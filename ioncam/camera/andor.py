"""Andor iXon (SDK2) driver on top of Andor's ``pyAndorSDK2`` wrapper.

Design notes (things the previous GUI got wrong, fixed here):

* **One thread owns the SDK.**  The engine calls everything from its camera
  thread; only :meth:`AndorCamera.cancel_wait` (``CancelWait``, documented for
  cross-thread use) is called from elsewhere.  SDK2 is not thread-safe.
* **Unsigned data.**  The wrapper's ``GetImages16``/``GetMostRecentImage16``
  allocate *signed* ``c_short`` buffers, so pixels above 32767 (EM-gain
  saturation, cosmic rays) come back negative.  The hot path here calls the
  DLL directly with a pre-allocated ``uint16`` numpy buffer (also avoiding a
  per-frame ctypes allocation and copy).
* **No SDK rotation/flip.**  ``SetImageRotate``/``SetImageFlip`` happen in host
  software and never reach the Camera Link port, so ROIs computed on rotated
  images would be wrong for the Grabber.  Data stays in raw sensor order.
* **Speeds and gains by value.**  Horizontal/vertical speeds are matched by
  value against the tables the camera reports, never by hard-coded index.
  ``SetVSSpeed`` is used for image mode (``SetFKVShiftSpeed`` only applies to
  fast kinetics).  The EM gain mode is set before any gain value.
* Every call is checked; failures raise :class:`CameraError` naming the
  function and the SDK error code.
"""

from __future__ import annotations

import ctypes
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ioncam.camera.base import (FAN_MODES, SHUTTER_MODES, AcquisitionSettings,
                                AppliedSettings, CameraBase, CameraBusy, CameraError,
                                nearest_index as _nearest, normalize_roi, roi_to_sdk,
                                vs_speed_index)

logger = logging.getLogger(__name__)

DRV_SUCCESS = 20002
DRV_NO_NEW_DATA = 20024
DRV_TEMP_OFF = 20034
DRV_TEMP_NOT_STABILIZED = 20035
DRV_TEMP_STABILIZED = 20036
DRV_TEMP_NOT_REACHED = 20037
DRV_TEMP_OUT_RANGE = 20038
DRV_TEMP_NOT_SUPPORTED = 20039
DRV_TEMP_DRIFT = 20040
DRV_P1INVALID = 20066
DRV_P3INVALID = 20068
DRV_P4INVALID = 20069
DRV_ACQUIRING = 20072
DRV_IDLE = 20073
DRV_NOT_INITIALIZED = 20075
DRV_INVALID_MODE = 20078
DRV_NOT_SUPPORTED = 20991
DRV_NOT_AVAILABLE = 20992

_FALLBACK_NAMES = {
    20002: "DRV_SUCCESS", 20003: "DRV_VXDNOTINSTALLED", 20013: "DRV_ERROR_ACK",
    20024: "DRV_NO_NEW_DATA", 20066: "DRV_P1INVALID", 20067: "DRV_P2INVALID",
    20068: "DRV_P3INVALID", 20069: "DRV_P4INVALID", 20072: "DRV_ACQUIRING",
    20073: "DRV_IDLE", 20075: "DRV_NOT_INITIALIZED", 20076: "DRV_P5INVALID",
    20077: "DRV_P6INVALID", 20078: "DRV_INVALID_MODE", 20089: "DRV_USBERROR",
    20095: "DRV_INVALID_TRIGGER_MODE", 20099: "DRV_BINNING_ERROR",
    20990: "DRV_ERROR_NOCAMERA", 20991: "DRV_NOT_SUPPORTED", 20992: "DRV_NOT_AVAILABLE",
}

_TEMP_STATUS = {
    DRV_TEMP_OFF: "off",
    DRV_TEMP_STABILIZED: "stabilized",
    DRV_TEMP_NOT_REACHED: "not_reached",
    DRV_TEMP_NOT_STABILIZED: "not_stabilized",
    DRV_TEMP_DRIFT: "drift",
    DRV_TEMP_OUT_RANGE: "out_of_range",
    DRV_TEMP_NOT_SUPPORTED: "not_supported",
}

ACQ_MODE = {"single": 1, "accumulate": 2, "run_till_abort": 5}
TRIGGER_MODE = {"internal": 0, "external": 1, "external_start": 6,
                "external_exposure": 7, "software": 10}
AMPLIFIER = {"em": 0, "conventional": 1}
# GetCapabilities ulFeatures bits
AC_FEATURES_SHUTTER = 1 << 3
AC_FEATURES_SHUTTEREX = 1 << 4
AC_FEATURES_FANCONTROL = 1 << 7
AC_FEATURES_MIDFANCONTROL = 1 << 8
EXTERNAL_TRIGGERS = ("external", "external_start", "external_exposure")


def error_name(code) -> str:
    try:
        from pyAndorSDK2.atmcd_errors import Error_Codes
        return Error_Codes(int(code)).name
    except Exception:
        return _FALLBACK_NAMES.get(int(code), "error")


def _text(value) -> str:
    v = getattr(value, "value", value)
    if isinstance(v, bytes):
        return v.decode("ascii", errors="replace").strip("\x00 ")
    return str(v)


class AndorCamera(CameraBase):
    """iXon (Ultra) camera through Andor SDK2.

    :param cfg: :class:`ioncam.config.CameraConfig`
    :param sdk: inject an ``atmcd``-compatible object (tests use
        :class:`ioncam.camera.fake_sdk.FakeAtmcd`); default loads pyAndorSDK2.
    """

    #: most frames fetched per wake-up; older ones are dropped (live view only
    #: needs the newest, and the SDK ring buffer would overwrite them anyway)
    max_batch = 256

    def __init__(self, cfg, sdk=None):
        self.cfg = cfg
        self._sdk = sdk
        self._dll = None
        self._open = False
        self._acquiring = False
        self._single_shot = False
        self._shape = (0, 0)
        self._applied: Optional[AppliedSettings] = None
        self._caps: Dict = {}
        self._last_temp: Tuple[float, str] = (float("nan"), "unknown")
        self.dropped_frames = 0
        self.info: Dict = {}
        self.shutter_note = ""

    # -- helpers --------------------------------------------------------------

    def _call(self, name: str, *args, tolerate=()):
        """Call SDK function ``name``; return its outputs (minus the return code).

        Raises :class:`CameraError` unless the code is DRV_SUCCESS or listed in
        ``tolerate`` (then returns None)."""
        fn = getattr(self._sdk, name)
        res = fn(*args)
        if isinstance(res, tuple):
            ret, vals = res[0], res[1:]
        else:
            ret, vals = res, ()
        if ret == DRV_SUCCESS:
            if not vals:
                return None
            return vals[0] if len(vals) == 1 else vals
        if ret in tolerate:
            logger.debug("%s%s -> %s (tolerated)", name, args, error_name(ret))
            return None
        raise CameraError("{}{} failed: {} ({})".format(name, args, error_name(ret), ret))

    def _try(self, name: str, *args, default=None):
        """Call an optional/informational SDK function; never raises."""
        try:
            fn = getattr(self._sdk, name)
            res = fn(*args)
        except Exception as e:  # missing function in older SDKs etc.
            logger.debug("%s unavailable: %s", name, e)
            return default
        if isinstance(res, tuple):
            if res[0] != DRV_SUCCESS:
                return default
            return res[1] if len(res) == 2 else res[1:]
        return res == DRV_SUCCESS

    # -- lifecycle ----------------------------------------------------------------

    def open(self) -> dict:
        if self._sdk is None:
            try:
                from pyAndorSDK2 import atmcd
            except Exception as e:
                raise CameraError(
                    "pyAndorSDK2 could not be imported ({}).  Install Andor's pyAndorSDK2 "
                    "package and the Andor SDK2 driver, or start with --simulate.".format(e)
                ) from e
            self._sdk = atmcd(self.cfg.sdk_dir or None)
        self._dll = self._sdk.dll

        n_cams = self._try("GetAvailableCameras", default=1)
        if n_cams == 0:
            raise CameraError("no Andor camera found (powered? USB connected?)")
        if self.cfg.camera_index > 0 or (n_cams or 1) > 1:
            if self.cfg.camera_index >= n_cams:
                raise CameraError("camera_index {} but only {} camera(s) found".format(
                    self.cfg.camera_index, n_cams))
            handle = self._call("GetCameraHandle", self.cfg.camera_index)
            self._call("SetCurrentCamera", handle)

        logger.info("initialising Andor SDK (this takes a few seconds)...")
        ret = self._sdk.Initialize(self.cfg.sdk_dir)
        if ret != DRV_SUCCESS:
            raise CameraError(
                "Initialize failed: {} ({}).  Is the camera powered and connected, and "
                "is Andor Solis (or another program) NOT using it?".format(error_name(ret), ret))
        self._open = True

        w, h = self._call("GetDetector")
        self.sensor_shape = (int(h), int(w))
        self._read_capabilities()

        self._call("SetReadMode", 4)
        # Raw orientation everywhere (see module docstring).
        self._call("SetImageRotate", 0, tolerate=(DRV_NOT_SUPPORTED, DRV_NOT_AVAILABLE))
        self._call("SetImageFlip", 0, 0, tolerate=(DRV_NOT_SUPPORTED, DRV_NOT_AVAILABLE))
        self._call("SetEMGainMode", self.cfg.em_gain_mode,
                   tolerate=(DRV_NOT_SUPPORTED, DRV_NOT_AVAILABLE))
        self._call("SetEMAdvanced", int(self.cfg.em_advanced),
                   tolerate=(DRV_NOT_SUPPORTED, DRV_NOT_AVAILABLE))
        tmin, tmax = self._caps.get("temperature_range", (-100, 30))
        target = int(min(max(self.cfg.target_temperature, tmin), tmax))
        self._call("SetTemperature", target)
        if self.cfg.cooler_on:
            self._call("CoolerON")
        self._call("SetCoolerMode", int(self.cfg.keep_cooler_on_exit),
                   tolerate=(DRV_NOT_SUPPORTED, DRV_NOT_AVAILABLE, DRV_P1INVALID))
        fan = {v: k for k, v in FAN_MODES.items()}.get(int(self.cfg.fan_mode), "full")
        try:
            self.set_fan(fan)
        except CameraError as e:
            logger.info("fan: %s", e)
        try:
            self.shutter_note = self.set_shutter(
                self.cfg.shutter, self.cfg.external_shutter, self.cfg.shutter_ttl_high,
                self.cfg.shutter_open_ms, self.cfg.shutter_close_ms)
        except CameraError as e:
            self.shutter_note = "the shutter could not be set: {}".format(e)
            logger.warning("%s", self.shutter_note)

        serial = self._try("GetCameraSerialNumber", default=0)
        head = _text(self._try("GetHeadModel", default=b"iXon"))
        pix = self._try("GetPixelSize", default=(16.0, 16.0))
        self.info = {"model": head, "serial": int(serial or 0),
                     "sensor_shape": list(self.sensor_shape),
                     "pixel_size_um": float(pix[0]) if pix else 16.0,
                     "simulated": False}
        logger.info("camera %s #%s, sensor %dx%d", head, serial, w, h)
        return dict(self.info)

    def _read_capabilities(self):
        caps: Dict = {}
        n_amp = self._try("GetNumberAmp", default=1) or 1
        caps["n_amplifiers"] = int(n_amp)
        hs = {}
        for amp_name, amp in AMPLIFIER.items():
            if amp >= n_amp:
                continue
            n = self._try("GetNumberHSSpeeds", 0, amp, default=0) or 0
            hs[amp_name] = [float(self._try("GetHSSpeed", 0, amp, i, default=0.0))
                            for i in range(n)]
        caps["hs_speeds_mhz"] = hs
        n_vs = self._try("GetNumberVSSpeeds", default=0) or 0
        caps["vs_speeds_us"] = [float(self._try("GetVSSpeed", i, default=0.0))
                                for i in range(n_vs)]
        fast = self._try("GetFastestRecommendedVSSpeed", default=None)
        caps["vs_fastest_recommended_index"] = int(fast[0]) if fast else 0
        if caps["vs_speeds_us"]:
            caps["vs_fastest_recommended"] = caps["vs_speeds_us"][
                min(caps["vs_fastest_recommended_index"], len(caps["vs_speeds_us"]) - 1)]
        n_pa = self._try("GetNumberPreAmpGains", default=1) or 1
        caps["preamp_gains"] = [float(self._try("GetPreAmpGain", i, default=1.0))
                                for i in range(n_pa)]
        caps["preamp_gain_texts"] = [_text(self._try("GetPreAmpGainText", i, 30,
                                                     default="Gain {}".format(i + 1)))
                                     for i in range(n_pa)]
        caps["n_vs_amplitudes"] = int(self._try("GetNumberVSAmplitudes", default=5) or 5)
        caps["readout_flipped"] = {name: bool(self._try("IsReadoutFlippedByAmplifier", amp,
                                                        default=0))
                                   for name, amp in AMPLIFIER.items() if amp < n_amp}
        caps_struct = self._try("GetCapabilities", default=None)
        feats = getattr(caps_struct, "ulFeatures", None)
        known = feats is not None
        internal = self._try("IsInternalMechanicalShutter", default=None)
        mins = self._try("GetShutterMinTimes", default=None)
        caps["shutter"] = {
            "control": not known or bool(feats & (AC_FEATURES_SHUTTER | AC_FEATURES_SHUTTEREX)),
            # None: not known (the capabilities could not be read): SetShutterEx is tried
            "independent": bool(feats & AC_FEATURES_SHUTTEREX) if known else None,
            "internal": None if internal is None else bool(internal),
            "min_close_ms": int(mins[0]) if mins else 0,
            "min_open_ms": int(mins[1]) if mins else 0}
        caps["fan"] = {"control": not known or bool(feats & AC_FEATURES_FANCONTROL),
                       "low": not known or bool(feats & AC_FEATURES_MIDFANCONTROL)}
        trange = self._try("GetTemperatureRange", default=None)
        caps["temperature_range"] = tuple(int(v) for v in trange) if trange else (-100, 30)
        self._caps = caps
        logger.info("HS speeds %s MHz, VS speeds %s us, pre-amp gains %s",
                    hs, caps["vs_speeds_us"], caps["preamp_gains"])

    def capabilities(self) -> dict:
        caps = dict(self._caps)
        caps["temperature_range"] = list(caps.get("temperature_range", ()))
        return caps

    def close(self) -> None:
        if not self._open:
            return
        try:
            self.stop()
        except CameraError as e:
            logger.warning("abort on close: %s", e)
        if not self.cfg.keep_cooler_on_exit:
            self._try("CoolerOFF")
        try:
            self._sdk.ShutDown()
        finally:
            self._open = False
            logger.info("Andor SDK shut down")

    # -- configuration --------------------------------------------------------------

    def configure(self, settings: AcquisitionSettings, tag: str = "") -> AppliedSettings:
        s = settings
        s.validate()
        if self._acquiring:
            self.stop()
        warnings: List[str] = []
        roi = normalize_roi(s.roi, self.sensor_shape, s.hbin, s.vbin)
        if s.roi is not None and tuple(s.roi) != roi:
            warnings.append("ROI {} adjusted to {} (sensor/binning)".format(list(s.roi), list(roi)))
        amp = AMPLIFIER[s.amplifier]
        caps = self._caps

        self._call("SetReadMode", 4)
        self._call("SetAcquisitionMode", ACQ_MODE[s.acquisition])
        self._call("SetOutputAmplifier", amp, tolerate=(DRV_NOT_SUPPORTED, DRV_NOT_AVAILABLE)
                   if amp == 0 else ())
        self._call("SetADChannel", 0, tolerate=(DRV_NOT_SUPPORTED, DRV_NOT_AVAILABLE,
                                                DRV_P1INVALID))

        speeds = caps.get("hs_speeds_mhz", {}).get(s.amplifier) or [s.hs_speed_mhz]
        hs_idx = _nearest(speeds, s.hs_speed_mhz)
        if abs(speeds[hs_idx] - s.hs_speed_mhz) > 0.05 * s.hs_speed_mhz:
            warnings.append("HS speed {} MHz not available; using {} MHz".format(
                s.hs_speed_mhz, speeds[hs_idx]))
        self._call("SetHSSpeed", amp, hs_idx)

        vs_list = caps.get("vs_speeds_us") or [0.0]
        vs_idx, note = vs_speed_index(vs_list, caps.get("vs_fastest_recommended_index", 0),
                                      s.vs_speed_us, s.vs_amplitude)
        if note:
            warnings.append(note)
        self._call("SetVSSpeed", vs_idx)
        self._call("SetVSAmplitude", int(s.vs_amplitude),
                   tolerate=(DRV_NOT_AVAILABLE, DRV_NOT_SUPPORTED) if s.vs_amplitude == 0 else ())

        pa = self._choose_preamp(amp, hs_idx, s.preamp_gain_index, warnings)
        self._call("SetPreAmpGain", pa)
        sensitivity, source = self._sensitivity(amp, hs_idx, pa)
        ret = self._sdk.SetBaselineClamp(int(s.baseline_clamp))
        if ret != DRV_SUCCESS and s.baseline_clamp:
            warnings.append("baseline clamp not available ({})".format(error_name(ret)))

        em_gain = 0
        em_range = (0, 0)
        if s.amplifier == "em":
            rng = self._try("GetEMGainRange", default=None)
            em_range = (int(rng[0]), int(rng[1])) if rng else (0, 255)
            em_gain = em_range[0] if s.em_gain <= 0 else int(min(max(s.em_gain, em_range[0]),
                                                                  em_range[1]))
            if s.em_gain > em_range[1]:
                warnings.append("EM gain {} above maximum {}; clamped".format(
                    s.em_gain, em_range[1]))
            self._call("SetEMCCDGain", em_gain)
            actual = self._try("GetEMCCDGain", default=em_gain)
            em_gain = int(actual) if actual is not None else em_gain

        self._call("SetFrameTransferMode", int(s.frame_transfer),
                   tolerate=(DRV_NOT_SUPPORTED, DRV_NOT_AVAILABLE) if not s.frame_transfer else ())

        trig = TRIGGER_MODE[s.trigger]
        ret = self._sdk.IsTriggerModeAvailable(trig)
        if ret != DRV_SUCCESS:
            raise CameraError("trigger mode '{}' not available with these settings ({})".format(
                s.trigger, error_name(ret)))
        self._call("SetTriggerMode", trig)
        if s.trigger in EXTERNAL_TRIGGERS:
            self._call("SetFastExtTrigger", int(s.fast_ext_trigger))
            ret = self._sdk.SetTriggerInvert(int(s.trigger_invert))
            if ret != DRV_SUCCESS and s.trigger_invert:
                raise CameraError("falling-edge trigger not available ({})".format(
                    error_name(ret)))

        self._call("SetImage", s.hbin, s.vbin, *roi_to_sdk(roi))
        self._call("SetExposureTime", float(s.exposure))
        if s.acquisition == "run_till_abort":
            self._call("SetKineticCycleTime", float(s.kinetic_cycle_time))
        elif s.acquisition == "accumulate":
            self._call("SetNumberAccumulations", int(s.n_accumulations))
            self._call("SetAccumulationCycleTime", 0.0)

        ret = self._sdk.SetCameraLinkMode(int(s.camera_link))
        if ret != DRV_SUCCESS:
            if s.camera_link:
                raise CameraError("Camera Link output not available on this camera ({})".format(
                    error_name(ret)))
            if ret not in (DRV_NOT_SUPPORTED, DRV_NOT_AVAILABLE):
                logger.warning("SetCameraLinkMode(0): %s", error_name(ret))

        timings = self._call("GetAcquisitionTimings")
        exposure, accumulate, kinetic = (float(v) for v in timings)
        readout = float(self._try("GetReadOutTime", default=0.0) or 0.0)
        keep_clean = float(self._try("GetKeepCleanTime", default=0.0) or 0.0)
        if abs(exposure - s.exposure) > max(1e-6, 0.02 * s.exposure):
            warnings.append("exposure {:.6g} s requested, camera uses {:.6g} s".format(
                s.exposure, exposure))
        self._call("PrepareAcquisition")

        h = (roi[3] - roi[1]) // s.vbin
        w = (roi[2] - roi[0]) // s.hbin
        self._shape = (h, w)
        self._single_shot = s.acquisition in ("single", "accumulate")
        pa_values = caps.get("preamp_gains") or [1.0]
        applied = AppliedSettings(
            settings=s, roi=roi, frame_shape=(h, w), exposure=exposure,
            accumulate_cycle=accumulate, kinetic_cycle=kinetic, readout_time=readout,
            keep_clean_time=keep_clean, hs_speed_mhz=float(speeds[hs_idx]),
            vs_speed_us=float(vs_list[vs_idx]) if vs_idx < len(vs_list) else 0.0,
            preamp_gain=float(pa_values[pa]) if pa < len(pa_values) else 1.0,
            em_gain=em_gain, em_gain_range=em_range, tag=tag, warnings=warnings,
            sensitivity=sensitivity, sensitivity_source=source,
            # the set gain is the multiplication only in RealGain mode (3)
            em_gain_real=s.amplifier != "em" or int(self.cfg.em_gain_mode) == 3)
        self._applied = applied
        for w_ in warnings:     # the engine reports them (once each)
            logger.debug("configure(%s): %s", tag, w_)
        return applied

    def _sensitivity(self, amp: int, hs_idx: int, pa: int) -> Tuple[float, str]:
        """Electrons per count at this readout: as configured (your camera's
        System Performance Sheet) or as the camera reports it - GetSensitivity
        (channel, horzShift, amplifier, pa), the argument order of Andor's
        header - else unknown (0)."""
        if self.cfg.sensitivity > 0:
            return float(self.cfg.sensitivity), "configuration"
        got = self._try("GetSensitivity", 0, hs_idx, amp, pa, default=None)
        try:
            value = float(got)
        except (TypeError, ValueError):
            return 0.0, ""
        return (value, "camera") if 0.01 < value < 1000.0 else (0.0, "")

    def _raw(self, name: str, *args):
        """An SDK call's return code (None: the function does not exist), logged."""
        fn = getattr(self._sdk, name, None)
        if fn is None:
            return None
        res = fn(*args)
        ret = res[0] if isinstance(res, tuple) else res
        logger.info("%s%s -> %s", name, args, error_name(ret))
        return ret

    def set_shutter(self, mode: str = "open", external: str = "open", ttl_high: bool = True,
                    open_ms: int = 0, close_ms: int = 0) -> str:
        """SetShutterEx (internal + external shutter) where the camera has it,
        else SetShutter.  Times below the camera's minimum (GetShutterMinTimes)
        are raised to it in the automatic modes, where the shutter must move in
        them, and whenever the camera rejects them - an iXon Ultra 888 may
        refuse 0 ms (Andor otherwise advises 0 ms for a permanently open
        shutter)."""
        if mode not in SHUTTER_MODES or external not in SHUTTER_MODES:
            raise CameraError("shutter mode must be one of {}".format(", ".join(SHUTTER_MODES)))
        sh = self._caps.get("shutter") or {}
        if sh.get("control") is False:
            raise CameraError("this camera has no shutter control")
        typ, m, e = int(bool(ttl_high)), SHUTTER_MODES[mode], SHUTTER_MODES[external]
        lo_c, lo_o = int(sh.get("min_close_ms") or 0), int(sh.get("min_open_ms") or 0)
        times = [(int(close_ms), int(open_ms))]
        if mode in ("auto", "series") or external in ("auto", "series"):
            times = [(max(c, lo_c), max(o, lo_o)) for c, o in times]
        if times[0] != (max(times[0][0], lo_c), max(times[0][1], lo_o)):
            times.append((max(times[0][0], lo_c), max(times[0][1], lo_o)))
        calls = ([] if sh.get("independent") is False else
                 [("SetShutterEx", lambda c, o: (typ, m, c, o, e))]) + \
            [("SetShutter", lambda c, o: (typ, m, c, o))]
        answers = []
        for name, args in calls:
            for c, o in times:
                ret = self._raw(name, *args(c, o))
                if ret is None:
                    break
                answers.append("{} {}".format(name, error_name(ret)))
                if ret == DRV_SUCCESS:
                    if name == "SetShutterEx" and sh.get("independent") is None:
                        sh["independent"] = True
                    raised = (c, o) != (int(close_ms), int(open_ms))
                    return "{} accepted{}".format(name, "; times raised to the camera's "
                                                  "minimum, closing {} / opening {} ms".format(
                                                      c, o) if raised else "")
                if ret == DRV_ACQUIRING:
                    raise CameraBusy("{}: acquisition in progress".format(name))
                if ret in (DRV_P3INVALID, DRV_P4INVALID):
                    continue                    # the times: with the minimum next
                break                           # not this call: the next one
        raise CameraError("the camera accepted no shutter setting ({})".format(
            "; ".join(answers) or "no shutter functions"))

    def set_fan(self, mode: str) -> str:
        if mode not in FAN_MODES:
            raise CameraError("fan mode must be one of {}".format(", ".join(FAN_MODES)))
        fan = self._caps.get("fan") or {}
        if fan.get("control") is False:
            raise CameraError("this camera has no fan control")
        if mode == "low" and fan.get("low") is False:
            raise CameraError("this camera's fan has no low setting")
        ret = self._raw("SetFanMode", FAN_MODES[mode])
        if ret == DRV_ACQUIRING:
            raise CameraBusy("SetFanMode: acquisition in progress")
        if ret != DRV_SUCCESS:
            raise CameraError("SetFanMode({}) failed: {} ({})".format(
                FAN_MODES[mode], error_name(ret), ret))
        return "SetFanMode accepted"

    def wait_idle(self, timeout: float = 3.0) -> bool:
        """Poll GetStatus until DRV_IDLE (AbortAcquisition may return before the
        camera has stopped)."""
        t_end = time.monotonic() + timeout
        while True:
            status = self._try("GetStatus", default=DRV_IDLE)
            if status == DRV_IDLE:
                return True
            if time.monotonic() >= t_end:
                logger.warning("camera not idle %.1f s after the abort (%s)", timeout,
                               error_name(status))
                return False
            time.sleep(0.02)

    def _choose_preamp(self, amp: int, hs_idx: int, requested: int, warnings: List[str]) -> int:
        """The pre-amp gain index: as requested if available at this speed, else
        the nearest lower (then any) one; ``requested < 0``: the highest
        available, without a warning (cameras differ: 897 three, 888 two)."""
        n = len(self._caps.get("preamp_gains") or [1.0])
        highest = requested < 0
        if highest:
            requested = n - 1
        candidates = [requested] + list(range(min(requested, n - 1), -1, -1)) + list(range(n))
        for pa in candidates:
            if not 0 <= pa < n:
                continue
            avail = self._try("IsPreAmpGainAvailable", 0, amp, hs_idx, pa, default=1)
            if avail:
                if pa != requested and not highest:
                    warnings.append("pre-amp gain index {} unavailable at this speed; "
                                    "using {}".format(requested, pa))
                return pa
        raise CameraError("no pre-amp gain available for this readout speed")

    # -- acquisition ----------------------------------------------------------------

    def start(self) -> None:
        if self._applied is None:
            raise CameraError("configure() must be called before start()")
        self._call("StartAcquisition")
        self._acquiring = True

    def stop(self) -> None:
        if not self._open:
            return
        ret = self._sdk.AbortAcquisition()
        self._acquiring = False
        if ret not in (DRV_SUCCESS, DRV_IDLE):
            raise CameraError("AbortAcquisition failed: {} ({})".format(error_name(ret), ret))

    def is_acquiring(self) -> bool:
        return self._acquiring

    def cancel_wait(self) -> None:
        dll = self._dll
        if dll is not None:
            try:
                dll.CancelWait()
            except Exception:  # pragma: no cover - never let a wake-up fail loudly
                pass

    def send_software_trigger(self) -> None:
        self._call("SendSoftwareTrigger")

    def wait_frames(self, timeout: float) -> List[Tuple[int, np.ndarray]]:
        if not self._acquiring:
            time.sleep(min(max(timeout, 0.0), 0.02))
            return []
        self._dll.WaitForAcquisitionTimeOut(ctypes.c_int(max(1, int(timeout * 1000))))
        return self._fetch_new()

    def _fetch_new(self) -> List[Tuple[int, np.ndarray]]:
        first = ctypes.c_int32(0)
        last = ctypes.c_int32(0)
        ret = self._dll.GetNumberNewImages(ctypes.pointer(first), ctypes.pointer(last))
        if ret != DRV_SUCCESS:
            self._check_finished()
            return []
        f, l = int(first.value), int(last.value)
        if l < f:
            return []
        if l - f + 1 > self.max_batch:
            self.dropped_frames += l - f + 1 - self.max_batch
            f = l - self.max_batch + 1
        n = l - f + 1
        h, w = self._shape
        buf = np.empty((n, h, w), dtype=np.uint16)
        vfirst = ctypes.c_int32(0)
        vlast = ctypes.c_int32(0)
        ret = self._dll.GetImages16(ctypes.c_int32(f), ctypes.c_int32(l),
                                    buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                                    ctypes.c_uint32(buf.size),
                                    ctypes.pointer(vfirst), ctypes.pointer(vlast))
        if ret == DRV_NO_NEW_DATA:
            return []
        if ret != DRV_SUCCESS:
            raise CameraError("GetImages16({}, {}) failed: {} ({})".format(
                f, l, error_name(ret), ret))
        vf, vl = int(vfirst.value), int(vlast.value)
        got = max(0, min(n, vl - vf + 1))
        if got < n:
            self.dropped_frames += n - got
        self._check_finished()
        return [(vf + i, buf[i]) for i in range(got)]

    def _check_finished(self) -> None:
        if not self._single_shot:
            return
        status = self._try("GetStatus", default=DRV_ACQUIRING)
        if status == DRV_IDLE:
            self._acquiring = False

    # -- temperature ------------------------------------------------------------------

    def get_temperature(self) -> Tuple[float, str]:
        ret, temp = self._sdk.GetTemperatureF()
        if ret == DRV_ACQUIRING or ret == DRV_NOT_INITIALIZED:
            return self._last_temp
        status = _TEMP_STATUS.get(ret, "unknown")
        self._last_temp = (float(temp), status)
        return self._last_temp

    def set_temperature(self, temperature: float) -> None:
        tmin, tmax = self._caps.get("temperature_range", (-100, 30))
        if not tmin <= temperature <= tmax:
            raise CameraError("temperature {} outside {}..{} C".format(temperature, tmin, tmax))
        self._call("SetTemperature", int(round(temperature)))

    def set_cooler(self, on: bool) -> None:
        self._call("CoolerON" if on else "CoolerOFF")
