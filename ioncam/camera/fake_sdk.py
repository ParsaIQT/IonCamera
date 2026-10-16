"""An in-process emulation of Andor's ``pyAndorSDK2.atmcd`` for an iXon Ultra 897
(or 888: ``simulator.camera_model``).

It lets the *real* :class:`ioncam.camera.andor.AndorCamera` driver code run
without hardware (``backend = "fake_sdk"``) and is what the driver tests use.
Images come from :class:`ioncam.camera.simulator.SimulatedCamera`.

Every call is recorded in :attr:`FakeAtmcd.calls` so tests can check the
exact SDK call sequence.  The ``dll`` attribute emulates the raw DLL functions
used on the driver's hot path, including ctypes pointer arguments.
"""

from __future__ import annotations

import ctypes
import threading
import time
from typing import Dict, List, Tuple

import numpy as np

from ioncam.camera.base import AcquisitionSettings, CameraError
from ioncam.camera.simulator import SimulatedCamera

S = 20002            # DRV_SUCCESS
NO_NEW_DATA = 20024
P1INVALID, P2INVALID, P3INVALID, P4INVALID = 20066, 20067, 20068, 20069
ACQUIRING, IDLE, NOT_INITIALIZED = 20072, 20073, 20075
INVALID_MODE = 20078
NOT_SUPPORTED = 20991
TEMP = {"off": 20034, "not_stabilized": 20035, "stabilized": 20036,
        "not_reached": 20037, "drift": 20040}

_ACQ = {1: "single", 2: "accumulate", 5: "run_till_abort"}
_TRIG = {0: "internal", 1: "external", 6: "external_start", 7: "external_exposure",
         10: "software"}
RING_BUFFER = 64
_SHUTTER = {0: "auto", 1: "open", 2: "closed", 4: "series", 5: "series"}


class _Capabilities:
    """The part of AndorCapabilities the driver reads."""

    def __init__(self, features: int):
        self.ulFeatures = features


class _FakeDll:
    """Raw DLL entry points used by the driver's hot path."""

    def __init__(self, owner: "FakeAtmcd"):
        self._o = owner

    def WaitForAcquisitionTimeOut(self, ms):
        o = self._o
        o.calls.append(("dll.WaitForAcquisitionTimeOut", (ms.value,)))
        if not o.sim.is_acquiring() and not o._pending():
            return NO_NEW_DATA
        frames = o.sim.wait_frames(ms.value / 1000.0)
        with o._lock:
            for _, img in frames:
                o._acquired += 1
                o._ring[o._acquired] = img
                o._ring.pop(o._acquired - RING_BUFFER, None)
        return S if frames else NO_NEW_DATA

    def GetNumberNewImages(self, pfirst, plast):
        o = self._o
        with o._lock:
            first = max(o._retrieved + 1, o._acquired - RING_BUFFER + 1)
            if o._acquired < first:
                return NO_NEW_DATA
            pfirst.contents.value = first
            plast.contents.value = o._acquired
        return S

    def GetImages16(self, first, last, arr, size, pvfirst, pvlast):
        o = self._o
        f, l, n = first.value, last.value, size.value
        o.calls.append(("dll.GetImages16", (f, l, n)))
        npix = o._npix()
        with o._lock:
            if l < f or f < 1 or l > o._acquired:
                return P1INVALID
            if n != (l - f + 1) * npix:
                return P4INVALID
            out = np.ctypeslib.as_array(arr, shape=(n,))
            vf = max(f, o._acquired - RING_BUFFER + 1)
            k = 0
            for idx in range(vf, l + 1):
                out[k * npix:(k + 1) * npix] = o._ring[idx].ravel()
                k += 1
            o._retrieved = max(o._retrieved, l)
        pvfirst.contents.value = vf
        pvlast.contents.value = l
        return S

    def CancelWait(self):
        self._o.sim.cancel_wait()
        return S


class FakeAtmcd:
    """Behaves like ``pyAndorSDK2.atmcd`` connected to an iXon Ultra 897 (or 888)."""

    def __init__(self, sim_cfg, serial: int = 12345):
        self.sim = SimulatedCamera(sim_cfg)
        self.model = m = self.sim.model
        self._hs = (m["hs"]["em"], m["hs"]["conventional"])
        self._vs = m["vs"]
        self._preamp = m["preamp"]
        self._h, self._w = m["sensor"]
        self.dll = _FakeDll(self)
        self.calls: List[Tuple[str, tuple]] = []
        self.serial = serial
        self._lock = threading.Lock()
        self._ring: Dict[int, np.ndarray] = {}
        self._acquired = 0
        self._retrieved = 0
        self._initialized = False
        # real-camera behaviours, for the tests
        self.busy_while_acquiring = False   # refuse fan / shutter settings while acquiring
        self.abort_lag = 0.0                # [s] still "acquiring" after AbortAcquisition
        self._abort_until = 0.0
        self.enforce_min_shutter_times = False  # refuse shutter times < 27 ms
        self.shutterex_only = False         # SetShutter accepted but ignored
        self.state = {
            "read_mode": 4, "acq_mode": 1, "trigger": 0, "amp": 0, "hs": 0, "vs": 0,
            "vs_amp": 0, "preamp": 0, "clamp": 0, "em_gain_mode": 0, "em_gain": 0,
            "em_advanced": 0, "ft": 0, "fast_ext": 0, "invert": 0, "camera_link": 0,
            "image": (1, 1, 1, self._w, 1, self._h), "exposure": 0.01, "kinetic": 0.0,
            "n_acc": 1, "rotate": 0, "flip": (0, 0), "cooler_mode": 0, "shutter": None,
        }

    # recording ------------------------------------------------------------------
    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        if name[:1].isupper() and callable(attr):
            calls = object.__getattribute__(self, "calls")

            def rec(*args, **kw):
                calls.append((name, args))
                return attr(*args, **kw)
            return rec
        return attr

    def _pending(self) -> bool:
        with self._lock:
            return self._acquired > self._retrieved

    def _npix(self) -> int:
        hb, vb, hs, he, vs, ve = self.state["image"]
        return ((he - hs + 1) // hb) * ((ve - vs + 1) // vb)

    def _need_idle(self):
        if not self._initialized:
            return NOT_INITIALIZED
        if self.sim.is_acquiring():
            return ACQUIRING
        return None

    def _set(self, key, value, valid=None):
        err = self._need_idle()
        if err:
            return err
        if valid is not None and value not in valid:
            return P1INVALID
        self.state[key] = value
        return S

    # discovery / init ----------------------------------------------------------------
    def GetAvailableCameras(self):
        return S, 1

    def GetCameraHandle(self, index):
        return (S, 100 + index) if index == 0 else (P1INVALID, 0)

    def SetCurrentCamera(self, handle):
        return S

    def Initialize(self, directory):
        self._initialized = True
        return S

    def ShutDown(self):
        self.sim.stop()
        self._initialized = False
        return S

    def GetDetector(self):
        return S, self._w, self._h

    def GetHeadModel(self):
        return S, self.model["head"].encode()

    def GetCameraSerialNumber(self):
        return S, self.serial

    def GetPixelSize(self):
        return S, self.model["pixel_um"], self.model["pixel_um"]

    def GetNumberAmp(self):
        return S, 2

    def GetNumberADChannels(self):
        return S, 1

    def GetNumberHSSpeeds(self, channel, typ):
        return S, len(self._hs[typ])

    def GetHSSpeed(self, channel, typ, index):
        speeds = self._hs[typ]
        return (S, speeds[index]) if 0 <= index < len(speeds) else (P3INVALID, 0.0)

    def GetNumberVSSpeeds(self):
        return S, len(self._vs)

    def GetVSSpeed(self, index):
        return (S, self._vs[index]) if 0 <= index < len(self._vs) else (P1INVALID, 0.0)

    def GetFastestRecommendedVSSpeed(self):
        k = self.model["vs_fastest"]
        return S, k, self._vs[k]

    def GetNumberPreAmpGains(self):
        return S, len(self._preamp)

    def GetPreAmpGain(self, index):
        return S, self._preamp[index]

    def GetPreAmpGainText(self, index, length):
        buf = ctypes.create_string_buffer(length)
        buf.value = "Gain {}".format(index + 1).encode()
        return S, buf

    def GetNumberVSAmplitudes(self):
        return S, 5

    def IsReadoutFlippedByAmplifier(self, amplifier):
        return S, int(amplifier == 1)

    def GetTemperatureRange(self):
        return S, -100, 20

    def IsPreAmpGainAvailable(self, channel, amplifier, index, pa):
        # emulate a camera where gain index 2 is not offered at 17 MHz
        ok = not (amplifier == 0 and index == 0 and pa == 2)
        return S, int(ok)

    # settings ------------------------------------------------------------------------
    def SetReadMode(self, mode):
        return self._set("read_mode", mode, (0, 1, 2, 3, 4))

    def SetAcquisitionMode(self, mode):
        return self._set("acq_mode", mode, (1, 2, 3, 4, 5))

    def SetOutputAmplifier(self, typ):
        return self._set("amp", typ, (0, 1))

    def SetADChannel(self, channel):
        return S if channel == 0 else P1INVALID

    def SetHSSpeed(self, typ, index):
        n = len(self._hs[typ]) if typ in (0, 1) else 0
        if not 0 <= index < n:
            return P2INVALID
        return self._set("hs", index)

    def SetVSSpeed(self, index):
        return self._set("vs", index, range(len(self._vs)))

    def SetFKVShiftSpeed(self, index):
        return S  # accepted but irrelevant for image mode (the old GUI's bug)

    def SetVSAmplitude(self, index):
        return self._set("vs_amp", index, range(5))

    def SetPreAmpGain(self, index):
        return self._set("preamp", index, range(len(self._preamp)))

    def SetBaselineClamp(self, state):
        return self._set("clamp", state, (0, 1))

    def SetEMGainMode(self, mode):
        return self._set("em_gain_mode", mode, (0, 1, 2, 3))

    def SetEMAdvanced(self, state):
        return self._set("em_advanced", state, (0, 1))

    def _em_range(self):
        mode = self.state["em_gain_mode"]
        if mode == 0:
            return 0, 255
        if mode == 1:
            return 0, 4095
        return (1, 1000) if self.state["em_advanced"] else (1, 300)

    def GetEMGainRange(self):
        lo, hi = self._em_range()
        return S, lo, hi

    def SetEMCCDGain(self, gain):
        lo, hi = self._em_range()
        if not lo <= gain <= hi:
            return P1INVALID
        return self._set("em_gain", gain)

    def GetEMCCDGain(self):
        return S, self.state["em_gain"]

    def SetFrameTransferMode(self, mode):
        return self._set("ft", mode, (0, 1))

    def IsTriggerModeAvailable(self, mode):
        return S if mode in _TRIG else INVALID_MODE

    def SetTriggerMode(self, mode):
        return self._set("trigger", mode, tuple(_TRIG))

    def SetFastExtTrigger(self, mode):
        self.state["fast_ext"] = mode
        return S

    def SetTriggerInvert(self, mode):
        return self._set("invert", mode, (0, 1))

    def SetImage(self, hbin, vbin, hstart, hend, vstart, vend):
        err = self._need_idle()
        if err:
            return err
        if hbin < 1 or vbin < 1:
            return P1INVALID
        if not (1 <= hstart <= hend <= self._w):
            return P3INVALID
        if not (1 <= vstart <= vend <= self._h):
            return 20076  # DRV_P5INVALID
        if (hend - hstart + 1) % hbin or (vend - vstart + 1) % vbin:
            return 20099  # DRV_BINNING_ERROR
        self.state["image"] = (hbin, vbin, hstart, hend, vstart, vend)
        return S

    def SetImageRotate(self, rotate):
        self.state["rotate"] = rotate
        return S

    def SetImageFlip(self, h, v):
        self.state["flip"] = (h, v)
        return S

    def SetExposureTime(self, t):
        return self._set("exposure", float(t))

    def SetKineticCycleTime(self, t):
        return self._set("kinetic", float(t))

    def SetNumberAccumulations(self, n):
        return self._set("n_acc", int(n))

    def SetAccumulationCycleTime(self, t):
        return S

    def SetCameraLinkMode(self, mode):
        return self._set("camera_link", mode, (0, 1))

    def SetCoolerMode(self, mode):
        self.state["cooler_mode"] = mode
        return S

    def _busy(self) -> bool:
        """Acquiring - or just aborted (``abort_lag``): settings are refused."""
        return self.sim.is_acquiring() or time.monotonic() < self._abort_until

    def SetFanMode(self, mode):
        if self.busy_while_acquiring and self._busy():
            return ACQUIRING
        if mode not in (0, 1, 2):
            return P1INVALID
        self.state["fan"] = mode
        return S

    def _shutter(self, typ, mode, closing, opening):
        if self.busy_while_acquiring and self._busy():
            return ACQUIRING
        if typ not in (0, 1):
            return P1INVALID
        if mode not in _SHUTTER:
            return P2INVALID
        if closing < 0 or (self.enforce_min_shutter_times and closing < 27):
            return P3INVALID
        if opening < 0 or (self.enforce_min_shutter_times and opening < 27):
            return P4INVALID
        self.sim.set_shutter(_SHUTTER[mode])
        return S

    def SetShutter(self, typ, mode, closing, opening):
        if self.shutterex_only:
            return S                        # accepted, and ignored: "you MUST use SetShutterEx"
        ret = self._shutter(typ, mode, closing, opening)
        if ret == S:
            self.state["shutter"] = (typ, mode)
        return ret

    def SetShutterEx(self, typ, mode, closing, opening, extmode):
        if extmode not in _SHUTTER:
            return 20076                    # DRV_P5INVALID
        ret = self._shutter(typ, mode, closing, opening)
        if ret == S:
            self.state["shutter"] = (typ, mode, extmode)
        return ret

    def IsInternalMechanicalShutter(self):
        return S, 1

    def GetShutterMinTimes(self):
        return S, 27, 27

    def GetCapabilities(self):
        # shutter (SetShutter + SetShutterEx), fan control with a low setting
        return S, _Capabilities((1 << 3) | (1 << 4) | (1 << 7) | (1 << 8))

    def GetSensitivity(self, channel, horz_shift, amplifier, pa):
        # Andor's header: (channel, horzShift, amplifier, pa)
        if channel != 0:
            return P1INVALID, 0.0
        if amplifier not in (0, 1):
            return P3INVALID, 0.0
        if not 0 <= horz_shift < len(self._hs[amplifier]):
            return P2INVALID, 0.0
        if not 0 <= pa < len(self._preamp):
            return P4INVALID, 0.0
        # what the emulated images use: the simulator's electrons per count
        return S, float(self.sim.sensitivity_for(pa))

    # timings ------------------------------------------------------------------------
    def _settings(self) -> AcquisitionSettings:
        st = self.state
        hb, vb, hs, he, vs, ve = st["image"]
        amp = "em" if st["amp"] == 0 else "conventional"
        speeds = self._hs[0 if st["amp"] == 0 else 1]
        return AcquisitionSettings(
            exposure=st["exposure"], em_gain=st["em_gain"], amplifier=amp,
            hs_speed_mhz=speeds[st["hs"]], vs_speed_us=self._vs[st["vs"]],
            vs_amplitude=st["vs_amp"], preamp_gain_index=st["preamp"],
            baseline_clamp=bool(st["clamp"]), hbin=hb, vbin=vb, roi=(hs - 1, vs - 1, he, ve),
            trigger=_TRIG[st["trigger"]], acquisition=_ACQ.get(st["acq_mode"], "single"),
            n_accumulations=st["n_acc"], frame_transfer=bool(st["ft"]),
            fast_ext_trigger=bool(st["fast_ext"]), trigger_invert=bool(st["invert"]),
            camera_link=bool(st["camera_link"]), kinetic_cycle_time=st["kinetic"])

    def GetAcquisitionTimings(self):
        if self.state["read_mode"] != 4:
            return INVALID_MODE, 0.0, 0.0, 0.0
        a = self.sim.configure(self._settings())
        return S, a.exposure, a.accumulate_cycle, a.kinetic_cycle

    def GetReadOutTime(self):
        return S, self.sim.configure(self._settings()).readout_time

    def GetKeepCleanTime(self):
        return S, self.sim.configure(self._settings()).keep_clean_time

    def PrepareAcquisition(self):
        return self._need_idle() or S

    # acquisition --------------------------------------------------------------------
    def StartAcquisition(self):
        err = self._need_idle()
        if err:
            return err
        self.sim.configure(self._settings(), tag="fake_sdk")
        with self._lock:
            self._ring.clear()
            self._acquired = 0
            self._retrieved = 0
        self.sim.start()
        return S

    def AbortAcquisition(self):
        if not self._initialized:
            return NOT_INITIALIZED
        if not self.sim.is_acquiring():
            return IDLE
        self.sim.stop()
        self._abort_until = time.monotonic() + self.abort_lag
        return S

    def GetStatus(self):
        return S, ACQUIRING if self._busy() else IDLE

    def CancelWait(self):
        self.sim.cancel_wait()
        return S

    def SendSoftwareTrigger(self):
        if self.state["trigger"] != 10:
            return INVALID_MODE
        try:
            self.sim.send_software_trigger()
        except CameraError:
            return IDLE
        return S

    # temperature ----------------------------------------------------------------------
    def SetTemperature(self, t):
        self.sim.set_temperature(t)
        return S

    def CoolerON(self):
        self.sim.set_cooler(True)
        return S

    def CoolerOFF(self):
        self.sim.set_cooler(False)
        return S

    def GetTemperatureF(self):
        t, status = self.sim.get_temperature()
        return TEMP.get(status, 20034), t

    def GetTemperature(self):
        t, status = self.sim.get_temperature()
        return TEMP.get(status, 20034), int(round(t))
