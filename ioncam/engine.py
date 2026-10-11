"""The camera brain.

Threads
-------
* **camera thread** - the only thread that talks to the camera.  It executes
  commands from a queue between frame waits (``CancelWait`` wakes it up
  immediately) and runs multi-step *procedures* (locate ions, calibrate and
  arm the Grabber readout) start to finish, so the camera can never be
  reconfigured half-way through one.
* **analysis thread** - detection + chain tracking on live frames (always the
  newest frame; stale ones are dropped), and cheap per-ROI sums of *every*
  readout frame (the iXon keeps sending frames over USB while it streams to
  the Grabber, so the brain keeps watching the ions during experiments).
* callers (RPC server, GUI) only touch thread-safe snapshots and the queue.

Modes: ``idle``, ``live`` (internal-trigger video), ``readout`` (armed:
external trigger + Camera Link), ``busy`` (a procedure is running).
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import logging
import math
import os
import queue
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ioncam import __version__, version_string
from ioncam.analysis.chain import equilibrium_positions, fit_chain
from ioncam.analysis.detection import DetectionResult, Detector
from ioncam.analysis.fitting import PSF, fit_gaussians
from ioncam.analysis.readout import (PlanError, ReadoutPlan, binning_advice,
                                     compare_grabber_frame, dark_pixel_mask, fit_threshold,
                                     plan_readout, reposition_plan, shot_histograms,
                                     verify_plan)
from ioncam.analysis.tracker import ChainTracker, adopt_learned_psf, align
from ioncam.camera.base import (AcquisitionSettings, AppliedSettings, CameraBase, CameraError,
                                Frame)
from ioncam.config import Config
from ioncam.events import EventLog

logger = logging.getLogger(__name__)

IDLE, LIVE, READOUT, BUSY, ERROR = "idle", "live", "readout", "busy", "error"


class ProcedureError(RuntimeError):
    """A procedure (locate, calibrate, ...) could not be completed."""


class Aborted(ProcedureError):
    pass


def _rebaseline_needed(old: Optional[AppliedSettings], new: AppliedSettings) -> bool:
    """Brightness is normalised by exposure x EM gain; anything else that changes
    counts per photon (binning, amplifier, speed, pre-amp) needs new references."""
    if old is None:
        return True
    a, b = old.settings, new.settings
    return ((a.hbin, a.vbin, a.amplifier, a.preamp_gain_index, old.hs_speed_mhz) !=
            (b.hbin, b.vbin, b.amplifier, b.preamp_gain_index, new.hs_speed_mhz))


def _merge_split_spots(positions, flux, n: int, sigma: float) -> np.ndarray:
    """More spots than the ``n`` ions known to be there: at the very high SNR of
    a long exposure a PSF estimate that is a little off makes the detector
    split a bright ion into two close spots with one ion's light between them.
    Merge such pairs (closest first, within ``2 sigma``) - real ions are
    further apart than the detector can resolve, and carry an ion's light each."""
    pos = np.asarray(positions, float).reshape(-1, 2).copy()
    f = np.asarray(flux, float).copy()
    while len(pos) > n:
        d = np.hypot(pos[:, None, 0] - pos[None, :, 0], pos[:, None, 1] - pos[None, :, 1])
        np.fill_diagonal(d, np.inf)
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        rest = np.delete(f, [i, j])
        one_ion = float(np.median(rest)) if len(rest) else f[i] + f[j]
        if d[i, j] > 2.0 * sigma or f[i] + f[j] > 1.5 * one_ion:
            break
        w = max(f[i] + f[j], 1e-12)
        pos[i] = (pos[i] * f[i] + pos[j] * f[j]) / w
        f[i] = f[i] + f[j]
        pos, f = np.delete(pos, j, axis=0), np.delete(f, j)
    return pos


def _crystal_spots(positions, sites, sigma: float, scale: float = 0.0) -> np.ndarray:
    """Which spots of a long exposure belong to the crystal the tracker follows
    (``sites``: its bright and dark ions, sensor px) - along the chain axis up
    to 1.5 ion spacings beyond its ends, across it within 3 PSF sigmas (at
    least 3 px) of its ions.  A long exposure also shows surface light
    (defects, dust, electrode edges) as ion-like spots anywhere in the field
    of view, unless a learned background removes it."""
    pos = np.asarray(positions, float).reshape(-1, 2)
    s = np.asarray(sites, float).reshape(-1, 2)
    if len(pos) == 0 or len(s) == 0:
        return np.ones(len(pos), bool)
    across = max(3.0 * float(sigma), 3.0)
    if len(s) == 1:
        reach = max(across, 1.5 * float(scale) if scale else 2.0 * across)
        return np.hypot(pos[:, 0] - s[0, 0], pos[:, 1] - s[0, 1]) <= reach
    center = s.mean(axis=0)
    _, _, vt = np.linalg.svd(s - center)
    axis, normal = vt[0], np.array([-vt[0][1], vt[0][0]])
    u, v = (s - center) @ axis, (s - center) @ normal
    ext = max(1.5 * float(np.ptp(u)) / (len(s) - 1), across)
    up, vp = (pos - center) @ axis, (pos - center) @ normal
    return (up >= u.min() - ext) & (up <= u.max() + ext) & (np.abs(vp) <= np.abs(v).max() + across)


def _pad_roi(roi, min_size: int, sensor_shape) -> tuple:
    """``roi`` (x0, y0, x1, y1) grown symmetrically to at least ``min_size``
    pixels per axis, kept on the sensor."""
    out = []
    for lo, hi, n in ((roi[0], roi[2], sensor_shape[1]), (roi[1], roi[3], sensor_shape[0])):
        size = min(max(hi - lo, min_size), n)
        lo = min(max(0, lo - (size - (hi - lo)) // 2), n - size)
        out.append((lo, lo + size))
    return (out[0][0], out[1][0], out[0][1], out[1][1])


class ReadoutMonitor:
    """Per-ROI sums of every readout frame (and the frame's background pixels,
    the dark reference of the shot histograms) + health checks."""

    def __init__(self, maxlen: int = 50000):
        self._lock = threading.Lock()
        self._maxlen = maxlen
        self.reset(None)

    def reset(self, plan: Optional[ReadoutPlan]) -> None:
        with self._lock:
            self.plan = plan
            self.sums: Deque[np.ndarray] = deque(maxlen=self._maxlen)
            self.dark: Deque[np.ndarray] = deque(maxlen=self._maxlen)
            self.dark_mask = dark_pixel_mask(plan) if plan is not None else None
            self.times: Deque[float] = deque(maxlen=self._maxlen)
            self.outside: Deque[float] = deque(maxlen=2000)
            self.frames: Deque[np.ndarray] = deque(maxlen=200)
            self.shots = 0
            self.first_shot = 0
            self.outside_baseline: Optional[float] = None
            self.geometry_warning = False

    def add(self, frames: Sequence[Frame]) -> List[dict]:
        events = []
        with self._lock:
            plan = self.plan
            if plan is None or not frames:
                return events
            w, h = plan.frame_size
            data = np.stack([f.data for f in frames if f.data.shape == (h, w)])
            if data.size == 0:
                return events
            sums = plan.roi_sums(data)
            dark = data[:, self.dark_mask]
            n = plan.n_ions
            area = np.array([(r[2] - r[0]) * (r[3] - r[1]) for r in plan.sdk_rois()], float)
            total = data.reshape(len(data), -1).sum(axis=1).astype(float)
            if plan.background_roi is not None:
                bias = sums[:, n] / area[n]
            else:
                bias = np.median(np.concatenate([data[:, 0, :], data[:, -1, :]], axis=1), axis=1)
            sig_total = total - bias * (w * h)
            sig_ions = sums[:, :n].sum(axis=1) - bias * area[:n].sum()
            for k, f in enumerate(frames[:len(data)]):
                self.frames.append(data[k])
                self.sums.append(sums[k])
                self.dark.append(dark[k])
                self.times.append(f.timestamp)
                self.shots += 1
                # fraction of the light outside the ion ROIs (only when ions shine)
                if sig_total[k] > 50.0 * math.sqrt(max(w * h, 1)):
                    self.outside.append(float(1.0 - sig_ions[k] / sig_total[k]))
            if len(self.outside) >= 50:
                recent = float(np.median(list(self.outside)[-50:]))
                if self.outside_baseline is None:
                    self.outside_baseline = recent
                elif not self.geometry_warning and recent > self.outside_baseline + 0.2:
                    self.geometry_warning = True
                    events.append({"kind": "readout_geometry_changed", "severity": "warning",
                                   "message": "{:.0%} of the ion light now falls outside the ROIs "
                                              "(was {:.0%} after arming): ions moved or were lost? "
                                              "Recalibrate.".format(recent, self.outside_baseline),
                                   "details": {"outside": recent,
                                               "baseline": self.outside_baseline}})
        return events

    def recent_frames(self) -> Optional[np.ndarray]:
        with self._lock:
            return np.asarray(list(self.frames), float) if self.frames else None

    def counts(self, since_shot: int = 0, max_shots: int = 5000) -> dict:
        with self._lock:
            first = self.shots - len(self.sums)
            start = max(since_shot, first, self.shots - max_shots)
            k0 = start - first
            sums = list(self.sums)[k0:]
            times = list(self.times)[k0:]
        return {"first_shot": start, "next_shot": start + len(sums),
                "sums": [[int(v) for v in s] for s in sums], "t": times}

    def shots_snapshot(self, last: int) -> Tuple[Optional[ReadoutPlan], np.ndarray,
                                                 np.ndarray, float]:
        """(plan, sums, dark pixels, time of the last shot) of the last shots."""
        with self._lock:
            k = len(self.sums) if last <= 0 else min(int(last), len(self.sums))
            sums = np.array(list(self.sums)[-k:]) if k else np.zeros((0, 0))
            dark = np.array(list(self.dark)[-k:]) if k else np.zeros((0, 0))
            t = self.times[-1] if self.times else 0.0
            return self.plan, sums, dark, t

    def stats(self, last: int = 2000) -> dict:
        with self._lock:
            plan = self.plan
            arr = np.array(list(self.sums)[-last:]) if self.sums else np.zeros((0, 0))
            shots = self.shots
            outside = float(np.median(list(self.outside)[-200:])) if self.outside else None
            baseline = self.outside_baseline
        out = {"shots": shots, "used": int(len(arr)), "rois": [],
               "outside_fraction": outside, "outside_baseline": baseline}
        if plan is None or arr.size == 0:
            return out
        rects = plan.sdk_rois()
        for i in range(arr.shape[1]):
            col = arr[:, i]
            u0, v0, u1, v1 = rects[i]
            px = max((u1 - u0) * (v1 - v0), 1)
            d = {"index": i, "kind": "ion" if i < plan.n_ions else "background",
                 "mean": float(col.mean()), "std": float(col.std()), "pixels": px,
                 "mean_per_pixel": float(col.mean()) / px}
            if i < plan.n_ions:
                th = fit_threshold(col)
                d["threshold"] = th
                if th.get("ok"):
                    d["bright_fraction"] = float(np.mean(col > th["threshold"]))
            out["rois"].append(d)
        return out


class IonCamEngine:
    """Owns the camera and implements everything the API exposes."""

    def __init__(self, config: Config, camera: CameraBase):
        self.config = config
        self.camera = camera
        self.detector = Detector(config.detection)
        self.tracker = ChainTracker(config.tracker)
        log_path = None
        if config.storage.event_log and config.storage.data_dir:
            log_path = os.path.join(config.storage.data_dir, "events.jsonl")
        self.events = EventLog(path=log_path)
        self.monitor = ReadoutMonitor()
        self.info: Dict[str, Any] = {}

        self._cmd: "queue.Queue[tuple]" = queue.Queue()
        self._camera_thread: Optional[threading.Thread] = None
        self._analysis_thread: Optional[threading.Thread] = None
        self._running = False
        self._started = time.time()

        self._mode = IDLE
        self._applied: Optional[AppliedSettings] = None
        self._live = dataclasses.replace(config.live)
        self._readout = dataclasses.replace(config.readout)
        self._plan: Optional[ReadoutPlan] = None
        self._plan_valid = False
        self._last_locate: Optional[dict] = None
        self._verify_shots: Optional[dict] = None     # the last verification's frames
        self._procedure: Optional[dict] = None
        self._abort = threading.Event()
        self._error: Optional[str] = None

        self._frame_cond = threading.Condition()
        self._latest: Optional[Frame] = None
        self._frame_index = 0
        self._fps = 0.0
        self._t_last_frame = 0.0
        self._analysis_cond = threading.Condition()
        self._analysis_queue: Deque[Frame] = deque(maxlen=512)
        self._tracker_lock = threading.RLock()
        self._tracker_cond = threading.Condition()
        self._snapshot: dict = self.tracker.snapshot()
        self._detection: Optional[DetectionResult] = None
        self._det_summary: dict = {}
        self._last_analysis = 0.0
        self._collector: Optional[List[Frame]] = None

        self._temperature: Tuple[float, str] = (float("nan"), "unknown")
        self._t_temperature = 0.0
        self._control: Optional[dict] = None
        self._safety_warned = 0.0

    # ======================================================================
    # lifecycle
    # ======================================================================

    def start(self, auto_live: bool = True) -> dict:
        self._running = True
        self._camera_thread = threading.Thread(target=self._camera_loop, name="ioncam-camera",
                                               daemon=True)
        self._analysis_thread = threading.Thread(target=self._analysis_loop,
                                                 name="ioncam-analysis", daemon=True)
        self._camera_thread.start()
        self._analysis_thread.start()
        self.info = self._call(self._open_camera, timeout=60.0)
        self.events.emit("started", "ioncam {} started with {}".format(
            version_string(), self.info.get("model", "camera")), "info", **self.info)
        if auto_live:
            self.start_live()
        return dict(self.info)

    def _open_camera(self) -> dict:
        info = self.camera.open()
        cc = self.config.camera
        try:
            self.camera.set_temperature(cc.target_temperature)
            if cc.cooler_on:
                self.camera.set_cooler(True)
        except CameraError as e:
            logger.warning("cooling setup: %s", e)
        self._temperature = self.camera.get_temperature()
        self._t_temperature = time.monotonic()
        return info

    def shutdown(self) -> None:
        if not self._running:
            return
        try:
            self._call(self._close_camera, timeout=30.0)
        except Exception as e:  # pragma: no cover
            logger.warning("camera close: %s", e)
        self._running = False
        self.camera.cancel_wait()
        with self._analysis_cond:
            self._analysis_cond.notify_all()
        for t in (self._camera_thread, self._analysis_thread):
            if t is not None:
                t.join(timeout=5.0)

    def _close_camera(self) -> None:
        try:
            self.camera.stop()
        finally:
            self.camera.close()
            self._mode = IDLE

    # ======================================================================
    # camera thread
    # ======================================================================

    def _call(self, fn: Callable, *args, timeout: Optional[float] = 120.0, **kwargs):
        """Run ``fn`` in the camera thread and return its result (re-raising errors)."""
        if threading.current_thread() is self._camera_thread:
            return fn(*args, **kwargs)
        if not self._running:
            raise CameraError("engine not running")
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._cmd.put((fn, args, kwargs, fut))
        self.camera.cancel_wait()
        return fut.result(timeout=timeout)

    def _camera_loop(self) -> None:
        while self._running:
            try:
                self._run_commands(block=not self.camera.is_acquiring())
                if self.camera.is_acquiring():
                    raw = self.camera.wait_frames(0.1)
                    if raw:
                        self._dispatch(raw)
                self._housekeeping()
            except CameraError as e:
                self._camera_failed(e)
            except Exception as e:  # pragma: no cover - never let the thread die
                logger.exception("camera thread")
                self._camera_failed(e)

    def _run_commands(self, block: bool) -> None:
        try:
            item = self._cmd.get(timeout=0.2) if block else self._cmd.get_nowait()
        except queue.Empty:
            return
        while True:
            fn, args, kwargs, fut = item
            if fut.set_running_or_notify_cancel():
                try:
                    fut.set_result(fn(*args, **kwargs))
                except BaseException as e:  # delivered to the caller
                    fut.set_exception(e)
            try:
                item = self._cmd.get_nowait()
            except queue.Empty:
                return

    def _camera_failed(self, e: Exception) -> None:
        self._error = str(e)
        self.events.emit("camera_error", "Camera error: {}".format(e), "alarm")
        try:
            self.camera.stop()
        except Exception:
            pass
        self._mode = ERROR
        time.sleep(0.2)

    def _housekeeping(self) -> None:
        now = time.monotonic()
        if now - self._t_temperature >= self.config.camera.temperature_poll_s:
            self._t_temperature = now
            old = self._temperature[1]
            try:
                self._temperature = self.camera.get_temperature()
            except CameraError as e:
                logger.debug("temperature: %s", e)
            new = self._temperature[1]
            if new != old and old != "unknown":
                if new == "stabilized":
                    self.events.emit("temperature_stabilized", "Sensor temperature stabilised "
                                     "at {:.1f} C".format(self._temperature[0]))
                elif old == "stabilized" and new in ("drift", "not_stabilized", "not_reached"):
                    self.events.emit("temperature_unstable", "Sensor temperature no longer "
                                     "stable ({}, {:.1f} C)".format(new, self._temperature[0]),
                                     "warning")
        c = self._control
        if c is not None and c["until"] is not None and time.time() > c["until"]:
            self._control = None
            self.events.emit("control_expired", "Control lease of '{}' expired".format(
                c["owner"]), "warning")

    def _apply(self, settings: AcquisitionSettings, tag: str) -> AppliedSettings:
        if self.camera.is_acquiring():
            self.camera.stop()
        applied = self.camera.configure(settings, tag)
        old, self._applied = self._applied, applied
        if tag == "live" and _rebaseline_needed(old if old and old.tag == "live" else None,
                                                applied):
            with self._tracker_lock:
                self.tracker.rebaseline()
        return applied

    def _dispatch(self, raw) -> List[Frame]:
        applied = self._applied
        frames = []
        now = time.time()
        for cam_index, data in raw:
            self._frame_index += 1
            f = Frame(data, self._frame_index, now, applied, cam_index)
            frames.append(f)
        if self._t_last_frame:
            dt = max(now - self._t_last_frame, 1e-6)
            inst = len(frames) / dt
            self._fps = inst if self._fps == 0 else 0.8 * self._fps + 0.2 * inst
        self._t_last_frame = now
        with self._frame_cond:
            self._latest = frames[-1]
            self._frame_cond.notify_all()
        if self._collector is not None:
            self._collector.extend(frames)
        with self._analysis_cond:
            self._analysis_queue.extend(frames)
            self._analysis_cond.notify()
        if applied is not None and applied.tag in ("live", "readout"):
            self._safety(frames[-1])
        return frames

    def _collect(self, n: int, timeout: float) -> List[Frame]:
        """Inside a procedure: acquire ``n`` frames from the running acquisition."""
        out: List[Frame] = []
        deadline = time.monotonic() + timeout
        while len(out) < n:
            if self._abort.is_set():
                raise Aborted("aborted")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProcedureError("timeout: got {} of {} frames".format(len(out), n))
            raw = self.camera.wait_frames(min(0.25, remaining))
            if raw:
                out.extend(self._dispatch(raw))
            elif not self.camera.is_acquiring() and len(out) < n:
                self.camera.start()      # single-scan modes need re-arming
        return out[:n]

    # -- safety ----------------------------------------------------------------------

    def _safety(self, frame: Frame) -> None:
        sc = self.config.safety
        a = frame.applied
        if not sc.enabled or a.settings.amplifier != "em":
            return
        floor_gain = max(sc.safe_em_gain, a.em_gain_range[0])
        if a.em_gain <= floor_gain:
            return
        n_sat = int(np.count_nonzero(frame.data >= sc.saturation_counts))
        trip = n_sat > sc.max_saturated_pixels
        if sc.bright_counts > 0:
            trip |= float(np.mean(frame.data > sc.bright_counts)) > sc.bright_fraction
        if not trip:
            return
        if a.tag == "live" and self._mode == LIVE:
            old = a.em_gain
            self._live = self._live.replace(em_gain=sc.safe_em_gain)
            self._start_live_internal()
            self.events.emit("overexposure", "Overexposure protection: {} saturated pixels at EM "
                             "gain {} - gain reduced to {}".format(n_sat, old,
                                                                 self._applied.em_gain),
                             "alarm", saturated=n_sat, old_gain=old)
        elif time.time() - self._safety_warned > 60.0:
            self._safety_warned = time.time()
            self.events.emit("overexposure", "Readout frames saturate ({} pixels at EM gain {}); "
                             "not changing settings during an experiment".format(n_sat, a.em_gain),
                             "warning", saturated=n_sat)

    # ======================================================================
    # analysis thread
    # ======================================================================

    def _analysis_loop(self) -> None:
        while self._running:
            with self._analysis_cond:
                if not self._analysis_queue:
                    self._analysis_cond.wait(0.25)
                batch = list(self._analysis_queue)
                self._analysis_queue.clear()
            if not batch:
                continue
            try:
                self._analyse(batch)
            except Exception:  # pragma: no cover - keep analysing
                logger.exception("analysis")

    def _analyse(self, batch: List[Frame]) -> None:
        readout = [f for f in batch if f.tag == "readout"]
        if readout:
            for ev in self.monitor.add(readout):
                self.events.add(ev)
                if ev["kind"] == "readout_geometry_changed":
                    self._plan_valid = False
        live = [f for f in batch if f.tag == "live"]
        if not live:
            return
        now = time.time()
        max_rate = self.config.detection.max_rate_hz
        if max_rate > 0 and now - self._last_analysis < 1.0 / max_rate:
            # keep the newest frame for the next round instead of analysing now
            with self._analysis_cond:
                self._analysis_queue.appendleft(live[-1])
            time.sleep(min(0.01, 1.0 / max_rate))
            return
        self._last_analysis = now
        frame = live[-1]
        with self._tracker_lock:
            prior = self.tracker.prior()
        det = self.detector.analyze(frame, prior=prior)
        with self._tracker_lock:
            events = self.tracker.update(det, frame.timestamp)
            snap = self.tracker.snapshot()
            psf_msg = adopt_learned_psf(self.tracker, self.config.detection)
        if psf_msg:
            self.events.emit("psf_updated", psf_msg, "info")
        self._detection = det
        self._det_summary = {
            "frame_index": frame.index, "n_spots": len(det.spots), "n_clouds": len(det.clouds),
            "background": round(det.background, 2), "noise": round(det.noise, 2),
            "noise_f": round(det.noise_f, 3), "focus": None if not math.isfinite(det.focus)
            else round(det.focus, 3), "saturated": det.saturated, "max_value": det.max_value,
            "spikes": det.spikes, "analysis_ms": round(1000 * det.analysis_time, 2),
            "spots": [s.to_dict() for s in det.spots], "clouds": [c.to_dict() for c in det.clouds],
        }
        with self._tracker_cond:
            self._snapshot = snap
            self._tracker_cond.notify_all()
        for ev in events:
            self.events.add(ev)
            sb = self.detector.static_background
            if ev["kind"] == "chain_moved" and sb is not None and not sb.info.get("moved"):
                sb.info["moved"] = True
                self.events.emit("background_check", "The crystal moved in the image: if the "
                                 "camera / imaging moved (not the ions), learn the background "
                                 "again - the surface structure moved with it", "info")
            if ev["kind"] in ("ion_lost", "ion_trapped", "chain_reconfigured", "dark_ion_moved",
                              "chain_moved") and self._plan is not None:
                if self._plan_valid:
                    self._plan_valid = False
                    self.events.emit("readout_plan_outdated", "The crystal changed since the "
                                     "readout calibration - recalibrate before the next "
                                     "experiment", "warning")

    # ======================================================================
    # public API (thread-safe; heavy work runs in the camera thread)
    # ======================================================================

    # -- status -----------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    def status(self) -> dict:
        a = self._applied
        snap = self._snapshot
        plan = self._plan
        vs = self._verify_shots
        temp, tstat = self._temperature
        last = self.events.since(max(0, self.events.last_id - 1), limit=1)
        return {
            "version": __version__,
            "mode": self._mode,
            "procedure": dict(self._procedure) if self._procedure else None,
            "acquiring": self.camera.is_acquiring(),
            "error": self._error,
            "camera": dict(self.info),
            "temperature": None if not math.isfinite(temp) else round(temp, 2),
            "temperature_status": tstat,
            "target_temperature": self.config.camera.target_temperature,
            "fps": round(self._fps, 2),
            "frame_index": self._frame_index,
            "settings": a.to_dict() if a is not None else None,
            "live_settings": self._live.to_dict(),
            "ions": {k: snap.get(k) for k in ("state", "n_bright", "n_dark", "n_ions", "stable",
                                               "stable_for", "fluorescence_level",
                                               "marker_radius", "chain",
                                               "pending", "hold", "n_spots", "n_clouds",
                                               "last_update")},
            "sites": snap.get("sites", []),
            "detection": {k: v for k, v in self._det_summary.items()
                          if k not in ("spots", "clouds")},
            "background": self.background_info(),
            "readout": {"armed": self._mode == READOUT, "has_plan": plan is not None,
                        "plan_valid": bool(plan is not None and self._plan_valid),
                        "n_ions": plan.n_ions if plan else 0,
                        "frame_size": list(plan.frame_size) if plan else None,
                        "shots": self.monitor.shots,
                        "verified": vs["t"] if vs and plan and vs["plan"] == plan.created
                        else None},
            "control": dict(self._control) if self._control else None,
            "last_event": last[0] if last else None,
            "last_event_id": self.events.last_id,
            "uptime": round(time.time() - self._started, 1),
        }

    def latest_frame(self, since_index: int = 0, timeout: float = 0.0) -> Optional[Frame]:
        with self._frame_cond:
            if timeout > 0 and (self._latest is None or self._latest.index <= since_index):
                self._frame_cond.wait(timeout)
            f = self._latest
        if f is None or f.index <= since_index:
            return None
        return f

    def detection_summary(self) -> dict:
        return dict(self._det_summary)

    def last_detection(self):
        """The most recent :class:`DetectionResult` (live frames) or None."""
        return self._detection

    def tracker_snapshot(self) -> dict:
        return dict(self._snapshot)

    # -- acquisition --------------------------------------------------------------------

    def start_live(self, **overrides) -> dict:
        if overrides:
            self._live = self._live.replace(**overrides)
            self._live.validate()
        return self._call(self._start_live_internal).to_dict()

    def _start_live_internal(self) -> AppliedSettings:
        applied = self._apply(self._live, "live")
        self.camera.start()
        self._mode = LIVE
        self._error = None
        return applied

    def update_live(self, **changes) -> dict:
        """Change live settings; applied immediately if live is running."""
        self._live = self._live.replace(**changes)
        self._live.validate()
        if self._mode == LIVE:
            return self._call(self._start_live_internal).to_dict()
        return self._live.to_dict()

    def stop(self) -> None:
        def _stop():
            self.camera.stop()
            self._mode = IDLE
        self._call(_stop)

    def abort(self) -> bool:
        """Abort a running procedure (locate / calibrate / snap)."""
        if self._procedure is None:
            return False
        self._abort.set()
        self.camera.cancel_wait()
        return True

    def _procedure_run(self, name: str, fn: Callable, *args, **kwargs):
        """Run a procedure in the camera thread, restoring the previous mode after."""
        def run():
            if self._procedure is not None:
                raise ProcedureError("'{}' is already running".format(self._procedure["name"]))
            self._abort.clear()
            self._procedure = {"name": name, "step": "starting", "started": time.time()}
            prev = self._mode
            if self.camera.is_acquiring():
                self.camera.stop()
            self._mode = BUSY
            restore = True
            try:
                result, restore = fn(*args, **kwargs)
                return result
            except Aborted:
                self.events.emit("procedure_aborted", "'{}' aborted".format(name), "warning")
                raise
            except (ProcedureError, PlanError, CameraError) as e:
                self.events.emit("procedure_failed", "'{}' failed: {}".format(name, e), "alarm")
                raise
            finally:
                self._procedure = None
                self._collector = None
                if self.camera.is_acquiring() and self._mode == BUSY:
                    self.camera.stop()
                if restore and self._mode == BUSY:
                    try:
                        self._restore(prev)
                    except CameraError as e:
                        self._camera_failed(e)
                elif self._mode == BUSY:
                    self._mode = IDLE
        return self._call(run, timeout=None)

    def _restore(self, prev: str) -> None:
        if prev == LIVE:
            self._start_live_internal()
        elif prev == READOUT and self._plan is not None:
            self._arm_internal(self._plan)
        else:
            self._mode = IDLE

    def _step(self, text: str) -> None:
        if self._procedure is not None:
            self._procedure["step"] = text
        logger.info("%s", text)

    def snap(self, n_frames: int = 1, **overrides) -> Tuple[np.ndarray, List[Frame]]:
        """Acquire ``n_frames`` with ``overrides`` applied to the live settings."""
        def proc():
            s = self._live.replace(**overrides) if overrides else self._live
            self._step("snap: {} frame(s)".format(n_frames))
            applied = self._apply(s, "snap")
            self.camera.start()
            frames = self._collect(n_frames, timeout=n_frames * (applied.kinetic_cycle * 3 + 1)
                                   + 10)
            self.camera.stop()
            mean = np.mean([f.data.astype(np.float32) for f in frames], axis=0)
            return (mean, frames), True
        return self._procedure_run("snap", proc)

    # -- locating ions ------------------------------------------------------------------

    def locate_ions(self, expected: int = 0, n_frames: Optional[int] = None,
                    include_dark: Optional[bool] = None, **overrides) -> dict:
        """Long-exposure, low-gain frames + joint fit: precise ion positions and PSF."""
        def proc():
            return self._locate(expected, n_frames, include_dark, overrides), True
        return self._procedure_run("locate_ions", proc)

    def _locate(self, expected: int, n_frames: Optional[int], include_dark: Optional[bool],
                overrides: dict) -> dict:
        cfg = self.config
        cal = cfg.calibration
        n_frames = int(n_frames or cal.n_frames)
        include_dark = cal.include_dark_sites if include_dark is None else include_dark
        s = cfg.locate.replace(**overrides) if overrides else cfg.locate
        with self._tracker_lock:
            known = self.tracker.positions(include_dark=False)
            snap = self.tracker.snapshot()
        if s.roi is None:
            if len(known) and cal.search_margin > 0:
                m = cal.search_margin
                roi = (int(math.floor(known[:, 0].min() - m)), int(math.floor(known[:, 1].min() - m)),
                       int(math.ceil(known[:, 0].max() + m)) + 1,
                       int(math.ceil(known[:, 1].max() + m)) + 1)
            else:
                roi = self._live.roi
            s = s.replace(roi=roi)
        if s.roi is not None:
            # background and noise statistics need some space around the crystal
            s = s.replace(roi=_pad_roi(s.roi, 64, self.camera.sensor_shape))
        if "hbin" not in overrides and "vbin" not in overrides:
            s = s.replace(hbin=1, vbin=1)
        self._step("locate: {} x {:.3g} s exposure(s), EM gain {}".format(n_frames, s.exposure,
                                                                         s.em_gain))
        applied = self._apply(s, "locate")
        self.camera.start()
        frames = self._collect(n_frames, timeout=n_frames * (applied.kinetic_cycle * 3 + 1) + 10)
        self.camera.stop()
        self._step("locate: fitting")
        mean = np.mean([f.data.astype(np.float32) for f in frames], axis=0)
        frame = Frame(mean, frames[-1].index, frames[-1].timestamp, applied)
        det = self.detector.analyze(frame)
        positions = det.positions
        spot_flux = np.array([sp.flux for sp in det.spots], float)
        quality: Dict[str, Any] = {"detected": len(positions)}
        if snap.get("stable") and snap.get("sites") and len(positions):
            # the crystal on screen is the one to locate: spots far from it are
            # surface light (and would make the fit and the plan slow and wrong)
            keep = _crystal_spots(positions, [[s_["x"], s_["y"]] for s_ in snap["sites"]],
                                  cfg.detection.psf_sigma,
                                  (snap.get("chain") or {}).get("scale_px") or 0.0)
            if keep.any() and not keep.all():
                positions, spot_flux = positions[keep], spot_flux[keep]
                quality["ignored"] = int((~keep).sum())
        n_det = len(positions)
        if n_det == 0:
            raise ProcedureError("no ions found in the long exposure "
                                 "(lasers on? ions in the field of view?)")
        hb, vb = applied.hbin, applied.vbin
        guesses = positions
        n_known = expected or (int(snap.get("n_bright") or 0) if snap.get("stable") else 0)
        if n_known and n_det > n_known:
            guesses = _merge_split_spots(positions, spot_flux, n_known, cfg.detection.psf_sigma)
        if expected and n_det != expected:
            guesses = self._chain_guesses(positions, expected, det, snap.get("chain")) \
                if expected > n_det else guesses
        elif not expected and snap.get("stable") and (snap.get("n_bright") or 0) > n_det:
            # the tracker sees more ions than this exposure resolved (a tight
            # pair merged into one spot): start the joint fit from its sites
            # (a guess that carries no light is dropped after the fit)
            sites = np.array([[s_["x"], s_["y"]] for s_ in snap["sites"]
                              if s_["state"] == "bright"], float).reshape(-1, 2)
            su, sv = det.to_crop(sites[:, 0], sites[:, 1])
            h_, w_ = det.residual.shape
            if np.all((su > 2) & (su < w_ - 3) & (sv > 2) & (sv < h_ - 3)):
                guesses = sites
        cu, cv = det.to_crop(guesses[:, 0], guesses[:, 1])
        psf = cfg.detection.psf_sigma
        halo = (0.0, 0.0)
        if cal.fit_psf:
            # core + halo (the halo - aberration wings - is isotropic in sensor px,
            # so only fitted without binning or with square binning)
            fit = fit_gaussians(det.residual, np.column_stack([cu, cv]),
                                sigma0=psf / math.sqrt(hb * vb), fit_offset=True,
                                halo=hb == vb)
            px = (fit.positions[:, 0] + det.crop[0]) * hb + applied.x0 + 0.5 * (hb - 1)
            py = (fit.positions[:, 1] + det.crop[1]) * vb + applied.y0 + 0.5 * (vb - 1)
            pos = np.column_stack([px, py])
            sigma = (fit.sigma[0] * hb, fit.sigma[1] * vb)
            if fit.halo_fraction > 0:
                halo = (fit.halo_fraction, fit.halo_sigma * hb)
            flux = fit.flux
            errors = fit.position_errors * np.array([hb, vb])
            quality.update(reduced_chi2=round(fit.reduced_chi2, 3), success=fit.success)
            # drop fitted "ions" that carry almost no light (spurious extra guesses)
            if len(flux) > 1:
                keep = flux > 0.2 * np.median(flux)
                if not np.all(keep):
                    pos, flux, errors = pos[keep], flux[keep], errors[keep]
        else:
            pos = positions
            sigma = (psf, psf)
            flux = spot_flux
            errors = np.full(pos.shape, np.nan)
        # dark ions known to the tracker: carry them over (shifted to the new frame)
        states = ["bright"] * len(pos)
        if include_dark and snap.get("n_dark"):
            dark = np.array([[s["x"], s["y"]] for s in snap["sites"] if s["state"] == "dark"])
            shift, pairs, _, _ = align(known, pos, 3.0, 20.0)
            if len(pairs) >= max(1, len(known) // 2):
                dark = dark + shift
                pos = np.vstack([pos, dark])
                flux = np.concatenate([flux, np.zeros(len(dark))])
                errors = np.vstack([errors, np.full(dark.shape, np.nan)])
                states += ["dark"] * len(dark)
        # order along the chain axis
        if len(pos) >= 2:
            cf = fit_chain(pos)
            order = np.argsort((pos - cf.center) @ cf.axis)
            chain = {"n": cf.n, "scale_px": round(cf.scale, 4), "rms_axial": round(cf.rms_axial, 4),
                     "rms_radial": round(cf.rms_radial, 4), "linear": cf.is_linear,
                     "angle_deg": round(math.degrees(math.atan2(cf.axis[1], cf.axis[0])), 3)}
        else:
            order = np.arange(len(pos))
            chain = None
        pos, flux, errors = pos[order], flux[order], errors[order]
        states = [states[i] for i in order]
        if cfg.detection.auto_psf and cal.fit_psf and 0.5 < sigma[0] < 6 and 0.5 < sigma[1] < 6:
            d = cfg.detection
            old_psf = PSF.from_sensor(d.psf_sigma, d.psf_halo_fraction, d.psf_halo_sigma)
            d.psf_sigma = round(float(math.sqrt(sigma[0] * sigma[1])), 4)
            if halo[0] > 0:
                # (a fit without a significant halo - e.g. read-noise limited -
                # leaves a halo measured before in place)
                d.psf_halo_fraction = round(float(halo[0]), 4)
                d.psf_halo_sigma = round(float(halo[1]), 4)
            with self._tracker_lock:
                self.tracker.psf_changed(old_psf, PSF.from_sensor(
                    d.psf_sigma, d.psf_halo_fraction, d.psf_halo_sigma))
        result = {
            "n_ions": int(len(pos)),
            "n_bright": int(sum(1 for s_ in states if s_ == "bright")),
            "positions": [[round(float(x), 4), round(float(y), 4)] for x, y in pos],
            "position_errors": [[round(float(x), 4), round(float(y), 4)] for x, y in errors],
            "states": states,
            "flux": [round(float(f), 1) for f in flux],
            "psf_sigma": [round(float(sigma[0]), 4), round(float(sigma[1]), 4)],
            "psf_halo": [round(float(halo[0]), 4), round(float(halo[1]), 4)],
            "chain": chain,
            "quality": quality,
            "exposure": applied.exposure,
            "em_gain": applied.em_gain,
            "n_frames": n_frames,
            "roi": list(applied.roi),
            "max_value": int(max(int(f.data.max()) for f in frames)),
            "t": time.time(),
        }
        if result["max_value"] >= 65000:
            result.setdefault("warnings", []).append("long exposure saturated - lower the "
                                                     "locate exposure or EM gain")
        if quality.get("ignored"):
            result.setdefault("warnings", []).append(
                "{} spot(s) away from the tracked crystal ignored - surface light? (learning "
                "the background removes it)".format(quality["ignored"]))
        self._last_locate = result
        self._save("locate", result, image=mean)
        msg = "Located {} ion(s), PSF sigma {:.2f} x {:.2f} px".format(result["n_ions"], sigma[0],
                                                                    sigma[1])
        if halo[0] > 0:
            msg += " + {:.0%} of the light in a halo of sigma {:.1f} px".format(*halo)
        self.events.emit("ions_located", msg, "info", n_ions=result["n_ions"])
        return result

    def _chain_guesses(self, positions: np.ndarray, n: int, det: Optional[DetectionResult] = None,
                       chain: Optional[dict] = None) -> np.ndarray:
        """Initial positions for ``n`` ions when the detector resolved fewer
        (dense chain): equilibrium positions spanning the detected ones.  From a
        single spot (ions merged into one) along the chain axis the tracker
        knows, spaced by its length scale - else along the spot's elongation."""
        if len(positions) == 0 or n <= len(positions):
            return positions
        if len(positions) == 1:
            u = equilibrium_positions(n)
            if chain and chain.get("scale_px"):
                a = math.radians(chain["angle_deg"])
                axis, scale = np.array([math.cos(a), math.sin(a)]), float(chain["scale_px"])
            else:
                axis, scale = self._elongation(det, positions[0], u)
            return positions[0][None, :] + scale * u[:, None] * axis[None, :]
        cf = fit_chain(positions)
        s = (positions - cf.center) @ cf.axis
        u = equilibrium_positions(n)
        scale = (s.max() - s.min()) / (u[-1] - u[0])
        center = cf.center + 0.5 * (s.max() + s.min()) * cf.axis
        return center[None, :] + scale * u[:, None] * cf.axis[None, :]

    def _elongation(self, det: Optional[DetectionResult], pos: np.ndarray, u: np.ndarray):
        """(axis, chain length scale) from the second moments of one spot that
        holds several ions in equilibrium positions ``u`` (along x, one PSF
        sigma apart, if nothing better is known)."""
        s0 = float(self.config.detection.psf_sigma)
        axis, scale = np.array([1.0, 0.0]), s0
        if det is None or det.residual.size == 0:
            return axis, scale
        cu, cv = det.to_crop(np.array([pos[0]]), np.array([pos[1]]))
        cu, cv = float(cu[0]), float(cv[0])
        m = int(math.ceil(4.0 * s0 / min(det.hbin, det.vbin))) + 2
        h, w = det.residual.shape
        u0, u1 = max(0, int(cu) - m), min(w, int(cu) + m + 1)
        v0, v1 = max(0, int(cv) - m), min(h, int(cv) + m + 1)
        patch = np.clip(det.residual[v0:v1, u0:u1].astype(np.float64), 0.0, None)
        tot = patch.sum()
        if tot <= 0:
            return axis, scale
        vv, uu = np.mgrid[v0:v1, u0:u1]
        x = (uu - cu) * det.hbin
        y = (vv - cv) * det.vbin
        mx, my = (patch * x).sum() / tot, (patch * y).sum() / tot
        cov = np.array([[(patch * (x - mx) ** 2).sum(), (patch * (x - mx) * (y - my)).sum()],
                        [0.0, (patch * (y - my) ** 2).sum()]]) / tot
        cov[1, 0] = cov[0, 1]
        ev, vec = np.linalg.eigh(cov)
        axis = vec[:, 1] if vec[0, 1] >= 0 else -vec[:, 1]
        excess = max(float(ev[1] - ev[0]), 0.25 * s0 * s0)
        return axis, math.sqrt(excess / max(float(np.mean(u ** 2)), 1e-6))

    # -- Grabber readout ------------------------------------------------------------------

    def calibrate_readout(self, n_ions: int = 0, hbin: int = 0, vbin: int = 0,
                          roi_size: Optional[Sequence[int]] = None, arm: bool = True,
                          verify: Optional[bool] = None, auto_binning: bool = False,
                          force: bool = False, locate_overrides: Optional[dict] = None,
                          **readout_overrides) -> dict:
        """Locate the ions precisely, plan the AQROI + Grabber ROIs, verify, arm."""
        def proc():
            plan = self._calibrate(n_ions, hbin, vbin, roi_size, arm, verify, auto_binning,
                                   force, locate_overrides or {}, readout_overrides)
            return plan.to_dict(), not arm
        return self._procedure_run("calibrate_readout", proc)

    def _calibrate(self, n_ions, hbin, vbin, roi_size, arm, verify, auto_binning, force,
                   locate_overrides, readout_overrides) -> ReadoutPlan:
        cfg = self.config
        cal = cfg.calibration
        expected = n_ions or cal.expected_ions
        loc = self._locate(expected, None, None, locate_overrides)
        if expected and loc["n_ions"] != expected and not force:
            raise ProcedureError("found {} ions but {} expected".format(loc["n_ions"], expected))
        if readout_overrides:
            self._readout = self._readout.replace(**readout_overrides)
            self._readout.validate()
        rs = self._readout
        hb, vb = int(hbin or rs.hbin), int(vbin or rs.vbin)
        pos = np.array(loc["positions"], float)
        sigma = tuple(loc["psf_sigma"])
        # the PSF halo (wings) sets the crosstalk: as just measured, else as known
        halo = tuple(loc["psf_halo"]) if loc.get("psf_halo", (0, 0))[0] > 0 else \
            (cfg.detection.psf_halo_fraction, cfg.detection.psf_halo_sigma)
        kw = dict(roi_size=tuple(roi_size) if roi_size else tuple(cal.roi_size),
                  frame_margin=cal.frame_margin, min_containment=cal.min_containment,
                  max_crosstalk=cal.max_crosstalk, max_rois=cfg.grabber.roi_engine_count,
                  background_roi=cal.background_roi, halo=halo)
        self._step("plan: {} ions, {}x{} binning".format(len(pos), hb, vb))
        advice = binning_advice(pos, sigma, self.camera.sensor_shape, **kw)
        if auto_binning:
            good = [a for a in advice if a.get("ok")]
            if good:
                pick = min(good, key=lambda a: (a["pixels"], -a["min_containment"]))
                hb, vb = pick["binning"]
        plan = plan_readout(pos, sigma, self.camera.sensor_shape, hb, vb,
                            flip_x=cfg.grabber.flip_x, flip_y=cfg.grabber.flip_y,
                            res_width=cfg.grabber.res_width, ion_states=loc["states"], **kw)
        for w in loc.get("warnings", []):
            plan.add_note(w)
        if self._temperature[1] != "stabilized":
            plan.add_note("sensor temperature not stabilised ({}, {:.1f} C)".format(
                self._temperature[1], self._temperature[0]))
        do_verify = cal.verify if verify is None else verify
        if do_verify:
            self._verify_with_camera(plan, force)
        plan.binning_advice = advice
        self._plan = plan
        self._plan_valid = True
        d = plan.to_dict()
        self.events.emit("readout_calibrated", "Readout planned for {} ion(s): AQROI {} "
                         "(Solis L,R,B,T), {}x{} binning, frame {}x{}, min containment "
                         "{:.0%}".format(plan.n_ions, d["aqroi_solis"], hb, vb,
                                         plan.frame_size[0], plan.frame_size[1],
                                         min(plan.containment)), "info",
                         aqroi=d["aqroi"], grabber_rois=d["grabber_rois"])
        for w in plan.warnings:
            self.events.emit("readout_warning", w, "warning")
        if arm:
            self._arm_internal(plan)
        self._save("readout_plan", plan.to_dict())
        return plan

    def _verify_with_camera(self, plan: ReadoutPlan, force: bool) -> None:
        """Frames with the readout settings (internal trigger): is every ion
        bright enough inside its ROI?  Fills ``plan.verification``."""
        cal = self.config.calibration
        vs = self._readout.replace(roi=plan.roi, hbin=plan.hbin, vbin=plan.vbin,
                                   trigger="internal", acquisition="run_till_abort",
                                   camera_link=False, fast_ext_trigger=False)
        self._step("verify: {} frames with the readout settings".format(cal.verify_frames))
        applied = self._apply(vs, "verify")
        self.camera.start()
        frames = self._collect(cal.verify_frames,
                               timeout=cal.verify_frames * (applied.kinetic_cycle * 3 + 0.5) + 10)
        self.camera.stop()
        data = [f.data for f in frames]
        enf = 2.0 if applied.settings.amplifier == "em" and applied.em_gain > 1 else 1.0
        plan.verification = verify_plan(plan, data, cal.min_verify_snr, enf=enf)
        w, h = plan.frame_size
        good = [d for d in data if d.shape == (h, w)]
        if good:
            # kept for the shot histograms (the ions shine all the time: bright shots)
            good = np.stack(good)
            self._verify_shots = {"plan": plan.created, "t": time.time(),
                                  "sums": plan.roi_sums(good),
                                  "dark": good[:, dark_pixel_mask(plan)], "enf": enf}
        if not plan.verification["ok"] and not force:
            raise ProcedureError("readout verification failed: {}".format(
                "; ".join(plan.verification["problems"])))

    def verify_readout(self, force: bool = True) -> dict:
        """Test frames with the current plan (e.g. after adjusting it by hand);
        the result goes into the plan.  ``force``: keep the plan even if the
        verification fails (default - it is only reported)."""
        def proc():
            if self._plan is None:
                raise ProcedureError("no readout plan - calibrate first")
            self._verify_with_camera(self._plan, force)
            return self._plan.to_dict(), True
        return self._procedure_run("verify_readout", proc)

    def readout_histograms(self, source: str = "auto", last: int = 20000) -> dict:
        """Histograms of the readout shots - every ion ROI and the dark
        reference - with a quick fidelity estimate per ion, every shot taken
        as prepared bright (see :func:`shot_histograms`).  ``source``:
        "trigger" (the externally triggered readout shots), "verify" (the
        frames of the last verification) or "auto" (whichever is newer)."""
        if source not in ("auto", "trigger", "verify"):
            raise ValueError('source must be "auto", "trigger" or "verify"')
        plan = self._plan
        out = {"source": None, "shots": 0, "ions": [], "series": [], "edges": [],
               "available": {"trigger": 0, "verify": 0}}
        if plan is None:
            return out
        mplan, sums, dark, t_trig = self.monitor.shots_snapshot(last)
        if mplan is None or mplan.created != plan.created:
            sums, dark, t_trig = np.zeros((0, plan.n_rois)), np.zeros((0, 0)), 0.0
        vs = self._verify_shots
        if vs is not None and vs["plan"] != plan.created:
            vs = None
        out["available"] = {"trigger": int(len(sums)), "verify": len(vs["sums"]) if vs else 0}
        if source == "auto":
            source = "verify" if vs is not None and (not len(sums) or vs["t"] > t_trig) \
                else "trigger"
        if source == "verify":
            if vs is None:
                return dict(out, source="verify")
            k = len(vs["sums"]) if last <= 0 else int(last)
            sums, dark, enf = vs["sums"][-k:], vs["dark"][-k:], vs["enf"]
        else:
            a = plan.applied or {}
            em = a.get("settings", {}).get("amplifier") == "em" and a.get("em_gain", 0) > 1
            enf = 2.0 if em else 1.0
        res = shot_histograms(plan, sums, dark, enf=enf)
        res.update(source=source, available=out["available"], plan_created=plan.created)
        return res

    def nudge_readout(self, dx: int = 0, dy: int = 0) -> dict:
        """Move the AQROI by ``(dx, dy)`` sensor px - the D-pad of the manual
        approach: the Grabber ROIs are frame coordinates and move with it, so
        the ions shift inside their ROIs (with binning by a fraction of a
        super-pixel) while the ROIs handed to ARTIQ stay the same.
        Containment and crosstalk are re-computed from the located positions;
        an armed readout is re-armed with the new AQROI."""
        def run():
            plan = self._editable_plan()
            new = reposition_plan(plan, int(dx), int(dy), self.camera.sensor_shape)
            self._set_plan(new, "AQROI moved by ({:+d}, {:+d}) px".format(int(dx), int(dy)))
            return new.to_dict()
        return self._call(run)

    def replan_readout(self, hbin: int = 0, vbin: int = 0,
                       aqroi: Optional[Sequence[int]] = None,
                       roi_size: Optional[Sequence[int]] = None) -> dict:
        """Plan again from the located positions without a new locate: another
        binning, and/or a fixed AQROI ``(x0, y0, x1, y1)`` (sensor px, half-open,
        a multiple of the binning) inside which the per-ion ROIs are placed.
        Close ions get overlapping ROIs rather than an error (see warnings).
        Re-arms an armed readout; the plan is not verified (verify_readout)."""
        def run():
            pos, sigma, halo, states, newer = self._plan_source()
            cfg = self.config
            cal = cfg.calibration
            hb = int(hbin) or (self._plan.hbin if self._plan else self._readout.hbin)
            vb = int(vbin) or (self._plan.vbin if self._plan else self._readout.vbin)
            kw = dict(roi_size=tuple(roi_size) if roi_size else tuple(cal.roi_size),
                      frame_margin=cal.frame_margin, min_containment=cal.min_containment,
                      max_crosstalk=cal.max_crosstalk, max_rois=cfg.grabber.roi_engine_count,
                      background_roi=cal.background_roi, halo=halo)
            plan = plan_readout(pos, sigma, self.camera.sensor_shape, hb, vb,
                                flip_x=cfg.grabber.flip_x, flip_y=cfg.grabber.flip_y,
                                res_width=cfg.grabber.res_width, ion_states=states,
                                aqroi=tuple(int(v) for v in aqroi) if aqroi else None, **kw)
            plan.binning_advice = binning_advice(pos, sigma, self.camera.sensor_shape, **kw)
            if newer:
                self._plan_valid = True
            self._set_plan(plan, "re-planned: {}x{} binning{}".format(
                hb, vb, ", AQROI fixed by hand" if aqroi else ""))
            return plan.to_dict()
        return self._call(run)

    def _editable_plan(self) -> ReadoutPlan:
        if self._procedure is not None:
            raise ProcedureError("busy with '{}'".format(self._procedure["name"]))
        if self._plan is None:
            raise ProcedureError("no readout plan - calibrate (or locate, then re-plan) first")
        return self._plan

    def _plan_source(self):
        """Located ions for a re-plan: the newer of the current plan and the
        last locate -> (positions, sigma, halo, states, from_new_locate)."""
        if self._procedure is not None:
            raise ProcedureError("busy with '{}'".format(self._procedure["name"]))
        plan, loc = self._plan, self._last_locate
        if loc is not None and (plan is None or loc["t"] > plan.created):
            halo = tuple(loc["psf_halo"]) if loc["psf_halo"][0] > 0 else \
                (self.config.detection.psf_halo_fraction, self.config.detection.psf_halo_sigma)
            return (np.array(loc["positions"], float), tuple(loc["psf_sigma"]), halo,
                    list(loc["states"]), True)
        if plan is None:
            raise ProcedureError("no located ions - locate them (or calibrate) first")
        return (np.array(plan.positions, float), tuple(plan.psf_sigma), tuple(plan.psf_halo),
                list(plan.ion_states), False)

    def _set_plan(self, plan: ReadoutPlan, what: str) -> None:
        """Adopt a plan changed by hand (re-arming an armed readout)."""
        self._plan = plan
        if self._mode == READOUT:
            self._arm_internal(plan)
        d = plan.to_dict()
        self.events.emit("readout_adjusted", "Readout {}: AQROI {} (Solis L,R,B,T), {}x{} "
                         "binning, containment {}".format(
                             what, d["aqroi_solis"], plan.hbin, plan.vbin,
                             " ".join("{:.0%}".format(c) for c in plan.containment)), "info",
                         aqroi=d["aqroi"], grabber_rois=d["grabber_rois"])

    def arm_readout(self) -> dict:
        """(Re-)arm the Grabber readout with the current plan."""
        def run():
            if self._plan is None:
                raise ProcedureError("no readout plan - call calibrate_readout first")
            self._arm_internal(self._plan)
            return self._plan.to_dict()
        return self._call(run)

    def _arm_internal(self, plan: ReadoutPlan) -> None:
        rs = self._readout.replace(roi=plan.roi, hbin=plan.hbin, vbin=plan.vbin)
        applied = self._apply(rs, "readout")
        w, h = plan.frame_size
        if applied.frame_shape != (h, w):
            raise ProcedureError("camera frame {} differs from the plan {}x{}".format(
                applied.frame_shape, w, h))
        plan.applied = applied.to_dict()
        self.monitor.reset(plan)
        self.camera.start()
        self._mode = READOUT
        self.events.emit("readout_armed", "Armed: {} trigger, exposure {:.4g} s, frame {}x{}, "
                         "min trigger period {:.4g} s".format(
                             rs.trigger, applied.exposure, w, h, applied.min_trigger_period),
                         "info", min_trigger_period=applied.min_trigger_period)

    def disarm_readout(self, resume_live: bool = True) -> None:
        def run():
            if self._mode == READOUT:
                self.camera.stop()
                self._mode = IDLE
                self.events.emit("readout_disarmed", "Readout disarmed")
            if resume_live:
                self._start_live_internal()
        self._call(run)

    def readout_plan(self) -> Optional[dict]:
        if self._plan is None:
            return None
        d = self._plan.to_dict()
        d["valid"] = self._plan_valid
        return d

    def check_grabber_mapping(self, image) -> dict:
        """Compare a Grabber-reconstructed frame with the SDK's view of the readout ROI."""
        if self._plan is None:
            raise ProcedureError("no readout plan")
        w, h = self._plan.frame_size
        img = np.asarray(image, float)
        if img.shape != (h, w):
            raise ProcedureError("image shape {} must be {}x{} (H x W)".format(img.shape, h, w))
        frames = self.monitor_recent_frames()
        if frames is None:
            raise ProcedureError("no readout frames received over USB to compare with")
        expected = frames.mean(axis=0)
        res = compare_grabber_frame(expected, img)
        if res.get("ambiguous"):
            res["hint"] = ("the image is nearly mirror-symmetric; make it asymmetric (e.g. one "
                           "ion dark, or a single ion off-centre) and repeat")
        return res

    def monitor_recent_frames(self) -> Optional[np.ndarray]:
        return self.monitor.recent_frames()

    # -- tracking ------------------------------------------------------------------------

    def hold_tracking(self, reason: str = "", timeout: Optional[float] = None) -> dict:
        with self._tracker_lock:
            info = self.tracker.hold(reason, timeout)
            self._snapshot = self.tracker.snapshot()
        self.events.emit("hold_started", "Tracking on hold: {}".format(reason or "external "
                                                                                "control"))
        return info

    def resume_tracking(self) -> Optional[dict]:
        with self._tracker_lock:
            info = self.tracker.release()
            self._snapshot = self.tracker.snapshot()
        if info is not None:
            self.events.emit("hold_released", "Tracking resumed; checking the crystal")
        return info

    def relearn(self) -> dict:
        det = self._detection
        with self._tracker_lock:
            evs = self.tracker.relearn(det)
            self._snapshot = self.tracker.snapshot()
        for ev in evs:
            self.events.add(ev)
        return dict(self._snapshot)

    def reset_tracking(self) -> None:
        with self._tracker_lock:
            self.tracker.reset()
            self._snapshot = self.tracker.snapshot()

    def ion_count_ok(self, n: int, at_least: bool = False, stable_time: Optional[float] = None,
                     bright_only: bool = True) -> Tuple[bool, dict]:
        snap = self._snapshot
        count = snap["n_bright"] if bright_only else snap["n_ions"]
        st = self.config.tracker.stable_time if stable_time is None else stable_time
        stable = snap["state"] in ("stable", "empty") and snap.get("pending") is None and \
            (snap.get("stable_for", 0.0) >= st or (snap["state"] == "empty" and n == 0))
        ok = (count >= n if at_least else count == n) and stable
        return ok, {"n_bright": snap["n_bright"], "n_dark": snap["n_dark"],
                    "n_ions": snap["n_ions"], "state": snap["state"], "stable": stable}

    def wait_for_ions(self, n: int, timeout: float = 60.0, at_least: bool = False,
                      stable_time: Optional[float] = None, bright_only: bool = True) -> dict:
        deadline = time.monotonic() + timeout
        with self._tracker_cond:
            while True:
                ok, info = self.ion_count_ok(n, at_least, stable_time, bright_only)
                remaining = deadline - time.monotonic()
                if ok or remaining <= 0:
                    info["ok"] = ok
                    return info
                self._tracker_cond.wait(min(remaining, 0.5))

    # -- background -------------------------------------------------------------------------

    def learn_background(self, n_frames: int = 50, mask_ions: bool = True,
                         full_sensor: bool = True, **overrides) -> dict:
        """Learn the static background structure (surface scatter, defects, hot
        pixels): ``n_frames`` with the live settings (plus ``overrides``, e.g.
        a longer exposure), by default of the full sensor without binning so
        the result applies to any later ROI and binning.

        ``mask_ions``: leave out the tracked ion sites (ions may stay trapped).
        Even better: learn with the ions dark (repumper off) or without ions,
        and ``mask_ions=False``."""
        def proc():
            s = self._live.replace(**overrides) if overrides else self._live
            if full_sensor:
                s = s.replace(roi=None, hbin=1, vbin=1)
            self._step("background: {} frame(s)".format(n_frames))
            applied = self._apply(s, "background")
            self.camera.start()
            frames = self._collect(int(n_frames),
                                   timeout=n_frames * (applied.kinetic_cycle * 3 + 1) + 10)
            self.camera.stop()
            with self._tracker_lock:
                mask = self.tracker.positions(include_dark=True) if mask_ions else None
                n_clouds = self.tracker.n_clouds
            sb = self.detector.learn_static_background(
                frames, mask, info={"mask_ions": bool(mask_ions)})
            with self._tracker_lock:
                self.tracker.background_changed()
            info = sb.summary()
            msg = "Static background learned from {} frame(s) of {:.3g} s ({} px of structure" \
                  "{})".format(len(frames), applied.exposure, info["structure_px"],
                               ", {} ion sites masked".format(info["masked_sites"])
                               if mask_ions else "")
            self.events.emit("background_learned", msg + ")", "info")
            if n_clouds:
                self.events.emit("background_warning", "A cloud was present while learning the "
                                 "background - it may have been learned as background",
                                 "warning")
            return info, True
        return self._procedure_run("learn_background", proc)

    def background_info(self) -> Optional[dict]:
        sb = self.detector.static_background
        return sb.summary() if sb is not None else None

    def clear_background(self) -> None:
        self.detector.clear_static_background()
        with self._tracker_lock:
            self.tracker.background_changed()

    def sim_trap_turned(self) -> None:
        """[simulation] The simulated trap was turned in the image, electrodes
        and surface with it - as if a different trap had been mounted: the
        learned background (of the old orientation) is dropped, the crystal is
        tracked afresh (the ions jumping to new places is nothing a real trap
        does; nor is the turned surface light an ion cloud) and a readout plan
        is outdated."""
        learned = self.detector.static_background is not None
        self.detector.clear_static_background()
        self.reset_tracking()
        if learned:
            self.events.emit("background_cleared", "The trap was turned (simulation): the "
                             "learned background was of the old orientation and is cleared - "
                             "learn it again", "warning")
        if self._plan is not None and self._plan_valid:
            self._plan_valid = False
            self.events.emit("readout_plan_outdated", "The trap was turned (simulation) - "
                             "recalibrate the readout", "warning")

    # -- control lease ---------------------------------------------------------------------

    def take_control(self, owner: str, reason: str = "", timeout: Optional[float] = 600.0,
                     force: bool = False) -> dict:
        c = self._control
        now = time.time()
        if c is not None and c["owner"] != owner and not force:
            raise ProcedureError("camera is controlled by '{}' ({})".format(c["owner"],
                                                                            c["reason"]))
        self._control = {"owner": owner, "reason": reason, "since": now,
                         "until": None if not timeout else now + timeout}
        if c is None or c["owner"] != owner:
            self.events.emit("control_taken", "'{}' took control: {}".format(owner, reason))
        return dict(self._control)

    def release_control(self, owner: Optional[str] = None, force: bool = False) -> bool:
        c = self._control
        if c is None:
            return False
        if owner is not None and c["owner"] != owner and not force:
            raise ProcedureError("control belongs to '{}'".format(c["owner"]))
        self._control = None
        self.events.emit("control_released", "'{}' released control".format(c["owner"]))
        return True

    def control(self) -> Optional[dict]:
        return dict(self._control) if self._control else None

    # -- temperature / config ----------------------------------------------------------------

    def set_temperature(self, temperature: float) -> None:
        self.config.camera.target_temperature = int(round(temperature))
        self._call(self.camera.set_temperature, temperature)

    def set_cooler(self, on: bool) -> None:
        self._call(self.camera.set_cooler, bool(on))

    def temperature(self) -> Tuple[float, str]:
        return self._temperature

    def set_config(self, section: str, key: str, value) -> Any:
        v = self.config.set(section, key, value)
        if section == "live":
            self._live = self._live.replace(**{key: v})
            if self._mode == LIVE:
                self._call(self._start_live_internal)
        elif section == "readout":
            self._readout = self._readout.replace(**{key: v})
        elif section == "camera" and key == "target_temperature":
            self._call(self.camera.set_temperature, v)
        return v

    # -- storage -----------------------------------------------------------------------------

    def _save(self, kind: str, data: dict, image: Optional[np.ndarray] = None) -> None:
        st = self.config.storage
        if not st.save_calibrations or not st.data_dir:
            return
        try:
            d = os.path.join(st.data_dir, "calibrations")
            os.makedirs(d, exist_ok=True)
            stem = os.path.join(d, time.strftime("%Y%m%d-%H%M%S") + "_" + kind)
            with open(stem + ".json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1, default=str)
            if image is not None:
                np.save(stem + ".npy", image.astype(np.float32))
        except OSError as e:  # pragma: no cover
            logger.warning("could not save %s: %s", kind, e)

    # -- simulation ----------------------------------------------------------------------------

    def sim_scene(self):
        """The simulated trap (for demos/tests) or None for real hardware."""
        cam = self.camera
        scene = getattr(cam, "scene", None)
        if scene is None:
            sdk = getattr(cam, "_sdk", None)
            sim = getattr(sdk, "sim", None)
            scene = getattr(sim, "scene", None)
        return scene

    def sim_camera(self):
        cam = self.camera
        if hasattr(cam, "trigger"):
            return cam
        sdk = getattr(cam, "_sdk", None)
        return getattr(sdk, "sim", None)
