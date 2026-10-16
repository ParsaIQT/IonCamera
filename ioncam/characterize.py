"""Camera characterization: dark frames at many readout settings, for SPAM.

With the shutter closed, the camera is run at a sweep of readout settings
around the readout's own (``[readout]``): horizontal readout speed x
pre-amp gain, vertical shift speed x clock voltage, EM gain, exposure
(mostly below 10 ms), on-chip binning, software-triggered single shots with
idle time between them (like Grabber readout shots), optionally the full
grid of readout electronics and other sensor temperatures.  Every setting's
dark frames give (:mod:`ioncam.analysis.darkframes`): bias and its
structure and drift, read noise, the measured EM gain, parallel and serial
clock-induced charge, hot pixels, charge trails, first-frame effects, and the
full histogram of counts.  Together: the dark current (against exposure and
temperature), the real EM gain against the set one, how binning adds noise
up, and how stable it all is over the run.  Finally the SPAM error every
setting gives for an ion of a given brightness, and the setting that
reaches a target error in the shortest detection time.

Everything goes into ``<storage.data_dir>/characterization/<time>_<camera>/``
as it is measured (an aborted run keeps what it has):

* ``plan.json`` - the settings to measure, the camera, the options;
* ``points.jsonl`` - one line per measured setting (all its numbers);
* ``hist/NNNN.npz`` - its histogram of counts above bias (and maps for some);
* ``summary.json`` / ``summary.csv`` - all of it, the global results and the
  SPAM ranking;
* ``camera_profile.json`` - the dark model at every setting, for
  simulations (``simulator.camera_profile``);
* ``README.txt`` - what the files hold.
"""

from __future__ import annotations

import concurrent.futures
import csv
import dataclasses
import datetime
import json
import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ioncam.analysis.darkframes import (analyze_dark_stack, detection_curve,
                                        fit_dark_histogram, scaled_model, shortest_detection)
from ioncam.camera.base import AcquisitionSettings, CameraError, normalize_roi
from ioncam.camera.profile import write_profile

logger = logging.getLogger("ioncam")

STAGES = {
    "baseline": "the readout settings, at the start, every stability_every_s and at the end "
                "(bias, CIC and temperature drift)",
    "sensor": "the whole sensor at the readout settings (hot pixels, CIC against the row, "
              "bias structure)",
    "electronics": "every horizontal readout speed x pre-amp gain (read noise, bias, serial CIC)",
    "vertical": "every vertical shift speed x clock voltage (parallel CIC)",
    "gain": "EM gains (the real gain against the set one, CIC against the gain)",
    "exposure": "exposures, mostly below 10 ms (CIC against dark current, light leaks)",
    "binning": "on-chip binnings (super-pixel noise of readout ROIs)",
    "triggered": "software-triggered single shots with idle time between them, like Grabber "
                 "readout shots",
    "conventional": "the conventional amplifier's speeds x pre-amp gains (read noise without "
                    "EM gain)",
    "grid": "every HS speed x pre-amp gain x VS speed x clock voltage at the grid gains",
    "temperature": "baseline and exposures at other sensor set points (dark current against "
                   "temperature)",
}
ORDER = ("baseline", "sensor", "electronics", "vertical", "gain", "exposure", "binning",
         "triggered", "conventional", "grid", "temperature")
MAX_BYTES = 256 * 2 ** 20           # frames of one setting kept in memory at most
LIGHT_RATE = 50.0                   # electrons/pixel/s over the region: light, not dark current
LIGHT_SPOT_RATE = 20.0              # ... in a 3x3 spot (ions, laser light)


@dataclass
class Point:
    """One setting to measure."""
    index: int
    stage: str
    label: str
    settings: AcquisitionSettings
    frames: int
    max_s: float
    spacing: Optional[float] = None    # software trigger: idle time between shots [s]
    temperature: Optional[float] = None
    keep_maps: bool = False
    stages: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"index": self.index, "stage": self.stage, "stages": list(self.stages),
                "label": self.label, "settings": self.settings.to_dict(),
                "frames": self.frames, "max_s": self.max_s, "spacing": self.spacing,
                "temperature": self.temperature, "keep_maps": self.keep_maps}

    def key(self) -> tuple:
        s = self.settings
        return (s.amplifier, round(s.hs_speed_mhz, 3), s.preamp_gain_index,
                round(s.vs_speed_us, 3), s.vs_amplitude, int(s.em_gain) if
                s.amplifier == "em" else 0, round(s.exposure, 7), s.hbin, s.vbin,
                tuple(s.roi) if s.roi else None, s.trigger, s.frame_transfer,
                None if self.spacing is None else round(self.spacing, 6), self.temperature)


def _readout_estimate(s: AcquisitionSettings, sensor_shape, vs_us: float, hs_mhz: float) -> float:
    """Rough readout time of one frame [s] (for planning; the camera's own
    number is recorded)."""
    x0, y0, x1, y1 = normalize_roi(s.roi, sensor_shape, s.hbin, s.vbin)
    rows, cols = (y1 - y0) // s.vbin, (x1 - x0) // s.hbin
    return sensor_shape[0] * max(vs_us, 0.3) * 1e-6 + rows * (cols + 20) / (max(hs_mhz, 0.05) *
                                                                            1e6) + 2e-4


def estimate_seconds(p: Point, sensor_shape, discard: int = 2) -> float:
    s = p.settings
    ro = _readout_estimate(s, sensor_shape, s.vs_speed_us or 0.5, s.hs_speed_mhz)
    if p.spacing is not None:
        per = s.exposure + ro + p.spacing + 2e-3
    elif s.frame_transfer:
        per = max(s.exposure, ro) + 1e-4
    else:
        per = s.exposure + ro + 2e-4
    return min(p.max_s, (p.frames + discard) * per) + 0.8


def default_region(engine, size: int) -> Tuple[int, int, int, int]:
    """A square region of ``size`` px: on the readout AQROI, else the ions,
    else the sensor centre."""
    h, w = engine.camera.sensor_shape
    cx, cy = w / 2.0, h / 2.0
    plan = getattr(engine, "_plan", None)
    if plan is not None:
        x0, y0, x1, y1 = plan.roi
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    else:
        try:
            pos = engine.tracker.positions()
            if len(pos):
                cx, cy = float(np.mean(pos[:, 0])), float(np.mean(pos[:, 1]))
        except Exception:
            pass
    size = int(min(max(size, 16), w, h))
    x0 = int(min(max(round(cx - size / 2), 0), w - size))
    y0 = int(min(max(round(cy - size / 2), 0), h - size))
    return (x0, y0, x0 + size, y0 + size)


def build_plan(opts: dict, readout: AcquisitionSettings, caps: dict, sensor_shape,
               region: Tuple[int, int, int, int]) -> List[Point]:
    """The settings to measure, stage by stage, around ``readout``; settings
    measured already in an earlier stage are not repeated (their point
    names all its stages)."""
    stages = [s for s in ORDER if s in set(opts["stages"])]
    hs_all = caps.get("hs_speeds_mhz") or {}
    hs_em = [float(v) for v in hs_all.get("em") or [readout.hs_speed_mhz]]
    hs_conv = [float(v) for v in hs_all.get("conventional") or []]
    vs_list = [float(v) for v in caps.get("vs_speeds_us") or [readout.vs_speed_us or 0.5]]
    fastest = caps.get("vs_fastest_recommended") or vs_list[min(
        int(caps.get("vs_fastest_recommended_index", 0)), len(vs_list) - 1)]
    n_pa = len(caps.get("preamp_gains") or [1.0])
    n_amp = int(caps.get("n_vs_amplitudes") or 5)
    amps = [a for a in (opts.get("vs_amplitudes") or range(n_amp)) if 0 <= int(a) < n_amp]
    em_lo, em_hi = (caps.get("em_gain_range") or [1, 300])[:2]
    em_lo, em_hi = max(int(em_lo), 1), max(int(em_hi), 1)

    def gain_ok(g):
        return em_lo <= g <= em_hi

    base = readout.replace(roi=tuple(region), hbin=1, vbin=1, trigger="internal",
                           acquisition="run_till_abort", camera_link=False,
                           kinetic_cycle_time=0.0, n_accumulations=1,
                           vs_speed_us=readout.vs_speed_us or float(fastest),
                           em_gain=min(max(int(readout.em_gain), em_lo), em_hi)
                           if readout.amplifier == "em" else readout.em_gain)
    if base.amplifier != "em":
        base = base.replace(amplifier="em", hs_speed_mhz=hs_em[0])
    if base.preamp_gain_index < 0:
        base = base.replace(preamp_gain_index=n_pa - 1)
    samples = float(opts["samples_per_point"])

    def frames_for(s: AcquisitionSettings, n_min: Optional[int] = None) -> int:
        x0, y0, x1, y1 = normalize_roi(s.roi, sensor_shape, s.hbin, s.vbin)
        npx = max(((x1 - x0) // s.hbin) * ((y1 - y0) // s.vbin), 1)
        n = int(math.ceil(samples / npx))
        n = min(max(n, int(n_min if n_min is not None else opts["min_frames"])),
                int(opts["max_frames"]), int(MAX_BYTES // (4 * npx)))
        return max(n, 6)

    pts: List[Point] = []
    seen: Dict[tuple, Point] = {}

    def add(stage, label, s, spacing=None, temperature=None, keep_maps=False, n_min=None,
            repeat=False):
        p = Point(len(pts), stage, label, s, frames_for(s, n_min), float(opts["max_point_s"]),
                  spacing, temperature, keep_maps, [stage])
        k = p.key()
        if not repeat and k in seen:
            if stage not in seen[k].stages:
                seen[k].stages.append(stage)
            return
        seen.setdefault(k, p)
        pts.append(p)

    def pa_label(i):
        texts = caps.get("preamp_gain_texts") or []
        return texts[i] if i < len(texts) else "pre-amp {}".format(i)

    for stage in stages:
        if stage == "baseline":
            add("baseline", "readout settings", base, keep_maps=True, repeat=True)
        elif stage == "sensor":
            add("sensor", "whole sensor", base.replace(roi=None), keep_maps=True, n_min=40)
        elif stage == "electronics":
            for hs in hs_em:
                for pa in range(n_pa):
                    add(stage, "{:g} MHz, {}".format(hs, pa_label(pa)),
                        base.replace(hs_speed_mhz=hs, preamp_gain_index=pa))
        elif stage == "vertical":
            for vs in vs_list:
                for a in amps:
                    add(stage, "VS {:g} us, voltage {}".format(vs, "+{}".format(a) if a else
                                                                 "normal"),
                        base.replace(vs_speed_us=vs, vs_amplitude=int(a)))
        elif stage == "gain":
            for g in opts["em_gains"]:
                if gain_ok(int(g)):
                    add(stage, "EM gain {}".format(int(g)), base.replace(em_gain=int(g)))
        elif stage == "exposure":
            for t in opts["exposures_ms"]:
                add(stage, "exposure {:g} ms".format(float(t)),
                    base.replace(exposure=float(t) / 1000.0))
        elif stage == "binning":
            bins = [(int(b), int(b)) for b in opts["binnings"]]
            rb = (int(readout.hbin), int(readout.vbin))
            if rb not in bins:
                bins.append(rb)
            for hb, vb in bins:
                add(stage, "binning {}x{}".format(hb, vb), base.replace(hbin=hb, vbin=vb),
                    n_min=int(opts["min_frames"]))
        elif stage == "triggered":
            trig = base.replace(trigger="software", frame_transfer=readout.frame_transfer,
                                fast_ext_trigger=readout.fast_ext_trigger)
            for sp in opts["trigger_spacings_ms"]:
                add(stage, "triggered, {:g} ms between shots".format(float(sp)), trig,
                    spacing=float(sp) / 1000.0, n_min=min(int(opts["min_frames"]), 60))
        elif stage == "conventional":
            for hs in hs_conv:
                for pa in range(n_pa):
                    add(stage, "conventional {:g} MHz, {}".format(hs, pa_label(pa)),
                        base.replace(amplifier="conventional", hs_speed_mhz=hs,
                                     preamp_gain_index=pa, em_gain=0))
        elif stage == "grid":
            gains = sorted({min(max(int(g) if int(g) > 0 else int(base.em_gain), em_lo), em_hi)
                            for g in opts["grid_gains"]})
            for g in gains:
                for hs in hs_em:
                    for pa in range(n_pa):
                        for vs in vs_list:
                            for a in amps:
                                add(stage, "{:g} MHz, {}, VS {:g} us{}, gain {}".format(
                                    hs, pa_label(pa), vs, " +{}".format(a) if a else "", g),
                                    base.replace(hs_speed_mhz=hs, preamp_gain_index=pa,
                                                 vs_speed_us=vs, vs_amplitude=int(a),
                                                 em_gain=g))
        elif stage == "temperature":
            for temp in opts["temperatures"]:
                add(stage, "{:g} C, readout settings".format(float(temp)), base,
                    temperature=float(temp), keep_maps=True)
                for t in opts["exposures_ms"]:
                    add(stage, "{:g} C, exposure {:g} ms".format(float(temp), float(t)),
                        base.replace(exposure=float(t) / 1000.0), temperature=float(temp))
    if "baseline" in stages and len(pts) > 1:
        add("baseline", "readout settings (end)", base, keep_maps=True, repeat=True)
    for i, p in enumerate(pts):
        p.index = i
    return pts


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


class Characterization:
    """One run: plan, measurement (in the engine's camera thread), analysis
    (in a worker thread), files and status."""

    def __init__(self, engine, options: Optional[dict] = None):
        self.eng = engine
        cfg = engine.config
        opts = dataclasses.asdict(cfg.characterize)
        unknown = set(options or {}) - set(opts) - {"region", "force", "stop_on_light"}
        if unknown:
            raise ValueError("unknown characterization option(s): {}".format(
                ", ".join(sorted(unknown))))
        opts.update({k: v for k, v in (options or {}).items() if k in opts})
        opts["stop_on_light"] = bool((options or {}).get("stop_on_light", True))
        self.light: Optional[dict] = None     # the light check (see _light_check)
        bad = [s for s in opts["stages"] if s not in STAGES]
        if bad:
            raise ValueError("unknown stage(s) {}; known: {}".format(
                ", ".join(bad), ", ".join(ORDER)))
        self.opts = opts
        cam = engine.camera
        self.caps = cam.capabilities()
        self.sensor_shape = tuple(cam.sensor_shape)
        region = (options or {}).get("region")
        self.region = tuple(int(v) for v in region) if region else \
            default_region(engine, int(opts["region_size"]))
        self.readout = dataclasses.replace(cfg.readout)
        self.points = build_plan(opts, self.readout, self.caps, self.sensor_shape, self.region)
        if not self.points:
            raise ValueError("nothing to measure: no stage selected")
        self.discard = int(opts["discard_frames"])
        self.estimate_s = sum(estimate_seconds(p, self.sensor_shape, self.discard)
                              for p in self.points)
        info = dict(engine.info)
        name = re.sub(r"[^A-Za-z0-9]+", "-", str(info.get("model", "camera"))).strip("-")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        root = cfg.storage.data_dir or "ioncam_data"
        self.directory = os.path.join(root, "characterization",
                                      "{}_{}".format(stamp, name[:40] or "camera"))
        self.camera_info = info
        self.results: List[dict] = []
        self._lock = threading.Lock()
        self._pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="characterize")
        self._pending: List[concurrent.futures.Future] = []
        self.state = "planned"
        self.started = 0.0
        self.finished = 0.0
        self.current: Optional[dict] = None
        self.notes: List[str] = []
        self.error: Optional[str] = None
        self.summary: Optional[dict] = None
        self._durations: List[Tuple[float, float]] = []    # (estimated, measured) s
        self._last_baseline = 0.0
        self._n_stability = 0
        self._next = 0
        self._applied_keys: Dict[tuple, int] = {}

    # -- description ------------------------------------------------------------------

    def describe(self) -> dict:
        counts: Dict[str, int] = {}
        for p in self.points:
            counts[p.stage] = counts.get(p.stage, 0) + 1
        return {"directory": os.path.abspath(self.directory), "points": len(self.points),
                "stages": counts, "estimate_s": round(self.estimate_s, 1),
                "region": list(self.region), "options": _jsonable(self.opts)}

    def status(self) -> dict:
        with self._lock:
            n_done = len(self.results)
            recent = [_brief(r) for r in self.results[-12:]]
        now = time.time()
        elapsed = (self.finished or now) - self.started if self.started else 0.0
        remaining = sum(estimate_seconds(p, self.sensor_shape, self.discard)
                        for p in self.points[self._next:]) if self.state == "running" else 0.0
        if self._durations:
            est, real = (sum(v) for v in zip(*self._durations))
            remaining *= real / max(est, 1e-9)
        return {"state": self.state, "directory": os.path.abspath(self.directory),
                "points": len(self.points), "done": n_done, "measured": self._next,
                "progress": n_done / max(len(self.points), 1),
                "elapsed_s": round(elapsed, 1), "eta_s": round(remaining, 1),
                "current": self.current, "notes": list(self.notes), "error": self.error,
                "recent": recent, "estimate_s": round(self.estimate_s, 1)}

    # -- the run (camera thread) ----------------------------------------------------------

    def proc(self):
        """Run inside the engine's procedure (camera thread)."""
        from ioncam.engine import Aborted
        eng, cam = self.eng, self.eng.camera
        os.makedirs(os.path.join(self.directory, "hist"), exist_ok=True)
        self.state = "running"
        self.started = time.time()
        self._write_json("plan.json", {"camera": self.camera_info, "capabilities": self.caps,
                                       "region": list(self.region),
                                       "readout": self.readout.to_dict(),
                                       "options": self.opts, "started": _now(),
                                       "points": [p.to_dict() for p in self.points]})
        prev_shutter = eng.shutter_settings()
        prev_target = eng.config.camera.target_temperature
        eng._long_procedure = "camera characterization"
        aborted = None
        final = "done"          # the state once the summary is written
        try:
            eng._step("characterization: closing the shutter")
            eng.set_shutter(mode="closed")
            internal = (self.caps.get("shutter") or {}).get("internal")
            if internal is False:
                self._note("this camera has no internal shutter: closing it only switches the "
                           "shutter output - cover the camera, or the frames are not dark")
            self._wait_temperature(prev_target)
            self._light_check()
            set_point = float(prev_target)
            for i, p in enumerate(self.points):
                self._next = i
                if p.temperature is not None and p.temperature != set_point:
                    set_point = float(p.temperature)
                    cam.set_temperature(set_point)
                    self._wait_temperature(set_point)
                elif p.temperature is None and set_point != float(prev_target):
                    set_point = float(prev_target)
                    cam.set_temperature(set_point)
                    self._wait_temperature(set_point)
                if self._stability_due(p):
                    self._run_point(self._stability_point(), extra=True)
                self._run_point(p)
            self._next = len(self.points)
        except Aborted as e:
            aborted = e
            final = "aborted"
        except Exception as e:
            final = "failed"
            self.error = str(e)
            logger.exception("camera characterization")
            raise
        finally:
            self.state = "finishing"
            eng._long_procedure = None
            try:
                cam.stop()
            except CameraError:
                pass
            try:
                if float(eng.config.camera.target_temperature) != float(prev_target) or \
                        any(p.temperature is not None for p in self.points):
                    cam.set_temperature(float(prev_target))
            except CameraError as e:
                self._note("could not restore the set point: {}".format(e))
            try:
                eng.set_shutter(mode=prev_shutter["mode"], external=prev_shutter["external"],
                                ttl_high=prev_shutter["ttl_high"],
                                open_ms=prev_shutter["open_ms"],
                                close_ms=prev_shutter["close_ms"])
            except Exception as e:
                self._note("could not restore the shutter: {}".format(e))
            eng._step("characterization: finishing the analysis")
            for f in list(self._pending):
                try:
                    f.result(timeout=600)
                except Exception as e:
                    logger.warning("characterization analysis: %s", e)
            self._pool.shutdown(wait=True)
            self.finished = time.time()
            try:
                self.summary = self._finish(final)
            except Exception as e:      # the measured points are on disk regardless
                logger.exception("characterization summary")
                self._note("summary failed: {}".format(e))
            self.state = final
        if aborted is not None:
            raise aborted
        return _brief_summary(self.summary), True

    def _note(self, text: str) -> None:
        self.notes.append(text)
        logger.info("characterization: %s", text)
        self.eng.events.emit("characterization", text, "warning")

    def _check_abort(self):
        from ioncam.engine import Aborted
        if self.eng._abort.is_set():
            raise Aborted("aborted")

    def _wait_temperature(self, target: float) -> None:
        cam = self.eng.camera
        timeout = float(self.opts["temperature_timeout_s"])
        t0 = time.monotonic()
        while True:
            t, st = cam.get_temperature()
            self.eng._temperature = (t, st)
            if timeout <= 0:
                return
            if st in ("stabilized", "off", "not_supported", "unknown"):
                if st == "off":
                    self._note("the cooler is off: dark current and CIC depend on the "
                               "temperature")
                return
            if time.monotonic() - t0 > timeout:
                self._note("the sensor temperature did not settle at {:g} C within {:.0f} s "
                           "({:.1f} C, {})".format(target, timeout, t, st))
                return
            self.eng._step("characterization: waiting for the sensor temperature ({:.1f} C, "
                           "{}, set point {:g} C)".format(t, st.replace("_", " "), target))
            for _ in range(20):
                self._check_abort()
                time.sleep(0.1)

    def _light_check(self) -> None:
        """Is the sensor dark?  20 ms frames against 1 ms ones at the readout
        settings, pixel by pixel: dark current is ~0.001 electrons/pixel/s
        on a cold sensor, so more than ``LIGHT_RATE`` over the region (a
        leak) or in a spot of a few pixels (ions, laser light: no internal
        shutter) - or saturated pixels - mean light gets in, and the run
        stops (``stop_on_light``) before it measures light as CIC or ages
        the EM register."""
        from scipy import ndimage
        eng, cam = self.eng, self.eng.camera
        base = self.points[0].settings
        g = int(base.em_gain) if base.amplifier == "em" else 1
        eng._step("characterization: is the sensor dark?")
        means = []
        for t in (0.001, 0.02):
            applied = eng._apply(base.replace(exposure=t, hbin=1, vbin=1), "characterize")
            frames = []
            deadline = time.monotonic() + 10.0
            cam.start()
            try:
                while len(frames) < 12 and time.monotonic() < deadline:
                    self._check_abort()
                    raw = cam.wait_frames(0.25)
                    if raw:
                        frames += [f.data for f in eng._dispatch(raw)]
                    elif not cam.is_acquiring():
                        cam.start()
            finally:
                cam.stop()
            if len(frames) < 4:
                self._note("light check skipped: only {} frames".format(len(frames)))
                return
            stack = np.stack(frames[2:] if len(frames) > 6 else frames)
            sat = float(np.mean(stack >= int(eng.config.safety.saturation_counts)))
            means.append((applied, stack.astype(np.float64).mean(axis=0), sat))
        (a1, m1, sat1), (a2, m2, sat2) = means
        sens = float(a2.sensitivity or 0.0)
        per_e = g / sens if sens > 0 else float(g)       # counts per photo-electron
        dt = max(a2.exposure - a1.exposure, 1e-6)
        d = m2 - m1                                      # counts, per pixel
        level = float(np.median(d))
        # light covers most of a 3 x 3 neighbourhood; a hot pixel (or a few)
        # does not and the median takes it out
        spot = ndimage.median_filter(d - level, size=3, mode="nearest")
        noise = 1.4826 * float(np.median(np.abs(spot - np.median(spot))))
        lit = spot > max(8.0 * noise, LIGHT_SPOT_RATE * dt * per_e)
        n_lit = int(np.count_nonzero(lit))
        rate = float(np.mean(d)) / dt / max(per_e, 1e-9)
        peak = float(np.max(spot)) / dt / max(per_e, 1e-9)
        y, x = np.unravel_index(int(np.argmax(spot)), spot.shape)
        x0, y0 = int(a2.roi[0]), int(a2.roi[1])
        temp, _ = cam.get_temperature()
        cold = temp == temp and temp < -30.0
        sat = max(sat1, sat2)
        leak = cold and rate > LIGHT_RATE and level > 3.0
        self.light = {"e_per_px_s": rate, "spot_e_per_px_s": peak, "spot_xy": [x0 + int(x),
                                                                                y0 + int(y)],
                      "lit_pixels": n_lit, "saturated_fraction": sat, "temperature": temp,
                      "em_gain": g, "light": bool(sat > 1e-3 or leak or (cold and n_lit >= 4))}
        if not self.light["light"]:
            return
        if sat > 1e-3:
            what = "pixels saturate"
        elif leak:
            what = "~{:.3g} electrons/pixel/s over the region".format(rate)
        else:
            what = ("a spot of {} pixels around ({}, {}) at up to {:.3g} electrons/pixel/s - "
                    "ions or laser light?".format(n_lit, x0 + int(x), y0 + int(y), peak))
        self._light_found("light reaches the sensor with the shutter closed: {} ({:.0f} C) - "
                          "cover the camera (no internal shutter, or a leak) and start "
                          "again".format(what, temp))

    def _light_found(self, msg: str) -> None:
        from ioncam.engine import Aborted
        self._note(msg)
        if self.opts.get("stop_on_light", True):
            self.error = msg
            raise Aborted(msg)

    def _saturated(self, stack: np.ndarray, applied) -> None:
        """Light during the run (a lamp, a shutter opened): frames that
        saturate at EM gain stop it, as the overexposure protection would."""
        sc = self.eng.config.safety
        if applied.settings.amplifier != "em" or applied.em_gain <= max(sc.safe_em_gain, 1):
            return
        per_frame = np.count_nonzero(stack >= int(sc.saturation_counts), axis=(1, 2))
        if float(np.median(per_frame)) > sc.max_saturated_pixels:
            self.light = dict(self.light or {}, light=True, saturated_during_run=True)
            self._light_found("frames saturate at EM gain {} ({:.0f} pixels per frame): light "
                              "reaches the sensor - stopping to protect the EM register".format(
                                  applied.em_gain, float(np.median(per_frame))))

    def _stability_due(self, p: Point) -> bool:
        every = float(self.opts["stability_every_s"])
        if every <= 0 or "baseline" not in self.opts["stages"] or p.stage == "baseline":
            return False
        return time.time() - self._last_baseline > every

    def _stability_point(self) -> Point:
        base = next((p for p in self.points if p.stage == "baseline"), self.points[0])
        self._n_stability += 1
        return dataclasses.replace(base, index=-self._n_stability, label="readout settings "
                                   "(stability {})".format(self._n_stability), stages=["baseline"])

    def _run_point(self, p: Point, extra: bool = False) -> None:
        eng, cam = self.eng, self.eng.camera
        self._check_abort()
        if p.stage == "baseline":
            self._last_baseline = time.time()
        self.current = {"index": p.index, "stage": p.stage, "label": p.label}
        n_total = len(self.points)
        eng._step("characterization {}/{}: {} - {}".format(
            max(self._next + 1, 1), n_total, p.stage, p.label))
        if eng._procedure is not None:
            eng._procedure["progress"] = round(len(self.results) / max(n_total, 1), 4)
            st = self.status()
            eng._procedure["eta_s"] = st["eta_s"]
        t0 = time.time()
        temp_before = cam.get_temperature()
        eng._temperature = temp_before
        try:
            applied = eng._apply(p.settings, "characterize")
        except (CameraError, ValueError) as e:
            self._record_failure(p, "the camera refused these settings: {}".format(e))
            return
        dup = self._duplicate(p, applied)
        if dup is not None and not extra:
            self._record_failure(p, "the camera used the same settings as point {} ({})".format(
                dup, "; ".join(applied.warnings) or "substituted"), skipped=True)
            return
        dropped0 = getattr(cam, "dropped_frames", 0)
        frames, first, stamps = [], [], []
        deadline = time.monotonic() + p.max_s
        want = p.frames + self.discard
        cam.start()
        try:
            if p.spacing is None:
                while len(frames) + len(first) < want and time.monotonic() < deadline:
                    self._check_abort()
                    raw = cam.wait_frames(0.25)
                    if raw:
                        for f in eng._dispatch(raw):
                            if len(first) < self.discard:
                                first.append(f.data)
                            else:
                                frames.append(f.data)
                                stamps.append(f.timestamp)
                    elif not cam.is_acquiring():
                        cam.start()
            else:
                wait = applied.exposure + applied.readout_time + 1.0
                while len(frames) + len(first) < want and time.monotonic() < deadline:
                    self._check_abort()
                    cam.send_software_trigger()
                    t_trig = time.monotonic()
                    got = None
                    while got is None and time.monotonic() - t_trig < wait:
                        raw = cam.wait_frames(0.1)
                        if raw:
                            got = eng._dispatch(raw)[-1]
                    if got is not None:
                        if len(first) < self.discard:
                            first.append(got.data)
                        else:
                            frames.append(got.data)
                            stamps.append(got.timestamp)
                    end = time.monotonic() + p.spacing
                    while time.monotonic() < end:
                        self._check_abort()
                        time.sleep(min(0.05, max(end - time.monotonic(), 0.0)))
        finally:
            cam.stop()
        temp_after = cam.get_temperature()
        eng._temperature = temp_after
        seconds = time.time() - t0
        if not extra:
            self._durations.append((estimate_seconds(p, self.sensor_shape, self.discard),
                                    seconds))
        meta = {"index": p.index, "stage": p.stage, "stages": list(p.stages), "label": p.label,
                "settings": p.settings.to_dict(), "applied": _applied_brief(applied),
                "spacing_s": p.spacing, "temperature_set": p.temperature,
                "temperature_before": list(temp_before), "temperature_after": list(temp_after),
                "time": t0, "duration_s": round(seconds, 3),
                "dropped": int(getattr(cam, "dropped_frames", 0) - dropped0),
                "keep_maps": p.keep_maps}
        if len(frames) < 4:
            self._record_failure(p, "only {} frames in {:.0f} s".format(len(frames), p.max_s),
                                 meta=meta)
            return
        stack = np.stack(frames)
        self._saturated(stack, applied)
        first_stack = np.stack(first) if first else None
        while len(self._pending) >= 3:       # keep the memory bounded
            self._pending.pop(0).result()
        self._pending.append(self._pool.submit(self._analyze, p, meta, stack, first_stack,
                                               stamps, applied))

    def _duplicate(self, p: Point, applied) -> Optional[int]:
        s = applied.settings
        key = (s.amplifier, round(applied.hs_speed_mhz, 3), round(applied.preamp_gain, 4),
               round(applied.vs_speed_us, 3), s.vs_amplitude, applied.em_gain,
               round(applied.exposure, 7), s.hbin, s.vbin, tuple(applied.roi), s.trigger,
               None if p.spacing is None else round(p.spacing, 6), p.temperature)
        if p.stage == "baseline":
            return None
        if key in self._applied_keys:
            return self._applied_keys[key]
        self._applied_keys[key] = p.index
        return None

    def _record_failure(self, p: Point, why: str, skipped: bool = False,
                        meta: Optional[dict] = None) -> None:
        r = dict(meta or {"index": p.index, "stage": p.stage, "stages": list(p.stages),
                          "label": p.label, "settings": p.settings.to_dict()},
                 ok=False, skipped=skipped, note=why)
        with self._lock:
            self.results.append(r)
        self._append_line(r)
        if not skipped:
            self._note("{}: {}".format(p.label, why))

    def _analyze(self, p: Point, meta: dict, stack, first, stamps, applied) -> None:
        a = applied
        try:
            stats, arrays = analyze_dark_stack(
                stack, em_gain=float(a.em_gain) if a.settings.amplifier == "em" else 1.0,
                sensitivity=float(a.sensitivity), first=first, origin=(a.roi[0], a.roi[1]),
                binning=(a.settings.hbin, a.settings.vbin), timestamps=stamps,
                keep_maps=p.keep_maps)
        except Exception as e:      # one setting's analysis must not stop the run
            logger.exception("characterization: analysis of %s", p.label)
            self._record_failure(p, "analysis failed: {}".format(e), meta=meta)
            return
        r = dict(meta, ok=bool(stats.get("fit", {}).get("ok")), stats=_jsonable(stats))
        name = "hist/{:04d}.npz".format(p.index) if p.index >= 0 else \
            "hist/stability{:02d}.npz".format(-p.index)
        try:
            np.savez_compressed(os.path.join(self.directory, name), **arrays)
            r["hist_file"] = name
        except OSError as e:
            logger.warning("characterization: %s", e)
        with self._lock:
            self.results.append(r)
        self._append_line(r)

    # -- files ------------------------------------------------------------------------------

    def _write_json(self, name: str, data) -> None:
        with open(os.path.join(self.directory, name), "w", encoding="utf-8") as f:
            json.dump(_jsonable(data), f, indent=1, default=str)

    def _append_line(self, r: dict) -> None:
        try:
            with open(os.path.join(self.directory, "points.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(_jsonable(r), default=str) + "\n")
        except OSError as e:
            logger.warning("characterization: %s", e)

    def _finish(self, state: str) -> dict:
        with self._lock:
            results = sorted(self.results, key=lambda r: (r.get("time") or 0.0))
        glob = global_results(results, self.caps)
        if refit_with_calibrated_gain(results, glob, self.directory):
            glob = global_results(results, self.caps)
        spam = spam_params(self.eng, self.opts, self.readout)
        ranking = spam_ranking(results, glob, spam)
        summary = {"camera": self.camera_info, "region": list(self.region),
                   "readout": self.readout.to_dict(), "options": self.opts,
                   "started": _iso(self.started), "finished": _iso(self.finished),
                   "state": state, "notes": self.notes, "light_check": self.light,
                   "global": glob, "spam": spam,
                   "ranking": ranking[:200], "points": results}
        self._write_json("summary.json", summary)
        write_csv(os.path.join(self.directory, "summary.csv"), results)
        write_profile_from(os.path.join(self.directory, "camera_profile.json"), results, glob,
                           self.camera_info)
        with open(os.path.join(self.directory, "README.txt"), "w", encoding="utf-8") as f:
            f.write(README)
        best = ranking[0] if ranking and ranking[0].get("t_min_ms") else None
        msg = "Camera characterization {}: {} settings measured in {:.0f} min".format(
            state, sum(1 for r in results if r.get("ok")), (self.finished - self.started)
            / 60.0)
        if best:
            msg += "; best for SPAM: {} ({:.2f} ms to reach {:g})".format(
                best["label"], best["t_min_ms"], spam["target_error"])
        self.eng.events.emit("characterization_done", msg, "info",
                             directory=os.path.abspath(self.directory))
        return summary


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _iso(t: float) -> Optional[str]:
    return datetime.datetime.fromtimestamp(t).isoformat(timespec="seconds") if t else None


def _applied_brief(a) -> dict:
    return {"exposure": a.exposure, "kinetic_cycle": a.kinetic_cycle,
            "readout_time": a.readout_time, "keep_clean_time": a.keep_clean_time,
            "hs_speed_mhz": a.hs_speed_mhz, "vs_speed_us": a.vs_speed_us,
            "preamp_gain": a.preamp_gain, "em_gain": a.em_gain,
            "em_gain_real": a.em_gain_real, "sensitivity": a.sensitivity,
            "sensitivity_source": a.sensitivity_source, "roi": list(a.roi),
            "frame_shape": list(a.frame_shape), "warnings": list(a.warnings)}


def _brief(r: dict) -> dict:
    """A point's main numbers (status, tables)."""
    st = r.get("stats") or {}
    a = r.get("applied") or {}
    s = r.get("settings") or {}
    fit = st.get("fit") or {}
    return {"index": r.get("index"), "stage": r.get("stage"), "label": r.get("label"),
            "ok": r.get("ok"), "note": r.get("note") or fit.get("note"),
            "amplifier": s.get("amplifier"), "hs_speed_mhz": a.get("hs_speed_mhz"),
            "preamp_gain": a.get("preamp_gain"), "vs_speed_us": a.get("vs_speed_us"),
            "vs_amplitude": s.get("vs_amplitude"), "em_gain": a.get("em_gain"),
            "exposure_ms": None if a.get("exposure") is None else 1e3 * a["exposure"],
            "binning": [s.get("hbin"), s.get("vbin")], "spacing_ms": None if
            r.get("spacing_s") is None else 1e3 * r["spacing_s"],
            "temperature": (r.get("temperature_after") or [None])[0],
            "frames": st.get("frames"), "bias": st.get("bias"),
            "read_noise": st.get("read_noise"), "read_noise_e": st.get("read_noise_e"),
            "g": fit.get("g"), "g_err": (fit.get("errors") or {}).get("g"),
            "em_gain_measured": st.get("em_gain_measured"),
            "cic_parallel": (st.get("model") or fit).get("lam_p"),
            "cic_serial": (st.get("model") or fit).get("lam_s"),
            "cic_parallel_fit": fit.get("lam_p"), "cic_serial_fit": fit.get("lam_s"),
            "model_source": st.get("model_source"), "events_per_px": st.get("events_per_px"),
            "n_hot": st.get("n_hot"), "chi2_dof": fit.get("chi2_dof"),
            "readout_time_ms": None if a.get("readout_time") is None else
            1e3 * a["readout_time"], "common_mode_rms": st.get("common_mode_rms")}


def _brief_summary(summary: Optional[dict]) -> dict:
    if not summary:
        return {}
    return {"state": summary.get("state"), "global": summary.get("global"),
            "best": (summary.get("ranking") or [None])[0]}


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


def _weighted_line(x, y, w):
    x, y, w = (np.asarray(v, float) for v in (x, y, w))
    if x.size < 2 or np.ptp(x) <= 0:
        return None
    sw = w.sum()
    xm, ym = (w * x).sum() / sw, (w * y).sum() / sw
    sxx = (w * (x - xm) ** 2).sum()
    slope = (w * (x - xm) * (y - ym)).sum() / sxx
    intercept = ym - slope * xm
    resid = y - (intercept + slope * x)
    dof = max(x.size - 2, 1)
    s2 = (w * resid ** 2).sum() / dof
    return slope, intercept, math.sqrt(max(s2 / sxx, 0.0))


def global_results(results: List[dict], caps: Optional[dict] = None) -> dict:
    """What the settings say together: dark current and light leaks, the EM
    gain calibration, binning, stability, first frames, triggered shots,
    hot pixels."""
    good = [r for r in results if r.get("ok") and r.get("stats")]
    out: Dict = {}

    def fit_of(r):
        """The fit the numbers come from: the gain-calibrated one where the
        dark frames did not pin the gain down (see refit_with_calibrated_gain)."""
        st = r["stats"]
        return st.get("fit_calibrated") or st.get("fit") or {}

    def temp_of(r):
        return (r.get("temperature_after") or [None])[0]

    # dark current: parallel events against the exposure (per pixel, 1x1)
    for label, rows in (("exposure", [r for r in good if r["stage"] == "exposure" or
                                      ("exposure" in r.get("stages", []) and
                                       r.get("temperature_set") is None)]),):
        rows = [r for r in rows if r["settings"]["hbin"] == 1 and r["settings"]["vbin"] == 1
                and fit_of(r).get("model") == "emccd"]
        if len(rows) >= 2:
            t = [r["applied"]["exposure"] for r in rows]
            lam = [fit_of(r)["lam_p"] for r in rows]
            err = [max(fit_of(r)["lam_p"] * ((fit_of(r).get("errors") or {}).get("lam_p") or
                                             0.1), 1e-7) for r in rows]
            line = _weighted_line(t, lam, [1.0 / e ** 2 for e in err])
            if line:
                d, cic, d_err = line
                out["dark_current"] = {"e_per_px_s": d, "error": d_err, "cic_at_zero": cic,
                                       "temperature": float(np.nanmean(
                                           [temp_of(r) for r in rows if temp_of(r) is not None]
                                           or [float("nan")])),
                                       "points": [[ti, li] for ti, li in zip(t, lam)]}
    by_temp: Dict[float, list] = {}
    for r in good:
        if r.get("temperature_set") is not None and fit_of(r).get("model") == "emccd":
            by_temp.setdefault(float(r["temperature_set"]), []).append(r)
    dct = []
    for temp, rows in sorted(by_temp.items()):
        if len(rows) >= 2:
            line = _weighted_line([r["applied"]["exposure"] for r in rows],
                                  [fit_of(r)["lam_p"] for r in rows],
                                  [1.0 for _ in rows])
            if line:
                dct.append({"set_point": temp, "temperature": float(np.nanmean(
                    [temp_of(r) for r in rows])), "e_per_px_s": line[0], "cic": line[1]})
    if dct:
        out["dark_current_temperature"] = dct
    dc = (out.get("dark_current") or {}).get("e_per_px_s")
    temp = (out.get("dark_current") or {}).get("temperature")
    if dc is not None and dc > 1.0 and (temp is None or not temp > -40):
        out["light_leak"] = ("{:.3g} electrons/pixel/s with the shutter closed at {:.0f} C - far "
                             "more than dark current: light reaches the sensor (no internal "
                             "shutter, or a leak)").format(dc, temp if temp is not None else
                                                           float("nan"))
    # EM gain: measured against set - by the free fit (a gain-calibrated one
    # only took the gain from here)
    gains = []
    for r in good:
        st = r["stats"]
        f = st.get("fit") or {}
        if st.get("em_gain_measured") and f.get("model") == "emccd" and not f.get("g_fixed"):
            gains.append({"index": r["index"], "stage": r["stage"],
                          "set": r["applied"]["em_gain"], "measured": st["em_gain_measured"],
                          "error": (f.get("errors") or {}).get("g"),
                          "hs_speed_mhz": r["applied"]["hs_speed_mhz"],
                          "preamp_gain": r["applied"]["preamp_gain"]})
    if gains:
        # the real / set gain: the error-weighted mean over the settings that
        # measure it (EM gain >= 20, gain error < 10 %)
        sure = [g for g in gains if g["error"] is not None and 0 < g["error"] < 0.10 and
                g["set"] >= 20]
        if sure:
            ratios = np.array([g["measured"] / g["set"] for g in sure])
            w = np.array([1.0 / g["error"] ** 2 for g in sure])
            mean = float(np.sum(w * ratios) / np.sum(w))
            out["gain_ratio"] = mean
            out["gain_ratio_error"] = float(mean / math.sqrt(np.sum(w)))
            out["gain_ratio_spread"] = float(np.std(ratios))
            if len(sure) > 1:
                chi2 = float(np.sum(w * (ratios / mean - 1.0) ** 2) / (len(sure) - 1))
                out["gain_ratio_chi2_dof"] = chi2
                # the settings disagree more than their errors say: a wider error
                out["gain_ratio_error"] *= math.sqrt(max(chi2, 1.0))
        out["gain"] = gains
    # binning: noise of a super-pixel against the pixels it adds up; the
    # reference is every unbinned setting with the same electronics
    def electronics(r):
        a, s = r["applied"], r["settings"]
        return (s["amplifier"], a["hs_speed_mhz"], a["preamp_gain"], a["vs_speed_us"],
                s["vs_amplitude"], a["em_gain"], r.get("spacing_s"), r.get("temperature_set"))

    rows = []
    for r in good:
        if "binning" not in r.get("stages", [r["stage"]]) or fit_of(r).get("model") != "emccd":
            continue
        hb, vb = r["settings"]["hbin"], r["settings"]["vbin"]
        f = fit_of(r)
        ref = [x for x in good if electronics(x) == electronics(r) and
               x["settings"]["hbin"] == 1 and x["settings"]["vbin"] == 1 and
               fit_of(x).get("model") == "emccd" and x["applied"]["exposure"] <= 0.02]
        exp_lam = None
        if ref:
            lam = np.array([fit_of(x)["lam_p"] for x in ref])
            rel = np.array([(fit_of(x).get("errors") or {}).get("lam_p") or 0.2 for x in ref])
            w = 1.0 / np.maximum(lam * rel, 1e-9) ** 2
            exp_lam = float(np.sum(w * lam) / np.sum(w)) * hb * vb
        rows.append({"binning": [hb, vb], "bias": r["stats"].get("bias"),
                     "read_noise": r["stats"].get("read_noise"),
                     "cic_parallel": f.get("lam_p"), "cic_serial": f.get("lam_s"),
                     "cic_parallel_expected": exp_lam, "g": f.get("g")})
    if rows:
        out["binning"] = sorted(rows, key=lambda d: d["binning"][0] * d["binning"][1])
    # stability: the baseline over the run
    base = sorted([r for r in good if r["stage"] == "baseline"], key=lambda r: r["time"])
    if len(base) >= 2:
        out["stability"] = {
            "times_min": [(r["time"] - base[0]["time"]) / 60.0 for r in base],
            "bias": [r["stats"].get("bias") for r in base],
            "cic_parallel": [fit_of(r).get("lam_p") for r in base],
            "read_noise": [r["stats"].get("read_noise") for r in base],
            "temperature": [temp_of(r) for r in base]}
    # the first frames after every start
    firsts = []
    for r in good:
        ff = r["stats"].get("first_frames")
        ev = r["stats"].get("events_per_px")
        if ff and ev:
            firsts.append([ff[0]["offset"], ff[0]["events_per_px"] / ev])
    if firsts:
        a = np.array(firsts)
        out["first_frame"] = {"offset_median": float(np.median(a[:, 0])),
                              "event_ratio_median": float(np.median(a[:, 1])),
                              "event_ratio_max": float(np.max(a[:, 1]))}
    trig = [r for r in good if r["stage"] == "triggered"]
    if trig:
        out["triggered"] = [{"spacing_ms": 1e3 * (r.get("spacing_s") or 0.0),
                             "bias": r["stats"].get("bias"),
                             "cic_parallel": fit_of(r).get("lam_p"),
                             "events_per_px": r["stats"].get("events_per_px")} for r in trig]
    sensor = [r for r in good if r["stage"] == "sensor"]
    if sensor:
        st = sensor[0]["stats"]
        out["hot_pixels"] = st.get("hot_pixels") or []
        out["cic_row_trend_per_100_rows"] = st.get("cic_row_trend_per_100_rows")
    trails = [r["stats"].get("trail") or {} for r in good]
    if trails:
        for k in ("right", "left", "down"):
            v = [t[k] for t in trails if k in t]
            if v:
                out.setdefault("trail", {})[k] = float(np.median(v))
    return _jsonable(out)


def refit_with_calibrated_gain(results: List[dict], glob: dict, directory: str) -> int:
    """Settings whose dark frames do not pin the gain down (error > 5 %, or
    unknown - low EM gain, few events): fitted again with the gain fixed at
    the set gain x the measured real/set ratio / electrons per count, so
    their CIC is the camera's and not a trade-off against a wrong gain.
    Their prediction model is that fit (``model_source``: "fit", "gain
    calibrated" or "uncertain").  Returns how many were fitted again."""
    ratio = glob.get("gain_ratio")
    n = 0
    for r in results:
        st = r.get("stats") or {}
        fit = st.get("fit") or {}
        if not r.get("ok") or fit.get("model") != "emccd":
            continue
        if st.get("fit_calibrated"):
            continue
        g_err = (fit.get("errors") or {}).get("g")
        if g_err is not None and g_err <= 0.05 and not fit.get("g_fixed"):
            st["model_source"] = "fit"
            continue
        a = r.get("applied") or {}
        sens = float(a.get("sensitivity") or 0.0)
        path = os.path.join(directory, r["hist_file"]) if r.get("hist_file") else ""
        if not ratio or sens <= 0 or not path or not os.path.exists(path):
            st["model_source"] = "uncertain"
            continue
        with np.load(path) as z:
            hist = np.asarray(z["hist"], np.int64)
            lo = int(np.asarray(z["hist_lo"]).ravel()[0])
        g_cal = float(a["em_gain"]) * float(ratio) / sens
        f2 = fit_dark_histogram(hist, lo, float(a["em_gain"]), g_hint=g_cal, fix_g=True)
        if not f2.get("ok"):
            st["model_source"] = "uncertain"
            continue
        widen = math.sqrt(1.0 + math.pi / max(int(st.get("frames") or 1), 1))
        st["fit_calibrated"] = _jsonable(f2)
        st["model"] = {"sigma": float(f2["sigma"]) / widen, "g": float(f2["g"]),
                       "lam_p": float(f2["lam_p"]), "lam_s": float(f2["lam_s"]),
                       "a": float(f2["a"]), "mu": float(f2["mu"])}
        st["model_source"] = "gain calibrated"
        n += 1
    return n


def spam_params(engine, opts: dict, readout: AcquisitionSettings) -> dict:
    """The ion and detection the SPAM ranking is for: options, else what
    ioncam measures (the live brightness, the readout plan's ROI light)."""
    rate = float(opts.get("ion_rate") or 0.0)
    if rate <= 0:
        try:
            rate = float(engine.brightness().get("mean_bright") or 0.0)
        except Exception:
            rate = 0.0
    source = "option" if opts.get("ion_rate") else ("measured live" if rate > 0 else "default")
    if rate <= 0:
        rate = 8000.0
    frac = float(opts.get("roi_fraction") or 0.0)
    if frac <= 0:
        plan = getattr(engine, "_plan", None)
        cont = getattr(plan, "containment", None) if plan is not None else None
        frac = float(np.mean(cont)) if cont else 0.6
    return {"ion_rate": rate, "ion_rate_source": source, "roi_fraction": frac,
            "roi_pixels": int(opts.get("roi_pixels") or 1),
            "background_rate": float(opts.get("background_rate") or 0.0),
            "binning": [int(readout.hbin), int(readout.vbin)],
            "detection_ms": [float(v) for v in opts.get("detection_ms") or []],
            "target_error": float(opts.get("target_error") or 1e-3),
            "bright_lifetime_ms": float(opts.get("bright_lifetime_ms") or 0.0),
            "dark_lifetime_ms": float(opts.get("dark_lifetime_ms") or 0.0)}


def spam_ranking(results: List[dict], glob: dict, spam: dict) -> List[dict]:
    """Every EM setting's SPAM error for the ion of ``spam`` (see
    :func:`spam_params`), best first: by the shortest detection time
    reaching the target error, then by the lowest error.  Each setting's
    dark model is moved to the readout binning and the detection time (dark
    current); the signal's counts per photo-electron are the fitted ones
    when the fit pins them down (< 3 %), else the set gain x the measured
    real/set ratio / electrons per count."""
    rows = []
    d_cur = (glob.get("dark_current") or {}).get("e_per_px_s") or 0.0
    d_cur = max(float(d_cur), 0.0)
    ratio = glob.get("gain_ratio")
    hb, vb = spam["binning"]
    k = int(spam["roi_pixels"])
    rate = float(spam["ion_rate"]) * float(spam["roi_fraction"])
    bg = float(spam["background_rate"])
    tb = float(spam["bright_lifetime_ms"]) / 1e3
    td = float(spam["dark_lifetime_ms"]) / 1e3
    seen = set()
    for r in results:
        st = r.get("stats") or {}
        f = st.get("fit") or {}
        model = st.get("model")
        if not r.get("ok") or f.get("model") != "emccd" or not model:
            continue
        if r.get("temperature_set") is not None:
            continue
        a, s = r["applied"], r["settings"]
        key = (a["hs_speed_mhz"], a["preamp_gain"], a["vs_speed_us"], s["vs_amplitude"],
               a["em_gain"], s["hbin"], s["vbin"], r.get("spacing_s"),
               round(a["exposure"], 6))
        if key in seen:
            continue
        seen.add(key)
        m = scaled_model(model, (hb, vb), (s["hbin"], s["vbin"]), 0.0, a["exposure"], d_cur)
        g_err = (f.get("errors") or {}).get("g")
        g_sig = 0.0
        if (g_err is None or g_err > 0.03) and ratio and a.get("sensitivity"):
            g_sig = float(a["em_gain"]) * float(ratio) / float(a["sensitivity"])
        kw = dict(background_rate=bg, n_pixels=k, bright_lifetime=tb, dark_lifetime=td,
                  dark_current=d_cur, area=hb * vb, g_signal=g_sig)
        best = shortest_detection(m, rate, float(spam["target_error"]), **kw)
        curve = detection_curve(m, rate, [t / 1e3 for t in spam["detection_ms"]], **kw)
        bias = float(st.get("bias") or 0.0)
        row = dict(_brief(r))
        row.update({
            "t_min_ms": None if best.get("time") is None else 1e3 * best["time"],
            "error_at_t_min": best.get("error"), "best_error": best.get("best_error"),
            "best_time_ms": None if best.get("best_time") is None else 1e3 * best["best_time"],
            "threshold": best.get("threshold"),
            "threshold_raw": None if best.get("threshold") is None else
            int(round(k * bias + best["threshold"])),
            "errors": {"{:g}".format(t): c["error"] for t, c in zip(spam["detection_ms"],
                                                                    curve)},
            "dark_errors": {"{:g}".format(t): c["dark_error"] for t, c in
                            zip(spam["detection_ms"], curve)},
            "bright_errors": {"{:g}".format(t): c["bright_error"] for t, c in
                              zip(spam["detection_ms"], curve)},
            "g_signal": g_sig or m["g"], "model": m,
            "readout_settings": {"amplifier": "em", "hs_speed_mhz": a["hs_speed_mhz"],
                                 "preamp_gain_index": s["preamp_gain_index"],
                                 "vs_speed_us": a["vs_speed_us"],
                                 "vs_amplitude": s["vs_amplitude"], "em_gain": a["em_gain"]}})
        rows.append(row)

    def order(row):
        if row["t_min_ms"] is not None:
            return (0, row["t_min_ms"], row["best_error"] or 1.0)
        return (1, row["best_error"] if row["best_error"] is not None else 1.0, 0.0)
    rows.sort(key=order)
    for i, row in enumerate(rows):
        row["rank"] = i + 1
    return _jsonable(rows)


CSV_COLUMNS = ("index", "stage", "label", "ok", "note", "amplifier", "hs_speed_mhz",
               "preamp_gain", "vs_speed_us", "vs_amplitude", "em_gain", "exposure_ms",
               "binning", "spacing_ms", "temperature", "frames", "bias", "common_mode_rms",
               "read_noise", "read_noise_e", "g", "g_err", "em_gain_measured", "cic_parallel",
               "cic_serial", "model_source", "cic_parallel_fit", "cic_serial_fit",
               "events_per_px", "n_hot", "chi2_dof", "readout_time_ms")


def write_csv(path: str, results: List[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in results:
            b = _brief(r)
            w.writerow(["x".join(str(v) for v in b[c]) if c == "binning" and b[c] else
                        ("" if b.get(c) is None else b[c]) for c in CSV_COLUMNS])


def write_profile_from(path: str, results: List[dict], glob: dict, camera: dict) -> None:
    points = []
    for r in results:
        st = r.get("stats") or {}
        f = st.get("fit") or {}
        if not r.get("ok") or not st.get("model"):
            continue
        if r.get("temperature_set") is not None or r.get("spacing_s") is not None:
            continue
        a, s = r["applied"], r["settings"]
        points.append({"stage": r["stage"], "amplifier": s["amplifier"],
                       "hs_speed_mhz": a["hs_speed_mhz"],
                       "preamp_gain_index": s["preamp_gain_index"],
                       "preamp_gain": a["preamp_gain"], "vs_speed_us": a["vs_speed_us"],
                       "vs_amplitude": s["vs_amplitude"], "em_gain": a["em_gain"],
                       "hbin": s["hbin"], "vbin": s["vbin"], "exposure": a["exposure"],
                       "bias": st.get("bias"), "read_noise": st["model"]["sigma"],
                       "sensitivity": a.get("sensitivity") or None,
                       "g": f.get("g") if f.get("model") == "emccd" else None,
                       "lam_p": f.get("lam_p"), "lam_s": f.get("lam_s"),
                       "common_mode_rms": st.get("common_mode_rms"),
                       "temperature": (r.get("temperature_after") or [None])[0]})
    dc = glob.get("dark_current") or {}
    dct = [[d["temperature"], d["e_per_px_s"]] for d in glob.get("dark_current_temperature", [])]
    write_profile(path, camera, points, dc.get("temperature"),
                  max(float(dc.get("e_per_px_s") or 0.0), 0.0), dct, glob.get("gain_ratio"),
                  glob.get("hot_pixels") or [], _now())


def load_results(directory: str) -> dict:
    """A characterization's results from its directory: the summary of a
    finished run, else what a stopped one measured."""
    path = os.path.join(directory, "summary.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    with open(os.path.join(directory, "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    points = _read_points(directory)
    glob = global_results(points, plan.get("capabilities"))
    if refit_with_calibrated_gain(points, glob, directory):
        glob = global_results(points, plan.get("capabilities"))
    return {"camera": plan.get("camera"), "region": plan.get("region"),
            "readout": plan.get("readout"), "options": plan.get("options"),
            "started": plan.get("started"), "state": "incomplete", "notes": [],
            "global": glob, "points": points}


def list_results(root: str) -> List[dict]:
    d = os.path.join(root or "ioncam_data", "characterization")
    if not os.path.isdir(d):
        return []
    out = []
    for name in sorted(os.listdir(d), reverse=True):
        p = os.path.join(d, name)
        if os.path.exists(os.path.join(p, "plan.json")):
            n = 0
            pl = os.path.join(p, "points.jsonl")
            if os.path.exists(pl):
                with open(pl, encoding="utf-8") as f:
                    n = sum(1 for line in f if line.strip())
            out.append({"name": name, "directory": os.path.abspath(p), "points": n,
                        "complete": os.path.exists(os.path.join(p, "summary.json"))})
    return out


def _read_points(directory: str) -> List[dict]:
    p = os.path.join(directory, "points.jsonl")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def point_detail(directory: str, index: int) -> dict:
    """One measured setting: its record, histogram and the fitted model's
    expected counts in every histogram bin."""
    from ioncam.analysis.darkframes import dark_model_pmf
    rec = next((r for r in _read_points(directory) if r.get("index") == index), None)
    if rec is None:
        raise KeyError("no setting {} in {}".format(index, directory))
    out = {"point": _brief(rec), "record": rec, "lo": None, "counts": None, "model": None}
    name = rec.get("hist_file")
    if name and os.path.exists(os.path.join(directory, name)):
        with np.load(os.path.join(directory, name)) as z:
            hist = np.asarray(z["hist"], np.int64)
            lo = int(np.asarray(z["hist_lo"]).ravel()[0])
        out["lo"] = lo
        out["counts"] = hist
        fit = (rec.get("stats") or {}).get("fit") or {}
        if fit.get("ok") and hist.size:
            hi = lo + hist.size - 1
            if fit.get("model") == "emccd":
                pmf = dark_model_pmf(lo, hi, fit["mu"], fit["sigma"], fit["g"], fit["lam_p"],
                                     fit["lam_s"], fit["a"])
            else:
                x = np.arange(lo, hi + 1, dtype=float)
                from scipy.special import ndtr
                pmf = ndtr((x + 0.5 - fit["mu"]) / fit["sigma"]) - \
                    ndtr((x - 0.5 - fit["mu"]) / fit["sigma"])
            out["model"] = pmf / max(pmf.sum(), 1e-300) * float(hist.sum())
    return out


def rank(results_or_summary: dict, engine=None, **params) -> Tuple[List[dict], dict]:
    """Re-rank a characterization's settings for another ion / detection
    (``params`` as the ``[characterize]`` SPAM options, ``binning`` [h, v]).
    Returns (ranking, the parameters used)."""
    summary = results_or_summary
    opts = dict(summary.get("options") or {})
    opts.update({k: v for k, v in params.items() if v is not None})
    ro = summary.get("readout") or {}
    readout = AcquisitionSettings.from_dict(ro) if ro else AcquisitionSettings()
    if params.get("binning"):
        readout = readout.replace(hbin=int(params["binning"][0]), vbin=int(params["binning"][1]))
    spam = spam_params(engine, opts, readout) if engine is not None else \
        spam_params(_NoEngine(), opts, readout)
    return spam_ranking(summary.get("points") or [], summary.get("global") or {}, spam), spam


class _NoEngine:
    _plan = None

    def brightness(self):
        return {}


README = """ioncam camera characterization
==============================

Dark frames (shutter closed) at many readout settings.  Counts are ADU of the
camera; "pixel" means the (binned) super-pixel of that setting.

plan.json         the camera, its capabilities, the options and every setting
points.jsonl      one JSON line per setting measured, in the order measured:
                    settings      what was asked (ioncam AcquisitionSettings)
                    applied       what the camera used (speeds, gain, exposure, readout
                                  time, electrons per count = sensitivity)
                    temperature_before / _after   [C, status]
                    stats         bias (median), bias_spatial_rms, bias_row_rms/_col_rms,
                                  common_mode_rms (frame-to-frame offset), bias_drift_per_min,
                                  read_noise (counts; _e: electrons), events_per_px (above
                                  event_threshold = 5 read noise), hot pixels [x, y,
                                  events/frame, mean counts], trail (charge left in the pixel
                                  read next, fraction), event_repeat (same pixel next frame,
                                  / chance), first_frames (offset, events of the frames right
                                  after the start), cic_row_trend_per_100_rows
                    stats.fit     the dark model fitted to the histogram (maximum likelihood):
                                  mu, sigma (counts), g (counts per input electron = EM gain
                                  / electrons per count), lam_p (parallel CIC + dark current,
                                  events/pixel/frame), lam_s (serial CIC), a (one output
                                  electron, counts), errors (1 sigma; relative except mu),
                                  chi2_dof, above (fraction of samples above 3/5 sigma,
                                  g/2 .. 4g: measured, model)
                    stats.model   the same with the read noise of the camera alone (the
                                  noise of the bias estimate taken out) - for predictions
hist/NNNN.npz     hist: samples per count above bias, from hist_lo; for some settings also
                  bias_map, event_map (events/frame) and frame_offsets
summary.json      everything, plus "global" (dark current against exposure and temperature,
                  light leak, real/set EM gain, binning, stability, first frames,
                  triggered shots, hot pixels) and "ranking" (SPAM, see below)
summary.csv       one row per setting
camera_profile.json   the dark model at every setting, for simulations
                  (ioncam: simulator.camera_profile = "<this file>")

The model of a pixel (Lantz et al. 2008, appendix A):
  x = bias + N(0, read noise) + EM register output,
  EM output: Poisson(lam_p) electrons through the whole register, each
  exponential with mean g counts (n of them: Gamma(n, g)), plus Poisson(lam_s)
  serial-CIC electrons born in the register, with a log-uniform mean between
  one output electron (a) and g.

SPAM ranking: for an ion giving ion_rate x roi_fraction photo-electrons/s in a
readout ROI of roi_pixels super-pixels at the readout binning, every setting's
dark model (CIC added up over the binned pixels, dark current for the
detection time) with the ion's counts (Poisson photo-electrons through the EM
register; lifetimes: the state changing during the detection) gives the
threshold and its error 1 - (P(bright > t) + P(dark <= t)) / 2 at each
detection time, and the shortest detection time reaching target_error.
threshold_raw is the ROI sum's threshold including the bias.
"""
