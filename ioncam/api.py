"""The RPC interface (sipyco target ``ioncam``) used by ARTIQ and the GUI.

Use from an ARTIQ experiment (device_db entry of type ``controller``)::

    self.setattr_device("ioncam")
    ...
    ok = self.ioncam.wait_for_ions(3, timeout=120)["ok"]
    plan = self.ioncam.calibrate_readout(n_ions=3)       # locate, plan, verify, arm
    x0, y0, x1, y1 = plan["grabber_x0"], plan["grabber_y0"], plan["grabber_x1"], plan["grabber_y1"]

or directly with ``sipyco.pc_rpc.Client(host, 3270, "ioncam")``.

Conventions: positions are raw sensor pixels (0-based, see
:mod:`ioncam.camera.base`); all return values are plain JSON-like Python types
(numpy arrays only for images).  Calls that may take time are coroutines on
the server side, so a long calibration never blocks other clients.
"""

# NOTE: no type annotations in this module - sipyco's method introspection
# (get_rpc_method_list) calls ``__name__`` on every annotation.

import asyncio
import functools
import time

import numpy as np

from ioncam import version_string
from ioncam.analysis.readout import grabber_roi_sums
from ioncam.camera.base import AcquisitionSettings
from ioncam.config import config_help


def _plain(o):
    """Recursively convert numpy scalars/tuples into JSON-friendly Python types."""
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray) and o.ndim <= 1 and o.size <= 1024:
        return [_plain(v) for v in o.tolist()]
    return o


_LIVE_KEYS = {f for f in AcquisitionSettings.__dataclass_fields__}


class IonCamAPI:
    """Ion camera brain: live monitoring, ion detection/tracking, Grabber readout."""

    def __init__(self, engine):
        self._engine = engine

    async def _run(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))

    # ------------------------------------------------------------------ basics

    def ping(self):
        """Liveness check used by artiq_ctlmgr; returns True."""
        return True

    def get_version(self):
        """ioncam version and the git commit it runs from ("installed copy" if it
        was copied into site-packages instead of installed with ``pip install -e``)."""
        return version_string()

    def get_info(self):
        """Camera model, serial number, sensor size, simulated flag, capabilities."""
        info = dict(self._engine.info)
        info["ioncam_version"] = version_string()
        try:
            info["capabilities"] = self._engine.camera.capabilities()
        except Exception:
            pass
        return _plain(info)

    def get_status(self):
        """Everything at a glance: mode, temperature, fps, ion counts, crystal
        state, readout state, control lease, last event (dict)."""
        return _plain(self._engine.status())

    def get_temperature(self):
        """[temperature_C, status] with status in stabilized / not_reached /
        not_stabilized / drift / off / unknown."""
        t, s = self._engine.temperature()
        return [None if t != t else round(t, 2), s]

    async def set_temperature(self, temperature):
        """Set the sensor temperature set point [deg C]."""
        await self._run(self._engine.set_temperature, float(temperature))

    async def set_cooler(self, on):
        """Switch the TEC cooler on/off."""
        await self._run(self._engine.set_cooler, bool(on))

    # ------------------------------------------------------------- acquisition

    async def start_live(self, **settings):
        """Start live video (internal trigger).  Optional keyword overrides of the
        live settings, e.g. ``exposure=0.1, em_gain=100, roi=[200,100,330,240],
        hbin=1``.  Returns the applied settings."""
        bad = set(settings) - _LIVE_KEYS
        if bad:
            raise ValueError("unknown setting(s): {}".format(", ".join(sorted(bad))))
        return _plain(await self._run(self._engine.start_live, **_settings(settings)))

    async def set_live(self, **settings):
        """Change live settings (applied at once when live is running)."""
        bad = set(settings) - _LIVE_KEYS
        if bad:
            raise ValueError("unknown setting(s): {}".format(", ".join(sorted(bad))))
        return _plain(await self._run(self._engine.update_live, **_settings(settings)))

    async def set_exposure(self, exposure):
        """Live exposure time [s]."""
        return await self.set_live(exposure=float(exposure))

    async def set_em_gain(self, gain):
        """Live EM gain (0 = minimum)."""
        return await self.set_live(em_gain=int(gain))

    async def set_roi(self, x0, y0, x1, y1):
        """Live acquisition ROI in sensor pixels (half-open)."""
        return await self.set_live(roi=[int(x0), int(y0), int(x1), int(y1)])

    async def reset_roi(self):
        """Live acquisition on the full sensor."""
        return await self.set_live(roi=None)

    async def stop(self):
        """Stop acquiring (also disarms the readout)."""
        await self._run(self._engine.stop)

    def abort(self):
        """Abort a running procedure (locate / calibrate / snap).  True if one was running."""
        return self._engine.abort()

    async def snap(self, n_frames=1, **settings):
        """Take ``n_frames`` with the live settings (plus overrides) and return the
        mean image (float32) and its geometry.  Live mode resumes afterwards."""
        mean, frames = await self._run(self._engine.snap, int(n_frames), **_settings(settings))
        a = frames[-1].applied
        return {"image": mean.astype(np.float32), "roi": list(a.roi), "hbin": a.hbin,
                "vbin": a.vbin, "exposure": a.exposure, "em_gain": a.em_gain,
                "n_frames": len(frames)}

    # ------------------------------------------------------------------ images

    def get_frame(self, since_index=0, bin=1, roi=None, view="raw"):
        """Newest frame if newer than ``since_index`` else None.

        Returns {"index", "t", "tag", "image" (raw orientation), "roi" (sensor
        rect of the image), "hbin", "vbin", "exposure", "em_gain", "view"}.
        ``bin`` > 1 averages bin x bin blocks (less network traffic for viewers);
        ``roi`` = [x0, y0, x1, y1] in sensor pixels crops the image.
        ``view="corrected"``: the analysed live frame with the background
        model removed (learned static background, scatter profile, bias) -
        float32 - as the ion detection sees it; other frames stay raw.
        ``view="background"``: the learned static background for the frame's
        pixels (float32, counts at its brightness now; zero where none was learned)."""
        f = self._engine.latest_frame(int(since_index))
        if f is None:
            return None
        a = f.applied
        out = {"index": f.index, "t": f.timestamp, "tag": f.tag, "exposure": a.exposure,
               "em_gain": a.em_gain, "view": "raw"}
        img, x0, y0, hb, vb = f.data, f.x0, f.y0, f.hbin, f.vbin
        if view == "background":
            sb = self._engine.detector.static_background
            m = sb.for_frame(f) if sb is not None else None
            img = np.zeros(f.data.shape, np.float32) if m is None else \
                (m * np.float32(a.gain_factor * sb.scale)).astype(np.float32)
            out["view"] = "background"
        elif view == "corrected" and f.tag == "live":
            det = self._engine.last_detection()
            if det is None or det.frame_index <= int(since_index):
                return None
            img = det.residual
            hb, vb = det.hbin, det.vbin
            x0, y0 = det.x0 + det.crop[0] * hb, det.y0 + det.crop[1] * vb
            out.update(index=det.frame_index, t=det.timestamp, view="corrected")
        if roi is not None:
            u0 = max(0, (int(roi[0]) - x0) // hb)
            v0 = max(0, (int(roi[1]) - y0) // vb)
            u1 = min(img.shape[1], -(-(int(roi[2]) - x0) // hb))
            v1 = min(img.shape[0], -(-(int(roi[3]) - y0) // vb))
            img = img[v0:v1, u0:u1]
            x0, y0 = x0 + u0 * hb, y0 + v0 * vb
        b = int(bin)
        if b > 1:
            h, w = (img.shape[0] // b) * b, (img.shape[1] // b) * b
            img = img[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3)).astype(img.dtype)
            hb, vb = hb * b, vb * b
        out.update(image=np.ascontiguousarray(img), hbin=hb, vbin=vb,
                   roi=[x0, y0, x0 + img.shape[1] * hb, y0 + img.shape[0] * vb])
        return out

    # -------------------------------------------------------------------- ions

    def get_ions(self):
        """Tracked ion sites: list of {index, x, y, state ("bright"/"dark"),
        level (brightness / reference), snr, inferred} in chain order."""
        return _plain(self._engine.tracker_snapshot().get("sites", []))

    def get_ion_count(self):
        """Number of *bright* ions in the confirmed crystal (int)."""
        return int(self._engine.tracker_snapshot().get("n_bright", 0))

    def get_ion_counts(self):
        """{"n_bright", "n_dark", "n_ions", "state", "stable"}."""
        s = self._engine.tracker_snapshot()
        return _plain({k: s.get(k) for k in ("n_bright", "n_dark", "n_ions", "state", "stable",
                                             "stable_for")})

    def get_crystal(self):
        """Full tracker snapshot (state, sites, chain fit, pending change, hold)."""
        return _plain(self._engine.tracker_snapshot())

    def get_detection(self):
        """Last frame's raw detection: spots, clouds, background, noise, focus..."""
        return _plain(self._engine.detection_summary())

    async def wait_for_ions(self, n, timeout=60.0, at_least=False, stable_time=None,
                            bright_only=True):
        """Block until exactly ``n`` (or >= n with at_least=True) bright ions form
        a stable crystal, or until ``timeout`` s.  Returns {"ok", "n_bright",
        "n_dark", "n_ions", "state", "stable"}.  Typical use when loading::

            ioncam.wait_for_ions(1, timeout=300)   # oven + PI lasers on before
        """
        deadline = time.monotonic() + float(timeout)
        while True:
            ok, info = self._engine.ion_count_ok(int(n), bool(at_least), stable_time,
                                                 bool(bright_only))
            if ok or time.monotonic() >= deadline:
                info["ok"] = ok
                return _plain(info)
            await asyncio.sleep(0.05)

    async def locate_ions(self, expected=0, n_frames=None, region=None, **settings):
        """Long-exposure, low-gain frames + joint Gaussian fit of all ions.

        Returns precise positions (sensor px, ~0.01 px), uncertainties, PSF
        sigma, fluxes, chain fit.  Live/readout mode is restored afterwards.
        Ions are searched for only inside the live camera ROI, the search
        region (set_search_region) and ``region`` = [x0, y0, x1, y1].
        Keyword overrides go to the [locate] settings (e.g. exposure=1.0)."""
        return _plain(await self._run(self._engine.locate_ions, int(expected), n_frames, None,
                                      [int(v) for v in region] if region else None,
                                      **_settings(settings)))

    def hold_tracking(self, reason="", timeout=None):
        """Ions will be moved on purpose (shuttling, camera moved): suspend
        lost/dark classification.  Call resume_tracking() afterwards; the crystal
        is then compared with the one before (events: hold_done, ion_lost)."""
        return _plain(self._engine.hold_tracking(str(reason), timeout))

    def resume_tracking(self):
        """End a hold_tracking() period."""
        return _plain(self._engine.resume_tracking())

    def relearn(self):
        """Accept the currently detected crystal as the reference."""
        return _plain(self._engine.relearn())

    def reset_tracking(self):
        """Forget the crystal (next ions are reported as newly detected)."""
        self._engine.reset_tracking()

    # ------------------------------------------------------------------ events

    def get_last_event_id(self):
        """Id of the newest event (use with get_events / wait_for_event)."""
        return self._engine.events.last_id

    def get_events(self, since_id=0, kinds=None, limit=200):
        """Events newer than ``since_id``: list of {id, t, time, kind, severity,
        message, n_bright, n_dark, n_ions, details}.  Kinds include ion_trapped,
        ion_lost, ion_dark, ion_bright, dark_ion_moved, crystal_melted,
        crystal_recovered, fluorescence_lost, fluorescence_restored,
        ions_presumed_lost, cloud_detected, chain_moved, readout_armed,
        readout_plan_outdated, readout_geometry_changed, overexposure, ..."""
        return _plain(self._engine.events.since(int(since_id), kinds, int(limit)))

    async def wait_for_event(self, kinds=None, since_id=None, timeout=60.0):
        """Block until an event (of ``kinds``) newer than ``since_id`` (default:
        now) arrives; returns it or None on timeout."""
        log = self._engine.events
        last = log.last_id if since_id is None else int(since_id)
        deadline = time.monotonic() + float(timeout)
        while True:
            evs = log.since(last, kinds, limit=1)
            if evs:
                return _plain(evs[0])
            if time.monotonic() >= deadline:
                return None
            await asyncio.sleep(0.05)

    # ----------------------------------------------------------------- readout

    async def calibrate_readout(self, n_ions=0, hbin=0, vbin=0, roi_size=None, arm=True,
                                verify=None, auto_binning=False, force=False, locate=None,
                                aqroi=None, **readout_settings):
        """Prepare the Grabber readout carefully and (by default) arm it.

        1. long exposure / low gain -> precise ion positions + PSF (locate);
        2. plan the AQROI (aligned to the binning grid, centred on the crystal)
           and one Grabber ROI per ion maximising the light each ROI contains;
        3. verify with the final readout settings (internal trigger) that every
           ion is inside its ROI;
        4. arm: external trigger + Camera Link (+ run-till-abort).

        ``n_ions`` > 0 makes it fail unless exactly that many ions are found.
        ``hbin``/``vbin`` override the [readout] binning; ``auto_binning`` picks
        the cheapest binning that meets the containment/crosstalk targets.
        Extra keywords override [readout] settings (exposure, em_gain, ...);
        ``locate`` is a dict of [locate] overrides.  Ions are searched for only
        inside the live camera ROI and the search region (set_search_region);
        ``aqroi`` = [x0, y0, x1, y1] keeps that AQROI (e.g. set by hand with
        manual_aqroi): the ions are searched for inside it and read out in it.

        Returns the plan: aqroi / aqroi_solis / sdk_image, frame_size (W, H),
        grabber_rois and grabber_x0/y0/x1/y1 (ions first, then background),
        ion_mask, full_mask, containment, crosstalk, verification, applied
        camera settings incl. min_trigger_period, binning_advice, warnings."""
        return _plain(await self._run(
            self._engine.calibrate_readout, int(n_ions), int(hbin), int(vbin), roi_size,
            bool(arm), verify, bool(auto_binning), bool(force), locate or {},
            aqroi=[int(v) for v in aqroi] if aqroi else None, **_settings(readout_settings)))

    async def nudge_readout(self, dx=0, dy=0):
        """Move the AQROI by (dx, dy) sensor px, e.g. 1 px steps to centre the
        ions in their ROIs by hand.  The Grabber ROIs are frame coordinates, so
        they move with it and stay the same numbers; containment and crosstalk
        are re-computed, an armed readout is re-armed.  Returns the plan."""
        return _plain(await self._run(self._engine.nudge_readout, int(dx), int(dy)))

    async def replan_readout(self, hbin=0, vbin=0, aqroi=None, roi_size=None):
        """Plan again from the located ions (no new locate): another binning
        and/or a fixed AQROI [x0, y0, x1, y1] (sensor px, half-open, a multiple
        of the binning) inside which the ROIs are placed.  Ions too close to
        separate get overlapping ROIs plus a warning, never an error.  Re-arms
        if armed; not verified (see verify_readout).  Returns the plan."""
        return _plain(await self._run(self._engine.replan_readout, int(hbin), int(vbin),
                                      aqroi, roi_size))

    async def manual_aqroi(self, cols, rows=0, x=None, y=None, hbin=0, vbin=0, roi_size=None):
        """An AQROI set by hand: cols x rows super-pixels (square if rows is 0)
        centred on (x, y) sensor px - by default on the located ions - on the
        binning-grid phase that centres those ions in their super-pixels.  The
        ions inside get Grabber ROIs, spots outside are left out (a warning
        lists them).  Re-arms if armed; not verified.  Returns the plan."""
        return _plain(await self._run(
            self._engine.manual_aqroi, int(cols), int(rows), None if x is None else float(x),
            None if y is None else float(y), int(hbin), int(vbin), roi_size))

    async def resize_readout(self, dcols=0, drows=0):
        """dcols / drows super-pixel columns / rows more (fewer if negative) in
        the AQROI - one at a time on the side with the narrower margin around
        the ions' ROIs, keeping the binning grid; the ROIs are placed again
        inside it.  Re-arms if armed.  Returns the plan."""
        return _plain(await self._run(self._engine.resize_readout, int(dcols), int(drows)))

    def set_search_region(self, x0, y0, x1, y1):
        """Look for ions only inside [x0, y0, x1, y1] (sensor px, half-open) -
        live tracking, locate and calibrate - so stray light elsewhere is never
        taken for an ion.  The crystal is tracked afresh.  Returns the region."""
        return self._engine.set_search_region((x0, y0, x1, y1))

    def clear_search_region(self):
        """Look for ions everywhere again (tracked afresh)."""
        return self._engine.set_search_region(None)

    async def verify_readout(self, force=True):
        """Test frames with the current plan (internal trigger): signal of
        every ion in its ROI, single-shot SNR.  Returns the plan with its
        "verification"."""
        return _plain(await self._run(self._engine.verify_readout, bool(force)))

    async def arm_readout(self):
        """Re-arm the Grabber readout with the existing plan (e.g. after live)."""
        return _plain(await self._run(self._engine.arm_readout))

    async def disarm_readout(self, resume_live=True):
        """Leave readout mode; by default go back to live video."""
        await self._run(self._engine.disarm_readout, bool(resume_live))

    def get_readout_plan(self):
        """The current readout plan (see calibrate_readout) or None."""
        return _plain(self._engine.readout_plan())

    def get_grabber_rois(self):
        """[[x0, y0, x1, y1], ...] Grabber ROIs (ions in chain order, then the
        background ROI if any) - pass straight to Grabber.setup_roi(n, ...)."""
        plan = self._engine.readout_plan()
        return [] if plan is None else [list(map(int, r)) for r in plan["grabber_rois"]]

    def get_grabber_roi_lists(self):
        """([x0...], [y0...], [x1...], [y1...]) - convenient for ARTIQ kernels."""
        rois = self.get_grabber_rois()
        return [[r[k] for r in rois] for k in range(4)]

    def is_readout_valid(self):
        """False when the crystal changed since the last calibration."""
        plan = self._engine.readout_plan()
        return bool(plan is not None and plan.get("valid"))

    def get_readout_counts(self, since_shot=0, max_shots=5000):
        """Per-ROI sums of every readout frame the camera also sent over USB:
        {"first_shot", "next_shot", "sums": [[roi0, roi1, ...], ...], "t": [...]}."""
        return _plain(self._engine.monitor.counts(int(since_shot), int(max_shots)))

    def get_readout_stats(self, last=2000):
        """Per-ROI mean/std, fitted bright/dark threshold and bright fraction over
        the last shots, plus the fraction of light outside the ion ROIs."""
        return _plain(self._engine.monitor.stats(int(last)))

    async def get_readout_histograms(self, source="auto", last=20000):
        """Histograms of the readout shots - every ion ROI and the background
        (dark reference) - with a quick readout-fidelity estimate per ion:
        threshold (bright if ROI sum > threshold), bright and dark error and
        the fidelity, taking every shot as prepared bright (prepare bright on
        purpose).  The dark reference is the background pixels of the same
        shots summed to the ion ROI's size; ``dark_error_neighbours_bright``
        adds the planned crosstalk of the other ions.  ``source``:
        "trigger" (externally triggered shots), "verify" (the last
        verification frames) or "auto" (the newer)."""
        return _plain(await self._run(self._engine.readout_histograms, str(source), int(last)))

    def check_grabber_mapping(self, image):
        """Compare an image reconstructed from Grabber single-pixel ROIs (H x W)
        with the frames received over USB; returns the best orientation and the
        grabber.flip_x / flip_y settings to use."""
        return _plain(self._engine.check_grabber_mapping(np.asarray(image, float)))

    # --------------------------------------------------------------- background

    async def learn_background(self, n_frames=50, mask_ions=True, full_sensor=True,
                               **settings):
        """Learn the static background - laser scatter from surface defects,
        dust and electrode edges (which can look like ions or clouds), hot
        pixels - from ``n_frames`` with the live settings (``settings``
        override them, e.g. ``exposure=0.2``), of the full sensor without
        binning by default so it applies to every later ROI and binning.

        ``mask_ions``: leave the tracked ion sites out (ions may stay in the
        trap).  Best: switch the repumper off (dark ions scatter nothing, the
        laser still does) or empty the trap, and pass ``mask_ions=False``.
        Used for detection from then on (subtracted, scaled with exposure x
        EM gain and the laser power); returns a summary."""
        return _plain(await self._run(self._engine.learn_background, int(n_frames),
                                      bool(mask_ions), bool(full_sensor), **_settings(settings)))

    def get_background(self):
        """Summary of the learned static background (None if there is none)."""
        return _plain(self._engine.background_info())

    def clear_background(self):
        """Forget the learned static background."""
        self._engine.clear_background()

    # ------------------------------------------------------------------ control

    def take_control(self, owner, reason="", timeout=600.0, force=False):
        """Advisory lease: tells GUIs/users that ``owner`` (e.g. an experiment)
        is driving the camera.  Expires after ``timeout`` s."""
        return _plain(self._engine.take_control(str(owner), str(reason), timeout, bool(force)))

    def release_control(self, owner=None, force=False):
        """Release the control lease."""
        return self._engine.release_control(owner, bool(force))

    def get_control(self):
        """Current control lease or None."""
        return _plain(self._engine.control())

    # ------------------------------------------------------------------- config

    def get_config(self):
        """The whole configuration as a dict of sections."""
        return _plain(self._engine.config.to_dict())

    def get_config_help(self):
        """What every configuration option means: {section: {key: help}}, the
        section's own description under ``""``."""
        return config_help()

    async def set_config(self, section, key, value):
        """Change one configuration option at runtime (not saved)."""
        return _plain(await self._run(self._engine.set_config, str(section), str(key), value))

    def save_config(self, path):
        """Write the current configuration (with comments) to ``path`` on the
        camera PC."""
        self._engine.config.save(str(path))

    # --------------------------------------------------------------- simulation

    def _scene(self):
        scene = self._engine.sim_scene()
        if scene is None:
            raise RuntimeError("not running a simulated camera")
        return scene

    def sim_state(self):
        """[simulation] Ground truth of the simulated trap."""
        return _plain(self._scene().state())

    def sim_load_ion(self, n=1, cloud_time=0.8):
        """[simulation] Load ``n`` ions (hot cloud for ``cloud_time`` s first)."""
        self._scene().load_ion(int(n), float(cloud_time))

    def sim_lose_ion(self, index=-1):
        """[simulation] Lose ion ``index`` (chain order, -1 = random)."""
        return self._scene().lose_ion(None if int(index) < 0 else int(index))

    def sim_set_dark(self, index, dark=True):
        """[simulation] Put ion ``index`` into a dark state (or back)."""
        self._scene().set_dark(int(index), bool(dark))

    def sim_shift(self, dx, dy):
        """[simulation] Move the crystal rigidly (shuttling / camera moved)."""
        self._scene().shift(float(dx), float(dy))

    def sim_set_scale(self, scale_px):
        """[simulation] Change the axial confinement (chain length scale, px)."""
        self._scene().set_scale(float(scale_px))

    def sim_melt(self, duration=1.0):
        """[simulation] Melt the crystal for ``duration`` s."""
        self._scene().melt(float(duration))

    def sim_set_lasers(self, on):
        """[simulation] Cooling/detection lasers on or off."""
        self._scene().set_lasers(bool(on))

    def sim_set_oven(self, on, load_rate=None):
        """[simulation] Oven + photo-ionisation on/off (random loading)."""
        if load_rate is not None:
            self._engine.sim_camera().cfg.load_rate = float(load_rate)
        self._scene().set_oven(bool(on))

    def sim_get_params(self):
        """[simulation] All simulator parameters (photon rate, PSF and halo,
        surface scatter / defects, NA, ion height, camera noise ...) and the
        surface presets."""
        return _plain(self._scene().parameters())

    def sim_set_params(self, **params):
        """[simulation] Change simulator parameters at run time, e.g.
        ``sim_set_params(surface="horrendous")``, ``scatter_rate=100``,
        ``beam_angle_deg=30``, ``scatter_sigma_across=20``, ``defects=300``,
        ``numerical_aperture=0.4``, ``ion_height_um=50``, ``photon_rate=8000``,
        ``psf_halo_fraction=0.2``, ``laser_power=0.7``, ``surface_seed=7`` (a
        new surface).  Turning the trap (``trap_angle_deg``) is like mounting
        a different trap: a learned background is cleared, the crystal is
        tracked afresh and a readout plan is outdated.  Returns all parameters."""
        scene = self._scene()
        angles = (scene.cfg.trap_angle_deg, scene.cfg.axis_angle_deg)
        out = scene.set_params(**params)
        if (scene.cfg.trap_angle_deg, scene.cfg.axis_angle_deg) != angles:
            self._engine.sim_trap_turned()
        return _plain(out)

    def sim_set_repumper(self, on):
        """[simulation] Repumper off: the ions are shelved (dark) while the
        lasers still scatter from the surface - e.g. to learn the background."""
        self._scene().set_repumper(bool(on))

    def sim_move_camera(self, dx, dy):
        """[simulation] Move the camera (imaging system) by (dx, dy) px: ions,
        scatter and surface structure all move in the image."""
        self._scene().move_camera(float(dx), float(dy))

    def sim_trigger(self, n=1, spacing=0.0):
        """[simulation] External trigger pulses (what the ARTIQ TTL would do).
        Returns how many were accepted."""
        cam = self._engine.sim_camera()
        if cam is None:
            raise RuntimeError("not running a simulated camera")
        return int(cam.trigger(int(n), float(spacing)))

    def sim_grabber_sums(self):
        """[simulation] What the Grabber would return for the last readout frame
        (ROI sums in Grabber ROI order)."""
        cam = self._engine.sim_camera()
        plan = self._engine.readout_plan()
        if cam is None or plan is None or cam.last_frame is None:
            return None
        rois = [tuple(r) for r in plan["grabber_rois"]]
        w, h = plan["frame_size"]
        if cam.last_frame.shape != (h, w):
            return None
        return [int(v) for v in grabber_roi_sums(cam.last_frame, rois)]


def _settings(d):
    """RPC-friendly settings -> AcquisitionSettings field values."""
    out = dict(d)
    if "roi" in out and out["roi"] is not None:
        out["roi"] = tuple(int(v) for v in out["roi"])
    return out
