"""Chain tracker: per-frame detections -> a stable picture of the crystal + events.

How changes are told apart
--------------------------
* **dark** - a site stops fluorescing while every other ion stays exactly where
  it was (common shift ~ 0).  Positions of a chain depend only on charge, so
  a shelved ion or a molecular ion keeps its place.
* **lost** - the remaining ions re-arrange: residual motion after removing the
  best common shift, or a missing ion *together with* a common shift (the
  chain re-centres).  The new configuration is fitted with the linear-chain
  model (with the old length scale and centre as priors) to find hidden dark
  ions, so "a dark ion was lost" and "a bright ion was lost" are both seen.
* **trapped** - a new, persistent configuration with more ions (hot clouds
  are reported separately and never counted as ions).
* **bright again / dark ion moved** - spots re-appear on dark sites.
* **melted** - most of the fluorescence turns diffuse and dim; when a crystal
  re-forms it is compared with the one before (ions may have been lost).
* **fluorescence lost** - everything dark (lasers off or all ions gone); the
  ions are only presumed lost after ``forget_time``.
* **moved** - a rigid translation with all ions matched (shuttling, camera).

Nothing is reported until the same interpretation has persisted for
``confirm_time`` and ``confirm_frames`` (single-frame glitches, cosmic rays,
noise peaks never generate events).  ``hold()`` suspends classification while
the experiment deliberately moves ions; ``release()`` then reports whether
all ions survived.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ioncam.analysis.chain import fit_chain
from ioncam.analysis.detection import DetectionResult, deblend_patch
from ioncam.analysis.fitting import PSF, fit_fixed_psf, fit_gaussians

BRIGHT, DARK = "bright", "dark"

# crystal states
EMPTY = "empty"
STABLE = "stable"
CHANGING = "changing"
MELTED = "melted"
NO_FLUORESCENCE = "no_fluorescence"
HOLD = "hold"


@dataclass
class Site:
    id: int
    x: float
    y: float
    state: str = BRIGHT
    ref: Optional[float] = None    # reference brightness (filtered peak / gain factor)
    level: float = 1.0             # latest brightness / ref
    snr: float = 0.0
    since: float = 0.0             # time of the last state change
    inferred: bool = False         # dark ion whose position comes from the chain model

    def to_dict(self, index: int) -> dict:
        return {"index": index, "id": self.id, "x": round(self.x, 3), "y": round(self.y, 3),
                "state": self.state, "level": round(float(self.level), 3),
                "snr": round(float(self.snr), 1), "since": self.since, "inferred": self.inferred}


@dataclass
class _Pending:
    """A candidate interpretation being confirmed over several frames.

    For configuration hypotheses every candidate ion position carries a hit
    count; only positions seen in at least half of the frames are trusted, so
    a single-frame noise peak can never become an ion."""
    key: tuple
    t0: float
    frames: int = 1
    positions: Optional[np.ndarray] = None
    hits: Optional[np.ndarray] = None
    avg: Optional[np.ndarray] = None       # summed image around the crystal (see _avg_frame)
    avg_n: int = 0
    avg_key: Optional[tuple] = None
    avg_noise2: float = 0.0
    avg_sq: Optional[np.ndarray] = None
    misses: int = 0                        # consecutive frames showing something else
    site_pos: Optional[np.ndarray] = None  # the sites when the change began
    dark_votes: Optional[dict] = None      # frames read as "these sites went dark"
    checked_n: int = 0                     # avg_n when the average was last checked
    looks_same: bool = False               # ... and still showed the old crystal

    def __post_init__(self):
        if self.positions is not None and self.hits is None:
            self.hits = np.ones(len(self.positions))

    def reliable(self) -> np.ndarray:
        if self.positions is None:
            return np.zeros((0, 2))
        keep = self.hits >= max(1.0, 0.5 * self.frames)
        return self.positions[keep]

    def merge(self, spots: np.ndarray, gate: float) -> None:
        pos = self.positions if self.positions is not None else np.zeros((0, 2))
        pairs = _assign(pos, spots, gate)
        for i, j in pairs:
            pos[i] += 0.3 * (spots[j] - pos[i])
            self.hits[i] += 1
        new = [j for j in range(len(spots)) if j not in {b for _, b in pairs}]
        if new:
            self.positions = np.vstack([pos, spots[new]])
            self.hits = np.concatenate([self.hits, np.ones(len(new))])


def adopt_learned_psf(tracker: "ChainTracker", detection_cfg) -> Optional[str]:
    """Use the PSF the tracker measured on averaged frames of the live crystal
    for the per-frame detection too (``detection.auto_psf``): the detector then
    works with the real ion image without a calibration.  Returns a message
    when the detection PSF changed, else None."""
    e = tracker.psf_estimate
    if not detection_cfg.auto_psf or e is None or e is tracker._psf_adopted:
        return None
    tracker._psf_adopted = e
    sx, sy = e["sigma"]
    sigma = math.sqrt(sx * sy)
    if not 0.5 < sigma < 6.0:
        return None
    f = float(min(max(e["halo_fraction"], 0.0), 0.5))
    hs = float(e["halo_sigma"]) if f > 0 else 0.0
    d = detection_cfg
    if abs(sigma - d.psf_sigma) < 0.04 * d.psf_sigma and abs(f - d.psf_halo_fraction) < 0.03:
        return None
    old = PSF.from_sensor(d.psf_sigma, d.psf_halo_fraction, d.psf_halo_sigma)
    d.psf_sigma = round(sigma, 4)
    d.psf_halo_fraction = round(f, 4)
    d.psf_halo_sigma = round(hs, 4)
    tracker.psf_changed(old, PSF.from_sensor(d.psf_sigma, d.psf_halo_fraction, d.psf_halo_sigma))
    msg = "PSF measured on the live crystal: sigma {:.2f} px".format(sigma)
    if f > 0:
        msg += " + {:.0%} of the light in a halo of sigma {:.1f} px".format(f, hs)
    return msg + " - used for detection from now on"


def photometry_scale(model: PSF, true: PSF) -> float:
    """What least-squares photometry with PSF ``model`` reads for an isolated
    source of unit flux whose image is ``true`` (both unit-flux sums of
    Gaussians, so the overlaps are analytic)."""
    def overlap(a, b):
        return sum(wa * wb / (2.0 * math.pi * math.sqrt((ax * ax + bx * bx) * (ay * ay + by * by)))
                   for wa, ax, ay in a.parts() for wb, bx, by in b.parts())
    return overlap(model, true) / overlap(model, model)


def align(a: np.ndarray, b: np.ndarray, gate: float, max_shift: float):
    """Best common shift of point set ``a`` onto ``b`` and the matching.

    Candidate shifts are all pairwise differences (so large rigid moves are
    found); the score is the number of points matched within ``gate``, ties
    broken by the smallest shift.  Returns ``(shift, pairs, unmatched_a,
    unmatched_b)`` with ``pairs`` a list of ``(i_a, i_b)``."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return np.zeros(2), [], list(range(na)), list(range(nb))
    cand = (b[None, :, :] - a[:, None, :]).reshape(-1, 2)
    cand = cand[np.hypot(cand[:, 0], cand[:, 1]) <= max_shift]
    cand = np.vstack([np.zeros((1, 2)), cand])
    if len(cand) > 4000:
        cand = cand[:4000]
    moved = a[None, :, :] + cand[:, None, :]                       # (S, na, 2)
    d = np.linalg.norm(moved[:, :, None, :] - b[None, None, :, :], axis=3)  # (S, na, nb)
    n_match = (d.min(axis=2) < gate).sum(axis=1)
    score = n_match - 1e-3 * np.hypot(cand[:, 0], cand[:, 1])
    shift = cand[int(np.argmax(score))]
    for _ in range(2):  # refine with the actual assignment
        pairs = _assign(a + shift, b, gate)
        if not pairs:
            break
        disp = np.array([b[j] - a[i] for i, j in pairs])
        # median: one mis-placed spot (e.g. two blended ions) must not drag the shift
        shift = np.median(disp, axis=0) if len(pairs) >= 3 else disp.mean(axis=0)
    pairs = _assign(a + shift, b, gate)
    ma = {i for i, _ in pairs}
    mb = {j for _, j in pairs}
    return (shift, pairs, [i for i in range(na) if i not in ma],
            [j for j in range(nb) if j not in mb])


def _variance_of_mean(mean: np.ndarray, sum_sq: np.ndarray, n: int) -> np.ndarray:
    """Variance of each pixel of an n-frame average, from the EMCCD noise law
    var = a + b * signal fitted to the per-pixel sample variances (which are
    themselves too noisy to weight a fit with)."""
    sample = np.maximum(sum_sq / n - mean ** 2, 0.0) * n / max(n - 1, 1)
    sig = np.maximum(mean, 0.0).ravel()
    v = sample.ravel()
    keep = np.ones(v.size, bool)
    # the background noise (half the pixels with the least light) bounds the
    # law from below: with few frames, a fit pulled by the bright pixels could
    # otherwise put the background variance at ~0 and give those pixels all
    # the weight
    low = sig <= np.median(sig)
    a0 = float(np.median(v[low])) if np.any(low) else float(np.median(v))
    a, b = a0, 0.0
    for _ in range(3):
        A = np.column_stack([np.ones(keep.sum()), sig[keep]])
        (a, b), *_ = np.linalg.lstsq(A, v[keep], rcond=None)
        a, b = max(a, 0.5 * a0, 1e-12), max(b, 0.0)
        pred = a + b * sig
        keep = np.abs(v - pred) < 4.0 * pred * math.sqrt(2.0 / max(n - 1, 1)) + 1e-12
        if keep.sum() < 10:
            break
    return ((a + b * np.maximum(mean, 0.0)) / n).reshape(mean.shape)


def _assign(a: np.ndarray, b: np.ndarray, gate: float) -> List[Tuple[int, int]]:
    if len(a) == 0 or len(b) == 0:
        return []
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    ia, ib = linear_sum_assignment(d)
    return [(int(i), int(j)) for i, j in zip(ia, ib) if d[i, j] < gate]


def _same_configuration(p: np.ndarray, q: np.ndarray, gate: float) -> bool:
    """Same ion configuration?  Larger chains may drop one spot (detector merges
    neighbours near the resolution limit) without counting as different."""
    if p is None or q is None:
        return False
    n = max(len(p), len(q))
    if n == 0:
        return True
    allowed = 1 if n >= 3 else 0
    if abs(len(p) - len(q)) > allowed:
        return False
    return len(_assign(p, q, gate)) >= n - allowed


class ChainTracker:
    """Tracks ion sites over time and emits events (dicts)."""

    def __init__(self, cfg, clock=time.time):
        self.cfg = cfg
        self.clock = clock
        self.reset()

    # -- public ----------------------------------------------------------------------

    def reset(self) -> None:
        self.sites: List[Site] = []
        self.state = EMPTY
        self._next_id = 1
        self._pending: Optional[_Pending] = None
        self._cloud_pending: Optional[_Pending] = None
        self._clouds_reported = 0
        self._stable_since: Optional[float] = None
        self._state_since = self.clock()
        self._first_config = True
        self._chain = None              # (scale, center(2), axis(2)) of the last good fit
        self._premelt: Optional[List[Site]] = None
        self._hold: Optional[dict] = None
        self._after_hold: Optional[dict] = None
        self._region_ref: Optional[float] = None
        self._diffuse_ref: Optional[float] = None
        self._region_var: Optional[float] = None   # its frame-to-frame scatter
        self._moved_accum = np.zeros(2)
        self._dim = False
        self.fluorescence_level = float("nan")
        self.last_shift = np.zeros(2)
        self.history: Deque[Tuple[float, float, List[float]]] = deque(maxlen=3000)
        self.n_spots = 0
        self.n_clouds = 0
        self.last_update = 0.0
        self._shift_noise = 0.2         # rms of the per-frame common shift [px]
        self._pos_noise = 0.2           # rms per-axis position noise of one ion per frame [px]
        self._cl: Optional[dict] = None   # faint-cloud integrator (see _integrated_clouds)
        self._fi: Optional[dict] = None   # faint-ion integrator (see _faint_ions)
        self._last_resid: List[float] = []
        self.psf_estimate: Optional[dict] = None   # PSF learned from averaged frames
        self._psf_adopted: Optional[dict] = None   # the estimate last given to detection
        self._budget_var: Optional[float] = None   # scatter of the crystal's light budget
        self._avg_reset()

    @property
    def n_bright(self) -> int:
        return sum(1 for s in self.sites if s.state == BRIGHT)

    @property
    def n_dark(self) -> int:
        return sum(1 for s in self.sites if s.state == DARK)

    def is_stable(self, t: Optional[float] = None, stable_time: Optional[float] = None) -> bool:
        t = self.clock() if t is None else t
        st = self.cfg.stable_time if stable_time is None else stable_time
        settled = not self._pending_significant(t)
        return (self.state in (STABLE, EMPTY) and settled and
                self._stable_since is not None and t - self._stable_since >= st)

    def _pending_significant(self, t: float) -> bool:
        """Has the pending hypothesis lasted long enough to mean something?  (A
        change of the crystal also has to show on the average of its frames.)"""
        p = self._pending
        if p is None or p.frames < 2 or t - p.t0 < 0.5 * self.cfg.confirm_time:
            return False
        return p.key != ("CHANGE",) or (p.checked_n > 0 and not p.looks_same)

    def prior(self) -> Optional[np.ndarray]:
        """Where bright ions are expected in the next frame (for the detector's
        crowded-field fits), or None while ions are being moved / melted / dark."""
        if self.state not in (STABLE, CHANGING) or self._hold is not None or not self.sites:
            return None
        return self.positions(include_dark=False)

    def positions(self, include_dark: bool = True) -> np.ndarray:
        return np.array([[s.x, s.y] for s in self.sites
                         if include_dark or s.state == BRIGHT], float).reshape(-1, 2)

    def snapshot(self) -> dict:
        t = self.clock()
        chain = None
        if self._chain is not None:
            scale, center, axis = self._chain
            chain = {"scale_px": round(float(scale), 4),
                     "center": [round(float(center[0]), 3), round(float(center[1]), 3)],
                     "angle_deg": round(math.degrees(math.atan2(axis[1], axis[0])), 3)}
        pend = None
        if self._pending_significant(t):
            pend = {"kind": self._pending.key[0], "for_s": round(t - self._pending.t0, 3),
                    "frames": self._pending.frames}
        return {
            "state": self.state,
            "state_since": self._state_since,
            "n_bright": self.n_bright,
            "n_dark": self.n_dark,
            "n_ions": len(self.sites),
            "stable": self.is_stable(t),
            "stable_for": (round(t - self._stable_since, 3)
                           if self._stable_since is not None and self.state == STABLE else 0.0),
            "sites": [s.to_dict(i) for i, s in enumerate(self.sites)],
            "fluorescence_level": (None if not math.isfinite(self.fluorescence_level)
                                   else round(float(self.fluorescence_level), 3)),
            "chain": chain,
            "pending": pend,
            "hold": dict(self._hold) if self._hold else None,
            "n_spots": self.n_spots,
            "n_clouds": self.n_clouds,
            "last_update": self.last_update,
        }

    def hold(self, reason: str = "", timeout: Optional[float] = None) -> dict:
        """Suspend change classification (ions are being moved on purpose)."""
        t = self.clock()
        if self._hold is None:
            self._hold = {"reason": reason, "since": t,
                          "until": t + (timeout if timeout else self.cfg.hold_timeout),
                          "n_bright": self.n_bright, "n_dark": self.n_dark,
                          "n_ions": len(self.sites)}
        else:
            self._hold["reason"] = reason or self._hold["reason"]
            self._hold["until"] = t + (timeout if timeout else self.cfg.hold_timeout)
        self._set_state(HOLD, t)
        self._pending = None
        return dict(self._hold)

    def release(self) -> Optional[dict]:
        """End a hold; the next stable configuration is compared with the one before."""
        if self._hold is None:
            return None
        info = dict(self._hold)
        self._hold = None
        self._pending = None
        self._chain = None              # trap centre/scale may have changed
        self._after_hold = info
        self._set_state(CHANGING, self.clock())
        return info

    def relearn(self, det: Optional[DetectionResult] = None) -> List[dict]:
        """Accept the current detections as the reference configuration now."""
        t = self.clock()
        self._pending = None
        self._premelt = None
        self._hold = None
        if det is None:
            return []
        self.sites = []
        self._chain = None
        self._adopt(det.positions, [], det, t)
        self._set_state(STABLE if self.sites else EMPTY, t)
        self._stable_since = t
        return [self._event("reference_set", "info",
                            "Reference set to the current crystal ({} ions)".format(len(self.sites)),
                            t)]

    def rebaseline(self) -> None:
        """Camera settings changed: re-learn brightness references from the next frame."""
        for s in self.sites:
            s.ref = None

    def psf_changed(self, old: PSF, new: PSF) -> None:
        """The detection PSF changed (``old`` -> ``new``, sensor px): photometry
        with it reads a different number for the same ion (a too-wide model
        without halo reads ~20 % more), so the brightness references are
        rescaled - with the best PSF known (learned, else ``new``) as the
        truth - and the light references that depend on them start afresh."""
        e = self.psf_estimate
        true = new if e is None else PSF.from_sensor(e["sigma"], e["halo_fraction"],
                                                     e["halo_sigma"])
        r = photometry_scale(new, true) / max(photometry_scale(old, true), 1e-12)
        for s in self.sites:
            if s.ref:
                s.ref *= r
        self._region_ref = None
        self._diffuse_ref = None
        self._region_var = None
        self._budget_var = None

    def background_changed(self) -> None:
        """The background model changed (static background learned / cleared):
        brightness references, the diffuse-light reference and the cloud
        integrator start afresh."""
        self.rebaseline()
        self._region_ref = None
        self._diffuse_ref = None
        self._region_var = None
        self._cl = None
        self._pending = None
        self._avg_reset()

    # -- main update -------------------------------------------------------------------

    def update(self, det: DetectionResult, t: Optional[float] = None) -> List[dict]:
        t = self.clock() if t is None else t
        self.last_update = t
        events: List[dict] = []
        spots = np.array([[s.x, s.y] for s in det.spots], float).reshape(-1, 2)
        self.n_spots = len(spots)
        self._update_clouds(det, t, events)

        if self._hold is not None:
            self._update_hold(det, spots, t, events)
            return events
        after_hold = self._after_hold
        if after_hold is not None:
            self._update_after_hold(det, spots, t, events, after_hold)
            return events
        if not self.sites:
            self._update_empty(det, spots, t, events)
            return events
        if self.state == MELTED:
            self._update_melted(det, spots, t, events)
            return events
        if self.state == NO_FLUORESCENCE:
            self._update_dark_crystal(det, spots, t, events)
            return events
        self._update_crystal(det, spots, t, events)
        return events

    # -- states ---------------------------------------------------------------------------

    def _faint_ions(self, det, t) -> np.ndarray:
        """Faint new ions (empty trap, loading): a ~0.5 s average of the
        matched-filter image compared with a slowly learned reference, so ions
        too dim for single frames are found and static defects never count."""
        f = det.filtered
        if f.size == 0 or det.noise_f_map is None:
            self._fi = None
            return np.zeros((0, 2))
        key = (det.crop, det.x0, det.y0, det.hbin, det.vbin, f.shape)
        c = self._fi
        if c is None or c["key"] != key:
            self._fi = {"key": key, "fast": f.astype(np.float64), "slow": f.astype(np.float64),
                        "n": 1, "t": t}
            return np.zeros((0, 2))
        dt = max(t - c["t"], 1e-3)
        c["t"] = t
        c["n"] += 1
        a_f = max(min(1.0, dt / 0.5), 1.0 / c["n"])
        a_s = max(min(1.0, dt / 30.0), 1.0 / c["n"])
        c["fast"] += a_f * (f - c["fast"])
        if c["n"] * min(1.0, dt / 0.5) < 3.0:
            # warming up: the reference is no older than the fast average yet
            # (the noise estimate below assumes it is) - learn, report nothing
            c["slow"] += a_s * (f - c["slow"])
            return np.zeros((0, 2))
        excess = c["fast"] - c["slow"]
        sig = det.noise_f_map * math.sqrt(a_f / (2.0 - a_f) + a_s / (2.0 - a_s))
        mask = excess > 6.0 * sig
        # the reference learns the usual image everywhere except where light
        # is (becoming) significantly new - updating it only in frames where
        # nothing stands out would freeze it on the first, noisy frames of a
        # large image, and its noise would then pass for faint ions
        c["slow"] += a_s * np.where(excess < 3.0 * sig, f - c["slow"], 0.0)
        if not np.any(mask):
            return np.zeros((0, 2))
        from ioncam.analysis.detection import _local_maxima
        idx = _local_maxima(excess, mask)
        idx = idx[np.argsort(-excess.ravel()[idx])][:20]
        vv, uu = np.divmod(idx, excess.shape[1])
        hb, vb = det.hbin, det.vbin
        return np.column_stack([(uu + det.crop[0]) * hb + det.x0 + 0.5 * (hb - 1),
                                (vv + det.crop[1]) * vb + det.y0 + 0.5 * (vb - 1)]).astype(float)

    def _update_empty(self, det, spots, t, events):
        faint = self._faint_ions(det, t)
        if len(faint):
            sep = 2.0 * max(det.psf.sx * det.hbin, det.psf.sy * det.vbin) if det.psf else 3.0
            extra = [p_ for p_ in faint if not len(spots) or
                     np.min(np.hypot(*(spots - p_).T)) > sep]
            if extra:
                spots = np.vstack([spots, np.array(extra)]) if len(spots) else np.array(extra)
        if len(spots) == 0:
            self._pending = None
            self._first_config = False      # we have seen the trap empty
            if self.state != EMPTY:
                self._set_state(EMPTY, t)
            if self._stable_since is None:
                self._stable_since = t
            return
        key = ("APPEAR",)
        committed = self._advance(key, t, spots)
        if self._pending is not None:
            self._pending_average(self._pending, det)
        if committed:
            p = self._pending
            self._pending = None
            self._learn_psf_on(p, det)
            found = self._average_fit(p, det, (p.reliable(),))
            self._adopt(found if found is not None and len(found) else p.reliable(), [], det, t)
            n = len(self.sites)
            if self._first_config:
                events.append(self._event("ions_detected", "info",
                                          "{} ion{} detected".format(n, "s" * (n != 1)), t,
                                          count=n))
            else:
                events.append(self._event("ion_trapped", "info",
                                          "Ion trapped! {} ion{} now".format(n, "s" * (n != 1)),
                                          t, count=n))
            self._first_config = False
            self._set_state(STABLE, t)
            self._stable_since = t
        elif self.state == EMPTY:
            self._stable_since = None

    def _observe(self, det: DetectionResult, spots: np.ndarray):
        """Brightness of every site + matching of spots to bright sites."""
        cfg = self.cfg
        pos = self.positions()
        # deblended photometry: a neighbour's light is not counted for a dark ion
        vals, noise, _ = det.photometry(pos[:, 0], pos[:, 1])
        cu, cv = det.to_crop(pos[:, 0], pos[:, 1])
        h, w = det.filtered.shape if det.filtered.size else (0, 0)
        inside = (np.atleast_1d(cu) > -0.5) & (np.atleast_1d(cu) < w - 0.5) & \
                 (np.atleast_1d(cv) > -0.5) & (np.atleast_1d(cv) < h - 0.5)
        norm = vals / max(det.gain_factor, 1e-12)
        refs = [s.ref for s in self.sites if s.state == BRIGHT and s.ref]
        default_ref = float(np.median(refs)) if refs else None
        levels = np.full(len(self.sites), np.nan)
        for k, s in enumerate(self.sites):
            ref = s.ref if s.ref else default_ref
            if ref:
                levels[k] = norm[k] / ref
        bright_idx = [k for k, s in enumerate(self.sites) if s.state == BRIGHT]
        dark_idx = [k for k, s in enumerate(self.sites) if s.state == DARK]
        shift, pairs, miss_b, extra = align(pos[bright_idx], spots, cfg.match_radius,
                                            cfg.max_global_shift)
        region, region_noise = self._region_flux(det, pos)
        # light around the crystal that the sites do not explain: scatter structure
        # (constant) plus, when melted, the smeared-out fluorescence
        explained = float(np.sum(np.clip(np.where(inside, vals, 0.0), 0.0, None))) / \
            max(det.gain_factor, 1e-12)
        diffuse = None if region is None else region - explained
        if det.prior_fit:
            # the detector already fitted the frame starting from these sites
            fitted = np.full((len(self.sites), 2), np.nan)
            _, prs, _, _ = align(pos, spots, cfg.match_radius, cfg.max_global_shift) \
                if len(spots) else (None, [], None, None)
            for k, j in prs:
                if self.sites[k].state == BRIGHT:
                    fitted[k] = spots[j]
        else:
            fitted = self._fit_sites(det, inside)
        pairs = [(bright_idx[i], j) for i, j in pairs]
        missing = [bright_idx[i] for i in miss_b]
        # A spot next to an expected ion without a spot of its own may be the
        # two seen as one, pulled in between: the common motion is measured on
        # the other ions when there are enough of them ("clean" shift)
        near, clean_shift = set(), not missing
        if missing and pairs:
            sp = self._local_spacing(pos)
            near = {k for k, _ in pairs
                    if min(np.hypot(*(pos[m] - pos[k])) for m in missing) < 1.3 * sp[k]}
            clean = [(k, j) for k, j in pairs if k not in near]
            if len(clean) >= 2:
                disp = np.array([spots[j] - pos[k] for k, j in clean])
                shift = np.median(disp, axis=0) if len(clean) >= 3 else disp.mean(axis=0)
                clean_shift = True
        return {"pos": pos, "vals": vals, "noise": noise, "norm": norm, "levels": levels,
                "inside": inside, "bright_idx": bright_idx, "dark_idx": dark_idx,
                "shift": shift, "pairs": pairs, "missing": missing, "extra": extra,
                "near_missing": near, "clean_shift": clean_shift,
                "default_ref": default_ref, "region": region, "region_noise": region_noise,
                "diffuse": diffuse,
                "fitted": fitted}

    def _fit_sites(self, det: DetectionResult, inside) -> Optional[np.ndarray]:
        """Positions of the bright sites fitted to this frame (sensor px; NaN for
        dark or unfitted sites), or None."""
        bright = [k for k, s in enumerate(self.sites) if s.state == BRIGHT and inside[k]]
        if not bright or det.psf is None:
            return None
        box = self._crystal_box(det)
        if box is None:
            return None
        ua, va, ub, vb = box
        pos = self.positions()[bright]
        cu, cv = det.to_crop(pos[:, 0], pos[:, 1])
        try:
            fit = fit_fixed_psf(det.residual[va:vb, ua:ub], np.column_stack([cu - ua, cv - va]),
                                det.psf, max_move=1.0, iterations=6)
        except np.linalg.LinAlgError:  # pragma: no cover
            return None
        out = np.full((len(self.sites), 2), np.nan)
        hb, vb_ = det.hbin, det.vbin
        good = fit.flux > 3.0 * np.maximum(fit.flux_err, 1e-12)
        for i, k in enumerate(bright):
            if good[i]:
                u = fit.positions[i, 0] + ua
                v = fit.positions[i, 1] + va
                out[k] = ((u + det.crop[0]) * hb + det.x0 + 0.5 * (hb - 1),
                          (v + det.crop[1]) * vb_ + det.y0 + 0.5 * (vb_ - 1))
        return out

    def _region_flux(self, det: DetectionResult, pos: np.ndarray):
        """Background-subtracted light in a box around the crystal and its
        noise (normalised), or (None, None).

        Tells a melted crystal (light still there, just diffuse) from lasers
        off / ions gone (no light at all)."""
        res = det.residual
        if len(pos) == 0 or res.size == 0:
            return None, None
        cu, cv = det.to_crop(pos[:, 0], pos[:, 1])
        mu = 3.0 * self.cfg.match_radius / det.hbin + 2.0
        mv = 3.0 * self.cfg.match_radius / det.vbin + 2.0
        h, w = res.shape
        u0 = int(max(0, math.floor(np.min(cu) - mu)))
        u1 = int(min(w, math.ceil(np.max(cu) + mu) + 1))
        v0 = int(max(0, math.floor(np.min(cv) - mv)))
        v1 = int(min(h, math.ceil(np.max(cv) + mv) + 1))
        if u1 <= u0 or v1 <= v0:
            return None, None
        g = max(det.gain_factor, 1e-12)
        n = (v1 - v0) * (u1 - u0)
        return float(res[v0:v1, u0:u1].sum()) / g, float(det.noise) * math.sqrt(n) / g

    def _local_spacing(self, pos: np.ndarray) -> np.ndarray:
        if len(pos) < 2:
            return np.full(len(pos), np.inf)
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        return d.min(axis=1)

    def _classify(self, det, spots, obs) -> Optional[tuple]:
        """Interpret one frame.  Returns a hypothesis key or None (undecided)."""
        cfg = self.cfg
        pos, levels, inside = obs["pos"], obs["levels"], obs["inside"]
        shift = obs["shift"]
        observed = [k for k in range(len(self.sites)) if inside[k]]
        if not observed:
            return None
        bright_obs = [k for k in obs["bright_idx"] if inside[k]]
        spacing = self._local_spacing(pos)
        # single-frame positions of dim ions scatter: never call that motion
        tol = np.maximum(np.maximum(cfg.reconfig_min_px, cfg.reconfig_fraction * spacing),
                         3.5 * self._pos_noise)
        # per-frame position noise (dim ions, blended neighbours) must not look
        # like a rigid move: the threshold follows the measured shift noise
        small_shift = float(np.hypot(*shift)) <= max(
            cfg.reconfig_min_px, 0.5 * float(np.min(tol)) if len(tol) else 1.0,
            4.0 * self._shift_noise)

        # Light budget, in single-ion units: two ions the detector saw as one
        # keep the crystal's light, a lost ion takes one ion's worth with it.
        # Joint photometry at the sites shares the light out among them, so the
        # sum keeps the total wherever the ions sit.  (Single spots are too
        # noisy at low light to tell a merged pair by its brightness; the
        # deficit is compared with the scatter of the total.)
        light_short = False
        bidx = [k for k in obs["bright_idx"] if inside[k]]
        lvb = levels[bidx] if bidx else np.zeros(0)
        if len(bidx) >= 2 and np.all(np.isfinite(lvb)):
            total = float(np.sum(lvb))
            refs = np.array([self.sites[k].ref or np.nan for k in bidx], float)
            nz = np.asarray(obs["noise"], float)[bidx] / max(det.gain_factor, 1e-12) / refs
            sig = math.sqrt(float(np.nansum(nz ** 2))) if np.any(np.isfinite(nz)) else 0.3
            if self._budget_var:
                sig = max(sig, math.sqrt(self._budget_var))
            light_short = len(bidx) - total > max(0.6, 3.0 * sig)
            if not obs["missing"] and not obs["extra"]:
                dev = (total - len(bidx)) ** 2
                self._budget_var = dev if self._budget_var is None else \
                    0.95 * self._budget_var + 0.05 * dev

        # residual motion of matched ions after the common shift
        reconf = False
        resid = []
        for k, j in obs["pairs"]:
            d = spots[j] - (pos[k] + shift)
            if k not in obs["near_missing"]:
                resid.append(float(np.hypot(*d)))     # (position noise estimate)
            if float(np.hypot(*d)) > tol[k]:
                # a spot pulled towards a missing neighbour is two ions seen as one
                # (whatever that neighbour's brightness reads), not motion - unless
                # light is missing: a tight chain that lost an ion re-centres its
                # ions towards the gap too
                toward = [m for m in obs["missing"]
                          if np.hypot(*(pos[m] - pos[k])) < 1.3 * spacing[k] and
                          float(np.dot(d, pos[m] - pos[k])) > 0]
                if toward and not light_short:
                    continue
                reconf = True
        self._last_resid = resid
        lv = levels[bright_obs] if bright_obs else np.array([])
        lv = lv[np.isfinite(lv)]
        med_level = float(np.median(lv)) if lv.size else float("nan")

        # melted: the light is still there, just not at the sites.  Measured above
        # the usual background structure around the crystal (laser scatter stays
        # when an ion goes dark; everything goes when the lasers are off)
        diffuse = False
        if obs["region"] is not None and self._region_ref and self._diffuse_ref is not None:
            crystal_light = max(self._region_ref - self._diffuse_ref, 1e-12)
            excess = obs["region"] - self._diffuse_ref
            # ... and by more than that light scatters from frame to frame
            # (around a single dark ion the background alone often makes 8 % of
            # its light; residual background errors are correlated, so the
            # scatter of the sum is measured, the pixel noise only bounds it)
            noise = obs["region_noise"] or 0.0
            if self._region_var is not None:
                noise = max(noise, math.sqrt(self._region_var))
            diffuse = excess >= cfg.melt_diffuse * crystal_light and excess > 3.0 * noise
        matched_frac = len(obs["pairs"]) / max(1, len(bright_obs))
        # melted: sites dim, but the light is still there - spread out
        if bright_obs and lv.size and med_level < cfg.melt_level and matched_frac < 0.5 and \
                (diffuse or det.clouds):
            return ("MELT",)
        # everything dark (no light around the crystal at all)?
        if bright_obs and len(spots) == 0 and not det.clouds and \
                (not lv.size or np.all(lv < cfg.dark_level)):
            return ("ALL_DARK",)

        newly_dark, ambiguous, unaccounted = [], False, False
        for k in obs["missing"]:
            if not inside[k]:
                continue
            lvk = levels[k]
            if not np.isfinite(lvk) or lvk < cfg.dark_level:
                newly_dark.append(k)
            elif lvk < cfg.bright_level:
                ambiguous = True
            elif light_short:
                # light at the site, but the crystal as a whole is an ion short:
                # not a merged pair - it changed
                unaccounted = True
            # else: still bright, the detector merged it with a neighbour
        newly_bright, unexplained = [], []
        dark_pos = pos[obs["dark_idx"]] + shift if obs["dark_idx"] else np.zeros((0, 2))
        used = set()
        for j in obs["extra"]:
            if len(dark_pos):
                dd = np.hypot(*(dark_pos - spots[j]).T)
                i = int(np.argmin(dd))
                if dd[i] < cfg.match_radius and i not in used:
                    used.add(i)
                    newly_bright.append(obs["dark_idx"][i])
                    continue
            unexplained.append(j)
        for i, k in enumerate(obs["dark_idx"]):   # dark site bright again but not detected
            if i not in used and inside[k] and np.isfinite(levels[k]) and \
                    levels[k] > cfg.bright_level:
                newly_bright.append(k)

        # A missing ion together with a common shift means the chain re-centred:
        # that is a loss, never "an end ion went dark".  A rigid move is only
        # accepted when every ion is accounted for.
        if reconf or unexplained or unaccounted or (obs["missing"] and not small_shift):
            return ("RECONFIG",)
        if ambiguous:
            return None
        if newly_dark or newly_bright:
            return ("SITES", tuple(sorted(newly_dark)), tuple(sorted(newly_bright)))
        if not small_shift:
            return ("MOVED",)
        return ("STABLE",)

    def _update_crystal(self, det, spots, t, events):
        obs = self._observe(det, spots)
        hyp = self._classify(det, spots, obs)
        self._record_history(obs, t)
        if hyp is None:
            return
        kind = hyp[0]
        moved = False
        if kind == "MOVED":
            # a rigid move (shuttling, camera) is followed once it persists
            if not self._advance(hyp, t, spots):
                return
            moved = True
        if kind in ("STABLE", "MOVED"):
            # an expected ion without its own spot (merged with a neighbour - or
            # a change under way): such a frame neither confirms nor cancels a
            # pending change, and must not drag the sites (nor the noise
            # estimates, nor the averaged image) along
            ambiguous = bool(obs["missing"]) and not moved
            p_ = self._pending
            if ambiguous and p_ is not None and not p_.looks_same:
                return
            if p_ is not None and p_.looks_same:
                self._avg_merge(p_)      # (see _avg_merge)
            self._pending = None
            if kind == "STABLE" and obs["clean_shift"]:
                # (frames with a merged pair count too - measured on the other
                # ions: frames where every ion has its spot are the bright ones)
                sh = float(np.hypot(*obs["shift"]))
                self._shift_noise = math.sqrt(0.95 * self._shift_noise ** 2 + 0.05 * sh * sh)
                r = getattr(self, "_last_resid", [])
                if r:
                    # 2D displacement -> per-axis rms
                    ms = float(np.median(np.square(r))) / 1.386
                    self._pos_noise = math.sqrt(0.95 * self._pos_noise ** 2 + 0.05 * ms)
            if not ambiguous:
                self._apply_tracking(obs, spots, t, full=moved)
            if moved:
                self._avg_reset()
            else:
                # (merged or not, the frame shows the crystal as it is)
                self._avg_add(det, t, events)
            if moved or np.hypot(*self._moved_accum) > 3.0:
                acc = obs["shift"] if moved else self._moved_accum
                self._moved_accum = np.zeros(2)
                events.append(self._event(
                    "chain_moved", "info",
                    "Crystal moved rigidly by ({:+.1f}, {:+.1f}) px".format(*acc), t,
                    shift=[float(acc[0]), float(acc[1])]))
            if self.state != STABLE:
                # a persistent-but-unconfirmed change faded away: stable again from
                # now - unless its average showed the old crystal (noise: nothing
                # happened)
                self._set_state(STABLE, t)
                if moved or p_ is None or not p_.looks_same:
                    self._stable_since = t
            if self._stable_since is None:
                self._stable_since = t
            self._check_dim(t, events)
            return
        # An ion missing from its site: single frames of a tight crystal cannot
        # always tell "it went dark, the rest stayed" from "it was lost, the rest
        # re-arranged" - one change, decided on the average of its frames
        dark_cand = None
        if kind == "SITES" and hyp[1] and not hyp[2]:
            dark_cand = hyp[1]
            hyp = ("CHANGE",)
        elif kind == "RECONFIG":
            hyp = ("CHANGE",)
        kind = hyp[0]
        committed = self._advance(hyp, t, spots)
        p_ = self._pending
        if kind in ("CHANGE", "SITES") and p_ is not None and p_.misses == 0:
            # the frames that show the change are its evidence (an odd frame that
            # showed something else must not be mixed in)
            self._pending_average(p_, det)
            if dark_cand is not None:
                p_.dark_votes = p_.dark_votes or {}
                p_.dark_votes[dark_cand] = p_.dark_votes.get(dark_cand, 0) + 1
            if kind == "CHANGE" and p_.avg is not None and p_.avg_n >= 4 and \
                    p_.avg_n >= 1.5 * p_.checked_n:
                # (decided as at the end: until the average shows the change, it
                # is not under way, and frames of the old crystal end it)
                p_.checked_n = p_.avg_n
                p_.looks_same = self._decide_change(p_, det)[0] == "same"
        if committed and kind == "CHANGE" and p_ is not None and \
                (p_.avg is None or p_.avg_n < 4):
            committed = False      # decided on the average of >= 4 frames
        # a few odd frames (noise, cosmic rays, dim ions) are not instability; only
        # a hypothesis that persists makes the crystal "changing"
        if self.state == STABLE and self._pending_significant(t):
            self._set_state(CHANGING, t)
        if not committed:
            return
        p = self._pending
        self._pending = None
        if kind == "SITES":
            self._commit_sites(hyp[1], hyp[2], obs, t, events)
        elif kind == "CHANGE":
            verdict, new_pos = self._decide_change(p, det)
            if verdict == "dark":
                # dark: the rest of the crystal is exactly where it was
                self._commit_sites(new_pos, (), obs, t, events)
                self._set_state(STABLE, t)
                self._stable_since = t
                return
            if verdict == "same":
                # the averaged frames still show the old crystal: single-frame noise
                self._avg_merge(p)
                self._set_state(STABLE, t)
                return
            if verdict == "hidden":
                # every old site is still there and one more ion shows up in the
                # average: it was there all along (too dim / too close to resolve)
                n_before = self.n_bright
                dark = self.positions(include_dark=True)[[s.state == DARK for s in self.sites]] \
                    if self.n_dark else np.zeros((0, 2))
                self._adopt(new_pos, dark, det, t)
                events.append(self._event(
                    "crystal_refined", "info",
                    "Re-measured on {} averaged frames: {} bright ions (single frames showed "
                    "{})".format(p.avg_n, len(new_pos), n_before), t,
                    before=n_before, after=len(new_pos), frames=p.avg_n))
                self._set_state(STABLE, t)
                self._stable_since = t
                return
            self._commit_reconfig(new_pos, det, t, events)
        elif kind == "MELT":
            self._premelt = [Site(**vars(s)) for s in self.sites]
            self._set_state(MELTED, t)
            events.append(self._event("crystal_melted", "warning",
                                      "Crystal melted / heated (collision?) - waiting for it "
                                      "to re-crystallise", t))
            return
        elif kind == "ALL_DARK":
            self._set_state(NO_FLUORESCENCE, t)
            if len(self.sites) == 1:
                msg = ("The ion went dark: dark state, lost, or lasers off (a single ion "
                       "cannot tell which) - presumed lost after {:.0f} s".format(
                           self.cfg.forget_time))
            else:
                msg = ("All ion fluorescence vanished at once (lasers blocked/off?) - ions "
                       "presumed lost after {:.0f} s".format(self.cfg.forget_time))
            events.append(self._event("fluorescence_lost", "alarm", msg, t))
            return
        self._set_state(STABLE, t)
        self._stable_since = t

    def _update_melted(self, det, spots, t, events):
        cfg = self.cfg
        if len(spots) == 0:
            self._pending = None
            if t - self._state_since > cfg.forget_time and self._premelt is not None:
                n = len(self._premelt)
                self.sites = []
                self._premelt = None
                self._set_state(EMPTY, t)
                events.append(self._event("ions_presumed_lost", "alarm",
                                          "Crystal did not re-form within {:.0f} s - {} ion(s) "
                                          "presumed lost".format(cfg.forget_time, n), t,
                                          count=n))
            return
        if not self._advance(("REFORM",), t, spots):
            return
        p = self._pending
        self._pending = None
        before = self._premelt or self.sites
        n_before = len(before)
        self.sites = [Site(**vars(s)) for s in before]
        self._premelt = None
        # compare the re-formed crystal with the one before melting
        pos = self.positions()
        reformed = p.reliable()
        shift, pairs, miss, extra = align(pos, reformed, self.cfg.match_radius,
                                          self.cfg.max_global_shift)
        if not miss and not extra and float(np.hypot(*shift)) <= self.cfg.reconfig_min_px:
            for k, j in pairs:
                self.sites[k].x, self.sites[k].y = map(float, reformed[j])
            self._set_state(STABLE, t)
            self._stable_since = t
            events.append(self._event("crystal_recovered", "info",
                                      "Crystal re-formed with all {} ions".format(n_before), t))
            return
        self._commit_reconfig(p.reliable(), det, t, events, recovered=True)
        self._set_state(STABLE, t)
        self._stable_since = t

    def _update_dark_crystal(self, det, spots, t, events):
        cfg = self.cfg
        if len(spots) == 0:
            self._pending = None
            if t - self._state_since > cfg.forget_time:
                n = len(self.sites)
                self.sites = []
                self._chain = None
                self._set_state(EMPTY, t)
                events.append(self._event("ions_presumed_lost", "alarm",
                                          "No fluorescence for {:.0f} s - {} ion(s) presumed "
                                          "lost".format(cfg.forget_time, n), t, count=n))
            return
        if not self._advance(("RESTORE",), t, spots):
            return
        p = self._pending
        self._pending = None
        pos = self.positions(include_dark=False)
        shift, pairs, miss, extra = align(pos, p.reliable(), cfg.match_radius,
                                          cfg.max_global_shift)
        if not miss and not extra and float(np.hypot(*shift)) <= cfg.reconfig_min_px:
            self._set_state(STABLE, t)
            self._stable_since = t
            events.append(self._event("fluorescence_restored", "info",
                                      "Fluorescence back - all ions still there", t))
            return
        self._commit_reconfig(p.reliable(), det, t, events, recovered=True)
        self._set_state(STABLE, t)
        self._stable_since = t

    def _update_hold(self, det, spots, t, events):
        h = self._hold
        if t > h["until"]:
            events.append(self._event("hold_expired", "warning",
                                      "Tracking hold ('{}') timed out - resuming".format(
                                          h["reason"]), t))
            self.release()
            return
        # follow the ions loosely so positions stay meaningful
        if self.sites and len(spots):
            pos = self.positions()
            shift, pairs, _, _ = align(pos, spots, 3 * self.cfg.match_radius,
                                       self.cfg.max_global_shift)
            for k in range(len(self.sites)):
                self.sites[k].x += float(shift[0])
                self.sites[k].y += float(shift[1])
            for k, j in pairs:
                self.sites[k].x, self.sites[k].y = map(float, spots[j])

    def _update_after_hold(self, det, spots, t, events, info):
        if not self._advance(("AFTER_HOLD",), t, spots):
            return
        p = self._pending
        self._pending = None
        self._after_hold = None
        # dark ions cannot be followed through a transport: compare bright ions
        n_before = info["n_bright"]
        self.sites = []
        self._adopt(p.reliable(), [], det, t)
        n_after = len(self.sites)
        reason = info.get("reason") or "hold"
        if info.get("n_dark"):
            reason += " ({} dark ion(s) not verified)".format(info["n_dark"])
        if n_after < n_before:
            events.append(self._event(
                "ion_lost", "alarm",
                "{} ion(s) lost during '{}' ({} -> {})".format(n_before - n_after, reason,
                                                              n_before, n_after),
                t, count=n_before - n_after, before=n_before, after=n_after))
        elif n_after > n_before:
            events.append(self._event(
                "ion_trapped", "info",
                "{} more ion(s) after '{}' ({} -> {})".format(n_after - n_before, reason,
                                                             n_before, n_after),
                t, count=n_after - n_before, before=n_before, after=n_after))
        events.append(self._event("hold_done", "info",
                                  "'{}' finished: {} ion(s) before, {} after".format(
                                      reason, n_before, n_after),
                                  t, before=n_before, after=n_after))
        self._set_state(STABLE if self.sites else EMPTY, t)
        self._stable_since = t

    # -- commits ------------------------------------------------------------------------

    def _commit_sites(self, newly_dark, newly_bright, obs, t, events):
        self._avg_reset()
        for k in newly_dark:
            s = self.sites[k]
            s.state, s.since = DARK, t
        for k in newly_bright:
            s = self.sites[k]
            s.state, s.since, s.inferred = BRIGHT, t, False
        if len(newly_dark) == 1 and len(newly_bright) == 1:
            events.append(self._event(
                "dark_ion_moved", "warning",
                "Dark ion moved from site {} to site {}".format(newly_bright[0], newly_dark[0]),
                t, from_site=newly_bright[0], to_site=newly_dark[0]))
            return
        for k in newly_dark:
            events.append(self._event(
                "ion_dark", "warning",
                "Ion {} went dark - the rest of the crystal is unperturbed (dark state or "
                "dark species, the ion is still trapped)".format(k), t, site=k,
                x=self.sites[k].x, y=self.sites[k].y))
        for k in newly_bright:
            events.append(self._event("ion_bright", "info",
                                      "Ion {} is bright again".format(k), t, site=k))

    def _commit_reconfig(self, new_pos: np.ndarray, det, t, events, recovered=False):
        self._avg_reset()
        n_before = len(self.sites)
        nb_before = self.n_bright
        nd_before = self.n_dark
        prior = self._chain
        self.sites = []
        hidden = self._infer_hidden(new_pos, prior, max(nd_before, 0))
        self._adopt(new_pos, hidden, det, t)
        n_after = len(self.sites)
        nb_after = self.n_bright
        nd_after = self.n_dark
        detail = dict(before=n_before, after=n_after, bright_before=nb_before,
                      bright_after=nb_after, dark_before=nd_before, dark_after=nd_after)
        if n_after < n_before:
            lost = n_before - n_after
            what = "dark " if nb_after == nb_before and nd_after < nd_before else ""
            events.append(self._event(
                "ion_lost", "alarm",
                "{} {}ion{} lost - crystal re-arranged ({} -> {} ions)".format(
                    lost, what, "s" * (lost != 1), n_before, n_after), t, count=lost, **detail))
        elif n_after > n_before:
            gained = n_after - n_before
            events.append(self._event(
                "ion_trapped", "info",
                "Ion trapped! {} -> {} ions".format(n_before, n_after), t, count=gained,
                **detail))
        else:
            events.append(self._event(
                "chain_reconfigured", "warning",
                "Crystal re-arranged with the same number of ions ({}){}".format(
                    n_after, " after re-crystallising" if recovered else ""), t, **detail))

    def _infer_hidden(self, bright_pos: np.ndarray, prior, n_dark_before: int) -> np.ndarray:
        cfg = self.cfg
        if not cfg.use_chain_model or n_dark_before == 0 or len(bright_pos) == 0:
            return np.zeros((0, 2))
        kw = {}
        if prior is not None:
            scale, center, axis = prior
            kw = dict(scale_prior=scale, center_prior=center, axis_hint=axis)
        else:
            return np.zeros((0, 2))
        fit = fit_chain(bright_pos, max_hidden=min(cfg.max_dark_inferred, n_dark_before), **kw)
        if fit is None or not fit.is_linear or not fit.hidden_sites:
            return np.zeros((0, 2))
        return fit.hidden_positions()

    def _adopt(self, bright_pos: np.ndarray, dark_pos, det, t) -> None:
        """Replace the sites by a new configuration (sorted along the chain)."""
        bright_pos = np.asarray(bright_pos, float).reshape(-1, 2)
        dark_pos = np.asarray(dark_pos, float).reshape(-1, 2)
        allpos = np.vstack([bright_pos, dark_pos])
        states = [BRIGHT] * len(bright_pos) + [DARK] * len(dark_pos)
        if len(allpos) == 0:
            self.sites = []
            return
        vals, _, _ = det.photometry(allpos[:, 0], allpos[:, 1])
        norm = vals / max(det.gain_factor, 1e-12)
        bright_refs = [norm[i] for i in range(len(bright_pos)) if norm[i] > 0]
        med = float(np.median(bright_refs)) if bright_refs else None
        fit = fit_chain(allpos) if len(allpos) >= 2 else None
        if fit is not None:
            order = np.argsort((allpos - fit.center) @ fit.axis)
        else:
            order = np.arange(len(allpos))
        sites = []
        for i in order:
            ref = float(norm[i]) if states[i] == BRIGHT and norm[i] > 0 else med
            sites.append(Site(id=self._next_id, x=float(allpos[i, 0]), y=float(allpos[i, 1]),
                              state=states[i], ref=ref, since=t,
                              inferred=states[i] == DARK))
            self._next_id += 1
        self.sites = sites
        self._region_ref = None
        self._diffuse_ref = None
        self._region_var = None
        self._avg_reset()
        self._update_chain_fit()

    def _update_chain_fit(self) -> None:
        pos = self.positions()
        if len(pos) >= 2:
            fit = fit_chain(pos)
            if fit is not None and fit.is_linear:
                self._chain = (fit.scale, fit.center, fit.axis)
                return
            self._chain = None
        elif len(pos) == 1 and self._chain is not None:
            scale, _, axis = self._chain
            self._chain = (scale, pos[0].copy(), axis)

    def _apply_tracking(self, obs, spots, t, full: bool = False) -> None:
        """Follow slow drifts / confirmed rigid moves and adapt brightness references.

        Single-frame positions of dim ions are noisy, so normally every site only
        moves part of the way (``position_smoothing``) towards its spot - dark
        sites by the same fraction of the common shift.  A confirmed rigid move
        (``full``) is applied at once."""
        cfg = self.cfg
        a = cfg.position_smoothing
        shift = np.asarray(obs["shift"], float)
        applied = shift if full else a * shift
        self._moved_accum = self._moved_accum + applied
        self.last_shift = shift
        matched = {k: j for k, j in obs["pairs"]}
        fitted = obs.get("fitted")
        if not full and fitted is not None and np.any(np.isfinite(fitted[:, 0])):
            # the sites fitted to this frame (as many components as bright ions, so
            # two blended ions can never pull a site in between): smooth towards
            # them; dark sites follow the mean motion of the bright ones
            ok = np.isfinite(fitted[:, 0])
            pos = self.positions()
            drift = np.median(fitted[ok] - pos[ok], axis=0)
            for k, s in enumerate(self.sites):
                tx, ty = fitted[k] if ok[k] else pos[k] + drift
                s.x += a * (float(tx) - s.x)
                s.y += a * (float(ty) - s.y)
            applied = a * drift
            self._moved_accum = self._moved_accum - a * shift + applied
        else:
            for k, s in enumerate(self.sites):
                if full:
                    s.x += float(shift[0])
                    s.y += float(shift[1])
                if k in matched:
                    x, y = spots[matched[k]]
                    s.x += a * (float(x) - s.x)
                    s.y += a * (float(y) - s.y)
                elif not full:
                    s.x += float(applied[0])
                    s.y += float(applied[1])
            lv = obs["levels"][k]
            if np.isfinite(lv):
                s.level = float(lv)
            s.snr = float(obs["vals"][k] / obs["noise"][k])
            if s.state == BRIGHT and obs["inside"][k]:
                nk = float(obs["norm"][k])
                if s.ref is None:
                    s.ref = nk if nk > 0 else obs["default_ref"]
                elif nk > 0 and 0.3 < nk / s.ref < 3.0:
                    s.ref += 0.02 * (nk - s.ref)
        region = obs.get("region")
        if region is not None and region > 0 and not obs["missing"]:
            if self._region_ref is None:
                self._region_ref = region
                self._diffuse_ref = obs["diffuse"]
            else:
                dev = region - self._region_ref
                self._region_var = dev * dev if self._region_var is None else \
                    0.95 * self._region_var + 0.05 * dev * dev
                self._region_ref += 0.05 * dev
                self._diffuse_ref += 0.05 * (obs["diffuse"] - self._diffuse_ref)
        if self._chain is not None:
            scale, center, axis = self._chain
            self._chain = (scale, center + applied, axis)
            if len(self.sites) >= 2 and np.random.random() < 0.05:
                self._update_chain_fit()

    # -- re-measuring the crystal on averaged frames --------------------------------

    def _avg_reset(self) -> None:
        self._avg = None
        self._avg_n = 0
        self._avg_key = None
        self._avg_t0 = None

    def _avg_add(self, det: DetectionResult, t: float, events) -> None:
        """Accumulate the image around the (stable) crystal; every
        ``verify_period`` re-measure it on the average."""
        cfg = self.cfg
        if cfg.verify_period <= 0 or det.psf is None or not self.sites or \
                det.residual.size == 0:
            return
        psf = det.psf
        pos = self.positions()
        cu, cv = det.to_crop(pos[:, 0], pos[:, 1])
        h, w = det.residual.shape
        m = int(math.ceil(3.0 * max(psf.sx, psf.sy))) + 3
        box = (int(max(0, math.floor(np.min(cu)) - m)), int(max(0, math.floor(np.min(cv)) - m)),
               int(min(w, math.ceil(np.max(cu)) + m + 1)),
               int(min(h, math.ceil(np.max(cv)) + m + 1)))
        if box[2] - box[0] < 3 or box[3] - box[1] < 3:
            return
        key = (det.crop, det.x0, det.y0, det.hbin, det.vbin, len(self.sites))
        if self._avg is None or key != self._avg_key or \
                box[0] < self._avg_box[0] - 1 or box[1] < self._avg_box[1] - 1 or \
                box[2] > self._avg_box[2] + 1 or box[3] > self._avg_box[3] + 1:
            self._avg_key = key
            self._avg_box = box
            ua, va, ub, vb = box
            self._avg = det.residual[va:vb, ua:ub].astype(np.float64)
            self._avg_sq = self._avg ** 2
            self._avg_n = 1
            self._avg_t0 = t
            self._avg_noise = float(det.noise_f) ** 2
            return
        ua, va, ub, vb = self._avg_box
        patch = det.residual[va:vb, ua:ub]
        self._avg += patch
        self._avg_sq += patch.astype(np.float64) ** 2
        self._avg_n += 1
        self._avg_noise += float(det.noise_f) ** 2
        if t - self._avg_t0 >= cfg.verify_period and self._avg_n >= 10:
            self._verify(det, t, events)
            self._avg_reset()

    def _avg_merge(self, p: "_Pending") -> None:
        """The frames of a change that did not happen show the stable crystal:
        add them to its average.  Picked for looking different, they are a
        biased sample on their own - together with the frames that did not
        look different they are all frames again."""
        if p.avg is None or self._avg is None or p.avg_key is None:
            return
        crop, x0, y0, hb, vb, (pa, pb, pc, pd) = p.avg_key
        if (crop, x0, y0, hb, vb, len(self.sites)) != self._avg_key:
            return
        ua, va, ub, vb_ = self._avg_box
        if ua < pa or va < pb or ub > pc or vb_ > pd:
            return
        sl = (slice(va - pb, vb_ - pb), slice(ua - pa, ub - pa))
        self._avg += p.avg[sl]
        self._avg_sq += p.avg_sq[sl]
        self._avg_n += p.avg_n
        self._avg_noise += p.avg_noise2

    def _psf_for(self, det: DetectionResult) -> PSF:
        """The PSF for analyses of averaged frames: learned if available."""
        e = self.psf_estimate
        if e is None:
            return det.psf
        return PSF.from_sensor(e["sigma"], e["halo_fraction"], e["halo_sigma"],
                               det.hbin, det.vbin)

    def _learn_psf(self, mean: np.ndarray, guesses: np.ndarray, det: DetectionResult,
                   var_mean: np.ndarray) -> PSF:
        """Fit core + halo to the averaged crystal.  The estimate is kept only if
        the model then explains the image down to the noise (with the wrong
        number of ions it would not, and a PSF bent to hide an ion is worse than
        none) and the core width is plausible."""
        cur = self._psf_for(det)
        if det.hbin != 1 or det.vbin != 1 or len(guesses) == 0:
            return cur
        try:
            g = fit_gaussians(mean, guesses, sigma0=float(det.psf.sx), halo=True)
        except (ValueError, np.linalg.LinAlgError):  # pragma: no cover
            return cur
        sx, sy = g.sigma
        # two unresolved ions look like one stretched along the chain: a learned
        # PSF must be round (a wrong ion count must not be "explained" by it)
        ok = (g.success and 0.6 * det.psf.sx < sx < 1.6 * det.psf.sx and
              0.6 * det.psf.sy < sy < 1.6 * det.psf.sy and 0.92 < sx / sy < 1.087)
        if ok:
            # residuals near the ions against the noise of the average
            c0, r0, c1, r1 = g.window
            m = int(math.ceil(3 * max(sx, sy)))
            pos = g.positions
            a0 = int(max(c0, math.floor(pos[:, 0].min()) - m))
            a1 = int(min(c1, math.ceil(pos[:, 0].max()) + m + 1))
            b0 = int(max(r0, math.floor(pos[:, 1].min()) - m))
            b1 = int(min(r1, math.ceil(pos[:, 1].max()) + m + 1))
            diff = mean[b0:b1, a0:a1] - g.model[b0 - r0:b1 - r0, a0 - c0:a1 - c0]
            chi2 = float(np.mean(diff ** 2 / var_mean[b0:b1, a0:a1])) if diff.size else np.inf
            ok = chi2 < 2.0
        if not ok:
            return cur
        self.psf_estimate = {"sigma": (float(sx), float(sy)),
                             "halo_fraction": float(g.halo_fraction),
                             "halo_sigma": float(g.halo_sigma), "n_ions": len(guesses)}
        return self._psf_for(det)

    def _verify(self, det: DetectionResult, t: float, events) -> None:
        """Compare the sites with what the averaged image shows."""
        n = self._avg_n
        mean = self._avg / n
        noise_f = math.sqrt(self._avg_noise / n) / math.sqrt(n)
        ua, va = self._avg_box[:2]
        bright = [s for s in self.sites if s.state == BRIGHT]
        if not bright:
            return
        bpos = np.array([[s.x, s.y] for s in bright])
        cu, cv = det.to_crop(bpos[:, 0], bpos[:, 1])
        guesses = np.column_stack([cu - ua, cv - va])
        allowed = np.ones(mean.shape, bool)
        e = int(math.ceil(1.5 * max(det.psf.sx, det.psf.sy)))
        allowed[:e, :] = allowed[-e:, :] = False
        allowed[:, :e] = allowed[:, -e:] = False
        # per-pixel variance of the average (includes the ions' own shot noise)
        var_mean = _variance_of_mean(mean, self._avg_sq, n)
        psf = self._learn_psf(mean, guesses, det, var_mean)
        fit = deblend_patch(mean, guesses, psf, noise_f, 6.0, det.sigma_f, allowed,
                            max_new=3, min_new_flux=0.5, chain_search=True, var=var_mean)
        if fit is None:
            return
        found = fit.positions + [ua, va]
        hb, vb = det.hbin, det.vbin
        fx = (found[:, 0] + det.crop[0]) * hb + det.x0 + 0.5 * (hb - 1)
        fy = (found[:, 1] + det.crop[1]) * vb + det.y0 + 0.5 * (vb - 1)
        found = np.column_stack([fx, fy])
        shift, pairs, miss, extra = align(bpos, found, self.cfg.match_radius, 2.0)
        if (not miss and not extra) or len(found) == len(bright):
            # same crystal, measured more precisely: refine the site positions
            for i, j in pairs:
                bright[i].x, bright[i].y = map(float, found[j])
            return
        dark = [s for s in self.sites if s.state == DARK and
                np.min(np.hypot(found[:, 0] - s.x, found[:, 1] - s.y)) > self.cfg.match_radius]
        n_before = len(bright)
        self._adopt(found, np.array([[s.x, s.y] for s in dark]).reshape(-1, 2), det, t)
        events.append(self._event(
            "crystal_refined", "info",
            "Re-measured on {} averaged frames: {} bright ion{} (single frames showed {})".format(
                n, len(found), "s" * (len(found) != 1), n_before), t,
            before=n_before, after=len(found), frames=n))

    def _crystal_box(self, det: DetectionResult, extra_pos=None):
        """Crop-coordinate box around the sites (and ``extra_pos``) with PSF margin."""
        pos = self.positions()
        if extra_pos is not None and len(extra_pos):
            pos = np.vstack([pos, extra_pos])
        if len(pos) == 0 or det.psf is None or det.residual.size == 0:
            return None
        cu, cv = det.to_crop(pos[:, 0], pos[:, 1])
        h, w = det.residual.shape
        m = int(math.ceil(3.0 * max(det.psf.sx, det.psf.sy))) + 3
        box = (int(max(0, math.floor(np.min(cu)) - m)), int(max(0, math.floor(np.min(cv)) - m)),
               int(min(w, math.ceil(np.max(cu)) + m + 1)),
               int(min(h, math.ceil(np.max(cv)) + m + 1)))
        if box[2] - box[0] < 3 or box[3] - box[1] < 3:
            return None
        return box

    def _pending_average(self, p: "_Pending", det: DetectionResult) -> None:
        """Sum the frames that support a pending re-configuration."""
        if p.avg is None:
            box = self._crystal_box(det, p.positions)
            if box is None:
                return
            p.avg_key = (det.crop, det.x0, det.y0, det.hbin, det.vbin, box)
            ua, va, ub, vb = box
            p.avg = det.residual[va:vb, ua:ub].astype(np.float64)
            p.avg_sq = p.avg ** 2
            p.avg_n, p.avg_noise2 = 1, float(det.noise_f) ** 2
            return
        key = p.avg_key
        if key[:5] != (det.crop, det.x0, det.y0, det.hbin, det.vbin):
            return
        ua, va, ub, vb = key[5]
        patch = det.residual[va:vb, ua:ub].astype(np.float64)
        p.avg += patch
        p.avg_sq += patch ** 2
        p.avg_n += 1
        p.avg_noise2 += float(det.noise_f) ** 2

    def _learn_psf_on(self, p: "_Pending", det: DetectionResult) -> None:
        """Measure the PSF on the averaged frames of a new crystal before its
        ions are counted: a detector still assuming the wrong PSF merges and
        splits ions in tight chains.  (Accepted only if the fit explains the
        average - see _learn_psf.)"""
        if p.avg is None or p.avg_n < 4 or det.psf is None:
            return
        n = p.avg_n
        mean = p.avg / n
        var_mean = _variance_of_mean(mean, p.avg_sq, n)
        crop, x0, y0, hb, vb, box = p.avg_key
        ua, va = box[:2]
        for start in (p.reliable(), p.positions):
            if start is None or len(start) == 0:
                continue
            g = np.column_stack([(start[:, 0] - x0 - 0.5 * (hb - 1)) / hb - crop[0] - ua,
                                 (start[:, 1] - y0 - 0.5 * (vb - 1)) / vb - crop[1] - va])
            before = self.psf_estimate
            self._learn_psf(mean, g, det, var_mean)
            if self.psf_estimate is not before:
                return

    def _average_fit(self, p: "_Pending", det: DetectionResult, starts):
        """Analyse the average of the frames that supported ``p``, starting from each
        position set in ``starts`` (sensor px); the best explanation wins (every
        ion costs a penalty).  Returns sensor positions or None."""
        if p.avg is None or p.avg_n < 4 or det.psf is None:
            return None
        n = p.avg_n
        mean = p.avg / n
        noise_f = math.sqrt(p.avg_noise2 / n) / math.sqrt(n)
        var_mean = _variance_of_mean(mean, p.avg_sq, n)
        crop, x0, y0, hb, vb, box = p.avg_key
        ua, va = box[:2]

        def to_patch(p_):
            return np.column_stack([(p_[:, 0] - x0 - 0.5 * (hb - 1)) / hb - crop[0] - ua,
                                    (p_[:, 1] - y0 - 0.5 * (vb - 1)) / vb - crop[1] - va])
        allowed = np.ones(mean.shape, bool)
        e = int(math.ceil(1.5 * max(det.psf.sx, det.psf.sy)))
        allowed[:e, :] = allowed[-e:, :] = False
        allowed[:, :e] = allowed[:, -e:] = False
        fit = None
        for start in starts:
            if start is None or len(start) == 0:
                continue
            f_ = deblend_patch(mean, to_patch(start), self._psf_for(det), noise_f, 6.0,
                               det.sigma_f, allowed, max_new=2, min_new_flux=0.5,
                               chain_search=True, var=var_mean)
            if f_ is None:
                continue
            if fit is None or (f_.cost + 25.0 * f_.rms ** 2 * len(f_.flux) <
                               fit.cost + 25.0 * fit.rms ** 2 * len(fit.flux)):
                fit = f_
        if fit is None:
            return None
        f = fit.positions + [ua, va]
        return np.column_stack([(f[:, 0] + crop[0]) * hb + x0 + 0.5 * (hb - 1),
                                (f[:, 1] + crop[1]) * vb + y0 + 0.5 * (vb - 1)])

    def _avg_costs(self, p: "_Pending", det: DetectionResult, configs):
        """Chi-square of the averaged frames of ``p`` for each ``(positions,
        max_move)`` (sensor px; one fixed-PSF fit each), or None."""
        if p.avg is None or p.avg_n < 4 or det.psf is None:
            return None
        n = p.avg_n
        mean = p.avg / n
        var_mean = _variance_of_mean(mean, p.avg_sq, n)
        crop, x0, y0, hb, vb, box = p.avg_key
        ua, va = box[:2]
        out = []
        for pos_, max_move in configs:
            if len(pos_) == 0:
                out.append(float(np.sum(mean ** 2 / var_mean)))
                continue
            g = np.column_stack([(pos_[:, 0] - x0 - 0.5 * (hb - 1)) / hb - crop[0] - ua,
                                 (pos_[:, 1] - y0 - 0.5 * (vb - 1)) / vb - crop[1] - va])
            try:
                out.append(fit_fixed_psf(mean, g, self._psf_for(det), max_move=max_move,
                                         var=var_mean).cost)
            except (ValueError, np.linalg.LinAlgError):  # pragma: no cover
                return None
        return out

    def _clearly_better(self, p: "_Pending", det: DetectionResult, old: np.ndarray,
                        new: np.ndarray, move: float = 0.5) -> bool:
        """Does configuration ``new`` explain the averaged frames of ``p``
        clearly better than ``old`` (chi-square lower by more than the usual
        penalty per ion of difference, at least one)?  Both are fitted with
        their own number of ions, ``old``'s ions free to move by ``move`` px
        (little when the counts differ: two ions pulled together would imitate
        one)."""
        if len(old) == 0:
            return True
        c = self._avg_costs(p, det, ((old, move), (new, 0.5)))
        if c is None:
            return True
        return c[0] - c[1] > 25.0 * max(1, abs(len(new) - len(old)))

    def _decide_change(self, p: "_Pending", det: DetectionResult):
        """What the averaged frames of a pending change show: ("dark", sites)
        if the rest of the crystal stayed exactly where it was, else the
        verdict of _confirm_on_average ("same", "hidden" or "changed" with
        positions)."""
        dark = max(p.dark_votes, key=p.dark_votes.get) if p.dark_votes else None
        if dark is not None and self._rest_moved(p, det, dark) == "unmoved":
            return "dark", dark
        return self._confirm_on_average(p, det)

    def _rest_moved(self, p: "_Pending", det: DetectionResult, newly_dark) -> str:
        """An ion that goes dark leaves the rest of the crystal exactly where
        it was (and its site dark); a lost ion makes the rest re-arrange.  In a
        tight chain the re-centred ions can still look like the old sites
        minus one (each pulled towards the gap, as if merged with a neighbour),
        so check on the average of the frames that showed the change, fitted
        from the remaining bright sites as they were before: "unmoved",
        "moved" or "unknown"."""
        rest = [k for k, s_ in enumerate(self.sites)
                if s_.state == BRIGHT and k not in set(newly_dark)]
        if not rest:
            return "unknown"
        allpos = self.positions()
        if p.site_pos is not None and len(p.site_pos) == len(allpos):
            allpos = p.site_pos                 # where they were before the change
        old = allpos[rest]
        found = self._average_fit(p, det, (old,))
        if found is None or len(found) == 0:
            return "unknown"
        if len(found) > len(old):
            return "unknown"                    # light where the 'dark' ion was
        pairs = _assign(old, found, 2.0 * self.cfg.match_radius)
        if len(found) < len(old) or len(pairs) < len(found):
            return "moved"
        d = max(float(np.hypot(*(found[j] - old[i]))) for i, j in pairs)
        if len(allpos) >= 2:
            dd = np.hypot(allpos[:, None, 0] - allpos[None, :, 0],
                          allpos[:, None, 1] - allpos[None, :, 1])
            np.fill_diagonal(dd, np.inf)
            spacing = float(dd.min())
        else:
            spacing = 10.0
        return "moved" if d > max(0.4, 0.15 * spacing) else "unmoved"

    def _confirm_on_average(self, p: "_Pending", det: DetectionResult):
        """Check a re-configuration on the average of the frames that showed it.

        Returns ``(verdict, positions)``: "changed" with the new bright
        positions (measured on the average, so more precise than single
        frames), "hidden" if every old site is still there plus ions that single
        frames could not resolve, or "same" if the averaged image still shows
        the old crystal (the "change" was single-frame noise)."""
        new_pos = p.reliable()
        old = self.positions(include_dark=False)
        found = self._average_fit(p, det, (new_pos, old))
        if found is None:
            return "changed", new_pos
        cfg = self.cfg
        shift, pairs, miss, extra = align(old, found, 1.5 * cfg.match_radius,
                                          cfg.max_global_shift)
        if not pairs:
            return "changed", found
        spacing = self._local_spacing(old)
        tol = np.maximum(cfg.reconfig_min_px, cfg.reconfig_fraction * spacing)
        moved = any(float(np.hypot(*(found[j] - old[i] - shift))) > tol[i] for i, j in pairs)
        # an ion that was hidden all along leaves the others exactly where they
        # were; a chain that gained or lost an ion re-centres and re-spaces
        shifted = float(np.hypot(*shift)) > max(cfg.reconfig_min_px, 3.0 * self._shift_noise)
        # The old crystal may move by up to the motion threshold when the
        # counts agree: on a short average of dim, blended ions the free fit
        # always finds positions that explain the noise a little better
        move = float(np.min(tol)) if len(found) == len(old) and len(tol) else 0.5
        if (miss or moved or shifted or extra) and \
                not self._clearly_better(p, det, old, found, move):
            # the crystal as it was explains the averaged frames as well: keep it
            # (single frames of a dim, dense chain change their mind; a real
            # change shows on the average)
            return "same", old
        if miss or moved or shifted:
            return "changed", found
        if extra:
            # hidden = too close to a neighbour to be resolved in single frames;
            # an extra ion well away from all others is a newly trapped one
            res = 3.0 * max(det.psf.sx * det.hbin, det.psf.sy * det.vbin)
            if all(np.min(np.hypot(*(old - found[j]).T)) < res for j in extra):
                return "hidden", found
            return "changed", found
        # Same ions.  The sites stay where they are: these frames were picked
        # for looking different, so their average is no unbiased measurement
        # of a crystal that did not change (the stable average re-measures it,
        # see _verify)
        return "same", old

    def _check_dim(self, t, events) -> None:
        lv = [s.level for s in self.sites if s.state == BRIGHT and np.isfinite(s.level)]
        if not lv:
            return
        level = float(np.median(lv))
        self.fluorescence_level = level
        if not self._dim and level < 0.5:
            self._dim = True
            events.append(self._event("fluorescence_dim", "warning",
                                      "Ion fluorescence dropped to {:.0f} % (check cooling laser "
                                      "power/detuning)".format(100 * level), t, level=level))
        elif self._dim and level > 0.75:
            self._dim = False
            events.append(self._event("fluorescence_ok", "info",
                                      "Ion fluorescence back to {:.0f} %".format(100 * level), t,
                                      level=level))

    # -- helpers ---------------------------------------------------------------------------

    def _advance(self, key: tuple, t: float, spots: np.ndarray) -> bool:
        """Debounce: True once ``key`` has persisted long enough (same configuration)."""
        cfg = self.cfg
        p = self._pending
        config_key = key[0] in ("CHANGE", "APPEAR", "REFORM", "RESTORE", "AFTER_HOLD")
        same = p is not None and p.key == key
        if same and config_key:
            if _same_configuration(p.reliable(), spots, 1.5 * cfg.match_radius):
                p.misses = 0
            else:
                # One odd frame (a noise peak fitted as an extra ion, two ions
                # merged) does not restart the confirmation - the per-position
                # hit counts keep it out; a different configuration that
                # persists does.
                p.misses += 1
                same = p.frames >= 2 and p.misses < 2
        if not same:
            self._pending = _Pending(key, t, 1, spots.copy(),
                                     site_pos=self.positions().copy() if self.sites else None)
            return cfg.confirm_frames <= 1 and cfg.confirm_time <= 0
        p.frames += 1
        if config_key:
            p.merge(spots, 1.5 * cfg.match_radius)
        return p.frames >= cfg.confirm_frames and t - p.t0 >= cfg.confirm_time

    def _integrated_clouds(self, det, t) -> List[tuple]:
        """Faint clouds: a ~0.3 s average of the cloud-scale map compared with a
        slowly learned reference (~30 s), so static scatter and surface defects
        never count, but new diffuse light does.  Returns [(x, y, snr), ...]."""
        if det.diffuse is None:
            self._cl = None
            return []
        f2, noise, sc, nb = det.diffuse
        key = (det.crop, det.x0, det.y0, det.hbin, det.vbin, f2.shape)
        c = self._cl
        if c is None or c["key"] != key:
            self._cl = {"key": key, "fast": f2.astype(np.float64), "slow": f2.astype(np.float64),
                        "n": 1, "t": t, "since": None}
            return []
        dt = max(t - c["t"], 1e-3)
        c["t"] = t
        c["n"] += 1
        a_f = max(min(1.0, dt / 0.3), 1.0 / c["n"])
        a_s = max(min(1.0, dt / 30.0), 1.0 / c["n"])
        c["fast"] += a_f * (f2 - c["fast"])
        if c["n"] * min(1.0, dt / 0.3) < 3.0:
            # warming up (see _faint_ions)
            c["slow"] += a_s * (f2 - c["slow"])
            c["active"] = []
            return []
        excess = c["fast"] - c["slow"]
        sig = np.broadcast_to(np.asarray(noise, np.float64), excess.shape) * \
            math.sqrt(a_f / (2.0 - a_f) + a_s / (2.0 - a_s))
        mask = excess > 6.0 * sig
        # hysteresis: a cloud already seen stays while it is above 4 sigma
        for iu, iv in c.get("active", []):
            r = int(math.ceil(2.0 * sc))
            win = (slice(max(0, iv - r), iv + r + 1), slice(max(0, iu - r), iu + r + 1))
            mask[win] |= excess[win] > 4.0 * sig[win]
        found = []
        if np.any(mask):
            pos = self.positions()
            cu, cv = det.to_crop(pos[:, 0], pos[:, 1]) if len(pos) else (np.zeros(0), np.zeros(0))
            cu, cv = np.atleast_1d(cu) / nb, np.atleast_1d(cv) / nb
            spots_uv = [((sp.u - det.crop[0]) / nb, (sp.v - det.crop[1]) / nb)
                        for sp in det.spots]
            from ioncam.analysis.detection import _local_maxima
            idx = _local_maxima(excess, mask)
            for i in idx[np.argsort(-excess.ravel()[idx])][:5]:
                iv, iu = divmod(int(i), excess.shape[1])
                near_ion = any(math.hypot(iu - a, iv - b) < 2.0 * sc for a, b in
                               list(zip(np.atleast_1d(cu), np.atleast_1d(cv))) + spots_uv)
                if near_ion or any(math.hypot(iu - a, iv - b) < 2.5 * sc for a, b, _ in found):
                    continue
                found.append((iu, iv, float(excess[iv, iu] / sig[iv, iu])))
        c["active"] = [(iu, iv) for iu, iv, _ in found]
        excess_snr = excess / sig
        if found:
            c["since"] = c["since"] or t
            if t - c["since"] > 20.0:
                # a "cloud" that just sits there is a change of the background
                c["slow"] = c["fast"].copy()
                c["since"] = None
                found = []
        else:
            c["since"] = None
        # the reference learns the usual background, but not light that is
        # (becoming) significant: a faint cloud would otherwise leak into it
        # before it is ever reported
        c["slow"] += a_s * np.where(excess_snr < 3.0, f2 - c["slow"], 0.0)
        c["excess_snr"] = excess_snr
        c["bin"] = nb
        hb, vb = det.hbin, det.vbin
        return [((nb * iu + 0.5 * (nb - 1) + det.crop[0]) * hb + det.x0 + 0.5 * (hb - 1),
                 (nb * iv + 0.5 * (nb - 1) + det.crop[1]) * vb + det.y0 + 0.5 * (vb - 1), snr)
                for iu, iv, snr in found]

    def _is_new_light(self, det, x, y) -> bool:
        """Is there more diffuse light at sensor (x, y) than usual (integrator)?"""
        c = self._cl
        if c is None or "excess_snr" not in c or c["n"] < 10:
            return True
        cu, cv = det.to_crop(x, y)
        h, w = c["excess_snr"].shape
        nb = c.get("bin", 1)
        iu, iv = int(round((float(cu) - 0.5 * (nb - 1)) / nb)), \
            int(round((float(cv) - 0.5 * (nb - 1)) / nb))
        if not (0 <= iu < w and 0 <= iv < h):
            return True
        return float(c["excess_snr"][iv, iu]) > 3.0

    def _update_clouds(self, det, t, events) -> None:
        faint = self._integrated_clouds(det, t)
        # broad blobs that were always there (defocused surface defects) are not clouds
        clouds = [(c.x, c.y, c.width) for c in det.clouds if self._is_new_light(det, c.x, c.y)]
        for x, y, _ in faint:
            if all(math.hypot(x - a, y - b) > max(w_, 6.0) for a, b, w_ in clouds):
                clouds.append((x, y, 0.0))
        self.n_clouds = len(clouds)
        n = len(clouds)
        if n == self._clouds_reported:
            self._cloud_pending = None
            return
        key = ("CLOUDS", n)
        p = self._cloud_pending
        if p is None or p.key != key:
            self._cloud_pending = _Pending(key, t)
            return
        p.frames += 1
        if p.frames >= self.cfg.confirm_frames and t - p.t0 >= self.cfg.confirm_time:
            if n > self._clouds_reported:
                cx, cy, cw = clouds[0]
                events.append(self._event(
                    "cloud_detected", "info",
                    "Hot, uncrystallised ion cloud near ({:.0f}, {:.0f}) - something is "
                    "trapped; it should crystallise with good cooling".format(cx, cy), t,
                    x=cx, y=cy, width=cw))
            else:
                events.append(self._event("cloud_gone", "info", "Ion cloud gone", t))
            self._clouds_reported = n
            self._cloud_pending = None

    def _record_history(self, obs, t) -> None:
        lv = obs["norm"]
        self.history.append((t, float(np.sum(lv)), [float(v) for v in lv]))

    def _set_state(self, state: str, t: float) -> None:
        if state != self.state:
            self.state = state
            self._state_since = t

    def _event(self, kind: str, severity: str, message: str, t: float, **details) -> dict:
        return {"kind": kind, "severity": severity, "message": message, "t": t,
                "n_bright": self.n_bright, "n_dark": self.n_dark, "n_ions": len(self.sites),
                "details": details}
