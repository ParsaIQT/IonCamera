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
from ioncam.analysis.detection import DetectionResult

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
        shift = disp.mean(axis=0) if len(pairs) >= 2 else disp[0]
    pairs = _assign(a + shift, b, gate)
    ma = {i for i, _ in pairs}
    mb = {j for _, j in pairs}
    return (shift, pairs, [i for i in range(na) if i not in ma],
            [j for j in range(nb) if j not in mb])


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
    allowed = 1 if n >= 6 else 0
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
        self._moved_accum = np.zeros(2)
        self._dim = False
        self.fluorescence_level = float("nan")
        self.last_shift = np.zeros(2)
        self.history: Deque[Tuple[float, float, List[float]]] = deque(maxlen=3000)
        self.n_spots = 0
        self.n_clouds = 0
        self.last_update = 0.0

    @property
    def n_bright(self) -> int:
        return sum(1 for s in self.sites if s.state == BRIGHT)

    @property
    def n_dark(self) -> int:
        return sum(1 for s in self.sites if s.state == DARK)

    def is_stable(self, t: Optional[float] = None, stable_time: Optional[float] = None) -> bool:
        t = self.clock() if t is None else t
        st = self.cfg.stable_time if stable_time is None else stable_time
        settled = self._pending is None or self._pending.frames < 2
        return (self.state in (STABLE, EMPTY) and settled and
                self._stable_since is not None and t - self._stable_since >= st)

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
        if self._pending is not None and self._pending.frames >= 2:
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

    # -- main update -------------------------------------------------------------------

    def update(self, det: DetectionResult, t: Optional[float] = None) -> List[dict]:
        t = self.clock() if t is None else t
        self.last_update = t
        events: List[dict] = []
        spots = np.array([[s.x, s.y] for s in det.spots], float).reshape(-1, 2)
        self.n_spots = len(spots)
        self.n_clouds = len(det.clouds)
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

    def _update_empty(self, det, spots, t, events):
        if len(spots) == 0:
            self._pending = None
            self._first_config = False      # we have seen the trap empty
            if self.state != EMPTY:
                self._set_state(EMPTY, t)
            if self._stable_since is None:
                self._stable_since = t
            return
        key = ("APPEAR",)
        if self._advance(key, t, spots):
            p = self._pending
            self._pending = None
            self._adopt(p.reliable(), [], det, t)
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
        vals, noise, rx, ry = det.sample(pos[:, 0], pos[:, 1], refine=1)
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
        region = self._region_flux(det, pos)
        pairs = [(bright_idx[i], j) for i, j in pairs]
        missing = [bright_idx[i] for i in miss_b]
        return {"pos": pos, "vals": vals, "noise": noise, "norm": norm, "levels": levels,
                "inside": inside, "bright_idx": bright_idx, "dark_idx": dark_idx,
                "shift": shift, "pairs": pairs, "missing": missing, "extra": extra,
                "default_ref": default_ref, "region": region}

    def _region_flux(self, det: DetectionResult, pos: np.ndarray) -> Optional[float]:
        """Background-subtracted light in a box around the crystal (normalised).

        Tells a melted crystal (light still there, just diffuse) from lasers
        off / ions gone (no light at all)."""
        res = det.residual
        if len(pos) == 0 or res.size == 0:
            return None
        cu, cv = det.to_crop(pos[:, 0], pos[:, 1])
        mu = 3.0 * self.cfg.match_radius / det.hbin + 2.0
        mv = 3.0 * self.cfg.match_radius / det.vbin + 2.0
        h, w = res.shape
        u0 = int(max(0, math.floor(np.min(cu) - mu)))
        u1 = int(min(w, math.ceil(np.max(cu) + mu) + 1))
        v0 = int(max(0, math.floor(np.min(cv) - mv)))
        v1 = int(min(h, math.ceil(np.max(cv) + mv) + 1))
        if u1 <= u0 or v1 <= v0:
            return None
        return float(res[v0:v1, u0:u1].sum()) / max(det.gain_factor, 1e-12)

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
        tol = np.maximum(cfg.reconfig_min_px, cfg.reconfig_fraction * spacing)
        small_shift = float(np.hypot(*shift)) <= max(cfg.reconfig_min_px,
                                                    0.5 * float(np.min(tol)) if len(tol) else 1.0)

        # bright sites without a spot that are still bright at their position:
        # the detector merged them with a neighbour (dense chains)
        merged = [k for k in obs["missing"] if inside[k] and np.isfinite(levels[k])
                  and levels[k] >= cfg.bright_level]
        # residual motion of matched ions after the common shift
        reconf = False
        for k, j in obs["pairs"]:
            d = spots[j] - (pos[k] + shift)
            if float(np.hypot(*d)) > tol[k]:
                # a spot pulled towards a merged neighbour is not motion
                if any(np.hypot(*(pos[m] - pos[k])) < 1.3 * spacing[k] for m in merged):
                    continue
                reconf = True
        lv = levels[bright_obs] if bright_obs else np.array([])
        lv = lv[np.isfinite(lv)]
        med_level = float(np.median(lv)) if lv.size else float("nan")

        region = obs["region"]
        region_level = (region / self._region_ref
                        if region is not None and self._region_ref else None)
        diffuse = region_level is not None and region_level >= 0.5 * cfg.melt_level
        matched_frac = len(obs["pairs"]) / max(1, len(bright_obs))
        # melted: sites dim, but the light is still there - spread out
        if bright_obs and lv.size and med_level < cfg.melt_level and matched_frac < 0.5 and \
                (diffuse or det.clouds):
            return ("MELT",)
        # everything dark (no light around the crystal at all)?
        if bright_obs and len(spots) == 0 and not det.clouds and \
                (not lv.size or np.all(lv < cfg.dark_level)):
            return ("ALL_DARK",)

        newly_dark, ambiguous = [], False
        for k in obs["missing"]:
            if not inside[k]:
                continue
            lvk = levels[k]
            if not np.isfinite(lvk) or lvk < cfg.dark_level:
                newly_dark.append(k)
            elif lvk < cfg.bright_level:
                ambiguous = True
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
        if reconf or unexplained or (obs["missing"] and not small_shift):
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
        if kind in ("STABLE", "MOVED"):
            self._pending = None
            self._apply_tracking(obs, spots, t)
            if kind == "MOVED" or np.hypot(*self._moved_accum) > 3.0:
                self._moved_accum = np.zeros(2)
                events.append(self._event(
                    "chain_moved", "info",
                    "Crystal moved rigidly by ({:+.1f}, {:+.1f}) px".format(*obs["shift"]), t,
                    shift=[float(obs["shift"][0]), float(obs["shift"][1])]))
            if self.state != STABLE:
                # a persistent-but-unconfirmed change faded away: stable again from now
                self._set_state(STABLE, t)
                self._stable_since = t
            if self._stable_since is None:
                self._stable_since = t
            self._check_dim(t, events)
            return
        committed = self._advance(hyp, t, spots)
        # a single odd frame (noise peak, cosmic ray) is not instability; only a
        # hypothesis that persists makes the crystal "changing"
        if self._pending is not None and self._pending.frames >= 2 and self.state == STABLE:
            self._set_state(CHANGING, t)
        if not committed:
            return
        p = self._pending
        self._pending = None
        if kind == "SITES":
            self._commit_sites(hyp[1], hyp[2], obs, t, events)
        elif kind == "RECONFIG":
            self._commit_reconfig(p.reliable(), det, t, events)
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
        vals, _, _, _ = det.sample(allpos[:, 0], allpos[:, 1])
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

    def _apply_tracking(self, obs, spots, t) -> None:
        """Follow slow drifts / rigid moves and adapt brightness references."""
        cfg = self.cfg
        a = cfg.position_smoothing
        shift = obs["shift"]
        self._moved_accum = self._moved_accum + shift
        self.last_shift = shift
        matched = {k: j for k, j in obs["pairs"]}
        for k, s in enumerate(self.sites):
            if k in matched:
                x, y = spots[matched[k]]
                s.x += float(shift[0]) + a * (float(x) - s.x - float(shift[0]))
                s.y += float(shift[1]) + a * (float(y) - s.y - float(shift[1]))
            else:
                s.x += float(shift[0])
                s.y += float(shift[1])
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
            else:
                self._region_ref += 0.05 * (region - self._region_ref)
        if self._chain is not None:
            scale, center, axis = self._chain
            self._chain = (scale, center + shift, axis)
            if len(self.sites) >= 2 and np.random.random() < 0.05:
                self._update_chain_fit()

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
        config_key = key[0] in ("RECONFIG", "APPEAR", "REFORM", "RESTORE", "AFTER_HOLD")
        same = p is not None and p.key == key
        if same and config_key:
            same = _same_configuration(p.reliable(), spots, 1.5 * cfg.match_radius)
        if not same:
            self._pending = _Pending(key, t, 1, spots.copy())
            return cfg.confirm_frames <= 1 and cfg.confirm_time <= 0
        p.frames += 1
        if config_key:
            p.merge(spots, 1.5 * cfg.match_radius)
        return p.frames >= cfg.confirm_frames and t - p.t0 >= cfg.confirm_time

    def _update_clouds(self, det, t, events) -> None:
        n = len(det.clouds)
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
                c = det.clouds[0]
                events.append(self._event(
                    "cloud_detected", "info",
                    "Hot, uncrystallised ion cloud near ({:.0f}, {:.0f}) - something is "
                    "trapped; it should crystallise with good cooling".format(c.x, c.y), t,
                    x=c.x, y=c.y, width=c.width))
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
