"""Grabber readout planning: AQROI + per-ion Grabber ROIs, verification, thresholds.

Geometry
--------
The camera reads the AQROI ``[x0, x1) x [y0, y1)`` (sensor px) with on-chip
binning ``hbin x vbin``; the resulting ``W x H`` frame is what the ARTIQ
Grabber receives over Camera Link.  Each ion gets its own rectangular ROI of
binned pixels (sizes may differ between ions).  Where the ion sits relative to the binning grid matters: an ion on a
super-pixel boundary splits its light.  The grid *phase* is set by the
AQROI origin, so the planner tries every phase (``hbin x vbin`` choices) and
every ROI size, computing for each ion the exact fraction of its fitted PSF
(Gaussian core plus, if measured, the broad halo of aberration wings that
dominates the crosstalk of real imaging systems) inside its own ROI
(*containment*) and inside every other ROI (*crosstalk*), and keeps the
placement with the best worst-case containment
(smallest ROIs first among those meeting ``min_containment``).  The AQROI is
then the bounding box of all ROIs plus a symmetric margin, i.e. centred on
the crystal.  Ions closer than the binning can separate still get a plan:
every ion its own best ROI, overlapping ROIs allowed (the Grabber sums each
ROI independently), with the crosstalk reported so it can be measured (e.g.
in a SPAM experiment) - the planner never refuses because ions are close.

By hand: :func:`reposition_plan` moves the AQROI in sensor-pixel steps (the
ROIs move with it - they are frame coordinates - so with binning this shifts
the ions by a fraction of a super-pixel inside their ROIs), and
``plan_readout(..., aqroi=...)`` places the ROIs inside a given AQROI.

Grabber semantics (from the ARTIQ gateware, ``artiq/gateware/grabber/core.py``)
-------------------------------------------------------------------------------
* coordinates are 0-based binned pixels in Camera Link stream order, which is
  the raw sensor order (Andor rotate/flip happen in host software only);
* a ROI ``setup_roi(n, x0, y0, x1, y1)`` sums ``x0 <= x < x1, y0 <= y < y1``;
* the x "in ROI" flag is only cleared when the pixel counter equals ``x1`` and
  otherwise persists across lines until end-of-frame - a ROI with
  ``x1 > W`` therefore silently sums whole rows.  :func:`validate_roi` rejects
  this and :func:`grabber_roi_sum_exact` emulates it for the tests.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import ndtr

Rect = Tuple[int, int, int, int]


class PlanError(ValueError):
    pass


@dataclass
class ReadoutPlan:
    n_ions: int
    positions: List[Tuple[float, float]]      # ion positions, sensor px
    ion_states: List[str]                     # "bright"/"dark" at calibration
    psf_sigma: Tuple[float, float]            # (sigma_x, sigma_y) sensor px
    hbin: int
    vbin: int
    roi: Rect                                 # AQROI, sensor px, half-open
    frame_size: Tuple[int, int]               # (W, H) binned pixels = Grabber frame
    roi_size: Tuple[int, int]                 # common ROI size (ru, rv); (0, 0) = per ion
    ion_rois: List[Rect]                      # in SDK frame coordinates (u0, v0, u1, v1)
    background_roi: Optional[Rect]            # SDK frame coordinates
    grabber_rois: List[Rect]                  # Grabber coordinates: ions, then background
    containment: List[float]                  # fraction of each ion's light in its ROI
    crosstalk: List[float]                    # worst fraction of another ion's light in ROI i
    crosstalk_matrix: List[List[float]]       # [i][j]: fraction of ion j's light in ROI i
    background_contamination: float           # ion light in the background ROI (in ions)
    flip_x: bool = False
    flip_y: bool = False
    psf_halo: Tuple[float, float] = (0.0, 0.0)  # (fraction, sigma [sensor px]) of the halo
    warnings: List[str] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    verification: Optional[dict] = None
    applied: Optional[dict] = None            # camera settings actually used (filled at arm)
    binning_advice: Optional[list] = None     # quality of alternative binnings
    notes: List[str] = field(default_factory=list)    # warnings not about the ROI placement
    targets: Tuple[float, float] = (0.6, 0.12)        # (min containment, max crosstalk)
    manual_offset: Tuple[int, int] = (0, 0)           # AQROI moved by hand [sensor px]

    @property
    def n_rois(self) -> int:
        return len(self.grabber_rois)

    @property
    def ion_mask(self) -> int:
        return (1 << self.n_ions) - 1

    @property
    def full_mask(self) -> int:
        return (1 << self.n_rois) - 1

    @property
    def background_index(self) -> int:
        return self.n_ions if self.background_roi is not None else -1

    @property
    def roi_labels(self) -> List[str]:
        """Grabber ROI n -> what it measures ("ion 0", "ion 1", ..., "bg")."""
        return ["ion {}".format(i) for i in range(self.n_ions)] + \
            (["bg"] if self.background_roi is not None else [])

    def add_note(self, text: str) -> None:
        """A warning that does not depend on where the ROIs sit."""
        self.notes.append(text)
        self.warnings.append(text)

    def sdk_image(self) -> Tuple[int, int, int, int, int, int]:
        """``SetImage`` arguments (hbin, vbin, hstart, hend, vstart, vend)."""
        x0, y0, x1, y1 = self.roi
        return self.hbin, self.vbin, x0 + 1, x1, y0 + 1, y1

    def sdk_rois(self) -> List[Rect]:
        """All ROIs (ions, then background) in SDK frame coordinates."""
        return list(self.ion_rois) + ([self.background_roi] if self.background_roi else [])

    def roi_sums(self, frames: np.ndarray) -> np.ndarray:
        """Per-ROI sums of SDK-oriented frames: (n_frames, n_rois)."""
        f = np.asarray(frames)
        if f.ndim == 2:
            f = f[None]
        f = f.astype(np.int64, copy=False)
        return np.stack([f[:, v0:v1, u0:u1].sum(axis=(1, 2))
                         for (u0, v0, u1, v1) in self.sdk_rois()], axis=1)

    def to_dict(self) -> dict:
        x0, y0, x1, y1 = self.roi
        g = self.grabber_rois
        return {
            "n_ions": self.n_ions,
            "positions": [[round(float(x), 3), round(float(y), 3)] for x, y in self.positions],
            "ion_states": list(self.ion_states),
            "psf_sigma": [round(float(s), 4) for s in self.psf_sigma],
            "psf_halo": [round(float(s), 4) for s in self.psf_halo],
            "binning": [self.hbin, self.vbin],
            "aqroi": [x0, y0, x1, y1],
            "aqroi_solis": [x0 + 1, x1, y0 + 1, y1],    # Left, Right, Bottom, Top
            "sdk_image": list(self.sdk_image()),
            "frame_size": list(self.frame_size),
            "roi_size": list(self.roi_size),
            "ion_rois": [list(r) for r in self.ion_rois],
            "background_roi": list(self.background_roi) if self.background_roi else None,
            "grabber_rois": [list(r) for r in g],
            "grabber_x0": [r[0] for r in g],
            "grabber_y0": [r[1] for r in g],
            "grabber_x1": [r[2] for r in g],
            "grabber_y1": [r[3] for r in g],
            "n_rois": self.n_rois,
            "ion_mask": self.ion_mask,
            "full_mask": self.full_mask,
            "background_index": self.background_index,
            "roi_labels": self.roi_labels,
            "containment": [round(float(c), 4) for c in self.containment],
            "crosstalk": [round(float(c), 4) for c in self.crosstalk],
            "background_contamination": round(float(self.background_contamination), 4),
            "flip_x": self.flip_x,
            "flip_y": self.flip_y,
            "warnings": list(self.warnings),
            "manual_offset": list(self.manual_offset),
            "created": self.created,
            "verification": self.verification,
            "applied": self.applied,
            "binning_advice": self.binning_advice,
        }


def _frac(a, b, mu, sigma):
    """Fraction of N(mu, sigma) inside [a, b)."""
    return ndtr((b - mu) / sigma) - ndtr((a - mu) / sigma)


def _light_matrix(ax, bx, ay, by, pos, sx, sy, halo=(0.0, 0.0)) -> np.ndarray:
    """F[i, j] = fraction of ion j's light inside rectangle i (sensor coords);
    PSF: Gaussian core (sx, sy) plus a fraction ``halo[0]`` in an isotropic
    Gaussian halo of sigma ``halo[1]``."""
    ax, bx = np.asarray(ax)[:, None], np.asarray(bx)[:, None]
    ay, by = np.asarray(ay)[:, None], np.asarray(by)[:, None]
    out = _frac(ax, bx, pos[None, :, 0], sx) * _frac(ay, by, pos[None, :, 1], sy)
    f, sh = float(halo[0]), float(halo[1])
    if f > 0 and sh > 0:
        out = (1.0 - f) * out + f * (_frac(ax, bx, pos[None, :, 0], sh) *
                                     _frac(ay, by, pos[None, :, 1], sh))
    return out


AREA_PENALTY = 0.005       # score cost per binned pixel of ROI (background/noise)
CROSSTALK_WEIGHT = 2.0     # score cost per unit of foreign ion light in a ROI


def _roi_limits(sx, sy, hb, vb):
    """Largest ROI extent tried, in binned pixels: up to ~6 sigma of the PSF."""
    return (int(min(12, max(2, math.ceil(6.0 * sx / hb) + 1))),
            int(min(12, max(2, math.ceil(6.0 * sy / vb) + 1))))


def _ion_candidates(i, pos, sx, sy, px, py, hb, vb, bounds, max_w, max_h, fixed=None,
                    halo=(0.0, 0.0)):
    """All ROIs (superpixel rects inside ``bounds`` = (x0, y0, x1, y1), sensor
    px: the sensor or a fixed AQROI) containing ion i's superpixel, with scores."""
    x, y = pos[i]
    bx0, by0, bx1, by1 = bounds
    ci = int(math.floor((x + 0.5 - px) / hb))
    ri = int(math.floor((y + 0.5 - py) / vb))
    widths = [fixed[0]] if fixed else range(1, max_w + 1)
    heights = [fixed[1]] if fixed else range(1, max_h + 1)
    f, s_h = float(halo[0]), float(halo[1])
    use_halo = f > 0 and s_h > 0
    out = []
    others = np.delete(pos, i, axis=0)
    for w in widths:
        for a in range(ci - w + 1, ci + 1):
            x0 = px + a * hb
            if x0 < bx0 or x0 + w * hb > bx1:
                continue
            xa, xb = x0 - 0.5, x0 + w * hb - 0.5
            fx_i = float(_frac(xa, xb, x, sx))
            fx_o = _frac(xa, xb, others[:, 0], sx) if len(others) else None
            if use_halo:
                hx_i = float(_frac(xa, xb, x, s_h))
                hx_o = _frac(xa, xb, others[:, 0], s_h) if len(others) else None
            for h in heights:
                for r in range(ri - h + 1, ri + 1):
                    y0 = py + r * vb
                    if y0 < by0 or y0 + h * vb > by1:
                        continue
                    ya, yb = y0 - 0.5, y0 + h * vb - 0.5
                    cont = fx_i * float(_frac(ya, yb, y, sy))
                    if use_halo:
                        cont = (1.0 - f) * cont + f * hx_i * float(_frac(ya, yb, y, s_h))
                    xt = 0.0
                    if fx_o is not None:
                        lx = fx_o * _frac(ya, yb, others[:, 1], sy)
                        if use_halo:
                            lx = (1.0 - f) * lx + f * hx_o * _frac(ya, yb, others[:, 1], s_h)
                        xt = float(np.max(lx))
                    score = cont - CROSSTALK_WEIGHT * xt - AREA_PENALTY * w * h
                    out.append((score, a, r, w, h, cont, xt))
    return out


def _best_layout(pos, sx, sy, px, py, hb, vb, bounds, fixed=None, along_x=None,
                 halo=(0.0, 0.0)):
    """Per-ion ROIs for one grid phase: maximise the worst ion score, no overlaps.

    Ions are processed along the dominant chain axis; a bottleneck dynamic
    programme over the right edge of the previous ROI guarantees disjoint
    ROIs (exact for chains, also works for zig-zags with distinct columns)."""
    n = len(pos)
    max_w, max_h = _roi_limits(sx, sy, hb, vb)
    if along_x is None:
        along_x = bool(np.ptp(pos[:, 0]) >= np.ptp(pos[:, 1])) if n > 1 else True
    order = np.argsort(pos[:, 0] if along_x else pos[:, 1], kind="stable")
    cands = []
    for i in order:
        c = _ion_candidates(i, pos, sx, sy, px, py, hb, vb, bounds, max_w, max_h, fixed,
                            halo)
        if not c:
            return None
        cands.append(c)
    # DP: state = end coordinate (exclusive, in superpixels) along the chain axis
    # value = (worst score so far, sum of scores) ; back-pointers for the choice
    states = {-10 ** 9: ((float("inf"), 0.0), None)}
    history = []
    for c in cands:
        new = {}
        for (score, a, r, w, h, cont, xt) in c:
            start, end = (a, a + w) if along_x else (r, r + h)
            best_prev = None
            for e, (val, _) in states.items():
                if e <= start and (best_prev is None or val > states[best_prev][0]):
                    best_prev = e
            if best_prev is None:
                continue
            pv = states[best_prev][0]
            val = (min(pv[0], score), pv[1] + score)
            if end not in new or val > new[end][0]:
                new[end] = (val, (best_prev, (score, a, r, w, h, cont, xt)))
        if not new:
            return None
        history.append(new)
        states = {e: (v, None) for e, (v, _) in new.items()}
    # backtrack
    end = max(history[-1], key=lambda e: history[-1][e][0])
    worst = history[-1][end][0][0]
    picks = [None] * n
    for k in range(n - 1, -1, -1):
        prev, choice = history[k][end][1]
        picks[order[k]] = choice
        end = prev
    return worst, picks


def _independent_layout(pos, sx, sy, px, py, hb, vb, bounds, fixed=None, halo=(0.0, 0.0)):
    """Every ion its own best ROI for one grid phase, overlaps allowed (for
    ions closer than the binning separates, where no disjoint layout exists)."""
    max_w, max_h = _roi_limits(sx, sy, hb, vb)
    picks = []
    for i in range(len(pos)):
        c = _ion_candidates(i, pos, sx, sy, px, py, hb, vb, bounds, max_w, max_h, fixed, halo)
        if not c:
            return None
        picks.append(max(c))
    return min(p[0] for p in picks), picks


def _assess(pos, sx, sy, halo, aq, hb, vb, ion_rois, bg_rect):
    """Light fractions of ROIs given in frame coordinates of the AQROI ``aq``:
    ``(F, containment, crosstalk, background)`` with F[i, j] the fraction of
    ion j's light in ROI i, the worst foreign light per ROI and the ion light
    in the background ROI (in ions)."""
    r = np.asarray(ion_rois, float).reshape(-1, 4)
    x0, y0 = aq[0], aq[1]
    f = _light_matrix(x0 + r[:, 0] * hb - 0.5, x0 + r[:, 2] * hb - 0.5,
                      y0 + r[:, 1] * vb - 0.5, y0 + r[:, 3] * vb - 0.5, pos, sx, sy, halo)
    cont = np.diag(f).copy()
    xt = f.copy()
    np.fill_diagonal(xt, 0.0)
    crosstalk = xt.max(axis=1) if len(pos) > 1 else np.zeros(1)
    bg = 0.0
    if bg_rect is not None:
        u0, v0, u1, v1 = bg_rect
        bg = float(_light_matrix([x0 + u0 * hb - 0.5], [x0 + u1 * hb - 0.5],
                                 [y0 + v0 * vb - 0.5], [y0 + v1 * vb - 0.5],
                                 pos, sx, sy, halo).sum())
    return f, cont, crosstalk, bg


def _overlap_warnings(ion_rois, hb, vb) -> List[str]:
    out = []
    for i in range(len(ion_rois)):
        for j in range(i + 1, len(ion_rois)):
            a, b = tuple(ion_rois[i]), tuple(ion_rois[j])
            if a == b:
                out.append("ions {} and {} have the same ROI (closer than one {}x{} super-pixel "
                           "separates): their counts cannot be told apart - use less binning"
                           .format(i, j, hb, vb))
            elif _overlap(a, b):
                out.append("the ROIs of ions {} and {} overlap (the ions are closer than the {}x{} "
                           "binning separates): the shared pixels count for both - measure the "
                           "crosstalk (e.g. SPAM) before thresholding".format(i, j, hb, vb))
    return out


def _background_warning(bg: float) -> List[str]:
    if bg <= 0.02:
        return []
    return ["background ROI picks up {:.1%} of an ion's light; use a frame_margin of 2 for a "
            "cleaner background".format(bg)]


def plan_readout(positions: Sequence[Sequence[float]], sigma: Tuple[float, float],
                 sensor_shape: Tuple[int, int], hbin: int, vbin: int,
                 roi_size: Tuple[int, int] = (0, 0), frame_margin: int = 1,
                 min_containment: float = 0.6, max_crosstalk: float = 0.12,
                 max_rois: int = 16, background_roi: bool = True,
                 flip_x: bool = False, flip_y: bool = False, res_width: int = 12,
                 ion_states: Optional[Sequence[str]] = None,
                 halo: Tuple[float, float] = (0.0, 0.0),
                 aqroi: Optional[Sequence[int]] = None) -> ReadoutPlan:
    """Plan the AQROI and the per-ion Grabber ROIs (see module docstring).

    ``roi_size = (0, 0)`` lets every ion get its own optimal ROI size (an ion
    centred in a super-pixel needs one, an ion on a boundary two); a fixed
    ``(ru, rv)`` forces the same size for all.  ``halo = (fraction, sigma)``
    adds the PSF's broad wings (sensor px) to the containment/crosstalk.
    ``aqroi = (x0, y0, x1, y1)`` (sensor px, half-open, a multiple of the
    binning) fixes the AQROI: the ROIs are placed inside it, on its grid.

    Ions too close for disjoint ROIs still get a plan: every ion its own best
    ROI, overlapping ROIs allowed (with a warning)."""
    pos = np.asarray(positions, float).reshape(-1, 2)
    n = len(pos)
    if n == 0:
        raise PlanError("no ions to plan a readout for")
    if n > max_rois:
        raise PlanError("{} ions but the Grabber has only {} ROI engines".format(n, max_rois))
    notes: List[str] = []
    if background_roi and n + 1 > max_rois:
        background_roi = False
        notes.append("no ROI engine left for a background ROI")
    sx, sy = (float(sigma[0]), float(sigma[1]))
    sh, sw = sensor_shape
    fixed = (int(roi_size[0]), int(roi_size[1])) if tuple(roi_size) != (0, 0) else None
    if aqroi is not None:
        aq_fixed = tuple(int(v) for v in aqroi)
        ax0, ay0, ax1, ay1 = aq_fixed
        if not (0 <= ax0 < ax1 <= sw and 0 <= ay0 < ay1 <= sh):
            raise PlanError("AQROI {} is not on the {}x{} sensor".format(list(aq_fixed), sw, sh))
        if (ax1 - ax0) % hbin or (ay1 - ay0) % vbin:
            raise PlanError("AQROI of {}x{} px is not a multiple of the {}x{} binning".format(
                ax1 - ax0, ay1 - ay0, hbin, vbin))
        bounds, where = aq_fixed, "AQROI"
        phases = [(ax0 % hbin, ay0 % vbin)]
    else:
        aq_fixed = None
        bounds, where = (0, 0, sw, sh), "sensor"
        phases = [(px, py) for px in range(hbin) for py in range(vbin)]
    outside = [str(i) for i, (x, y) in enumerate(pos)
               if not (bounds[0] - 0.5 <= x < bounds[2] - 0.5 and
                       bounds[1] - 0.5 <= y < bounds[3] - 0.5)]
    if outside:
        raise PlanError("ion{} {} outside the {}".format("s" * (len(outside) > 1),
                                                         ", ".join(outside), where))

    best = None
    dominant = bool(np.ptp(pos[:, 0]) >= np.ptp(pos[:, 1])) if n > 1 else True
    # partition along the chain axis first; for 2D crystals also try the other axis
    layouts = []
    for along_x in (dominant, not dominant):
        for px, py in phases:
            res = _best_layout(pos, sx, sy, px, py, hbin, vbin, bounds, fixed, along_x, halo)
            if res is not None:
                layouts.append((px, py, res))
        if layouts:
            break
    if not layouts:
        # No disjoint ROIs: ions closer than the binning (or a forced ROI size)
        # separates.  Every ion still gets its own best ROI - the Grabber sums
        # overlapping ROIs independently - and the crosstalk is reported, to be
        # measured (e.g. with a SPAM experiment) before thresholding.
        for px, py in phases:
            res = _independent_layout(pos, sx, sy, px, py, hbin, vbin, bounds, fixed, halo)
            if res is not None:
                layouts.append((px, py, res))
    for px, py, (worst, picks) in layouts:
        xs0 = np.array([px + p[1] * hbin for p in picks])
        ys0 = np.array([py + p[2] * vbin for p in picks])
        xs1 = xs0 + np.array([p[3] for p in picks]) * hbin
        ys1 = ys0 + np.array([p[4] for p in picks]) * vbin
        f = _light_matrix(xs0 - 0.5, xs1 - 0.5, ys0 - 0.5, ys1 - 0.5, pos, sx, sy, halo)
        cont = np.diag(f).copy()
        xt = f.copy()
        np.fill_diagonal(xt, 0.0)
        worst_x = float(xt.max()) if n > 1 else 0.0
        ok = cont.min() >= min_containment and worst_x <= max_crosstalk
        key = (int(ok), round(worst, 6), -sum(p[3] * p[4] for p in picks))
        if best is None or key > best[0]:
            best = (key, px, py, picks, f, cont, xt)
    if best is None:
        raise PlanError("no {}x{} ROI fits around the ions inside the {} (ions at its "
                        "edge?)".format(hbin, vbin, where))
    key, px, py, picks, f, cont, xt = best
    geometry: List[str] = []
    if not key[0]:
        geometry.append("no placement reaches containment >= {:.0%} with crosstalk <= {:.0%}; "
                        "best worst-ion containment {:.1%}, worst crosstalk {:.1%} - consider "
                        "less binning (see binning_advice) or a larger ion spacing".format(
                            min_containment, max_crosstalk, cont.min(),
                            float(xt.max()) if n > 1 else 0.0))
    m0 = np.array([p[1] for p in picks])
    n0 = np.array([p[2] for p in picks])
    wid = np.array([p[3] for p in picks])
    hgt = np.array([p[4] for p in picks])
    sizes = sorted(set(zip(wid.tolist(), hgt.tolist())))

    if aq_fixed is not None:
        aq = aq_fixed
        u_min, u_max = (aq[0] - px) // hbin, (aq[2] - px) // hbin
        v_min, v_max = (aq[1] - py) // vbin, (aq[3] - py) // vbin
    else:
        # -- AQROI: bounding box of the ROIs + symmetric margin, kept on the sensor
        u_lo, u_hi = int(m0.min()), int((m0 + wid).max())
        v_lo, v_hi = int(n0.min()), int((n0 + hgt).max())
        mx = my = int(frame_margin)
        if background_roi and frame_margin < 1:
            my = 1
        left = min(mx, (px + u_lo * hbin) // hbin)
        right = min(mx, (sw - (px + u_hi * hbin)) // hbin)
        bottom = min(my, (py + v_lo * vbin) // vbin)
        top = min(my, (sh - (py + v_hi * vbin)) // vbin)
        if min(left, right) < mx or min(bottom, top) < my:
            notes.append("crystal close to the sensor edge: AQROI margin reduced")
        u_min, u_max = u_lo - left, u_hi + right
        v_min, v_max = v_lo - bottom, v_hi + top
        aq = (px + u_min * hbin, py + v_min * vbin, px + u_max * hbin, py + v_max * vbin)
    fw, fh = u_max - u_min, v_max - v_min
    ion_rois = [(int(m0[i] - u_min), int(n0[i] - v_min), int(m0[i] - u_min + wid[i]),
                 int(n0[i] - v_min + hgt[i])) for i in range(n)]

    # -- background ROI: a margin strip with the least ion light ---------------
    bg_rect: Optional[Rect] = None
    bg_cont = 0.0
    if background_roi:
        cu0 = min(r[0] for r in ion_rois)
        cu1 = max(r[2] for r in ion_rois)
        cands = []
        for v in [0, fh - 1]:
            rect = (cu0, v, cu1, v + 1)
            if any(_overlap(rect, r) for r in ion_rois):
                continue
            a_x = aq[0] + rect[0] * hbin - 0.5
            b_x = aq[0] + rect[2] * hbin - 0.5
            a_y = aq[1] + rect[1] * vbin - 0.5
            b_y = aq[1] + rect[3] * vbin - 0.5
            c = float(_light_matrix([a_x], [b_x], [a_y], [b_y], pos, sx, sy, halo).sum())
            cands.append((c, rect))
        if cands:
            bg_cont, bg_rect = min(cands)
            geometry += _background_warning(bg_cont)
        else:
            notes.append("no room for a background ROI")

    rois = ion_rois + ([bg_rect] if bg_rect else [])
    grabber = [map_rect(r, fw, fh, flip_x, flip_y) for r in rois]
    lim = (1 << res_width) - 1
    for r in grabber:
        problems = validate_roi(r, fw, fh, lim)
        if problems:
            raise PlanError("internal error, invalid Grabber ROI {}: {}".format(r, problems))
    crosstalk = xt.max(axis=1).tolist() if n > 1 else [0.0]
    geometry = _overlap_warnings(ion_rois, hbin, vbin) + geometry
    return ReadoutPlan(
        n_ions=n, positions=[(float(x), float(y)) for x, y in pos],
        ion_states=list(ion_states) if ion_states is not None else ["bright"] * n,
        psf_sigma=(sx, sy), hbin=hbin, vbin=vbin, roi=tuple(int(v) for v in aq),
        frame_size=(fw, fh), roi_size=sizes[0] if len(sizes) == 1 else (0, 0),
        ion_rois=ion_rois,
        background_roi=bg_rect, grabber_rois=grabber,
        containment=[float(c) for c in cont], crosstalk=[float(c) for c in crosstalk],
        crosstalk_matrix=[[float(v) for v in row] for row in f],
        background_contamination=bg_cont, flip_x=flip_x, flip_y=flip_y,
        psf_halo=(float(halo[0]), float(halo[1])), warnings=notes + geometry, notes=notes,
        targets=(float(min_containment), float(max_crosstalk)))


def reposition_plan(plan: ReadoutPlan, dx: int, dy: int,
                    sensor_shape: Tuple[int, int]) -> ReadoutPlan:
    """``plan`` with its AQROI moved by ``(dx, dy)`` sensor px.

    The Grabber ROIs are frame coordinates and move with the AQROI, so the
    ions shift by ``(-dx, -dy)`` relative to their ROIs - with binning by a
    fraction of a super-pixel: the manual counterpart of the planner's
    grid-phase search.  Containment and crosstalk are re-computed for the
    located ion positions; the Grabber ROIs themselves do not change."""
    sh, sw = sensor_shape
    x0, y0, x1, y1 = plan.roi
    dx, dy = int(dx), int(dy)
    aq = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
    if aq[0] < 0 or aq[1] < 0 or aq[2] > sw or aq[3] > sh:
        raise PlanError("the AQROI {} would leave the {}x{} sensor".format(list(aq), sw, sh))
    pos = np.asarray(plan.positions, float).reshape(-1, 2)
    sx, sy = plan.psf_sigma
    f, cont, crosstalk, bg = _assess(pos, sx, sy, plan.psf_halo, aq, plan.hbin, plan.vbin,
                                     plan.ion_rois, plan.background_roi)
    offset = (plan.manual_offset[0] + dx, plan.manual_offset[1] + dy)
    geometry = _overlap_warnings(plan.ion_rois, plan.hbin, plan.vbin)
    min_cont, max_xt = plan.targets
    for i in range(plan.n_ions):
        if cont[i] < min_cont:
            geometry.append("ion {}: only {:.1%} of its light in its ROI (target {:.0%})".format(
                i, cont[i], min_cont))
        if crosstalk[i] > max_xt:
            geometry.append("ROI {}: {:.1%} of a neighbour's light (target <= {:.0%})".format(
                i, crosstalk[i], max_xt))
    geometry += _background_warning(bg)
    if offset != (0, 0):
        geometry.append("AQROI moved by hand by ({:+d}, {:+d}) px from the computed plan".format(
            *offset))
    return dataclasses.replace(
        plan, roi=aq, containment=[float(c) for c in cont],
        crosstalk=[float(c) for c in crosstalk],
        crosstalk_matrix=[[float(v) for v in row] for row in f],
        background_contamination=bg, warnings=list(plan.notes) + geometry,
        notes=list(plan.notes), manual_offset=offset, verification=None, applied=None,
        created=time.time())


def _overlap(a: Rect, b: Rect) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def map_rect(r: Rect, w: int, h: int, flip_x: bool, flip_y: bool) -> Rect:
    """SDK frame rectangle -> Grabber rectangle for a mirrored Camera Link stream."""
    u0, v0, u1, v1 = r
    if flip_x:
        u0, u1 = w - u1, w - u0
    if flip_y:
        v0, v1 = h - v1, h - v0
    return int(u0), int(v0), int(u1), int(v1)


def validate_roi(r: Rect, w: int, h: int, limit: int = 4095) -> List[str]:
    """Problems with a Grabber ROI for a ``w x h`` frame (empty list = fine)."""
    x0, y0, x1, y1 = r
    problems = []
    if min(x0, y0) < 0:
        problems.append("negative coordinate")
    if x0 >= x1 or y0 >= y1:
        problems.append("empty ROI (x0 >= x1 or y0 >= y1)")
    if x1 > w:
        problems.append("x1 = {} > frame width {}: the gateware never clears its x flag and "
                        "would sum whole rows".format(x1, w))
    if y1 > h:
        problems.append("y1 = {} > frame height {}".format(y1, h))
    if max(x0, y0, x1, y1) > limit:
        problems.append("coordinate above {}".format(limit))
    return problems


def grabber_roi_sum_exact(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                          shift: int = 0) -> int:
    """Cycle-faithful model of one ARTIQ Grabber ROI engine for one frame.

    Mirrors ``ROI`` in ``artiq/gateware/grabber/core.py``: the in-ROI flags are
    registers updated when the pixel counters *equal* x0/x1 (y0/y1) and reset
    only at end-of-frame; during the first horizontal-blanking cycle after a
    line the x counter holds W, afterwards 0 and the y counter is incremented."""
    h, w = frame.shape
    x_good = y_good = False

    def cmp(px, py):
        nonlocal x_good, y_good
        if py == y0:
            y_good = True
        if py == y1:
            y_good = False
        if px == x0:
            x_good = True
        if px == x1:
            x_good = False

    total = 0
    cmp(0, 0)                         # vertical blanking before the first line
    for y in range(h):
        for x in range(w):
            cmp(x, y)
            if x_good and y_good:
                total += int(frame[y, x]) >> shift
        cmp(w, y)                     # first blanking cycle: x counter still at W
        cmp(0, y + 1)                 # rest of the blanking
    return total


def grabber_roi_sums(frame: np.ndarray, rois: Sequence[Rect]) -> np.ndarray:
    """Fast per-ROI sums for *valid* ROIs (what the Grabber returns)."""
    f = np.asarray(frame, np.int64)
    return np.array([f[y0:y1, x0:x1].sum() for (x0, y0, x1, y1) in rois], np.int64)


# ---------------------------------------------------------------------------
# verification and thresholds
# ---------------------------------------------------------------------------

def verify_plan(plan: ReadoutPlan, frames: Sequence[np.ndarray], min_snr: float = 3.0) -> dict:
    """Check, on frames taken with the final readout settings (SDK orientation),
    that every ion is really inside its ROI."""
    f = np.asarray(frames)
    problems: List[str] = []
    w, h = plan.frame_size
    if f.ndim != 3 or f.shape[1:] != (h, w):
        problems.append("frame shape {} differs from the planned {}x{}".format(
            tuple(f.shape[1:]), w, h))
        return {"ok": False, "problems": problems}
    sums = plan.roi_sums(f).astype(float)
    n = plan.n_ions
    area = [(r[2] - r[0]) * (r[3] - r[1]) for r in plan.sdk_rois()]
    if plan.background_roi is not None:
        bg = sums[:, n] / area[n]                         # per binned pixel
        bg_level = float(bg.mean())
        bg_noise = float(bg.std(ddof=1)) if len(bg) > 1 else 0.0
    else:
        # border pixels of the frame as background estimate
        border = np.concatenate([f[:, 0, :].ravel(), f[:, -1, :].ravel()]).astype(float)
        bg_level, bg_noise = float(np.median(border)), float(border.std())
    ions = []
    nets = []
    n_f = len(sums)
    for i in range(n):
        s = sums[:, i]
        net = s - area[i] * bg_level
        std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        noise = max(std, math.sqrt(area[i]) * max(bg_noise, 1.0))
        shot = float(net.mean()) / noise if noise > 0 else float("inf")
        # is the ion in its ROI?  judged on the mean of all frames: with ~10-20
        # photons per shot a single shot is legitimately noisy
        snr = shot * math.sqrt(n_f)
        # rough single-shot bright/dark error: dark = background scaled to the ROI
        dark_sd = math.sqrt(area[i]) * max(bg_noise, 1e-9)
        err = _gauss_overlap(float(net.mean()), std, 0.0, dark_sd)
        nets.append(float(net.mean()))
        ions.append({"mean": round(float(s.mean()), 1), "std": round(std, 1),
                     "net": round(float(net.mean()), 1), "snr": round(snr, 1),
                     "snr_single_shot": round(shot, 2),
                     "discrimination_error": round(err, 4)})
        if plan.ion_states[i] == "bright" and snr < min_snr:
            problems.append("ion {}: SNR {:.1f} < {} - not in its ROI?".format(i, snr, min_snr))
    bright = [nets[i] for i in range(n) if plan.ion_states[i] == "bright"]
    if len(bright) >= 2:
        med = float(np.median(bright))
        for i in range(n):
            if plan.ion_states[i] == "bright" and med > 0 and nets[i] < 0.4 * med:
                problems.append("ion {} gives only {:.0%} of the median ion signal".format(
                    i, nets[i] / med))
    return {"ok": not problems, "problems": problems, "n_frames": int(f.shape[0]),
            "background_per_pixel": round(bg_level, 2), "background_noise": round(bg_noise, 2),
            "ions": ions, "max_count": int(f.max())}


def _gauss_overlap(mu_b: float, sd_b: float, mu_d: float, sd_d: float) -> float:
    """Mean error of the best threshold between two Gaussians (bright, dark)."""
    if mu_b <= mu_d:
        return 0.5
    sd_b, sd_d = max(sd_b, 1e-9), max(sd_d, 1e-9)
    xs = np.linspace(mu_d, mu_b, 201)
    err = 0.5 * (ndtr((mu_d - xs) / sd_d) + ndtr((xs - mu_b) / sd_b))
    return float(err.min())


def fit_threshold(counts: Sequence[float]) -> dict:
    """Bright/dark discrimination threshold from a histogram of ROI sums.

    Otsu initialisation + two-Gaussian EM; the threshold minimises the total
    misclassification probability.  Works on raw Grabber sums (bias included)."""
    x = np.asarray(counts, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 20:
        return {"ok": False, "reason": "need at least 20 shots", "n": int(n)}
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return {"ok": False, "reason": "all values equal", "n": int(n), "mean": lo}
    hist, edges = np.histogram(x, bins=int(min(256, max(16, n // 8))))
    centers = 0.5 * (edges[1:] + edges[:-1])
    p = hist / hist.sum()
    w0 = np.cumsum(p)
    m0 = np.cumsum(p * centers)
    mt = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mt * w0 - m0) ** 2 / (w0 * (1.0 - w0))
    between[~np.isfinite(between)] = -1
    t = float(centers[int(np.argmax(between))])
    a, b = x[x <= t], x[x > t]
    if a.size < 3 or b.size < 3:
        return {"ok": False, "reason": "only one population (all bright or all dark)",
                "n": int(n), "mean": float(x.mean()), "std": float(x.std())}
    wa, ma, sa, mb, sb = a.size / n, a.mean(), a.std() + 1e-9, b.mean(), b.std() + 1e-9
    for _ in range(200):
        pa = wa * np.exp(-0.5 * ((x - ma) / sa) ** 2) / sa
        pb = (1 - wa) * np.exp(-0.5 * ((x - mb) / sb) ** 2) / sb
        r = pa / np.maximum(pa + pb, 1e-300)
        wa_new = float(r.mean())
        if wa_new <= 1e-6 or wa_new >= 1 - 1e-6:
            break
        ma_new = float((r * x).sum() / r.sum())
        mb_new = float(((1 - r) * x).sum() / (1 - r).sum())
        sa = float(np.sqrt((r * (x - ma_new) ** 2).sum() / r.sum())) + 1e-9
        sb = float(np.sqrt(((1 - r) * (x - mb_new) ** 2).sum() / (1 - r).sum())) + 1e-9
        done = abs(ma_new - ma) < 1e-6 * (abs(ma) + 1) and abs(mb_new - mb) < 1e-6 * (abs(mb) + 1)
        wa, ma, mb = wa_new, ma_new, mb_new
        if done:
            break
    if ma > mb:
        wa, ma, sa, mb, sb = 1 - wa, mb, sb, ma, sa
    separation = float((mb - ma) / math.sqrt(sa * sa + sb * sb))
    if separation < 2.0 or min(wa, 1 - wa) < 0.01:
        return {"ok": False, "reason": "no clear bright/dark separation (one population?)",
                "n": int(n), "mean": float(x.mean()), "std": float(x.std()),
                "separation": separation}
    grid = np.linspace(ma, mb, 2001)
    err = wa * (1 - ndtr((grid - ma) / sa)) + (1 - wa) * ndtr((grid - mb) / sb)
    k = int(np.argmin(err))
    thr = float(grid[k])
    p_dark_as_bright = float(1 - ndtr((thr - ma) / sa))
    p_bright_as_dark = float(ndtr((thr - mb) / sb))
    return {"ok": True, "threshold": thr, "dark_mean": float(ma), "dark_std": float(sa),
            "bright_mean": float(mb), "bright_std": float(sb), "bright_fraction": float(1 - wa),
            "fidelity": 1.0 - 0.5 * (p_dark_as_bright + p_bright_as_dark),
            "p_dark_as_bright": p_dark_as_bright, "p_bright_as_dark": p_bright_as_dark,
            "separation": float((mb - ma) / math.sqrt(sa * sa + sb * sb)), "n": int(n)}


def compare_grabber_frame(expected: np.ndarray, measured: np.ndarray) -> dict:
    """Find how a Grabber-reconstructed frame is oriented relative to the SDK frame.

    ``measured`` is built in an ARTIQ experiment by reading single-pixel ROIs.
    Returns the best of identity / flip_x / flip_y / rot180 by normalised
    correlation, with the suggested ``grabber.flip_x``/``flip_y`` settings."""
    e = np.asarray(expected, float)
    m = np.asarray(measured, float)
    if e.shape != m.shape:
        return {"ok": False, "reason": "shape {} vs expected {}".format(m.shape, e.shape)}
    cands = {"identity": (False, False), "flip_x": (True, False),
             "flip_y": (False, True), "rot180": (True, True)}
    scores = {}
    for name, (fx, fy) in cands.items():
        t = e[:, ::-1] if fx else e
        t = t[::-1, :] if fy else t
        a = t - t.mean()
        b = m - m.mean()
        den = math.sqrt(float((a * a).sum() * (b * b).sum()))
        scores[name] = float((a * b).sum() / den) if den > 0 else 0.0
    best = max(scores, key=scores.get)
    ranked = sorted(scores.values(), reverse=True)
    margin = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]
    return {"ok": ranked[0] > 0.5 and margin > 0.05, "best": best,
            "flip_x": cands[best][0], "flip_y": cands[best][1],
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "ambiguous": margin <= 0.05}


def binning_advice(positions: Sequence[Sequence[float]], sigma: Tuple[float, float],
                   sensor_shape: Tuple[int, int],
                   candidates: Sequence[Tuple[int, int]] = ((1, 1), (2, 2), (3, 3), (4, 4),
                                                            (2, 4), (2, 8), (1, 4), (4, 8)),
                   **kw) -> List[dict]:
    """Quality of the readout layout for several binnings (to help choose one)."""
    out = []
    for hb, vb in candidates:
        try:
            p = plan_readout(positions, sigma, sensor_shape, hb, vb, **kw)
        except PlanError as e:
            out.append({"binning": [hb, vb], "ok": False, "reason": str(e)})
            continue
        xt = max(p.crosstalk) if p.crosstalk else 0.0
        out.append({"binning": [hb, vb], "ok": not any("no placement" in w for w in p.warnings),
                    "min_containment": round(min(p.containment), 4),
                    "max_crosstalk": round(xt, 4), "frame_size": list(p.frame_size),
                    "pixels": p.frame_size[0] * p.frame_size[1]})
    return out
