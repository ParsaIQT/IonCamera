"""Grabber readout planning: AQROI + per-ion Grabber ROIs, verification, thresholds.

Geometry
--------
The camera reads the AQROI ``[x0, x1) x [y0, y1)`` (sensor px) with on-chip
binning ``hbin x vbin``; the resulting ``W x H`` frame is what the ARTIQ
Grabber receives over Camera Link.  Each ion gets its own rectangular ROI of
binned pixels.  By default every ion gets one super-pixel (``roi_size = (1,
1)``): each super-pixel adds its own dark counts (clock-induced charge,
scatter, read noise) to the sum, so the fewest pixels around the brightest
part of the ion image usually discriminate bright from dark best.  Where the
ion sits relative to the binning grid matters: an ion on a super-pixel
boundary splits its light.  The grid *phase* is set by the AQROI origin, so
the planner tries every phase (``hbin x vbin`` choices) - which centres the
ions in their super-pixels as well as their spacing allows - and with
``roi_size = (0, 0)`` every ROI size, computing for each ion the exact
fraction of its fitted PSF (Gaussian core plus, if measured, the broad halo
of aberration wings that dominates the crosstalk of real imaging systems)
inside its own ROI (*containment*) and inside every other ROI (*crosstalk*),
and keeps the placement with the best worst-case containment (smallest ROIs
first among those meeting ``min_containment``).  The AQROI is
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
* coordinates are 0-based pixels of the Camera Link stream: x counts the
  stream's valid pixels (one 16-bit pixel per clock) in a line, y the lines
  of a frame.  Andor rotate/flip happen in host software only and never
  reach the stream.  How the camera's binned frame sits in the stream -
  stream pixels per binned pixel, pixels before the image, the direction -
  is a :class:`GrabberGeometry`: the identity (one pixel per binned pixel)
  is what an iXon Ultra 897 sends; other cameras are measured with the
  Grabber mapping check (:func:`infer_grabber_geometry`);
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
    aqroi_by_hand: bool = False                       # AQROI set by hand, ROIs placed in it
    positions_t: float = 0.0          # when the ion positions were measured (0: at ``created``)
    grabber: dict = field(default_factory=dict)       # GrabberGeometry the ROIs were mapped with
    grabber_frame: Tuple[int, int] = (0, 0)           # the Grabber frame (W, H) they assume
    grabber_source: str = ""                          # where that geometry comes from
    grabber_counts: Tuple[float, float] = (1.0, 0.0)  # Grabber sum = gain x USB sum + offset x px

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

    def to_grabber_counts(self, value: float, roi: int) -> float:
        """An ROI sum of the USB frames (e.g. a threshold) as the Grabber
        counts it for ROI ``roi``: a stream that is not binned reads every
        sensor pixel with its own bias (``grabber_counts``)."""
        u0, v0, u1, v1 = self.sdk_rois()[roi]
        gain, offset = self.grabber_counts
        return float(gain) * float(value) + float(offset) * (u1 - u0) * (v1 - v0)

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
            "grabber_geometry": GrabberGeometry.from_dict(self.grabber).to_dict(),
            "grabber_frame": list(self.grabber_frame) if any(self.grabber_frame)
            else list(self.frame_size),
            "grabber_source": self.grabber_source,
            "grabber_counts": [round(float(v), 4) for v in self.grabber_counts],
            "grabber_summary": GrabberGeometry.from_dict(self.grabber).describe(self.hbin,
                                                                                self.vbin),
            "warnings": list(self.warnings),
            "manual_offset": list(self.manual_offset),
            "aqroi_by_hand": self.aqroi_by_hand,
            "created": self.created,
            "positions_t": self.positions_t or self.created,
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


def _intervals(c, sizes, origin, binning, lo, hi):
    """Super-pixel intervals ``[a, a + s)`` holding super-pixel ``c`` (``s`` in
    ``sizes``, ``a`` from ``c - s + 1`` to ``c``) whose sensor extent lies
    inside ``[lo, hi)``, ordered by size, then start: ``(starts, sizes)``."""
    starts, lens = [], []
    for s in sizes:
        for a in range(c - s + 1, c + 1):
            x0 = origin + a * binning
            if lo <= x0 and x0 + s * binning <= hi:
                starts.append(a)
                lens.append(s)
    return np.array(starts, dtype=np.int64), np.array(lens, dtype=np.int64)


class _Candidates:
    """The candidate ROIs of one ion, as arrays in the planner's order (width,
    left, height, bottom): ``score, a, r, w, h, cont, xt``."""
    __slots__ = ("score", "a", "r", "w", "h", "cont", "xt")

    def __init__(self, score, a, r, w, h, cont, xt):
        self.score, self.a, self.r, self.w, self.h = score, a, r, w, h
        self.cont, self.xt = cont, xt

    def __len__(self):
        return len(self.score)

    def pick(self, k) -> tuple:
        """Candidate ``k`` as ``(score, a, r, w, h, cont, xt)`` (plain Python numbers)."""
        return (float(self.score[k]), int(self.a[k]), int(self.r[k]), int(self.w[k]),
                int(self.h[k]), float(self.cont[k]), float(self.xt[k]))


def _ion_candidates(i, pos, sx, sy, px, py, hb, vb, bounds, max_w, max_h, fixed=None,
                    halo=(0.0, 0.0)) -> _Candidates:
    """All ROIs (superpixel rects inside ``bounds`` = (x0, y0, x1, y1), sensor
    px: the sensor or a fixed AQROI) containing ion i's superpixel, with scores:
    the light of ion i inside (containment), the most light of any other ion
    inside (crosstalk) - computed for all of them at once, as separable
    products of the 1D fractions of the PSF core (and halo)."""
    x, y = pos[i]
    bx0, by0, bx1, by1 = bounds
    ci = int(math.floor((x + 0.5 - px) / hb))
    ri = int(math.floor((y + 0.5 - py) / vb))
    widths = [fixed[0]] if fixed else range(1, max_w + 1)
    heights = [fixed[1]] if fixed else range(1, max_h + 1)
    a, w = _intervals(ci, widths, px, hb, bx0, bx1)
    r, h = _intervals(ri, heights, py, vb, by0, by1)
    kx, ky = len(a), len(r)
    if kx == 0 or ky == 0:
        e = np.zeros(0)
        return _Candidates(e, e.astype(np.int64), e.astype(np.int64), e.astype(np.int64),
                           e.astype(np.int64), e, e)
    f, s_h = float(halo[0]), float(halo[1])
    use_halo = f > 0 and s_h > 0
    others = np.delete(pos, i, axis=0)
    x0 = px + a * hb
    xa, xb = x0 - 0.5, x0 + w * hb - 0.5
    y0 = py + r * vb
    ya, yb = y0 - 0.5, y0 + h * vb - 0.5
    cont = _frac(xa, xb, x, sx)[:, None] * _frac(ya, yb, y, sy)[None, :]
    if use_halo:
        cont = (1.0 - f) * cont + f * _frac(xa, xb, x, s_h)[:, None] * \
            _frac(ya, yb, y, s_h)[None, :]
    if len(others):
        lx = _frac(xa[:, None], xb[:, None], others[None, :, 0], sx)[:, None, :] * \
            _frac(ya[:, None], yb[:, None], others[None, :, 1], sy)[None, :, :]
        if use_halo:
            lx = (1.0 - f) * lx + \
                f * _frac(xa[:, None], xb[:, None], others[None, :, 0], s_h)[:, None, :] * \
                _frac(ya[:, None], yb[:, None], others[None, :, 1], s_h)[None, :, :]
        xt = lx.max(axis=2)
    else:
        xt = np.zeros((kx, ky))
    score = cont - CROSSTALK_WEIGHT * xt - AREA_PENALTY * w[:, None] * h[None, :]
    return _Candidates(score.ravel(), np.repeat(a, ky), np.tile(r, kx), np.repeat(w, ky),
                       np.tile(h, kx), cont.ravel(), xt.ravel())


def _all_candidates(pos, sx, sy, px, py, hb, vb, bounds, fixed=None, halo=(0.0, 0.0)):
    """``_ion_candidates`` of every ion for the grid phase ``(px, py)``."""
    max_w, max_h = _roi_limits(sx, sy, hb, vb)
    return [_ion_candidates(i, pos, sx, sy, px, py, hb, vb, bounds, max_w, max_h, fixed, halo)
            for i in range(len(pos))]


def _best_layout(pos, sx, sy, px, py, hb, vb, bounds, fixed=None, along_x=None,
                 halo=(0.0, 0.0), cands=None):
    """Per-ion ROIs for one grid phase: maximise the worst ion score, no overlaps.

    Ions are processed along the dominant chain axis; a bottleneck dynamic
    programme over the right edge of the previous ROI guarantees disjoint
    ROIs (exact for chains, also works for zig-zags with distinct columns).
    ``cands``: the ions' candidates (:func:`_all_candidates`), if known."""
    n = len(pos)
    if cands is None:
        cands = _all_candidates(pos, sx, sy, px, py, hb, vb, bounds, fixed, halo)
    if along_x is None:
        along_x = bool(np.ptp(pos[:, 0]) >= np.ptp(pos[:, 1])) if n > 1 else True
    order = np.argsort(pos[:, 0] if along_x else pos[:, 1], kind="stable")
    if any(len(cands[i]) == 0 for i in order):
        return None
    # DP: state = end coordinate (exclusive, in superpixels) along the chain axis
    # value = (worst score so far, sum of scores) ; back-pointers for the choice.
    # (dicts keep the insertion order: ties go to the first state / candidate)
    states = {-10 ** 9: (float("inf"), 0.0)}
    history = []
    for i in order:
        c = cands[i]
        start = c.a if along_x else c.r
        end = start + (c.w if along_x else c.h)
        # best previous state for every start: the first of the best values
        # among the states ending at or before it
        prev = {}
        for s in np.unique(start).tolist():
            best = None
            for e, val in states.items():
                if e <= s and (best is None or val > states[best]):
                    best = e
            prev[s] = best
        ok = np.array([prev[s] is not None for s in start.tolist()], bool)
        if not ok.any():
            return None
        idx = np.nonzero(ok)[0]
        bp = [prev[s] for s in start[idx].tolist()]
        p0 = np.array([states[e][0] for e in bp])
        p1 = np.array([states[e][1] for e in bp])
        sc = c.score[idx]
        v0 = np.minimum(p0, sc)
        v1 = p1 + sc
        ends = end[idx]
        new = {}
        # ends in the order they first appear; per end the first candidate with
        # the lexicographically largest (worst, sum)
        _, first = np.unique(ends, return_index=True)
        for k0 in np.sort(first).tolist():
            e = int(ends[k0])
            sel = np.nonzero(ends == e)[0]
            sel = sel[v0[sel] == v0[sel].max()]
            k = int(sel[np.argmax(v1[sel] == v1[sel].max())])
            new[e] = ((float(v0[k]), float(v1[k])), (bp[k], c.pick(idx[k])))
        history.append(new)
        states = {e: v for e, (v, _) in new.items()}
    # backtrack
    end = max(history[-1], key=lambda e: history[-1][e][0])
    worst = history[-1][end][0][0]
    picks = [None] * n
    for k in range(n - 1, -1, -1):
        prev_end, choice = history[k][end][1]
        picks[order[k]] = choice
        end = prev_end
    return worst, picks


def _independent_layout(pos, sx, sy, px, py, hb, vb, bounds, fixed=None, halo=(0.0, 0.0),
                        cands=None):
    """Every ion its own best ROI for one grid phase, overlaps allowed (for
    ions closer than the binning separates, where no disjoint layout exists)."""
    if cands is None:
        cands = _all_candidates(pos, sx, sy, px, py, hb, vb, bounds, fixed, halo)
    picks = []
    for c in cands:
        if len(c) == 0:
            return None
        top = np.nonzero(c.score == c.score.max())[0]
        picks.append(max(c.pick(k) for k in top))
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
                 aqroi: Optional[Sequence[int]] = None,
                 grabber: Optional["GrabberGeometry"] = None) -> ReadoutPlan:
    """Plan the AQROI and the per-ion Grabber ROIs (see module docstring).

    ``roi_size = (0, 0)`` lets every ion get its own optimal ROI size (an ion
    centred in a super-pixel needs one, an ion on a boundary two); a fixed
    ``(ru, rv)`` forces the same size for all.  ``halo = (fraction, sigma)``
    adds the PSF's broad wings (sensor px) to the containment/crosstalk.
    ``aqroi = (x0, y0, x1, y1)`` (sensor px, half-open, a multiple of the
    binning) fixes the AQROI: the ROIs are placed inside it, on its grid.
    ``grabber``: how the frame sits in the Camera Link stream (default: one
    stream pixel per binned pixel, mirrored as ``flip_x`` / ``flip_y`` say).

    Ions too close for disjoint ROIs still get a plan: every ion its own best
    ROI, overlapping ROIs allowed (with a warning)."""
    pos = np.asarray(positions, float).reshape(-1, 2)
    n = len(pos)
    if n == 0:
        raise PlanError("no ions to plan a readout for")
    notes: List[str] = []
    if aqroi is not None:
        # an AQROI set by hand reads out the ions inside it; spots outside
        # (other ions, stray light taken for ions) are left out, not an error
        ax0, ay0, ax1, ay1 = (int(v) for v in aqroi)
        inside = ((pos[:, 0] >= ax0 - 0.5) & (pos[:, 0] < ax1 - 0.5) &
                  (pos[:, 1] >= ay0 - 0.5) & (pos[:, 1] < ay1 - 0.5))
        if not inside.any():
            raise PlanError("no ion inside the AQROI {} (located: {})".format(
                [ax0, ay0, ax1, ay1], ", ".join("({:.1f}, {:.1f})".format(x, y)
                                                for x, y in pos[:8])))
        if not inside.all():
            notes.append("{} located spot(s) outside the AQROI set by hand not read out: "
                         "{}".format(int((~inside).sum()), ", ".join(
                             "({:.1f}, {:.1f})".format(x, y) for x, y in pos[~inside][:8])))
            pos = pos[inside]
            if ion_states is not None:
                ion_states = [st for st, k in zip(ion_states, inside) if k]
            n = len(pos)
    if n > max_rois:
        raise PlanError("{} ions but the Grabber has only {} ROI engines".format(n, max_rois))
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
    cache = {}

    def cands(px, py):
        # the candidates do not depend on the partition axis: computed once a phase
        if (px, py) not in cache:
            cache[(px, py)] = _all_candidates(pos, sx, sy, px, py, hbin, vbin, bounds, fixed,
                                              halo)
        return cache[(px, py)]

    for along_x in (dominant, not dominant):
        for px, py in phases:
            res = _best_layout(pos, sx, sy, px, py, hbin, vbin, bounds, fixed, along_x, halo,
                               cands(px, py))
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
            res = _independent_layout(pos, sx, sy, px, py, hbin, vbin, bounds, fixed, halo,
                                      cands(px, py))
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
        hint = ("a super-pixel smaller than the ion image holds little of its light: more "
                "binning or a larger roi_size" if fixed else
                "consider less binning (see binning_advice) or a larger ion spacing")
        geometry.append("no placement reaches containment >= {:.0%} with crosstalk <= {:.0%}; "
                        "best worst-ion containment {:.1%}, worst crosstalk {:.1%} - {}".format(
                            min_containment, max_crosstalk, cont.min(),
                            float(xt.max()) if n > 1 else 0.0, hint))
    m0 = np.array([p[1] for p in picks])
    n0 = np.array([p[2] for p in picks])
    wid = np.array([p[3] for p in picks])
    hgt = np.array([p[4] for p in picks])
    sizes = sorted(set(zip(wid.tolist(), hgt.tolist())))

    if aq_fixed is not None:
        aq = aq_fixed
        u_min, u_max = (aq[0] - px) // hbin, (aq[2] - px) // hbin
        v_min, v_max = (aq[1] - py) // vbin, (aq[3] - py) // vbin
        bg_row = None
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
        bg_row = None
        if background_roi:
            # the background strip: a super-pixel row across the ion columns
            # holding < 2 % of an ion's light, as close to the ions as that
            # allows (small ROIs leave the ions' wings in the next row; every
            # extra row costs readout time)
            c0, c1 = int(m0.min()), int((m0 + wid).max())
            best_bg = None
            for extra in range(3):
                for row in (v_min - extra, v_max - 1 + extra):
                    y_a = py + row * vbin
                    if y_a < 0 or y_a + vbin > sh or \
                            any(n0[i] <= row < n0[i] + hgt[i] for i in range(n)):
                        continue
                    c = float(_light_matrix([px + c0 * hbin - 0.5], [px + c1 * hbin - 0.5],
                                            [y_a - 0.5], [y_a + vbin - 0.5], pos, sx, sy,
                                            halo).sum())
                    if best_bg is None or c < best_bg[0]:
                        best_bg = (c, row)
                if best_bg is not None and best_bg[0] <= 0.02:
                    break
            if best_bg is not None:
                bg_row = best_bg[1]
                v_min, v_max = min(v_min, bg_row), max(v_max, bg_row + 1)
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
        for v in ([0, fh - 1] if bg_row is None else [bg_row - v_min]):
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
    geom = grabber if grabber is not None else GrabberGeometry(flip_x=flip_x, flip_y=flip_y)
    gw, gh = geom.frame(fw, fh, hbin, vbin)
    grabber_rects = [geom.rect(r, fw, fh, hbin, vbin) for r in rois]
    lim = (1 << res_width) - 1
    for r in grabber_rects:
        problems = validate_roi(r, gw, gh, lim)
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
        background_roi=bg_rect, grabber_rois=grabber_rects,
        containment=[float(c) for c in cont], crosstalk=[float(c) for c in crosstalk],
        crosstalk_matrix=[[float(v) for v in row] for row in f],
        background_contamination=bg_cont, flip_x=geom.flip_x, flip_y=geom.flip_y,
        grabber=geom.to_dict(), grabber_frame=(gw, gh),
        psf_halo=(float(halo[0]), float(halo[1])), warnings=notes + geometry, notes=notes,
        targets=(float(min_containment), float(max_crosstalk)), aqroi_by_hand=aqroi is not None)


def square_aqroi(positions: Sequence[Sequence[float]], sigma: Tuple[float, float],
                 sensor_shape: Tuple[int, int], hbin: int, vbin: int, cols: int,
                 rows: Optional[int] = None, center: Optional[Sequence[float]] = None,
                 **kw) -> Rect:
    """An AQROI of ``cols`` x ``rows`` super-pixels (square by default) set by
    hand, centred on ``center`` (sensor px; default: the ions), on the
    binning-grid phase the planner picks for the ions it holds - so they sit
    as centrally in their super-pixels as an automatic plan puts them.
    ``kw``: :func:`plan_readout` options (roi_size, halo, ...).  Returns
    ``(x0, y0, x1, y1)`` (sensor px, half-open)."""
    rows = int(cols if rows is None else rows)
    cols = int(cols)
    sh, sw = sensor_shape
    if cols < 1 or rows < 1:
        raise PlanError("an AQROI needs at least one super-pixel column and row")
    w, h = cols * hbin, rows * vbin
    if w > sw or h > sh:
        raise PlanError("an AQROI of {} x {} super-pixels ({} x {} px) is larger than the "
                        "{} x {} sensor".format(cols, rows, w, h, sw, sh))
    pos = np.asarray(positions, float).reshape(-1, 2)
    if center is None:
        if not len(pos):
            raise PlanError("no ions to place the AQROI around")
        cx, cy = (float(v) for v in pos.mean(axis=0))
    else:
        cx, cy = float(center[0]), float(center[1])
    near = pos[(np.abs(pos[:, 0] - cx) < 0.5 * w) & (np.abs(pos[:, 1] - cy) < 0.5 * h)] \
        if len(pos) else pos
    px = py = 0
    if len(near):
        opts = {k: v for k, v in kw.items() if k in ("roi_size", "halo", "min_containment",
                                                     "max_crosstalk")}
        p = plan_readout(near, sigma, sensor_shape, hbin, vbin, frame_margin=0,
                         background_roi=False, **opts)
        x0, y0 = p.roi[0], p.roi[1]
        px, py = x0 % hbin, y0 % vbin
        if center is None:
            # centred on the ions' ROIs, not just on the ions
            u0 = min(r[0] for r in p.ion_rois)
            u1 = max(r[2] for r in p.ion_rois)
            v0 = min(r[1] for r in p.ion_rois)
            v1 = max(r[3] for r in p.ion_rois)
            cx = x0 + 0.5 * (u0 + u1) * hbin - 0.5
            cy = y0 + 0.5 * (v0 + v1) * vbin - 0.5
    out = []
    for c, size, b, phase, n_px in ((cx, w, hbin, px, sw), (cy, h, vbin, py, sh)):
        a = phase + b * int(round((c + 0.5 - 0.5 * size - phase) / b))
        while a < 0:
            a += b
        while a + size > n_px:
            a -= b
        out.append(max(0, a))
    x0, y0 = out
    return (x0, y0, x0 + w, y0 + h)


def resize_aqroi(plan: ReadoutPlan, dcols: int = 0, drows: int = 0,
                 sensor_shape: Optional[Tuple[int, int]] = None) -> Rect:
    """The AQROI of ``plan`` with ``dcols`` super-pixel columns and ``drows``
    rows more (or fewer, if negative), on the same grid: one at a time on the
    side with the narrower margin around the ion ROIs (removed from the wider
    one), so the ions stay centred; never through an ion's ROI."""
    x0, y0, x1, y1 = plan.roi
    hb, vb = plan.hbin, plan.vbin
    sh, sw = sensor_shape if sensor_shape is not None else (1 << 30, 1 << 30)
    fw, fh = plan.frame_size
    rois = plan.ion_rois
    lo = [min(r[0] for r in rois), min(r[1] for r in rois)]         # margins in super-px
    hi = [fw - max(r[2] for r in rois), fh - max(r[3] for r in rois)]
    edges = [[x0, x1], [y0, y1]]
    for axis, (d, b, n_px) in enumerate(((int(dcols), hb, sw), (int(drows), vb, sh))):
        name = ("columns", "rows")[axis]
        for _ in range(abs(d)):
            a, z = edges[axis]
            if d > 0:
                if (lo[axis] <= hi[axis] and a - b >= 0) or z + b > n_px:
                    if a - b < 0:
                        raise PlanError("the AQROI cannot get more {}: it reaches both edges "
                                        "of the sensor".format(name))
                    edges[axis][0] = a - b
                    lo[axis] += 1
                else:
                    edges[axis][1] = z + b
                    hi[axis] += 1
            else:
                if lo[axis] >= hi[axis] and lo[axis] >= 1:
                    edges[axis][0] = a + b
                    lo[axis] -= 1
                elif hi[axis] >= 1:
                    edges[axis][1] = z - b
                    hi[axis] -= 1
                else:
                    raise PlanError("the AQROI cannot lose more {}: the ion ROIs fill it "
                                    "(use less binning or smaller ROIs)".format(name))
    (x0, x1), (y0, y1) = edges
    return (x0, y0, x1, y1)


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


@dataclass(frozen=True)
class GrabberGeometry:
    """Where the camera's binned frame (as the SDK delivers it over USB) sits
    in the Camera Link stream the Grabber counts, per axis: stream pixels per
    binned pixel (``x_scale``; 0 = the binning: the stream is not binned
    along that axis), stream pixels before the image in every line / lines
    before it in every frame (``x_offset`` / ``y_offset``: pre-scan pixels)
    and whether the stream runs the other way (``flip_x`` / ``flip_y``; the
    offset comes first either way).  The identity is what an iXon Ultra 897
    sends."""
    x_scale: int = 1
    y_scale: int = 1
    x_offset: int = 0
    y_offset: int = 0
    flip_x: bool = False
    flip_y: bool = False

    def scales(self, hbin: int, vbin: int) -> Tuple[int, int]:
        return (int(self.x_scale) or int(hbin), int(self.y_scale) or int(vbin))

    def frame(self, w: int, h: int, hbin: int, vbin: int) -> Tuple[int, int]:
        """The Grabber frame (stream pixels) holding a ``w x h`` binned frame
        (at least: pixels after the image are not known)."""
        sx, sy = self.scales(hbin, vbin)
        return int(self.x_offset + w * sx), int(self.y_offset + h * sy)

    def rect(self, r: Rect, w: int, h: int, hbin: int, vbin: int) -> Rect:
        """A rectangle of the ``w x h`` binned (SDK) frame -> Grabber ROI."""
        sx, sy = self.scales(hbin, vbin)
        u0, v0, u1, v1 = map_rect(r, w, h, self.flip_x, self.flip_y)
        return (int(self.x_offset + u0 * sx), int(self.y_offset + v0 * sy),
                int(self.x_offset + u1 * sx), int(self.y_offset + v1 * sy))

    @property
    def identity(self) -> bool:
        return self == GrabberGeometry()

    def describe(self, hbin: int, vbin: int) -> str:
        """In words, e.g. "4 stream pixels per super-pixel in x, 1 in y"."""
        if self.identity:
            return "one Camera Link pixel per super-pixel, no offset, not mirrored"
        sx, sy = self.scales(hbin, vbin)
        out = ["{} stream pixel{} per super-pixel in x{}, {} in y{}".format(
            sx, "s" * (sx != 1), " (unbinned)" if sx == hbin != 1 else "", sy,
            " (unbinned)" if sy == vbin != 1 else "")]
        if self.x_offset or self.y_offset:
            out.append("the image starts {} pixel{} into each line, {} line{} into the "
                       "frame".format(self.x_offset, "s" * (self.x_offset != 1),
                                      self.y_offset, "s" * (self.y_offset != 1)))
        flips = [a for a, f in (("x", self.flip_x), ("y", self.flip_y)) if f]
        out.append("mirrored in " + " and ".join(flips) if flips else "not mirrored")
        return "; ".join(out)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "GrabberGeometry":
        d = d or {}
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: (bool(v) if k.startswith("flip") else int(v))
                      for k, v in d.items() if k in names})


def camera_link_frame(frame: np.ndarray, geom: GrabberGeometry, hbin: int, vbin: int,
                      bias: float = 0.0, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """The binned (SDK) ``frame`` as a camera streaming it with ``geom`` puts
    it on Camera Link (simulation and tests): a binned pixel's light shared
    by its stream pixels, each read with its own ``bias``; offset pixels at
    the bias.  ``size``: (W, H) of the stream frame, if larger."""
    f = np.asarray(frame, float)
    h, w = f.shape
    sx, sy = geom.scales(hbin, vbin)
    if geom.flip_x:
        f = f[:, ::-1]
    if geom.flip_y:
        f = f[::-1, :]
    block = np.repeat(np.repeat((f - bias) / (sx * sy), sy, axis=0), sx, axis=1) + bias
    gw, gh = geom.frame(w, h, hbin, vbin)
    if size is not None:
        gw, gh = max(gw, int(size[0])), max(gh, int(size[1]))
    out = np.full((gh, gw), float(bias))
    out[geom.y_offset:geom.y_offset + h * sy, geom.x_offset:geom.x_offset + w * sx] = block
    return out


def infer_grabber_geometry(expected: np.ndarray, measured: np.ndarray, hbin: int = 1,
                           vbin: int = 1, max_tries: int = 200000,
                           pixel_batch: Optional[np.ndarray] = None) -> dict:
    """How a frame the Grabber saw (``measured``: H_g x W_g stream pixels,
    e.g. rebuilt from single-pixel ROIs) holds the camera's binned frame
    (``expected``: H x W, the mean of USB frames of the same light): every
    geometry that fits - stream pixels per binned pixel, the image's offset,
    the direction, per axis - is tried, the measured frame summed back into
    binned pixels and compared with the expected one (correlation).  The best
    wins if it explains the light (r >= 0.6) and every other way round - the
    other mirrorings - fits clearly worse (by 0.03), and so does every other
    layout (scale or offset: by 0.03, or with at least twice the misfit 1 - r):
    a symmetric image cannot tell the mirrorings apart, a noisy one (too few
    frames averaged) the layouts.  Also the
    intensity: summed stream pixels = ``gain`` x USB pixel + ``offset`` (each
    stream pixel carries its own bias).

    ``pixel_batch``: when the stream pixels were measured in different shots
    (n_engines at a time), the batch of each (H_g x W_g indices) and
    ``expected`` the USB mean frame of each batch's shots (n x H x W): each
    stream pixel is compared with the light of its own shots - an ion's
    brightness differs from batch to batch (EM gain, shot noise, drifts) far
    more than a layout's fit."""
    e = np.asarray(expected, float)
    m = np.asarray(measured, float)
    out: dict = {"ok": False, "frame": [int(m.shape[1]), int(m.shape[0])] if m.ndim == 2 else None}
    stack = None
    if pixel_batch is not None:
        stack = e if e.ndim == 3 else e[None]
        pb = np.asarray(pixel_batch, int)
        if m.ndim != 2 or pb.shape != m.shape or pb.min() < 0 or pb.max() >= len(stack):
            return dict(out, reason="pixel_batch must give a batch of the expected frames for "
                                    "every pixel of the Grabber frame")
        e = stack.mean(axis=0)
    if e.ndim != 2 or m.ndim != 2:
        return dict(out, reason="frames must be 2-D")
    h, w = e.shape
    gh, gw = m.shape
    if gh < h or gw < w:
        return dict(out, reason="the Grabber frame ({} x {}) is smaller than the camera's "
                                "({} x {})".format(gw, gh, w, h))

    def axis(n, ng):
        return [(s_, o, f) for s_ in range(1, ng // n + 1) for o in range(ng - n * s_ + 1)
                for f in (False, True)]

    cx, cy = axis(w, gw), axis(h, gh)
    if len(cx) * len(cy) > max_tries:
        # (large frames: the plausible scales only - binned, or not binned)
        cx = [c for c in cx if c[0] in (1, hbin)]
        cy = [c for c in cy if c[0] in (1, vbin)]
    ez = e - e.mean()
    en = float(np.sqrt((ez * ez).sum()))
    if en <= 0:
        return dict(out, reason="the camera's frame is uniform: put light into it (an ion)")
    ev = ez.ravel() / (en * en)
    jj = np.arange(h)[:, None, None, None]
    ii = np.arange(w)[None, None, :, None]
    best: dict = {}
    top: list = []                  # the 8 best (score, r, geometry): the best rival layout
    for sx, ox, fx in cx:
        cols = m[:, ox:ox + w * sx].reshape(gh, w, sx).sum(axis=2)
        if stack is not None:
            bx = pb[:, ox:ox + w * sx].reshape(gh, w, sx)
        if fx:
            cols = cols[:, ::-1]
            if stack is not None:
                bx = bx[:, ::-1]
        for sy, oy, fy in cy:
            b = cols[oy:oy + h * sy].reshape(h, sy, w).sum(axis=1)
            if fy:
                b = b[::-1]
            ez_g, en_g, ev_g = ez, en, ev
            if stack is not None:
                # the expected light of each binned pixel: the USB frames of
                # the shots its stream pixels were measured in
                by = bx[oy:oy + h * sy].reshape(h, sy, w, sx)
                if fy:
                    by = by[::-1]
                eg = stack[by, jj, ii].mean(axis=(1, 3))
                ez_g = eg - eg.mean()
                en_g = float(np.sqrt((ez_g * ez_g).sum()))
                if en_g <= 0:
                    continue
                ev_g = ez_g.ravel() / (en_g * en_g)
            bz = b - b.mean()
            bn = float(np.sqrt((bz * bz).sum()))
            r = float((ez_g * bz).sum()) / (en_g * bn) if bn > 0 else 0.0
            # The stream pixels of a binned pixel hold its light: summed back,
            # gain 1 (or as many times as copies of it are sent).  A geometry
            # that sums part of the light, or a neighbour's, is no explanation.
            gain = float(np.dot(ev_g, bz.ravel()))
            n_px = sx * sy
            if not (abs(gain - 1.0) <= 0.15 or abs(gain - n_px) <= 0.15 * n_px):
                continue
            # (equal fits: the simpler geometry)
            score = r - 1e-9 * (sx + sy + ox + oy + fx + fy)
            key = (fx, fy)
            if key not in best or score > best[key][0]:
                best[key] = (score, r, GrabberGeometry(sx, sy, ox, oy, fx, fy), b,
                             ez_g + (e.mean() if stack is None else eg.mean()))
            if len(top) < 8 or score > top[-1][0]:
                top = sorted(top + [(score, r, (sx, sy, ox, oy, fx, fy))],
                             key=lambda v: -v[0])[:8]
    if not best:
        return dict(out, reason="no geometry sums the Grabber's pixels back to the camera's "
                                "light: is the readout armed with this plan, and light on it?")
    ranked = sorted(best.values(), key=lambda v: -v[0])
    _, r, geom, b, e_fit = ranked[0]
    other = ranked[1][1] if len(ranked) > 1 else -1.0
    # the best layout other than the winner's (any mirroring): at most 4 of
    # the top 8 are the winner's layout
    layout = (geom.x_scale, geom.y_scale, geom.x_offset, geom.y_offset)
    rival = next((v for v in top if v[2][:4] != layout), None)
    r_rival = rival[1] if rival is not None else -1.0
    # the intensity: binned stream pixels against USB pixels
    a_, c_ = np.polyfit(e_fit.ravel(), b.ravel(), 1) if np.ptp(e_fit) > 0 else (1.0, 0.0)
    sx, sy = geom.scales(hbin, vbin)
    # a scale equal to the binning is the stream not binned: stored as 0, so
    # it follows the binning (a scale of 1 stays 1)
    stored = GrabberGeometry(0 if sx == hbin != 1 else sx, 0 if sy == vbin != 1 else sy,
                             geom.x_offset, geom.y_offset, geom.flip_x, geom.flip_y)
    # clearly better than any other layout: by 0.03, or half its misfit (near r = 1)
    clear = r - r_rival >= 0.03 or 1.0 - r <= 0.5 * (1.0 - r_rival)
    ok = r >= 0.6 and r - other >= 0.03 and clear
    out.update(ok=bool(ok), geometry=stored.to_dict(), r=round(r, 4),
               r_other_mirroring=round(other, 4), r_other_layout=round(r_rival, 4),
               ambiguous=bool(r - other < 0.03 or not clear),
               gain=round(float(a_), 4), offset=round(float(c_), 2),
               summary=stored.describe(hbin, vbin))
    if not ok:
        if r < 0.6:
            out["reason"] = ("no geometry explains the Grabber's frame (r = {:.2f}): is the "
                             "readout armed with this plan and light on the ions?".format(r))
        elif r - other < 0.03:
            out["reason"] = ("two ways round fit about as well - the picture looks alike "
                             "mirrored: make it asymmetric (one ion dark, or the AQROI moved "
                             "by a super-pixel so the light is off-centre both ways) and repeat")
        else:
            sx_, sy_, ox_, oy_, fx_, fy_ = rival[2]
            other_geom = GrabberGeometry(sx_, sy_, ox_, oy_, fx_, fy_)
            out["reason"] = ("two layouts fit about as well: {} (r = {:.3f}) and {} (r = "
                             "{:.3f}) - average more frames per pixel or put more light into "
                             "the AQROI, and repeat".format(
                                 geom.describe(hbin, vbin), r, other_geom.describe(hbin, vbin),
                                 r_rival))
    return out


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

def verify_plan(plan: ReadoutPlan, frames: Sequence[np.ndarray], min_snr: float = 3.0,
                enf: float = 2.0) -> dict:
    """Check, on frames taken with the final readout settings (SDK orientation),
    that every ion is really inside its ROI.  The ions shine in every frame,
    so the frames also give the quick fidelity estimate of
    :func:`shot_histograms` (``enf``: 2 with EM gain, 1 without)."""
    f = np.asarray(frames)
    problems: List[str] = []
    w, h = plan.frame_size
    if f.ndim != 3 or f.shape[1:] != (h, w):
        problems.append("frame shape {} differs from the planned {}x{}".format(
            tuple(f.shape[1:]), w, h))
        return {"ok": False, "problems": problems}
    raw = plan.roi_sums(f)
    sums = raw.astype(float)
    n = plan.n_ions
    area = [(r[2] - r[0]) * (r[3] - r[1]) for r in plan.sdk_rois()]
    quick = shot_histograms(plan, raw, f[:, dark_pixel_mask(plan)], enf=enf)["ions"]
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
        nets.append(float(net.mean()))
        ions.append({"mean": round(float(s.mean()), 1), "std": round(std, 1),
                     "net": round(float(net.mean()), 1), "snr": round(snr, 1),
                     "snr_single_shot": round(shot, 2)})
        q = quick[i]
        if q.get("ok") and plan.ion_states[i] == "bright":
            # the frames' own threshold, bright and dark error (background as dark)
            ions[-1].update(threshold=q["threshold"], fidelity=round(q["fidelity"], 4),
                            discrimination_error=round(1.0 - q["fidelity"], 4))
            if "dark_error_neighbours_bright" in q:
                ions[-1]["dark_error_neighbours_bright"] = round(
                    q["dark_error_neighbours_bright"], 4)
        if plan.ion_states[i] == "bright" and snr < min_snr:
            problems.append("ion {}: SNR {:.1f} < {} - not in its ROI?".format(i, snr, min_snr))
    # every ROI against the light the plan expects in it (containment and the
    # neighbours' share, for whatever ROI sizes and overlaps the plan has)
    lit = [i for i in range(n) if plan.ion_states[i] == "bright"]
    F = np.asarray(plan.crosstalk_matrix, float).reshape(n, n) if plan.crosstalk_matrix \
        else np.eye(n)
    expected = F[:, lit].sum(axis=1) if lit else np.zeros(n)
    ratio = {i: nets[i] / expected[i] for i in lit if expected[i] > 0.02}
    if len(ratio) >= 2:
        med = float(np.median(list(ratio.values())))
        for i, r in ratio.items():
            if med > 0 and r < 0.4 * med:
                problems.append("ion {} gives only {:.0%} of the signal expected in its ROI "
                                "(compared with the other ions)".format(i, r / med))
    return {"ok": not problems, "problems": problems, "n_frames": int(f.shape[0]),
            "background_per_pixel": round(bg_level, 2), "background_noise": round(bg_noise, 2),
            "ions": ions, "max_count": int(f.max())}


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


# ---------------------------------------------------------------------------
# shot histograms: a quick fidelity estimate from shots prepared bright
# ---------------------------------------------------------------------------

def dark_pixel_mask(plan: ReadoutPlan, max_light: float = 0.02,
                    max_pixels: int = 64) -> np.ndarray:
    """Frame pixels (binned, SDK orientation) that serve as the dark
    reference: the background ROI or, without one, the (at most
    ``max_pixels``) super-pixels holding the least ion light, all below
    ``max_light`` of an ion's light (planned PSF)."""
    w, h = plan.frame_size
    mask = np.zeros((h, w), bool)
    if plan.background_roi is not None:
        u0, v0, u1, v1 = plan.background_roi
        mask[v0:v1, u0:u1] = True
        return mask
    vv, uu = np.mgrid[0:h, 0:w]
    ax = plan.roi[0] + uu.ravel() * plan.hbin - 0.5
    ay = plan.roi[1] + vv.ravel() * plan.vbin - 0.5
    f = _light_matrix(ax, ax + plan.hbin, ay, ay + plan.vbin,
                      np.asarray(plan.positions, float).reshape(-1, 2), *plan.psf_sigma,
                      plan.psf_halo)
    light = f.sum(axis=1)
    keep = np.argsort(light, kind="stable")[:max_pixels]
    mask.ravel()[keep[light[keep] < max_light]] = True
    return mask


def _dark_sums(px: np.ndarray, area: int) -> np.ndarray:
    """Dark-reference sums for a ROI of ``area`` super-pixels from the dark
    pixels ``px`` (shots x pixels): neighbouring pixels of one shot added up
    (pixels of several shots when a shot has too few)."""
    px = np.asarray(px, np.int64)
    if px.ndim != 2 or px.size == 0:
        return np.zeros(0, np.int64)
    if area <= 1:
        return px.ravel()
    if area <= px.shape[1]:
        c = np.concatenate([np.zeros((len(px), 1), np.int64), np.cumsum(px, axis=1)], axis=1)
        return (c[:, area:] - c[:, :-area]).ravel()
    flat = px.ravel()
    k = flat.size // area
    return flat[:k * area].reshape(k, area).sum(axis=1)


def threshold_fidelity(bright: Sequence[float], dark: Sequence[float]) -> dict:
    """The threshold between ROI sums of shots prepared bright and dark
    (bright if sum > threshold) that misclassifies the fewest, and the
    readout fidelity it gives, ``1 - (P(bright <= t) + P(dark > t)) / 2`` -
    counted on the shots themselves, no model of the distributions."""
    b = np.sort(np.rint(np.asarray(bright, float)).astype(np.int64).ravel())
    d = np.sort(np.rint(np.asarray(dark, float)).astype(np.int64).ravel())
    if b.size == 0 or d.size == 0:
        return {"ok": False, "reason": "no {} shots".format("bright" if b.size == 0 else "dark")}
    cand = np.unique(np.concatenate([b, d]))
    eb = np.searchsorted(b, cand, "right") / b.size          # bright read as dark
    ed = 1.0 - np.searchsorted(d, cand, "right") / d.size    # dark read as bright
    err = eb + ed
    best = np.flatnonzero(err <= err.min() + 1e-12)
    k = int(best[len(best) // 2])
    # every integer from cand[k] to cand[k + 1] - 1 classifies the same: the middle
    t = int(cand[k]) if k + 1 >= len(cand) else int((cand[k] + cand[k + 1] - 1) // 2)
    nb_err = int(np.searchsorted(b, t, "right"))
    nd_err = int(d.size - np.searchsorted(d, t, "right"))
    e_b, e_d = nb_err / b.size, nd_err / d.size
    return {"ok": True, "threshold": t, "fidelity": 1.0 - 0.5 * (e_b + e_d),
            "bright_error": e_b, "dark_error": e_d, "bright_errors": nb_err,
            "dark_errors": nd_err, "n_bright": int(b.size), "n_dark": int(d.size),
            "bright_mean": float(b.mean()), "dark_mean": float(d.mean())}


def shot_histograms(plan: ReadoutPlan, sums, dark_px, enf: float = 2.0, bins: int = 80,
                    seed: int = 0) -> dict:
    """Histograms of readout shots and a quick readout-fidelity estimate.

    Every shot is taken as prepared **bright** (prepare bright on purpose), so
    each ion ROI's sums are its bright distribution.  The dark reference is
    the background pixels of the same shots (:func:`dark_pixel_mask`) added
    up to the ion ROI's size: what the ROI reads with the ion dark and no
    light from its neighbours.  :func:`threshold_fidelity` puts the threshold
    where the two overlap least.

    A dark ion next to bright ones also gets their crosstalk.  The planned
    crosstalk matrix turns the measured signals into the light every ion
    puts into the other ROIs; added to the dark reference as EMCCD
    photo-electrons (Poisson, each amplified with an exponential gain
    distribution for ``enf`` = 2 - EM gain on - or fixed for ``enf`` = 1;
    the counts per photo-electron from the bright shots' excess variance)
    it gives the dark error, at the same threshold, with all neighbours
    bright."""
    n, n_rois = plan.n_ions, plan.n_rois
    s = np.asarray(sums, np.int64).reshape(-1, n_rois)
    px = np.asarray(dark_px, np.int64)
    if px.ndim != 2 or len(px) != len(s):
        px = np.zeros((len(s), 0), np.int64)
    areas = [(r[2] - r[0]) * (r[3] - r[1]) for r in plan.sdk_rois()[:n]]
    dark = {a: _dark_sums(px, a) for a in sorted(set(areas))}
    out = {"shots": int(len(s)), "n_ions": n, "dark_pixels": int(px.shape[1]),
           "dark_reference": "background ROI" if plan.background_roi is not None
           else "frame pixels with < 2% of an ion's light",
           "ions": [], "series": [], "edges": []}
    if len(s) == 0:
        return out
    nets, excess = [], []
    for i in range(n):
        d, b = dark[areas[i]], s[:, i]
        r = threshold_fidelity(b, d) if d.size else \
            {"ok": False, "reason": "no dark reference (no background pixels in the frame)"}
        r.update(index=i, label="ion {}".format(i), roi_pixels=areas[i])
        if d.size:
            nets.append(float(b.mean() - d.mean()))
            excess.append(float(b.var() - d.var()))
            r["signal"] = nets[-1]
        out["ions"].append(r)
    if n > 1 and len(nets) == n and plan.crosstalk_matrix:
        F = np.asarray(plan.crosstalk_matrix, float).reshape(n, n)
        net = np.maximum(np.array(nets), 0.0)
        diag = np.maximum(np.diag(F), 1e-9)
        light = net / diag
        if np.linalg.cond(F) < 1e6:
            light = np.linalg.solve(F, net)
        light = np.maximum(light, 0.0)
        xt = F @ light - np.diag(F) * light            # neighbours' light in each ROI
        g = max(sum(excess), 0.0) / max(enf * float(net.sum()), 1e-12)   # counts / e-
        rng = np.random.default_rng(seed)
        for i, r in enumerate(out["ions"]):
            r["crosstalk"] = float(xt[i])
            if not r.get("ok"):
                continue
            d = dark[areas[i]]
            if g > 0:
                k = rng.poisson(xt[i] / g, d.size)
                x = rng.gamma(k, g) if enf > 1.5 else k * g
            else:
                x = np.full(d.size, xt[i])
            e = float(np.mean(np.rint(d + x) > r["threshold"]))
            r["dark_error_neighbours_bright"] = e
            r["fidelity_neighbours_bright"] = 1.0 - 0.5 * (r["bright_error"] + e)
    # common bins for all curves: the ion ROIs, then the dark reference(s)
    series = [("ion {}".format(i), s[:, i]) for i in range(n)]
    series += [("bg" if len(dark) == 1 else "bg ({} px)".format(a), v)
               for a, v in dark.items() if v.size]
    lo = math.floor(min(np.percentile(v, 0.02) for _, v in series))
    hi = math.ceil(max(np.percentile(v, 99.98) for _, v in series))
    width = max(1, int(math.ceil((hi - lo + 1) / max(int(bins), 1))))
    edges = lo - 0.5 + width * np.arange(int(math.ceil((hi - lo + 1) / width)) + 1)
    out["edges"] = edges.tolist()
    for name, v in series:
        h, _ = np.histogram(v, edges)
        out["series"].append({"label": name, "n": int(v.size),
                              "fraction": (h / v.size).tolist()})
    return out


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
