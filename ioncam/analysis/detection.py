"""Fast, robust ion spot detection for EMCCD frames.

Pipeline per frame (float32; ~10 ms for a full 512x512 frame, ~1 ms for a
typical analysis ROI):

1. optional subtraction of a learned *static* background structure (fine
   scatter features, hot pixels), scaled with exposure x gain;
2. smooth background from tile medians, bilinearly interpolated with cached
   separable interpolation matrices (follows drifting scatter and blobs);
   in small frames the ions are masked out of the tile medians;
3. a *noise map* from tile-wise MAD of the residual: EMCCD noise grows with
   the background, so a single global noise figure would under-threshold
   bright scatter regions.  Its scale comes from the upper tail of the
   pixels away from the ions (:func:`filtered_noise_scale`), so a camera ROI
   hugging the crystal works too;
4. matched filter (Gaussian, ``filter_scale`` x PSF so that dense chains stay
   resolved at ~90 % of the optimal SNR) and local maxima above
   ``threshold_sigma`` x local filtered noise;
5. per peak: sub-pixel position (parabolic), source width from the log
   curvature of the filtered peak (nearest neighbours only, so chain
   neighbours barely bias it) and a broadness probe on a ring around the
   peak.  This separates ions (PSF sized), uncrystallised clouds (broad in
   every direction) and cosmic rays / CIC spikes (single pixel) without fits.

All positions are returned in *sensor* coordinates (see
:mod:`ioncam.camera.base`) so results from different ROIs/binnings compare.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from ioncam.analysis.fitting import PSF, fit_fixed_psf, fit_gaussians, linear_photometry
from ioncam.camera.base import Frame

try:  # optional acceleration
    import cv2 as _cv2
except Exception:  # pragma: no cover - OpenCV is optional
    _cv2 = None

_RING = np.array([[math.cos(a), math.sin(a)] for a in np.arange(8) * math.pi / 4])


@dataclass
class Spot:
    x: float            # sensor coordinates
    y: float
    u: float            # frame coordinates (binned pixels)
    v: float
    peak: float         # matched-filter response (counts)
    amplitude: float    # estimated PSF peak height (counts per frame pixel)
    flux: float         # estimated integrated counts
    snr: float          # peak / local filtered noise
    width: float        # estimated source sigma [sensor px]
    kind: str = "ion"   # "ion" | "cloud"
    blended: bool = False  # position/flux from a joint fit with overlapping neighbours

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "amplitude": self.amplitude, "flux": self.flux,
                "snr": self.snr, "width": self.width, "kind": self.kind,
                "blended": self.blended}


@dataclass
class DetectionResult:
    frame_index: int
    timestamp: float
    spots: List[Spot]                 # ions
    clouds: List[Spot]                # broad, uncrystallised blobs
    background: float                 # median background level [counts / frame px]
    noise: float                      # median pixel noise rms [counts]
    noise_f: float                    # median matched-filter noise rms [counts]
    filtered: np.ndarray              # matched-filter image of the analysed crop
    residual: np.ndarray              # background-subtracted crop
    crop: Tuple[int, int, int, int]   # (u0, v0, u1, v1) of the crop inside the frame
    x0: int = 0
    y0: int = 0
    hbin: int = 1
    vbin: int = 1
    sigma_f: Tuple[float, float] = (1.0, 1.0)   # filter sigma (sy, sx) [frame px]
    gain_factor: float = 1.0          # exposure x EM gain of the frame
    saturated: int = 0
    max_value: int = 0
    focus: float = float("nan")       # median ion width [sensor px]
    spikes: int = 0                   # rejected single-pixel events
    noise_f_map: Optional[np.ndarray] = None
    analysis_time: float = 0.0
    psf: Optional[PSF] = None         # PSF model in frame pixels (core + halo)
    diffuse: Optional[tuple] = None   # (map, noise map, sigma, bin): light the ions do
                                      # not explain, filtered at cloud scale (crop px / bin)
    prior_fit: bool = False           # ion positions are joint fits started from the prior

    # -- geometry helpers ------------------------------------------------------
    def to_crop(self, x, y):
        """sensor -> fractional crop coordinates (col, row)."""
        cu = (np.asarray(x, float) - self.x0 - 0.5 * (self.hbin - 1)) / self.hbin - self.crop[0]
        cv = (np.asarray(y, float) - self.y0 - 0.5 * (self.vbin - 1)) / self.vbin - self.crop[1]
        return cu, cv

    def sample(self, x, y, refine: int = 0):
        """Matched-filter response and local noise at sensor positions.

        With ``refine > 0`` each position first moves to the local maximum
        within ``refine`` crop pixels.  Returns ``(values, noise, x, y)``;
        positions outside the analysed crop give value 0."""
        cu, cv = self.to_crop(x, y)
        cu = np.atleast_1d(np.asarray(cu, float)).copy()
        cv = np.atleast_1d(np.asarray(cv, float)).copy()
        f = self.filtered
        h, w = f.shape
        if h == 0 or w == 0:
            z = np.zeros(len(cu))
            return z, z + max(self.noise_f, 1e-9), np.atleast_1d(x), np.atleast_1d(y)
        if refine > 0:
            for k in range(len(cu)):
                iu, iv = int(round(cu[k])), int(round(cv[k]))
                ua, ub = max(0, iu - refine), min(w, iu + refine + 1)
                va, vb = max(0, iv - refine), min(h, iv + refine + 1)
                if ua >= ub or va >= vb:
                    continue
                win = f[va:vb, ua:ub]
                j = int(np.argmax(win))
                pv, pu = divmod(j, win.shape[1])
                du, dv = _parabola_offsets(f, va + pv, ua + pu)
                cu[k], cv[k] = ua + pu + du, va + pv + dv
        vals = ndimage.map_coordinates(f, [cv, cu], order=1, mode="nearest")
        inside = (cu >= -0.5) & (cu <= w - 0.5) & (cv >= -0.5) & (cv <= h - 0.5)
        vals = np.where(inside, vals, 0.0)
        if self.noise_f_map is not None:
            nz = ndimage.map_coordinates(self.noise_f_map, [cv, cu], order=1, mode="nearest")
        else:
            nz = np.full(len(cu), self.noise_f)
        xs = (cu + self.crop[0]) * self.hbin + self.x0 + 0.5 * (self.hbin - 1)
        ys = (cv + self.crop[1]) * self.vbin + self.y0 + 0.5 * (self.vbin - 1)
        return vals, np.maximum(nz, 1e-9), xs, ys

    def photometry(self, x, y):
        """Deblended fluxes of point sources at sensor positions.

        Linear least squares with the PSF model (core + halo) on the
        background-subtracted image: light of overlapping neighbours is
        attributed to them, so a dark ion between two bright ones reads ~0.
        Returns ``(flux, flux_err, inside)`` (counts per frame)."""
        cu, cv = self.to_crop(x, y)
        cu = np.atleast_1d(np.asarray(cu, float))
        cv = np.atleast_1d(np.asarray(cv, float))
        n = len(cu)
        flux = np.zeros(n)
        err = np.full(n, np.inf)
        res = self.residual
        h, w = res.shape if res.size else (0, 0)
        inside = (cu > -0.5) & (cu < w - 0.5) & (cv > -0.5) & (cv < h - 0.5)
        psf = self.psf
        if psf is None or not np.any(inside):
            return flux, err, inside
        idx = np.nonzero(inside)[0]
        pts = np.column_stack([cu[idx], cv[idx]])
        for group in _groups(pts, 4.0 * max(psf.sx, psf.sy) + 2.0):
            gp = pts[group]
            m = int(math.ceil(3.0 * max(psf.sx, psf.sy))) + 2
            ua = int(max(0, math.floor(gp[:, 0].min()) - m))
            ub = int(min(w, math.ceil(gp[:, 0].max()) + m + 1))
            va = int(max(0, math.floor(gp[:, 1].min()) - m))
            vb = int(min(h, math.ceil(gp[:, 1].max()) + m + 1))
            f, e, _, _ = linear_photometry(res[va:vb, ua:ub], gp - [ua, va], psf,
                                           noise=max(self.noise, 1e-9))
            flux[idx[group]] = f
            err[idx[group]] = e
        return flux, err, inside

    @property
    def positions(self) -> np.ndarray:
        return np.array([[s.x, s.y] for s in self.spots], float).reshape(-1, 2)


def mad_std(a: np.ndarray) -> float:
    a = np.asarray(a, np.float32).ravel()
    if a.size == 0:
        return 0.0
    k = a.size // 2
    med = np.partition(a, k)[k]
    return float(1.4826 * np.partition(np.abs(a - med), k)[k])


@lru_cache(maxsize=64)
def _interp_matrix(n: int, nb: int) -> np.ndarray:
    """(n, nb) matrix linearly interpolating ``nb`` tile centres onto ``n`` pixels."""
    if nb == 1:
        return np.ones((n, 1), np.float32)
    centers = (np.arange(nb) + 0.5) * n / nb - 0.5
    x = np.clip(np.arange(n, dtype=float), centers[0], centers[-1])
    j = np.clip(np.searchsorted(centers, x, side="right") - 1, 0, nb - 2)
    t = (x - centers[j]) / (centers[j + 1] - centers[j])
    m = np.zeros((n, nb), np.float32)
    m[np.arange(n), j] = 1.0 - t
    m[np.arange(n), j + 1] = t
    return m


def _tiles(img: np.ndarray, block: int):
    """(nby, nbx, n) contiguous array of the pixels of ~block-sized tiles."""
    h, w = img.shape
    nby, nbx = max(1, h // block), max(1, w // block)
    by, bx = h // nby, w // nbx
    core = img[:nby * by, :nbx * bx]
    return np.ascontiguousarray(
        core.reshape(nby, by, nbx, bx).transpose(0, 2, 1, 3)).reshape(nby, nbx, by * bx)


def _quantiles_last(t: np.ndarray, qs) -> List[np.ndarray]:
    """Quantiles along the last axis by one partial sort (order statistics, no
    interpolation - plenty for the hundreds of samples per tile used here)."""
    n = t.shape[-1]
    ks = sorted({min(n - 1, max(0, int(round(q * (n - 1))))) for q in qs})
    part = np.partition(t, ks, axis=-1)
    return [part[..., min(n - 1, max(0, int(round(q * (n - 1)))))] for q in qs]


def _expand(coarse: np.ndarray, shape) -> np.ndarray:
    h, w = shape
    nby, nbx = coarse.shape
    if nby == 1 and nbx == 1:
        return np.full((h, w), coarse[0, 0], np.float32)
    return _interp_matrix(h, nby) @ coarse.astype(np.float32) @ _interp_matrix(w, nbx).T


def _subsample(img: np.ndarray, block: int):
    """Stride-2 view for large tiles (tile statistics need ~1000 samples, not 4096)."""
    if block >= 16 and min(img.shape) >= 32:
        return img[::2, ::2], block // 2
    return img, block


def block_background(img: np.ndarray, block: int,
                     exclude: Optional[np.ndarray] = None) -> np.ndarray:
    """Smooth background: medians of ~``block``-sized tiles, bilinearly interpolated.

    Pixels flagged in ``exclude`` (objects) are left out; a tile with too few
    remaining pixels takes the median of the other tiles."""
    sub, b = _subsample(img, block)
    t = _tiles(sub, b)
    if exclude is None:
        return _expand(_quantiles_last(t, [0.5])[0], img.shape)
    keep = ~_tiles(*_subsample(exclude, block))
    ok = keep.sum(axis=2) >= max(8, t.shape[2] // 5)
    if not ok.any():
        rest = t[keep]
        return np.full(img.shape, np.median(rest if rest.size else t), np.float32)
    med = np.empty(t.shape[:2], np.float32)
    for iy, ix in zip(*np.nonzero(ok)):
        med[iy, ix] = np.median(t[iy, ix][keep[iy, ix]])
    med[~ok] = np.median(med[ok])
    return _expand(med, img.shape)


def block_noise(res: np.ndarray, block: int) -> np.ndarray:
    """Noise map: robust (MAD) standard deviation per tile, interpolated."""
    sub, b = _subsample(res, block)
    t = _tiles(sub, b)
    med = _quantiles_last(t, [0.5])[0]
    return _expand(1.4826 * _quantiles_last(np.abs(t - med[..., None]), [0.5])[0], res.shape)


def filtered_noise_scale(filt: np.ndarray, tile: int, spot_sigma: Tuple[float, float],
                         significant: float = 6.0,
                         levels: Sequence[float] = (0.995, 0.98, 0.95),
                         filled_fallback: bool = True) -> float:
    """Noise scale of the matched-filtered image, for the detection threshold.

    EMCCD noise is strongly right-skewed at low light (single amplified
    photons), so the scale comes from the upper quantile, (q99.5 - q50) /
    2.576: exact for Gaussian noise, conservative for skewed noise.  Bright
    objects must not inflate it:

    * tiles with an outlying scale (clouds, scatter, a chain) are excluded and
      the quantile is taken over the pooled pixels of the clean tiles;
    * a tight ROI around a crystal is a single tile in which the ions *are*
      the upper tail.  So objects are masked in two passes: the first masks
      everything above a high quantile (``levels``, tried in turn, for
      crystals covering more of the frame), grown by 3 ``spot_sigma`` (the
      (x, y) width of a spot in ``filt``) to include the wings of very bright
      ions; the second, using the scale of the rest, only pixels above
      ``significant`` times that scale, grown by the 1.5 widths in which a
      spot falls from there to the noise.  So in pure noise the tail that
      defines the scale stays (almost) intact.  The masked estimate is
      adopted only if it is substantially lower;
    * the lower-tail scale, which positive objects cannot inflate, is a floor
      - and, in a small frame (``filled_fallback``), the estimate when the
      objects leave no background at all.
    """
    sub, b = _subsample(filt, tile)
    step = tile // b
    tl = _tiles(sub, b)
    t50, t995 = _quantiles_last(tl, [0.5, 0.995])
    ts = t995 - t50
    clean = ts <= 2.0 * float(np.median(ts))
    if not np.any(clean):
        clean[...] = True
    vals = tl[clean]
    q005, q50, q995 = _quantiles_last(vals.ravel(), [0.005, 0.5, 0.995])
    s_all = float(q995 - q50) / 2.576
    s_low = float(q50 - q005) / 2.576
    if clean.size >= 9 and s_all <= 4.0 * s_low:
        # enough tiles for the clean-tile selection to work, and no more
        # skew than EMCCD noise has: nothing left to mask (saves time)
        return max(s_all, mad_std(vals[:, ::4]), 1e-6)
    min_rest = max(64, vals.size // 10)

    def unmasked(level, widths):
        """Pixels farther than ``widths`` spot widths from any pixel above
        ``level`` (None: there is no such pixel)."""
        hot = sub > level
        if not hot.any():
            return None
        size = tuple(2 * int(math.ceil((widths * t + 1.0) / step)) + 1
                     for t in (spot_sigma[1], spot_sigma[0]))
        grown = ndimage.maximum_filter(hot, size=size, mode="constant")
        return vals[~_tiles(grown, b)[clean]]

    def upper_scale(x):
        r50, r995 = np.quantile(x, [0.5, 0.995])
        return float(r50), float(r995 - r50) / 2.576

    # each level gives a candidate; a level that masked only part of the
    # crystal still gives an inflated one, so the smallest candidate wins
    scale, rest = s_all, vals
    for p in levels:
        first = unmasked(float(np.quantile(vals, p)), 3.0)
        if first is None or first.size < min_rest:
            break                                  # nothing would be left
        r50, s_first = upper_scale(first)
        second = unmasked(r50 + significant * s_first, 1.5)
        if second is None:
            continue                               # nothing significant (yet)
        if second.size >= min_rest:
            cand, cand_rest, limit = upper_scale(second)[1], second, 0.75
        else:
            # the crystal fills the frame: only the first pass can be used
            cand, cand_rest, limit = s_first, first, 0.5
        if cand < limit * s_all and cand < scale:
            scale, rest = cand, cand_rest
    if filled_fallback and filt.size <= 128 * 128 and clean.size < 9 and \
            s_all > 3.5 * s_low and scale > 2.0 * s_low:
        # Far more skew than Gaussian noise has, and masking found no clean
        # background.  Either the light fills the frame (blurred ions in a
        # tight ROI: a few large, bright blobs) or the noise counts photons
        # (single amplified electrons: many tiny ones, just as skewed).  Only
        # in the first case is the lower tail the best estimate left (widened
        # by the wings the median sits on, i.e. conservative).
        sig = sub > q50 + significant * s_low
        frac = float(sig.mean())
        area = frac * sig.size / max(1, len(_local_maxima(sub, sig)))
        if frac >= 0.15 and area >= 1.5 * math.pi * spot_sigma[0] * spot_sigma[1] / step ** 2:
            return max(s_low, 1e-6)
    return max(scale, s_low, mad_std(rest), 1e-6)


def _gaussian(img: np.ndarray, sy: float, sx: float) -> np.ndarray:
    """Gaussian filter with zero padding (OpenCV if available - ~5x faster)."""
    if _cv2 is not None:
        kx = 2 * int(math.ceil(3.5 * sx)) + 1
        ky = 2 * int(math.ceil(3.5 * sy)) + 1
        return _cv2.GaussianBlur(img, (kx, ky), sigmaX=sx, sigmaY=sy,
                                 borderType=_cv2.BORDER_CONSTANT)
    return ndimage.gaussian_filter(img, sigma=(sy, sx), mode="constant", truncate=3.5)


def _bandpass(img: np.ndarray, s1: Tuple[float, float], s2: Tuple[float, float]) -> np.ndarray:
    """Difference of Gaussians (linear: unlike median backgrounds it is not
    biased by skewed noise); edges reflected."""
    a = np.asarray(img, np.float32)
    a = a - np.float32(np.median(a[::4, ::4]))
    return (ndimage.gaussian_filter(a, s1, mode="reflect", truncate=3.5) -
            ndimage.gaussian_filter(a, s2, mode="reflect", truncate=3.5)).astype(np.float32)


def _local_maxima(f: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Flat indices of pixels in ``mask`` that are >= their 8 neighbours.

    Only the (few) masked pixels are tested, which is much cheaper than a
    full-image maximum filter.  The outermost ring is never a candidate."""
    h, w = f.shape
    mask = mask.copy()
    mask[[0, -1], :] = False
    mask[:, [0, -1]] = False
    iv, iu = np.nonzero(mask)
    if iv.size == 0:
        return np.zeros(0, np.int64)
    c = f[iv, iu]
    keep = np.ones(iv.size, bool)
    for dv in (-1, 0, 1):
        for du in (-1, 0, 1):
            if dv or du:
                keep &= c >= f[iv + dv, iu + du]
    return iv[keep] * w + iu[keep]


def _parabola_offsets(f: np.ndarray, iv: int, iu: int) -> Tuple[float, float]:
    h, w = f.shape

    def off(a, b, c):
        den = a - 2.0 * b + c
        if den >= 0:
            return 0.0
        return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))

    du = off(f[iv, iu - 1], f[iv, iu], f[iv, iu + 1]) if 0 < iu < w - 1 else 0.0
    dv = off(f[iv - 1, iu], f[iv, iu], f[iv + 1, iu]) if 0 < iv < h - 1 else 0.0
    return du, dv


def _log_variances(f: np.ndarray, iv: int, iu: int) -> Tuple[Optional[float], Optional[float]]:
    """Variance (x, y) of the filtered peak from its log-curvature.

    For a sampled Gaussian ``f(-1) f(+1) / f(0)^2 = exp(-1/sigma^2)``
    whatever the sub-pixel offset; only nearest neighbours are used."""
    h, w = f.shape
    c = float(f[iv, iu])

    def var(a, b):
        if c <= 0 or a <= 0 or b <= 0:
            return None
        q = float(a) * float(b) / (c * c)
        if not 0.0 < q < 1.0:
            return None
        return -1.0 / math.log(q)

    vx = var(f[iv, iu - 1], f[iv, iu + 1]) if 0 < iu < w - 1 else None
    vy = var(f[iv - 1, iu], f[iv + 1, iu]) if 0 < iv < h - 1 else None
    return vx, vy


def _groups(pts: np.ndarray, link: float) -> List[List[int]]:
    """Connected components of points closer than ``link`` (single linkage)."""
    n = len(pts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    if n > 1:
        d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
        for i, j in zip(*np.nonzero(np.triu(d < link, 1))):
            parent[find(i)] = find(j)
    out: dict = {}
    for i in range(n):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def deblend_patch(patch: np.ndarray, guesses, psf: PSF, noise_f: np.ndarray, thr: float,
                  sf: Tuple[float, float], allowed: Optional[np.ndarray] = None,
                  max_new: int = 3, min_new_flux: float = 0.25,
                  chain_search: Optional[bool] = None, var: Optional[np.ndarray] = None,
                  propose: Optional[float] = None):
    """Crowded-field analysis of one patch (background-subtracted image).

    Starting from ``guesses`` (patch coordinates) the sources are fitted jointly
    with the PSF; insignificant or coincident components are dropped, the
    residual image is searched for hidden sources (accepted only if they explain
    significantly more light - model selection), merged-looking results are
    re-tried as linear chains at their equilibrium positions, and components
    squeezed between neighbours are removed if not needed.  ``noise_f`` is the
    matched-filter noise (map or scalar) of ``patch``; a new source must carry
    at least ``min_new_flux`` of the typical flux (on long averages, where
    small PSF-model errors become significant, use ~0.5).  The chain-model
    search runs when the result looks merged (``chain_search=None``) or, with
    ``True``, whenever the sources lie on a line.  ``var`` (per-pixel variance,
    e.g. of an averaged image - on top of bright ions the noise is several
    times the background noise) makes all fits weighted, so the model
    selection compares true chi-squares.  Residual peaks above ``propose``
    (filtered-noise units; default 0.6 thr) are tried as new sources.  Returns a
    CrowdedFit (or None if nothing significant is left)."""
    sy, sx = sf
    ph, pw = patch.shape
    s_max = max(psf.sx, psf.sy)
    min_sep = 0.6 * min(psf.sx, psf.sy) + 0.3
    penalty = 25.0           # cost (in noise variances) one more source has to explain
    if allowed is None:
        allowed = np.ones(patch.shape, bool)
    vv, uu = np.mgrid[0:ph, 0:pw]

    def fit_pruned(g, move=1.5 * s_max + 0.5):
        """Fit, drop insignificant / duplicate components, refit until clean."""
        f = None
        for _ in range(len(g) + 1):
            f = fit_fixed_psf(patch, g, psf, max_move=move, var=var)
            keep = f.flux >= 0.75 * thr * np.maximum(f.flux_err, 1e-9)
            order = np.argsort(-f.flux)
            for a_i, a in enumerate(order):
                for b in order[a_i + 1:]:
                    if keep[a] and keep[b] and \
                            np.hypot(*(f.positions[a] - f.positions[b])) < min_sep:
                        keep[b] = False
            if np.all(keep):
                return f
            if not np.any(keep):
                return None
            g = f.positions[keep]
        return f

    def better(a, b):
        """Is fit ``b`` significantly better than ``a`` (per extra component)?"""
        if b is None:
            return False
        if a is None:
            return True
        extra = len(b.flux) - len(a.flux)
        s2 = min(a.rms, b.rms) ** 2
        return a.cost - b.cost > penalty * s2 * extra if extra > 0 else \
            a.cost - b.cost > -penalty * s2 * (-extra)

    g = np.asarray(guesses, float).reshape(-1, 2)
    if len(g) == 0:
        return None
    fit = fit_pruned(g)
    for _ in range(max_new):
        if fit is None:
            break
        rf = _gaussian((patch - fit.model).astype(np.float32), sy, sx)
        snr = np.where(allowed, rf / np.maximum(noise_f, 1e-12), 0.0)
        for (pu, pv) in fit.positions:
            snr[np.hypot(uu - pu, vv - pv) <= min_sep] = 0.0
        typical = float(np.median(fit.flux))
        accepted = False
        cand = _local_maxima(snr, snr >= (0.6 * thr if propose is None else propose))
        for idx in cand[np.argsort(-snr.ravel()[cand])][:2]:
            cv_, cu_ = divmod(int(idx), pw)
            trial = fit_pruned(np.vstack([fit.positions, [[cu_, cv_]]]))
            if trial is not None and len(trial.flux) > len(fit.flux) and \
                    float(np.min(trial.flux)) >= min_new_flux * typical and better(fit, trial):
                fit, accepted = trial, True
                break
        if not accepted:
            break
    if fit is not None and len(fit.flux) >= 2 and chain_search is not False:
        fl = fit.flux
        merged = fl.max() > 1.6 * np.median(fl) or fl.sum() > (len(fl) + 0.6) * np.median(fl)
        linear = False
        if chain_search and len(fl) >= 2:
            d = fit.positions - fit.positions.mean(axis=0)
            ev = np.linalg.eigvalsh(d.T @ d)
            linear = ev[0] <= 0.05 * ev[1]
        if merged or linear:
            fit = _chain_search(patch, fit, psf, fit_pruned, better, min_new_flux, var)
    if fit is not None and len(fit.flux) >= 3:
        fit = _drop_squeezed(fit, fit_pruned, better)
    return fit


def _blob_alternative(patch: np.ndarray, fit, psf: PSF, factor: float):
    """Is the light of a group of components one broad blob instead - a hot,
    uncrystallised cloud that shot noise breaks into ion-sized lumps?  Model
    selection: a round blob much broader than the PSF that explains the patch
    as well as the ions do (up to the usual penalty for their extra parameters)
    wins.  Returns (x, y, sx, sy, flux) in patch coordinates, or None."""
    fl = np.maximum(fit.flux, 0.0)
    if len(fl) < 2 or fl.sum() <= 0:
        return None
    c = (fit.positions * fl[:, None]).sum(axis=0) / fl.sum()
    wide = factor * max(psf.sx, psf.sy)
    try:
        g = fit_gaussians(patch, [c], sigma0=wide, fit_offset=True, max_nfev=60, margin=3.0)
    except (ValueError, np.linalg.LinAlgError):  # pragma: no cover
        return None
    bx, by = g.sigma
    if not (min(bx, by) > wide and max(bx, by) < 2.0 * min(bx, by) and g.flux[0] > 0):
        return None
    ph, pw = patch.shape
    (gx, gy), = g.positions
    model = g.offset + g.flux[0] * PSF(bx, by).images(pw, ph, np.array([gx]),
                                                      np.array([gy]))[0]
    cost = float(np.sum((patch - model) ** 2))
    extra = 3 * len(fl) - 5                # parameters the ions need beyond the blob's
    if cost > fit.cost + 8.0 * fit.rms ** 2 * max(extra, 0):
        return None
    return float(gx), float(gy), float(bx), float(by), float(g.flux[0])


def _deblend(spots: List[Spot], res: np.ndarray, filt: np.ndarray, noise_map: np.ndarray,
             thr: float, psf: PSF, sf: Tuple[float, float], frame: Frame, origin,
             search, max_group: int, prior: Optional[np.ndarray] = None,
             cloud_factor: float = 0.0) -> Tuple[List[Spot], List[Spot]]:
    """Crowded-field refinement of ion spots whose images overlap.

    In a dense chain of blurred ions the matched-filter peaks are pulled
    towards each other, two ions can show up as one peak and noise can split a
    peak in two.  Each group of overlapping spots is analysed jointly with the
    PSF model (:func:`deblend_patch`).  ``prior`` (crop coordinates) are where
    ions are expected (the tracked crystal): fits start there, which is both
    faster and more robust; the data still decide (model selection).  Groups
    not started from the crystal are also tried as one broad blob (a cloud,
    ``cloud_factor`` x the PSF sigma or wider); returns (ions, clouds)."""
    if len(spots) < 2 and (prior is None or len(prior) < 2):
        return spots, []
    u0, v0 = origin
    h, w = res.shape
    pts = np.array([[s.u - u0, s.v - v0] for s in spots], float).reshape(-1, 2)
    s_max = max(psf.sx, psf.sy)
    out: List[Spot] = []
    blobs: List[Spot] = []
    if prior is not None and len(prior):
        # group spots together with the expected positions (one chain = one group)
        allp = np.vstack([pts, prior])
        groups = [[i for i in g if i < len(pts)] for g in _groups(allp, 5.5 * s_max + 1.0)]
        groups = [g for g in groups if g]
    else:
        groups = _groups(pts, 5.5 * s_max + 1.0)
    for group in groups:
        n_prior = 0 if prior is None else int(np.sum(np.min(np.hypot(
            prior[:, None, 0] - pts[group][None, :, 0],
            prior[:, None, 1] - pts[group][None, :, 1]), axis=1) < 5.5 * s_max + 1.0))
        if (len(group) < 2 and n_prior < 2) or len(group) > max_group:
            out.extend(spots[i] for i in group)
            continue
        gp = pts[group]
        if prior is not None and n_prior:
            near = np.min(np.hypot(prior[:, None, 0] - gp[None, :, 0],
                                   prior[:, None, 1] - gp[None, :, 1]), axis=1) < 5.5 * s_max + 1
            gp_box = np.vstack([gp, prior[near]])
        else:
            gp_box = gp
        m = int(math.ceil(3.0 * s_max)) + 2
        ua = int(max(0, math.floor(gp_box[:, 0].min()) - m))
        ub = int(min(w, math.ceil(gp_box[:, 0].max()) + m + 1))
        va = int(max(0, math.floor(gp_box[:, 1].min()) - m))
        vb = int(min(h, math.ceil(gp_box[:, 1].max()) + m + 1))
        patch = res[va:vb, ua:ub]
        ph, pw = patch.shape
        start = gp - [ua, va]
        used_prior = False
        if prior is not None and len(prior):
            # the crystal we expect (tracker): start from it where it fits the spots
            pr = prior[(prior[:, 0] > ua - 0.5) & (prior[:, 0] < ub - 0.5) &
                       (prior[:, 1] > va - 0.5) & (prior[:, 1] < vb - 0.5)] - [ua, va]
            if len(pr):
                lone = [q for q in start if np.min(np.hypot(*(pr - q).T)) > 2.0 * s_max]
                start = np.vstack([pr] + ([np.array(lone)] if lone else []))
                used_prior = True
        allowed = np.ones(patch.shape, bool) if search is None else search[va:vb, ua:ub].copy()
        # new components only where the patch sees all of an ion's light
        edge = int(math.ceil(1.5 * s_max))
        if ua > 0:
            allowed[:, :edge] = False
        if ub < w:
            allowed[:, pw - edge:] = False
        if va > 0:
            allowed[:edge, :] = False
        if vb < h:
            allowed[ph - edge:, :] = False
        # started from the tracked crystal, only clear residual peaks are new ions
        fit = deblend_patch(patch, start, psf, noise_map[va:vb, ua:ub], thr, sf,
                            allowed, max_new=len(group) // 2 + 2,
                            propose=thr if used_prior else None)
        if fit is None:
            out.extend(spots[i] for i in group)
            continue
        blob = None
        if cloud_factor > 0 and not used_prior and len(fit.flux) >= 2:
            blob = _blob_alternative(patch, fit, psf, cloud_factor)
        if blob is not None:
            bu, bv, bx, by, bflux = blob
            pu, pv = bu + ua, bv + va
            cu, cv = u0 + pu, v0 + pv
            x, y = frame.to_sensor(cu, cv)
            peak = float(ndimage.map_coordinates(filt, [[pv], [pu]], order=1)[0])
            # matched-filter significance of the blob (pixel noise from the
            # filtered noise: a Gaussian filter of sigma s divides it by
            # sqrt(4 pi sx sy))
            nf = float(noise_map[int(np.clip(round(pv), 0, h - 1)),
                                 int(np.clip(round(pu), 0, w - 1))])
            pix = nf * math.sqrt(4.0 * math.pi * sf[0] * sf[1])
            snr = bflux / max(pix * math.sqrt(4.0 * math.pi * bx * by), 1e-12)
            blobs.append(Spot(float(x), float(y), float(cu), float(cv), peak,
                              bflux / (2.0 * math.pi * bx * by), bflux, float(snr),
                              math.sqrt(bx * frame.hbin * by * frame.vbin), "cloud"))
            continue
        sig = fit.flux / np.maximum(fit.flux_err, 1e-9)
        width = float(np.median([spots[i].width for i in group]))
        for k in range(len(fit.flux)):
            pu, pv = fit.positions[k] + [ua, va]
            if not (-0.5 < pu < w - 0.5 and -0.5 < pv < h - 0.5):
                continue
            peak = float(ndimage.map_coordinates(filt, [[pv], [pu]], order=1)[0])
            cu, cv = u0 + pu, v0 + pv
            x, y = frame.to_sensor(cu, cv)
            amp = float(fit.flux[k]) * (1.0 - psf.halo) / (2.0 * math.pi * psf.sx * psf.sy)
            out.append(Spot(float(x), float(y), float(cu), float(cv), peak, amp,
                            float(fit.flux[k]), float(sig[k]), width, "ion", blended=True))
    # a patch reaching into a neighbouring group may have refitted one of its ions
    out.sort(key=lambda sp: (not sp.blended, -sp.snr))
    final: List[Spot] = []
    for sp in out:
        if all(math.hypot((sp.u - q.u) / psf.sx, (sp.v - q.v) / psf.sy) >= 1.0 for q in final):
            final.append(sp)
    final.sort(key=lambda sp: -sp.peak)
    return final, blobs


def _drop_squeezed(fit, fit_pruned, better):
    """A component squeezed between neighbours (much closer than the typical
    spacing) usually shares their light without being an ion: remove it if the
    fit without it is not significantly worse."""
    for _ in range(len(fit.flux) - 2):
        pos = fit.positions
        d = np.hypot(pos[:, None, 0] - pos[None, :, 0], pos[:, None, 1] - pos[None, :, 1])
        np.fill_diagonal(d, np.inf)
        nn = d.min(axis=1)
        typical = float(np.median(nn))
        close = np.nonzero(nn < 0.75 * typical)[0]
        if close.size == 0:
            return fit
        weakest = close[np.argmin(fit.flux[close])]
        trial = fit_pruned(np.delete(pos, weakest, axis=0))
        if trial is None or better(trial, fit):
            return fit
        fit = trial
    return fit


def _chain_search(patch, fit, psf: PSF, fit_pruned, better, min_flux: float = 0.25,
                  var=None):
    """Try linear chains of N = n, n+1, n+2 ions (equilibrium positions) as
    starting points; keep whatever explains the patch best (every ion must carry
    at least ``min_flux`` of the median flux)."""
    from ioncam.analysis.chain import equilibrium_positions
    from ioncam.analysis.fitting import fit_chain_psf
    pos, fl = fit.positions, fit.flux
    c = (pos * fl[:, None]).sum(axis=0) / max(fl.sum(), 1e-12)
    d = pos - c
    cov = (d * fl[:, None]).T @ d
    evals, evecs = np.linalg.eigh(cov)
    ax = evecs[:, 1]
    if ax[0] < 0:
        ax = -ax
    t = d @ ax
    if np.ptp(t) < 1e-6:
        return fit
    angle = math.atan2(ax[1], ax[0])
    best = fit
    n0 = len(fl)
    for n in (n0, n0 + 1, n0 + 2):
        u = equilibrium_positions(n)
        scale = max(abs(t.min()), abs(t.max())) / max(u[-1], 1e-6)
        try:
            ch = fit_chain_psf(patch, n, c, scale, angle, psf, var=var)
        except (np.linalg.LinAlgError, ValueError):  # pragma: no cover
            continue
        spacing = np.min(np.diff(np.sort((ch.positions - c) @ ax))) if n > 1 else 1.0
        trial = fit_pruned(ch.positions, move=max(0.35 * spacing, 0.3))
        if trial is None or (len(trial.flux) > len(best.flux) and
                             trial.flux.min() < min_flux * np.median(trial.flux)):
            continue
        if better(best, trial) if len(trial.flux) > len(best.flux) else trial.cost < best.cost:
            best = trial
    return best


def _diffuse_clouds(res: np.ndarray, spots: List[Spot], clouds: List[Spot], psf: PSF,
                    frame: Frame, origin, crop, search, thr: float, tile: int):
    """Hot, uncrystallised ions spread their (Doppler-reduced) light over many
    pixels: invisible to the ion-sized matched filter, so the light that the
    ions do not explain is filtered once more at cloud scale (on a 2x2-binned
    image: clouds are broad, and this keeps it cheap).  Returns the clouds
    found in this frame and ``(map, noise map, sigma, binning)`` (binned crop
    pixels) for temporal integration (a faint cloud only shows up over several
    frames)."""
    u0, v0 = origin
    h, w = res.shape
    r2 = res.astype(np.float32, copy=True)
    s_max = max(psf.sx, psf.sy)
    m = int(math.ceil(4.0 * max(s_max, psf.hx if psf.halo else 0.0))) + 1
    for sp in spots:                       # remove the ions' own light
        pu, pv = sp.u - u0, sp.v - v0
        ua, ub = int(max(0, math.floor(pu) - m)), int(min(w, math.ceil(pu) + m + 1))
        va, vb = int(max(0, math.floor(pv) - m)), int(min(h, math.ceil(pv) + m + 1))
        if ua >= ub or va >= vb:
            continue
        img = psf.images(ub - ua, vb - va, np.array([pu - ua]), np.array([pv - va]))[0]
        r2[va:vb, ua:ub] -= np.float32(sp.flux) * img.astype(np.float32)
    nb = 4 if min(h, w) >= 192 else 2      # clouds are broad: coarse pixels are fine
    hb2, wb2 = h // nb, w // nb
    if hb2 < 4 or wb2 < 4:
        return [], None
    r2b = r2[:nb * hb2, :nb * wb2].reshape(hb2, nb, wb2, nb).sum(axis=(1, 3))
    sc = max(2.5 * max(s_max, 1.0) / nb, 1.25)  # cloud-scale filter sigma [binned px]
    f2 = _gaussian(r2b, sc, sc)
    tile_b = max(16, tile // nb, int(8 * sc))
    noise = filtered_noise_scale(f2, tile_b, (sc * 1.2, sc * 1.2), filled_fallback=False)
    # The noise is far from uniform: at high EM gain the photon noise of the
    # laser scatter dominates, several times the noise beside the beam.  Scale
    # it by the tile-wise MAD (insensitive to the clouds themselves), never
    # much below the global scale (estimation noise of a tile).
    rel = block_noise(f2, tile_b)
    rel /= max(float(np.median(rel)), 1e-12)
    noise_map = (noise * np.clip(rel, 0.85, 4.0)).astype(np.float32)
    mask = f2 > thr * noise_map
    if search != crop:
        inside = np.zeros_like(mask)
        inside[(search[1] - v0) // nb:(search[3] - v0 + nb - 1) // nb,
               (search[0] - u0) // nb:(search[2] - u0 + nb - 1) // nb] = True
        mask &= inside
    out: List[Spot] = []
    taken = [((c.u - u0) / nb, (c.v - v0) / nb, c.width / max(frame.hbin, frame.vbin) / nb)
             for c in clouds]
    idx = _local_maxima(f2, mask)
    for i in idx[np.argsort(-f2.ravel()[idx])][:5]:
        iv, iu = divmod(int(i), wb2)
        if any(math.hypot(iu - a, iv - b) < max(2.0 * sc, r) for a, b, r in taken):
            continue
        # spread of the excess light around the peak -> width
        rr = int(math.ceil(3 * sc))
        ua, ub = max(0, iu - rr), min(wb2, iu + rr + 1)
        va, vb = max(0, iv - rr), min(hb2, iv + rr + 1)
        win = np.clip(r2b[va:vb, ua:ub], 0, None)
        tot = float(win.sum())
        if tot <= 0 or float(win.max()) > 0.35 * tot:
            continue                       # nothing, or one amplified photon / cosmic ray
        yy, xx = np.mgrid[va:vb, ua:ub]
        mx, my = float((win * xx).sum() / tot), float((win * yy).sum() / tot)
        var = float((win * ((xx - mx) ** 2 + (yy - my) ** 2)).sum() / tot) / 2.0
        w_frame = nb * math.sqrt(max(var, sc * sc))
        width = w_frame * math.sqrt(frame.hbin * frame.vbin)
        # position: centroid of the excess light (the peak is quantised to the
        # coarse pixels)
        cu, cv = u0 + nb * mx + 0.5 * (nb - 1), v0 + nb * my + 0.5 * (nb - 1)
        x, y = frame.to_sensor(cu, cv)
        peak = float(f2[iv, iu])
        # amplitude: surface brightness [counts / frame px], comparable with ions'
        amp = tot / (2.0 * math.pi * w_frame * w_frame)
        out.append(Spot(float(x), float(y), float(cu), float(cv), peak, amp, tot,
                        peak / max(float(noise_map[iv, iu]), 1e-12), width, "cloud"))
        taken.append((iu, iv, width / max(frame.hbin, frame.vbin) / nb))
    return out, (f2, noise_map, sc, nb)


def _merge_clouds(clouds: List[Spot], spots: List[Spot]):
    """Merge fragments of one cloud and drop 'ions' that are lumps inside it:
    shot-noise lumps on a cloud are about as bright (per pixel) as the cloud
    itself, a crystallised ion next to or in front of a cloud is many times
    brighter (its light is concentrated in the PSF)."""
    merged: List[Spot] = []
    for c in sorted(clouds, key=lambda s: -s.snr):
        if all(math.hypot(c.x - m.x, c.y - m.y) > 2.0 * max(c.width, m.width) for m in merged):
            merged.append(c)
    kept = [s for s in spots
            if all(math.hypot(s.x - c.x, s.y - c.y) > 3.0 * c.width or
                   s.amplitude > 4.0 * c.amplitude for c in merged)]
    return merged, kept


class StaticBackground:
    """Learned static structure of the background: laser light scattered by
    surface defects, dust, scratches and electrode edges (in focus they look
    like ions, out of focus like clouds), the fine structure of the scatter
    profile, hot pixels.

    Stored in *sensor* coordinates per sensor pixel and per unit gain factor
    (exposure x EM gain), so it applies to any later ROI, binning, exposure
    and gain.  Only significant structure is kept (subtracting the noise of
    the learning frames would add noise to every frame).  ``scale`` follows
    the laser power (and the EM gain calibration): it is refined on every
    frame from the structure left in the residual."""

    def __init__(self, image: np.ndarray, region, info: Optional[dict] = None,
                 pixel_noise: float = 0.0):
        self.image = np.asarray(image, np.float32)      # counts / sensor px / gain factor
        self.region = tuple(int(v) for v in region)     # (x0, y0, x1, y1) sensor px
        self.info = dict(info or {})
        self.pixel_noise = float(pixel_noise)           # noise of the map per sensor px
        self.scale = 1.0
        self._cache: dict = {}

    def for_frame(self, frame: Frame) -> Optional[np.ndarray]:
        """The map in the frame's pixels (binned sums; zero where the frame is
        outside the learned region), at gain factor 1 and scale 1; None if the
        frame does not overlap it."""
        x0, y0, x1, y1 = frame.applied.roi
        hb, vb = frame.hbin, frame.vbin
        h, w = frame.data.shape
        key = (x0, y0, hb, vb, h, w)
        m = self._cache.get(key)
        if m is not None or key in self._cache:
            return m
        rx0, ry0, rx1, ry1 = self.region
        sx1, sy1 = x0 + w * hb, y0 + h * vb
        ix0, iy0, ix1, iy1 = max(x0, rx0), max(y0, ry0), min(sx1, rx1), min(sy1, ry1)
        if ix0 >= ix1 or iy0 >= iy1:
            m = None
        else:
            full = np.zeros((h * vb, w * hb), np.float32)
            full[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = \
                self.image[iy0 - ry0:iy1 - ry0, ix0 - rx0:ix1 - rx0]
            m = full.reshape(h, vb, w, hb).sum(axis=(1, 3)) if hb * vb > 1 else full
            if not np.any(m):
                m = None
        if len(self._cache) > 16:
            self._cache.clear()
        self._cache[key] = m
        return m

    def bandpass_for_frame(self, frame: Frame, s1: Tuple[float, float],
                           s2: Tuple[float, float]) -> Optional[np.ndarray]:
        """:meth:`for_frame` band-passed (difference of Gaussians ``s1`` - ``s2``,
        (sy, sx) frame px) - the structure the scale is fitted on."""
        m = self.for_frame(frame)
        if m is None:
            return None
        key = ("bandpass", frame.x0, frame.y0, frame.hbin, frame.vbin, m.shape, s1, s2)
        out = self._cache.get(key)
        if out is None:
            out = _bandpass(m, s1, s2)
            self._cache[key] = out
        return out

    def summary(self) -> dict:
        nz = self.image != 0
        return dict(self.info, region=list(self.region), scale=round(float(self.scale), 4),
                    structure_px=int(nz.sum()))

    # (kept for callers of the old API)
    def matches(self, frame: Frame) -> bool:
        return self.for_frame(frame) is not None


class Detector:
    """Per-frame ion detector (configuration: :class:`ioncam.config.DetectionConfig`)."""

    filter_scale = 0.65     # matched-filter sigma / PSF sigma

    def __init__(self, cfg):
        self.cfg = cfg
        self.static_background: Optional[StaticBackground] = None

    def analyze(self, frame: Frame, analysis_roi: Optional[Sequence[int]] = None,
                threshold_sigma: Optional[float] = None,
                prior: Optional[np.ndarray] = None) -> DetectionResult:
        """Find ions and clouds in ``frame``.  ``prior`` (sensor px) are positions
        where bright ions are expected (the tracked crystal), used as starting
        points of the crowded-field fits."""
        t_start = time.perf_counter()
        cfg = self.cfg
        img = frame.data
        hb, vb = frame.hbin, frame.vbin
        hgt, wid = img.shape
        crop = (0, 0, wid, hgt)
        roi = analysis_roi if analysis_roi is not None else cfg.analysis_roi
        if roi is not None:
            ua, va = frame.from_sensor(roi[0], roi[1])
            ub, vbb = frame.from_sensor(roi[2] - 1, roi[3] - 1)
            crop = (int(max(0, math.floor(ua))), int(max(0, math.floor(va))),
                    int(min(wid, math.ceil(ub) + 1)), int(min(hgt, math.ceil(vbb) + 1)))
            if crop[2] <= crop[0] or crop[3] <= crop[1]:
                crop = (0, 0, 0, 0)
        block_px = max(4, int(round(cfg.background_block / max(hb, vb))))
        # ions are searched for inside `search`; background/noise statistics use
        # some context around it (small analysis ROIs would otherwise be dominated
        # by the ions themselves)
        search = crop
        if roi is not None and crop[2] > crop[0]:
            m = max(16, block_px)
            crop = (max(0, search[0] - m), max(0, search[1] - m),
                    min(wid, search[2] + m), min(hgt, search[3] + m))
        u0, v0, u1, v1 = crop
        raw = img[v0:v1, u0:u1]
        saturated = int(np.count_nonzero(img >= 65000))
        max_value = int(img.max()) if img.size else 0
        gain_factor = frame.applied.gain_factor
        psf = cfg.psf_sigma
        sy = max(0.5, self.filter_scale * psf / vb)
        sx = max(0.5, self.filter_scale * psf / hb)
        thr_sigma = cfg.threshold_sigma if threshold_sigma is None else threshold_sigma
        if raw.size == 0:
            z = np.zeros((0, 0), np.float32)
            return DetectionResult(frame.index, frame.timestamp, [], [], 0.0, 0.0, 0.0, z, z,
                                   crop, frame.x0, frame.y0, hb, vb, (sy, sx), gain_factor,
                                   saturated, max_value)

        sub = raw.astype(np.float32)
        sb = self.static_background
        static = None
        if sb is not None:
            m = sb.for_frame(frame)
            if m is not None:
                static = m[v0:v1, u0:u1] * np.float32(gain_factor)
                sub -= static * np.float32(sb.scale)
        # small frames (tight camera ROI): at least two background tiles per
        # axis, so a scattered-light gradient is modelled instead of left in
        # the residual (where it would inflate the noise scale)
        floor = max(4, int(round(16 / max(hb, vb))))
        bg_block = min(block_px, max(floor, min(sub.shape) // 2))
        bg = block_background(sub, bg_block)
        res = sub - bg
        filt = _gaussian(res, sy, sx)
        psf_x = math.sqrt((psf / hb) ** 2 + 1.0 / 12.0)   # source sigma, frame px
        psf_y = math.sqrt((psf / vb) ** 2 + 1.0 / 12.0)
        tpx = math.sqrt(psf_x ** 2 + sx ** 2)              # spot sigma in `filt`
        tpy = math.sqrt(psf_y ** 2 + sy ** 2)
        # Noise of the filtered image: the scale from the upper tail with
        # bright objects excluded (see filtered_noise_scale), the spatial
        # variation (scatter blobs, gradients) from tile-wise MAD.
        tile = max(2 * block_px, 32)
        sigma_up = filtered_noise_scale(filt, tile, (tpx, tpy))
        if sub.size <= 128 * 128:
            # A crystal filling a good part of a small frame biases the tile
            # medians of the background, and with them the residual and its
            # noise: redo the background without the ions.  (In large frames
            # the bias is diluted to a fraction of the noise.)
            reach = (2 * int(math.ceil(1.5 * tpy + 1)) + 1, 2 * int(math.ceil(1.5 * tpx + 1)) + 1)
            objects = ndimage.maximum_filter(filt > 4.0 * sigma_up, size=reach, mode="constant")
            if _tiles(*_subsample(objects, bg_block)).mean(axis=2).max() > 0.1:
                bg = block_background(sub, bg_block, exclude=objects)
                res = sub - bg
                filt = _gaussian(res, sy, sx)
                sigma_up = filtered_noise_scale(filt, tile, (tpx, tpy))
        background = float(np.median(bg))
        noise = mad_std(res[::4, ::4] if res.size > 20000 else res)
        mad_map = block_noise(filt, tile)
        rel = mad_map / max(float(np.median(mad_map)), 1e-9)
        # the map's job is to raise the threshold on bright scatter; never let
        # estimation noise in a tile lower it much
        noise_f_map = sigma_up * np.clip(rel, 0.85, 4.0)
        noise_f = sigma_up

        snr_map = filt / noise_f_map
        mask = snr_map > thr_sigma
        if search != crop:
            inside = np.zeros_like(mask)
            inside[search[1] - v0:search[3] - v0, search[0] - u0:search[2] - u0] = True
            mask &= inside
        cand = _local_maxima(filt, mask)
        spots: List[Spot] = []
        clouds: List[Spot] = []
        spikes = 0
        # non-maximum suppression distance in frame pixels
        sep_u = cfg.min_separation / hb
        sep_v = cfg.min_separation / vb
        if cand.size:
            cand = cand[np.argsort(filt.ravel()[cand])[::-1][: 4 * cfg.max_spots]]
            resolved = min(psf_x, psf_y) >= 0.7               # can widths be measured?
            spike_limit = max(4.0, 0.6 * 2.0 * math.pi * sx * sy)
            # broadness probe: ring at 2.5 point-source widths of the filtered image
            ring = _RING * np.array([2.5 * tpx, 2.5 * tpy])
            k = cfg.cloud_width_factor
            cloud_ratio = 0.85 * math.exp(-3.125 / (k * k))   # 0.85: noise-boosted peaks
            taken: List[Tuple[float, float]] = []
            for idx in cand:
                iv, iu = divmod(int(idx), filt.shape[1])
                peak = float(filt[iv, iu])
                du, dv = _parabola_offsets(filt, iv, iu)
                pu, pv = iu + du, iv + dv
                # suppress weaker maxima closer than min_separation to a stronger one
                if any(((pu - a) / sep_u) ** 2 + ((pv - b) / sep_v) ** 2 < 1.0
                       for a, b in taken):
                    continue
                probe = ndimage.map_coordinates(filt, [pv + ring[:, 1], pu + ring[:, 0]],
                                                order=1, mode="nearest")
                # 3rd smallest: neighbours in a chain / 2D crystal hit at most a
                # few ring directions, a cloud is broad in all of them
                broad = float(np.sort(probe)[2]) / peak if peak > 0 else 0.0
                vx, vy = _log_variances(filt, iv, iu)
                tx2 = vx if vx is not None else tpx ** 2
                ty2 = vy if vy is not None else tpy ** 2
                if resolved:
                    sharp = float(res[iv, iu]) / peak if peak > 0 else 0.0
                    narrow = min((tx2 - sx * sx) / psf_x ** 2, (ty2 - sy * sy) / psf_y ** 2)
                    if sharp > spike_limit or narrow < 0.45 ** 2:
                        spikes += 1                # cosmic ray / hot pixel / CIC spike
                        continue
                taken.append((pu, pv))
                kind = "ion"
                if broad > cloud_ratio:
                    kind = "cloud"
                    # width from the ring ratio: broad = exp(-r^2 / (2 t^2))
                    t2x = (2.5 * tpx) ** 2 / (2.0 * math.log(1.0 / max(broad, 1e-6)))
                    t2y = (2.5 * tpy) ** 2 / (2.0 * math.log(1.0 / max(broad, 1e-6)))
                    tx2, ty2 = max(tx2, t2x), max(ty2, t2y)
                srcx = math.sqrt(max(tx2 - sx * sx, 1e-4))
                srcy = math.sqrt(max(ty2 - sy * sy, 1e-4))
                peak_true = peak * math.exp(0.5 * (du * du / tx2 + dv * dv / ty2))
                flux = 2.0 * math.pi * math.sqrt(tx2 * ty2) * peak_true
                amp = flux / (2.0 * math.pi * srcx * srcy)
                cu, cv = u0 + pu, v0 + pv
                x, y = frame.to_sensor(cu, cv)
                spot = Spot(float(x), float(y), float(cu), float(cv), peak, amp, flux,
                            float(snr_map[iv, iu]), math.sqrt(srcx * hb * srcy * vb), kind)
                (clouds if kind == "cloud" else spots).append(spot)
                if len(spots) >= cfg.max_spots:
                    break
        if clouds:
            clouds, spots = _merge_clouds(clouds, spots)
        psf_model = PSF.from_sensor(psf, cfg.psf_halo_fraction, cfg.psf_halo_sigma, hb, vb)
        prior_crop = None
        if prior is not None and len(prior):
            pu, pv = frame.from_sensor(np.asarray(prior)[:, 0], np.asarray(prior)[:, 1])
            prior_crop = np.column_stack([np.atleast_1d(pu) - u0, np.atleast_1d(pv) - v0])
            hh, ww = filt.shape
            prior_crop = prior_crop[(prior_crop[:, 0] > -0.5) & (prior_crop[:, 0] < ww - 0.5) &
                                    (prior_crop[:, 1] > -0.5) & (prior_crop[:, 1] < hh - 0.5)]
        if cfg.deblend and (len(spots) >= 2 or (prior_crop is not None and len(prior_crop) >= 2)):
            search_mask = None
            if search != crop:
                search_mask = np.zeros(filt.shape, bool)
                search_mask[search[1] - v0:search[3] - v0, search[0] - u0:search[2] - u0] = True
            spots, blobs = _deblend(spots, res, filt, noise_f_map, thr_sigma, psf_model,
                                    (sy, sx), frame, (u0, v0), search_mask, cfg.deblend_max,
                                    prior_crop, cfg.cloud_width_factor)
            clouds = clouds + blobs
        diffuse = None
        if cfg.cloud_threshold_sigma > 0 and res.shape[0] >= 8 and res.shape[1] >= 8:
            more, diffuse = _diffuse_clouds(res, spots, clouds, psf_model, frame, (u0, v0),
                                            crop, search, cfg.cloud_threshold_sigma, tile)
            clouds = clouds + more
        if clouds:
            # (again: the deblending may have re-introduced lumps of a cloud)
            clouds, spots = _merge_clouds(clouds, spots)
        if static is not None and cfg.static_scale_tracking:
            self._refine_static_scale(sb, frame, sub, spots + clouds, (u0, v0, u1, v1),
                                      (sy, sx), gain_factor)
        widths = [s.width for s in spots if s.snr > 2 * thr_sigma]
        focus = float(np.median(widths)) if widths else float("nan")
        return DetectionResult(
            frame.index, frame.timestamp, spots, clouds, background, noise, noise_f,
            filt, res, crop, frame.x0, frame.y0, hb, vb, (sy, sx), gain_factor,
            saturated, max_value, focus, spikes, noise_f_map,
            time.perf_counter() - t_start, psf_model, diffuse,
            bool(cfg.deblend and prior_crop is not None and len(prior_crop) >= 1))

    # -- static background ----------------------------------------------------------------

    def _refine_static_scale(self, sb: StaticBackground, frame: Frame, sub: np.ndarray,
                             objects: List[Spot], crop, sf: Tuple[float, float],
                             gain_factor: float) -> None:
        """Follow the scale of the static light (laser power, EM gain
        calibration) from the structure left in (or over-subtracted from) the
        frame ``sub`` (already corrected with the current scale), slowly
        averaged over frames.  Fitted on a linear band-pass of frame and map
        (the median-based background is biased by skewed EMCCD noise, more in
        bright structure), away from ions, clouds and the frame edges, where
        the structure is well above the noise of the frame and of the map."""
        u0, v0, u1, v1 = crop
        hb, vb = frame.hbin, frame.vbin
        s1 = (float(sf[0]), float(sf[1]))
        s2 = (2.5 * s1[0], 2.5 * s1[1])
        hm = sb.bandpass_for_frame(frame, s1, s2)
        if hm is None:
            return
        hm = hm[v0:v1, u0:u1] * np.float32(gain_factor)
        hs = _bandpass(sub, s1, s2)
        h, w = hs.shape
        noise_h = mad_std(hs[::2, ::2] if hs.size > 20000 else hs)
        # noise of the band-passed map: pixel noise x |kernel| (isotropic approx.)
        a2, b2 = s1[0] * s1[1], s2[0] * s2[1]
        knorm = math.sqrt(max(1.0 / (4 * math.pi * a2) + 1.0 / (4 * math.pi * b2)
                              - 1.0 / (math.pi * (a2 + b2)), 1e-12))
        map_noise = sb.pixel_noise * math.sqrt(hb * vb) * gain_factor * knorm
        sel = np.abs(hm) > max(3.0 * noise_h, 8.0 * map_noise)
        e = int(math.ceil(3.0 * max(s2)))
        sel[:e, :] = False
        sel[h - e:, :] = False
        sel[:, :e] = False
        sel[:, w - e:] = False
        r_obj = 3.0 * max(s2[0] * vb, s2[1] * hb) + 2.0
        for o in objects:
            rr = max(r_obj, 2.5 * o.width)
            cu, cv = o.u - u0, o.v - v0
            ua, ub = int(max(0, cu - rr / hb)), int(min(w, cu + rr / hb + 1))
            va, vb_ = int(max(0, cv - rr / vb)), int(min(h, cv + rr / vb + 1))
            sel[va:vb_, ua:ub] = False
        if np.count_nonzero(sel) < 12:
            return
        x = hm[sel].astype(np.float64)
        d = float(np.dot(x, hs[sel]) / max(float(np.dot(x, x)), 1e-12))
        sb.scale = float(np.clip(sb.scale + 0.3 * np.clip(d, -1.0, 1.0), 0.0, 5.0))

    def learn_static_background(self, frames: Sequence[Frame],
                                mask_positions: Optional[np.ndarray] = None,
                                mask_radius: Optional[float] = None,
                                info: Optional[dict] = None) -> StaticBackground:
        """Learn the static background structure from ``frames`` (one geometry;
        best the full sensor without binning and many photons: many or long
        exposures, lasers on).

        Positions in ``mask_positions`` (sensor px: the ions) are left out with
        ``mask_radius`` (default: PSF core and halo), so the ions may stay in
        the trap; with the repumper off the ions are dark while the laser
        still scatters - then nothing needs masking.  Structure is kept where
        it is significant (at the PSF scale or as single hot pixels)."""
        if not frames:
            raise ValueError("no frames")
        f0 = frames[0]
        hb, vb = f0.hbin, f0.vbin
        cube = np.stack([f.data for f in frames]).astype(np.float32)
        stack = cube.mean(axis=0)
        # hot pixels are high in every frame; one amplified photon (EMCCD) in
        # one frame lifts a pixel of the mean as much - the median is not fooled
        med = np.median(cube, axis=0) if len(frames) >= 3 else stack
        del cube
        block = max(4, int(round(self.cfg.background_block / max(hb, vb))))
        c = self.cfg
        pos = np.zeros((0, 2)) if mask_positions is None else \
            np.asarray(mask_positions, float).reshape(-1, 2)
        if len(pos):
            uc, vc = f0.from_sensor(pos[:, 0], pos[:, 1])
            uv = np.column_stack([np.atleast_1d(uc), np.atleast_1d(vc)])
            h, w = stack.shape
            uv = uv[(uv[:, 0] > -0.5) & (uv[:, 0] < w - 0.5) &
                    (uv[:, 1] > -0.5) & (uv[:, 1] < h - 0.5)]
            if len(uv):
                # the ions' light - wings (halo) included - is not background:
                # fit and remove it, then only the cores need masking
                psf = PSF.from_sensor(c.psf_sigma, c.psf_halo_fraction, c.psf_halo_sigma,
                                      hb, vb)
                reach = 4.0 * max(psf.sx, psf.sy, psf.hx if psf.halo else 0.0,
                                  psf.hy if psf.halo else 0.0) + 2.0
                ua, ub = int(max(0, uv[:, 0].min() - reach)), int(min(w, uv[:, 0].max() + reach + 1))
                va, vb_ = int(max(0, uv[:, 1].min() - reach)), int(min(h, uv[:, 1].max() + reach + 1))
                near = np.zeros(stack.shape, bool)
                near[va:vb_, ua:ub] = True
                patch = stack[va:vb_, ua:ub] - block_background(stack, block, exclude=near)[va:vb_, ua:ub]
                _, _, bgp, model = linear_photometry(patch, uv - [ua, va], psf)
                stack[va:vb_, ua:ub] -= (model - bgp).astype(np.float32)
        # all the static light (above the bias / uniform level, which do not
        # scale with exposure x gain - the per-frame background takes them):
        # smooth part everywhere (its noise averages away), fine structure
        # where it is significant (at the PSF scale - defects, edges - or per
        # pixel - hot pixels); the rest of the fine part is noise of the mean
        light = stack - float(np.min(block_background(stack, block)))
        sx = max(0.5, c.psf_sigma / hb)
        sy = max(0.5, c.psf_sigma / vb)
        s_smooth = (max(1.0, 3.0 / vb), max(1.0, 3.0 / hb))
        grow = (2 * int(math.ceil(2.0 * sy)) + 1, 2 * int(math.ceil(2.0 * sx)) + 1)
        smooth = ndimage.gaussian_filter(light, s_smooth, mode="nearest", truncate=3.0)
        keep = np.zeros(light.shape, bool)
        for _ in range(3):
            # the smooth part must not contain the sharp structure (a smeared
            # copy around every feature would be subtracted from the frames):
            # it is re-made with the significant structure left out
            if keep.any():
                smooth = ndimage.gaussian_filter(np.where(keep, smooth, light), s_smooth,
                                                 mode="nearest", truncate=3.0)
            fine = light - smooth
            noise = mad_std(fine[::2, ::2] if fine.size > 20000 else fine)
            fs = _gaussian(fine, sy, sx)
            fs_noise = mad_std(fs[::2, ::2] if fs.size > 20000 else fs)
            keep = ndimage.maximum_filter(np.abs(fs) > 3.0 * fs_noise, size=grow,
                                          mode="constant")
        # single pixels far above their neighbourhood: hot pixels if the
        # per-frame median agrees, else one amplified photon in one frame
        # (EMCCD) - those must not be learned (they would be subtracted from
        # every later frame)
        base = float(np.min(block_background(stack, block)))
        ring = np.ones((3, 3), bool)
        ring[1, 1] = False
        local = ndimage.median_filter(light, footprint=ring, mode="nearest")
        excess = light - local
        med_light = med - base
        med_excess = med_light - ndimage.median_filter(med_light, footprint=ring, mode="nearest")
        high = excess > 5.0 * noise
        steady = med_excess > 0.5 * excess          # in most frames: real (hot pixel, sharp defect)
        light = np.where(high & ~steady, local, light)
        keep |= high & steady
        smooth = smooth.astype(np.float32)
        fine = np.where(keep, light, smooth).astype(np.float32)
        n_masked = 0
        if len(pos):
            if mask_radius is None:
                mask_radius = 3.0 * c.psf_sigma      # cores (the model is least exact there)
            h, w = fine.shape
            uc, vc = f0.from_sensor(pos[:, 0], pos[:, 1])
            r_u, r_v = mask_radius / hb + 1.0, mask_radius / vb + 1.0
            for a, b in zip(np.atleast_1d(uc), np.atleast_1d(vc)):
                ua, ub = int(max(0, math.floor(a - r_u))), int(min(w, math.ceil(a + r_u) + 1))
                va, vb_ = int(max(0, math.floor(b - r_v))), int(min(h, math.ceil(b + r_v) + 1))
                if ua >= ub or va >= vb_:
                    continue
                vv, uu = np.mgrid[va:vb_, ua:ub]
                inside = ((uu - a) / r_u) ** 2 + ((vv - b) / r_v) ** 2 <= 1.0
                fine[va:vb_, ua:ub][inside] = smooth[va:vb_, ua:ub][inside]
                n_masked += 1
        # per sensor pixel and unit gain factor (a binned pixel's structure is
        # spread evenly over its sensor pixels)
        per_px = fine / np.float32(hb * vb * max(f0.applied.gain_factor, 1e-12))
        image = np.repeat(np.repeat(per_px, vb, axis=0), hb, axis=1)
        x0, y0 = f0.x0, f0.y0
        region = (x0, y0, x0 + image.shape[1], y0 + image.shape[0])
        meta = {"n_frames": len(frames), "exposure": float(f0.applied.exposure),
                "em_gain": int(f0.applied.em_gain), "binning": [hb, vb],
                "masked_sites": n_masked, "created": time.time(),
                "noise_of_mean": float(noise)}
        meta.update(info or {})
        sb = StaticBackground(image, region, meta,
                              pixel_noise=noise / (hb * vb * max(f0.applied.gain_factor, 1e-12)))
        self.static_background = sb
        return sb

    def clear_static_background(self) -> None:
        self.static_background = None
