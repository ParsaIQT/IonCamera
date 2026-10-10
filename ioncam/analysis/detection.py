"""Fast, robust ion spot detection for EMCCD frames.

Pipeline per frame (float32; ~10 ms for a full 512x512 frame, ~1 ms for a
typical analysis ROI):

1. optional subtraction of a learned *static* background structure (fine
   scatter features, hot pixels), scaled with exposure x gain;
2. smooth background from tile medians, bilinearly interpolated with cached
   separable interpolation matrices (follows drifting scatter and blobs);
3. a *noise map* from tile-wise MAD of the residual: EMCCD noise grows with
   the background, so a single global noise figure would under-threshold
   bright scatter regions;
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

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "amplitude": self.amplitude, "flux": self.flux,
                "snr": self.snr, "width": self.width, "kind": self.kind}


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

    @property
    def positions(self) -> np.ndarray:
        return np.array([[s.x, s.y] for s in self.spots], float).reshape(-1, 2)


def mad_std(a: np.ndarray) -> float:
    a = np.asarray(a, np.float32).ravel()
    if a.size == 0:
        return 0.0
    med = np.median(a)
    return float(1.4826 * np.median(np.abs(a - med)))


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
    h, w = img.shape
    nby, nbx = max(1, h // block), max(1, w // block)
    by, bx = h // nby, w // nbx
    core = img[:nby * by, :nbx * bx]
    return core.reshape(nby, by, nbx, bx).transpose(0, 2, 1, 3).reshape(nby, nbx, by * bx)


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


def block_background(img: np.ndarray, block: int) -> np.ndarray:
    """Smooth background: medians of ~``block``-sized tiles, bilinearly interpolated."""
    sub, b = _subsample(img, block)
    return _expand(np.median(_tiles(sub, b), axis=2), img.shape)


def block_noise(res: np.ndarray, block: int) -> np.ndarray:
    """Noise map: robust (MAD) standard deviation per tile, interpolated."""
    sub, b = _subsample(res, block)
    t = _tiles(sub, b)
    med = np.median(t, axis=2, keepdims=True)
    return _expand(1.4826 * np.median(np.abs(t - med), axis=2), res.shape)


def _gaussian(img: np.ndarray, sy: float, sx: float) -> np.ndarray:
    """Gaussian filter with zero padding (OpenCV if available - ~5x faster)."""
    if _cv2 is not None:
        kx = 2 * int(math.ceil(3.5 * sx)) + 1
        ky = 2 * int(math.ceil(3.5 * sy)) + 1
        return _cv2.GaussianBlur(img, (kx, ky), sigmaX=sx, sigmaY=sy,
                                 borderType=_cv2.BORDER_CONSTANT)
    return ndimage.gaussian_filter(img, sigma=(sy, sx), mode="constant", truncate=3.5)


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


def _merge_clouds(clouds: List[Spot], spots: List[Spot]):
    """Merge fragments of one cloud and drop 'ions' that are lumps inside it."""
    merged: List[Spot] = []
    for c in sorted(clouds, key=lambda s: -s.peak):
        if all(math.hypot(c.x - m.x, c.y - m.y) > max(c.width, m.width) for m in merged):
            merged.append(c)
    kept = [s for s in spots
            if all(math.hypot(s.x - c.x, s.y - c.y) > c.width or s.peak > c.peak
                   for c in merged)]
    return merged, kept


class StaticBackground:
    """Learned fine structure of the background (scatter features, hot pixels)."""

    def __init__(self, image: np.ndarray, roi, hbin: int, vbin: int, gain_factor: float):
        self.image = image.astype(np.float32)
        self.roi = tuple(roi)
        self.hbin, self.vbin = hbin, vbin
        self.gain_factor = gain_factor

    def matches(self, frame: Frame) -> bool:
        return (tuple(frame.applied.roi) == self.roi and frame.hbin == self.hbin
                and frame.vbin == self.vbin and frame.data.shape == self.image.shape)


class Detector:
    """Per-frame ion detector (configuration: :class:`ioncam.config.DetectionConfig`)."""

    filter_scale = 0.65     # matched-filter sigma / PSF sigma

    def __init__(self, cfg):
        self.cfg = cfg
        self.static_background: Optional[StaticBackground] = None

    def analyze(self, frame: Frame, analysis_roi: Optional[Sequence[int]] = None,
                threshold_sigma: Optional[float] = None) -> DetectionResult:
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
        if sb is not None and sb.matches(frame):
            sub -= sb.image[v0:v1, u0:u1] * (gain_factor / max(sb.gain_factor, 1e-12))
        bg = block_background(sub, block_px)
        res = sub - bg
        background = float(np.median(bg))
        stride = 2 if res.size > 20000 else 1
        noise = mad_std(res[::2 * stride, ::2 * stride] if res.size > 20000 else res)

        filt = _gaussian(res, sy, sx)
        # Noise of the filtered image.  EMCCD noise is strongly right-skewed at
        # low light (single amplified photons), so the scale comes from the
        # upper quantile (exact for Gaussian noise, conservative for skewed
        # noise, insensitive to ions which cover << 0.5 % of the pixels); the
        # spatial variation (scatter blobs, gradients) from tile-wise MAD.
        # Tiles dominated by bright objects (clouds, chains) are identified by an
        # outlying per-tile scale and excluded; the quantile is then taken over
        # the pooled pixels of the clean tiles (unbiased, unlike a median of
        # per-tile extreme-quantile estimates).
        tile = max(2 * block_px, 32)
        sub_f, b_f = _subsample(filt, tile)
        tl = _tiles(sub_f, b_f)
        tq = np.quantile(tl, [0.5, 0.995], axis=2)
        ts = tq[1] - tq[0]
        clean = ts <= 2.0 * float(np.median(ts))
        pooled = tl[clean].ravel() if np.any(clean) else tl.ravel()
        q50, q995 = np.quantile(pooled, [0.5, 0.995])
        sigma_up = max(float(q995 - q50) / 2.576, mad_std(filt[::stride, ::stride]), 1e-6)
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
            psf_x = math.sqrt((psf / hb) ** 2 + 1.0 / 12.0)   # source sigma, frame px
            psf_y = math.sqrt((psf / vb) ** 2 + 1.0 / 12.0)
            resolved = min(psf_x, psf_y) >= 0.7               # can widths be measured?
            spike_limit = max(4.0, 0.6 * 2.0 * math.pi * sx * sy)
            # broadness probe: ring at 2.5 point-source widths of the filtered image
            tpx = math.sqrt(psf_x ** 2 + sx ** 2)
            tpy = math.sqrt(psf_y ** 2 + sy ** 2)
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
        widths = [s.width for s in spots if s.snr > 2 * thr_sigma]
        focus = float(np.median(widths)) if widths else float("nan")
        return DetectionResult(
            frame.index, frame.timestamp, spots, clouds, background, noise, noise_f,
            filt, res, crop, frame.x0, frame.y0, hb, vb, (sy, sx), gain_factor,
            saturated, max_value, focus, spikes, noise_f_map,
            time.perf_counter() - t_start)

    # -- static background ----------------------------------------------------------------

    def learn_static_background(self, frames: Sequence[Frame],
                                mask_positions: Optional[np.ndarray] = None,
                                mask_radius: float = 4.0) -> StaticBackground:
        """Learn the fine background structure from ``frames`` (same geometry).

        Ion sites in ``mask_positions`` (sensor coordinates) are excluded, so
        this can be done with ions present."""
        if not frames:
            raise ValueError("no frames")
        f0 = frames[0]
        stack = np.mean([f.data.astype(np.float32) for f in frames], axis=0)
        smooth = block_background(stack, max(4, int(round(self.cfg.background_block /
                                                          max(f0.hbin, f0.vbin)))))
        fine = stack - smooth
        # keep only significant *positive* structure (hot pixels, sharp scatter
        # features): subtracting the residual noise of the mean image - or its
        # negative outliers - would add noise and bumps
        level = 6.0 * mad_std(fine[::2, ::2] if fine.size > 20000 else fine)
        fine[fine < level] = 0.0
        if mask_positions is not None and len(mask_positions):
            h, w = fine.shape
            vv, uu = np.mgrid[0:h, 0:w]
            pos = np.asarray(mask_positions, float).reshape(-1, 2)
            uc, vc = f0.from_sensor(pos[:, 0], pos[:, 1])
            r_u = mask_radius / f0.hbin + 1.0
            r_v = mask_radius / f0.vbin + 1.0
            mask = np.zeros_like(fine, bool)
            for a, b in zip(np.atleast_1d(uc), np.atleast_1d(vc)):
                mask |= ((uu - a) / r_u) ** 2 + ((vv - b) / r_v) ** 2 <= 1.0
            fine[mask] = 0.0
        sb = StaticBackground(fine, f0.applied.roi, f0.hbin, f0.vbin, f0.applied.gain_factor)
        self.static_background = sb
        return sb

    def clear_static_background(self) -> None:
        self.static_background = None
