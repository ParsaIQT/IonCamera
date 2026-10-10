"""Precise joint fit of all ions (used for calibration, not on the live path).

Model: ``offset + sum_i F_i * Ey_i(v) * Ex_i(u)`` where ``Ex_i(u)`` is the
fraction of a Gaussian PSF (shared ``sigma_x``, ``sigma_y``) centred at
``x_i`` that falls into pixel ``u`` (pixel-integrated, i.e. exact for
pixels comparable to the PSF).  All ions are fitted simultaneously, so
overlapping PSFs in dense chains do not bias each other.  Analytic Jacobian,
``scipy.optimize.least_squares`` (trust region reflective, bounded).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr

_SQRT2PI = math.sqrt(2.0 * math.pi)


@dataclass
class GaussFit:
    positions: np.ndarray          # (N, 2) image coordinates (col, row) of the fitted region
    sigma: tuple                   # (sigma_x, sigma_y) [image px]
    flux: np.ndarray               # (N,) integrated counts above offset
    offset: float
    position_errors: np.ndarray    # (N, 2) 1-sigma statistical uncertainties
    sigma_errors: tuple
    reduced_chi2: float
    success: bool
    message: str = ""
    model: Optional[np.ndarray] = field(default=None, repr=False)


def _axis_terms(n: int, centers: np.ndarray, sigma: float):
    """Pixel-integrated 1D Gaussian fractions and their derivatives.

    Returns (E, dE/dc, dE/dsigma), each (N, n) for pixels 0..n-1."""
    edges = np.arange(n + 1, dtype=float) - 0.5
    z = (edges[None, :] - centers[:, None]) / sigma           # (N, n+1)
    cdf = ndtr(z)
    pdf = np.exp(-0.5 * z * z) / _SQRT2PI
    e = cdf[:, 1:] - cdf[:, :-1]
    de_dc = -(pdf[:, 1:] - pdf[:, :-1]) / sigma
    zp = pdf * z
    de_ds = -(zp[:, 1:] - zp[:, :-1]) / sigma
    return e, de_dc, de_ds


def fit_gaussians(img: np.ndarray, guesses: Sequence[Sequence[float]], sigma0: float,
                  fit_offset: bool = True, max_nfev: int = 200,
                  margin: float = 4.0) -> GaussFit:
    """Jointly fit Gaussian PSFs to ``img`` around initial ``guesses`` ((col, row) list).

    Only a bounding box around the guesses (``margin`` sigmas) is used.
    """
    img = np.asarray(img, float)
    g = np.asarray(guesses, float).reshape(-1, 2)
    n_ions = len(g)
    if n_ions == 0:
        raise ValueError("no ions to fit")
    h, w = img.shape
    pad = margin * sigma0 + 2
    c0 = int(max(0, math.floor(g[:, 0].min() - pad)))
    c1 = int(min(w, math.ceil(g[:, 0].max() + pad) + 1))
    r0 = int(max(0, math.floor(g[:, 1].min() - pad)))
    r1 = int(min(h, math.ceil(g[:, 1].max() + pad) + 1))
    data = img[r0:r1, c0:c1]
    hh, ww = data.shape
    gx = g[:, 0] - c0
    gy = g[:, 1] - r0

    off0 = float(np.median(data)) if fit_offset else 0.0
    # initial fluxes from aperture sums
    flux0 = []
    for x, y in zip(gx, gy):
        ix, iy = int(round(x)), int(round(y))
        r = max(1, int(round(1.5 * sigma0)))
        win = data[max(0, iy - r):iy + r + 1, max(0, ix - r):ix + r + 1]
        flux0.append(max(float(win.sum() - off0 * win.size) * 1.6, 1.0))
    p0 = np.concatenate([[off0, sigma0, sigma0], flux0, gx, gy])

    n = n_ions
    lo = np.concatenate([[-np.inf if fit_offset else off0 - 1e-9, 0.3, 0.3],
                         np.zeros(n), gx - 3.0, gy - 3.0])
    hi = np.concatenate([[np.inf if fit_offset else off0 + 1e-9, 10.0 * sigma0 + 3, 10.0 * sigma0 + 3],
                         np.full(n, np.inf), gx + 3.0, gy + 3.0])
    p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)

    def unpack(p):
        return p[0], p[1], p[2], p[3:3 + n], p[3 + n:3 + 2 * n], p[3 + 2 * n:3 + 3 * n]

    def model(p):
        off, sx, sy, fl, xs, ys = unpack(p)
        ex, _, _ = _axis_terms(ww, xs, sx)
        ey, _, _ = _axis_terms(hh, ys, sy)
        return off + (ey.T * fl[None, :]) @ ex

    def residuals(p):
        return (model(p) - data).ravel()

    def jac(p):
        off, sx, sy, fl, xs, ys = unpack(p)
        ex, dex_dc, dex_ds = _axis_terms(ww, xs, sx)
        ey, dey_dc, dey_ds = _axis_terms(hh, ys, sy)
        j = np.empty((hh * ww, 3 + 3 * n))
        j[:, 0] = 1.0
        j[:, 1] = ((ey.T * fl[None, :]) @ dex_ds).ravel()
        j[:, 2] = ((dey_ds.T * fl[None, :]) @ ex).ravel()
        for i in range(n):
            j[:, 3 + i] = np.outer(ey[i], ex[i]).ravel()
            j[:, 3 + n + i] = (fl[i] * np.outer(ey[i], dex_dc[i])).ravel()
            j[:, 3 + 2 * n + i] = (fl[i] * np.outer(dey_dc[i], ex[i])).ravel()
        if not fit_offset:
            j[:, 0] = 0.0
        return j

    sol = least_squares(residuals, p0, jac=jac, bounds=(lo, hi), method="trf",
                        x_scale="jac", max_nfev=max_nfev)
    p = sol.x
    off, sx, sy, fl, xs, ys = unpack(p)
    res = sol.fun
    dof = max(1, res.size - p.size)
    s2 = float(np.sum(res ** 2) / dof)
    try:
        cov = np.linalg.pinv(sol.jac.T @ sol.jac) * s2
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:  # pragma: no cover
        err = np.full(p.size, np.nan)
    # reduced chi2 relative to a Poisson-like variance estimate of the data
    var = np.maximum(np.abs(model(p) - off), 1.0) + max(1.0, float(np.var(data[data < np.median(data)])))
    red = float(np.sum(res ** 2 / var.ravel()) / dof)
    pos = np.column_stack([xs + c0, ys + r0])
    perr = np.column_stack([err[3 + n:3 + 2 * n], err[3 + 2 * n:3 + 3 * n]])
    return GaussFit(positions=pos, sigma=(float(sx), float(sy)), flux=np.asarray(fl, float),
                    offset=float(off), position_errors=perr,
                    sigma_errors=(float(err[1]), float(err[2])), reduced_chi2=red,
                    success=bool(sol.success), message=str(sol.message),
                    model=model(p))
