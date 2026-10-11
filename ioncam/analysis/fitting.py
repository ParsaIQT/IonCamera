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
    sigma: tuple                   # (sigma_x, sigma_y) of the core [image px]
    flux: np.ndarray               # (N,) integrated counts above offset (core + halo)
    offset: float
    position_errors: np.ndarray    # (N, 2) 1-sigma statistical uncertainties
    sigma_errors: tuple
    reduced_chi2: float
    success: bool
    message: str = ""
    model: Optional[np.ndarray] = field(default=None, repr=False)
    halo_fraction: float = 0.0     # fraction of the light in the halo (0 = single Gaussian)
    halo_sigma: float = 0.0        # halo sigma [image px]
    halo_errors: tuple = (0.0, 0.0)
    window: tuple = (0, 0, 0, 0)   # (c0, r0, c1, r1): the image region fitted (= model's extent)


def _axis_terms(n: int, centers: np.ndarray, sigma: float, start: int = 0):
    """Pixel-integrated 1D Gaussian fractions and their derivatives.

    Returns (E, dE/dc, dE/dsigma), each (N, n) for pixels start..start+n-1."""
    edges = (np.arange(n + 1, dtype=float) - 0.5) if start == 0 else \
        (np.arange(start, start + n + 1, dtype=float) - 0.5)
    z = (edges[None, :] - centers[:, None]) / sigma           # (N, n+1)
    cdf = ndtr(z)
    pdf = np.exp(-0.5 * z * z) / _SQRT2PI
    e = cdf[:, 1:] - cdf[:, :-1]
    de_dc = -(pdf[:, 1:] - pdf[:, :-1]) / sigma
    zp = pdf * z
    de_ds = -(zp[:, 1:] - zp[:, :-1]) / sigma
    return e, de_dc, de_ds


def _box(g: np.ndarray, pad: float, w: int, h: int):
    """Bounding box (c0, r0, c1, r1) of the guesses ``g`` plus ``pad``, on the image."""
    return (int(max(0, math.floor(g[:, 0].min() - pad))),
            int(max(0, math.floor(g[:, 1].min() - pad))),
            int(min(w, math.ceil(g[:, 0].max() + pad) + 1)),
            int(min(h, math.ceil(g[:, 1].max() + pad) + 1)))


def _windows(g: np.ndarray, pad: float, w: int, h: int):
    """Disjoint image boxes covering every guess +- ``pad``: the guesses are
    grouped while their boxes overlap, each group fitted in its bounding box
    (one box - the bounding box of all - for a compact crystal)."""
    groups = [[i] for i in range(len(g))]
    while True:
        boxes = [_box(g[k], pad, w, h) for k in groups]
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                p, q = boxes[a], boxes[b]
                if p[0] < q[2] and q[0] < p[2] and p[1] < q[3] and q[1] < p[3]:
                    groups[a] += groups.pop(b)
                    break
            else:
                continue
            break
        else:
            return boxes


def fit_gaussians(img: np.ndarray, guesses: Sequence[Sequence[float]], sigma0: float,
                  fit_offset: bool = True, max_nfev: int = 200,
                  margin: float = 4.0, halo: bool = False) -> GaussFit:
    """Jointly fit Gaussian PSFs to ``img`` around initial ``guesses`` ((col, row) list).

    Only the pixels within ``margin`` sigmas of the guesses are used: the
    bounding box of a compact crystal, one such box per group of guesses far
    apart (so spots scattered over the sensor cost no more than a crystal).
    With ``halo`` the PSF is a core plus a broad, isotropic Gaussian halo
    (fraction and sigma fitted, shared by all ions) - real imaging systems put
    a significant part of the light into such wings, which is what sets the
    crosstalk between neighbouring readout ROIs.  The halo is kept only if it
    is significant; otherwise the single-Gaussian fit is returned.
    """
    img = np.asarray(img, float)
    g = np.asarray(guesses, float).reshape(-1, 2)
    n_ions = len(g)
    if n_ions == 0:
        raise ValueError("no ions to fit")
    h, w = img.shape
    pad = (max(margin, 9.0) if halo else margin) * sigma0 + 2
    c0, r0, c1, r1 = _box(g, pad, w, h)
    data = img[r0:r1, c0:c1]
    hh, ww = data.shape
    gx = g[:, 0] - c0
    gy = g[:, 1] - r0
    # the fitted windows (frame coordinates relative to (c0, r0)) and their pixels
    wins = [(a - c0, b - r0, cc - c0, d - r0) for a, b, cc, d in _windows(g, pad, w, h)]
    parts = [data[b:d, a:cc] for a, b, cc, d in wins]
    fitted = parts[0].ravel() if len(parts) == 1 else np.concatenate([q.ravel() for q in parts])
    n_px = fitted.size

    off0 = float(np.median(fitted)) if fit_offset else 0.0
    # initial fluxes from aperture sums
    flux0 = []
    for x, y in zip(gx, gy):
        ix, iy = int(round(x)), int(round(y))
        r = max(1, int(round(1.5 * sigma0)))
        win = data[max(0, iy - r):iy + r + 1, max(0, ix - r):ix + r + 1]
        flux0.append(max(float(win.sum() - off0 * win.size) * 1.6, 1.0))
    n = n_ions
    nh = 2 if halo else 0
    p0 = np.concatenate([[off0, sigma0, sigma0], flux0, gx, gy, [0.15, 2.5 * sigma0][:nh]])
    lo = np.concatenate([[-np.inf if fit_offset else off0 - 1e-9, 0.3, 0.3],
                         np.zeros(n), gx - 3.0, gy - 3.0, [0.0, 1.2 * 0.3][:nh]])
    hi = np.concatenate([[np.inf if fit_offset else off0 + 1e-9, 10.0 * sigma0 + 3, 10.0 * sigma0 + 3],
                         np.full(n, np.inf), gx + 3.0, gy + 3.0, [0.5, 5.0 * sigma0][:nh]])
    p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)

    def unpack(p):
        f_h, s_h = (p[3 + 3 * n], p[4 + 3 * n]) if halo else (0.0, 1.0)
        return (p[0], p[1], p[2], p[3:3 + n], p[3 + n:3 + 2 * n], p[3 + 2 * n:3 + 3 * n],
                f_h, s_h)

    def terms(p, win):
        off, sx, sy, fl, xs, ys, f_h, s_h = unpack(p)
        a, b, cc, d = win
        ex = _axis_terms(cc - a, xs, sx, a)
        ey = _axis_terms(d - b, ys, sy, b)
        hx = _axis_terms(cc - a, xs, max(s_h, sx * 1.01), a) if halo else None
        hy = _axis_terms(d - b, ys, max(s_h, sy * 1.01), b) if halo else None
        return off, sx, sy, fl, xs, ys, f_h, s_h, ex, ey, hx, hy

    def model_window(p, win):
        off, sx, sy, fl, xs, ys, f_h, s_h, ex, ey, hx, hy = terms(p, win)
        m = off + (1.0 - f_h) * (ey[0].T * fl[None, :]) @ ex[0]
        if halo:
            m = m + f_h * (hy[0].T * fl[None, :]) @ hx[0]
        return m

    def model(p):
        if len(wins) == 1:
            return model_window(p, wins[0]).ravel()
        return np.concatenate([model_window(p, win).ravel() for win in wins])

    def residuals(p):
        return model(p) - fitted

    def jac_window(p, win, j):
        off, sx, sy, fl, xs, ys, f_h, s_h, ex, ey, hx, hy = terms(p, win)
        c = 1.0 - f_h
        j[:, 0] = 1.0 if fit_offset else 0.0
        j[:, 1] = c * ((ey[0].T * fl[None, :]) @ ex[2]).ravel()
        j[:, 2] = c * ((ey[2].T * fl[None, :]) @ ex[0]).ravel()
        for i in range(n):
            core = np.outer(ey[0][i], ex[0][i])
            gx_ = c * np.outer(ey[0][i], ex[1][i])
            gy_ = c * np.outer(ey[1][i], ex[0][i])
            if halo:
                hal = np.outer(hy[0][i], hx[0][i])
                j[:, 3 + i] = (c * core + f_h * hal).ravel()
                gx_ = gx_ + f_h * np.outer(hy[0][i], hx[1][i])
                gy_ = gy_ + f_h * np.outer(hy[1][i], hx[0][i])
            else:
                j[:, 3 + i] = core.ravel()
            j[:, 3 + n + i] = (fl[i] * gx_).ravel()
            j[:, 3 + 2 * n + i] = (fl[i] * gy_).ravel()
        if halo:
            core_all = (ey[0].T * fl[None, :]) @ ex[0]
            halo_all = (hy[0].T * fl[None, :]) @ hx[0]
            j[:, 3 + 3 * n] = (halo_all - core_all).ravel()
            j[:, 4 + 3 * n] = (f_h * ((hy[2].T * fl[None, :]) @ hx[0] +
                                      (hy[0].T * fl[None, :]) @ hx[2])).ravel()

    def jac(p):
        j = np.empty((n_px, 3 + 3 * n + nh))
        k = 0
        for win in wins:
            size = (win[2] - win[0]) * (win[3] - win[1])
            jac_window(p, win, j[k:k + size])
            k += size
        return j

    sol = least_squares(residuals, p0, jac=jac, bounds=(lo, hi), method="trf",
                        x_scale="jac", max_nfev=max_nfev)
    p = sol.x
    off, sx, sy, fl, xs, ys, f_h, s_h = unpack(p)
    res = sol.fun
    dof = max(1, res.size - p.size)
    m_fit = model(p)
    # the model image over the whole box around the guesses (for the callers)
    m = m_fit.reshape(hh, ww) if len(wins) == 1 else model_window(p, (0, 0, ww, hh))
    # EMCCD noise grows with the signal (shot noise x excess factor): the
    # variance law var = a + b * signal is fitted to the squared residuals, and
    # the parameter errors of this unweighted fit follow from the sandwich
    # estimator (a single average variance would understate the errors of the
    # positions, which are set by the noisy pixels on the ions)
    sig = np.maximum(m_fit - off, 0.0)
    r2 = res * res * (res.size / dof)
    a_ = np.column_stack([np.ones_like(sig), sig])
    coef = np.linalg.lstsq(a_, r2, rcond=None)[0]
    v = max(float(coef[0]), 1e-9) + max(float(coef[1]), 0.0) * sig
    if not np.all(np.isfinite(v)) or v.mean() <= 0:  # pragma: no cover
        v = np.full_like(sig, float(np.mean(r2)))
    try:
        jj = sol.jac
        inv = np.linalg.pinv(jj.T @ jj)
        cov = inv @ ((jj.T * v) @ jj) @ inv
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:  # pragma: no cover
        err = np.full(p.size, np.nan)
    if halo:
        f_err = float(err[3 + 3 * n])
        # a "halo" at its width limit is broad background structure, not the PSF
        # (and a halo is a minority of the light, clearly broader than the core -
        # otherwise core and halo just share the work of one Gaussian)
        if not (0.02 < f_h < 0.48 and f_h > 3.0 * f_err and
                1.5 * max(sx, sy) < s_h < 4.8 * sigma0):
            # no significant halo: the plain Gaussian model describes the data
            return fit_gaussians(img, guesses, sigma0, fit_offset, max_nfev, margin, False)
    # reduced chi2 relative to a Poisson-like variance estimate of the data
    var = np.maximum(np.abs(m_fit - off), 1.0) + \
        max(1.0, float(np.var(fitted[fitted < np.median(fitted)])))
    red = float(np.sum(res ** 2 / var) / dof)
    pos = np.column_stack([xs + c0, ys + r0])
    perr = np.column_stack([err[3 + n:3 + 2 * n], err[3 + 2 * n:3 + 3 * n]])
    return GaussFit(positions=pos, sigma=(float(sx), float(sy)), flux=np.asarray(fl, float),
                    offset=float(off), position_errors=perr,
                    sigma_errors=(float(err[1]), float(err[2])), reduced_chi2=red,
                    success=bool(sol.success), message=str(sol.message), model=m,
                    halo_fraction=float(f_h) if halo else 0.0,
                    halo_sigma=float(s_h) if halo else 0.0,
                    halo_errors=(float(err[3 + 3 * n]), float(err[4 + 3 * n])) if halo
                    else (0.0, 0.0), window=(c0, r0, c1, r1))


# -- crowded fields: fixed PSF (core + halo), used on every live frame ----------------

@dataclass
class PSF:
    """PSF in *frame* pixels: Gaussian core plus an optional broad Gaussian halo."""
    sx: float
    sy: float
    halo: float = 0.0             # fraction of the light in the halo
    hx: float = 0.0               # halo sigmas (frame px)
    hy: float = 0.0

    @classmethod
    def from_sensor(cls, sigma, halo_fraction, halo_sigma, hbin=1, vbin=1) -> "PSF":
        sx, sy = (sigma, sigma) if np.isscalar(sigma) else sigma
        f = float(halo_fraction) if halo_sigma > 0 else 0.0
        return cls(sx / hbin, sy / vbin, min(max(f, 0.0), 0.9), halo_sigma / hbin,
                   halo_sigma / vbin)

    def parts(self):
        out = [(1.0 - self.halo, self.sx, self.sy)]
        if self.halo > 0:
            out.append((self.halo, self.hx, self.hy))
        return out

    def images(self, w: int, h: int, xs: np.ndarray, ys: np.ndarray, deriv: bool = False):
        """Unit-flux images of point sources at (xs, ys) on a w x h pixel grid:
        (N, h, w), plus d/dx and d/dy if ``deriv``."""
        img = dx = dy = 0.0
        for wgt, sx, sy in self.parts():
            ex, dex, _ = _axis_terms(w, xs, sx)
            ey, dey, _ = _axis_terms(h, ys, sy)
            img = img + wgt * ey[:, :, None] * ex[:, None, :]
            if deriv:
                dx = dx + wgt * ey[:, :, None] * dex[:, None, :]
                dy = dy + wgt * dey[:, :, None] * ex[:, None, :]
        return (img, dx, dy) if deriv else img


@dataclass
class CrowdedFit:
    positions: np.ndarray         # (N, 2) patch coordinates (col, row)
    flux: np.ndarray
    flux_err: np.ndarray
    position_errors: np.ndarray
    background: float
    model: np.ndarray
    rms: float                    # rms of the residuals
    cost: float = 0.0             # sum of squared residuals


def linear_photometry(data: np.ndarray, positions, psf: PSF, noise: Optional[float] = None,
                      background: bool = True, var: Optional[np.ndarray] = None):
    """Fluxes of point sources at fixed ``positions`` (patch coords) by linear least
    squares, so overlapping neighbours are separated.  ``var`` (per-pixel
    variance) makes it a weighted fit.  Returns (flux, flux_err, background,
    model)."""
    data = np.asarray(data, float)
    h, w = data.shape
    pos = np.asarray(positions, float).reshape(-1, 2)
    n = len(pos)
    tmpl = psf.images(w, h, pos[:, 0], pos[:, 1]).reshape(n, -1)
    cols = [tmpl] + ([np.ones((1, h * w))] if background else [])
    a = np.vstack(cols).T                                   # (npix, n [+1])
    sw = np.ones(h * w) if var is None else 1.0 / np.sqrt(np.maximum(var.ravel(), 1e-12))
    aw = a * sw[:, None]
    ata = aw.T @ aw
    ata[np.diag_indices_from(ata)] += 1e-9 * (np.trace(ata) / max(len(ata), 1) + 1e-12)
    sol = np.linalg.solve(ata, aw.T @ (data.ravel() * sw))
    model = (a @ sol).reshape(h, w)
    if var is not None:
        noise = 1.0
    elif noise is None:
        dof = max(1, h * w - len(sol))
        noise = math.sqrt(float(np.sum((data - model) ** 2)) / dof)
    cov = np.linalg.inv(ata) * noise ** 2
    err = np.sqrt(np.clip(np.diag(cov)[:n], 0, None))
    bg = float(sol[n]) if background else 0.0
    return sol[:n], err, bg, model


def _levenberg_marquardt(evaluate, p, lo, hi, iterations):
    """Minimise |r|^2 for ``evaluate(p) -> (r, J)`` within box bounds."""
    r, j = evaluate(p)
    cost = float(r @ r)
    lam = 1e-3
    for _ in range(iterations):
        jtj = j.T @ j
        grad = j.T @ r
        diag = np.diag(jtj).copy() + 1e-12
        accepted, gain = False, 0.0
        for _ in range(5):
            try:
                step = -np.linalg.solve(jtj + lam * np.diag(diag), grad)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            q = np.clip(p + step, lo, hi)
            r_new, j_new = evaluate(q)
            c_new = float(r_new @ r_new)
            if c_new <= cost:
                gain = cost - c_new
                p, r, j, cost = q, r_new, j_new, c_new
                lam = max(lam / 5.0, 1e-7)
                accepted = True
                break
            lam *= 8.0
        if not accepted or gain < 1e-4 * cost:
            break
    return p, r, j, cost


def _crowded_result(p, r, j, cost, d, h, w, n, xs, ys) -> CrowdedFit:
    dof = max(1, r.size - p.size)
    s2 = cost / dof
    try:
        cov = np.linalg.pinv(j.T @ j) * s2
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:  # pragma: no cover
        err = np.full(p.size, np.nan)
    return CrowdedFit(positions=np.column_stack([xs, ys]), flux=p[1:1 + n].copy(),
                      flux_err=err[1:1 + n], position_errors=np.zeros((n, 2)),
                      background=float(p[0]), model=(r + d).reshape(h, w), rms=math.sqrt(s2),
                      cost=cost)


def fit_fixed_psf(data: np.ndarray, guesses, psf: PSF, max_move: float = 2.5,
                  iterations: int = 8, var: Optional[np.ndarray] = None) -> CrowdedFit:
    """Joint fit of positions and fluxes of all sources in ``data`` (a background-
    subtracted patch) with a known PSF and a constant local background.

    A small Levenberg-Marquardt loop (a few ms for a chain of ions, cheap enough
    for every live frame); positions stay within ``max_move`` of the guesses and
    fluxes non-negative.  With ``var`` (per-pixel variance) the fit is weighted
    and ``cost`` is the chi-square (``rms`` ~ 1 for a good model)."""
    data = np.asarray(data, float)
    h, w = data.shape
    g = np.asarray(guesses, float).reshape(-1, 2)
    n = len(g)
    sw = None if var is None else 1.0 / np.sqrt(np.maximum(np.asarray(var, float).ravel(),
                                                           1e-12))
    d = data.ravel() if sw is None else data.ravel() * sw
    flux0, _, bg0, _ = linear_photometry(data, g, psf, var=var)
    p = np.concatenate([[bg0], np.maximum(flux0, 0.0), g[:, 0], g[:, 1]])
    lo = np.concatenate([[-np.inf], np.zeros(n), g[:, 0] - max_move, g[:, 1] - max_move])
    hi = np.concatenate([[np.inf], np.full(n, np.inf), g[:, 0] + max_move, g[:, 1] + max_move])

    def evaluate(q):
        bg, fl, xs, ys = q[0], q[1:1 + n], q[1 + n:1 + 2 * n], q[1 + 2 * n:]
        img, dx, dy = psf.images(w, h, xs, ys, deriv=True)
        m = bg + np.tensordot(fl, img, axes=1).ravel()
        j = np.empty((h * w, 1 + 3 * n))
        j[:, 0] = 1.0
        j[:, 1:1 + n] = img.reshape(n, -1).T
        j[:, 1 + n:1 + 2 * n] = (fl[:, None, None] * dx).reshape(n, -1).T
        j[:, 1 + 2 * n:] = (fl[:, None, None] * dy).reshape(n, -1).T
        if sw is not None:
            return m * sw - d, j * sw[:, None]
        return m - d, j

    p, r, j, cost = _levenberg_marquardt(evaluate, p, lo, hi, iterations)
    out = _crowded_result(p, r, j, cost, d, h, w, n, p[1 + n:1 + 2 * n], p[1 + 2 * n:])
    if sw is not None:
        out.model = (r + d).reshape(h, w) / sw.reshape(h, w)
    try:
        cov = np.linalg.pinv(j.T @ j) * (cost / max(1, r.size - p.size))
        e = np.sqrt(np.clip(np.diag(cov), 0, None))
        out.position_errors = np.column_stack([e[1 + n:1 + 2 * n], e[1 + 2 * n:]])
    except np.linalg.LinAlgError:  # pragma: no cover
        pass
    return out


def fit_chain_psf(data: np.ndarray, n_ions: int, center, scale: float, angle: float,
                  psf: PSF, iterations: int = 15, var: Optional[np.ndarray] = None) -> CrowdedFit:
    """Fit a linear chain of ``n_ions`` at their equilibrium positions
    (centre, length scale, axis angle; one flux per site, a site may be dark)
    to ``data`` with a known PSF.  Robust where ions are too close to be
    separated individually: the physics fixes the spacing pattern."""
    from ioncam.analysis.chain import equilibrium_positions
    data = np.asarray(data, float)
    h, w = data.shape
    sw = None if var is None else 1.0 / np.sqrt(np.maximum(np.asarray(var, float).ravel(),
                                                           1e-12))
    d = data.ravel() if sw is None else data.ravel() * sw
    n = int(n_ions)
    u = equilibrium_positions(n)

    def sites(q):
        c, s_, a = q[1 + n:3 + n], q[3 + n], q[4 + n]
        ax = np.array([math.cos(a), math.sin(a)])
        return c[None, :] + s_ * u[:, None] * ax[None, :], ax

    p = np.concatenate([[0.0], np.zeros(n), np.asarray(center, float), [scale, angle]])
    pos0, _ = sites(p)
    fl0, _, bg0, _ = linear_photometry(data, pos0, psf, var=var)
    p[0], p[1:1 + n] = bg0, np.maximum(fl0, 0.0)
    lo = np.concatenate([[-np.inf], np.zeros(n), [-np.inf, -np.inf, 0.3 * scale,
                                                 angle - 0.5]])
    hi = np.concatenate([[np.inf], np.full(n, np.inf), [np.inf, np.inf, 3.0 * scale,
                                                       angle + 0.5]])

    def evaluate(q):
        pos, ax = sites(q)
        fl = q[1:1 + n]
        img, dx, dy = psf.images(w, h, pos[:, 0], pos[:, 1], deriv=True)
        r = q[0] + np.tensordot(fl, img, axes=1).ravel() - d
        fdx = (fl[:, None, None] * dx).reshape(n, -1)
        fdy = (fl[:, None, None] * dy).reshape(n, -1)
        perp = np.array([-ax[1], ax[0]])
        j = np.empty((h * w, 5 + n))
        j[:, 0] = 1.0
        j[:, 1:1 + n] = img.reshape(n, -1).T
        j[:, 1 + n] = fdx.sum(axis=0)
        j[:, 2 + n] = fdy.sum(axis=0)
        j[:, 3 + n] = (u[:, None] * (fdx * ax[0] + fdy * ax[1])).sum(axis=0)
        j[:, 4 + n] = (q[3 + n] * u[:, None] * (fdx * perp[0] + fdy * perp[1])).sum(axis=0)
        if sw is not None:
            return (r + d) * sw - d, j * sw[:, None]
        return r, j

    p, r, j, cost = _levenberg_marquardt(evaluate, p, lo, hi, iterations)
    pos, _ = sites(p)
    out = _crowded_result(p, r, j, cost, d, h, w, n, pos[:, 0], pos[:, 1])
    if sw is not None:
        out.model = (r + d).reshape(h, w) / sw.reshape(h, w)
    return out
