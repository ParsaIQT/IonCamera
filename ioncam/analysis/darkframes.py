"""Dark frames of an EMCCD: what the camera adds to a pixel when no light
reaches it, measured, modelled and turned into state-detection (SPAM) errors.

Used by the camera characterization (:mod:`ioncam.characterize`).

The model (Lantz et al. 2008, appendix A; Harpsoe et al. 2012).  A pixel
(super-pixel when binned) reads, in counts above its bias,

    x = read noise + EM register output

* read noise: Gaussian, ``sigma`` counts (offset ``mu`` ~ 0);
* electrons present when the pixel enters the EM register - parallel
  clock-induced charge (CIC), dark current, light: Poisson with mean
  ``lam_p`` per pixel and frame, each multiplied to an exponential number
  of counts with mean ``g`` = counts per input electron = EM gain /
  electrons per count (n of them: Gamma(n, g));
* serial CIC: ``lam_s`` electrons per pixel and frame born inside the EM
  register at a random stage, so they get only part of the gain - one born
  at stage k of m comes out exponential with mean ~ g^(1 - k/m).  Averaged
  over the stages (log-uniform mean between one output electron, ``a``
  counts, and ``g``) its density is

      (exp(-x/g) - exp(-x/a)) / (x ln(g/a)).

Counts are integers: every distribution here is a probability per integer
count.  The sums of Poisson numbers of events are computed through their
characteristic function (one FFT), so a bright ion's tens of electrons cost
no more than a dark pixel's rare one.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import fft as sfft
from scipy.optimize import minimize
from scipy.special import exp1, ndtr
from scipy.stats import poisson

# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


def _single_bins(k: int, g: float) -> np.ndarray:
    """One electron through the EM register: exponential with mean ``g``
    counts, over the count bins 0..k-1 (bin 0 = [0, 1/2))."""
    edges = np.arange(k, dtype=float) + 0.5
    cdf = -np.expm1(-edges / max(g, 1e-9))
    return np.diff(np.concatenate([[0.0], cdf]))


def _serial_bins(k: int, g: float, a: float) -> np.ndarray:
    """One serial-CIC electron over the count bins 0..k-1: the log-uniform
    mixture of exponentials, integrated exactly (Frullani)."""
    g, a = float(g), float(a)
    lg = math.log(g / a)
    edges = np.arange(k, dtype=float) + 0.5
    # integral from 0 to c of (exp(-x/g) - exp(-x/a)) / x dx = ln(g/a) - E1(c/g) + E1(c/a)
    cum = lg - exp1(edges / g) + exp1(edges / a)
    return np.clip(np.diff(np.concatenate([[0.0], cum])) / lg, 0.0, None)


def noise_kernel(mu: float, sigma: float) -> Tuple[np.ndarray, int]:
    """Gaussian read noise integrated over count bins: (kernel, J) with the
    kernel over offsets -J..J."""
    sigma = max(float(sigma), 1e-3)
    j_max = int(math.ceil(abs(mu) + 8.0 * sigma)) + 1
    j = np.arange(-j_max, j_max + 1, dtype=float)
    return ndtr((j + 0.5 - mu) / sigma) - ndtr((j - 0.5 - mu) / sigma), j_max


def readout_pmf(means: Sequence[float], g: float, sigma: float, lam_s: float = 0.0,
                a: float = 1.0, mu: float = 0.0, hi: Optional[int] = None,
                weights: Optional[Sequence[float]] = None) -> Tuple[int, np.ndarray]:
    """Counts above bias: full-gain electrons Poisson with mean ``means[i]``
    (a mixture with ``weights``; CIC, dark current and light add up into
    one mean), serial CIC and read noise.  Returns ``(x0, pmf)`` with
    ``pmf[i]`` the probability of ``x0 + i`` counts (``x0 = -J``), up to
    ``hi`` counts (default: far enough for the largest mean)."""
    means = np.atleast_1d(np.asarray(means, float))
    w = np.ones(means.size) / means.size if weights is None else \
        np.asarray(weights, float) / np.sum(weights)
    g = max(float(g), 1e-6)
    m_top = float(means.max()) if means.size else 0.0
    if hi is None:
        hi = int(math.ceil(g * (m_top + 8.0 * math.sqrt(m_top) + 25.0)))
    kern, j_max = noise_kernel(mu, sigma)
    k = sfft.next_fast_len(int(hi) + 2 * j_max + 2, real=True)
    f1 = sfft.rfft(_single_bins(k, g))
    cf = np.zeros(f1.size, complex)
    for m, wi in zip(means, w):
        cf += wi * np.exp(max(m, 0.0) * (f1 - 1.0))
    if lam_s > 0.0 and g > a * 1.0001:
        cf *= np.exp(lam_s * (sfft.rfft(_serial_bins(k, g, a)) - 1.0))
    circ = np.zeros(k)
    circ[:j_max + 1] = kern[j_max:]
    circ[-j_max:] = kern[:j_max]
    full = sfft.irfft(cf * sfft.rfft(circ), k)
    pmf = np.concatenate([full[-j_max:], full[:int(hi) + 1]])
    return -j_max, np.clip(pmf, 0.0, None)


def dark_model_pmf(lo: int, hi: int, mu: float, sigma: float, g: float, lam_p: float,
                   lam_s: float = 0.0, a: float = 1.0) -> np.ndarray:
    """The dark model's probability of every count from ``lo`` to ``hi``."""
    x0, pmf = readout_pmf([lam_p], g, sigma, lam_s, a, mu, hi=max(hi, 1))
    out = np.zeros(hi - lo + 1)
    s0, d0 = max(lo - x0, 0), max(x0 - lo, 0)
    n = min(pmf.size - s0, out.size - d0)
    if n > 0:
        out[d0:d0 + n] = pmf[s0:s0 + n]
    return out


# ---------------------------------------------------------------------------
# fitting the dark histogram
# ---------------------------------------------------------------------------


def robust_sigma(r: np.ndarray, center: float = 0.0) -> float:
    """Gaussian sigma of the core from the samples below ``center``: events
    only add counts, so the low side is read noise alone."""
    r = np.asarray(r, float).ravel()
    low = r[r < center] - center
    if low.size < 10:
        return float(1.4826 * np.median(np.abs(r - np.median(r)))) if r.size else 0.0
    s0 = max(1.4826 * float(np.median(-low)), 0.3)
    core = low[low > -6.0 * s0]
    return float(math.sqrt(max(np.mean(core * core), 1e-6)))


def _group(n: int, lo: int, core_hi: int, width: int) -> np.ndarray:
    """Start index of every likelihood bin: single counts up to ``core_hi``,
    ``width`` counts beyond (the tail is smooth) - fewer terms, the same
    information."""
    x = np.arange(lo, lo + n)
    starts = list(np.flatnonzero(x <= core_hi))
    i = len(starts)
    while i < n:
        starts.append(i)
        i += width
    return np.array(starts, int)


def _chi2_dof(observed, expected, n_par) -> float:
    o = np.asarray(observed, float)
    e = np.asarray(expected, float)
    keep = e > 5.0
    if keep.sum() <= n_par:
        return float("nan")
    return float(np.sum((o[keep] - e[keep]) ** 2 / e[keep]) / (keep.sum() - n_par))


_FIT_NAMES = ("mu", "sigma", "g", "lam_p", "lam_s")


def fit_dark_histogram(hist: Sequence[int], lo: int, em_gain: float = 1.0,
                       g_hint: float = 0.0, fix_g: bool = False) -> dict:
    """Maximum-likelihood fit of the dark model to a histogram of counts above
    bias (``hist[i]`` samples with ``lo + i`` counts).

    ``em_gain`` (set) > 1.5: the full model (mu, sigma, g, lam_p, lam_s), the
    serial events from one output electron (g / em_gain counts) up;
    otherwise Gaussian read noise only (single electrons hide in it).
    ``g_hint``: counts per electron expected (EM gain / sensitivity) - the
    start, and kept when there are too few events to measure it (or
    ``fix_g``).  ``errors``: 1-sigma statistical errors - of ``mu`` and
    ``lam_s`` absolute, of the others relative; the error of ``g`` from the
    likelihood profile (the others fitted again at a gain 4 % off): at
    moderate gain (g a few times the read noise) serial CIC and a lower
    gain explain a dark histogram almost equally well, and it says so."""
    hist = np.asarray(hist, float)
    n_tot = float(hist.sum())
    x = np.arange(lo, lo + hist.size, dtype=float)
    out: Dict = {"ok": False, "n_samples": int(n_tot)}
    if n_tot < 100:
        out["note"] = "too few samples"
        return out
    order = np.cumsum(hist)
    med = float(x[min(np.searchsorted(order, 0.5 * n_tot), hist.size - 1)])
    w_low = hist * (x <= med)
    sigma0 = max(math.sqrt(float(np.sum(w_low * (x - med) ** 2) / max(w_low.sum(), 1.0))), 0.3)
    thr = med + 5.0 * sigma0
    tail = x > thr
    n_ev = float(hist[tail].sum())
    out["n_events"] = int(n_ev)
    if em_gain <= 1.5:
        core = np.abs(x - med) <= 6.0 * sigma0
        wc = hist * core
        mu = float(np.sum(wc * x) / max(np.sum(wc), 1.0))
        p = ndtr((x[core] + 0.5 - mu) / sigma0) - ndtr((x[core] - 0.5 - mu) / sigma0)
        out.update(ok=True, model="gaussian", mu=mu, sigma=sigma0, g=0.0, lam_p=0.0,
                   lam_s=0.0, a=0.0, excess=float(n_ev / n_tot),
                   note="no EM gain: single electrons are hidden in the read noise")
        out["chi2_dof"] = _chi2_dof(hist[core], p / max(p.sum(), 1e-30) * np.sum(wc), 2)
        return out
    g0 = float(np.sum(hist[tail] * (x[tail] - thr)) / n_ev) if n_ev > 0 else 0.0
    fix = bool((fix_g and g_hint > 0) or (n_ev < 50 and g_hint > 0))
    if g_hint > 0 and (fix or not (0.3 * g_hint < g0 < 3.0 * g_hint)):
        g0 = g_hint
    if not fix:
        g0 = max(g0, 2.0 * sigma0, 1.0)
    lam0 = min(max(n_ev / n_tot * math.exp(5.0 * sigma0 / g0), 1e-7), 1.0)
    # the likelihood bins: single counts in the core, the tail in steps of g/10
    hi = int(min(x[-1], med + 12.0 * sigma0 + 25.0 * g0))
    n_use = int(hi - lo + 1)
    h_use = hist[:n_use]
    starts = _group(n_use, lo, int(math.ceil(med + 10.0 * sigma0)) + 1, max(1, int(g0 / 10.0)))
    h_grp = np.add.reduceat(h_use, starts)

    def a_of(g):
        return g / max(em_gain, 1.0001)

    # parameters: mu, ln sigma, ln g, ln lam_p, lam_s / lam0 (linear: it can be 0)
    def unpack(t, g_fixed=None):
        mu, ls, lg, lp, ss = t
        g = g_fixed if g_fixed is not None else (g0 if fix else math.exp(lg))
        return mu, math.exp(ls), g, math.exp(lp), ss * lam0

    def nll(t, g_fixed=None):
        mu, s, g, lp, lsr = unpack(t, g_fixed)
        if lsr < 0.0 or lsr > 2.0:
            return 1e30
        p = np.add.reduceat(dark_model_pmf(lo, hi, mu, s, g, lp, lsr, a_of(g)), starts)
        tot = p.sum()
        if not tot > 0:
            return 1e30
        return -float(np.sum(h_grp * np.log(np.maximum(p / tot, 1e-300))))

    bounds = [(med - 5 * sigma0, med + 5 * sigma0),
              (math.log(0.2 * sigma0), math.log(5 * sigma0)),
              (math.log(max(0.1 * g0, 0.5)), math.log(20 * g0)),
              (math.log(1e-8), 0.0), (0.0, 2.0 / lam0)]

    def optimize(t0, g_fixed=None):
        f = (lambda t: nll(t, g_fixed))
        res = minimize(f, t0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 300, "eps": 1e-5})
        # settle in the valley (gain against serial and parallel CIC is nearly
        # flat at moderate gain): a small simplex from the best point
        x0 = np.array(res.x, float)
        steps = np.diag([0.02 * sigma0, 0.01, 0.0 if g_fixed is not None or fix else 0.02,
                         0.05, 0.3])
        simplex = np.array([x0] + [x0 + d for d in steps])
        pol = minimize(f, x0, method="Nelder-Mead",
                       options={"initial_simplex": simplex, "xatol": 1e-5, "fatol": 0.05,
                                "maxiter": 1500})
        if pol.fun < res.fun:
            return np.array(pol.x, float), float(pol.fun), int(res.nit)
        return x0, float(res.fun), int(res.nit)

    best = None
    for ss0 in (0.05, 1.0):
        t0 = np.array([med, math.log(sigma0), math.log(g0), math.log(lam0), ss0])
        cand = optimize(t0)
        if best is None or cand[1] < best[1]:
            best = cand
    t_best, f_best, n_it = best
    mu, s, g, lp, lsr = unpack(t_best)
    pmf = dark_model_pmf(lo, hi, mu, s, g, lp, lsr, a_of(g))
    pmf = pmf / max(pmf.sum(), 1e-300)
    out.update(ok=bool(np.isfinite(f_best)), model="emccd", mu=mu, sigma=s, g=g, lam_p=lp,
               lam_s=lsr, a=a_of(g), nll=f_best, g_fixed=fix, iterations=n_it)
    out["chi2_dof"] = _chi2_dof(h_grp, np.add.reduceat(pmf, starts) * n_tot, 5)
    errors = _errors(nll, t_best, fix, lam0)
    if not fix:
        # the gain's error from the profile likelihood: the others fitted
        # again at a gain 4 % off on either side
        d = 0.04
        prof = []
        for sgn in (-1.0, 1.0):
            gt = g * math.exp(sgn * d)
            t0 = np.array(t_best, float)
            prof.append(optimize(t0, g_fixed=gt)[1])
        curv = (prof[0] + prof[1] - 2.0 * f_best) / (d * d)
        errors["g"] = float(1.0 / math.sqrt(curv)) if curv > 0 else None
        out["g_profile"] = [prof[0] - f_best, prof[1] - f_best]
    out["errors"] = errors
    # how well the model gets the tail that decides SPAM: measured and model
    # fractions above a few thresholds
    xs = np.arange(lo, hi + 1)
    out["above"] = {}
    for name, t in (("3sigma", mu + 3 * s), ("5sigma", mu + 5 * s), ("g/2", mu + 0.5 * g),
                    ("g", mu + g), ("2g", mu + 2 * g), ("4g", mu + 4 * g)):
        k = xs > t
        out["above"][name] = [float(h_use[k].sum() / n_tot), float(pmf[k].sum())]
    if fix and not fix_g:
        out["note"] = "few events: g kept at EM gain / sensitivity"
    return out


def _errors(nll, t, fixed_g: bool, lam0: float) -> Dict[str, Optional[float]]:
    """1-sigma errors from the curvature of the negative log-likelihood: of
    ``mu`` and ``lam_s`` absolute, of the others relative (they are fitted
    as logs).  A serial CIC at its bound (0) is left out."""
    at_bound = t[4] <= 1e-6
    idx = [i for i in range(5) if not (fixed_g and i == 2) and not (at_bound and i == 4)]
    n = len(idx)
    hs = [1e-3, 1e-3, 1e-3, 1e-3, max(1e-3, 0.01 * abs(t[4]))]
    f0 = nll(t)
    hess = np.zeros((n, n))

    def at(*steps):
        tt = np.array(t, float)
        for i, dd in steps:
            tt[i] += dd
        return nll(tt)

    for p_ in range(n):
        i = idx[p_]
        hi_ = hs[i]
        hess[p_, p_] = (at((i, hi_)) - 2.0 * f0 + at((i, -hi_))) / (hi_ * hi_)
        for q_ in range(p_ + 1, n):
            j = idx[q_]
            hj = hs[j]
            v = (at((i, hi_), (j, hj)) - at((i, hi_), (j, -hj)) - at((i, -hi_), (j, hj)) +
                 at((i, -hi_), (j, -hj))) / (4.0 * hi_ * hj)
            hess[p_, q_] = hess[q_, p_] = v
    out: Dict[str, Optional[float]] = {name: None for name in _FIT_NAMES}
    try:
        d = np.diag(np.linalg.inv(hess))
    except np.linalg.LinAlgError:
        return out
    for p_, i in enumerate(idx):
        if d[p_] > 0 and np.isfinite(d[p_]):
            e = float(math.sqrt(d[p_]))
            out[_FIT_NAMES[i]] = e * lam0 if i == 4 else e
    return out


# ---------------------------------------------------------------------------
# a stack of dark frames
# ---------------------------------------------------------------------------


def analyze_dark_stack(frames: np.ndarray, em_gain: float = 1.0, sensitivity: float = 0.0,
                       first: Optional[np.ndarray] = None, origin: Tuple[int, int] = (0, 0),
                       binning: Tuple[int, int] = (1, 1), timestamps=None,
                       keep_maps: bool = False, max_hot: int = 50) -> Tuple[dict, dict]:
    """Everything one setting's dark frames say: ``frames`` (N, H, W) counts
    of the steady acquisition, ``first`` (K, H, W) the frames right after
    the start, ``origin`` / ``binning`` to give pixels sensor coordinates.

    Returns ``(stats, arrays)``: JSON-friendly numbers, and the arrays worth
    keeping (the histogram; maps with ``keep_maps``)."""
    f = np.asarray(frames)
    n, h, w = f.shape
    hb, vb = binning
    stats: Dict = {"frames": int(n), "pixels": int(h * w), "frame_shape": [int(h), int(w)]}
    arrays: Dict = {}
    if n < 4:
        stats["note"] = "too few frames"
        return stats, arrays
    bias_map = np.median(f, axis=0)
    # every frame against the bias of the other half of the frames: a bias
    # taken from the frame itself would pull its own samples in (the core of
    # the histogram ~1/N too narrow, and not Gaussian); the noise of that
    # bias (median of N/2 frames) is taken out of the read noise below
    b_even = np.rint(np.median(f[0::2], axis=0)).astype(np.int32)
    b_odd = np.rint(np.median(f[1::2], axis=0)).astype(np.int32)
    r = f.astype(np.int32)
    r[0::2] -= b_odd[None]
    r[1::2] -= b_even[None]
    bias_int = np.rint(bias_map).astype(np.int32)
    widen = math.sqrt(1.0 + math.pi / n)        # var(median of N/2) = pi sigma^2 / N
    offs = np.median(r.reshape(n, -1), axis=1).astype(float)
    stats["bias"] = float(np.median(bias_map))
    stats["bias_spatial_rms"] = float(np.std(bias_map))
    stats["bias_row_rms"] = float(np.std(bias_map.mean(axis=1)))
    stats["bias_col_rms"] = float(np.std(bias_map.mean(axis=0)))
    stats["common_mode_rms"] = float(np.std(offs))
    t = np.asarray(timestamps, float) if timestamps is not None and len(timestamps) == n \
        else None
    if t is not None and t[-1] > t[0]:
        stats["bias_drift_per_min"] = float(60.0 * np.polyfit(t - t[0], offs, 1)[0])
        stats["frame_rate"] = float((n - 1) / (t[-1] - t[0]))
    sub = r.reshape(-1)
    if sub.size > 2_000_000:
        sub = sub[:: sub.size // 2_000_000 + 1]
    sigma0 = robust_sigma(sub, float(np.median(sub)))
    # hot pixels first: their events are not the camera's CIC
    thr = max(5.0 * sigma0, 1.0)
    ev = r > thr
    counts = ev.sum(axis=0)
    rate = float(counts.sum()) / (n * h * w)
    hot = np.zeros((h, w), bool)
    for _ in range(4):
        k_hot = max(3, int(poisson.isf(1e-6 / max(h * w, 1), max(rate * n, 1e-9))) + 1)
        new_hot = counts >= k_hot
        if np.array_equal(new_hot, hot):
            break
        hot = new_hot
        rate = float(counts[~hot].sum()) / (n * max(int((~hot).sum()), 1))
    good = ~hot
    stats["event_threshold"] = thr
    stats["events_per_px"] = rate
    stats["n_hot"] = int(hot.sum())
    if hot.any():
        hv, hu = np.nonzero(hot)
        order = np.argsort(-counts[hot])[:max_hot]
        stats["hot_pixels"] = [
            [float(origin[0] + hb * hu[i] + 0.5 * (hb - 1)),
             float(origin[1] + vb * hv[i] + 0.5 * (vb - 1)),
             float(counts[hv[i], hu[i]] / n),
             float(np.mean(r[:, hv[i], hu[i]]))] for i in order]
    rg = r[:, good]
    g_hint = em_gain / sensitivity if (sensitivity > 0 and em_gain > 1.5) else 0.0
    g_range = g_hint if g_hint > 0 else (float(np.mean(rg[rg > thr]) - thr)
                                         if np.any(rg > thr) else 1.0)
    lo = int(max(int(rg.min()), -int(12 * sigma0) - 5))
    hi = int(min(int(rg.max()), int(12 * sigma0) + int(40 * max(g_range, 1.0)) + 10))
    hi = max(hi, lo + 1)
    vals = rg.ravel()
    under = int(np.sum(vals < lo))
    over = vals[vals > hi]
    hist = np.bincount(np.clip(vals, lo, hi) - lo, minlength=hi - lo + 1).astype(np.int64)
    hist[0] -= under
    hist[-1] -= over.size
    arrays["hist"] = hist
    arrays["hist_lo"] = np.array([lo])
    stats.update(hist_lo=lo, hist_hi=hi, underflow=under, overflow=int(over.size))
    if over.size:
        stats["overflow_max"] = int(over.max())
    fit = fit_dark_histogram(hist, lo, em_gain, g_hint=g_hint)
    stats["fit"] = fit
    sig = (float(fit["sigma"]) if fit.get("ok") else sigma0) / widen
    stats["read_noise"] = sig
    if sensitivity > 0:
        stats["read_noise_e"] = sig * sensitivity
    if fit.get("ok"):
        # the camera's own noise, for predictions (the bias estimate's noise out)
        stats["model"] = {"sigma": sig, "g": float(fit["g"]), "lam_p": float(fit["lam_p"]),
                          "lam_s": float(fit["lam_s"]), "a": float(fit["a"]),
                          "mu": float(fit["mu"])}
    if fit.get("ok") and fit.get("model") == "emccd":
        stats["cic_parallel"] = fit["lam_p"]
        stats["cic_serial"] = fit["lam_s"]
        stats["counts_per_electron"] = fit["g"]
        if sensitivity > 0:
            stats["em_gain_measured"] = fit["g"] * sensitivity
            stats["gain_ratio"] = fit["g"] * sensitivity / max(em_gain, 1e-9)
            stats["read_noise_input_e"] = sig / fit["g"]
    # where the events are: rows (distance from the readout register), trails
    ev_good = ev & good[None]
    row_rate = ev_good.sum(axis=(0, 2)) / (n * np.maximum(good.sum(axis=1), 1))
    if h >= 8 and row_rate.mean() > 0:
        slope = np.polyfit(np.arange(h, dtype=float), row_rate, 1)[0]
        stats["cic_row_trend_per_100_rows"] = float(slope * (100.0 / vb) / row_rate.mean())
    stats["trail"] = _trails(r, ev_good)
    if n > 2 and rate > 0:
        both = np.sum(ev_good[1:] & ev_good[:-1])
        stats["event_repeat"] = float(both / max(ev_good[:-1].sum(), 1) / rate)
    if first is not None and len(first):
        fr = np.asarray(first).astype(np.int32) - bias_int[None]
        stats["first_frames"] = [
            {"offset": float(np.median(xx)),
             "events_per_px": float(np.mean((xx > thr)[good]))} for xx in fr]
    if keep_maps:
        arrays["bias_map"] = bias_map.astype(np.float32)
        arrays["event_map"] = (counts / n).astype(np.float32)
        arrays["frame_offsets"] = offs.astype(np.float32)
    return stats, arrays


def _trails(r: np.ndarray, ev: np.ndarray) -> dict:
    """Charge left behind by an event, as a fraction of its counts, in the
    pixel read next (either direction along a row - which one depends on the
    amplifier - and the next row)."""
    amp = r[ev]
    if amp.size < 20:
        return {}
    a = float(np.mean(amp))
    out = {}
    for name, src, dst in (
            ("right", (slice(None), slice(None), slice(None, -1)),
             (slice(None), slice(None), slice(1, None))),
            ("left", (slice(None), slice(None), slice(1, None)),
             (slice(None), slice(None), slice(None, -1))),
            ("down", (slice(None), slice(None, -1), slice(None)),
             (slice(None), slice(1, None), slice(None)))):
        e = ev[src]
        if e.sum() < 20:
            continue
        nxt = r[dst][e & ~ev[dst]]
        if nxt.size:
            out[name] = float(np.mean(nxt) / a)
    return out


# ---------------------------------------------------------------------------
# state detection (SPAM)
# ---------------------------------------------------------------------------


def _change_mixture(mean: float, lifetime: float, duration: float, start_dark: bool,
                    n_steps: int = 48) -> Tuple[np.ndarray, np.ndarray]:
    """Poisson means and weights of an ion that may change state during the
    detection: a bright one stops scattering (depumping), a dark one
    (``start_dark``) starts; it collects ``mean`` pro rata of its bright
    time."""
    if lifetime <= 0.0 or duration <= 0.0:
        return np.array([0.0 if start_dark else mean]), np.ones(1)
    p_stay = math.exp(-duration / lifetime)
    q = (np.arange(n_steps) + 0.5) / n_steps * (1.0 - p_stay)
    frac = -lifetime * np.log1p(-q) / duration        # change time / duration, in [0, 1)
    if start_dark:
        frac = 1.0 - frac
    means = np.concatenate([[0.0 if start_dark else mean], mean * frac])
    weights = np.concatenate([[p_stay], np.full(n_steps, (1.0 - p_stay) / n_steps)])
    return means, weights


def spam_errors(dark: Tuple[int, np.ndarray], bright: Tuple[int, np.ndarray]) -> dict:
    """The threshold (bright if counts > threshold) that minimises the mean of
    both errors, for two count distributions ``(x0, pmf)``."""
    (xd, pd), (xb, pb) = dark, bright
    lo, hi = min(xd, xb), max(xd + pd.size, xb + pb.size)
    d = np.zeros(hi - lo)
    b = np.zeros(hi - lo)
    d[xd - lo:xd - lo + pd.size] = pd / max(pd.sum(), 1e-300)
    b[xb - lo:xb - lo + pb.size] = pb / max(pb.sum(), 1e-300)
    e_dark = np.clip(1.0 - np.cumsum(d), 0.0, 1.0)     # dark read as bright, threshold lo + i
    e_bright = np.clip(np.cumsum(b), 0.0, 1.0)         # bright read as dark
    err = 0.5 * (e_dark + e_bright)
    i = int(np.argmin(err))
    return {"threshold": int(lo + i), "error": float(err[i]), "dark_error": float(e_dark[i]),
            "bright_error": float(e_bright[i])}


class _Readout:
    """Count distributions of one camera model (per ROI sum of ``n_pixels``
    super-pixels) for any Poisson mixture of full-gain electrons, on one
    grid up to ``hi`` counts: the spectra of one electron, of serial CIC and
    of the read noise are computed once."""

    def __init__(self, model: dict, n_pixels: int, hi: int, g_signal: float = 0.0):
        k_pix = max(int(n_pixels), 1)
        self.sigma = float(model["sigma"]) * math.sqrt(k_pix)
        self.g = float(g_signal) if g_signal > 0 else max(float(model["g"]), 1e-6)
        self.base = float(model["lam_p"]) * k_pix
        lam_s = float(model.get("lam_s", 0.0)) * k_pix
        a = float(model.get("a", 1.0)) or 1.0
        kern, self.j = noise_kernel(0.0, self.sigma)
        self.hi = int(hi)
        self.k = sfft.next_fast_len(self.hi + 2 * self.j + 2, real=True)
        self.f1 = sfft.rfft(_single_bins(self.k, self.g))
        circ = np.zeros(self.k)
        circ[:self.j + 1] = kern[self.j:]
        circ[-self.j:] = kern[:self.j]
        self.fixed = sfft.rfft(circ)
        if lam_s > 0.0 and self.g > a * 1.0001:
            self.fixed = self.fixed * np.exp(lam_s * (sfft.rfft(_serial_bins(self.k, self.g, a))
                                                      - 1.0))

    def pmf(self, means, weights) -> Tuple[int, np.ndarray]:
        cf = np.zeros(self.f1.size, complex)
        for m, w in zip(means, weights):
            cf += w * np.exp(max(float(m), 0.0) * (self.f1 - 1.0))
        full = sfft.irfft(cf * self.fixed, self.k)
        return -self.j, np.clip(np.concatenate([full[-self.j:], full[:self.hi + 1]]), 0.0, None)

    def spam(self, signal: float, background: float = 0.0, bright_lifetime: float = 0.0,
             dark_lifetime: float = 0.0, duration: float = 0.0) -> dict:
        base = self.base + max(float(background), 0.0)
        mb, wb = _change_mixture(signal, bright_lifetime, duration, False)
        md, wd = _change_mixture(signal, dark_lifetime, duration, True)
        out = spam_errors(self.pmf(base + md, wd), self.pmf(base + mb, wb))
        out.update(signal=float(signal), background=float(background))
        return out


def _hi_for(model: dict, n_pixels: int, electrons: float, g_signal: float = 0.0) -> int:
    g = float(g_signal) if g_signal > 0 else max(float(model["g"]), 1e-6)
    m = float(model["lam_p"]) * max(int(n_pixels), 1) + max(electrons, 0.0)
    return int(math.ceil(g * (m + 8.0 * math.sqrt(m) + 25.0)))


def predict_spam(model: dict, signal: float, background: float = 0.0, n_pixels: int = 1,
                 bright_lifetime: float = 0.0, dark_lifetime: float = 0.0,
                 duration: float = 0.0, g_signal: float = 0.0) -> dict:
    """SPAM errors of thresholding the sum of ``n_pixels`` (super-)pixels of a
    camera described by ``model`` (per super-pixel: sigma, g, lam_p, lam_s,
    a - the dark model at the detection's exposure and binning) for an ion
    giving ``signal`` photo-electrons in them when bright, with
    ``background`` photo-electrons of stray light.  Lifetimes (same unit as
    ``duration``) put state changes during the detection in.  ``g_signal``:
    counts per photo-electron if better known than the dark fit's (at
    moderate gain the dark frames alone pin it down to a few per cent)."""
    r = _Readout(model, n_pixels, _hi_for(model, n_pixels, signal + background, g_signal),
                 g_signal)
    out = r.spam(signal, background, bright_lifetime, dark_lifetime, duration)
    out["n_pixels"] = max(int(n_pixels), 1)
    return out


def scaled_model(model: dict, binning: Tuple[int, int] = (1, 1),
                 measured_binning: Tuple[int, int] = (1, 1), exposure: float = 0.0,
                 measured_exposure: float = 0.0, dark_current: float = 0.0) -> dict:
    """The dark model at another binning and exposure: parallel CIC and dark
    current add up over the binned pixels, serial CIC and read noise come
    once per super-pixel; dark current (electrons/pixel/s) for the exposure
    change."""
    area = binning[0] * binning[1] / max(measured_binning[0] * measured_binning[1], 1)
    lam_p = float(model["lam_p"]) * area + max(dark_current, 0.0) * \
        binning[0] * binning[1] * (exposure - measured_exposure)
    return dict(model, lam_p=max(lam_p, 0.0))


def detection_curve(model: dict, rate: float, times: Sequence[float],
                    background_rate: float = 0.0, n_pixels: int = 1,
                    bright_lifetime: float = 0.0, dark_lifetime: float = 0.0,
                    dark_current: float = 0.0, area: int = 1, g_signal: float = 0.0,
                    _readout: Optional[_Readout] = None) -> list:
    """SPAM errors at every detection time in ``times`` [s] for an ion giving
    ``rate`` photo-electrons/s in its ROI (lifetimes in s; dark current in
    electrons/pixel/s over the ``area`` pixels of a super-pixel)."""
    times = [float(t) for t in times]
    if not times:
        return []
    k = max(int(n_pixels), 1)
    t_top = max(times)
    r = _readout or _Readout(model, k, _hi_for(model, k, (rate + background_rate +
                                                          dark_current * area * k) * t_top,
                                               g_signal), g_signal)
    out = []
    for t in times:
        bg = background_rate * t + dark_current * area * k * t
        e = r.spam(rate * t, bg, bright_lifetime, dark_lifetime, t)
        e["n_pixels"] = k
        out.append(e)
    return out


def shortest_detection(model: dict, rate: float, target: float = 1e-3,
                       t_min: float = 2e-5, t_max: float = 0.03, **kw) -> dict:
    """The shortest detection time [s] reaching a SPAM error <= ``target``:
    bisection in log time (without state changes the error falls with the
    time; with them it has a minimum, found first).  ``kw`` as for
    :func:`detection_curve`.  ``best_time`` / ``best_error``: the lowest
    error (at ``t_max`` without state changes)."""
    k = max(int(kw.get("n_pixels", 1)), 1)
    total = rate + kw.get("background_rate", 0.0) + kw.get("dark_current", 0.0) * \
        kw.get("area", 1) * k
    readout = _Readout(model, k, _hi_for(model, k, total * t_max, kw.get("g_signal", 0.0)),
                       kw.get("g_signal", 0.0))

    def err(t):
        return detection_curve(model, rate, [t], _readout=readout, **kw)[0]

    lo, hi = math.log(t_min), math.log(t_max)
    e_hi = err(t_max)
    if kw.get("bright_lifetime") or kw.get("dark_lifetime"):
        # golden-section search for the best time first
        a_, b_ = lo, hi
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        c, d = b_ - gr * (b_ - a_), a_ + gr * (b_ - a_)
        ec, ed = err(math.exp(c))["error"], err(math.exp(d))["error"]
        for _ in range(24):
            if ec < ed:
                b_, d, ed = d, c, ec
                c = b_ - gr * (b_ - a_)
                ec = err(math.exp(c))["error"]
            else:
                a_, c, ec = c, d, ed
                d = a_ + gr * (b_ - a_)
                ed = err(math.exp(d))["error"]
        hi = 0.5 * (a_ + b_)
        e_hi = err(math.exp(hi))
    out = {"best_time": math.exp(hi), "best_error": e_hi["error"]}
    if e_hi["error"] > target:
        out.update(time=None, reached=False, error=None, threshold=None)
        return out
    e_lo = err(math.exp(lo))
    if e_lo["error"] <= target:
        out.update(time=math.exp(lo), reached=True, error=e_lo["error"],
                   threshold=e_lo["threshold"])
        return out
    best = e_hi
    while hi - lo > 0.01:
        mid = 0.5 * (lo + hi)
        e = err(math.exp(mid))
        if e["error"] <= target:
            hi, best = mid, e
        else:
            lo = mid
    out.update(time=math.exp(hi), reached=True, error=best["error"],
               threshold=best["threshold"])
    return out
