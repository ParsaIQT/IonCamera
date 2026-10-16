"""Is the crystal still there?  The ion light in averaged Grabber readout frames.

During an experiment the iXon sends every readout frame (the AQROI, binned)
over USB as well.  One shot holds a few photo-electrons per ion, but the mean
of many shots says which ions shine where, whatever states the experiment
prepares in each shot:

    mean signal of super-pixel k  =  sum_j K[k, j] a_j  +  b

``K[k, j]``: the fraction of ion j's light in super-pixel k (its fitted PSF,
core and halo, at its position); ``a_j``: its mean light per shot (bright
fraction x brightness, in counts); ``b``: a uniform background (laser scatter
off the trap, clock-induced charge).  This is fitted for the crystal as
located and for the crystals of one, two, ... ions fewer: the ions that are
left re-arrange around the trap centre (two ions -> one in the middle).
The description that fits best tells what happened:

* ``ok``: every ion at its place, shining;
* ``dark``: an ion's light is gone while the others stay in place - a dark
  state that does not end (a molecular ion, a long-lived level) - or the
  experiment keeps that ion dark in every shot;
* ``lost``: the light comes from fewer ions at their new places: the crystal
  lost an ion and re-arranged;
* ``no_light``: no ion light at all (lasers off, all dark, all lost);
* ``unclear``: the light fits no crystal expected here (moved? recalibrate).

The bias (counts of a super-pixel without a photo-electron) is the read-noise
peak of the counts: shots in which a super-pixel caught no photo-electron are
common in its darkest super-pixel even with the ions bright in every shot.  A
small error in it is uniform and taken up by ``b``.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares, lsq_linear
from scipy.special import ndtr

from ioncam.analysis.chain import equilibrium_positions
from ioncam.analysis.readout import ReadoutPlan, _light_matrix, dark_pixel_mask

VERDICTS = ("ok", "dark", "lost", "no_light", "unclear", "uncertain")


# ---------------------------------------------------------------------------
# the bias: the read-noise peak of EMCCD counts
# ---------------------------------------------------------------------------


def _smoothed_mode(x: np.ndarray, width: float) -> Tuple[int, np.ndarray, int]:
    base = int(x.min())
    h = np.bincount(x - base).astype(float)
    r = np.arange(-int(4 * width) - 1, int(4 * width) + 2)
    k = np.exp(-0.5 * (r / width) ** 2)
    hs = np.convolve(h, k / k.sum(), mode="same")
    return base + int(np.argmax(hs)), h, base


def _left_sigma(x: np.ndarray, center: float) -> float:
    """Gaussian sigma from the samples below ``center`` (light only adds counts)."""
    low = x[x < center] - center
    if low.size < 10:
        return float(1.4826 * np.median(np.abs(x - np.median(x)))) or 1.0
    s0 = max(1.4826 * float(np.median(-low)), 0.5)
    core = low[low > -6.0 * s0]
    return max(float(math.sqrt(np.mean(core * core))), 0.5)


def zero_light_level(samples, sigma: Optional[float] = None, g: float = 0.0) -> dict:
    """The counts of a super-pixel that caught no photo-electron, from
    ``samples`` (shots x super-pixels sharing the bias): the read-noise peak
    of their counts, fitted near the peak - common level and read noise, for
    every super-pixel its own share of shots in the peak and its pedestal of
    one-electron events (an electron through the EM register: exponential,
    mean ``g`` counts, convolved with the read noise).  The sharpest peak
    (the darkest super-pixel's) sets the start.  ``sigma``: the read noise if
    known (counts).  Returns level, its error, sigma, the fraction of samples
    in the peak and ``ok`` (a peak worth the name)."""
    x = np.rint(np.asarray(samples, float)).astype(np.int64)
    if x.ndim == 1:
        x = x[:, None]
    x = x.reshape(len(x), -1)
    n_s, n_c = x.shape
    if n_s * n_c < 20:
        lvl = float(np.median(x)) if x.size else 0.0
        return {"level": lvl, "sigma": float(sigma or 0.0), "fraction": 0.0, "ok": False,
                "error": float("inf")}
    modes, sigmas, hists = [], [], []
    for c in range(n_c):
        m0, h, base = _smoothed_mode(x[:, c], 2.0)
        modes.append(m0)
        sigmas.append(_left_sigma(x[:, c], m0 + 0.5))
        hists.append((h, base))
    k0 = int(np.argmin(sigmas))
    s0 = float(sigma) if sigma else float(sigmas[k0])
    m0 = float(modes[k0])
    # the super-pixels whose peak is there (a lit one may have none)
    use = [c for c in range(n_c) if abs(modes[c] - m0) <= 3.0 * s0] or [k0]
    g = float(g) if g > 2.0 * s0 else 10.0 * s0
    lo, hi = int(math.floor(m0 - 5.0 * s0)), int(math.ceil(m0 + 3.0 * s0))
    xs = np.arange(lo, hi + 1)
    counts = []
    for c in use:
        h, base = hists[c]
        idx = xs - base
        counts.append(np.where((idx >= 0) & (idx < h.size), h[np.clip(idx, 0, h.size - 1)],
                               0.0))
    counts = np.array(counts)
    nu = len(use)
    fix_sigma = sigma is not None and sigma > 0

    def model(p):
        mu, sig = p[0], p[1]
        f0, c1 = p[2:2 + nu], p[2 + nu:]
        y = xs - mu
        gauss = np.exp(-0.5 * (y / sig) ** 2) / (math.sqrt(2.0 * math.pi) * sig)
        ped = np.exp(0.5 * (sig / g) ** 2 - y / g) * ndtr(y / sig - sig / g) / g
        return n_s * (f0[:, None] * gauss[None, :] + c1[:, None] * ped[None, :])

    def resid(p):
        m = model(p)
        return ((m - counts) / np.sqrt(np.maximum(m, 1.0))).ravel()

    f_start = np.clip(2.0 * counts[:, xs <= m0].sum(axis=1) / n_s, 1e-3, 1.0)
    p0 = np.concatenate([[m0, s0], f_start, np.full(nu, 0.1)])
    s_lo, s_hi = (0.999 * s0, 1.001 * s0 + 1e-9) if fix_sigma else (0.5 * s0, 2.0 * s0)
    lower = np.concatenate([[m0 - 2.0 * s0, s_lo], np.zeros(nu), np.zeros(nu)])
    upper = np.concatenate([[m0 + 2.0 * s0, s_hi], np.full(nu, 1.2), np.full(nu, 20.0)])
    try:
        r = least_squares(resid, np.clip(p0, lower, upper), bounds=(lower, upper))
        mu, sig = float(r.x[0]), float(r.x[1])
        f0 = r.x[2:2 + nu]
    except ValueError:
        mu, sig, f0 = m0, s0, np.zeros(nu)
    n_peak = float(np.sum(f0) * n_s)
    ok = n_peak >= 30 and float(np.max(f0)) >= 0.03
    if not ok:
        mu, sig = m0, s0
    return {"level": mu, "sigma": sig, "fraction": float(np.max(f0)), "ok": bool(ok),
            "error": float(sig / math.sqrt(max(n_peak, 1.0))), "pixels": len(use)}


# ---------------------------------------------------------------------------
# where the ions are: planned and re-arranged crystals
# ---------------------------------------------------------------------------


def chain_geometry(positions) -> Tuple[np.ndarray, np.ndarray, float]:
    """(centre, unit axis, length scale) of a linear crystal: the ions sit at
    ``centre + scale * u_i * axis`` (u_i the equilibrium positions)."""
    p = np.asarray(positions, float).reshape(-1, 2)
    c = p.mean(axis=0)
    if len(p) < 2:
        return c, np.array([1.0, 0.0]), 0.0
    _, _, vt = np.linalg.svd(p - c)
    axis = vt[0]
    s = np.sort((p - c) @ axis)
    u = equilibrium_positions(len(p))
    return c, axis, float(np.dot(u, s) / np.dot(u, u))


def crystal_positions(positions, m: int) -> np.ndarray:
    """Where ``m`` ions sit in the trap of the crystal ``positions`` (fewer
    ions re-arrange around the same centre, along the same axis: the length
    scale is the trap's)."""
    if m <= 0:
        return np.zeros((0, 2))
    c, axis, scale = chain_geometry(positions)
    return c[None, :] + scale * equilibrium_positions(m)[:, None] * axis[None, :]


def frame_light(plan: ReadoutPlan, positions, halo: Optional[Tuple[float, float]] = None
                ) -> np.ndarray:
    """K[k, j]: the fraction of ion j's light (at ``positions``, sensor px) in
    super-pixel k of the readout frame (row-major, SDK orientation)."""
    w, h = plan.frame_size
    vv, uu = np.mgrid[0:h, 0:w]
    ax = plan.roi[0] + uu.ravel() * plan.hbin - 0.5
    ay = plan.roi[1] + vv.ravel() * plan.vbin - 0.5
    pos = np.asarray(positions, float).reshape(-1, 2)
    if not len(pos):
        return np.zeros((w * h, 0))
    sx, sy = plan.psf_sigma
    return _light_matrix(ax, ax + plan.hbin, ay, ay + plan.vbin, pos, sx, sy,
                         plan.psf_halo if halo is None else halo)


# ---------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------


def _fit(signal, stat2, K, b_bounds, systematic, floor2):
    """Bounded weighted least squares of signal = K a + b (a >= 0); the
    errors take the model's own light into account (systematic)."""
    p, m = K.shape
    A = np.column_stack([K, np.ones(p)])
    lo = np.concatenate([np.zeros(m), [b_bounds[0]]])
    hi = np.concatenate([np.full(m, np.inf), [b_bounds[1]]])
    ion = np.zeros(p)
    for _ in range(3):
        err2 = stat2 + (systematic * ion) ** 2 + floor2
        w = 1.0 / np.sqrt(err2)
        res = lsq_linear(A * w[:, None], signal * w, bounds=(lo, hi), method="bvls")
        x = res.x
        ion = K @ x[:m]
    err2 = stat2 + (systematic * ion) ** 2 + floor2
    pred = A @ x
    chi2 = float(np.sum((signal - pred) ** 2 / err2))
    # statistical covariance of the amplitudes (unbounded linearisation)
    w = 1.0 / np.sqrt(stat2 + floor2)
    try:
        cov = np.linalg.pinv((A * w[:, None]).T @ (A * w[:, None]))
    except np.linalg.LinAlgError:
        cov = np.full((m + 1, m + 1), np.nan)
    errs = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return x[:m], float(x[m]), chi2, pred, errs[:m], float(errs[m]), cov[:m, :m]


def _names(idx) -> str:
    """0 / 0 and 2 / 0, 1 and 3"""
    idx = [str(i) for i in idx]
    return idx[0] if len(idx) == 1 else ", ".join(idx[:-1]) + " and " + idx[-1]


def check_crystal(plan: ReadoutPlan, frames, crystal: Optional[Sequence] = None,
                  reference: Optional[dict] = None, dark_level: float = 0.15,
                  systematic: float = 0.05, bias: Optional[dict] = None,
                  g: float = 0.0, margin: float = 16.0) -> dict:
    """Which ions shine where, from ``frames`` (shots, H, W) of the readout
    plan ``plan`` (SDK orientation).

    ``crystal``: every ion of the crystal (sensor px; default the plan's
    ions) - ions outside the AQROI count for where the others go when one
    is lost.  ``reference``: an earlier result with the crystal as planned
    (its ``reference`` entry): the ions' light and the background then, so
    an ion is judged against its own brightness and the background cannot
    pose as a centred ion.  ``dark_level``: an ion with less than this
    fraction of the others' light (each against its reference) is dark.
    ``systematic``: relative error of the PSF model per super-pixel.
    ``bias``: a known zero-light level (``zero_light_level``); ``g``: counts
    per photo-electron (for the bias fit).  ``margin``: how much better (in
    chi-square) another crystal must fit before the planned one is given up.
    """
    f = np.asarray(frames, float)
    w, h = plan.frame_size
    out: Dict = {"verdict": "unclear", "n_shots": int(len(f)), "expected": None}
    if f.ndim != 3 or f.shape[1:] != (h, w) or len(f) < 20:
        out["message"] = "too few readout shots ({})".format(len(f))
        out["verdict"] = None
        return out
    n_shots = len(f)
    flat = f.reshape(n_shots, -1)
    crystal = np.asarray(plan.positions if crystal is None else crystal, float).reshape(-1, 2)
    n_cr = len(crystal)
    out["expected"] = n_cr
    if reference is not None and "verdict" in reference:     # a whole earlier result
        reference = reference.get("reference")
    ref = reference or None
    sys_eff = float(max(systematic, (ref or {}).get("systematic") or 0.0))

    # -- the zero-light level: dark pixels if the frame has them, else the
    #    read-noise peak of the darkest super-pixel
    dark = dark_pixel_mask(plan).ravel()
    sigma_ref = (ref or {}).get("sigma")
    lvl = None
    if dark.any():
        lvl = dict(zero_light_level(flat[:, dark].reshape(-1, 1), sigma_ref, g),
                   source="dark pixels")
    if lvl is None or not lvl["ok"]:
        darkest = np.argsort(np.median(flat, axis=0), kind="stable")[:4]
        lvl = dict(zero_light_level(flat[:, darkest], sigma_ref, g),
                   source="read-noise peak of the darkest super-pixels")
    if bias is not None and (not lvl["ok"] or bias.get("error", 9e9) < lvl.get("error", 9e9)):
        lvl = dict(bias, source=bias.get("source", "earlier"))
    level = float(lvl["level"])
    out["bias"] = {k: lvl.get(k) for k in ("level", "sigma", "fraction", "ok", "source", "error")}

    signal = flat.mean(axis=0) - level
    stat2 = flat.var(axis=0) / n_shots
    bias_err = float(lvl.get("error") or 1.0)
    floor2 = 0.25 + bias_err ** 2
    p_pix = signal.size

    # the background: physical (>= 0, up to a bias error); with a reference
    # it cannot grow far beyond what it was (it would pose as a centred ion)
    b_lo = -(3.0 * bias_err + 2.0)
    b_hi = np.inf
    if ref is not None and ref.get("background") is not None:
        light_px = float(np.sum(ref.get("light") or [0.0])) / max(p_pix, 1)
        b_hi = float(ref["background"]) + 0.25 * abs(float(ref["background"])) + \
            0.03 * light_px + 3.0 * bias_err + 2.0

    # -- the crystals: as planned, then one, two, ... ions fewer
    hyps = []
    for m in range(n_cr, -1, -1):
        pos = crystal if m == n_cr else crystal_positions(crystal, m)
        K = frame_light(plan, pos)
        a, b, chi2, pred, a_err, b_err, cov = _fit(signal, stat2, K, (b_lo, b_hi), sys_eff,
                                                   floor2)
        dof = max(p_pix - m - 1, 1)
        hyps.append({"ions": m, "positions": pos.tolist(), "light": a.tolist(),
                     "light_error": a_err.tolist(), "background": b,
                     "background_error": b_err, "chi2": chi2, "dof": dof,
                     "cost": chi2 + 2.0 * (m + 1), "in_frame": K.sum(axis=0).tolist(),
                     "model": pred, "cov": cov})
    planned = hyps[0]
    by_cost = sorted(hyps, key=lambda d: d["cost"])
    if by_cost[0] is not planned and planned["cost"] - by_cost[0]["cost"] < margin:
        best, rival = planned, by_cost[0]   # not clearly another crystal: the planned one
    else:
        best = by_cost[0]
        rival = next((d for d in by_cost if d is not best), None)
    conf = float(rival["cost"] - best["cost"]) if rival is not None else None
    out["confidence"] = conf
    out["fit"] = {"chi2": best["chi2"], "dof": best["dof"],
                  "chi2_dof": best["chi2"] / best["dof"]}
    out["hypotheses"] = [{k: v for k, v in d.items() if k not in ("model", "cov")}
                         for d in hyps]
    out["signal"] = signal.reshape(h, w).tolist()
    out["model"] = best["model"].reshape(h, w).tolist()
    out["background"] = best["background"]
    out["n_ions"] = best["ions"]

    # -- what the best crystal says, ion by ion
    def shining(d):
        """Ion light, relative light (each against its reference, the
        brightest = 1) and states ("bright", "dark", "unseen") of a fit."""
        a = np.asarray(d["light"], float)
        seen = np.asarray(d["in_frame"], float) >= 0.1      # >= 10 % of its light in frame
        if d["ions"] == n_cr and ref_light is not None and len(ref_light) == n_cr:
            rel = np.where(ref_light > 0, a / np.maximum(ref_light, 1e-9), 0.0)
        else:
            rel = a.copy()
        top = float(np.max(rel[seen])) if seen.any() else 0.0
        q = rel / top if top > 0 else np.zeros_like(rel)
        return a, seen, q, ["bright" if (qq >= dark_level and sn) else
                            ("dark" if sn else "unseen") for qq, sn in zip(q, seen)]

    ref_light = np.asarray(ref.get("light"), float) if ref is not None and \
        ref.get("light") is not None else None
    ref_total = float(np.sum(ref_light)) if ref_light is not None else None
    a, seen, q, states = shining(best)
    a_err = np.asarray(best["light_error"], float)
    out["ions"] = [{"index": j, "x": round(float(best["positions"][j][0]), 3),
                    "y": round(float(best["positions"][j][1]), 3), "light": float(a[j]),
                    "light_error": float(a_err[j]), "relative": float(q[j]),
                    "reference": None if ref_light is None or best["ions"] != n_cr or
                    j >= len(ref_light) else float(ref_light[j]), "state": states[j]}
                   for j in range(len(a))]
    planned_bright = list(plan.ion_states) if len(plan.ion_states) == n_cr else \
        ["bright"] * n_cr
    # no ion light: the background alone explains the frames about as well as
    # any crystal, and the ions give clearly less than half their light after
    # arming (before there is a reference: no significant light)
    def dim(d) -> bool:
        """Little ion light in the fit ``d``: its total (the error of the sum:
        ions close together share their light, anti-correlated amplitudes)."""
        a_d = np.asarray(d["light"], float)
        seen_d = np.asarray(d["in_frame"], float) >= 0.1
        if not seen_d.any():
            return True
        cov_d = np.asarray(d["cov"], float)[np.ix_(seen_d, seen_d)]
        err = float(math.sqrt(max(float(np.sum(cov_d)), 0.0))) if np.all(np.isfinite(cov_d)) \
            else 0.0
        tot = float(np.sum(a_d[seen_d]))
        if ref_total is not None:
            return tot + 2.0 * err < 0.5 * ref_total
        return tot < max(3.0 * err, 3.0)

    # (every crystal that fits about as well: one may put a centred ion's even
    # light into the background, another into the ion)
    near = [d for d in hyps if d["ions"] and d["cost"] - by_cost[0]["cost"] < margin]
    bg_only = hyps[-1]["cost"] - by_cost[0]["cost"] < margin
    no_light = n_cr == 0 or (bg_only and all(dim(d) for d in near))
    lasers = ""
    if ref is not None and ref.get("background") is not None:
        b_ref = float(ref["background"])
        if b_ref > 3.0 * bias_err + 3.0 and best["background"] < 0.3 * b_ref:
            lasers = " - and no laser scatter either: lasers off?"
        elif b_ref > 3.0 * bias_err + 3.0:
            lasers = " - the laser scatter is still there: ions dark or lost"
    window = "last {} shots".format(n_shots)
    chi2_dof = best["chi2"] / best["dof"]
    n_bright = sum(1 for st in states if st == "bright")
    alike = rival is not None and conf is not None and conf < margin
    if no_light:
        out["verdict"] = "no_light"
        out["message"] = "no ion light in the {}{}".format(window, lasers)
    elif best["ions"] < n_cr:
        k = n_cr - best["ions"]
        dark_left = [j for j, st in enumerate(states) if st == "dark"]
        out["verdict"] = "lost"
        out["lost"] = k
        out["message"] = "{} lost: the light fits {} ion{} at {} new place{} (the crystal " \
            "re-arranged), not {} ({}){}".format(
                "an ion was" if k == 1 else "{} ions were".format(k), best["ions"],
                "s" if best["ions"] != 1 else "", "its" if best["ions"] == 1 else "their",
                "s" if best["ions"] != 1 else "", n_cr, window,
                "; {} of them dark".format(len(dark_left)) if dark_left else "")
    else:
        dark_now = [j for j, st in enumerate(states) if st == "dark" and
                    planned_bright[j] == "bright"]
        if dark_now:
            out["verdict"] = "dark"
            out["dark"] = dark_now
            one = len(dark_now) == 1
            rest = n_cr - len(dark_now)
            out["message"] = "{} {} dark: no light at {} place{} in the {} while the other " \
                "ion{} stay{} in place - a dark state that does not end (e.g. a molecular " \
                "ion), or the experiment keeps {} dark".format(
                    "ion" if one else "ions", _names(dark_now), "its" if one else "their",
                    "" if one else "s", window, "" if rest == 1 else "s",
                    "s" if rest == 1 else "", "it" if one else "them")
            if alike and rival["ions"] < n_cr and \
                    sum(1 for st in shining(rival)[3] if st == "bright") == n_bright:
                out["message"] += " - or {} lost: {} ion{} left would look the same".format(
                    n_cr - rival["ions"], rival["ions"], "s" if rival["ions"] != 1 else "")
        elif alike and rival["ions"] < n_cr:
            out["verdict"] = "uncertain"
            out["message"] = "{} ions in place, or {} at new places - they fit the light " \
                "about as well ({}); more shots will tell".format(n_cr, rival["ions"], window)
        else:
            out["verdict"] = "ok"
            out["message"] = "{} ion{} in place, shining ({})".format(
                n_cr, "s" if n_cr != 1 else "", window)
    if chi2_dof > 25.0 and out["verdict"] != "no_light" and (conf is None or conf < 4.0 * margin):
        out["verdict"] = "unclear"
        out["message"] = "the light pattern fits no crystal expected here (chi2/dof {:.0f}; " \
            "best: {} ion{}) - ions moved? recalibrate".format(
                chi2_dof, best["ions"], "s" if best["ions"] != 1 else "")
    # backed by the light: another crystal fits clearly worse (or no light at all)
    out["decided"] = bool(out["verdict"] in ("no_light", "unclear") or
                          (conf is not None and conf >= 0.5 * margin) or
                          (out["verdict"] == "dark" and rival is not None))
    if out["verdict"] == "ok" and out["decided"]:
        out["reference"] = {"light": a.tolist(), "background": best["background"],
                            "sigma": lvl.get("sigma") if lvl.get("ok") else None,
                            "systematic": float(min(max(
                                systematic * math.sqrt(max(chi2_dof, 1.0)), systematic), 0.3)),
                            "n_shots": n_shots}
    return out


def same_crystal(a: Optional[dict], b: Optional[dict]) -> bool:
    """Two checks say the same about the crystal."""
    return bool(a and b and a.get("verdict") == b.get("verdict") and
                a.get("n_ions") == b.get("n_ions") and a.get("dark") == b.get("dark"))


def combine(full: dict, recent: Optional[dict]) -> dict:
    """The verdict to report from the check of the whole window and of its
    most recent part: they agree - the whole window (more shots); they
    differ - the crystal changed within the window: the recent part if its
    light decides it, else the whole window."""
    if recent is None or recent.get("verdict") is None:
        return dict(full, window="full")
    if full.get("verdict") is None:
        return dict(recent, window="recent")
    if same_crystal(full, recent) or not recent.get("decided"):
        return dict(full, window="full")
    out = dict(recent, window="recent")
    out["message"] = out["message"] + " - changed within the last {} shots".format(
        full.get("n_shots"))
    return out
