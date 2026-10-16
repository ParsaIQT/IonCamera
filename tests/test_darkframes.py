"""The EMCCD dark-frame model: fits of synthetic dark frames with known
parameters, SPAM predictions against Monte Carlo shots."""

import math

import numpy as np
import pytest

from ioncam.analysis.darkframes import (analyze_dark_stack, detection_curve,
                                        fit_dark_histogram, predict_spam, scaled_model,
                                        shortest_detection)
from ioncam.analysis.readout import threshold_fidelity


def emccd_electrons(rng, shape, gain, lam_p, lam_s, signal=0.0):
    """EM register output [electrons]: Poisson(lam_p + signal) electrons
    through the whole register (n of them: Gamma(n, gain)), Poisson(lam_s)
    serial-CIC electrons with a log-uniform gain between 1 and ``gain``."""
    n = rng.poisson(lam_p + np.broadcast_to(signal, shape))
    out = np.zeros(shape)
    hit = n > 0
    out[hit] = rng.gamma(n[hit], gain)
    ser = rng.poisson(lam_s, shape)
    while ser.max() > 0:
        m = ser > 0
        out[m] += rng.exponential(np.power(gain, rng.random(int(m.sum()))))
        ser[m] -= 1
    return out


def dark_frames(n, h, w, gain, sens, read_e, lam_p, lam_s, bias=100.0, seed=0, hot=()):
    """Dark frames [counts]: a fixed bias pattern, read noise, the register
    output; ``hot`` pixels (y, x, extra electrons per frame)."""
    rng = np.random.default_rng(seed)
    pattern = bias + rng.normal(0.0, 2.0, (h, w))
    lam = np.full((h, w), float(lam_p))
    for y, x, extra in hot:
        lam[y, x] += extra
    e = emccd_electrons(rng, (n, h, w), gain, lam, lam_s)
    counts = e / sens + pattern[None] + rng.normal(0.0, read_e / sens, (n, h, w))
    return np.clip(np.rint(counts), 0, 65535).astype(np.uint16)


def test_fit_recovers_a_synthetic_emccd():
    """Read noise, counts per electron, parallel CIC, hot pixels and the real
    EM gain of known frames come back (serial CIC is weakly pinned down by
    a million samples and only bounded here)."""
    g_true, sens, read_e, lam_p, lam_s = 300.0, 4.0, 60.0, 0.005, 0.005
    f = dark_frames(80, 128, 128, g_true, sens, read_e, lam_p, lam_s, seed=1,
                    hot=[(10, 20, 1.0), (100, 50, 0.6)])
    st, arr = analyze_dark_stack(f, em_gain=g_true, sensitivity=sens, keep_maps=True,
                                 origin=(30, 40))
    fit = st["fit"]
    assert fit["ok"] and fit["model"] == "emccd"
    assert st["read_noise"] == pytest.approx(read_e / sens, rel=0.03)
    # (~3 sigma: gain, parallel and serial CIC trade off against each other)
    assert fit["g"] == pytest.approx(g_true / sens, rel=0.11)
    assert st["gain_ratio"] == pytest.approx(1.0, abs=0.11)
    assert fit["lam_p"] == pytest.approx(lam_p, rel=0.3)
    assert 0.0 <= fit["lam_s"] < 3.0 * lam_s
    assert fit["chi2_dof"] < 1.5
    assert st["bias"] == pytest.approx(100.0, abs=1.0)
    # the hot pixels, in sensor coordinates, are not taken for CIC
    assert st["n_hot"] == 2
    assert sorted((round(x), round(y)) for x, y, *_ in st["hot_pixels"]) == [(50, 50), (80, 140)]
    # the errors are statistical 1-sigma ones: relative, of the right size
    assert 0.003 < fit["errors"]["g"] < 0.06
    assert 0.01 < fit["errors"]["lam_p"] < 0.15
    # what is kept for every setting
    assert arr["hist"].sum() == st["frames"] * (128 * 128 - 2)
    assert arr["bias_map"].shape == (128, 128) and arr["event_map"].shape == (128, 128)
    assert st["model"]["sigma"] == pytest.approx(read_e / sens, rel=0.03)


def test_conventional_amplifier_reads_gaussian_noise():
    rng = np.random.default_rng(2)
    f = np.rint(500.0 + rng.normal(0.0, 6.0, (40, 64, 64))).astype(np.uint16)
    st, _ = analyze_dark_stack(f, em_gain=1.0, sensitivity=4.0)
    assert st["fit"]["model"] == "gaussian"
    assert st["read_noise"] == pytest.approx(6.0, rel=0.03)
    assert st["read_noise_e"] == pytest.approx(24.0, rel=0.03)
    assert st["n_hot"] == 0


def test_low_gain_fit_with_the_gain_fixed():
    """At moderate gain (g ~ 2 read noise) serial CIC and the gain trade off
    and the free gain is uncertain - and says so; with the gain fixed at the
    calibrated value the parallel CIC is the camera's."""
    g_true, sens, read_e, lam_p, lam_s = 100.0, 4.0, 60.0, 0.01, 0.003
    f = dark_frames(80, 128, 128, g_true, sens, read_e, lam_p, lam_s, seed=3)
    st, arr = analyze_dark_stack(f, em_gain=g_true, sensitivity=sens)
    free = st["fit"]
    fixed = fit_dark_histogram(arr["hist"], int(arr["hist_lo"][0]), g_true,
                               g_hint=g_true / sens, fix_g=True)
    assert fixed["ok"] and fixed["g_fixed"] and fixed["g"] == g_true / sens
    assert fixed["errors"]["g"] is None
    assert fixed["lam_p"] == pytest.approx(lam_p, rel=0.2)
    # the free fit's gain error covers how far off it is
    if free["errors"]["g"] is not None:
        assert abs(free["g"] / (g_true / sens) - 1.0) < 4.0 * free["errors"]["g"]


def shots(rng, n, signal, gain, sens, read_e, lam_p, lam_s, k=1, bright_lifetime=0.0,
          duration=1.0):
    """ROI sums of ``k`` pixels for an ion giving ``signal`` photo-electrons
    (decaying to dark with ``bright_lifetime``)."""
    sig = np.full(n, float(signal))
    if bright_lifetime > 0:
        sig = signal * np.minimum(rng.exponential(bright_lifetime, n), duration) / duration
    e = emccd_electrons(rng, (n,), gain, lam_p * k + sig, lam_s * k)
    return np.rint(e / sens + rng.normal(0.0, read_e / sens * math.sqrt(k), n))


@pytest.mark.parametrize("signal,k,lifetime", [(4.0, 1, 0.0), (6.0, 4, 0.0), (8.0, 1, 5.0)])
def test_spam_prediction_matches_monte_carlo(signal, k, lifetime):
    g, sens, read_e, lam_p, lam_s = 300.0, 4.0, 60.0, 0.004, 0.01
    rng = np.random.default_rng(int(signal * 10 + k))
    n = 400_000
    dark = shots(rng, n, 0.0, g, sens, read_e, lam_p, lam_s, k)
    bright = shots(rng, n, signal, g, sens, read_e, lam_p, lam_s, k, bright_lifetime=lifetime)
    mc = threshold_fidelity(bright, dark)
    model = {"sigma": read_e / sens, "g": g / sens, "lam_p": lam_p, "lam_s": lam_s,
             "a": 1.0 / sens}
    p = predict_spam(model, signal, n_pixels=k, bright_lifetime=lifetime,
                     duration=1.0 if lifetime else 0.0)
    err_mc = 1.0 - mc["fidelity"]
    tol = 3.0 * math.sqrt(err_mc / n) + 0.01 * err_mc
    assert abs(p["error"] - err_mc) < tol, (p, mc)
    # the error is flat around its best threshold: the thresholds are close
    assert abs(p["threshold"] - mc["threshold"]) <= 0.15 * g / sens + 2


def test_detection_time_binning_and_dark_current():
    model = {"sigma": 15.0, "g": 75.0, "lam_p": 0.004, "lam_s": 0.01, "a": 0.25}
    rate = 4000.0                                          # photo-electrons/s in the ROI
    best = shortest_detection(model, rate, 1e-3)
    t = best["time"]
    assert best["reached"] and best["error"] <= 1e-3 and 2e-4 < t < 0.03
    before, at = detection_curve(model, rate, [0.97 * t, t])
    assert before["error"] > 1e-3 >= at["error"]
    # errors fall with the detection time
    errs = [c["error"] for c in detection_curve(model, rate, [2e-4, 5e-4, 1e-3, 2e-3])]
    assert all(a > b for a, b in zip(errs, errs[1:]))
    # depumping: the error has a minimum, and the target may be out of reach
    dep = shortest_detection(model, rate, 1e-3, bright_lifetime=2e-3)
    assert dep["best_error"] > best["best_error"] and dep["best_time"] < 0.03
    assert dep["time"] is None or dep["time"] > t
    # binning adds parallel CIC up, not serial CIC or read noise; dark current
    # comes in with the exposure
    b4 = scaled_model(model, (4, 4), (1, 1), 0.002, 0.001, dark_current=0.5)
    assert b4["lam_p"] == pytest.approx(16 * 0.004 + 16 * 0.5 * 0.001)
    assert b4["lam_s"] == model["lam_s"] and b4["sigma"] == model["sigma"]
    assert shortest_detection(b4, rate, 1e-3)["time"] > t
