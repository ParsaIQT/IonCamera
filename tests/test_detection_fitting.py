import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from ioncam.analysis.detection import Detector
from ioncam.analysis.fitting import fit_gaussians
from ioncam.camera.base import Frame
from ioncam.camera.simulator import SimulatedCamera
from ioncam.config import Config


def _grab(cam, settings, n, t0=1000.0):
    applied = cam.configure(settings, "live")
    return [Frame(cam._render(applied, t0 + k), k, t0 + k, applied) for k in range(n)]


def _match(found, truth, r=1.5):
    if len(found) == 0:
        return np.zeros((0, 2)), 0, len(truth)
    d = np.linalg.norm(found[:, None, :] - truth[None, :, :], axis=2)
    i, j = linear_sum_assignment(d)
    ok = d[i, j] < r
    return found[i[ok]] - truth[j[ok]], len(found) - ok.sum(), len(truth) - ok.sum()


@pytest.fixture
def setup():
    cfg = Config()
    cfg.simulator.seed = 3
    cfg.detection.psf_sigma = 1.3
    cam = SimulatedCamera(cfg.simulator)
    return cfg, cam, Detector(cfg.detection)


def test_positions_accurate_and_complete(setup):
    cfg, cam, det = setup
    cam.scene.set_n_ions(4)
    cam.scene.set_scale(7.0)
    truth = np.array(cam.scene.state()["positions"])
    errs, fp, miss = [], 0, 0
    for f in _grab(cam, cfg.live.replace(exposure=0.05, em_gain=3), 10):
        r = det.analyze(f)
        e, a, b = _match(r.positions, truth)
        errs.append(e)
        fp += a
        miss += b
    e = np.vstack(errs)
    assert fp == 0 and miss == 0
    assert np.sqrt((e ** 2).mean()) < 0.05
    r = det.analyze(_grab(cam, cfg.live, 1)[0])
    assert r.focus == pytest.approx(np.sqrt(1.3 ** 2 + 1 / 12), abs=0.08)


@pytest.mark.parametrize("gain", [1, 50, 300])
def test_false_alarm_rate_on_empty_frames(setup, gain):
    cfg, cam, det = setup
    cam.scene.set_n_ions(0)
    n = 0
    for f in _grab(cam, cfg.live.replace(exposure=0.02, em_gain=gain), 15):
        r = det.analyze(f)
        n += len(r.spots) + len(r.clouds)
    assert n <= 2          # a few isolated noise peaks at most; the tracker debounces them


def test_cloud_vs_ion(setup):
    cfg, cam, det = setup
    cam.scene.set_n_ions(0)
    cam.scene.load_ion(1, cloud_time=1e9)
    r = det.analyze(_grab(cam, cfg.live.replace(exposure=0.05, em_gain=50), 1)[0])
    assert len(r.clouds) == 1 and len(r.spots) == 0
    assert r.clouds[0].width > 3.0


def test_single_pixel_spike_rejected(setup):
    cfg, cam, det = setup
    cam.scene.set_n_ions(1)
    f = _grab(cam, cfg.live.replace(exposure=0.05, em_gain=50), 1)[0]
    data = f.data.copy()
    data[30, 40] = 60000          # cosmic ray far from the ion
    r = det.analyze(Frame(data, 1, 0.0, f.applied))
    assert len(r.spots) == 1 and r.spikes >= 1


def test_static_background_keeps_ions_and_removes_hot_pixels(setup):
    cfg, cam, det = setup
    cam.scene.set_n_ions(2)
    frames = _grab(cam, cfg.live.replace(exposure=0.05, em_gain=50), 10)
    for f in frames:
        f.data[100, 100] = 9000            # a hot pixel in every frame
    r0 = det.analyze(frames[0])
    assert any(abs(s.x - 100) < 1 and abs(s.y - 100) < 1 for s in r0.spots + []) or \
        r0.spikes > 0
    det.learn_static_background(frames, r0.positions[[0, 1]] if len(r0.positions) > 2
                                else r0.positions, mask_radius=5.0)
    r1 = det.analyze(frames[1])
    near = [s for s in r1.spots if min(np.hypot(s.x - p[0], s.y - p[1])
                                       for p in r0.positions) < 1.0]
    assert len(near) >= 2                  # ions were masked, not learned as background
    assert min(s.snr for s in near) > 0.5 * min(s.snr for s in r0.spots[:2])
    assert not any(abs(s.x - 100) < 2 and abs(s.y - 100) < 2 for s in r1.spots)


def test_analysis_roi_crop(setup):
    cfg, cam, det = setup
    cam.scene.set_n_ions(3)
    f = _grab(cam, cfg.live, 1)[0]
    x, y = cam.scene.state()["positions"][0]
    r = det.analyze(f, analysis_roi=(int(x) - 3, int(y) - 10, int(x) + 4, int(y) + 10))
    assert len(r.spots) == 1


@pytest.mark.parametrize("n_ions", [4, 9])
@pytest.mark.parametrize("mode", ["live", "locate"])
def test_tight_camera_roi(setup, n_ions, mode):
    """A camera ROI hugging the crystal: the ions are a large part of the
    frame (and the whole upper tail of its pixel distribution) - they must
    not inflate the noise estimate and hide themselves."""
    cfg, cam, det = setup
    cam.scene.set_n_ions(n_ions)
    cam.scene.set_scale(6.0)
    truth = np.array(cam.scene.state()["positions"])
    settings = getattr(cfg, mode)
    full = det.analyze(_grab(cam, settings, 1)[0])
    x0, y0 = np.floor(truth.min(axis=0)).astype(int) - 8
    x1, y1 = np.ceil(truth.max(axis=0)).astype(int) + 9
    # crystal +- 8 px: the ions fill the frame (the noise is then overestimated,
    # there is hardly any background left to measure it on)
    for f in _grab(cam, settings.replace(roi=(x0, y0, x1, y1)), 3):
        r = det.analyze(f)
        _, fp, miss = _match(r.positions, truth)
        assert fp == 0 and miss == 0, len(r.spots)
    # (few pixels: single-frame noise estimates scatter by ~10 %)
    roi = (x0 - 20, y0 - 12, x1 + 20, y1 + 12)
    noise = []
    for f in _grab(cam, settings.replace(roi=roi), 5):
        r = det.analyze(f)
        _, fp, miss = _match(r.positions, truth)
        assert fp == 0 and miss == 0, len(r.spots)
        noise.append(r.noise_f)
    assert np.median(noise) < 1.3 * full.noise_f
    cam.scene.set_n_ions(0)
    noise = []
    for f in _grab(cam, settings.replace(roi=(x0, y0, x1, y1)), 5):
        r = det.analyze(f)
        assert len(r.spots) == 0
        noise.append(r.noise_f)
    assert np.median(noise) < 1.3 * full.noise_f


def test_joint_fit_dense_chain():
    cfg = Config()
    cfg.simulator.seed = 5
    cam = SimulatedCamera(cfg.simulator)
    cam.scene.set_n_ions(9)
    cam.scene.set_scale(5.5)        # 3.3 px minimum spacing (2.5 sigma)
    truth = np.array(cam.scene.state()["positions"])
    applied = cam.configure(cfg.locate, "locate")
    img = np.mean([cam._render(applied, 100.0 + k).astype(float) for k in range(2)], axis=0)
    guesses = truth + np.random.default_rng(0).normal(0, 0.4, truth.shape)
    fit = fit_gaussians(img, guesses, 1.4)
    assert fit.success
    order = np.argsort(fit.positions[:, 0])
    assert np.abs(fit.positions[order] - truth).max() < 0.03
    assert fit.sigma == pytest.approx((1.3, 1.3), abs=0.01)


def test_fit_jacobian_matches_numeric():
    import ioncam.analysis.fitting as F
    captured = {}
    orig = F.least_squares

    def spy(fun, x0, jac=None, **kw):
        captured.update(fun=fun, jac=jac, x0=x0)
        return orig(fun, x0, jac=jac, **kw)

    F.least_squares = spy
    try:
        img = np.random.default_rng(0).normal(100, 3, (20, 30))
        F.fit_gaussians(img, [(10.3, 9.7), (15.1, 10.2)], 1.3)
    finally:
        F.least_squares = orig
    x0 = captured["x0"] + np.r_[0, 0.1, -0.1, 50, 70, 0.2, -0.3, 0.1, 0.2]
    ja = captured["jac"](x0)
    eps = 1e-6
    jn = np.column_stack([(captured["fun"](x0 + eps * e) - captured["fun"](x0 - eps * e)) / (2 * eps)
                          for e in np.eye(len(x0))])
    assert np.abs(ja - jn).max() < 1e-6 * max(1.0, np.abs(jn).max())
