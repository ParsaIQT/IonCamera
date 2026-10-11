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
    """Realistic simulator (13k photons/s per ion, blurred PSF with a halo,
    laser scatter, surface defects); the detector knows the PSF as a
    calibration (locate) would have measured it."""
    cfg = Config()
    cfg.simulator.seed = 3
    cfg.detection.psf_sigma = cfg.simulator.psf_sigma
    cfg.detection.psf_halo_fraction = cfg.simulator.psf_halo_fraction
    cfg.detection.psf_halo_sigma = cfg.simulator.psf_halo_sigma
    cam = SimulatedCamera(cfg.simulator)
    return cfg, cam, Detector(cfg.detection)


def test_positions_accurate_and_complete(setup):
    """~470 photo-electrons per ion and frame: every ion found, positions at
    the photon limit (a few tenths of a pixel for a 1.8 px PSF)."""
    cfg, cam, det = setup
    cam.scene.set_n_ions(4)
    cam.scene.set_scale(7.0)
    truth = np.array(cam.scene.state()["positions"])
    errs, fp, miss = [], 0, 0
    for f in _grab(cam, cfg.live.replace(exposure=0.05, em_gain=100), 10):
        r = det.analyze(f)
        e, a, b = _match(r.positions, truth)
        errs.append(e)
        fp += a
        miss += b
    e = np.vstack(errs)
    assert fp == 0 and miss == 0
    assert np.sqrt((e ** 2).mean()) < 0.25
    r = det.analyze(_grab(cam, cfg.live, 1)[0])
    assert r.focus == pytest.approx(np.sqrt(1.8 ** 2 + 1 / 12), abs=0.2)


@pytest.mark.parametrize("gain", [1, 50, 300])
def test_false_alarm_rate_on_empty_frames(setup, gain):
    cfg, cam, det = setup
    cam.scene.set_n_ions(0)
    n = 0
    for f in _grab(cam, cfg.live.replace(exposure=0.02, em_gain=gain), 15):
        r = det.analyze(f)
        n += len(r.spots) + len(r.clouds)
    assert n <= 2          # a few isolated noise peaks at most; the tracker debounces them


@pytest.mark.parametrize("seed", [3, 4, 5])
def test_cloud_vs_ion(setup, seed):
    """A hot, uncrystallised ion is one broad cloud - not a cluster of faint
    'ions' (shot-noise lumps) - and an ion next to it stays an ion.  (A single
    hot ion is only ~30 photons/s/px: visible per frame in a long exposure; in
    short frames the tracker's temporal integration finds it.)"""
    cfg, cam, det = setup
    cam.scene.set_params(surface="clean", surface_seed=seed)
    cam.scene.set_n_ions(1)
    ion = np.array(cam.scene.state()["positions"])
    cam.scene.load_ion(1, cloud_time=1e9)
    cloud = np.array(cam.scene.state()["clouds"])[0]
    # (around the trap: elsewhere defocused surface defects are broad blobs
    # too in such a long exposure - always the same ones, which the tracker and
    # the learned background take care of)
    x, y = ion[0]
    trap = (int(x) - 40, int(y) - 40, int(x) + 40, int(y) + 40)
    for f in _grab(cam, cfg.live.replace(exposure=0.5, em_gain=50), 3, t0=1000.0 + seed):
        r = det.analyze(f, analysis_roi=trap)
        assert len(r.clouds) == 1 and len(r.spots) == 1, (len(r.clouds), len(r.spots))
        c = r.clouds[0]
        assert c.width > 3.0 and np.hypot(c.x - cloud[0], c.y - cloud[1]) < 2 * c.width
        assert np.hypot(*(r.positions[0] - ion[0])) < 1.0


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


@pytest.mark.parametrize("ions_dark", [False, True])
def test_learned_background_removes_surface_defects(setup, ions_dark):
    """A dirty surface with dust close to focus: defects that look like ions.
    The static background, learned once on the full sensor (with the ions
    masked, or dark - repumper off), removes them in any later ROI and
    binning; the ions stay."""
    cfg, cam, det = setup
    cam.scene.set_params(surface="dirty", defect_max_height_um=40.0, ion_height_um=3.0)
    live = cfg.live.replace(roi=(150, 90, 380, 250))
    cam.scene.set_n_ions(0)
    assert sum(len(det.analyze(f).spots) for f in _grab(cam, live, 5, t0=2000)) >= 20
    cam.scene.set_n_ions(5)
    cam.scene.set_scale(7.0)      # resolvable in single frames despite the scatter's noise
    truth = np.array(cam.scene.state()["positions"])
    cam.scene.set_repumper(not ions_dark)
    sb = det.learn_static_background(_grab(cam, cfg.live, 50, t0=3000),
                                     None if ions_dark else truth)
    cam.scene.set_repumper(True)
    assert sb.region == (0, 0, 512, 512)
    for settings in (live, live.replace(hbin=2, vbin=2)):
        fp = miss = 0
        for f in _grab(cam, settings, 5, t0=4000):
            r = det.analyze(f)
            _, a, b = _match(r.positions, truth, r=2.0)
            fp, miss = fp + a, miss + b
        assert miss == 0 and fp <= 2, (settings.hbin, fp, miss)
    assert 0.8 < sb.scale < 1.2
    # the laser 30 % brighter: the subtracted structure follows
    cam.scene.set_params(laser_power=1.3)
    for f in _grab(cam, live, 15, t0=5000):
        det.analyze(f)
    assert sb.scale == pytest.approx(1.3, abs=0.12)


def test_analysis_roi_crop(setup):
    cfg, cam, det = setup
    cam.scene.set_n_ions(3)
    f = _grab(cam, cfg.live, 1)[0]
    x, y = cam.scene.state()["positions"][0]
    r = det.analyze(f, analysis_roi=(int(x) - 3, int(y) - 10, int(x) + 4, int(y) + 10))
    assert len(r.spots) == 1


@pytest.mark.parametrize("n_ions, scale", [(4, 6.0), (7, 8.0)])
@pytest.mark.parametrize("mode", ["live", "locate"])
@pytest.mark.parametrize("gain", [None, 300])
def test_tight_camera_roi(setup, n_ions, scale, mode, gain):
    """A camera ROI hugging the crystal: the blurred ions fill the frame (and
    the whole upper tail of its pixel distribution) - they must not inflate
    the noise estimate and hide themselves, nor may a photon-counting frame
    (EM gain 300: just as skewed) be taken for one full of ions."""
    if gain and mode == "locate":
        pytest.skip("locate is a low-gain mode")
    cfg, cam, det = setup
    cam.scene.set_params(surface="clean")     # (surface defects: see test_learned_background)
    cam.scene.set_n_ions(n_ions)
    cam.scene.set_scale(scale)
    truth = np.array(cam.scene.state()["positions"])
    settings = getattr(cfg, mode)
    if gain:
        settings = settings.replace(em_gain=gain)
    full = det.analyze(_grab(cam, settings, 1)[0])
    # the noise where the crystal is (the laser scatter is brightest there)
    cu, cv = full.to_crop(truth[:, 0], truth[:, 1])
    local = float(np.median(full.noise_f_map[np.round(cv).astype(int),
                                             np.round(cu).astype(int)]))
    x0, y0 = np.floor(truth.min(axis=0)).astype(int) - 8
    x1, y1 = np.ceil(truth.max(axis=0)).astype(int) + 9
    # crystal +- 8 px: the ions fill the frame (the noise is then overestimated,
    # there is hardly any background left to measure it on); a crystal +- 20 px.
    # A dense chain may show one extra component in an odd frame (the tracker
    # needs persistence) - never a missing ion.
    extra, noise = 0, []
    for roi, n in (((x0, y0, x1, y1), 3), ((x0 - 20, y0 - 12, x1 + 20, y1 + 12), 5)):
        for f in _grab(cam, settings.replace(roi=roi), n):
            r = det.analyze(f)
            _, fp, miss = _match(r.positions, truth)
            assert miss == 0, len(r.spots)
            extra += fp
            if n == 5:
                noise.append(r.noise_f)
    assert extra <= 1
    assert np.median(noise) < 1.3 * local
    cam.scene.set_n_ions(0)
    noise, spots = [], 0
    for f in _grab(cam, settings.replace(roi=(x0, y0, x1, y1)), 5):
        r = det.analyze(f)
        spots += len(r.spots)
        noise.append(r.noise_f)
    assert spots <= (1 if gain else 0)
    assert np.median(noise) < 1.3 * local


def test_joint_fit_dense_chain():
    """Joint core + halo fit of a chain denser than the PSF (3.3 px spacing,
    1.8 sigma): unbiased positions and the true PSF, given enough light."""
    cfg = Config()
    cfg.simulator.seed = 5
    cfg.simulator.photon_rate = 2.0e5         # test the model, not the noise
    cam = SimulatedCamera(cfg.simulator)
    cam.scene.set_n_ions(9)
    cam.scene.set_scale(5.5)
    truth = np.array(cam.scene.state()["positions"])
    applied = cam.configure(cfg.locate, "locate")
    img = np.mean([cam._render(applied, 100.0 + k).astype(float) for k in range(4)], axis=0)
    guesses = truth + np.random.default_rng(0).normal(0, 0.4, truth.shape)
    fit = fit_gaussians(img, guesses, 1.8, halo=True)
    assert fit.success
    order = np.argsort(fit.positions[:, 0])
    # so dense that neighbours trade light: ~0.08 px along the chain even with
    # 240k photo-electrons per ion (noise-free images fit to < 0.001 px)
    assert np.abs(fit.positions[order] - truth).max() < 0.35
    # ... and the reported uncertainties say so (EMCCD noise law, sandwich)
    pulls = (fit.positions[order] - truth) / fit.position_errors[order]
    assert 0.6 < np.sqrt(np.mean(pulls ** 2)) < 1.6
    assert fit.sigma == pytest.approx((1.8, 1.8), abs=0.03)
    assert fit.halo_fraction == pytest.approx(0.15, abs=0.01)
    assert fit.halo_sigma == pytest.approx(4.5, abs=0.15)


@pytest.mark.parametrize("halo", [False, True])
def test_fit_jacobian_matches_numeric(halo):
    import ioncam.analysis.fitting as F
    captured = {}
    orig = F.least_squares

    def spy(fun, x0, jac=None, **kw):
        if not captured:                  # the first (core + halo) fit
            captured.update(fun=fun, jac=jac, x0=x0)
        return orig(fun, x0, jac=jac, **kw)

    F.least_squares = spy
    try:
        img = np.random.default_rng(0).normal(100, 3, (24, 34))
        F.fit_gaussians(img, [(12.3, 11.7), (17.1, 12.2)], 1.3, halo=halo)
    finally:
        F.least_squares = orig
    x0 = captured["x0"] + np.r_[0, 0.1, -0.1, 50, 70, 0.2, -0.3, 0.1, 0.2,
                                [0.05, 0.4][:2 * halo]]
    ja = captured["jac"](x0)
    eps = 1e-6
    jn = np.column_stack([(captured["fun"](x0 + eps * e) - captured["fun"](x0 - eps * e)) / (2 * eps)
                          for e in np.eye(len(x0))])
    assert np.abs(ja - jn).max() < 1e-6 * max(1.0, np.abs(jn).max())


def test_locate_merges_an_ion_split_in_two():
    """A locate exposure where the detector, with its first live PSF estimate
    (sigma 1.69 px + 25 % halo; the ions' is 1.8 + 15 %), split the middle of
    3 ions into two spots of half an ion's light each: told 3 ions, locate
    merges them.  Real ions - further apart, an ion's light each - stay."""
    from ioncam.engine import _merge_split_spots
    spots = [[268.17, 168.01], [255.82, 168.03], [263.04, 167.77], [260.98, 168.25]]
    g = _merge_split_spots(spots, [9541, 9491, 5459, 5520], 3, 1.69)
    assert len(g) == 3 and g[np.argsort(g[:, 0])][1] == pytest.approx([262.0, 168.0], abs=0.15)
    assert len(_merge_split_spots([[0, 0], [6, 0]], [1e4, 1e4], 1, 1.8)) == 2
    assert len(_merge_split_spots([[0, 0], [2, 0], [9, 0]], [1e4, 1e4, 1e4], 2, 1.8)) == 3
