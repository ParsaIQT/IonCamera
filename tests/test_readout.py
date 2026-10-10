import numpy as np
import pytest

from ioncam.analysis.chain import equilibrium_positions
from ioncam.analysis.readout import (PlanError, binning_advice, compare_grabber_frame,
                                     fit_threshold, grabber_roi_sum_exact, grabber_roi_sums,
                                     map_rect, plan_readout, validate_roi, verify_plan)
from ioncam.camera.simulator import SimulatedCamera
from ioncam.config import Config

SENSOR = (512, 512)


def chain(n, scale=7.4, center=(262.0, 168.0), angle=0.0):
    u = equilibrium_positions(n)
    a = np.deg2rad(angle)
    return np.column_stack([center[0] + scale * u * np.cos(a), center[1] + scale * u * np.sin(a)])


def test_single_ion_matches_hand_made_setup():
    """One ion at 4x4 binning: 8x8 px AQROI = 2x2 super-pixels around the ion
    (like the AQROI 257,264,161,168 used with Solis)."""
    p = plan_readout([(260.6, 164.3)], (1.3, 1.3), SENSOR, 4, 4, frame_margin=0,
                     background_roi=False)
    d = p.to_dict()
    x0, y0, x1, y1 = d["aqroi"]
    assert (x1 - x0, y1 - y0) == (8, 8) and d["frame_size"] == [2, 2]
    assert d["grabber_rois"] == [[0, 0, 2, 2]]
    assert p.containment[0] > 0.99
    # the ion sits at the centre of the 8x8 block
    assert abs((x0 + x1 - 1) / 2 - 260.6) <= 0.6 and abs((y0 + y1 - 1) / 2 - 164.3) <= 0.6
    assert d["aqroi_solis"] == [x0 + 1, x1, y0 + 1, y1]


@pytest.mark.parametrize("n, scale, angle, hb, vb", [
    (2, 7.4, 0, 4, 4), (3, 7.4, 0, 4, 4), (5, 7.4, 0, 2, 2), (8, 9.0, 0, 1, 4),
    (4, 7.4, 25, 2, 2), (4, 7.4, 90, 4, 4), (12, 10.0, 0, 2, 4),
])
def test_plan_geometry_invariants(n, scale, angle, hb, vb):
    pos = chain(n, scale, angle=angle)
    p = plan_readout(pos, (1.3, 1.3), SENSOR, hb, vb)
    x0, y0, x1, y1 = p.roi
    w, h = p.frame_size
    assert (x1 - x0) == w * hb and (y1 - y0) == h * vb       # aligned to the binning
    assert 0 <= x0 < x1 <= 512 and 0 <= y0 < y1 <= 512
    rois = p.grabber_rois
    assert len(rois) == n + 1
    for r in rois:
        assert validate_roi(r, w, h) == []
    for i in range(len(rois)):                              # no overlapping ROIs
        for j in range(i + 1, len(rois)):
            a, b = rois[i], rois[j]
            assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])
    for i, (x, y) in enumerate(pos):                        # every ion inside its ROI
        u0, v0, u1, v1 = p.ion_rois[i]
        assert x0 + u0 * hb - 0.5 <= x <= x0 + u1 * hb - 0.5
        assert y0 + v0 * vb - 0.5 <= y <= y0 + v1 * vb - 0.5
    assert np.allclose(np.diag(p.crosstalk_matrix), p.containment)


def test_roomy_layouts_meet_targets():
    p = plan_readout(chain(3), (1.3, 1.3), SENSOR, 2, 2)
    assert min(p.containment) > 0.9 and max(p.crosstalk) < 0.05 and not p.warnings
    p = plan_readout(chain(5, 7.4), (1.3, 1.3), SENSOR, 1, 4)
    assert min(p.containment) > 0.9 and max(p.crosstalk) < 0.05


def test_crowded_layout_warns_and_advises():
    pos = chain(5, 6.0)
    p = plan_readout(pos, (1.3, 1.3), SENSOR, 4, 4)
    assert any("no placement" in w for w in p.warnings)
    adv = binning_advice(pos, (1.3, 1.3), SENSOR)
    assert any(a.get("ok") for a in adv)


def test_flip_mapping():
    assert map_rect((1, 0, 3, 2), 8, 4, True, False) == (5, 0, 7, 2)
    assert map_rect((1, 0, 3, 2), 8, 4, False, True) == (1, 2, 3, 4)
    p = plan_readout(chain(3), (1.3, 1.3), SENSOR, 4, 4, flip_x=True)
    q = plan_readout(chain(3), (1.3, 1.3), SENSOR, 4, 4)
    w = p.frame_size[0]
    assert p.grabber_rois[0] == (w - q.grabber_rois[0][2], q.grabber_rois[0][1],
                                 w - q.grabber_rois[0][0], q.grabber_rois[0][3])


def test_errors():
    with pytest.raises(PlanError):
        plan_readout([], (1.3, 1.3), SENSOR, 4, 4)
    with pytest.raises(PlanError):
        plan_readout(chain(17, 10.0), (1.3, 1.3), SENSOR, 1, 1, max_rois=16)


def _sim_frames(n_ions=5, n_frames=20, roi=None, plan_kw=None):
    cfg = Config()
    cfg.simulator.seed = 4
    cfg.simulator.n_ions = n_ions
    cam = SimulatedCamera(cfg.simulator)
    pos = np.array(cam.scene.state()["positions"])
    plan = plan_readout(pos, (cfg.simulator.psf_sigma,) * 2, SENSOR, 4, 4, **(plan_kw or {}))
    a = cam.configure(cfg.readout.replace(roi=roi or plan.roi, trigger="internal"), "verify")
    return plan, [cam._render(a, 1000.0 + k) for k in range(n_frames)]


def test_gateware_model_and_plan_sums_agree():
    plan, frames = _sim_frames()
    for f in frames[:5]:
        exact = [grabber_roi_sum_exact(f, *r) for r in plan.grabber_rois]
        assert exact == list(grabber_roi_sums(f, plan.grabber_rois))
        assert exact == list(plan.roi_sums(f)[0])


def test_gateware_quirk_x1_beyond_width_sums_rows():
    f = np.arange(6 * 8, dtype=np.int64).reshape(6, 8)
    assert grabber_roi_sum_exact(f, 2, 1, 5, 3) == f[1:3, 2:5].sum()
    assert grabber_roi_sum_exact(f, 0, 0, 8, 6) == f.sum()            # x1 == W is fine
    # x1 > W: the x flag (set at x == x0 in every line, already in line 0) is never
    # cleared, so the ROI silently becomes the full rows y0..y1-1
    assert grabber_roi_sum_exact(f, 2, 1, 9, 3) == f[1:3, :].sum()
    assert validate_roi((2, 1, 9, 3), 8, 6)


def test_verification_ok_and_detects_misplaced_aqroi():
    plan, frames = _sim_frames()
    v = verify_plan(plan, frames)
    assert v["ok"], v
    assert all(i["snr"] > 5 for i in v["ions"])
    # the camera AQROI one super-pixel off: ions are no longer in their ROIs
    x0, y0, x1, y1 = plan.roi
    plan2, frames2 = _sim_frames(roi=(x0 + 4, y0 + 4, x1 + 4, y1 + 4))
    v2 = verify_plan(plan, frames2)
    assert not v2["ok"]


def test_threshold_fit():
    rng = np.random.default_rng(0)
    dark = rng.normal(400, 30, 3000)
    bright = rng.normal(1200, 150, 2000)
    r = fit_threshold(np.r_[dark, bright])
    assert r["ok"] and 450 < r["threshold"] < 900
    assert r["bright_fraction"] == pytest.approx(0.4, abs=0.02)
    assert r["fidelity"] > 0.99
    assert not fit_threshold(rng.normal(1000, 100, 3000))["ok"]      # one population
    assert not fit_threshold([1, 2, 3])["ok"]


def test_grabber_mapping_check():
    plan, frames = _sim_frames(n_ions=3)
    exp = np.mean(frames, axis=0)
    exp[1, 1] += 5000                    # make the image asymmetric
    rng = np.random.default_rng(1)
    for name, m in [("identity", exp), ("flip_x", exp[:, ::-1]), ("flip_y", exp[::-1]),
                    ("rot180", exp[::-1, ::-1])]:
        r = compare_grabber_frame(exp, m + rng.normal(0, 30, m.shape))
        assert r["best"] == name
