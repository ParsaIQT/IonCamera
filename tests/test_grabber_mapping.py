"""How the camera puts its binned frame on Camera Link (the Grabber's pixels):
the geometry, its fit to a frame the Grabber saw, and the engine measuring it
for the camera in use - an iXon Ultra 888 whose ROIs at x = 1 and x = 3 both
read the first super-pixel: its lines are not binned on Camera Link."""

import json
import os
import time

import numpy as np
import pytest

from ioncam.analysis.readout import (GrabberGeometry, camera_link_frame,
                                     infer_grabber_geometry, plan_readout)
from ioncam.config import Config
from ioncam.server import build

G = GrabberGeometry
SENSOR = (1024, 1024)

# a 4 x 4 frame of super-pixels with two ions (USB counts, bias 500)
FRAME = np.array([[0, 900, 0, 30], [0, 0, 20, 0], [10, 0, 0, 700], [0, 0, 0, 0]], float) + 500


def test_geometry_maps_rois():
    """A super-pixel's Grabber ROI: as many stream pixels as it has, after the
    offset, from the other side if mirrored (the offset comes first)."""
    assert G().rect((1, 0, 2, 1), 4, 4, 4, 4) == (1, 0, 2, 1)
    assert G(0, 1).rect((1, 0, 2, 1), 4, 4, 4, 4) == (4, 0, 8, 1)        # x not binned
    assert G(0, 1).frame(4, 4, 4, 4) == (16, 4)
    assert G(1, 1, 3, 1).rect((1, 0, 2, 1), 4, 4, 4, 4) == (4, 1, 5, 2)   # pre-scan
    assert G(0, 1, 2, 0, True).rect((0, 0, 1, 1), 4, 4, 4, 4) == (14, 0, 18, 1)
    assert G(0, 0).scales(3, 2) == (3, 2) and G(2, 1).scales(4, 4) == (2, 1)
    assert G().identity and not G(0, 1).identity
    assert "unbinned" in G(0, 1).describe(4, 4)
    assert G.from_dict(G(0, 1, 2, 0, True).to_dict()) == G(0, 1, 2, 0, True)


def test_plan_maps_its_rois_with_the_geometry():
    """The planner hands ARTIQ ROIs in the stream's pixels; the frame it
    expects in the core log is the stream's."""
    pos = [(262.0, 168.0)]
    plain = plan_readout(pos, (0.8, 0.8), SENSOR, 4, 4, background_roi=True)
    wide = plan_readout(pos, (0.8, 0.8), SENSOR, 4, 4, background_roi=True, grabber=G(0, 1))
    assert wide.ion_rois == plain.ion_rois and wide.roi == plain.roi
    for (x0, y0, x1, y1), (u0, v0, u1, v1) in zip(wide.grabber_rois, plain.sdk_rois()):
        assert (x0, y0, x1, y1) == (4 * u0, v0, 4 * u1, v1)
    d = wide.to_dict()
    assert d["grabber_frame"] == [4 * plain.frame_size[0], plain.frame_size[1]]
    assert d["grabber_geometry"] == G(0, 1).to_dict()


@pytest.mark.parametrize("geom", [G(), G(4, 1), G(4, 4), G(1, 1, 4, 0), G(1, 1, 3, 1),
                                  G(4, 1, 0, 0, True), G(1, 1, 0, 0, True, True), G(0, 1, 2, 0)])
def test_fit_finds_the_layout(geom):
    """Every layout - not binned, offset, mirrored, combined - is recovered
    from a noisy stream frame with the light of two ions."""
    rng = np.random.default_rng(1)
    stream = camera_link_frame(FRAME, geom, 4, 4, bias=500.0)
    res = infer_grabber_geometry(FRAME, stream + rng.normal(0, 3, stream.shape), 4, 4)
    sx, sy = geom.scales(4, 4)
    want = G(0 if sx == 4 else sx, 0 if sy == 4 else sy, geom.x_offset, geom.y_offset,
             geom.flip_x, geom.flip_y)
    assert res["ok"] and G.from_dict(res["geometry"]) == want, res
    assert res["gain"] == pytest.approx(1.0, abs=0.03)
    assert res["offset"] == pytest.approx((sx * sy - 1) * 500.0, abs=15)


def test_fit_says_when_it_cannot_tell():
    """Two equal ions on a diagonal of a 2 x 2 frame look the same turned by
    180 deg: not applied, with what to do; a frame without its light (wrong
    plan) explains nothing."""
    pair = np.array([[800, 50], [40, 780]], float) + 500
    res = infer_grabber_geometry(pair, camera_link_frame(pair, G(0, 0), 4, 4, 500.0), 4, 4)
    assert not res["ok"] and res["ambiguous"] and "asymmetric" in res["reason"]
    flat = np.full((4, 16), 500.0)
    res = infer_grabber_geometry(FRAME, flat + np.random.default_rng(2).normal(0, 3, flat.shape))
    assert not res["ok"] and res["reason"]


def _simulated_888(tmp_path, link, seed=5):
    cfg = Config()
    cfg.storage.data_dir = str(tmp_path)
    cfg.simulator.camera_model = "888"
    cfg.simulator.n_ions = 1
    cfg.simulator.seed = seed
    cfg.simulator.camera_link = list(link)
    cfg.tracker.confirm_time = 0.3
    return build(cfg, simulate=True)


def _shots(engine, api, n):
    """n triggered shots: the USB ROI sums, the Grabber's, the mean stream frame."""
    sim = engine.sim_camera()
    usb, grab, img = [], [], 0.0
    for _ in range(n):
        k0 = engine.monitor.shots
        sim.trigger(1, 0.0)
        t0 = time.time()
        while engine.monitor.shots < k0 + 1 and time.time() - t0 < 5:
            time.sleep(0.005)
        usb.append(engine._plan.roi_sums(sim.last_frame[None])[0])
        grab.append(api.sim_grabber_sums())
        img = img + np.array(api.sim_grabber_frame(), float)
    return np.array(usb, float), np.array(grab, float), img / n


def test_888_with_unbinned_camera_link(tmp_path):
    """A simulated 888 that sends its lines unbinned on Camera Link, 4 x 4
    binning, a 4 x 4 AQROI: the ROIs as planned for a binned stream read the
    wrong pixels (x = 1 lies in the first super-pixel); the mapping check
    finds the layout, keeps it for this camera and maps the ROIs anew - the
    Grabber's sums are then the ROIs' light (plus a bias per extra stream
    pixel, in grabber_threshold too); kept across a restart; forgotten on
    request."""
    engine, api = _simulated_888(tmp_path, [0, 1, 0, 0, 0, 0])
    engine.start()
    try:
        assert engine.wait_for_ions(1, timeout=40)["ok"]
        engine.calibrate_readout(n_ions=1, hbin=4, vbin=4)
        plan = engine.manual_aqroi(4, 4)
        assert plan["grabber_source"].startswith("assumed")
        assert plan["grabber_rois"][0][2] - plan["grabber_rois"][0][0] == 1
        usb, grab, img = _shots(engine, api, 25)
        assert grab[:, 0].mean() < 0.6 * usb[:, 0].mean()        # the wrong pixels
        assert img.shape == (4, 16)
        res = engine.check_grabber_mapping(img)
        assert res["ok"] and res["applied"], res
        assert G.from_dict(res["geometry"]) == G(0, 1)
        p = res["plan"]
        assert p["grabber_rois"] == [[4 * u0, v0, 4 * u1, v1] for u0, v0, u1, v1 in
                                     plan["ion_rois"] + [plan["background_roi"]]]
        assert p["grabber_frame"] == [16, 4] and p["grabber_source"].startswith("measured")
        bias = float(engine.sim_camera()._bias)
        assert p["grabber_counts"][1] == pytest.approx(3 * bias, rel=0.05)
        usb, grab, _ = _shots(engine, api, 25)
        np.testing.assert_allclose(grab, usb + 3 * bias, atol=3)
        # thresholds in Grabber counts
        stats = engine.monitor.stats()
        th = stats["rois"][0]["threshold"]
        if th.get("ok"):
            assert th["grabber_threshold"] == pytest.approx(th["threshold"] + 3 * bias, abs=1)
        try:
            from ioncam.gui.main_window import artiq_snippet
        except Exception:  # pragma: no cover - no usable Qt
            artiq_snippet = None
        if artiq_snippet is not None:
            text = artiq_snippet(engine.readout_plan())
            assert "16 x 4 Camera Link frame" in text and "unbinned" in text
            assert "NOT measured" not in text and "grabber_threshold" in text
        with open(os.path.join(str(tmp_path), "grabber_mapping.json"), encoding="utf-8") as f:
            saved = json.load(f)
        assert list(saved) == [engine._camera_id()]
    finally:
        engine.shutdown()
    # a restart with the same camera: measured, used at once
    engine, api = _simulated_888(tmp_path, [0, 1, 0, 0, 0, 0], seed=6)
    engine.start()
    try:
        m = api.get_grabber_mapping()
        assert m["source"].startswith("measured") and "unbinned" in m["summary"]
        assert engine.wait_for_ions(1, timeout=40)["ok"]
        plan = engine.calibrate_readout(n_ions=1, hbin=4, vbin=4)
        assert plan["grabber_source"].startswith("measured")
        assert plan["grabber_frame"][0] == 4 * plan["frame_size"][0]
        engine.forget_grabber_mapping()
        assert engine.readout_plan()["grabber_source"].startswith("assumed")
    finally:
        engine.shutdown()


@pytest.mark.parametrize("levels", [[0.4, 1.6, 0.7, 1.3, 1.0], [1.8, 0.5, 1.2, 0.4, 1.1]])
def test_fit_compares_each_pixel_with_its_own_shots(levels):
    """The mapping check reads the stream 15 pixels at a time, each batch in
    other shots - and an ion's brightness differs from batch to batch.  Against
    the USB frames of each pixel's own shots the layout is exact (r = 1);
    against the mean of all frames the light does not add up: refused (no
    layout, or two that fit alike), never a wrong layout applied."""
    geom, k = G(0, 1), 15
    h, w = FRAME.shape
    gh, gw = h, 4 * w
    usb = [(FRAME - 500) * lv + 500 for lv in levels]
    stream, batch = np.zeros((gh, gw)), np.zeros((gh, gw), int)
    for i in range(gh * gw):
        y, x = divmod(i, gw)
        stream[y, x] = camera_link_frame(usb[i // k], geom, 4, 4, bias=500.0)[y, x]
        batch[y, x] = i // k
    res = infer_grabber_geometry(np.array(usb), stream, 4, 4, pixel_batch=batch)
    assert res["ok"] and G.from_dict(res["geometry"]) == geom and res["r"] > 0.999, res
    assert res["gain"] == pytest.approx(1.0, abs=0.01)
    plain = infer_grabber_geometry(np.mean(usb, axis=0), stream, 4, 4)
    assert not plain["ok"] and plain["reason"], plain
    bad = infer_grabber_geometry(np.array(usb), stream, 4, 4, pixel_batch=batch[:, :-1])
    assert not bad["ok"] and "pixel_batch" in bad["reason"]
