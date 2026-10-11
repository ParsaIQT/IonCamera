"""End to end: engine + sipyco RPC server + clients (as ARTIQ would use it)."""

import os
import socket
import threading
import time

import numpy as np
import pytest

sipyco = pytest.importorskip("sipyco")
from sipyco.pc_rpc import Client  # noqa: E402

from ioncam.config import Config  # noqa: E402
from ioncam.server import ServerThread, build  # noqa: E402


def free_port():
    s = socket.socket()
    s.bind(("localhost", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def make_system(tmp_path, backend="simulator", **sim):
    cfg = Config()
    cfg.storage.data_dir = str(tmp_path)
    cfg.camera.backend = backend
    cfg.simulator.seed = 11
    cfg.simulator.n_ions = sim.pop("n_ions", 3)
    for k, v in sim.items():
        setattr(cfg.simulator, k, v)
    cfg.tracker.confirm_time = 0.3
    cfg.live.roi = (160, 100, 360, 240)
    cfg.live.exposure = 0.03
    engine, api = build(cfg, simulate=(backend == "simulator"))
    engine.start()
    port = free_port()
    server = ServerThread(api, ["localhost"], port).start()
    return engine, server, port


@pytest.fixture(scope="module")
def system(tmp_path_factory):
    engine, server, port = make_system(tmp_path_factory.mktemp("ioncam"))
    yield engine, port
    server.stop()
    engine.shutdown()


def client(port):
    return Client("localhost", port, "ioncam", timeout=60)


def test_rpc_introspection_and_basics(system):
    engine, port = system
    c = client(port)
    try:
        assert c.ping() is True
        doc = c.get_rpc_method_list()          # must not choke on annotations
        assert "calibrate_readout" in doc["methods"] and "wait_for_ions" in doc["methods"]
        info = c.get_info()
        assert info["simulated"] and info["sensor_shape"] == [512, 512]
        st = c.get_status()
        assert st["mode"] == "live"
        r = c.wait_for_ions(3, timeout=15)
        assert r["ok"] and r["n_bright"] == 3
        f = c.get_frame(0)
        assert f["image"].dtype == np.uint16 and f["roi"] == [160, 100, 360, 240]
        assert c.get_frame(f["index"]) is None or c.get_frame(f["index"])["index"] > f["index"]
        assert c.get_frame(0, bin=4)["image"].shape == (35, 50)
        ions = c.get_ions()
        assert len(ions) == 3 and all(isinstance(i["x"], float) for i in ions)
    finally:
        c.close_rpc()


def test_calibrate_arm_trigger_monitor(system):
    engine, port = system
    c = client(port)
    try:
        assert c.wait_for_ions(3, timeout=15)["ok"]
        plan = c.calibrate_readout(n_ions=3)
        assert c.get_status()["mode"] == "readout"
        assert plan["n_rois"] == 4 and plan["ion_mask"] == 0b111 and plan["full_mask"] == 0b1111
        for k in ("grabber_x0", "grabber_y0", "grabber_x1", "grabber_y1"):
            assert all(type(v) is int for v in plan[k])
        assert plan["verification"]["ok"]
        a = plan["applied"]
        assert a["settings"]["trigger"] == "external" and a["settings"]["camera_link"]
        assert a["exposure"] == pytest.approx(0.0015) and a["em_gain"] == 200
        assert c.get_grabber_roi_lists() == [plan["grabber_x0"], plan["grabber_y0"],
                                             plan["grabber_x1"], plan["grabber_y1"]]
        # the verification frames (ions shining) give the first shot histograms:
        # ion ROIs bright, the background pixels of the same frames dark
        h = c.get_readout_histograms()
        n_verify = plan["verification"]["n_frames"]
        assert h["source"] == "verify" and h["shots"] == n_verify
        assert [s["label"] for s in h["series"]] == ["ion 0", "ion 1", "ion 2", "bg"]
        assert all(r["ok"] and r["fidelity"] > 0.6 for r in h["ions"])
        assert [r["threshold"] for r in h["ions"]] == \
            [i["threshold"] for i in plan["verification"]["ions"]]
        assert c.sim_trigger(60, 0.004) == 60
        time.sleep(0.8)
        h = c.get_readout_histograms()               # the newer shots: triggered
        assert h["source"] == "trigger" and h["shots"] >= 60
        assert h["available"] == {"trigger": h["shots"], "verify": n_verify}
        assert c.get_readout_histograms("verify")["shots"] == n_verify
        counts = c.get_readout_counts(0)
        assert counts["next_shot"] >= 60
        assert counts["sums"][-1] == c.sim_grabber_sums()     # USB copy == Grabber view
        stats = c.get_readout_stats()
        assert len(stats["rois"]) == 4
        # ion ROI brighter than the background ROI, per super-pixel (the default
        # ion ROI is a single super-pixel, the background strip several)
        assert stats["rois"][0]["mean_per_pixel"] > stats["rois"][3]["mean_per_pixel"]
        assert c.is_readout_valid()
        c.disarm_readout()
        assert c.get_status()["mode"] == "live"
    finally:
        c.close_rpc()


def test_locate_ions_precise(system):
    engine, port = system
    c = client(port)
    try:
        assert c.wait_for_ions(3, timeout=15)["ok"]
        res = c.locate_ions(3)
        truth = np.array(c.sim_state()["positions"])
        pos = np.array(res["positions"])
        # photon limited: 2 x 0.5 s of 13k photons/s per ion, PSF 1.8 px + halo
        err = np.array(res["position_errors"])
        assert np.abs(pos - truth).max() < 0.2
        assert np.all(np.abs(pos - truth) < 4 * err + 0.02)
        assert res["psf_sigma"][0] == pytest.approx(1.8, abs=0.1)
        frac, sigma = res["psf_halo"]
        assert 0.05 < frac < 0.3 and 3.0 < sigma < 7.0
        # the PSF is used from now on (auto_psf): detection and readout planning
        cfg = c.get_config()["detection"]
        assert cfg["psf_halo_fraction"] == pytest.approx(frac, abs=1e-3)
        assert c.get_status()["mode"] == "live"      # previous mode restored
        # a camera ROI hugging the crystal works just as well (it is padded)
        x0, y0 = np.floor(truth.min(axis=0)).astype(int) - 4
        x1, y1 = np.ceil(truth.max(axis=0)).astype(int) + 5
        res = c.locate_ions(3, roi=[int(x0), int(y0), int(x1), int(y1)])
        assert np.abs(np.array(res["positions"]) - truth).max() < 0.2
    finally:
        c.close_rpc()


def test_wait_does_not_block_other_clients(system):
    engine, port = system
    c1, c2 = client(port), client(port)
    try:
        result = {}
        th = threading.Thread(target=lambda: result.update(
            ev=c1.wait_for_event(["never_happens"], None, 2.0)))
        th.start()
        n = 0
        t = time.time()
        while th.is_alive() and time.time() - t < 5:
            c2.get_status()
            n += 1
        th.join()
        assert result["ev"] is None and n > 20
    finally:
        c1.close_rpc()
        c2.close_rpc()


def test_loading_event_and_plan_invalidation(system):
    engine, port = system
    c = client(port)
    try:
        assert c.wait_for_ions(3, timeout=15)["ok"]
        last = c.get_last_event_id()
        c.sim_load_ion(1, 0.3)
        ev = c.wait_for_event(["ion_trapped"], last, 15)
        assert ev is not None and ev["n_bright"] == 4
        assert c.wait_for_ions(4, timeout=10)["ok"]
        if c.get_readout_plan() is not None:
            assert not c.is_readout_valid()
        c.sim_lose_ion(-1)
        ev = c.wait_for_event(["ion_lost"], ev["id"], 15)
        assert ev is not None and ev["details"]["after"] == 3
    finally:
        c.close_rpc()


def test_shuttling_with_hold(system):
    engine, port = system
    c = client(port)
    try:
        assert c.wait_for_ions(3, timeout=15)["ok"]
        last = c.get_last_event_id()
        c.hold_tracking("shuttle to zone B", 30)
        c.sim_shift(25.0, 0.0)
        time.sleep(0.5)
        c.sim_shift(-25.0, 0.0)
        time.sleep(0.3)
        c.resume_tracking()
        ev = c.wait_for_event(["hold_done"], last, 15)
        assert ev is not None and ev["details"] == {"before": 3, "after": 3}
        kinds = [e["kind"] for e in c.get_events(last)]
        assert "ion_lost" not in kinds
    finally:
        c.close_rpc()


def test_control_lease_and_config(system):
    engine, port = system
    c = client(port)
    try:
        c.take_control("artiq", "experiment 42", 60)
        assert c.get_control()["owner"] == "artiq"
        with pytest.raises(Exception):
            c.take_control("someone else", "", 60)
        assert c.release_control("artiq") is True
        assert c.get_control() is None
        assert c.set_config("tracker", "confirm_time", 0.3) == 0.3
        assert c.get_config()["tracker"]["confirm_time"] == 0.3
    finally:
        c.close_rpc()


def test_overexposure_protection(tmp_path):
    engine, server, port = make_system(tmp_path, n_ions=2, photon_rate=5e7)
    c = client(port)
    try:
        last = c.get_last_event_id()
        c.set_live(em_gain=300)
        ev = c.wait_for_event(["overexposure"], last, 10)
        assert ev is not None
        assert c.get_status()["settings"]["em_gain"] <= 1
    finally:
        c.close_rpc()
        server.stop()
        engine.shutdown()


def test_background_and_simulation_controls(tmp_path):
    """Over RPC: a dirty trap surface, learn the background with the ions dark
    (repumper off), see it used, view the corrected image, forget it."""
    engine, server, port = make_system(tmp_path)
    c = client(port)
    try:
        assert c.wait_for_ions(3, timeout=15)["ok"]
        p = c.sim_get_params()
        assert "scatter_rate" in p and "horrendous" in p["presets"]
        p = c.sim_set_params(surface="dirty", beam_angle_deg=30.0, laser_power=0.8)
        assert p["surface"] == "dirty" and p["beam_angle_deg"] == 30.0
        with pytest.raises(Exception):
            c.sim_set_params(no_such_knob=1)
        assert c.get_background() is None
        c.sim_set_repumper(False)
        info = c.learn_background(20, False)
        c.sim_set_repumper(True)
        assert info["n_frames"] == 20 and info["mask_ions"] is False
        assert info["region"] == [0, 0, 512, 512] and info["shot_noise"] > 0
        assert c.get_status()["mode"] == "live"          # live resumed
        assert c.get_status()["background"]["n_frames"] == 20
        fr, t = None, time.time()
        while (fr is None or fr["view"] != "corrected") and time.time() - t < 10:
            fr = c.get_frame(0, 1, None, "corrected")    # (non-live frames stay raw)
            time.sleep(0.1)
        assert fr["view"] == "corrected" and fr["image"].dtype == np.float32
        assert abs(float(np.median(fr["image"]))) < 5.0   # bias and scatter removed
        c.sim_move_camera(3.0, 0.0)
        c.clear_background()
        assert c.get_background() is None
    finally:
        c.close_rpc()
        server.stop()
        engine.shutdown()


def test_full_stack_on_emulated_andor_sdk(tmp_path):
    """Same workflow through the real AndorCamera driver (fake SDK underneath)."""
    engine, server, port = make_system(tmp_path, backend="fake_sdk")
    c = client(port)
    try:
        assert c.get_info()["model"] == "DU897_BV"
        assert c.wait_for_ions(3, timeout=15)["ok"]
        plan = c.calibrate_readout(n_ions=3)
        assert plan["verification"]["ok"]
        sdk = engine.camera._sdk
        assert ("SetCameraLinkMode", (1,)) in sdk.calls
        hb, vb, hs, he, vs, ve = plan["sdk_image"]
        assert ("SetImage", (hb, vb, hs, he, vs, ve)) in sdk.calls
        assert sdk.sim.trigger(20, 0.004) == 20
        time.sleep(0.5)
        assert c.get_readout_counts(0)["next_shot"] >= 20
    finally:
        c.close_rpc()
        server.stop()
        engine.shutdown()


def test_tight_trap_readout_by_hand(tmp_path):
    """PSF 1.2 px, l = 3.4 px (the lab's trap): coarse binning gives
    overlapping ROIs instead of an error; the AQROI is then moved by hand in
    1 px steps (the Grabber ROIs stay), re-planned and verified."""
    engine, server, port = make_system(tmp_path, n_ions=3, psf_sigma=1.2, length_scale_px=3.4)
    c = client(port)
    try:
        assert c.wait_for_ions(3, timeout=20)["ok"]
        plan = c.calibrate_readout(n_ions=3, hbin=8, vbin=8)
        assert c.get_status()["mode"] == "readout"
        assert plan["roi_labels"] == ["ion 0", "ion 1", "ion 2", "bg"]
        assert any("overlap" in w or "same ROI" in w for w in plan["warnings"])
        moved = c.nudge_readout(1, 0)
        x0, y0, x1, y1 = plan["aqroi"]
        assert moved["aqroi"] == [x0 + 1, y0, x1 + 1, y1]
        assert moved["grabber_rois"] == plan["grabber_rois"] and moved["manual_offset"] == [1, 0]
        assert moved["applied"]["roi"] == moved["aqroi"]          # re-armed with it
        assert c.get_status()["mode"] == "readout"
        fine = c.replan_readout(2, 2, None, [2, 2])      # 2x2 super-pixels per ion
        assert fine["binning"] == [2, 2] and min(fine["containment"]) > 0.6
        aq = fine["aqroi"]
        fixed = c.replan_readout(2, 2, [aq[0] - 4, aq[1] - 2, aq[2] + 4, aq[3] + 2])
        assert fixed["aqroi"] == [aq[0] - 4, aq[1] - 2, aq[2] + 4, aq[3] + 2]
        checked = c.verify_readout()
        assert checked["verification"]["ok"] and c.get_status()["mode"] == "readout"
        c.disarm_readout()
    finally:
        c.close_rpc()
        server.stop()
        engine.shutdown()


def test_locate_splits_a_merged_pair(tmp_path):
    """Two ions 2.8 px apart look like one spot: told how many to expect,
    locate fits two along the spot's elongation."""
    engine, server, port = make_system(tmp_path, n_ions=2, psf_sigma=1.2, length_scale_px=2.2)
    try:
        time.sleep(2.0)
        res = engine.locate_ions(2)
        truth = np.array(engine.sim_scene().state()["positions"])
        pos = np.array(res["positions"])
        assert res["n_ions"] == 2
        assert np.abs(pos[np.argsort(pos[:, 0])] - truth[np.argsort(truth[:, 0])]).max() < 0.3
    finally:
        server.stop()
        engine.shutdown()


def test_turning_the_trap_starts_afresh(tmp_path):
    """Simulator: turning the trap is like mounting a different one - the
    learned background (old orientation) is cleared, the crystal is tracked
    afresh (no "lost" or "trapped" for the jump) and the turned electrode
    scatter is not reported as ion clouds."""
    engine, server, port = make_system(tmp_path, n_ions=3, psf_sigma=1.2, length_scale_px=3.4)
    try:
        time.sleep(2.0)
        engine.learn_background(20)
        last = engine.events.last_id
        server.api.sim_set_params(trap_angle_deg=90)
        assert engine.background_info() is None
        time.sleep(4.0)
        kinds = [e["kind"] for e in engine.events.since(last)]
        assert "background_cleared" in kinds and "ions_detected" in kinds
        assert not {"cloud_detected", "ion_lost", "ion_trapped"} & set(kinds)
        assert engine.tracker.n_bright == 3 and engine.tracker.n_clouds == 0
        d = engine.tracker.positions()[-1] - engine.tracker.positions()[0]
        assert abs(d[0]) < 0.2 * abs(d[1])           # the chain now runs along y
    finally:
        server.stop()
        engine.shutdown()


def test_calibration_files_written(system, tmp_path_factory):
    engine, port = system
    d = os.path.join(engine.config.storage.data_dir, "calibrations")
    assert os.path.isdir(d) and any(f.endswith("_readout_plan.json") for f in os.listdir(d))
    assert os.path.exists(os.path.join(engine.config.storage.data_dir, "events.jsonl"))


def test_gui_smoke(tmp_path):
    pytest.importorskip("pyqtgraph")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtWidgets
        from ioncam.gui.backend import LocalBackend
        from ioncam.gui.main_window import MainWindow
        app = pg.mkQApp("ioncam-test")
    except Exception as e:  # pragma: no cover - no usable Qt
        pytest.skip("Qt not usable: {}".format(e))
    engine, server, port = make_system(tmp_path)
    backend = LocalBackend(server.api, server)
    try:
        win = MainWindow(backend, backend.call("get_config").result(10),
                         backend.call("get_info").result(10))
        win.show()
        t = time.time()
        while time.time() - t < 3.0:
            app.processEvents()
            time.sleep(0.01)
        assert win.image.img.image is not None
        assert "ion" in win.banner.count.text()
        x0, y0, x1, y1 = win.image.visible_sensor_rect()
        assert x0 <= 170 and x1 >= 350          # whole live ROI visible
        # brightness levels dragged by hand stay put (auto levels switch off)
        win.image.hist.item.region.setRegion((100.0, 400.0))
        t = time.time()
        while time.time() - t < 0.5:
            app.processEvents()
            time.sleep(0.01)
        assert not win.image.auto_levels and not win.auto_levels.isChecked()
        assert tuple(win.image.img.levels) == (100.0, 400.0)
        win.image.set_auto_levels(True)
        assert tuple(win.image.img.levels) != (100.0, 400.0)
        # the D-pad moves things on screen: through the display rotation
        win.image.set_orientation(90)
        # (90 deg clockwise: sensor +x points down, +y right on screen)
        assert win.image.screen_step(1, 0) == (0, 1) and win.image.screen_step(0, 1) == (-1, 0)
        win.image.set_orientation(0)
        assert win.image.screen_step(1, 0) == (1, 0) and win.image.screen_step(0, 1) == (0, 1)
        # every Grabber ROI is labelled on the image
        win._got_plan({"aqroi": [250, 160, 270, 176], "aqroi_solis": [251, 270, 161, 176],
                       "binning": [2, 2], "n_ions": 2, "ion_rois": [[2, 2, 5, 6], [5, 2, 8, 6]],
                       "background_roi": [2, 0, 8, 1], "containment": [0.8, 0.75],
                       "created": 1.0, "frame_size": [10, 8], "sdk_image": [2, 2, 251, 270,
                                                                              161, 176],
                       "grabber_rois": [[2, 2, 5, 6], [5, 2, 8, 6], [2, 0, 8, 1]],
                       "positions": [[255.0, 168.0], [262.0, 168.0]], "crosstalk": [0.1, 0.1],
                       "ion_states": ["bright", "bright"], "background_contamination": 0.0,
                       "ion_mask": 3, "full_mask": 7})
        labels = [it.textItem.toPlainText() for it in win.image._rects
                  if isinstance(it, pg.TextItem)]
        assert labels[:3] == ["ion 0 80%", "ion 1 75%", "bg"] and "AQROI" in labels
        # shot histograms: every ROI a curve, each ion its threshold and fidelity
        win._got_hist({"source": "trigger", "shots": 3, "dark_reference": "background ROI",
                       "dark_pixels": 2, "available": {"trigger": 3, "verify": 0},
                       "edges": [0.5, 1.5, 2.5, 3.5],
                       "series": [{"label": "ion 0", "n": 3, "fraction": [0.0, 0.5, 0.5]},
                                  {"label": "bg", "n": 6, "fraction": [1.0, 0.0, 0.0]}],
                       "ions": [{"ok": True, "label": "ion 0", "index": 0, "threshold": 1,
                                 "fidelity": 0.9, "bright_error": 0.1, "dark_error": 0.1,
                                 "bright_errors": 1, "n_bright": 3, "dark_errors": 1,
                                 "n_dark": 6, "dark_error_neighbours_bright": 0.2}]})
        assert set(win.hist.curves) == {"ion 0", "bg"} and len(win.hist.lines) == 1
        assert "90.00%" in win.hist_table.text() and "3 externally" in win.hist_info.text()
        # the simulator's rotate buttons turn the trap
        rotate = [b for b in win.findChildren(QtWidgets.QPushButton) if b.text() == "45 deg"]
        rotate[0].click()
        t = time.time()
        while time.time() - t < 5.0 and engine.sim_scene().cfg.trap_angle_deg != 45.0:
            app.processEvents()
            time.sleep(0.01)
        assert engine.sim_scene().cfg.trap_angle_deg == 45.0
        win.close()
    finally:
        backend.close()
        server.stop()
        engine.shutdown()
