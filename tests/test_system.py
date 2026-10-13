"""End to end: engine + sipyco RPC server + clients (as ARTIQ would use it)."""

import errno
import os
import socket
import sys
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
        # a cloud candidate of a frame or two is never reported, but may still be
        # counted at this instant on a busy computer
        t = time.time()
        while time.time() - t < 8.0 and not (engine.tracker.n_bright == 3
                                              and engine.tracker.n_clouds == 0):
            time.sleep(0.05)
        kinds = [e["kind"] for e in engine.events.since(last)]
        assert "background_cleared" in kinds and "ions_detected" in kinds
        assert not {"cloud_detected", "ion_lost", "ion_trapped"} & set(kinds)
        assert engine.tracker.n_bright == 3 and engine.tracker.n_clouds == 0
        d = engine.tracker.positions()[-1] - engine.tracker.positions()[0]
        assert abs(d[0]) < 0.2 * abs(d[1])           # the chain now runs along y
    finally:
        server.stop()
        engine.shutdown()


def _stray_light(engine, offsets, rate=13000.0, sigma=1.82):
    """Static, ion-like stray light: Gaussian spots on the simulated surface,
    at ``offsets`` (px) from the first ion; returns that ion's position."""
    scene = engine.sim_scene()
    x, y = scene.state()["positions"][0]
    h, w = scene.laser_map.shape
    yy, xx = np.mgrid[0:h, 0:w]
    for dx, dy in offsets:
        scene.laser_map += (rate * np.exp(-((xx - x - dx) ** 2 + (yy - y - dy) ** 2) /
                                          (2 * sigma ** 2)) /
                            (2 * np.pi * sigma ** 2)).astype(np.float32)
    return x, y


def test_stray_light_and_the_aqroi_by_hand(tmp_path):
    """Stray light that looks like ions (a bright ion, PSF 1.82 px, as on an
    iXon Ultra 888): ioncam says so; cropped around the ion, Calibrate + arm
    only searches the crop (not the padding the long exposure adds); a search
    region does the same without cropping; an AQROI typed in or set by hand
    reads out the ion inside and leaves the stray light out."""
    engine, server, port = make_system(tmp_path, n_ions=1, psf_sigma=1.82, surface="clean")
    c = client(port)
    try:
        x, y = _stray_light(engine, [(22, 3), (80, 40)])
        assert c.wait_for_ions(3, timeout=20)["ok"]                  # taken for ions
        t = time.time()
        while time.time() - t < 5 and not engine.events.since(0, ["stray_light_hint"]):
            time.sleep(0.1)
        assert engine.events.since(0, ["stray_light_hint"])
        c.set_roi(int(x) - 12, int(y) - 12, int(x) + 13, int(y) + 13)   # crop around the ion
        assert c.wait_for_ions(1, timeout=20)["ok"]
        plan = c.calibrate_readout(hbin=4, vbin=4, arm=False)
        assert plan["n_ions"] == 1 and abs(plan["positions"][0][0] - x) < 0.5
        c.set_roi(160, 100, 360, 240)
        c.set_search_region(int(x) - 15, int(y) - 15, int(x) + 16, int(y) + 16)
        assert c.get_status()["search_region"] == [int(x) - 15, int(y) - 15, int(x) + 16,
                                                   int(y) + 16]
        assert c.wait_for_ions(1, timeout=20)["ok"]
        assert c.locate_ions()["n_ions"] == 1
        c.clear_search_region()
        assert c.locate_ions()["n_ions"] == 3                         # everywhere again
        typed = c.replan_readout(4, 4, [int(x) - 10, int(y) - 10, int(x) + 10, int(y) + 10])
        assert typed["n_ions"] == 1 and any("outside the AQROI" in w for w in typed["warnings"])
        square = c.manual_aqroi(5, 0, x, y)
        assert square["aqroi_by_hand"] and square["frame_size"] == [5, 5]
        wider = c.resize_readout(1, 0)
        assert wider["frame_size"] == [6, 5] and wider["n_ions"] == 1
        final = c.calibrate_readout(hbin=4, vbin=4, aqroi=wider["aqroi"])
        assert final["aqroi"] == wider["aqroi"] and final["n_ions"] == 1
        assert final["verification"]["ok"] and c.get_status()["readout"]["armed"]
    finally:
        c.close_rpc()
        server.stop()
        engine.shutdown()


def test_readout_camera_settings_are_enforced(tmp_path):
    """The camera settings of the readout are used exactly as set, on an
    (emulated) iXon Ultra 888 whose camera recommends a slow vertical shift:
    a speed it has is never replaced; set while armed it is re-armed at once;
    calibrate's overrides, the settings dialog and set_acquisition all change
    the one [readout] section."""
    engine, server, port = make_system(tmp_path, backend="fake_sdk", camera_model="888")
    sdk = engine.camera._sdk
    try:
        # a camera recommending its slowest speed (4.33 us): 1.13 us is still 1.13 us
        engine.camera._caps["vs_fastest_recommended_index"] = 3
        res = engine.set_acquisition("readout", vs_speed_us=1.13, vs_amplitude=0,
                                     hs_speed_mhz=30.0)
        assert res["applied"] is None and res["settings"]["vs_speed_us"] == 1.13
        assert engine.config.readout.hs_speed_mhz == 30.0
        assert engine.wait_for_ions(3, timeout=20)["ok"]
        plan = engine.calibrate_readout(n_ions=3, exposure=0.002)
        assert engine.config.readout.exposure == 0.002          # one [readout], not a copy
        a = plan["applied"]
        assert (a["vs_speed_us"], a["hs_speed_mhz"], a["exposure"]) == (1.13, 30.0, 0.002)
        assert any("faster than the 4.33 us" in w for w in a["warnings"])
        # set while armed: re-armed at once with them
        start = len(sdk.calls)
        res = engine.set_acquisition("readout", vs_speed_us=2.2, em_gain=150)
        assert engine.status()["readout"]["armed"]
        assert res["applied"]["vs_speed_us"] == 2.2 and res["applied"]["em_gain"] == 150
        assert ("SetVSSpeed", (2,)) in sdk.calls[start:]
        assert engine.readout_plan()["applied"]["vs_speed_us"] == 2.2
        # the settings dialog (set_config) changes the same section
        engine.set_config("readout", "vs_speed_us", 4.33)
        assert engine.status()["readout_settings"]["vs_speed_us"] == 4.33
        with pytest.raises(ValueError):
            engine.set_acquisition("live", exposure=1.0)
        with pytest.raises(KeyError):
            engine.set_acquisition("readout", vs_speed=1.0)
        with pytest.raises(ValueError):                         # rejected: nothing changes
            engine.set_acquisition("readout", exposure=-1.0, em_gain=10)
        assert engine.config.readout.em_gain == 150
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
        from pyqtgraph.Qt import QtCore, QtWidgets
        from ioncam.gui.backend import LocalBackend
        from ioncam.gui.main_window import MainWindow
        app = pg.mkQApp("ioncam-test")
    except Exception as e:  # pragma: no cover - no usable Qt
        pytest.skip("Qt not usable: {}".format(e))
    engine, server, port = make_system(tmp_path)
    backend = LocalBackend(server.api, server)
    # dialogs are modal: recorded instead (a test must not block on one)
    errors, warning = [], QtWidgets.QMessageBox.warning
    infos, information = [], QtWidgets.QMessageBox.information
    QtWidgets.QMessageBox.warning = staticmethod(lambda *a, **k: errors.append(a[1:]))
    QtWidgets.QMessageBox.information = staticmethod(lambda *a, **k: infos.append(a[1:]))
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
        assert win.rpc_chip.text() == "RPC :{}".format(port)
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

        def wait(cond, seconds=10.0):
            t = time.time()
            while time.time() - t < seconds and not cond():
                app.processEvents()
                time.sleep(0.01)
            return cond()
        # the AQROI by hand: a square one on the ions in view, the size pad,
        # one placed by a click; Calibrate + arm would keep it
        win._new_aqroi()
        assert wait(lambda: (win._plan or {}).get("aqroi_by_hand"))
        assert win._plan["frame_size"] == [5, 5] and win.keep_aqroi.isChecked()
        square = list(win._plan["aqroi"])
        x0, y0 = square[:2]
        win._resize(1, 0)                                # one super-pixel wider on screen
        assert wait(lambda: sorted(win._plan["frame_size"]) == [5, 6])
        assert win.size_label.text() == "6x5"
        win._new_aqroi()                                 # back to 5 x 5 on the ions
        assert wait(lambda: win._plan["aqroi"] == square)
        win.place_btn.setChecked(True)
        assert win.image.pick_mode
        cx, cy = x0 + 14.0, y0 + 10.0                    # a little to the side
        win.image.clicked.emit(cx, cy)
        assert wait(lambda: win._plan["aqroi"] != square)
        aq = win._plan["aqroi"]                          # centred on the click
        assert abs((aq[0] + aq[2] - 1) / 2 - cx) <= 2 and abs((aq[1] + aq[3] - 1) / 2 - cy) <= 2
        assert not win.place_btn.isChecked() and not win.image.pick_mode
        # search only where the user looks
        win.image.zoom_to([(260.0, 168.0)], margin=20)
        win._search_view()
        assert wait(lambda: engine.status()["search_region"] is not None)
        assert wait(lambda: win.search_btn.text() == "search: region")
        win._cmd("clear_search_region")
        assert wait(lambda: engine.status()["search_region"] is None)
        # the simulator's rotate buttons turn the trap
        rotate = [b for b in win.findChildren(QtWidgets.QPushButton) if b.text() == "45 deg"]
        rotate[0].click()
        t = time.time()
        while time.time() - t < 5.0 and engine.sim_scene().cfg.trap_angle_deg != 45.0:
            app.processEvents()
            time.sleep(0.01)
        assert engine.sim_scene().cfg.trap_angle_deg == 45.0
        # overlays one kind at a time: the ROIs go, the ion markers stay
        win._set_overlay("rois", False)
        assert not win.image._rects and not win.overlay_buttons["rois"].isChecked()
        win._set_overlay("rois", True)
        assert any(isinstance(it, pg.TextItem) for it in win.image._rects)
        # what can run now: nothing to abort while idle
        assert win.act["live"].isEnabled() and not win.act["abort"].isEnabled()
        # the event log filtered by severity and text
        win.events.add_events([{"id": 10 ** 6, "time": "x 12:00:00", "kind": "k1",
                                "severity": "alarm", "message": "boom"},
                               {"id": 10 ** 6 + 1, "time": "x 12:00:01", "kind": "k2",
                                "severity": "info", "message": "fine"}])
        win.events.set_filter(min_severity=2)
        assert all("boom" in win.events.item(i).text() for i in range(win.events.count()))
        win.events.set_filter(min_severity=0, text="fine")
        assert win.events.count() == 1
        # the image on screen, saved with its geometry and settings
        path = win.save_image(tmp_path / "frame.npy")
        assert np.load(path).shape == win.image._last[0].shape
        assert (tmp_path / "frame.json").exists()
        # light and dark theme
        win.set_theme("light")
        win.set_theme("dark")
        # the settings dialog changes an option at run time
        from ioncam.gui.main_window import ConfigDialog
        dlg = ConfigDialog(win, backend.call("get_config").result(10),
                           backend.call("get_config_help").result(10))
        items = dlg.tree.findItems("verify_frames", QtCore.Qt.MatchFlag.MatchExactly |
                                   QtCore.Qt.MatchFlag.MatchRecursive, 0)
        assert items and items[0].toolTip(1)
        items[0].setText(1, "150")
        dlg._apply()
        t = time.time()
        while time.time() - t < 5.0 and engine.config.calibration.verify_frames != 150:
            app.processEvents()
            time.sleep(0.01)
        assert engine.config.calibration.verify_frames == 150
        dlg.close()
        # the camera settings of the long exposure and the readout (Readout tab): the
        # camera's own choices, and what is applied is what is used
        vs = win.acq["readout"]["vs_speed_us"]
        labels = [vs.itemText(k) for k in range(vs.count())]
        assert labels[0].startswith("[0.3") and labels[1].endswith("(recommended)")
        assert [vs.itemData(k) for k in range(vs.count())] == [0.3, 0.5, 0.9, 1.7, 3.3]
        hs = win.acq["readout"]["hs_speed_mhz"]
        assert [hs.itemData(k) for k in range(hs.count())] == [17.0, 10.0, 5.0, 1.0]
        assert win.acq_state.text() == "in use"
        vs.setCurrentIndex(vs.findData(0.9))
        win.acq["locate"]["em_gain"].setValue(12)
        assert win._acq_dirty and win.acq_state.text().startswith("changed")
        win._apply_acq()
        t = time.time()
        while time.time() - t < 5.0 and (engine.config.readout.vs_speed_us != 0.9
                                         or win.acq_state.text() != "in use"):
            app.processEvents()
            time.sleep(0.01)
        assert engine.config.readout.vs_speed_us == 0.9 and engine.config.locate.em_gain == 12
        assert win.acq_state.text() == "in use" and not win._acq_dirty
        assert win.acq_now.text().startswith("camera now (live)")
        # changed elsewhere (the settings dialog, ARTIQ): the tab follows
        engine.set_config("readout", "vs_speed_us", 1.7)
        t = time.time()
        while time.time() - t < 5.0 and vs.currentData() != 1.7:
            app.processEvents()
            time.sleep(0.01)
        assert vs.currentData() == 1.7
        # pending changes go first: Locate uses what is shown
        win.acq["locate"]["exposure"].setValue(0.3)
        win._locate()
        t = time.time()
        while time.time() - t < 20.0 and engine.config.locate.exposure != 0.3:
            app.processEvents()
            time.sleep(0.01)
        assert engine.config.locate.exposure == 0.3
        t = time.time()
        while time.time() - t < 30.0 and not infos:
            app.processEvents()
            time.sleep(0.02)
        assert infos and "Located" in infos[0][1]
        assert not errors, errors
        win.close()
    finally:
        QtWidgets.QMessageBox.warning = warning
        QtWidgets.QMessageBox.information = information
        backend.close()
        server.stop()
        engine.shutdown()


def _pump(app, until, timeout=5.0):
    t = time.time()
    while time.time() - t < timeout and not until():
        app.processEvents()
        time.sleep(0.01)
    return until()


def test_gui_works_without_the_rpc_server(tmp_path, monkeypatch, caplog):
    """The GUI does not depend on the RPC server it runs for ARTIQ: with
    sipyco missing, the port in use, a server that fails after a slow start or
    one stopped by a terminate request, every action still works and the
    reason is shown.  (It used to hang 10 s at start-up and then fail every
    action with "Event loop is closed".)"""
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
    from ioncam import server as server_mod
    cfg = Config()
    cfg.storage.data_dir = str(tmp_path)
    cfg.simulator.seed = 5
    engine, api = build(cfg, simulate=True)
    engine.start()
    errors, warning = [], QtWidgets.QMessageBox.warning
    QtWidgets.QMessageBox.warning = staticmethod(lambda *a, **k: errors.append(a[1:]))
    backends, servers = [], []

    def backend_for(st):
        b = LocalBackend(api, st)
        backends.append(b)
        if st is not None:
            servers.append(st)
        return b

    def actions_work(b):
        b.call("stop").result(10)
        assert b.call("start_live").result(10)["frame_shape"]
        b.call("set_exposure", 0.04).result(10)
        b.call("set_config", "detection", "threshold_sigma", 7.0).result(10)

    try:
        # sipyco missing: fails at once, says so, the GUI works
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "sipyco.pc_rpc", None)
            st = ServerThread(api, ["localhost"], free_port())
            b = backend_for(st)
            t = time.time()
            with pytest.raises(RuntimeError, match="sipyco"):
                st.start()
            assert time.time() - t < 5.0
        rpc = b.rpc_state()
        assert rpc["state"] == "failed" and "pip install" in rpc["hint"]
        actions_work(b)
        win = MainWindow(b, b.call("get_config").result(10), b.call("get_info").result(10))
        win.show()
        assert _pump(app, lambda: errors)               # told once, at start-up
        assert "ARTIQ cannot connect" in errors[0][1] and "sipyco" in errors[0][1]
        assert win.rpc_chip.text() == "RPC failed"
        errors.clear()
        win._cmd("set_exposure", 0.05)
        assert _pump(app, lambda: win.msg_label.text() == "set_exposure: ok")
        assert not errors, errors
        win.close()

        # the port in use
        port = free_port()
        st1 = ServerThread(api, ["localhost"], port).start()
        servers.append(st1)
        st2 = ServerThread(api, ["localhost"], port)
        b = backend_for(st2)
        with pytest.raises(OSError):
            st2.start()
        assert "port {}".format(port) in b.rpc_state()["hint"]
        actions_work(b)

        # a terminate request (artiq_ctlmgr) closes ioncam like any controller
        b = backend_for(st1)
        win = MainWindow(b, b.call("get_config").result(10), b.call("get_info").result(10))
        win.show()
        assert _pump(app, lambda: win.rpc_chip.text() == "RPC :{}".format(port))
        c = Client("localhost", port, "ioncam", timeout=10)
        c.terminate()
        c.close_rpc()
        assert _pump(app, lambda: not win.isVisible())
        assert st1.state == "stopped" and not errors
        actions_work(b)

        # a server that fails after start() gave up waiting: logged and reported
        def slow_failure(api):
            time.sleep(0.4)
            raise OSError(errno.EADDRINUSE, "address already in use")
        monkeypatch.setattr(server_mod, "make_server", slow_failure)
        st = ServerThread(api, ["localhost"], free_port())
        b = backend_for(st)
        stopped = []
        b.on_rpc_stop = lambda *a: stopped.append(a)
        st.start(timeout=0.05)
        assert st.state == "starting" and b.rpc_state()["state"] == "starting"
        assert _pump(app, lambda: stopped)
        assert stopped[0][0] == "failed" and "already in use" in stopped[0][1]
        assert "ARTIQ cannot connect" in caplog.text
        actions_work(b)

        # closing the GUI cancels a long wait (the process could not exit)
        b = backend_for(None)
        assert b.rpc_state()["state"] == "off"
        fut = b.call("wait_for_ions", 50, 3600.0, channel="wait")
        time.sleep(0.3)
        t = time.time()
        b.close()
        with pytest.raises(RuntimeError, match="closing"):
            fut.result(5)
        assert time.time() - t < 5.0
    finally:
        QtWidgets.QMessageBox.warning = warning
        for b in backends:
            b.close()
        for st in servers:
            st.stop()
        engine.shutdown()
