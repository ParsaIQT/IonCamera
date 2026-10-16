"""Camera characterization (dark frames at many readout settings, for SPAM)
end to end on the simulated camera, through the Andor driver, over RPC and
in the GUI; the camera profile it writes drives the simulator."""

import json
import os
import socket
import time

import pytest

from ioncam import characterize as ch
from ioncam.camera.profile import CameraProfile, write_profile
from ioncam.config import Config
from ioncam.engine import ProcedureError
from ioncam.server import build

# a few seconds per setting
QUICK = dict(samples_per_point=200000, min_frames=20, max_frames=400, max_point_s=3.0,
             temperature_timeout_s=0, stability_every_s=0)
DARK_AT_M60 = 0.5                   # simulated dark current [e-/px/s] at -60 C


def make_engine(tmp_path, backend="simulator", **sim):
    cfg = Config()
    cfg.storage.data_dir = str(tmp_path)
    cfg.camera.backend = backend
    cfg.simulator.seed = 7
    cfg.simulator.n_ions = sim.pop("n_ions", 2)
    for k, v in sim.items():
        setattr(cfg.simulator, k, v)
    engine, api = build(cfg, simulate=(backend == "simulator"))
    engine.start()
    # a cold sensor now (the simulated one takes ~40 s to cool down)
    engine.sim_camera()._temp = float(cfg.camera.target_temperature)
    return engine, api


def run(engine, timeout=180.0, **options):
    info = engine.start_characterization(**options)
    t0 = time.time()
    while engine.characterization_status()["state"] in ("planned", "running", "finishing"):
        assert time.time() - t0 < timeout, engine.characterization_status()
        time.sleep(0.2)
    return info, engine.characterization_status()


def wait_mode(engine, mode, timeout=10.0):
    t0 = time.time()
    while engine.mode != mode and time.time() - t0 < timeout:
        time.sleep(0.1)
    return engine.mode


def by_label(res):
    return {p["label"]: p for p in res["points"]}


def test_characterization_on_the_simulator(tmp_path):
    """The sweep with the shutter closed: the camera's real EM gain, CIC
    against the clock voltage and the binning, dark current from the
    exposures; files, the SPAM ranking, everything restored afterwards; other
    camera calls fail fast meanwhile."""
    engine, api = make_engine(tmp_path, cic_serial=0.004, em_gain_scale=0.93,
                              dark_current=DARK_AT_M60 / 2 ** (20 / 6.3))
    try:
        assert wait_mode(engine, "live") == "live"
        shutter = engine.config.camera.shutter
        opts = dict(QUICK, stages=["baseline", "vertical", "gain", "exposure", "binning",
                                   "triggered"],
                    vs_amplitudes=[0, 2], em_gains=[100, 300], exposures_ms=[0.5, 5.0, 100.0],
                    binnings=[1, 4], trigger_spacings_ms=[0.0, 20.0])
        plan = engine.start_characterization(dry_run=True, **opts)
        assert engine.characterization_status()["state"] == "idle"     # nothing started
        assert plan["stages"]["vertical"] == 9 and plan["stages"]["triggered"] == 2
        assert 0 < plan["estimate_s"] < 120 and len(plan["region"]) == 4
        with pytest.raises(ValueError):
            engine.start_characterization(dry_run=True, stages=["nonsense"])
        with pytest.raises(ValueError):
            engine.start_characterization(dry_run=True, frames_per_point=10)

        info = engine.start_characterization(**opts)
        assert info["points"] == plan["points"]
        t0 = time.time()
        while engine.characterization_status()["done"] < 1:
            st = engine.characterization_status()
            assert st["state"] in ("planned", "running") and time.time() - t0 < 60, st
            time.sleep(0.1)
        assert engine.camera.shutter == "closed"
        with pytest.raises(ProcedureError, match="camera characterization running"):
            engine.set_fan("low")
        while engine.characterization_status()["state"] in ("running", "finishing"):
            assert time.time() - t0 < 180
            time.sleep(0.2)
        st = engine.characterization_status()
        assert st["state"] == "done" and st["done"] == st["points"] == info["points"]
        assert st["error"] is None and not st["notes"]
        # restored
        assert wait_mode(engine, "live") == "live"
        assert engine.config.camera.shutter == shutter == engine.camera.shutter

        d = info["directory"]
        for name in ("plan.json", "points.jsonl", "summary.json", "summary.csv",
                     "camera_profile.json", "README.txt", "hist/0000.npz"):
            assert os.path.exists(os.path.join(d, name)), name
        with open(os.path.join(d, "points.jsonl")) as f:
            assert sum(1 for _ in f) == info["points"]

        res = engine.characterization_results()
        assert res["state"] == "done" and res["directory"] == d
        assert res["light_check"]["light"] is False
        g = res["global"]
        assert g["gain_ratio"] == pytest.approx(0.93, abs=0.05)
        assert g["dark_current"]["e_per_px_s"] == pytest.approx(DARK_AT_M60, rel=0.2)
        assert "light_leak" not in g
        pts = by_label(res)
        for vs in ("0.5", "1.7", "3.3"):         # +2 clock voltage: x 1.6^2 parallel CIC
            ratio = pts["VS {} us, voltage +2".format(vs)]["cic_parallel"] / \
                (pts["VS {} us, voltage normal".format(vs)] if vs != "0.5" else
                 pts["readout settings"])["cic_parallel"]
            assert 1.6 < ratio < 4.0, (vs, ratio)
        b4 = next(b for b in g["binning"] if b["binning"] == [4, 4])
        assert b4["cic_parallel"] == pytest.approx(b4["cic_parallel_expected"], rel=0.35)
        assert [t["spacing_ms"] for t in g["triggered"]] == [0.0, 20.0]
        assert all(p["ok"] for p in res["points"])
        # SPAM: every EM setting ranked, the shortest detection first
        rk = res["ranking"]
        assert [r["rank"] for r in rk] == list(range(1, len(rk) + 1)) and len(rk) >= 10
        times = [r["t_min_ms"] for r in rk if r["t_min_ms"] is not None]
        assert times and times == sorted(times) and times[0] < 10.0
        assert rk[0]["readout_settings"]["em_gain"] in (100, 200, 300)
        # a brighter ion is detected sooner
        bright = engine.characterization_results(ion_rate=4 * res["spam"]["ion_rate"])
        assert bright["spam"]["ion_rate_source"] == "option"
        assert bright["ranking"][0]["t_min_ms"] < 0.5 * times[0]
        # the same ranking again from the files alone
        again, spam = ch.rank(ch.load_results(d), ion_rate=res["spam"]["ion_rate"],
                              roi_fraction=res["spam"]["roi_fraction"])
        assert [r["label"] for r in again] == [r["label"] for r in rk]
        assert [r["t_min_ms"] for r in again] == pytest.approx([r["t_min_ms"] for r in rk])
        detail = engine.characterization_point(0)
        assert detail["counts"].sum() > 1e5 and detail["model"].shape == detail["counts"].shape
        assert detail["model"].sum() == pytest.approx(detail["counts"].sum(), rel=1e-6)
        assert engine.list_characterizations()[0]["complete"]
        # the profile for simulations
        prof = CameraProfile.load(os.path.join(d, "camera_profile.json"))
        assert prof.dark_current == pytest.approx(DARK_AT_M60, rel=0.2)
        assert prof.temperature == pytest.approx(-60.0, abs=1.0)
        near = prof.nearest("em", 5.0, 2, 0.5, 0, 150)
        assert near["em_gain"] in (100, 200) and near["vs_amplitude"] == 0
    finally:
        engine.shutdown()


def test_abort_keeps_what_was_measured(tmp_path):
    engine, api = make_engine(tmp_path)
    try:
        info = engine.start_characterization(**dict(QUICK, stages=["baseline", "gain"],
                                                    em_gains=[20, 50, 100, 150, 250, 300]))
        t0 = time.time()
        while engine.characterization_status()["done"] < 2:
            st = engine.characterization_status()
            assert st["state"] in ("planned", "running") and time.time() - t0 < 60, st
            time.sleep(0.1)
        assert engine.abort()
        while engine.characterization_status()["state"] in ("running", "finishing"):
            assert time.time() - t0 < 60
            time.sleep(0.1)
        st = engine.characterization_status()
        assert st["state"] == "aborted" and 2 <= st["done"] < info["points"]
        with open(os.path.join(info["directory"], "summary.json")) as f:
            summary = json.load(f)
        assert summary["state"] == "aborted" and len(summary["points"]) == st["done"]
        assert wait_mode(engine, "live") == "live"
        assert engine.camera.shutter == engine.config.camera.shutter == "open"
    finally:
        engine.shutdown()


def test_light_on_the_sensor_stops_the_run(tmp_path):
    """A camera whose shutter does not close (no internal shutter, not
    covered): the ions are seen in the first check and nothing is measured;
    with ``stop_on_light=False`` it is said and the run goes on."""
    engine, api = make_engine(tmp_path)
    sim = engine.sim_camera()
    keep_open = sim.set_shutter
    sim.set_shutter = lambda mode="open", *a, **k: keep_open("open", *a, **k)
    try:
        info, st = run(engine, **dict(QUICK, stages=["baseline"]))
        assert st["state"] == "aborted" and st["done"] == 0
        assert "light reaches the sensor" in st["error"] and "ions or laser light" in st["error"]
        assert any(e["kind"] == "characterization" and "light" in e["message"]
                   for e in engine.events.since(0))
        res = engine.characterization_results()
        assert res["light_check"]["light"] and res["light_check"]["lit_pixels"] >= 4
        assert wait_mode(engine, "live") == "live"

        info, st = run(engine, **dict(QUICK, stages=["baseline"], stop_on_light=False))
        assert st["state"] == "done" and st["done"] == 1
        assert any("light reaches the sensor" in n for n in st["notes"])
    finally:
        engine.shutdown()


def test_through_the_andor_driver(tmp_path):
    """The emulated iXon Ultra 897 under the real driver: the shutter closed
    with SetShutterEx, software-triggered single shots."""
    engine, api = make_engine(tmp_path, backend="fake_sdk")
    sdk = engine.camera._sdk
    try:
        info, st = run(engine, **dict(QUICK, stages=["baseline", "triggered"],
                                      trigger_spacings_ms=[0.0, 5.0], min_frames=12))
        assert st["state"] == "done" and st["done"] == info["points"] == 4, st
        names = [n for n, _ in sdk.calls]
        assert "SendSoftwareTrigger" in names and ("SetTriggerMode", (10,)) in sdk.calls
        closed = [a for n, a in sdk.calls if n == "SetShutterEx" and a[1] == 2]
        assert closed, [c for c in sdk.calls if c[0].startswith("SetShutter")]
        res = engine.characterization_results()
        assert res["camera"]["model"] == "DU897_BV"
        assert len(res["global"]["triggered"]) == 2 and res["ranking"]
        assert sdk.sim.shutter == "open"
    finally:
        engine.shutdown()


def test_the_profile_drives_the_simulator(tmp_path):
    """The profile a characterization writes, read back by the simulator: the
    simulated camera has the measured bias, read noise, gain and CIC."""
    path = str(tmp_path / "profile.json")
    point = {"stage": "baseline", "amplifier": "em", "hs_speed_mhz": 5.0,
             "preamp_gain_index": 2, "preamp_gain": 3.0, "vs_speed_us": 0.5,
             "vs_amplitude": 0, "em_gain": 200, "hbin": 1, "vbin": 1, "exposure": 0.0015,
             "bias": 150.0, "read_noise": 9.0, "sensitivity": 5.0, "g": 30.0, "lam_p": 0.02,
             "lam_s": 0.0, "temperature": -60.0}
    other = dict(point, hs_speed_mhz=17.0, read_noise=20.0, bias=180.0, g=35.0)
    write_profile(path, {"model": "test"}, [point, other], -60.0, 0.01,
                  [[-80.0, 0.001], [-60.0, 0.01]], 0.75, [], "now")
    prof = CameraProfile.load(path)
    assert prof.dark_current_at(-70.0) == pytest.approx(10 ** -2.5, rel=1e-6)
    near = prof.nearest("em", 5.0, 2, 0.5, 0, 100)
    assert near["bias"] == 150.0 and near["g"] == pytest.approx(15.0)
    assert prof.nearest("em", 16.0, 2, 0.5, 0, 200)["read_noise"] == 20.0
    assert prof.nearest("conventional", 1.0, 0, 0.5, 0, 1) is None
    with open(path) as f:
        data = json.load(f)
    data["format"] = "something else"
    with open(str(tmp_path / "bad.json"), "w") as f:
        json.dump(data, f)
    with pytest.raises(ValueError):
        CameraProfile.load(str(tmp_path / "bad.json"))

    engine, api = make_engine(tmp_path, camera_profile=path)
    try:
        info, st = run(engine, **dict(QUICK, stages=["baseline"]))
        assert st["state"] == "done"
        p = engine.characterization_results()["points"][0]
        assert p["bias"] == pytest.approx(150.0, abs=1.0)
        assert p["read_noise"] == pytest.approx(9.0, rel=0.05)
        assert p["em_gain_measured"] == pytest.approx(150.0, rel=0.1)   # g x sensitivity
        assert p["cic_parallel"] == pytest.approx(0.02, rel=0.2)
    finally:
        engine.shutdown()


def test_default_plans():
    """The default sweep on both cameras: mostly exposures below 10 ms, every
    setting once, the readout's at the start and the end, under an hour."""
    for model in ("897", "888"):
        cfg = Config()
        cfg.simulator.camera_model = model
        engine, api = build(cfg, simulate=True)
        engine.start()
        try:
            c = ch.Characterization(engine, {})
            pts = c.points
            assert 100 < len(pts) < 1000 and c.estimate_s < 3600, (model, len(pts),
                                                                     c.estimate_s)
            assert len({p.key() for p in pts if p.stage != "baseline"}) == \
                sum(1 for p in pts if p.stage != "baseline")
            assert pts[0].label == "readout settings" and pts[-1].label.endswith("(end)")
            short = sum(1 for p in pts if p.settings.exposure < 0.01)
            assert short > 0.9 * len(pts)
            caps = engine.camera.capabilities()
            assert {p.settings.vs_speed_us for p in pts} == set(caps["vs_speeds_us"])
        finally:
            engine.shutdown()


def test_characterize_dialog(tmp_path):
    """Start from the GUI (the question answered Yes), see the ranking and
    use its best setting for the readout."""
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
    cfg = Config()
    cfg.storage.data_dir = str(tmp_path)
    cfg.simulator.n_ions = 2
    for k, v in dict(QUICK, stages=["baseline", "gain"], em_gains=[100, 300]).items():
        setattr(cfg.characterize, k, v)
    engine, api = build(cfg, simulate=True)
    engine.start()
    engine.sim_camera()._temp = -60.0
    backend = LocalBackend(api)
    question, warning = QtWidgets.QMessageBox.question, QtWidgets.QMessageBox.warning
    asked, warned = [], []
    QtWidgets.QMessageBox.question = staticmethod(
        lambda *a, **k: asked.append(a[2]) or QtWidgets.QMessageBox.StandardButton.Yes)
    QtWidgets.QMessageBox.warning = staticmethod(lambda *a, **k: warned.append(a[2]))
    win = None

    def pump(until, timeout):
        t0 = time.time()
        while not until() and time.time() - t0 < timeout:
            app.processEvents()
            time.sleep(0.02)
        return until()
    try:
        win = MainWindow(backend, api.get_config(), api.get_info())
        win.show()
        win._characterize()
        dlg = win._char_dialog
        assert pump(lambda: "settings" in dlg.estimate.text(), 10), dlg.estimate.text()
        assert dlg.options()["stages"] == ["baseline", "gain"]
        dlg._start()
        assert asked and "Close the shutter" in asked[-1]
        assert pump(lambda: dlg._state == "done" and dlg.results is not None and
                    dlg.rank_table.rowCount() > 0, 120), dlg.state_label.text()
        assert dlg.settings_table.rowCount() == len(dlg.results["points"])
        assert "best" in dlg.spam_info.text() and "EM gain" in dlg.summary.toPlainText()
        best = dlg._ranking[0]
        dlg.rank_table.selectRow(0)
        assert dlg.use_btn.isEnabled()
        dlg._use()
        ro = engine.config.readout
        assert pump(lambda: ro.em_gain == best["readout_settings"]["em_gain"] and
                    abs(ro.exposure - best["t_min_ms"] / 1e3) < 1.1e-4, 10)
        assert not warned
    finally:
        QtWidgets.QMessageBox.question, QtWidgets.QMessageBox.warning = question, warning
        if win is not None:
            win.close()
        backend.close()
        engine.shutdown()


def free_port():
    s = socket.socket()
    s.bind(("localhost", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_over_rpc(tmp_path):
    pytest.importorskip("sipyco")
    from sipyco.pc_rpc import Client
    from ioncam.server import ServerThread
    engine, api = make_engine(tmp_path)
    port = free_port()
    server = ServerThread(api, ["localhost"], port).start()
    c = Client("localhost", port, "ioncam", timeout=60)
    try:
        plan = c.start_characterization(dry_run=True, **dict(QUICK, stages=["baseline"]))
        assert plan["points"] == 1
        c.start_characterization(**dict(QUICK, stages=["baseline", "gain"], em_gains=[300]))
        t0 = time.time()
        while c.characterization_status()["state"] in ("planned", "running", "finishing"):
            assert time.time() - t0 < 60
            time.sleep(0.2)
        assert c.characterization_status()["state"] == "done"
        res = c.characterization_results(ion_rate=20000.0, target_error=1e-2)
        assert res["spam"]["target_error"] == 1e-2 and res["ranking"][0]["t_min_ms"] > 0
        assert c.list_characterizations()[0]["points"] == 3
        point = c.characterization_point(1)
        assert len(point["counts"]) == len(point["model"]) > 10
    finally:
        c.close_rpc()
        server.stop()
        engine.shutdown()
