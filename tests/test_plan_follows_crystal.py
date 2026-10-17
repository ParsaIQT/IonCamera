"""The readout plan follows the crystal on screen: a locate or a plan made
before the live tracking saw the crystal change (an ion trapped or lost, the
crystal re-arranged or moved) is not planned for or armed again - with two
ions in the trap and a locate of one, a 2 x 2 AQROI by hand read out one ion
and the armed banner said "1 ion"."""

import time

import numpy as np

from ioncam.analysis.readout import square_aqroi
from ioncam.config import Config
from ioncam.server import build


def _engine(tmp_path, n_ions, angle=45.0, seed=11):
    cfg = Config()
    cfg.storage.data_dir = str(tmp_path)
    cfg.simulator.seed = seed
    cfg.simulator.n_ions = n_ions
    cfg.simulator.psf_sigma = 1.2
    cfg.simulator.length_scale_px = 3.4
    cfg.simulator.trap_angle_deg = angle
    cfg.tracker.confirm_time = 0.3
    return build(cfg, simulate=True)[0]


def _banner(engine):
    try:
        from ioncam.gui.widgets import readout_crystal_view
    except Exception:  # pragma: no cover - no usable Qt
        return None
    return readout_crystal_view(engine.status()["readout"]["crystal"])[0]


def _shots(engine, n=1200):
    """``n`` externally triggered shots, qubit states random; the check."""
    engine.sim_scene().set_params(qubit_dark_probability=0.3)
    k0 = engine.monitor.shots
    engine.sim_camera().trigger(n, 0.002)
    t0 = time.time()
    while engine.monitor.shots < k0 + n and time.time() - t0 < 60:
        time.sleep(0.05)
    time.sleep(0.3)
    return engine.readout_check()


def _settled(engine, n, outdated=False):
    """``n`` ions tracked, settled; ``outdated``: once the tracker has seen
    the crystal change under the plan."""
    t0 = time.time()
    while outdated and engine.readout_plan()["valid"] and time.time() - t0 < 20:
        time.sleep(0.05)
    assert engine.wait_for_ions(n, timeout=30)["ok"]
    time.sleep(0.5)


def _adjusted(engine, since):
    return [e["message"] for e in engine.events.since(since) if e["kind"] == "readout_adjusted"]


def test_plan_source_follows_the_crystal_on_screen():
    """Where the ions to plan for come from (no camera needed): the locate
    while the crystal on screen is the one it found, else the live tracking -
    keeping the plan's own positions for ions within 1 px of them."""
    engine, _ = build(Config(), simulate=True)
    one = [[100.0, 100.0]]
    two = [[97.9, 100.1], [102.1, 99.9]]

    def view(positions, t):
        engine._snapshot = {"stable": True, "last_update": t, "sites": [
            {"x": x, "y": y, "state": "bright"} for x, y in positions]}

    engine._last_locate = {"positions": one, "states": ["bright"], "psf_sigma": [1.2, 1.2],
                           "psf_halo": [0.0, 0.0], "t": 100.0, "search_region": None}
    view(two, 200.0)
    # the tracking counts two, but saw no change since the locate: the locate wins
    assert engine._plan_source(True)[5] == "locate"
    # a second ion trapped after the locate: the ions on screen
    engine._crystal_changed_t = 150.0
    pos, _, _, states, newer, source, t = engine._plan_source(True)
    assert source == "tracking" and newer and t == 200.0 and len(pos) == 2
    # a plan made from them (New AQROI): kept as it was while they stay (no
    # jitter between plans by hand)
    kw = engine._plan_kw(None, (0.0, 0.0))
    rect = square_aqroi(pos, (1.2, 1.2), engine.camera.sensor_shape, 4, 4, 2, 2, None, **kw)
    engine._plan = engine._plan_from(pos, (1.2, 1.2), states, 4, 4, kw, source, t, rect)
    plan = engine._plan.to_dict()
    assert plan["n_ions"] == 2 and plan["positions_t"] == 200.0
    view([[x + 0.3, y - 0.2] for x, y in two], 260.0)
    assert engine._plan_source(True)[5] == "plan"
    # one ion more on screen: the plan's ions at their planned positions, the new one too
    view([[x + 0.3, y - 0.2] for x, y in two] + [[106.3, 100.0]], 300.0)
    engine._crystal_changed_t = 280.0
    pos, _, _, _, _, source, _ = engine._plan_source(True)
    assert source == "tracking" and len(pos) == 3
    assert np.allclose(pos[:2], plan["positions"], atol=1e-3)
    # the crystal moved by 3 px since: the ions on screen, the plan outdated
    view([[x + 3.0, y] for x, y in two], 400.0)
    engine._crystal_moved_t = 350.0
    assert engine._plan_outdated(engine._plan)
    assert np.allclose(engine._plan_source(True)[0], [[x + 3.0, y] for x, y in two])


def test_second_ion_after_a_one_ion_plan(tmp_path):
    """Calibrated with one ion, a second loaded: a 2 x 2 AQROI by hand holds
    both and the armed readout says 2 ions (it said 1); after a loss, Arm
    places the AQROI for the ion left."""
    engine = _engine(tmp_path, 1)
    engine.start()
    try:
        _settled(engine, 1)
        engine.calibrate_readout(n_ions=1, hbin=4, vbin=4)
        old = engine.manual_aqroi(2, 2)
        assert old["n_ions"] == 1 and engine.mode == "readout"
        engine.disarm_readout()
        engine.sim_scene().load_ion(1, cloud_time=0.0)
        _settled(engine, 2, outdated=True)
        sites = engine.status()["sites"]
        cx, cy = np.mean([s["x"] for s in sites]), np.mean([s["y"] for s in sites])
        plan = engine.manual_aqroi(2, 2, cx, cy, 4, 4)            # New AQROI in the GUI
        assert plan["n_ions"] == 2 and plan["frame_size"] == [2, 2]
        assert any("crystal changed since the locate" in w for w in plan["warnings"])
        assert engine.readout_plan()["valid"]
        engine.arm_readout()
        c = _shots(engine)
        assert c["verdict"] == "ok" and c["decided"] and c["expected"] == 2, c["message"]
        assert _banner(engine) in (None, "2 ions")
        # an ion lost while live: Arm re-centres the AQROI on the ion left
        engine.disarm_readout()
        engine.sim_scene().lose_ion(1)
        _settled(engine, 1, outdated=True)
        last = engine.events.last_id
        armed = engine.arm_readout()
        assert armed["n_ions"] == 1 and armed["frame_size"] == [2, 2]
        assert any("ions changed since the plan (2 -> 1 ion read out)" in m
                   for m in _adjusted(engine, last))
        c = _shots(engine)
        assert c["verdict"] == "ok" and c["expected"] == 1, c["message"]
    finally:
        engine.shutdown()


def test_arm_follows_a_moved_crystal(tmp_path):
    """The crystal moved by (+3, -2) px since the plan: Arm moves the AQROI
    with it, the Grabber ROIs (ARTIQ's) unchanged; nothing changed: the plan
    is armed as it is; the D-pad on an outdated plan places the ROIs again for
    the ions on screen inside the AQROI it moved (none inside yet: the ROIs
    stay, no error, so it can be walked over to the ions)."""
    engine = _engine(tmp_path, 2)
    engine.start()
    try:
        _settled(engine, 2)
        engine.calibrate_readout(n_ions=2, hbin=4, vbin=4)
        plan = engine.manual_aqroi(2, 2)
        engine.disarm_readout()
        time.sleep(1.0)
        same = engine.arm_readout()
        assert same["created"] == plan["created"]             # nothing changed
        engine.disarm_readout()
        engine.sim_scene().shift(3.0, -2.0)
        _settled(engine, 2, outdated=True)
        assert not engine.readout_plan()["valid"]
        last = engine.events.last_id
        moved = engine.arm_readout()
        x0, y0, x1, y1 = plan["aqroi"]
        assert moved["aqroi"] == [x0 + 3, y0 - 2, x1 + 3, y1 - 2]
        assert moved["grabber_rois"] == plan["grabber_rois"] and moved["n_ions"] == 2
        assert engine.readout_plan()["valid"]
        assert any("crystal moved by" in m and "AQROI moved with it, ROIs unchanged" in m
                   for m in _adjusted(engine, last))
        c = _shots(engine)
        assert c["verdict"] == "ok" and c["decided"], c["message"]
        # the D-pad on an outdated plan: walked back to the ions (no ion inside
        # on the way: the ROIs as they were), then their ROIs placed again
        engine.disarm_readout()
        engine.sim_scene().shift(-3.0, 2.0)
        _settled(engine, 2, outdated=True)
        last = engine.events.last_id
        away = engine.nudge_readout(8, 0)
        assert away["grabber_rois"] == plan["grabber_rois"]
        nudged = engine.nudge_readout(-11, 2)
        assert nudged["aqroi"] == plan["aqroi"] and nudged["n_ions"] == 2
        msgs = _adjusted(engine, last)
        assert "no ion on screen inside it yet" in msgs[0]
        assert "ROIs placed again for the ions on screen" in msgs[1]
        assert max(abs(a - b) for p, q in zip(nudged["positions"], plan["positions"])
                   for a, b in zip(p, q)) < 0.5
    finally:
        engine.shutdown()
