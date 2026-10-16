"""Is the crystal still there?  The ion light in averaged readout frames: an
ion dark, an ion lost (the others re-arranged), no ion light - on synthetic
EMCCD frames with a known truth, then on the simulator while armed."""

import asyncio
import dataclasses
import time

import numpy as np
import pytest

from ioncam.analysis.chain import equilibrium_positions
from ioncam.analysis.readout import plan_readout, square_aqroi
from ioncam.analysis.readout_check import (check_crystal, combine, crystal_positions,
                                           frame_light, zero_light_level)
from ioncam.camera.simulator import emccd_counts
from ioncam.config import Config
from ioncam.server import build

SENSOR = (512, 512)
PSF = (1.2, 1.2)
HALO = (0.18, 3.4)
SCALE = 3.4                        # trap length scale [px] (the lab's trap)
EM_GAIN, SENS, READ_E = 200.0, 4.5, 33.0


def chain(n, angle_deg=0.0, center=(262.0, 168.0), scale=SCALE):
    a = np.radians(angle_deg)
    axis = np.array([np.cos(a), np.sin(a)])
    return np.asarray(center)[None, :] + scale * equilibrium_positions(n)[:, None] * axis[None]


def two_ions_2x2():
    """Two ions at 45 deg, each in a diagonal super-pixel of a 2 x 2 AQROI of
    3 x 3 binning - the lab's readout."""
    pos = chain(2, 45.0)
    rect = square_aqroi(pos, PSF, SENSOR, 3, 3, 2, halo=HALO)
    return pos, plan_readout(pos, PSF, SENSOR, 3, 3, aqroi=rect, halo=HALO)


def three_ion_chain():
    """Three ions in a row, 4 x 4 binning, the automatic AQROI with its
    background row."""
    pos = chain(3)
    return pos, plan_readout(pos, PSF, SENSOR, 4, 4, halo=HALO)


def shots(plan, ions, n, seed, bright_p=1.0, photons=12.0, background=0.25, psf=PSF,
          halo=HALO):
    """Readout frames: every ion at ``ions`` (sensor px) bright with
    probability ``bright_p`` in each shot (``photons`` photo-electrons, all
    its light), a uniform ``background`` (photo-electrons per super-pixel),
    the EMCCD (EM gain, read noise, bias 100)."""
    rng = np.random.default_rng(seed)
    w, h = plan.frame_size
    ions = np.asarray(ions, float).reshape(-1, 2)
    K = frame_light(plan, ions, halo) if len(ions) else np.zeros((w * h, 0))
    if psf != PSF:
        K = frame_light(dataclasses.replace(plan, psf_sigma=psf), ions, halo)
    lit = rng.random((n, len(ions))) < bright_p
    el = (lit * photons) @ K.T + background
    return emccd_counts(el.reshape(n, h, w), EM_GAIN, True, SENS, READ_E, 100.0, 0.0, rng)


def test_zero_light_level():
    """The bias from the read-noise peak, also with the ions bright in every
    shot (few shots without a photo-electron in a super-pixel)."""
    pos, plan = two_ions_2x2()
    for frames in (shots(plan, pos, 1000, 1), shots(plan, pos, 1000, 2, bright_p=0.5),
                   shots(plan, [], 1000, 3)):
        flat = frames.reshape(len(frames), -1)
        darkest = np.argsort(np.median(flat, axis=0))[:4]
        z = zero_light_level(flat[:, darkest], g=EM_GAIN / SENS)
        assert z["ok"] and z["level"] == pytest.approx(100.0, abs=1.5)
        assert z["sigma"] == pytest.approx(READ_E / SENS, rel=0.15)


def test_rearranged_crystals():
    pos = chain(3, 30.0)
    two = crystal_positions(pos, 2)
    one = crystal_positions(pos, 1)
    assert one[0] == pytest.approx(pos.mean(axis=0))
    assert np.hypot(*(two[1] - two[0])) == pytest.approx(
        SCALE * float(np.diff(equilibrium_positions(2))[0]), rel=1e-6)
    d = (two[1] - two[0]) / np.hypot(*(two[1] - two[0]))
    assert abs(d @ np.array([np.cos(np.radians(30)), np.sin(np.radians(30))])) == \
        pytest.approx(1.0)


@pytest.mark.parametrize("p", [1.0, 0.5])
def test_two_ions_in_a_2x2_aqroi(p):
    """The lab's readout: dark, lost (the other ion goes to the middle) and no
    light are told apart in 1000 shots and in 250, whether the ions are
    bright in every shot or in half of them."""
    pos, plan = two_ions_2x2()
    assert plan.frame_size == (2, 2) and plan.background_roi is None
    ref = check_crystal(plan, shots(plan, pos, 1000, 10, p), g=EM_GAIN / SENS)
    assert ref["verdict"] == "ok" and ref["decided"] and ref["n_ions"] == 2
    one = crystal_positions(pos, 1)
    cases = {"ok": pos, "dark0": pos[1:], "dark1": pos[:1], "lost": one}
    for n in (1000, 250):
        for name, ions in cases.items():
            for r in (None, ref):
                c = check_crystal(plan, shots(plan, ions, n, 20 + n, p), reference=r,
                                  g=EM_GAIN / SENS)
                assert c["decided"], (name, n, c["message"])
                if name == "ok":
                    assert c["verdict"] == "ok", (n, c["message"])
                elif name.startswith("dark"):
                    assert c["verdict"] == "dark" and c["dark"] == [int(name[-1])], \
                        (name, n, c["message"])
                else:
                    assert c["verdict"] == "lost" and c["n_ions"] == 1, (n, c["message"])
        # no ion light: lasers off, or the ions gone with the laser scatter there
        off = check_crystal(plan, shots(plan, [], n, 30, background=0.0), reference=ref,
                            g=EM_GAIN / SENS)
        gone = check_crystal(plan, shots(plan, [], n, 31), reference=ref, g=EM_GAIN / SENS)
        assert off["verdict"] == gone["verdict"] == "no_light"
        assert "lasers off" in off["message"] and "scatter is still there" in gone["message"]
    # the light the fit gives each ion: its photo-electrons in the frame x gain
    c = check_crystal(plan, shots(plan, pos, 1000, 40, p), reference=ref, g=EM_GAIN / SENS)
    for ion in c["ions"]:
        assert ion["light"] == pytest.approx(12.0 * p * EM_GAIN / SENS, rel=0.12)
        assert ion["state"] == "bright"


def test_three_ion_chain():
    """A chain of three with a background row: a dark ion at any place, one
    lost (the two left re-arrange), two lost (one left in the middle looks
    like the outer ions dark: said so)."""
    pos, plan = three_ion_chain()
    assert plan.background_roi is not None
    ref = check_crystal(plan, shots(plan, pos, 1000, 50, 0.5), g=EM_GAIN / SENS)
    assert ref["verdict"] == "ok"
    for k in range(3):
        c = check_crystal(plan, shots(plan, np.delete(pos, k, axis=0), 1000, 60 + k, 0.5),
                          reference=ref, g=EM_GAIN / SENS)
        assert c["verdict"] == "dark" and c["dark"] == [k], c["message"]
    c = check_crystal(plan, shots(plan, crystal_positions(pos, 2), 1000, 70, 0.5),
                      reference=ref, g=EM_GAIN / SENS)
    assert c["verdict"] == "lost" and c["n_ions"] == 2 and c["lost"] == 1
    c = check_crystal(plan, shots(plan, crystal_positions(pos, 1), 1000, 71, 0.5),
                      reference=ref, g=EM_GAIN / SENS)
    assert c["verdict"] == "dark" and c["dark"] == [0, 2]
    assert "2 lost: 1 ion left would look the same" in c["message"]


def test_robust_to_a_psf_off_by_15_percent():
    """The planned PSF a little off (real optics): still ok, dark, lost."""
    pos, plan = two_ions_2x2()
    psf = (1.38, 1.38)
    ref = check_crystal(plan, shots(plan, pos, 1000, 80, 0.5, psf=psf), g=EM_GAIN / SENS)
    assert ref["verdict"] == "ok"
    for ions, verdict in ((pos, "ok"), (pos[:1], "dark"), (crystal_positions(pos, 1), "lost")):
        c = check_crystal(plan, shots(plan, ions, 1000, 81, 0.5, psf=psf), reference=ref,
                          g=EM_GAIN / SENS)
        assert c["verdict"] == verdict, c["message"]


def test_a_bad_plan_is_unclear():
    """Light where the plan has no ion (e.g. a wrong calibration): unclear -
    recalibrate - never "no ion light"."""
    pos, plan = two_ions_2x2()
    wrong = dataclasses.replace(plan, positions=[tuple(pos[0]), tuple(pos[0] + [1.0, 0.2])])
    c = check_crystal(wrong, shots(plan, pos, 1000, 90), g=EM_GAIN / SENS)
    assert c["verdict"] == "unclear" and "recalibrate" in c["message"]


def test_combine_recent_and_full_windows():
    ok = {"verdict": "ok", "n_ions": 2, "decided": True, "message": "ok", "n_shots": 1000}
    dark = {"verdict": "dark", "n_ions": 2, "dark": [1], "decided": True, "message": "dark",
            "n_shots": 250}
    unsure = dict(dark, decided=False)
    assert combine(ok, dict(ok, n_shots=250))["window"] == "full"
    r = combine(ok, dark)
    assert r["verdict"] == "dark" and r["window"] == "recent" and "changed" in r["message"]
    assert combine(ok, unsure)["verdict"] == "ok"


def test_crystal_check_while_armed(tmp_path):
    """The simulator, armed with the lab's readout (two ions at 45 deg in a
    2 x 2 AQROI), qubit states random: nothing said while the crystal is as
    planned; an ion dark, back, lost - each an event, the plan outdated after
    the loss; the banner shows it."""
    cfg = Config()
    cfg.storage.data_dir = str(tmp_path)
    cfg.simulator.seed = 4
    cfg.simulator.n_ions = 2
    cfg.simulator.psf_sigma = 1.2
    cfg.simulator.length_scale_px = 3.4
    cfg.simulator.trap_angle_deg = 45.0
    cfg.tracker.confirm_time = 0.3
    engine, api = build(cfg, simulate=True)
    engine.start()
    try:
        assert engine.wait_for_ions(2, timeout=30)["ok"]
        engine.calibrate_readout(n_ions=2, hbin=3, vbin=3)
        plan = engine.manual_aqroi(2, 2)
        assert plan["frame_size"] == [2, 2] and engine.mode == "readout"
        assert engine.readout_check()["verdict"] is None
        sim, scene = engine.sim_camera(), engine.sim_scene()
        scene.set_params(qubit_dark_probability=0.3)
        last = engine.events.last_id

        def run(n):
            nonlocal last
            k0 = engine.monitor.shots
            sim.trigger(n, 0.002)
            t0 = time.time()
            while engine.monitor.shots < k0 + n and time.time() - t0 < 60:
                time.sleep(0.05)
            time.sleep(0.3)
            kinds = [e["kind"] for e in engine.events.since(last)]
            last = engine.events.last_id
            return engine.readout_check(), kinds

        c, kinds = run(1200)
        assert c["verdict"] == "ok" and c["reference"] is not None, c["message"]
        assert not [k for k in kinds if k.startswith("readout_") and k != "readout_armed"]
        assert engine.status()["readout"]["crystal"]["verdict"] == "ok"
        scene.set_dark(1)
        c, kinds = run(600)
        assert c["verdict"] == "dark" and c["dark"] == [1] and "readout_ion_dark" in kinds
        scene.set_dark(1, False)
        c, kinds = run(600)
        assert c["verdict"] == "ok" and "readout_crystal_ok" in kinds
        scene.lose_ion(1)
        c, kinds = run(600)
        assert c["verdict"] == "lost" and c["n_ions"] == 1
        assert "readout_ion_lost" in kinds and "readout_plan_outdated" in kinds
        assert not engine.status()["readout"]["plan_valid"]
        # over RPC, and checked on demand on the last shots
        now = asyncio.run(api.get_readout_check(300))
        assert now["verdict"] == "lost" and now["n_shots"] == 300 and len(now["signal"]) == 2
        try:
            from ioncam.gui.widgets import readout_crystal_view
        except Exception:  # pragma: no cover - no usable Qt
            return
        text, desc, _ = readout_crystal_view(engine.status()["readout"]["crystal"])
        assert text == "1 ion" and "ION LOST" in desc
    finally:
        engine.shutdown()
