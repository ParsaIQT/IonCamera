"""Crystal tracking scenarios in virtual time (simulated trap + EMCCD)."""

import numpy as np
import pytest

from scenario import Scenario


def kinds(events):
    return [e["kind"] for e in events]


def settle(n=5, **kw):
    sc = Scenario(n_ions=n, **kw)
    sc.run(2.5)
    ev = sc.take()
    assert kinds(ev) == ["ions_detected"], ev
    assert sc.tracker.n_bright == n and sc.tracker.is_stable()
    return sc


def test_loading_through_a_cloud_counts_as_trapped():
    sc = Scenario(n_ions=0)
    sc.run(1.0)
    assert sc.take() == []
    sc.scene.load_ion(1, cloud_time=1.0)
    sc.run(2.5)
    k = kinds(sc.take())
    assert "cloud_detected" in k and k[-1] == "ion_trapped"
    assert sc.tracker.n_bright == 1
    sc.scene.load_ion(2, cloud_time=0.0)
    sc.run(2.0)
    ev = sc.take()
    assert kinds(ev) == ["ion_trapped"] and ev[0]["details"]["after"] == 3


@pytest.mark.parametrize("site", [0, 2, 4])
def test_dark_ion_is_not_lost(site):
    """A dark ion (end or middle) leaves the rest of the crystal unperturbed."""
    sc = settle(5)
    sc.scene.set_dark(site, True)
    sc.run(1.5)
    ev = sc.take()
    assert kinds(ev) == ["ion_dark"], ev
    assert ev[0]["details"]["site"] == site
    assert (sc.tracker.n_bright, sc.tracker.n_dark) == (4, 1)
    sc.scene.set_dark(site, False)
    sc.run(1.5)
    assert kinds(sc.take()) == ["ion_bright"]
    assert (sc.tracker.n_bright, sc.tracker.n_dark) == (5, 0)


@pytest.mark.parametrize("site", [0, 2, 4])
def test_lost_ion_detected_by_rearrangement(site):
    sc = settle(5)
    sc.scene.lose_ion(site)
    sc.run(1.5)
    ev = sc.take()
    assert kinds(ev) == ["ion_lost"], ev
    assert ev[0]["details"]["before"] == 5 and ev[0]["details"]["after"] == 4
    assert sc.tracker.n_bright == 4


def test_loss_while_another_ion_is_dark_keeps_the_dark_ion():
    sc = settle(5)
    sc.scene.set_dark(0, True)
    sc.run(1.5)
    sc.take()
    sc.scene.lose_ion(4)              # a bright ion at the other end is lost
    sc.run(1.5)
    ev = sc.take()
    assert kinds(ev) == ["ion_lost"]
    assert (sc.tracker.n_bright, sc.tracker.n_dark) == (3, 1)
    # the dark ion's position was inferred from the chain model: when it lights
    # up again it must be recognised as "bright again", not as a new ion
    sc.scene.set_dark(0, False)
    sc.run(1.5)
    assert kinds(sc.take()) == ["ion_bright"]
    assert sc.tracker.n_bright == 4


def test_dark_ion_lost():
    sc = settle(4)
    sc.scene.set_dark(1, True)
    sc.run(1.5)
    sc.take()
    sc.scene.lose_ion(1)              # the dark one goes
    sc.run(1.5)
    ev = sc.take()
    assert kinds(ev) == ["ion_lost"] and "dark" in ev[0]["message"]
    assert (sc.tracker.n_bright, sc.tracker.n_dark) == (3, 0)


def test_melting_and_recrystallisation():
    sc = settle(4)
    sc.scene.melt(1.0)
    sc.run(3.0)
    assert kinds(sc.take()) == ["crystal_melted", "crystal_recovered"]


def test_melting_with_loss():
    sc = settle(4)
    sc.scene.melt(1.0)
    sc.run(0.6)
    sc.scene.lose_ion(0)
    sc.run(2.5)
    k = kinds(sc.take())
    assert k[0] == "crystal_melted" and "ion_lost" in k
    assert sc.tracker.n_bright == 3


def test_lasers_off_is_not_an_ion_loss():
    sc = settle(3)
    sc.scene.set_lasers(False)
    sc.run(1.5)
    assert kinds(sc.take()) == ["fluorescence_lost"]
    sc.scene.set_lasers(True)
    sc.run(1.5)
    assert kinds(sc.take()) == ["fluorescence_restored"]
    assert sc.tracker.n_bright == 3


def test_no_fluorescence_for_long_means_presumed_lost():
    sc = settle(2)
    sc.cfg.tracker.forget_time = 2.0
    sc.scene.set_lasers(False)
    sc.run(3.5)
    k = kinds(sc.take())
    assert k == ["fluorescence_lost", "ions_presumed_lost"]
    assert sc.tracker.n_bright == 0


def test_rigid_shift_is_a_move_not_a_loss():
    sc = settle(4)
    sc.scene.shift(5.0, 1.0)
    sc.run(1.0)
    ev = sc.take()
    assert kinds(ev) == ["chain_moved"]
    assert sc.tracker.n_bright == 4
    s0 = sc.tracker.sites[0]
    truth = np.array(sc.scene.state()["positions"][0])
    assert np.hypot(s0.x - truth[0], s0.y - truth[1]) < 0.5


def test_hold_during_shuttling_reports_survivors():
    sc = settle(4)
    sc.tracker.hold("shuttle", 30)
    sc.scene.shift(30.0, 0.0)
    sc.run(0.5)
    sc.scene.lose_ion(1)
    sc.run(0.5)
    sc.scene.shift(-30.0, 0.0)
    sc.run(0.5)
    assert sc.take() == []                 # nothing reported while on hold
    sc.tracker.release()
    sc.run(1.5)
    ev = sc.take()
    assert kinds(ev) == ["ion_lost", "hold_done"]
    assert ev[1]["details"] == {"before": 4, "after": 3}


def test_hold_without_losses():
    sc = settle(3)
    sc.tracker.hold("move camera", 30)
    sc.scene.shift(12.0, -3.0)
    sc.run(1.0)
    sc.tracker.release()
    sc.run(1.5)
    assert kinds(sc.take()) == ["hold_done"]


def test_stable_crystal_produces_no_events():
    sc = settle(7, exposure=0.01, em_gain=100)
    sc.run(10.0)
    assert sc.take() == []
    assert sc.tracker.is_stable()


def test_single_ion_dark_is_ambiguous():
    sc = settle(1)
    sc.scene.set_dark(0, True)
    sc.run(1.5)
    ev = sc.take()
    assert kinds(ev) == ["fluorescence_lost"] and "single ion" in ev[0]["message"]
    sc.scene.set_dark(0, False)
    sc.run(1.5)
    assert kinds(sc.take()) == ["fluorescence_restored"]
