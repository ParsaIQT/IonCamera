"""Crystal tracking scenarios in virtual time (simulated trap + EMCCD)."""

import math

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


@pytest.mark.parametrize("n,site", [(2, 1), (5, 2)])
def test_dark_ion_stays_put_and_the_chain_with_it(n, site):
    """One ion dark for long: the bright ones are tracked frame by frame and
    re-measured on averaged frames again and again, and the dark one must keep
    its place among them exactly - it used to wander a little at every update,
    and the chain's length and angle changed on their own.  The chain is fitted
    to the positions measured on averaged frames (steady), and the PSF measured
    meanwhile is not handed to the detection anew every few seconds."""
    sc = settle(n, psf_sigma=1.82)
    sc.scene.set_dark(site, True)
    sc.run(1.5)
    assert kinds(sc.take()) == ["ion_dark"]
    truth = np.array(sc.scene.state()["positions"])
    i0 = len(sc.psf_messages)
    rel, errors, scales, angles = [], [], [], []
    for _ in range(30):
        sc.run(1.0)
        assert [s.state for s in sc.tracker.sites] == \
            ["dark" if k == site else "bright" for k in range(n)]
        pos = sc.tracker.positions()
        rel.append(pos[site] - np.delete(pos, site, axis=0).mean(axis=0))
        errors.append(np.hypot(*(pos[site] - truth[site])))
        chain = sc.tracker.snapshot()["chain"]
        scales.append(chain["scale_px"])
        angles.append(chain["angle_deg"])
    assert sc.take() == []
    assert np.ptp(np.array(rel), axis=0).max() < 0.05, rel
    assert max(errors) < 0.6, errors       # (the bright ions' own tracking noise)
    assert np.ptp(scales) < 0.02 * np.mean(scales), scales
    assert np.ptp(angles) < 1.0, angles
    assert sum(map(bool, sc.psf_messages[i0:])) <= 3, [m for m in sc.psf_messages if m]


def test_dark_ion_moves_with_a_slowly_drifting_crystal():
    """The whole crystal drifts slowly (the imaging warming up, the trap
    charging), so the 2 s averages of the re-measurement lag behind the latest
    frames, and every re-measurement moved the bright ion back the same way and
    left the dark one ahead: the dark ion drifted steadily in the direction of
    the motion, the chain growing and turning.  It must move with the crystal."""
    sc = settle(2, psf_sigma=1.82)
    sc.run(4.0)
    sc.scene.set_dark(1, True)
    sc.run(1.5)
    assert kinds(sc.take()) == ["ion_dark"]
    chain0 = sc.tracker.snapshot()["chain"]
    gaps = []
    for _ in range(120):                    # 30 s at 3 px/min along the chain, 1 across
        sc.scene.shift(3.0 / 240, 1.0 / 240)
        sc.run(0.25)
        pos = sc.tracker.positions()
        gaps.append(pos[1] - pos[0])
    assert kinds(sc.take()) == []           # (1.6 px is no "chain_moved" yet)
    assert np.ptp(np.array(gaps), axis=0).max() < 0.05, gaps
    truth = np.array(sc.scene.state()["positions"])
    assert np.hypot(*(sc.tracker.positions()[1] - truth[1])) < 0.4
    chain = sc.tracker.snapshot()["chain"]
    assert chain["scale_px"] == pytest.approx(chain0["scale_px"], rel=0.005)
    assert chain["angle_deg"] == pytest.approx(chain0["angle_deg"], abs=0.2)


def test_frames_placing_the_ions_off_the_average_are_no_drift():
    """Single frames may place the ions a little off the averaged measurement,
    always the same way, so every re-measurement corrects them the same way:
    that must neither move a dark ion nor add up to a "chain moved" that never
    happened (which also marks the readout plan outdated)."""
    sc = settle(2, psf_sigma=1.82)
    analyze = sc.detector.analyze

    def biased(frame, prior=None):
        det = analyze(frame, prior=prior)
        for s in det.spots:
            s.x += 0.25
            s.y += 0.10
        return det
    sc.detector.analyze = biased
    sc.run(35.0)
    assert sc.take() == []
    sc.scene.set_dark(1, True)
    sc.run(1.5)
    assert kinds(sc.take()) == ["ion_dark"]
    pos = sc.tracker.positions()
    gap0 = pos[1] - pos[0]
    sc.run(30.0)
    assert sc.take() == []
    pos = sc.tracker.positions()
    assert np.hypot(*(pos[1] - pos[0] - gap0)) < 0.05


def test_learned_psf_is_smoothed():
    """Successive PSF fits on averaged frames are averaged; a fit that sees no
    halo keeps the halo known; one stray fit is ignored, a change the next fit
    confirms (a refocus) is taken at once."""
    from ioncam.analysis.tracker import _smoothed_psf

    def fit(s, f=0.15, h=4.5):
        return {"sigma": (s, s), "halo_fraction": f, "halo_sigma": h, "n_ions": 2}
    e = _smoothed_psf(None, fit(1.80, f=0.0, h=0.0))
    e = _smoothed_psf(e, fit(1.86))           # the first halo seen is taken at once
    assert (e["halo_fraction"], e["halo_sigma"]) == pytest.approx((0.15, 4.5))
    for s in (1.76, 1.84, 1.78):
        e = _smoothed_psf(e, fit(s))
    assert e["sigma"][0] == pytest.approx(1.81, abs=0.02)
    e = _smoothed_psf(e, fit(1.80, f=0.01, h=1.5))
    assert (e["halo_fraction"], e["halo_sigma"]) == pytest.approx((0.15, 4.5))
    stray = _smoothed_psf(e, fit(2.4))
    assert stray["sigma"] == e["sigma"]
    assert _smoothed_psf(stray, fit(1.81))["sigma"][0] == pytest.approx(1.81, abs=0.02)
    assert _smoothed_psf(stray, fit(2.42, f=0.35))["sigma"] == e["sigma"]    # no match
    refocused = _smoothed_psf(stray, fit(2.42))
    assert refocused["sigma"][0] == pytest.approx(2.41, abs=0.01)


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


# A tight trap like the lab's: PSF core 1.2 px, chain length scale 3.4 px - two
# ions 4.3 px apart, so single frames often show them as one spot.
TIGHT = dict(psf_sigma=1.2, scale=3.4, exposure=0.05)


@pytest.mark.parametrize("n,site", [(2, 0), (2, 1), (3, 0), (3, 1)])
def test_tight_chain_dark_then_lost(n, site):
    """Dark: the rest stays where it was.  Lost: the rest re-centres (in a tight
    chain much like two ions seen as one) - decided on the averaged frames."""
    sc = settle(n, **TIGHT)
    sc.scene.set_dark(site, True)
    sc.run(2.5)
    assert kinds(sc.take()) == ["ion_dark"]
    sc.scene.set_dark(site, False)
    sc.run(2.5)
    assert kinds(sc.take()) == ["ion_bright"]
    sc.scene.lose_ion(site)
    sc.run(3.0)
    ev = sc.take()
    assert kinds(ev) == ["ion_lost"], ev
    assert (sc.tracker.n_bright, sc.tracker.n_dark) == (n - 1, 0)


def test_tight_chain_psf_is_learned_from_the_crystal():
    """Detection starts with the default PSF (1.8 px); the tracker measures the
    real one (1.2 px) on averaged frames and hands it to the detector."""
    sc = settle(2, **TIGHT)
    sc.run(5.0)
    assert sc.take() == []
    assert any(sc.psf_messages)
    assert abs(sc.cfg.detection.psf_sigma - 1.2) < 0.15


def test_full_sensor_start():
    """The whole 512 x 512 sensor: the faint-ion integrator learns the static
    background instead of reporting its noise as ions."""
    sc = Scenario(n_ions=3)
    sc.settings = sc.settings.replace(roi=None)
    sc.applied = sc.cam.configure(sc.settings, "live")
    sc.run(4.0)
    assert kinds(sc.take()) == ["ions_detected"]
    assert sc.tracker.n_bright == 3
    sc.run(4.0)
    assert sc.take() == []


def test_brightness_levels_follow_every_ion():
    """The per-ion brightness (the GUI's trace) is measured in every frame: a
    dark ion reads ~0 at once, the others ~1."""
    sc = settle(3, **TIGHT)
    sc.scene.set_dark(1, True)
    sc.step(3)
    lv = [s.level for s in sc.tracker.sites]
    assert lv[1] < 0.3 and lv[0] > 0.6 and lv[2] > 0.6
    assert 0.8 < sc.tracker.fluorescence_level < 1.2


def test_encircled_radius():
    from ioncam.analysis.tracker import encircled_radius
    assert encircled_radius(1.0, fraction=0.8) == pytest.approx(1.794, abs=1e-3)
    assert encircled_radius(1.2, 0.15, 4.5) > encircled_radius(1.2)      # the halo counts


@pytest.mark.parametrize("seed", [1, 3])
def test_refocusing_is_not_a_change_of_the_crystal(seed):
    """Defocus (PSF core 1.2 -> 1.8 px): a detector still assuming the sharp
    PSF splits the ions, but the same ions with a re-fitted wider PSF explain
    the image - "focus_changed", no ions trapped.  The markers (80 % of the
    light) follow within the settling time."""
    from ioncam.analysis.tracker import encircled_radius
    sc = settle(2, seed=seed, **TIGHT)
    sc.scene.set_params(psf_sigma=1.8)
    sc.run(8.0)
    assert kinds(sc.take()) == ["focus_changed"]
    assert sc.tracker.n_bright == 2
    halo = sc.cfg.simulator.psf_halo_fraction, sc.cfg.simulator.psf_halo_sigma
    assert sc.tracker.marker_radius == pytest.approx(encircled_radius(1.8, *halo), rel=0.12)


@pytest.mark.parametrize("angle", [45, 90])
def test_rotated_trap(angle):
    """The simulator turns the whole trap in the image (sim trap_angle_deg):
    dark vs lost work along any chain direction."""
    sc = settle(3, trap_angle_deg=angle, **TIGHT)
    pos = sc.tracker.positions()
    d = pos[-1] - pos[0]
    assert math.degrees(math.atan2(abs(d[1]), abs(d[0]))) == pytest.approx(angle, abs=3)
    sc.scene.set_dark(2, True)
    sc.run(2.5)
    assert kinds(sc.take()) == ["ion_dark"]
    sc.scene.set_dark(2, False)
    sc.run(2.5)
    sc.take()
    sc.scene.lose_ion(0)
    sc.run(3.0)
    assert kinds(sc.take()) == ["ion_lost"] and sc.tracker.n_bright == 2


# the lab's trap and imaging: PSF core 0.8 px + 26 % in a halo of 1.8 px (as
# located), l = 3.4 px, the crystal at 45 deg
LAB = dict(scale=3.4, psf_sigma=0.8, psf_halo_fraction=0.26, psf_halo_sigma=1.8,
           exposure=0.1, em_gain=10, trap_angle_deg=45.0)


def _lab(n=2, seed=1):
    sc = Scenario(n_ions=n, seed=seed, **LAB)
    d = sc.cfg.detection
    d.psf_sigma, d.psf_halo_fraction, d.psf_halo_sigma = 0.8, 0.26, 1.8
    return sc


def _with_spot(sc, dx, dy, fraction):
    """A static point of light (stray light, a reflection) at (dx, dy) from
    ion 0 as it sits now, ``fraction`` of an ion's light, in every frame."""
    from scipy.special import ndtr
    scene = sc.scene
    x0, y0 = scene._chain_positions(len(scene.crystallized(sc.clock.t)))[0] + [dx, dy]
    render = scene.expected_electrons

    def with_spot(roi, hbin, vbin, exposure, *args, **kw):
        img = render(roi, hbin, vbin, exposure, *args, **kw)
        h, w = img.shape
        xe = roi[0] - 0.5 + hbin * np.arange(w + 1)
        ye = roi[1] - 0.5 + vbin * np.arange(h + 1)
        sg = scene.psf_sigma
        ex = ndtr((xe[1:] - x0) / sg) - ndtr((xe[:-1] - x0) / sg)
        ey = ndtr((ye[1:] - y0) / sg) - ndtr((ye[:-1] - y0) / sg)
        rate = scene.photon_rate * exposure * scene.cfg.quantum_efficiency * fraction
        return img + rate * np.outer(ey, ex)
    scene.expected_electrons = with_spot


@pytest.mark.parametrize("spot", [(2.5, -1.0, 0.45), (1.5, 1.5, 0.4), (6.0, 6.0, 0.3)])
def test_faint_spot_next_to_the_crystal_is_no_ion(spot):
    """Two ions and a static spot of stray light beside them (or between
    them): found with the crystal, it is dropped as no ion within seconds -
    it shines far less than the ions - and never taken for one again (it
    was "3 bright ions", flipping to 2 and back)."""
    sc = _lab()
    _with_spot(sc, *spot)
    sc.run(10.0)
    ev = sc.take()
    assert sc.tracker.n_bright == 2 and sc.tracker.n_dark == 0, kinds(ev)
    assert any("not an ion" in e["message"] for e in ev if e["kind"] == "crystal_refined")
    sc.run(30.0)
    assert sc.tracker.n_bright == 2 and sc.tracker.n_dark == 0
    assert not {"crystal_refined", "ion_trapped", "ion_dark"} & set(kinds(sc.take()))


def test_blinking_ions_stay_ions():
    """Ions dark in half the frames at random (qubit states, blinking): each
    shines like the others whenever lit - none is dropped as stray light."""
    sc = _lab(seed=2)
    sc.run(3.0)
    sc.take()
    sc.scene.set_params(qubit_dark_probability=0.5)
    sc.run(40.0)
    assert not any("not an ion" in e["message"] for e in sc.take())
    assert len(sc.tracker.sites) == 2


def test_no_hidden_ion_closer_than_the_trap_allows():
    """An extra source the averaged frames show next to an ion is no hidden
    ion when ions in this trap cannot come that close (two ions 4.3 px
    apart: l = 3.4 px, three would be >= 3.7 px apart), nor where a spot was
    found to be no ion; a third ion where a chain of three would put it is."""
    from ioncam.analysis.chain import equilibrium_positions
    from ioncam.analysis.tracker import ChainTracker, Site
    from ioncam.config import Config
    tr = ChainTracker(Config().tracker)
    two = 100.0 + 3.4 * equilibrium_positions(2)
    tr.sites = [Site(id=1, x=float(two[0]), y=100.0), Site(id=2, x=float(two[1]), y=100.0)]
    tr._update_chain_fit()
    close = np.array([[two[0], 100.0], [two[1], 100.0], [two[0] + 1.6, 100.0]])
    assert tr._not_ions(close, [2]) == {2}
    three = 100.0 + 3.4 * equilibrium_positions(3)
    chain3 = np.column_stack([three, np.full(3, 100.0)])
    assert tr._not_ions(chain3, [1]) == set()
    tr._not_ion_at = [np.array([100.3, 100.0])]
    assert tr._not_ions(chain3, [1]) == {1}
