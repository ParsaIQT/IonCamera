import numpy as np
import pytest

from ioncam.analysis.chain import equilibrium_positions, fit_chain, min_spacing
from ioncam.camera.base import normalize_roi, roi_from_sdk, roi_to_sdk
from ioncam.config import Config


# --------------------------------------------------------------------- config

def test_config_roundtrip(tmp_path):
    c = Config()
    c.readout.roi = (1, 2, 9, 10)
    c.detection.analysis_roi = (5, 6, 70, 80)
    p = tmp_path / "ioncam.toml"
    c.save(str(p))
    c2 = Config.load(str(p))
    assert c2.to_dict() == c.to_dict()
    text = p.read_text()
    assert "[readout]" in text and "# exposure time [s]" in text


def test_config_defaults_match_typical_grabber_settings():
    r = Config().readout
    assert (r.exposure, r.em_gain, r.hs_speed_mhz, r.vs_speed_us) == (0.0015, 200, 5.0, 0.5)
    assert (r.vs_amplitude, r.preamp_gain_index, r.hbin, r.vbin) == (0, -1, 4, 4)
    assert r.trigger == "external" and r.camera_link and r.baseline_clamp
    assert not r.frame_transfer       # exposure = set time, not trigger spacing


def test_config_coercion_and_errors():
    c = Config()
    assert c.set("live", "exposure", "0.2") == 0.2
    assert c.set("live", "em_gain", 10.0) == 10
    assert c.set("live", "roi", [0, 0, 100, 100]) == (0, 0, 100, 100)
    assert c.set("live", "roi", []) is None
    with pytest.raises(KeyError):
        c.set("live", "nope", 1)
    with pytest.raises(ValueError):
        c.set("live", "em_gain", 1.5)
    with pytest.raises(ValueError):
        c.set("live", "trigger", "sometimes")
    warnings = c.update_from_dict({"bogus": {}, "live": {"exposure": 0.3, "nope": 2}})
    assert len(warnings) == 2 and c.live.exposure == 0.3


def test_roi_conventions():
    # Solis "Left, Right, Bottom, Top" 257,264,161,168 == 8x8 px
    roi = roi_from_sdk(257, 264, 161, 168)
    assert roi == (256, 160, 264, 168)
    assert roi_to_sdk(roi) == (257, 264, 161, 168)
    assert normalize_roi((256, 160, 266, 171), (512, 512), 4, 4) == (256, 160, 264, 168)
    assert normalize_roi(None, (512, 512), 4, 4) == (0, 0, 512, 512)
    assert normalize_roi((510, 0, 600, 3), (512, 512), 4, 4) == (508, 0, 512, 4)


# ---------------------------------------------------------------------- chain

@pytest.mark.parametrize("n, expected", [
    (2, [0.62996]),
    (3, [0.0, 1.0772]),
    (4, [0.4544, 1.4368]),
    (5, [0.0, 0.8221, 1.7429]),
])
def test_james_equilibrium_positions(n, expected):
    u = equilibrium_positions(n)
    assert np.allclose(u, -u[::-1], atol=1e-12)          # symmetric
    assert np.all(np.diff(u) > 0)                        # ordered
    assert np.allclose(sorted(set(np.round(np.abs(u), 4))), expected, atol=2e-4)


def test_min_spacing_scaling():
    # James: min spacing ~ 2.018 / N^0.559
    for n in (10, 30, 60):
        assert min_spacing(n) == pytest.approx(2.018 / n ** 0.559, rel=0.04)


def _chain(n, scale=6.0, center=(260.0, 170.0), angle=0.0):
    u = equilibrium_positions(n)
    a = np.deg2rad(angle)
    return np.column_stack([center[0] + scale * u * np.cos(a), center[1] + scale * u * np.sin(a)])


def test_fit_chain_infers_hidden_middle_ion():
    pos = _chain(3)
    f = fit_chain(pos[[0, 2]], max_hidden=2, scale_prior=6.0, center_prior=(260.0, 170.0))
    assert f.n == 3 and f.hidden_sites == (1,)
    assert np.allclose(f.hidden_positions(), pos[[1]], atol=1e-6)


def test_fit_chain_prefers_loss_when_ions_rearranged():
    f = fit_chain(_chain(2), max_hidden=2, scale_prior=6.0, center_prior=(260.0, 170.0))
    assert f.n == 2 and f.hidden_sites == ()


def test_fit_chain_hidden_end_ion_needs_priors():
    pos = _chain(3)[[1, 2]]
    assert fit_chain(pos, max_hidden=2, scale_prior=6.0,
                     center_prior=(260.0, 170.0)).hidden_sites == (0,)


def test_fit_chain_single_bright_with_dark_partner():
    pos = _chain(2)[[1]]
    f = fit_chain(pos, max_hidden=1, scale_prior=6.0, center_prior=(260.0, 170.0))
    assert f.n == 2 and f.hidden_sites == (0,)


def test_fit_chain_tilted_long_chain_with_noise():
    rng = np.random.default_rng(1)
    pos = _chain(12, angle=20.0)
    keep = [i for i in range(12) if i not in (3, 8)]
    f = fit_chain(pos[keep] + rng.normal(0, 0.15, (10, 2)), max_hidden=3, scale_prior=6.0,
                  center_prior=(260.0, 170.0))
    assert f.n == 12 and f.hidden_sites == (3, 8)
    assert f.is_linear
