"""The real Andor driver on the emulated SDK (no hardware)."""

import time

import numpy as np
import pytest

from ioncam.camera.andor import AndorCamera
from ioncam.camera.base import CameraError
from ioncam.camera.fake_sdk import FakeAtmcd
from ioncam.config import Config


@pytest.fixture
def cam():
    cfg = Config()
    cfg.simulator.seed = 2
    sdk = FakeAtmcd(cfg.simulator)
    c = AndorCamera(cfg.camera, sdk=sdk)
    c.open()
    yield c, sdk, cfg
    c.close()


def names(sdk, since=0):
    return [n for n, _ in sdk.calls[since:]]


def test_open_sets_raw_orientation_and_cooling(cam):
    c, sdk, cfg = cam
    n = names(sdk)
    assert ("SetImageRotate", (0,)) in sdk.calls and ("SetImageFlip", (0, 0)) in sdk.calls
    assert ("SetEMGainMode", (3,)) in sdk.calls
    assert "CoolerON" in n and ("SetCoolerMode", (1,)) in sdk.calls
    assert ("SetTemperature", (-60,)) in sdk.calls
    assert c.sensor_shape == (512, 512)
    caps = c.capabilities()
    assert caps["hs_speeds_mhz"]["em"] == [17.0, 10.0, 5.0, 1.0]
    assert caps["preamp_gain_texts"][2] == "Gain 3"


def test_readout_configuration_calls(cam):
    c, sdk, cfg = cam
    start = len(sdk.calls)
    a = c.configure(cfg.readout.replace(roi=(244, 156, 284, 180)), "readout")
    calls = sdk.calls[start:]
    get = {name: args for name, args in calls}
    assert get["SetAcquisitionMode"] == (5,)            # run till abort: no re-arming
    assert get["SetOutputAmplifier"] == (0,)             # EM amplifier
    assert get["SetHSSpeed"] == (0, 2)                   # 5 MHz found by value
    assert get["SetVSSpeed"] == (1,)                     # fastest recommended: 0.5 us
    #                                                     (not SetFKVShiftSpeed!)
    assert "SetFKVShiftSpeed" not in get
    assert get["SetVSAmplitude"] == (0,)                 # Normal
    assert get["SetPreAmpGain"] == (2,)                  # "Gain 3"
    assert get["SetBaselineClamp"] == (1,)
    assert get["SetEMCCDGain"] == (200,)
    assert get["SetFrameTransferMode"] == (0,)
    assert get["SetTriggerMode"] == (1,)                 # external
    assert get["SetCameraLinkMode"] == (1,)
    assert get["SetImage"] == (4, 4, 245, 284, 157, 180)
    assert get["SetExposureTime"] == (0.0015,)
    order = [n for n, _ in calls]
    assert order.index("SetOutputAmplifier") < order.index("SetHSSpeed") < \
        order.index("SetPreAmpGain")
    assert a.frame_shape == (6, 10) and a.em_gain == 200 and a.hs_speed_mhz == 5.0
    assert a.min_trigger_period > a.exposure


def test_frames_are_unsigned_16_bit(cam):
    c, sdk, cfg = cam
    sdk.sim.scene.photon_rate = 3e6                      # saturate at high EM gain
    c.configure(cfg.live.replace(em_gain=300, exposure=0.05, roi=(200, 120, 330, 220)), "live")
    c.start()
    frames = []
    t = time.time()
    while len(frames) < 2 and time.time() - t < 5:
        frames += c.wait_frames(0.3)
    c.stop()
    img = frames[-1][1]
    assert img.dtype == np.uint16 and img.shape == (100, 130)
    assert img.max() > 32767 and img.min() >= 0          # no signed wrap-around
    assert [i for i, _ in frames][:2] == [1, 2]


def test_external_trigger_frames(cam):
    c, sdk, cfg = cam
    c.configure(cfg.readout.replace(roi=(244, 156, 284, 180)), "readout")
    c.start()
    assert c.wait_frames(0.05) == []                     # nothing without triggers
    sdk.sim.trigger(3, spacing=0.01)
    time.sleep(0.06)
    fr = c.wait_frames(0.2)
    c.stop()
    assert len(fr) == 3 and fr[0][1].shape == (6, 10)


def test_preamp_fallback_and_roi_adjustment(cam):
    c, sdk, cfg = cam
    a = c.configure(cfg.live.replace(hs_speed_mhz=17.0, preamp_gain_index=2,
                                     roi=(10, 10, 103, 50), hbin=4, vbin=4), "live")
    assert any("pre-amp" in w for w in a.warnings)
    assert ("SetPreAmpGain", (1,)) in sdk.calls
    assert a.roi == (10, 10, 102, 50)                    # width made a multiple of 4
    assert any("adjusted" in w for w in a.warnings)


def test_single_scan_finishes(cam):
    c, sdk, cfg = cam
    c.configure(cfg.locate.replace(acquisition="single", exposure=0.05,
                                   roi=(200, 120, 330, 220)), "snap")
    c.start()
    frames = []
    t = time.time()
    while c.is_acquiring() and time.time() - t < 5:
        frames += c.wait_frames(0.2)
    assert len(frames) == 1 and not c.is_acquiring()


def test_invalid_trigger_mode_raises(cam):
    c, sdk, cfg = cam
    with pytest.raises(ValueError):
        c.configure(cfg.live.replace(trigger="whenever"), "live")


def test_missing_sdk_gives_clear_error():
    cfg = Config()
    c = AndorCamera(cfg.camera)
    import importlib.util
    if importlib.util.find_spec("pyAndorSDK2") is not None:
        pytest.skip("pyAndorSDK2 installed")
    with pytest.raises(CameraError, match="pyAndorSDK2"):
        c.open()


def test_ixon_ultra_888_readout_speeds():
    """An iXon Ultra 888 (emulated): 1024 x 1024 sensor; a speed it does not
    have is taken by ratio - the 5 MHz of the readout settings is 10 MHz on
    the 888 (30, 20, 10, 1 MHz), not 1 MHz; the default pre-amp gain is the
    highest it has (no warning), and one it does not have is replaced by the
    nearest available one (with a warning)."""
    from ioncam.camera.andor import _nearest
    assert _nearest([30.0, 20.0, 10.0, 1.0], 5.0) == 2 and _nearest([17.0, 10.0, 5.0, 1.0], 5.0) == 2
    cfg = Config()
    cfg.simulator.camera_model = "888"
    sdk = FakeAtmcd(cfg.simulator)
    c = AndorCamera(cfg.camera, sdk=sdk)
    info = c.open()
    try:
        assert c.sensor_shape == (1024, 1024) and info["pixel_size_um"] == 13.0
        a = c.configure(cfg.readout.replace(roi=(600, 700, 616, 716), trigger="internal"),
                        "readout")
        assert a.hs_speed_mhz == 10.0 and a.frame_shape == (4, 4)
        assert a.preamp_gain == 2.0 and not any("pre-amp" in w for w in a.warnings)  # Gain 2
        b = c.configure(cfg.readout.replace(roi=(600, 700, 616, 716), trigger="internal",
                                            preamp_gain_index=2), "readout")
        assert b.preamp_gain == 2.0 and any("pre-amp" in w for w in b.warnings)
    finally:
        c.close()


def test_ixon_ultra_888_vertical_shift_speed(tmp_path):
    """An iXon Ultra 888 does not shift the charge at 0.6 us without a higher
    vertical clock voltage (Andor's hardware guide), and the 0.5 us of the old
    locate / readout settings was matched to 0.6 us on it: the long exposure,
    the verification and the armed readout were garbled.  Now a speed the
    camera has is used exactly as set (one faster than recommended at normal
    voltage with a warning, never replaced), one it does not have is never
    guessed (the fastest recommended, with a warning), and the default is the
    fastest recommended.  Each warning is said once.  The 897 keeps 0.5 us."""
    from ioncam.camera.base import vs_speed_index
    from ioncam.camera.simulator import SimulatedCamera
    from ioncam.server import build
    s888, s897 = [0.6, 1.13, 2.2, 4.33], [0.3, 0.5, 0.9, 1.7, 3.3]
    assert vs_speed_index(s888, 1, 0.0, 0) == (1, "")
    i, note = vs_speed_index(s888, 1, 0.5, 0)                 # an 897 value: not guessed
    assert i == 1 and "not one of this camera's" in note and "using 1.13 us" in note
    i, note = vs_speed_index(s888, 1, 0.6, 0)                 # as set, with a warning
    assert i == 0 and "faster than the 1.13 us" in note
    assert vs_speed_index(s888, 1, 0.6, 2) == (0, "")         # a higher voltage: no warning
    assert vs_speed_index(s888, 1, 2.2, 0) == (2, "")
    i, note = vs_speed_index(s888, 3, 1.13, 0)                # a camera recommending 4.33 us
    assert i == 1 and "faster than the 4.33 us" in note       # still gets the 1.13 us asked
    assert vs_speed_index(s897, 1, 0.5, 0) == (1, "")
    assert vs_speed_index(s897, 1, 0.0, 0) == (1, "")

    cfg = Config()
    cfg.simulator.camera_model = "888"
    where = dict(roi=(600, 700, 616, 716), trigger="internal")
    c = AndorCamera(cfg.camera, sdk=FakeAtmcd(cfg.simulator))
    c.open()
    try:
        assert c.capabilities()["vs_fastest_recommended"] == 1.13
        for settings in (cfg.locate, cfg.readout):                 # the defaults
            a = c.configure(settings.replace(**where), "test")
            assert a.vs_speed_us == 1.13 and not any("vertical" in w for w in a.warnings)
        old = c.configure(cfg.readout.replace(vs_speed_us=0.5, **where), "readout")
        assert old.vs_speed_us == 1.13
        assert any("not one of this camera's" in w for w in old.warnings)
        fast = c.configure(cfg.readout.replace(vs_speed_us=0.6, **where), "readout")
        assert fast.vs_speed_us == 0.6 and ("SetVSSpeed", (0,)) in c._sdk.calls
        assert any("faster than the 1.13 us" in w for w in fast.warnings)
        boost = c.configure(cfg.readout.replace(vs_speed_us=0.6, vs_amplitude=2, **where),
                            "readout")
        assert boost.vs_speed_us == 0.6 and not any("vertical" in w for w in boost.warnings)
        slow = c.configure(cfg.readout.replace(vs_speed_us=4.33, **where), "readout")
        assert slow.vs_speed_us == 4.33 and not any("vertical" in w for w in slow.warnings)
    finally:
        c.close()
    sim = SimulatedCamera(cfg.simulator)
    a = sim.configure(cfg.readout.replace(vs_speed_us=0.5, **where), "readout")
    assert a.vs_speed_us == 1.13 and any("not one of this camera's" in w for w in a.warnings)
    assert sim.configure(cfg.readout.replace(vs_speed_us=2.2, **where), "readout").vs_speed_us \
        == 2.2

    # an old configuration file on an 888: said once in the event log, not at every start
    cfg.storage.data_dir = str(tmp_path)
    cfg.camera.backend = "fake_sdk"
    cfg.live.vs_speed_us = 0.5
    engine, _ = build(cfg)
    engine.start()
    try:
        engine.update_live(exposure=0.04)
        engine.update_live(exposure=0.05)
        notes = [e for e in engine.events.since(0) if e["kind"] == "camera_settings"]
        vs = [e for e in notes if "vertical shift speed" in e["message"]]
        assert len(vs) == 1 and vs[0]["severity"] == "warning"
        assert engine.camera._applied.vs_speed_us == 1.13
    finally:
        engine.shutdown()


def test_shutter_fan_and_electrons_per_count(cam):
    """Shutter and fan through the real driver (SetShutterEx when the camera
    controls the internal and external shutters separately), and the
    electrons per count of the readout speed and pre-amp gain in use, asked of
    the camera (GetSensitivity in the argument order of Andor's header:
    channel, horizontal speed index, amplifier, pre-amp) unless configured."""
    c, sdk, cfg = cam
    caps = c.capabilities()
    assert caps["shutter"] == {"control": True, "independent": True, "internal": True,
                               "min_close_ms": 27, "min_open_ms": 27}
    assert caps["fan"] == {"control": True, "low": True}
    start = len(sdk.calls)
    c.set_shutter("closed", "series", False, 27, 30)
    # SetShutterEx(typ, mode, closingtime, openingtime, extmode)
    assert ("SetShutterEx", (0, 2, 30, 27, 5)) in sdk.calls[start:]
    assert sdk.sim.shutter == "closed"
    c.set_shutter("open")
    c.set_fan("low")
    c.set_fan("full")
    assert [a for n, a in sdk.calls[start:] if n == "SetFanMode"] == [(1,), (0,)]
    with pytest.raises(CameraError):
        c.set_shutter("ajar")
    with pytest.raises(CameraError):
        c.set_fan("turbo")

    where = dict(roi=(244, 156, 284, 180), trigger="internal")
    a = c.configure(cfg.readout.replace(**where), "readout")       # 5 MHz, "Gain 3"
    assert ("GetSensitivity", (0, 2, 0, 2)) in sdk.calls
    assert (a.sensitivity, a.sensitivity_source, a.em_gain_real) == (4.0, "camera", True)
    assert a.pe_factor == pytest.approx(4.0 / (a.exposure * a.em_gain))
    b = c.configure(cfg.readout.replace(preamp_gain_index=0, **where), "readout")
    assert b.sensitivity == 12.0                                  # "Gain 1": 3x the electrons
    # normalised brightness is photoelectrons/s: the same whatever the gain settings
    assert a.gain_factor * a.pe_factor == pytest.approx(1.0)
    assert b.gain_factor * b.pe_factor == pytest.approx(1.0)
    c.cfg.sensitivity = 3.7                                       # the performance sheet's
    d = c.configure(cfg.readout.replace(**where), "readout")
    assert (d.sensitivity, d.sensitivity_source) == (3.7, "configuration")
    c.cfg.sensitivity = 0.0
    c.cfg.em_gain_mode = 0                                        # DAC units: not the gain
    e = c.configure(cfg.readout.replace(**where), "readout")
    assert not e.em_gain_real and e.pe_factor == 0.0
