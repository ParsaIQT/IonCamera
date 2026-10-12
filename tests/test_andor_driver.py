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
    assert get["SetVSSpeed"] == (1,)                     # 0.5 us (not SetFKVShiftSpeed!)
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
