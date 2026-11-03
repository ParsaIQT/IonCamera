"""The ARTIQ mapping check (examples/artiq/repository/ioncam_grabber_mapping_check.py)
run against an emulated Grabber: ROI engines summing as the gateware does
(artiq/gateware/grabber/core.py), fed with the Camera Link frames of a camera.

On an iXon Ultra 888 with a 3 x 2 super-pixel AQROI the check stopped with "the
Grabber frame is larger than 512 pixels", once at the width, twice at the
height: a column (row) counted as inside the frame if any one of 20 frames had
light in it, so a single frame that was not the AQROI's (a keep-clean cycle,
a frame without a trigger) made every column look lit.  Now engine 0 sums the
whole frame beside every probe, frames far off the median are left out and
counted, and the report says what reached the Grabber."""

import asyncio
import importlib.util
import inspect
import os
import sys
import time
import types

import numpy as np
import pytest

from ioncam.analysis.readout import GrabberGeometry, grabber_roi_sum_exact
from ioncam.config import Config
from ioncam.server import build

EXAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "examples",
                       "artiq", "repository", "ioncam_grabber_mapping_check.py")


def roi_sum(frame, x0, y0, x1, y1):
    """One ROI engine over one frame, as the gateware works it cycle by cycle:
    a flag is set when the pixel counter equals x0 (y0) and cleared when it
    equals x1 (y1) or at the end of the frame - not at the end of a line - and
    a pixel counts when both flags are set after its own cycle.  Between
    frames the counters stand at (0, 0); a line ends with a cycle at x = W."""
    h, w = frame.shape

    def fy(g, y):
        g = True if y == y0 else g
        return False if y == y1 else g

    def fx(g, x):
        g = True if x == x0 else g
        return False if x == x1 else g

    xg, yg = fx(False, 0), fy(False, 0)          # after the frame before (FVAL low)
    total = 0.0
    events = sorted({e for e in (x0, x1) if 0 <= e < w})
    for y in range(h):
        yg, xg = fy(yg, y), fx(xg, 0)           # line blanking: x = 0
        start = 0
        for e in events:
            if xg and yg:
                total += frame[y, start:e].sum()
            xg, start = fx(xg, e), e
        if xg and yg:
            total += frame[y, start:].sum()
        xg = fx(xg, w)                           # the cycle after the last pixel
    return total


def test_roi_sum_emulates_the_gateware():
    """x0 <= x < x1, y0 <= y < y1; an x1 beyond the line carries the ROI into
    the next lines (the gateware clears x only at x1 or the frame's end) - and
    so does an x0 equal to the line's length W: the counter stands at W for
    the cycle after the line's last pixel.  On the first line nothing carries
    over (the flags are cleared at the end of the frame before)."""
    f = np.arange(12, dtype=float).reshape(3, 4)
    assert roi_sum(f, 1, 0, 2, 3) == f[:, 1].sum()
    assert roi_sum(f, 0, 1, 4, 2) == f[1].sum()
    assert roi_sum(f, 0, 0, 4095, 4095) == f.sum()
    assert roi_sum(f, 0, 3, 4, 4) == 0 and roi_sum(f, 5, 0, 6, 4095) == 0
    assert roi_sum(f, 2, 1, 6, 2) == f[1].sum()     # x on since x = 2 of line 0: all of line 1
    assert roi_sum(f, 2, 0, 6, 4095) == f.sum() - f[0, :2].sum()   # from (2, 0) on
    assert roi_sum(f, 4, 0, 5, 4095) == f[1:].sum()          # a column at x = W: lit
    assert roi_sum(f, 4, 0, 5, 1) == 0 and roi_sum(f, 3, 0, 4, 1) == f[0, 3]
    assert roi_sum(f, 4, 1, 5, 2) == f[1].sum()             # pixel (W, 1): all of line 1
    # the same as ioncam's pixel-by-pixel model of the ROI engine
    rng = np.random.default_rng(3)
    for _ in range(300):
        h, w = rng.integers(1, 6, 2)
        g = rng.integers(0, 1000, (h, w))
        x0, x1 = rng.integers(0, w + 3, 2)
        y0, y1 = rng.integers(0, h + 3, 2)
        assert roi_sum(g.astype(float), x0, y0, x1, y1) == \
            grabber_roi_sum_exact(g, x0, y0, x1, y1), (g, x0, y0, x1, y1)


class Timeline:
    def __init__(self):
        self.now = 0
        self.par = None

    def advance(self, mu):
        if self.par is None:
            self.now += int(mu)
        else:                       # in a parallel block every statement starts together
            self.par[1] = max(self.par[1], self.now + int(mu))


class GrabberTimeoutException(Exception):
    pass


class RTIOOverflow(Exception):
    pass


class FakeGrabber:
    """ROI engines over the frames a camera delivers while the gate is open."""

    def __init__(self, t, camera):
        self.t, self.camera = t, camera
        self.rois, self.fifo, self.open = {}, [], 0
        self.depth, self.overflow = 64, False      # input FIFO entries (sentinel + sums)

    def gate_roi(self, mask):
        self.open = mask

    def setup_roi(self, n, x0, y0, x1, y1):
        self.rois[n] = (x0, y0, x1, y1)
        self.t.advance(32)

    def gate_roi_pulse(self, mask, dt):
        self.open = mask
        for frame in self.camera.untriggered():
            self.deliver(frame)
        self.t.advance(round(dt * 1e9))

    def deliver(self, frame):
        n = [i for i in range(16) if self.open >> i & 1]
        if n:
            if sum(1 + len(s) for s in self.fifo) + 1 + len(n) > self.depth:
                self.overflow = True
                return
            self.fifo.append([int(round(roi_sum(frame, *self.rois[i]))) for i in n])

    def input_mu(self, data, timeout_mu=-1):
        if self.overflow:
            self.overflow = False
            raise RTIOOverflow("RTIO input overflow")
        if not self.fifo:
            raise GrabberTimeoutException("Timeout before Grabber frame available")
        sums = self.fifo.pop(0)
        assert len(sums) == len(data), "out of sync"
        data[:] = sums


class FakeTTL:
    def __init__(self, t, grabber, camera):
        self.t, self.grabber, self.camera = t, grabber, camera

    def pulse(self, dt):
        for frame in self.camera.trigger():
            self.grabber.deliver(frame)
        self.t.advance(round(dt * 1e9))


class FakeCore:
    def __init__(self, t, grabber):
        self.t, self.grabber = t, grabber

    def reset(self):
        self.grabber.fifo.clear()
        self.grabber.overflow = False
        self.t.advance(125000)

    def break_realtime(self):
        pass

    def seconds_to_mu(self, s):
        return int(round(s * 1e9))


def load_example(monkeypatch, t):
    """The example module with ARTIQ's names faked (no ARTIQ needed)."""
    class Parallel:
        def __enter__(self):
            t.par = [t.now, t.now]

        def __exit__(self, *exc):
            t.now, t.par = t.par[1], None

    class Value:
        def __init__(self, default, **kwargs):
            self.default = default

    class EnvExperiment:
        def __init__(self, devices, arguments=None):
            self._devices, self._arguments, self.datasets = devices, arguments or {}, {}
            self.build()

        def setattr_device(self, key):
            setattr(self, key, self._devices[key])

        def get_device(self, key):
            return self._devices[key]

        def get_device_db(self):
            return {}

        def setattr_argument(self, key, processor):
            setattr(self, key, self._arguments.get(key, processor.default))

        def set_dataset(self, key, value, **kwargs):
            self.datasets[key] = value

    def rpc(f=None, flags=None):
        return f if callable(f) else (lambda g: g)

    experiment = types.ModuleType("artiq.experiment")
    names = dict(EnvExperiment=EnvExperiment, NumberValue=Value, BooleanValue=Value,
                 kernel=lambda f: f, rpc=rpc, parallel=Parallel(), ms=1e-3, us=1e-6, TNone=None,
                 delay=lambda s: t.advance(round(s * 1e9)), now_mu=lambda: t.now)
    experiment.__dict__.update(names)
    experiment.__all__ = list(names)
    grabber = types.ModuleType("artiq.coredevice.grabber")
    grabber.GrabberTimeoutException = GrabberTimeoutException
    exceptions = types.ModuleType("artiq.coredevice.exceptions")
    exceptions.RTIOOverflow = RTIOOverflow
    for name, mod in (("artiq", types.ModuleType("artiq")), ("artiq.experiment", experiment),
                      ("artiq.coredevice", types.ModuleType("artiq.coredevice")),
                      ("artiq.coredevice.grabber", grabber),
                      ("artiq.coredevice.exceptions", exceptions)):
        monkeypatch.setitem(sys.modules, name, mod)
    spec = importlib.util.spec_from_file_location("ioncam_grabber_mapping_check", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_experiment(mod, t, camera, ioncam, **arguments):
    grabber = FakeGrabber(t, camera)
    devices = {"core": FakeCore(t, grabber), "grabber0": grabber,
               "ttl_camera_trigger": FakeTTL(t, grabber, camera), "ioncam": ioncam}
    mod.get_ioncam = lambda exp: ioncam
    return mod.IonCamGrabberMappingCheck(devices, arguments)


class Simulated888:
    """ioncam's simulated 888, its readout frames on Camera Link
    (simulator.camera_link): every trigger is a shot ioncam receives over USB
    and a frame on Camera Link; ``odd_every`` triggers bring an odd frame first
    (a big bias-only frame, ``odd_shape``); ``idle`` odd frames come in a gate
    without a trigger."""

    def __init__(self, engine, api, odd_every=0, odd_shape=(3, 700), idle=0):
        self.engine, self.api = engine, api
        self.odd_every, self.odd_shape, self.idle = odd_every, odd_shape, idle
        self.n = 0

    def odd(self):
        return np.full(self.odd_shape, 500.0)

    def trigger(self):
        k0 = self.engine.monitor.shots
        self.engine.sim_camera().trigger(1, 0.0)
        t0 = time.time()
        while self.engine.monitor.shots < k0 + 1 and time.time() - t0 < 5:
            time.sleep(0.002)
        frame = np.rint(np.array(self.api.sim_grabber_frame(), float))
        self.n += 1
        return ([self.odd()] if self.odd_every and self.n % self.odd_every == 0 else []) + [frame]

    def untriggered(self):
        out, self.idle = [self.odd()] * self.idle, 0      # in the first gate only
        return out


class ApiAsClient:
    """The API object as ARTIQ's sipyco client presents it (no coroutines)."""

    def __init__(self, api):
        self.api = api

    def __getattr__(self, name):
        f = getattr(self.api, name)

        def call(*args, **kwargs):
            r = f(*args, **kwargs)
            return asyncio.run(r) if inspect.iscoroutine(r) else r
        return call


def test_mapping_check_on_the_888_despite_odd_frames(tmp_path, monkeypatch, capsys):
    """The simulated 888 sends its lines unbinned (4 x 4 binning, 4 x 4 AQROI:
    16 x 4 stream pixels); every 4th trigger an odd 700 x 3 frame ends first
    in the gate, and two come in the gate without a trigger.  The check finds
    the 16 x 4 frame, reads it, ioncam finds the layout and maps the ROIs;
    the report counts the odd frames."""
    cfg = Config()
    cfg.storage.data_dir = str(tmp_path)
    cfg.simulator.camera_model = "888"
    cfg.simulator.n_ions = 1
    cfg.simulator.seed = 5
    cfg.simulator.camera_link = [0, 1, 0, 0, 0, 0]
    cfg.tracker.confirm_time = 0.3
    engine, api = build(cfg, simulate=True)
    engine.start()
    try:
        assert engine.wait_for_ions(1, timeout=40)["ok"]
        engine.calibrate_readout(n_ions=1, hbin=4, vbin=4)
        engine.manual_aqroi(4, 4)
        t = Timeline()
        mod = load_example(monkeypatch, t)
        cam = Simulated888(engine, api, odd_every=4, idle=2)
        exp = make_experiment(mod, t, cam, ApiAsClient(api), frames_per_batch=8,
                              frames_per_pixel=8)
        exp.prepare()
        exp.run()
        exp.analyze()
        out = capsys.readouterr().out
        assert exp.size == (16, 4), out
        assert exp.image.shape == (4, 16)
        # every batch of the pixel read has the USB frames of its own shots
        assert len(exp.usb_means) == 5 and all(m is not None for m in exp.usb_means)
        assert exp.n_odd > 0 and exp.n_got == exp.n_gates, out
        assert "frames with the gate open 200 ms and no trigger: 2" in out
        assert "odd frames (more than 3x off): {}".format(exp.n_odd) in out
        assert "ioncam over USB:" in out and "Grabber / USB whole frame:" in out
        assert "Grabber frame: 16 x 4 stream pixels" in out
        assert "applied:" in out and "fit r = 1.0" in out, out
        assert "Grabber frame, pixel by pixel" in out and "ioncam's USB frame" in out
        plan = engine.readout_plan()
        assert GrabberGeometry.from_dict(plan["grabber_geometry"]) == GrabberGeometry(0, 1)
        assert plan["grabber_source"].startswith("measured")
    finally:
        engine.shutdown()


class StaticCamera:
    """Every trigger: the same stream frame."""

    def __init__(self, frame):
        self.frame = np.asarray(frame, float)

    def trigger(self):
        return [self.frame]

    def untriggered(self):
        return []


class StaticIoncam:
    plan = {"frame_size": [3, 2], "binning": [4, 4],
            "applied": {"min_trigger_period": 0.004}}

    def get_readout_plan(self):
        return self.plan

    def arm_readout(self):
        return self.plan

    def take_control(self, *args):
        pass

    def release_control(self, *args):
        pass

    def get_readout_frames(self, last):
        return {"n": 10, "shape": [2, 3], "frame_sums": [3000.0] * 10}


@pytest.mark.parametrize("shape, error", [((2, 3), None),
                                          ((7, 700), "too large to read pixel by pixel"),
                                          ((2, 4200), "lit up to 4095"),
                                          ((700, 7), "too large to read pixel by pixel")])
def test_mapping_check_measures_any_frame_size(monkeypatch, capsys, shape, error):
    """The frame size is found up to the Grabber's 4096 (a 3 x 2 frame, a
    700-pixel line, 700 lines); a frame too large to read pixel by pixel, or
    beyond the counters, stops the check with the report printed."""
    t = Timeline()
    mod = load_example(monkeypatch, t)
    frame = np.full(shape, 500.0)
    exp = make_experiment(mod, t, StaticCamera(frame), StaticIoncam(), frames_per_batch=3)
    exp.prepare()
    if error is None:
        exp.run()
        assert exp.size == (shape[1], shape[0])
        np.testing.assert_allclose(exp.image, frame)
    else:
        with pytest.raises(mod.MappingCheckError, match=error):
            exp.run()
    out = capsys.readouterr().out
    whole = roi_sum(frame, *mod.WHOLE)          # a line beyond 4095 is cut there
    assert "whole-frame sum: median {:.0f}".format(whole) in out
    assert "Grabber / USB whole frame: {:.2f}".format(whole / 3000.0) in out
    if error is None or "too large" in error:
        assert "Grabber frame: {} x {} stream pixels".format(shape[1], shape[0]) in out


class FreeRunning(StaticCamera):
    """40 frames arrive in the first gate without a trigger."""

    def __init__(self, frame):
        super().__init__(frame)
        self.first = True

    def untriggered(self):
        out = [self.frame] * 40 if self.first else []
        self.first = False
        return out


def test_mapping_check_says_when_the_camera_runs_without_the_trigger(monkeypatch, capsys):
    """More frames without a trigger than the Grabber's input FIFO holds: the
    report says the camera sends frames on its own (no RTIOOverflow crash),
    and the check goes on."""
    t = Timeline()
    mod = load_example(monkeypatch, t)
    exp = make_experiment(mod, t, FreeRunning(np.full((2, 3), 500.0)), StaticIoncam(),
                          frames_per_batch=3)
    exp.prepare()
    exp.run()
    out = capsys.readouterr().out
    assert "more than the Grabber's input FIFO holds - the camera sends frames without " \
           "the trigger" in out
    assert exp.size == (3, 2)
