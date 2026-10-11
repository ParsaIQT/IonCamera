"""Virtual-time harness: simulated trap + EMCCD -> Detector -> ChainTracker."""

from ioncam.analysis.detection import Detector
from ioncam.analysis.tracker import ChainTracker, adopt_learned_psf
from ioncam.camera.base import Frame
from ioncam.camera.simulator import SimulatedCamera
from ioncam.config import Config


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class Scenario:
    """Drive the simulator frame by frame in virtual time (no sleeping)."""

    def __init__(self, n_ions=3, seed=1, exposure=0.03, em_gain=50, scale=6.0,
                 confirm_time=0.3, **sim_overrides):
        self.cfg = Config()
        self.cfg.simulator.seed = seed
        self.cfg.simulator.n_ions = n_ions
        self.cfg.simulator.length_scale_px = scale
        for k, v in sim_overrides.items():
            setattr(self.cfg.simulator, k, v)
        self.cfg.tracker.confirm_time = confirm_time
        self.clock = Clock()
        self.cam = SimulatedCamera(self.cfg.simulator)
        self.cam.scene.clock = self.clock
        self.scene = self.cam.scene
        self.settings = self.cfg.live.replace(exposure=exposure, em_gain=em_gain,
                                              roi=(160, 100, 360, 240))
        self.applied = self.cam.configure(self.settings, "live")
        self.detector = Detector(self.cfg.detection)
        self.tracker = ChainTracker(self.cfg.tracker, clock=self.clock)
        self.events = []
        self.psf_messages = []
        self.index = 0
        self.period = max(exposure, 0.02)

    def step(self, n=1):
        for _ in range(n):
            self.clock.t += self.period
            self.index += 1
            img = self.cam._render(self.applied, self.clock.t)
            frame = Frame(img, self.index, self.clock.t, self.applied)
            det = self.detector.analyze(frame, prior=self.tracker.prior())
            self.events += self.tracker.update(det, self.clock.t)
            # as the engine does: the PSF measured on the crystal feeds detection
            self.psf_messages.append(adopt_learned_psf(self.tracker, self.cfg.detection))
        return self

    def run(self, seconds):
        return self.step(int(round(seconds / self.period)))

    def kinds(self):
        return [e["kind"] for e in self.events]

    def take(self):
        ev, self.events = self.events, []
        return ev
