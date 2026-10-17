"""One-off check that Grabber ROI coordinates match ioncam's camera coordinates.

Every pixel of the (small) Grabber frame is read with single-pixel ROIs, the
image is rebuilt and ioncam compares it with the frames it receives over USB.
It reports whether the Camera Link stream is mirrored relative to the SDK
image (it should not be: ioncam never uses the SDK's rotate/flip) and the
grabber.flip_x / flip_y settings to use if it is.

Make the image asymmetric for an unambiguous answer: e.g. a single ion, or
one ion of the chain dark (otherwise a symmetric chain looks the same when
mirrored).
"""

import numpy as np

from artiq.experiment import *


class IonCamGrabberMappingCheck(EnvExperiment):
    """ioncam: check Grabber orientation"""

    def build(self):
        self.setattr_device("core")
        self.setattr_device("ioncam")
        self.setattr_device("grabber0")
        self.setattr_device("ttl_camera_trigger")
        self.setattr_argument("frames_per_batch", NumberValue(20, precision=0, step=1,
                                                              min=1, max=1000))
        self.setattr_argument("n_engines", NumberValue(16, precision=0, step=1, min=1, max=32))

    def prepare(self):
        if self.ioncam.get_readout_plan() is None:
            plan = self.ioncam.calibrate_readout()
        else:
            # the plan armed: it follows the crystal on screen if that changed
            plan = self.ioncam.arm_readout()
        self.w, self.h = plan["frame_size"]
        n_pix = self.w * self.h
        k = int(self.n_engines)
        self.batches = []
        for start in range(0, n_pix, k):
            self.batches.append(list(range(start, min(start + k, n_pix))))
        self.n_batches = len(self.batches)
        self.batch_len = [len(b) for b in self.batches]
        self.batch_pix = [b + [0] * (k - len(b)) for b in self.batches]  # padded, rectangular
        self.sums = [0] * k
        self.image = np.zeros(n_pix)
        applied = plan["applied"]
        self.gate_time = applied["exposure"] + applied["readout_time"] + 2 * ms
        self.period = max(5 * ms, applied["min_trigger_period"] + self.gate_time)
        self.n_frames = int(self.frames_per_batch)

    @kernel
    def run(self):
        self.core.reset()
        for b in range(self.n_batches):
            n = self.batch_len[b]
            self.grabber0.gate_roi(0)
            for i in range(n):
                p = self.batch_pix[b][i]
                x = p % self.w
                y = p // self.w
                self.grabber0.setup_roi(i, x, y, x + 1, y + 1)
            delay(20 * ms)
            sums = [0] * n
            for f in range(self.n_frames):
                self.core.break_realtime()
                with parallel:
                    self.grabber0.gate_roi_pulse((1 << n) - 1, self.gate_time)
                    self.ttl_camera_trigger.pulse(10 * us)
                self.grabber0.input_mu(sums)
                self.accumulate(b, sums)
                delay(self.period)

    @rpc(flags={"async"})
    def accumulate(self, b, sums):
        for i, v in enumerate(sums):
            self.image[self.batches[b][i]] += v

    def analyze(self):
        img = self.image.reshape(self.h, self.w) / max(1, self.n_frames)
        self.set_dataset("ioncam.grabber_image", img, broadcast=True)
        result = self.ioncam.check_grabber_mapping(img)
        print("Grabber vs ioncam orientation:", result)
        if result.get("ok") and result["best"] != "identity":
            print("Set in ioncam.toml:  [grabber] flip_x = {}, flip_y = {}  and recalibrate".format(
                str(result["flip_x"]).lower(), str(result["flip_y"]).lower()))
