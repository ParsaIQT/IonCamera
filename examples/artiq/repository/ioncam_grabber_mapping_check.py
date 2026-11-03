"""Measure how the camera puts its frame on Camera Link, so the Grabber ROIs are right.

The Grabber counts Camera Link pixels: x the valid pixels of a line, y the lines
of a frame.  How the camera's binned frame (what ioncam sees over USB) sits in
that stream depends on the camera: an iXon Ultra 897 sends one pixel per
super-pixel, another camera may send its lines unbinned, start them with
pre-scan pixels, or run them the other way.  With the readout armed and the
ions shining - one ion dark or a single ion off-centre, so the picture is not
mirror-symmetric - this experiment

1. finds the Grabber's frame size: a single-column (single-row) ROI sums the
   camera's bias inside the frame and nothing beyond it;
2. reads every pixel of that frame with single-pixel ROIs (n_engines at a
   time, frames_per_batch frames each);
3. hands the picture to ioncam, which compares it with the frames it receives
   over USB, finds the layout - stream pixels per super-pixel, offset,
   mirroring - and, if the answer is unambiguous, keeps it for this camera and
   maps the plan's Grabber ROIs with it (thresholds too: ``grabber_threshold``).

Run it once per camera, and again after changing the binning.  A small AQROI
with some structure works best (e.g. 4 x 4 super-pixels with one ion off-centre;
at most 64 x 64 stream pixels are read).  The core device log's
"grabber0 frame size: WxH" should agree with the size found here.
"""

import numpy as np

from artiq.experiment import *


class IonCamGrabberMappingCheck(EnvExperiment):
    """ioncam: measure the Grabber mapping (Camera Link layout)"""

    def build(self):
        self.setattr_device("core")
        self.setattr_device("ioncam")
        self.setattr_device("grabber0")
        self.setattr_device("ttl_camera_trigger")
        self.setattr_argument("frames_per_batch", NumberValue(20, precision=0, step=1,
                                                              min=1, max=1000))
        self.setattr_argument("n_engines", NumberValue(16, precision=0, step=1, min=1, max=16))
        self.setattr_argument("apply", BooleanValue(True))

    def prepare(self):
        if self.ioncam.get_readout_plan() is None:
            plan = self.ioncam.calibrate_readout()
        else:
            # the plan armed: it follows the crystal on screen if that changed
            plan = self.ioncam.arm_readout()
        self.plan_frame = plan["frame_size"]
        applied = plan["applied"]
        # the gate must be open from the trigger until the frame has arrived
        self.gate_time = applied["exposure"] + applied["readout_time"] + 2 * ms
        self.period = max(5 * ms, applied["min_trigger_period"] + self.gate_time)
        self.n_frames = int(self.frames_per_batch)
        self.k = int(self.n_engines)
        self.n_batches = 0
        self.batch_len = [0]
        self.rois = [0, 0, 1, 1]
        self.acc = np.zeros((1, self.k))

    def run(self):
        owner = "artiq:grabber_mapping_check"
        self.ioncam.take_control(owner, "Grabber mapping check", 900)
        try:
            w = self.extent(lambda x: (x, 0, x + 1, 4095))
            h = self.extent(lambda y: (0, y, w, y + 1))
            print("Grabber frame: {} x {} stream pixels (ioncam's frame: {} x {} "
                  "super-pixels)".format(w, h, *self.plan_frame))
            if w * h > 64 * 64:
                raise ValueError("the Grabber frame is too large to read pixel by pixel: "
                                 "make the AQROI smaller")
            px = self.sums_of([(x, y, x + 1, y + 1) for y in range(h) for x in range(w)])
            self.image = np.array(px, float).reshape(h, w)
        finally:
            self.ioncam.release_control(owner)

    def sums_of(self, rois):
        """The mean Grabber sum of every ROI (x0, y0, x1, y1), n_engines at a time."""
        k = self.k
        batches = [rois[i:i + k] for i in range(0, len(rois), k)]
        self.n_batches = len(batches)
        self.batch_len = [len(b) for b in batches]
        flat = []
        for b in batches:
            for r in list(b) + [(0, 0, 1, 1)] * (k - len(b)):
                flat += [int(v) for v in r]
        self.rois = flat
        self.acc = np.zeros((len(batches), k))
        self.measure()
        return [float(self.acc[i // k, i % k]) / self.n_frames for i in range(len(rois))]

    def extent(self, roi):
        """The frame's width (or height): the first column (row) whose ROI sums
        nothing - coarse to fine."""
        cand = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512][:self.k]
        inside, outside = -1, None
        while outside is None or outside > inside + 1:
            for c, v in zip(cand, self.sums_of([roi(c) for c in cand])):
                if v > 0:
                    inside = max(inside, c)
                elif outside is None or c < outside:
                    outside = c
            if inside < 0:
                raise ValueError("the Grabber sums nothing: is the camera armed, Camera Link "
                                 "connected and the trigger wired?")
            if outside is None:
                raise ValueError("the Grabber frame is larger than {} pixels: make the AQROI "
                                 "smaller".format(cand[-1]))
            step = max(1, (outside - inside) // (self.k + 1))
            cand = list(range(inside + step, outside, step))[:self.k] or [inside + 1]
        return outside

    @kernel
    def measure(self):
        self.core.reset()
        for b in range(self.n_batches):
            n = self.batch_len[b]
            self.grabber0.gate_roi(0)
            for i in range(n):
                j = 4 * (b * self.k + i)
                self.grabber0.setup_roi(i, self.rois[j], self.rois[j + 1], self.rois[j + 2],
                                        self.rois[j + 3])
            delay(50 * ms)      # ROI engines must stay disabled > 1 frame after setup_roi
            sums = [0] * n
            for f in range(self.n_frames):
                self.core.break_realtime()
                with parallel:
                    self.grabber0.gate_roi_pulse((1 << n) - 1, self.gate_time)
                    self.ttl_camera_trigger.pulse(10 * us)
                # ARTIQ < 8: drop the timeout argument
                self.grabber0.input_mu(sums, now_mu() + self.core.seconds_to_mu(500 * ms))
                self.accumulate(b, sums)
                delay(self.period)

    @rpc(flags={"async"})
    def accumulate(self, b, sums):
        self.acc[b, :len(sums)] += np.asarray(sums, float)

    def analyze(self):
        self.set_dataset("ioncam.grabber_image", self.image, broadcast=True)
        res = self.ioncam.check_grabber_mapping(self.image.tolist(), bool(self.apply))
        print("Camera Link layout of {}: {}".format(res.get("camera"), res.get("summary")))
        print("  fit r = {} (other mirroring {}), Grabber sum = {} x USB sum + {} per "
              "super-pixel".format(res.get("r"), res.get("r_other_mirroring"), res.get("gain"),
                                   res.get("offset")))
        if not res.get("ok"):
            print("  NOT applied: " + str(res.get("reason")))
        elif res.get("applied"):
            p = res["plan"]
            print("  applied: Grabber ROIs (ions, then background) now {}".format(
                p["grabber_rois"]))
            print("  ROI_X0 = {}\n  ROI_Y0 = {}\n  ROI_X1 = {}\n  ROI_Y1 = {}".format(
                p["grabber_x0"], p["grabber_y0"], p["grabber_x1"], p["grabber_y1"]))
