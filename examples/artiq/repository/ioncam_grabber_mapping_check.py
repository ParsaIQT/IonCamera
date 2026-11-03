"""Measure how the camera puts its frame on Camera Link, so the Grabber ROIs are right.

The Grabber counts Camera Link pixels: x the valid pixels of a line, y the lines
of a frame.  How the camera's binned frame (what ioncam sees over USB) sits in
that stream depends on the camera: an iXon Ultra 897 sends one pixel per
super-pixel, another camera may send its lines unbinned, start them with
pre-scan pixels, or run them the other way.  With the readout armed and the
ions shining - one ion dark or a single ion off-centre, so the picture is not
mirror-symmetric - this experiment

1. counts the frames that reach the Grabber with the gate open and no trigger
   (there should be none);
2. finds the Grabber's frame size: a ROI on one pixel of the first line (on
   one row) sums the camera's bias inside the frame and nothing beyond it - up
   to the Grabber's limit of 4096 x 4096.  (Not a column ROI: the gateware
   switches a ROI on when the pixel counter equals x0, and after a line's last
   pixel the counter stands at the line's length W for a cycle - a ROI at
   x0 = W stays on into the next lines.);
3. reads every pixel of that frame with single-pixel ROIs (n_engines - 1 at a
   time, averaged over frames_per_pixel frames) and, after each batch, fetches
   from ioncam the USB frames of the same shots - an ion's brightness changes
   from batch to batch, so each pixel is compared with its own shots;
4. hands the picture to ioncam, which compares it with the frames it receives
   over USB, finds the layout - stream pixels per super-pixel, offset,
   mirroring - and, if the answer is unambiguous, keeps it for this camera and
   maps the plan's Grabber ROIs with it (thresholds too: ``grabber_threshold``).

ROI engine 0 sums the whole frame in every measurement.  A frame whose whole sum
is more than 3x off the median is not one of the AQROI's frames (a frame with
no trigger, a partial one): it is counted and left out, and a column or row is
lit only if it is lit in most frames - one odd frame cannot fake a large frame.
The report, printed also when the measurement fails, says how many frames came
per trigger, how many were odd, the whole-frame sum next to ioncam's USB
frames, and each probe's sums.

Run it once per camera, and again after changing the binning.  A small AQROI
with some structure works best (e.g. 4 x 4 super-pixels with one ion
off-centre; at most 64 x 64 stream pixels are read).  The core device log
(``artiq_coremgmt log``) reports "grabber0 frame size: WxH" whenever the
Grabber sees a new frame size: it should agree with the size found here.

Needs ARTIQ 8 or later (``input_mu`` with a timeout, ``GrabberTimeoutException``).
"""

import socket
import time

import numpy as np

from artiq.experiment import *
from artiq.coredevice.exceptions import RTIOOverflow
from artiq.coredevice.grabber import GrabberTimeoutException


WHOLE = (0, 0, 4095, 4095)      # every pixel of a frame: the Grabber's x and y are 12-bit
SIZES = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 256, 1024, 4095]
MAX_PIXELS = 64 * 64            # read pixel by pixel at most
ODD = 3.0                       # a whole-frame sum this far off the median: an odd frame


def get_ioncam(exp, timeout=5.0):
    """The "ioncam" controller, once it answers - call it in prepare().

    ARTIQ connects a controller requested in build() with no time limit of its
    own: when it cannot reach ioncam, the experiment dies after 15 s with
    "Timeout receiving data from worker" and no word of why."""
    db = exp.get_device_db()
    desc = db["ioncam"]
    while isinstance(desc, str):            # an alias
        desc = db[desc]
    host, port = desc["host"], desc["port"]
    try:
        socket.create_connection((host, port), timeout).close()
    except OSError as e:
        raise ConnectionError(
            "ioncam does not answer at {}:{} ({}). On the camera PC: is ioncam_gui (or "
            "aqctl_ioncam) running, started with --bind '*' (otherwise only that PC can "
            "connect), and does its firewall let TCP port {} in? Is {} that PC's address? "
            "Test from here: sipyco_rpctool {} {} list-targets".format(
                host, port, e, port, host, host, port)) from None
    return exp.get_device("ioncam")


class MappingCheckError(Exception):
    pass


class IonCamGrabberMappingCheck(EnvExperiment):
    """ioncam: measure the Grabber mapping (Camera Link layout)"""

    def build(self):
        self.setattr_device("core")
        self.setattr_device("grabber0")
        self.setattr_device("ttl_camera_trigger")
        # frames per measurement: finding the frame size, reading its pixels
        self.setattr_argument("frames_per_batch", NumberValue(20, precision=0, step=1,
                                                              min=1, max=1000))
        self.setattr_argument("frames_per_pixel", NumberValue(30, precision=0, step=10,
                                                              min=1, max=10000))
        # engine 0 sums the whole frame; the others probe
        self.setattr_argument("n_engines", NumberValue(16, precision=0, step=1, min=2, max=16))
        self.setattr_argument("apply", BooleanValue(True))

    def prepare(self):
        self.ioncam = get_ioncam(self)
        if self.ioncam.get_readout_plan() is None:
            plan = self.ioncam.calibrate_readout()
        else:
            # the plan armed: it follows the crystal on screen if that changed
            plan = self.ioncam.arm_readout()
        self.plan_frame = plan["frame_size"]
        self.binning = plan["binning"]
        applied = plan["applied"]
        # the gate is open from the trigger until the frame has arrived: the
        # camera may first finish a keep-clean cycle, then exposes and reads out
        self.gate_time = applied["min_trigger_period"] + 2 * ms
        self.period = max(10 * ms, applied["min_trigger_period"] + self.gate_time)
        # after the gate has closed, a frame's sums are in the input FIFO
        self.wait_mu = self.core.seconds_to_mu(5 * ms)
        self.idle_time = 200 * ms
        self.n_frames = int(self.frames_per_batch)
        self.usb = None
        self.match_usb = False      # fetch each batch's USB frames (the pixel read)
        self.usb_means = []         # the USB mean frame of each batch's shots
        self.usb_frames = []        # each batch's USB frames, one per trigger
        self.usb_count = 0          # ioncam's shot count after the last batch
        self.k = int(self.n_engines)
        self.n_batches = 0
        self.batch_len = [0]
        self.rois = [0, 0, 1, 1]
        self.image = None
        self.size = None
        self.notes = []
        self.wholes = []            # whole-frame sum of every frame read first in its gate
        self.odd_wholes = []        # ... of the odd ones (the first few)
        self.extra_wholes = []      # frames after the first in the same gate
        self.idle_wholes = []       # frames with the gate open and no trigger (-1: overflow)
        self.n_gates = 0
        self.n_got = 0
        self.n_odd = 0

    def run(self):
        owner = "artiq:grabber_mapping_check"
        self.ioncam.take_control(owner, "Grabber mapping check", 900)
        try:
            self.count_idle_frames()
            # the width on the first line only: a ROI whose x0 is the line's
            # length W is switched on by the cycle after the line's last pixel
            # and stays on into the next lines - a column ROI at x = W is lit
            w = self.extent("columns", lambda x: (x, 0, x + 1, 1))
            h = self.extent("rows", lambda y: (0, y, min(w, 4095), y + 1))
            self.size = (w, h)
            if w * h > MAX_PIXELS:
                raise MappingCheckError(
                    "the Grabber frame is {} x {} stream pixels: too large to read pixel by "
                    "pixel (at most {}) - ioncam's frame is {} x {} super-pixels; see the "
                    "report above".format(w, h, MAX_PIXELS, *self.plan_frame))
            self.n_frames = int(self.frames_per_pixel)
            self.match_usb = True
            px = self.sums_of([(x, y, x + 1, y + 1) for y in range(h) for x in range(w)])
            self.match_usb = False
            self.image = np.array([m for m, _ in px], float).reshape(h, w)
        finally:
            self.ioncam.release_control(owner)
            self.report()

    def extent(self, what, roi):
        """The frame's width (height) in stream pixels: the first column (row)
        after the last lit one - coarse to fine.  Lit: the median sum over the
        AQROI's frames is above 0."""
        lit = {}
        cand = list(SIZES)
        while True:
            res = self.sums_of([roi(c) for c in cand])
            if not lit:
                self.notes.append("{} probed (mean sum over the AQROI's frames): {}".format(
                    what, ", ".join("{}: {:.0f}".format(c, m) for c, (m, _) in zip(cand, res))))
            for c, (_, median) in zip(cand, res):
                lit[c] = median > 0
            on = [c for c in lit if lit[c]]
            if not on:
                raise MappingCheckError(
                    "the Grabber sums nothing in its {}: is the camera armed, Camera Link "
                    "connected and the trigger wired?".format(what))
            inside = max(on)
            off = [c for c in lit if not lit[c] and c > inside]
            if not off:
                raise MappingCheckError(
                    "the Grabber's {} are lit up to 4095: the camera's frames on Camera Link "
                    "are at least 4096 pixels {} (the Grabber's counters wrap), not ioncam's "
                    "{} x {} super-pixels - see the report above".format(
                        what, "wide" if what == "columns" else "high", *self.plan_frame))
            outside = min(off)
            if outside == inside + 1:
                break
            step = max(1, (outside - inside) // (len(SIZES) + 1))
            cand = list(range(inside + step, outside, step))[:len(SIZES)] or [inside + 1]
        gaps = sorted(c for c in lit if not lit[c] and c < inside)
        if gaps:
            self.notes.append("{} without light inside the frame: {}".format(what, gaps))
        return outside

    def sums_of(self, rois):
        """Every ROI's (mean, median) sum over the AQROI's frames, n_engines - 1
        ROIs at a time beside the whole frame (engine 0)."""
        k = self.k - 1
        batches = [rois[i:i + k] for i in range(0, len(rois), k)]
        flat = []
        for b in batches:
            for r in [WHOLE] + list(b) + [(0, 0, 1, 1)] * (k - len(b)):
                flat += [int(v) for v in r]
        self.rois = flat
        self.n_batches = len(batches)
        self.batch_len = [1 + len(b) for b in batches]
        self.got = np.zeros((self.n_batches, self.n_frames), bool)
        self.data = np.zeros((self.n_batches, self.n_frames, self.k))
        if self.match_usb:
            self.usb_means = [None] * self.n_batches
            self.usb_frames = [None] * self.n_batches
            try:
                self.usb_count = self.ioncam.get_readout_frames(1)["shots"]
            except Exception as e:      # an ioncam without get_readout_frames
                self.notes.append("USB frames not matched to the batches: {}".format(e))
                self.match_usb = False
        self.measure()
        normal = self.sort_frames()
        if self.match_usb:
            # the USB frames of exactly the shots kept (the newest 200 at most)
            for b, fr in enumerate(self.usb_frames):
                if fr is None:
                    continue
                keep = normal[b][self.n_frames - len(fr):]
                if keep.any():
                    self.usb_means[b] = np.asarray(fr, float)[keep].mean(axis=0).tolist()
        out = []
        for i in range(len(rois)):
            b, j = divmod(i, k)
            v = self.data[b, normal[b], j + 1]
            if not len(v):
                raise MappingCheckError("none of {} triggers gave one of the AQROI's frames - "
                                        "see the report above".format(self.n_frames))
            out.append((float(np.mean(v)), float(np.median(v))))
        return out

    def sort_frames(self):
        """Which frames of the last measurement are the AQROI's: whole-frame
        sum within ODD x of the median of all frames so far.  Counted for the
        report."""
        whole = self.data[:, :, 0]
        self.n_gates += self.got.size
        self.n_got += int(self.got.sum())
        self.wholes += [float(v) for v in whole[self.got]]
        ref = float(np.median(self.wholes)) if self.wholes else 0.0
        normal = self.got.copy()
        if ref > 0:
            normal &= (whole >= ref / ODD) & (whole <= ref * ODD)
        odd = self.got & ~normal
        self.n_odd += int(odd.sum())
        self.odd_wholes += [float(v) for v in whole[odd]][:max(0, 8 - len(self.odd_wholes))]
        return normal

    def report(self):
        """What reached the Grabber - printed also when the measurement fails."""
        try:
            ref = float(np.median(self.wholes)) if self.wholes else 0.0
            idle = [v for v in self.idle_wholes if v >= 0]
            lines = ["Grabber: {} triggers, a frame within the gate for {}{}; another frame in "
                     "the same gate {} times{}".format(
                         self.n_gates, self.n_got,
                         "" if self.n_got == self.n_gates else
                         " (no frame for {})".format(self.n_gates - self.n_got),
                         len(self.extra_wholes), _values(self.extra_wholes)),
                     "  whole-frame sum: median {:.0f}; odd frames (more than {:g}x off): "
                     "{}{}".format(ref, ODD, self.n_odd, _values(self.odd_wholes)),
                     "  frames with the gate open {:.0f} ms and no trigger: {}{}".format(
                         self.idle_time / ms, len(idle) if -1 not in self.idle_wholes else
                         "more than the Grabber's input FIFO holds - the camera sends frames "
                         "without the trigger", _values(idle))]
            try:
                usb = self.ioncam.get_readout_frames(200)
            except Exception as e:      # an ioncam without get_readout_frames
                usb = {"n": 0}
                lines.append("  ioncam's USB frames: not available ({})".format(e))
            self.usb = usb
            if usb.get("n"):
                h, w = usb["shape"]
                u = float(np.median(usb["frame_sums"]))
                lines.append("  ioncam over USB: {} frames of {} x {} super-pixels ({} x {} "
                             "binning), frame sum median {:.0f}".format(
                                 usb["n"], w, h, self.binning[0], self.binning[1], u))
                if u > 0 and ref > 0:
                    lines.append("  Grabber / USB whole frame: {:.2f} (1: the stream is the "
                                 "binned frame; {}: not binned at all; much more: the stream "
                                 "carries more than the AQROI)".format(
                                     ref / u, self.binning[0] * self.binning[1]))
            if self.size is not None:
                lines.append("  Grabber frame: {} x {} stream pixels; ioncam's frame: {} x {} "
                             "super-pixels".format(self.size[0], self.size[1],
                                                   *self.plan_frame))
            lines += ["  " + n for n in self.notes]
            if self.image is not None and self.image.shape[1] <= 32 and self.image.shape[0] <= 32:
                lines.append("  Grabber frame, pixel by pixel (mean over the AQROI's frames; "
                             "line 0 first):")
                lines += ["    " + " ".join("{:6.0f}".format(v) for v in row)
                          for row in self.image]
                if usb.get("n") and usb.get("mean") is not None:
                    lines.append("  ioncam's USB frame (mean; SDK order, row 0 first):")
                    lines += ["    " + " ".join("{:6.0f}".format(v) for v in row)
                              for row in usb["mean"]]
            lines.append("  The core device log (artiq_coremgmt log) names each frame size the "
                         "Grabber sees: \"grabber0 frame size: WxH\".")
            for line in lines:
                print(line)
        except Exception as e:      # never hide the measurement's own error
            print("Grabber report failed: {}".format(e))

    @kernel
    def count_idle_frames(self):
        """Frames that reach the Grabber with the gate open and no trigger."""
        self.core.reset()
        self.grabber0.gate_roi(0)
        self.grabber0.setup_roi(0, 0, 0, 4095, 4095)
        delay(50 * ms)      # ROI engines must stay disabled > 1 frame after setup_roi
        whole = [0]
        self.grabber0.gate_roi_pulse(1, self.idle_time)
        end = now_mu()
        more = True
        while more:
            try:
                self.grabber0.input_mu(whole, end + self.wait_mu)
                self.record_idle(whole[0])
            except GrabberTimeoutException:
                more = False
            except RTIOOverflow:
                self.record_idle(-1)    # more frames than the input FIFO holds
                more = False

    @kernel
    def measure(self):
        self.core.reset()
        for b in range(self.n_batches):
            self.core.break_realtime()
            n = self.batch_len[b]
            self.grabber0.gate_roi(0)
            for i in range(n):
                j = 4 * (b * self.k + i)
                self.grabber0.setup_roi(i, self.rois[j], self.rois[j + 1], self.rois[j + 2],
                                        self.rois[j + 3])
            delay(50 * ms)      # ROI engines must stay disabled > 1 frame after setup_roi
            sums = [0] * n
            spare = [0] * n
            for f in range(self.n_frames):
                self.core.break_realtime()
                with parallel:
                    self.grabber0.gate_roi_pulse((1 << n) - 1, self.gate_time)
                    self.ttl_camera_trigger.pulse(10 * us)
                end = now_mu()
                got = True
                try:
                    self.grabber0.input_mu(sums, end + self.wait_mu)
                except GrabberTimeoutException:
                    got = False
                more = got
                while more:     # any further frame that ended in this gate
                    try:
                        self.grabber0.input_mu(spare, end + self.wait_mu)
                        self.record_extra(spare[0])
                    except GrabberTimeoutException:
                        more = False
                self.record(b, f, got, sums)
                delay(self.period)
            self.batch_done(b)

    def batch_done(self, b) -> TNone:
        """After batch b (pixel read): the USB frames of its shots - each
        trigger is one shot ioncam receives (the newest 200 are kept)."""
        if not self.match_usb:
            return
        want = self.usb_count + self.n_frames
        t0 = time.time()
        while True:
            usb = self.ioncam.get_readout_frames(1)
            if usb["shots"] >= want or time.time() - t0 > 3.0:
                break
            time.sleep(0.01)
        if usb["shots"] == want:
            usb = self.ioncam.get_readout_frames(min(self.n_frames, 200), True)
            if usb["shots"] == want:
                self.usb_frames[b] = usb["frames"]
        self.usb_count = usb["shots"]

    @rpc(flags={"async"})
    def record(self, b, f, got, sums):
        if got:
            self.got[b, f] = True
            self.data[b, f, :len(sums)] = sums

    @rpc(flags={"async"})
    def record_extra(self, whole):
        self.extra_wholes.append(whole)

    @rpc(flags={"async"})
    def record_idle(self, whole):
        self.idle_wholes.append(whole)

    def analyze(self):
        if self.image is None:
            return
        self.set_dataset("ioncam.grabber_image", self.image, broadcast=True)
        h, w = self.image.shape
        if self.usb_means and all(m is not None for m in self.usb_means):
            # each pixel against the USB frames of its own batch of shots
            batch = [[(y * w + x) // (self.k - 1) for x in range(w)] for y in range(h)]
            res = self.ioncam.check_grabber_mapping(self.image.tolist(), bool(self.apply),
                                                    self.usb_means, batch)
        else:
            print("  (the USB frames of some batches were not received: compared with "
                  "ioncam's recent frames)")
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


def _values(v, n=4):
    """A few of the whole-frame sums, for the report."""
    return " ({}{})".format(", ".join("{:.0f}".format(x) for x in v[:n]),
                            ", ..." if len(v) > n else "") if v else ""
