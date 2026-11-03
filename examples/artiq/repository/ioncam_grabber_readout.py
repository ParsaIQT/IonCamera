"""State detection of an ion chain with the iXon + ARTIQ Grabber.

prepare():  ioncam locates the ions with a long, low-gain exposure, centres the
            camera AQROI on them (aligned to the binning grid), gives every ion
            its own Grabber ROI, verifies the ROIs with the final readout
            settings and arms the camera (external trigger + Camera Link).
kernel:     sets up the Grabber ROIs; every shot opens the ROI gate, triggers
            the camera and reads one sum per ROI (ions in chain order, then a
            background ROI).
analyze():  bright/dark thresholds per ion from the histograms.  With
            ``all_bright`` (every shot prepared bright on purpose) the
            background pixels of the same shots are the dark reference:
            ioncam's quick fidelity estimate per ion.

Wiring: a TTL output -> iXon EXT TRIG input; iXon Camera Link -> Grabber
(base configuration).  After arming, the core device log should report
"grabber0 frame size: WxH" with the Grabber frame printed below.  On a camera
not yet measured (the 897's layout is assumed), run ioncam_grabber_mapping_check
once: it measures how the camera puts its frame on Camera Link.
"""

import numpy as np

from artiq.experiment import *
from artiq.coredevice.grabber import GrabberTimeoutException


class IonCamGrabberReadout(EnvExperiment):
    """ioncam: Grabber readout"""

    def build(self):
        self.setattr_device("core")
        self.setattr_device("ioncam")
        self.setattr_device("grabber0")
        self.setattr_device("ttl_camera_trigger")
        self.setattr_argument("n_ions", NumberValue(2, precision=0, step=1, min=1, max=15))
        self.setattr_argument("n_shots", NumberValue(1000, precision=0, step=100, min=1,
                                                     max=1000000))
        self.setattr_argument("shot_period", NumberValue(10 * ms, unit="ms", min=1 * ms))
        self.setattr_argument("recalibrate", BooleanValue(True))
        # every shot prepared bright: threshold + quick fidelity with the
        # background as the dark reference (otherwise a two-population fit)
        self.setattr_argument("all_bright", BooleanValue(True))

    def prepare(self):
        n = int(self.n_ions)
        if self.recalibrate or not self.ioncam.is_readout_valid():
            plan = self.ioncam.calibrate_readout(n_ions=n)
        else:
            # the plan armed: it follows the crystal on screen if that changed
            plan = self.ioncam.arm_readout()
        for w in plan["warnings"]:
            print("ioncam warning:", w)
        gw, gh = plan.get("grabber_frame", plan["frame_size"])
        print("AQROI (Solis L,R,B,T) {}, binning {}, Grabber frame {}x{} - check the core log "
              "for 'grabber0 frame size: {}x{}'; Camera Link layout {}".format(
                  plan["aqroi_solis"], plan["binning"], gw, gh, gw, gh,
                  plan.get("grabber_source", "")))
        self.n_rois = int(plan["n_rois"])
        self.n_ion_rois = int(plan["n_ions"])
        self.roi_x0 = [int(v) for v in plan["grabber_x0"]]
        self.roi_y0 = [int(v) for v in plan["grabber_y0"]]
        self.roi_x1 = [int(v) for v in plan["grabber_x1"]]
        self.roi_y1 = [int(v) for v in plan["grabber_y1"]]
        self.mask = int(plan["full_mask"])
        applied = plan["applied"]
        # the gate must be open from the trigger until the frame has arrived
        self.gate_time = applied["exposure"] + applied["readout_time"] + 2 * ms
        if self.shot_period < applied["min_trigger_period"] + self.gate_time:
            raise ValueError("shot_period shorter than exposure + readout ({:.2f} ms)".format(
                1e3 * (applied["min_trigger_period"] + self.gate_time)))
        self.n_shots_int = int(self.n_shots)
        self.roi_sums = [0] * self.n_rois
        self.missed = 0
        self.set_dataset("ioncam.roi_sums",
                         np.zeros((self.n_shots_int, self.n_rois), dtype=np.int32),
                         broadcast=True)
        self.kernel_invariants = {"n_rois", "roi_x0", "roi_y0", "roi_x1", "roi_y1", "mask",
                                  "gate_time", "n_shots_int"}

    def run(self):
        owner = "artiq:grabber_readout"
        self.ioncam.take_control(owner, "Grabber readout", 3600)
        try:
            self.run_kernel()
        finally:
            self.ioncam.release_control(owner)

    @kernel
    def run_kernel(self):
        self.core.reset()
        self.grabber0.gate_roi(0)
        for i in range(self.n_rois):
            self.grabber0.setup_roi(i, self.roi_x0[i], self.roi_y0[i],
                                    self.roi_x1[i], self.roi_y1[i])
        delay(50 * ms)      # ROI engines must stay disabled > 1 frame after setup_roi
        for shot in range(self.n_shots_int):
            self.core.break_realtime()
            delay(1 * ms)
            # ---- prepare / manipulate the ions here (cooling, pumping, gates, ...) ----
            with parallel:
                self.grabber0.gate_roi_pulse(self.mask, self.gate_time)
                self.ttl_camera_trigger.pulse(10 * us)
                # switch the detection light on for the exposure here
            try:
                # ARTIQ < 8: drop the timeout argument
                self.grabber0.input_mu(self.roi_sums,
                                       now_mu() + self.core.seconds_to_mu(200 * ms))
            except GrabberTimeoutException:
                self.missed_frame(shot)
                continue
            self.store(shot, self.roi_sums)
            delay(self.shot_period)

    @rpc(flags={"async"})
    def store(self, shot, sums):
        self.mutate_dataset("ioncam.roi_sums", shot, np.array(sums, dtype=np.int32))

    @rpc(flags={"async"})
    def missed_frame(self, shot):
        self.missed += 1
        print("no Grabber frame for shot", shot)

    def analyze(self):
        sums = np.asarray(self.get_dataset("ioncam.roi_sums"))
        if self.missed:
            print("{} shot(s) without a frame (trigger during readout?)".format(self.missed))
        thresholds = []
        if self.all_bright:
            # the frames the camera also sent over USB: ion ROIs bright, the
            # background pixels of the same shots dark
            est = self.ioncam.get_readout_histograms("trigger", self.n_shots_int)
            for r in est["ions"]:
                if r.get("ok"):
                    # (in Grabber counts: a stream that is not binned adds a bias
                    # per stream pixel - the Grabber mapping check measures it)
                    th = r.get("grabber_threshold", r["threshold"])
                    thresholds.append(th)
                    print("ion {}: bright if sum > {}, fidelity ~{:.2%} (bright->dark {:.2%}, "
                          "dark->bright {:.2%}; {:.2%} with the neighbours bright)".format(
                              r["index"], th, r["fidelity"], r["bright_error"],
                              r["dark_error"], r.get("dark_error_neighbours_bright",
                                                     r["dark_error"])))
                else:
                    thresholds.append(float("nan"))
                    print("ion {}: no estimate ({})".format(r["index"], r.get("reason")))
            self.set_dataset("ioncam.thresholds", np.array(thresholds), broadcast=True)
            return
        # mixed bright and dark shots: server-side two-population fits (on the
        # USB copies of the shots; grabber_threshold is in Grabber counts)
        stats = self.ioncam.get_readout_stats(self.n_shots_int)
        for i in range(self.n_ion_rois):
            th = stats["rois"][i].get("threshold", {}) if i < len(stats["rois"]) else {}
            if th.get("ok"):
                t = th.get("grabber_threshold", th["threshold"])
                thresholds.append(t)
                print("ion {}: threshold {:.0f}, est. fidelity {:.4f}, bright {:.1%}".format(
                    i, t, th["fidelity"], float(np.mean(sums[:, i] > t))))
            else:
                thresholds.append(float("nan"))
                print("ion {}: no bright/dark separation ({})".format(i, th.get("reason")))
        self.set_dataset("ioncam.thresholds", np.array(thresholds), broadcast=True)
