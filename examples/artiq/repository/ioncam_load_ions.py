"""Load ions until the camera confirms a stable crystal of N ions.

ioncam decides when an ion is trapped (it ignores hot clouds and noise and
waits until the crystal is stable), so the loading sequence simply switches the
oven / photo-ionisation on, waits for ioncam, and switches everything off.
"""

from artiq.experiment import *


class IonCamLoadIons(EnvExperiment):
    """ioncam: load N ions"""

    def build(self):
        self.setattr_device("core")
        self.setattr_device("scheduler")
        self.setattr_device("ioncam")
        # ARTIQ < 7: use ndecimals=0 instead of precision=0
        self.setattr_argument("n_ions", NumberValue(1, precision=0, step=1, min=1, max=50))
        self.setattr_argument("at_least", BooleanValue(False))
        self.setattr_argument("timeout", NumberValue(300.0, unit="s", min=1.0, max=3600.0))
        # self.setattr_device("ttl_oven")        # your oven / PI-laser shutters
        # self.setattr_device("ttl_pi_shutter")

    def run(self):
        n = int(self.n_ions)
        before = self.ioncam.get_ion_counts()
        print("ioncam sees {} bright / {} dark ions before loading".format(
            before["n_bright"], before["n_dark"]))
        if before["n_bright"] >= n and before["stable"]:
            print("already enough ions")
            return
        owner = "artiq:load_ions"
        self.ioncam.take_control(owner, "loading {} ion(s)".format(n), self.timeout + 60)
        try:
            self.loading(True)
            result = self.wait(n)
        finally:
            self.loading(False)
            self.ioncam.release_control(owner)
        self.set_dataset("ioncam.n_ions", result["n_bright"], broadcast=True)
        if result["ok"]:
            print("loaded: {} ion(s), crystal stable".format(result["n_bright"]))
        else:
            print("loading timed out with {} ion(s) (state {})".format(
                result["n_bright"], result["state"]))

    def wait(self, n):
        # wait in short slices so that the experiment can be paused/terminated
        remaining = float(self.timeout)
        while True:
            r = self.ioncam.wait_for_ions(n, timeout=min(remaining, 5.0),
                                          at_least=bool(self.at_least))
            remaining -= 5.0
            if r["ok"] or remaining <= 0 or self.scheduler.check_termination():
                return r
            if not self.at_least and r["n_bright"] > n:
                print("too many ions ({}) - stop and dump some".format(r["n_bright"]))
                return r

    def loading(self, on):
        """Switch oven / photo-ionisation lasers (adapt to your setup)."""
        # self.loading_kernel(on)
        print("loading", "ON" if on else "OFF")

    # @kernel
    # def loading_kernel(self, on):
    #     self.core.break_realtime()
    #     if on:
    #         self.ttl_oven.on()
    #         self.ttl_pi_shutter.on()
    #     else:
    #         self.ttl_oven.off()
    #         self.ttl_pi_shutter.off()
