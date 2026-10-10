"""Moving ions on purpose (shuttling, changing trap voltages) with ioncam watching.

hold_tracking() tells ioncam that the ions are about to move, so it does not
raise "ion lost" alarms while they travel.  After resume_tracking() it waits
for a stable crystal again and reports how many ions survived ("hold_done"
event).  The second part shows how to use ioncam as a position sensor: scan a
voltage and measure the chain centre/scale precisely at every step.
"""

import numpy as np

from artiq.experiment import *


class IonCamShuttle(EnvExperiment):
    """ioncam: shuttle with hold + position scan"""

    def build(self):
        self.setattr_device("core")
        self.setattr_device("ioncam")
        self.setattr_argument("scan_points", NumberValue(0, precision=0, step=1, min=0, max=100))
        # self.setattr_device("zotino0")    # your trap DACs

    def run(self):
        before = self.ioncam.get_ion_counts()
        last = self.ioncam.get_last_event_id()
        self.ioncam.hold_tracking("shuttling", 60)
        try:
            self.shuttle()                  # your DAC waveform
        finally:
            self.ioncam.resume_tracking()
        done = self.ioncam.wait_for_event(["hold_done"], last, 30)
        if done is None:
            print("crystal did not settle after shuttling")
        else:
            d = done["details"]
            print("shuttling: {} ion(s) before, {} after".format(d["before"], d["after"]))
            if d["after"] < d["before"]:
                print("ION LOST DURING SHUTTLING")
        if int(self.scan_points):
            self.position_scan()

    def shuttle(self):
        """Replace by your transport kernel (e.g. Zotino/Fastino waveforms)."""
        pass

    def position_scan(self):
        """Chain position vs. a control voltage, measured with long exposures."""
        voltages = np.linspace(-0.1, 0.1, int(self.scan_points))
        centres, scales = [], []
        for v in voltages:
            self.set_voltage(v)
            loc = self.ioncam.locate_ions()
            pos = np.array(loc["positions"])
            centres.append(pos.mean(axis=0))
            scales.append(loc["chain"]["scale_px"] if loc["chain"] else np.nan)
            print("V = {:+.3f}: centre {} px, chain scale {} px".format(v, centres[-1],
                                                                         scales[-1]))
        self.set_dataset("ioncam.scan.voltage", voltages, broadcast=True)
        self.set_dataset("ioncam.scan.centre", np.array(centres), broadcast=True)
        self.set_dataset("ioncam.scan.scale", np.array(scales), broadcast=True)

    def set_voltage(self, v):
        """Replace by your DAC update."""
        pass
