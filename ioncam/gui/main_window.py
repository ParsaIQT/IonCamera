"""The ioncam main window."""

import time

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ioncam.gui.widgets import (EventList, HistogramPlot, ImagePanel, SiteTable,
                                StatusBanner, TracePlot)


def _button(text, slot, tip=None):
    b = QtWidgets.QPushButton(text)
    b.clicked.connect(slot)
    if tip:
        b.setToolTip(tip)
    return b


def _group(title, widgets):
    g = QtWidgets.QGroupBox(title)
    lay = QtWidgets.QVBoxLayout(g)
    for w in widgets:
        if isinstance(w, QtWidgets.QLayout):
            lay.addLayout(w)
        else:
            lay.addWidget(w)
    return g


def _row(*widgets):
    lay = QtWidgets.QHBoxLayout()
    for w in widgets:
        if isinstance(w, str):
            lay.addWidget(QtWidgets.QLabel(w))
        else:
            lay.addWidget(w)
    return lay


def format_plan(plan):
    if not plan:
        return "No readout plan yet.\n\nPress 'Calibrate + arm' (or call calibrate_readout() " \
               "from ARTIQ) with the cooling light on."
    w, h = plan["frame_size"]
    lines = []
    lines.append("Readout plan  {}   {}".format(
        time.strftime("%H:%M:%S", time.localtime(plan["created"])),
        "VALID" if plan.get("valid", True) else "OUTDATED - crystal changed, recalibrate"))
    lines.append("ions {}   binning {}x{}   Grabber frame {} x {} (W x H)".format(
        plan["n_ions"], plan["binning"][0], plan["binning"][1], w, h))
    lines.append("  -> core device log should say: grabber0 frame size: {}x{}".format(w, h))
    lines.append("AQROI (Solis Left, Right, Bottom, Top): {}".format(
        ", ".join(str(v) for v in plan["aqroi_solis"])))
    lines.append("SDK SetImage{}".format(tuple(plan["sdk_image"])))
    lines.append("")
    lines.append(" ROI    x0  y0  x1  y1   contain  crosstalk   ion - ROI centre [px]")
    ax0, ay0 = plan["aqroi"][:2]
    hb, vb = plan["binning"]
    for i, r in enumerate(plan["grabber_rois"]):
        if i < plan["n_ions"]:
            u0, v0, u1, v1 = plan["ion_rois"][i]
            px, py = plan["positions"][i]
            dx = px - (ax0 + 0.5 * (u0 + u1) * hb - 0.5)
            dy = py - (ay0 + 0.5 * (v0 + v1) * vb - 0.5)
            lines.append(" ion{:<3d}{:3d} {:3d} {:3d} {:3d}   {:6.1%}   {:6.1%}     {:+5.1f} {:+5.1f}{}".format(
                i, *r, plan["containment"][i], plan["crosstalk"][i], dx, dy,
                "  (dark)" if plan["ion_states"][i] == "dark" else ""))
        else:
            lines.append(" bg    {:3d} {:3d} {:3d} {:3d}   ({:.2%} of an ion's light)".format(
                *r, plan["background_contamination"]))
    lines.append("masks: ions {}  all {}".format(bin(plan["ion_mask"]), bin(plan["full_mask"])))
    lines.append("(ROI edges lie on the {}x{} binning grid - dotted lines - so ions are only as "
                 "centred as the grid allows; the layout maximises the light in each ROI)".format(
                     hb, vb))
    v = plan.get("verification")
    if v:
        lines.append("")
        ions_v = v.get("ions", [])
        lines.append("verification: {}  ({} frames)".format("OK" if v.get("ok") else "FAILED",
                                                             v.get("n_frames", "?")))
        if ions_v:
            lines.append("  single-shot SNR   {}".format(
                [i.get("snr_single_shot", i["snr"]) for i in ions_v]))
            if "discrimination_error" in ions_v[0]:
                lines.append("  est. bright/dark error per shot {}".format(
                    ["{:.1%}".format(i["discrimination_error"]) for i in ions_v]))
        for p in v.get("problems", []):
            lines.append("  ! " + p)
    a = plan.get("applied")
    if a:
        s = a["settings"]
        lines.append("")
        lines.append("camera: exposure {:.4g} s, EM gain {}, {} MHz, VS {} us, pre-amp {}x".format(
            a["exposure"], a["em_gain"], a["hs_speed_mhz"], a["vs_speed_us"], a["preamp_gain"]))
        lines.append("        trigger {}, {}, Camera Link {}, frame transfer {}".format(
            s["trigger"], s["acquisition"], "on" if s["camera_link"] else "off",
            "on" if s["frame_transfer"] else "off"))
        lines.append("        readout {:.3g} ms -> minimum trigger period {:.3g} ms".format(
            1e3 * a["readout_time"], 1e3 * a["min_trigger_period"]))
    adv = plan.get("binning_advice")
    if adv:
        lines.append("")
        lines.append("binning advice (min containment / max crosstalk / frame):")
        for b in adv:
            if "min_containment" in b:
                lines.append("  {}x{}: {:5.1%} / {:5.1%} / {}x{} {}".format(
                    b["binning"][0], b["binning"][1], b["min_containment"], b["max_crosstalk"],
                    b["frame_size"][0], b["frame_size"][1], "ok" if b["ok"] else ""))
    if plan.get("warnings"):
        lines.append("")
        for wtxt in plan["warnings"]:
            lines.append("warning: " + wtxt)
    return "\n".join(lines)


def artiq_snippet(plan):
    if not plan:
        return ""
    g = plan["grabber_rois"]
    return ("# ioncam readout plan, AQROI (Solis L,R,B,T) = {aq}, binning {b}\n"
            "N_IONS = {n}\n"
            "ROI_X0 = {x0}\nROI_Y0 = {y0}\nROI_X1 = {x1}\nROI_Y1 = {y1}\n"
            "ION_MASK = {im}\nFULL_MASK = {fm}\n").format(
        aq=plan["aqroi_solis"], b="x".join(str(v) for v in plan["binning"]), n=plan["n_ions"],
        x0=[r[0] for r in g], y0=[r[1] for r in g], x1=[r[2] for r in g], y1=[r[3] for r in g],
        im=bin(plan["ion_mask"]), fm=bin(plan["full_mask"]))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, backend, config, info):
        super().__init__()
        self.backend = backend
        self.config = config            # dict of sections (from get_config)
        self.info = info
        self.simulated = bool(info.get("simulated"))
        backend.on_error = self._error
        disp = _Display(config.get("display", {}))
        self.setWindowTitle("ioncam - {} ({})".format(info.get("model", "camera"),
                                                       backend.describe()))
        self.resize(1500, 900)

        self.banner = StatusBanner()
        self.image = ImagePanel(disp)
        self.image.hovered.connect(lambda s: self.hover_label.setText(s))
        self.trace = TracePlot()
        self.events = EventList()

        left = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        left.addWidget(self.image)
        left.addWidget(self.trace)
        left.setSizes([650, 200])
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._live_tab(disp), "Live")
        self.tabs.addTab(self._readout_tab(), "Readout")
        self.tabs.addTab(self._ions_tab(), "Ions")
        self.tabs.addTab(self.events, "Events")
        self.tabs.addTab(self._camera_tab(), "Camera")
        if self.simulated:
            self.tabs.addTab(self._sim_tab(), "Simulation")
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(self.tabs)
        split.setSizes([1000, 480])
        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self.banner)
        lay.addWidget(split, 1)
        self.setCentralWidget(central)
        self.hover_label = QtWidgets.QLabel("")
        self.msg_label = QtWidgets.QLabel("")
        self.statusBar().addWidget(self.hover_label, 1)
        self.statusBar().addPermanentWidget(self.msg_label)

        self._frame_index = 0
        self._last_event = 0
        self._plan_created = None
        self._plan = None
        self._polling = False
        self._status = {}
        self._tick = 0
        self._stats = None
        self._remote_bin = 1
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(int(1000 / max(1.0, disp.max_fps)))
        self.slow = QtCore.QTimer(self)
        self.slow.timeout.connect(self._slow_tick)
        self.slow.start(500)

    # -------------------------------------------------------------- tabs
    def _live_tab(self, disp):
        live = self.config.get("live", {})
        self.exp = QtWidgets.QDoubleSpinBox()
        self.exp.setDecimals(4)
        self.exp.setRange(1e-4, 60.0)
        self.exp.setSingleStep(0.01)
        self.exp.setSuffix(" s")
        self.exp.setValue(live.get("exposure", 0.05))
        self.exp.editingFinished.connect(lambda: self._cmd("set_live", exposure=self.exp.value()))
        self.gain = QtWidgets.QSpinBox()
        self.gain.setRange(0, 1000)
        self.gain.setValue(live.get("em_gain", 50))
        self.gain.editingFinished.connect(lambda: self._cmd("set_live", em_gain=self.gain.value()))
        self.binning = QtWidgets.QComboBox()
        self.binning.addItems(["1", "2", "4", "8"])
        self.binning.setCurrentText(str(live.get("hbin", 1)))
        self.binning.activated.connect(lambda _: self._cmd(
            "set_live", hbin=int(self.binning.currentText()), vbin=int(self.binning.currentText())))
        acq = _group("Live video", [
            _row(_button("Start live", lambda: self._cmd("start_live")),
                 _button("Stop", lambda: self._cmd("stop"))),
            _row("Exposure", self.exp, "EM gain", self.gain),
            _row("Binning", self.binning,
                 _button("Crop to view", self._crop_view, "acquire only the visible area"),
                 _button("Full sensor", lambda: self._cmd("reset_roi"))),
            _row(_button("Crop around ions", self._crop_ions, "acquire 40 px around the ions")),
        ])
        self.auto_levels = QtWidgets.QCheckBox("auto levels")
        self.auto_levels.setChecked(True)
        self.auto_levels.toggled.connect(lambda v: setattr(self.image, "auto_levels", v))
        self.overlays = QtWidgets.QCheckBox("overlays")
        self.overlays.setChecked(True)
        self.overlays.toggled.connect(self._toggle_overlays)
        self.rotation = QtWidgets.QComboBox()
        self.rotation.addItems(["0", "90", "180", "270"])
        self.rotation.setCurrentText(str(disp.rotation))
        self.flipx = QtWidgets.QCheckBox("flip x")
        self.flipx.setChecked(disp.flip_x)
        self.flipy = QtWidgets.QCheckBox("flip y")
        self.flipy.setChecked(disp.flip_y)
        for w in (self.rotation,):
            w.activated.connect(lambda _: self._orientation())
        for w in (self.flipx, self.flipy):
            w.toggled.connect(lambda _: self._orientation())
        self.cmap = QtWidgets.QComboBox()
        self.cmap.addItems(["inferno", "viridis", "magma", "plasma", "CET-L1", "CET-L8"])
        self.cmap.setCurrentText(disp.colormap)
        self.cmap.activated.connect(lambda _: self.image.set_colormap(self.cmap.currentText()))
        display = _group("Display (orientation is display-only)", [
            _row(self.auto_levels, self.overlays, "colormap", self.cmap),
            _row("rotate (clockwise)", self.rotation, self.flipx, self.flipy),
        ])
        self.hold_btn = QtWidgets.QPushButton("Hold tracking (moving ions)")
        self.hold_btn.setCheckable(True)
        self.hold_btn.toggled.connect(self._hold)
        analysis = _group("Ion detection", [
            _row(_button("Learn background", lambda: self._cmd("learn_background", 10),
                         "learn scatter structure / hot pixels (ions are masked)"),
                 _button("Clear background", lambda: self._cmd("clear_background"))),
            _row(_button("Accept crystal as reference", lambda: self._cmd("relearn")),
                 _button("Forget crystal", lambda: self._cmd("reset_tracking"))),
            self.hold_btn,
        ])
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        for g in (acq, display, analysis):
            lay.addWidget(g)
        lay.addStretch(1)
        return w

    def _readout_tab(self):
        ro = self.config.get("readout", {})
        self.n_ions = QtWidgets.QSpinBox()
        self.n_ions.setRange(0, 64)
        self.n_ions.setSpecialValueText("any")
        self.hbin = QtWidgets.QSpinBox()
        self.hbin.setRange(1, 16)
        self.hbin.setValue(ro.get("hbin", 4))
        self.vbin = QtWidgets.QSpinBox()
        self.vbin.setRange(1, 16)
        self.vbin.setValue(ro.get("vbin", 4))
        self.auto_bin = QtWidgets.QCheckBox("auto binning")
        self.verify = QtWidgets.QCheckBox("verify")
        self.verify.setChecked(True)
        self.ro_exp = QtWidgets.QDoubleSpinBox()
        self.ro_exp.setDecimals(5)
        self.ro_exp.setRange(1e-5, 1.0)
        self.ro_exp.setSingleStep(0.0005)
        self.ro_exp.setSuffix(" s")
        self.ro_exp.setValue(ro.get("exposure", 0.0015))
        self.ro_gain = QtWidgets.QSpinBox()
        self.ro_gain.setRange(0, 1000)
        self.ro_gain.setValue(ro.get("em_gain", 200))
        self.plan_text = QtWidgets.QPlainTextEdit()
        self.plan_text.setReadOnly(True)
        f = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        self.plan_text.setFont(f)
        self.plan_text.setPlainText(format_plan(None))
        self.hist = HistogramPlot()
        self.hist_ion = QtWidgets.QSpinBox()
        self.hist_ion.setRange(0, 63)
        self.hist_label = QtWidgets.QLabel("")
        box = _group("Grabber readout", [
            _row("ions", self.n_ions, "binning", self.hbin, "x", self.vbin, self.auto_bin,
                 self.verify),
            _row("exposure", self.ro_exp, "EM gain", self.ro_gain),
            _row(_button("Locate ions", lambda: self._cmd("locate_ions", self.n_ions.value(),
                                                          callback=self._located),
                         "long exposure + precise fit (does not arm)"),
                 _button("Calibrate + arm", self._calibrate,
                         "locate -> plan AQROI/ROIs -> verify -> external trigger + Camera Link"),
                 _button("Arm", lambda: self._cmd("arm_readout")),
                 _button("Disarm -> live", lambda: self._cmd("disarm_readout")),
                 _button("Abort", lambda: self._cmd("abort"))),
        ])
        copy = _button("Copy ARTIQ snippet", self._copy_snippet)
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(box)
        lay.addWidget(self.plan_text, 2)
        lay.addLayout(_row(copy))
        lay.addLayout(_row("histogram of ion", self.hist_ion, self.hist_label))
        lay.addWidget(self.hist, 1)
        return w

    def _ions_tab(self):
        self.sites = SiteTable()
        self.crystal_label = QtWidgets.QLabel("")
        self.crystal_label.setWordWrap(True)
        self.wait_n = QtWidgets.QSpinBox()
        self.wait_n.setRange(1, 64)
        self.wait_n.setValue(1)
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(self.crystal_label)
        lay.addWidget(self.sites, 1)
        lay.addLayout(_row("loading: alert when", self.wait_n,
                           _button("ions trapped", self._wait_ions,
                                   "beeps and pops up when this many ions are trapped")))
        return w

    def _camera_tab(self):
        self.info_text = QtWidgets.QPlainTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setPlainText("\n".join("{}: {}".format(k, v) for k, v in self.info.items()))
        self.setpoint = QtWidgets.QSpinBox()
        self.setpoint.setRange(-100, 20)
        self.setpoint.setValue(int(self.config.get("camera", {}).get("target_temperature", -60)))
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(_group("Cooling", [
            _row("set point [C]", self.setpoint,
                 _button("Set", lambda: self._cmd("set_temperature", self.setpoint.value())),
                 _button("Cooler on", lambda: self._cmd("set_cooler", True)),
                 _button("Cooler off", lambda: self._cmd("set_cooler", False)))]))
        lay.addWidget(_group("Control (advisory lease shown to all clients)", [
            _row(_button("Take control (GUI)", lambda: self._cmd("take_control", "gui",
                                                                 "manual operation", 1800)),
                 _button("Release", lambda: self._cmd("release_control", None, True)))]))
        lay.addWidget(self.info_text, 1)
        return w

    def _sim_tab(self):
        self.sim_idx = QtWidgets.QSpinBox()
        self.sim_idx.setRange(0, 63)
        c = self._cmd
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel("Training / test: change the simulated trap and watch "
                                       "how ioncam reacts."))
        lay.addWidget(_group("Ions", [
            _row(_button("Load ion", lambda: c("sim_load_ion", 1, 1.0)),
                 _button("Lose random ion", lambda: c("sim_lose_ion", -1))),
            _row("ion", self.sim_idx,
                 _button("dark", lambda: c("sim_set_dark", self.sim_idx.value(), True)),
                 _button("bright", lambda: c("sim_set_dark", self.sim_idx.value(), False)),
                 _button("lose", lambda: c("sim_lose_ion", self.sim_idx.value()))),
            _row(_button("Melt 1 s", lambda: c("sim_melt", 1.0)),
                 _button("Shift +5 px", lambda: c("sim_shift", 5.0, 0.0)),
                 _button("Shift -5 px", lambda: c("sim_shift", -5.0, 0.0))),
        ]))
        lay.addWidget(_group("Lab", [
            _row(_button("Lasers off", lambda: c("sim_set_lasers", False)),
                 _button("Lasers on", lambda: c("sim_set_lasers", True))),
            _row(_button("Oven on (loading)", lambda: c("sim_set_oven", True, 0.3)),
                 _button("Oven off", lambda: c("sim_set_oven", False))),
            _row(_button("100 external triggers", lambda: c("sim_trigger", 100, 0.005))),
        ]))
        lay.addStretch(1)
        return w

    # ------------------------------------------------------------ actions
    def _cmd(self, name, *args, callback=None, **kwargs):
        self.msg_label.setText("{}...".format(name))

        def done(res):
            self.msg_label.setText("{}: ok".format(name))
            if callback is not None:
                callback(res)
        return self.backend.call(name, *args, callback=done, **kwargs)

    def _error(self, name, exc):
        self.msg_label.setText("{} failed: {}".format(name, exc))
        if name != "refresh":
            QtWidgets.QMessageBox.warning(self, "ioncam", "{} failed:\n{}".format(name, exc))

    def _crop_view(self):
        self._cmd("set_roi", *self.image.visible_sensor_rect())

    def _crop_ions(self):
        sites = self._status.get("sites") or []
        if not sites:
            return
        xs = [s["x"] for s in sites]
        ys = [s["y"] for s in sites]
        m = 40
        self._cmd("set_roi", int(min(xs) - m), int(min(ys) - m), int(max(xs) + m) + 1,
                  int(max(ys) + m) + 1)

    def _toggle_overlays(self, on):
        self.image.show_overlays = on
        self.image.set_plan(self._plan)

    def _orientation(self):
        self.image.set_orientation(int(self.rotation.currentText()), self.flipx.isChecked(),
                                   self.flipy.isChecked())

    def _hold(self, on):
        if on:
            self._cmd("hold_tracking", "moved from GUI", 600)
        else:
            self._cmd("resume_tracking")

    def _calibrate(self):
        kw = dict(n_ions=self.n_ions.value(), hbin=self.hbin.value(), vbin=self.vbin.value(),
                  auto_binning=self.auto_bin.isChecked(), verify=self.verify.isChecked(),
                  exposure=self.ro_exp.value(), em_gain=self.ro_gain.value())
        self._cmd("calibrate_readout", callback=self._got_plan, **kw)

    def _located(self, res):
        QtWidgets.QMessageBox.information(
            self, "ioncam", "Located {} ion(s)\nPSF sigma {} px\npositions:\n{}".format(
                res["n_ions"], res["psf_sigma"],
                "\n".join("  {:.3f}, {:.3f}  ({})".format(p[0], p[1], s)
                          for p, s in zip(res["positions"], res["states"]))))

    def _got_plan(self, plan):
        self._plan = plan
        self._plan_created = plan.get("created")
        self.plan_text.setPlainText(format_plan(plan))
        self.image.set_plan(plan)

    def _copy_snippet(self):
        QtWidgets.QApplication.clipboard().setText(artiq_snippet(self._plan))
        self.msg_label.setText("ARTIQ snippet copied")

    def _wait_ions(self):
        n = self.wait_n.value()

        def done(res):
            if res.get("ok"):
                QtWidgets.QApplication.beep()
                QtWidgets.QMessageBox.information(self, "ioncam",
                                                  "{} ion(s) trapped!".format(res["n_bright"]))
        self.backend.call("wait_for_ions", n, 3600.0, True, callback=done, channel="wait")
        self.msg_label.setText("waiting for {} ion(s)...".format(n))

    # ------------------------------------------------------------ polling
    def _poll(self):
        if self._polling:
            return
        self._polling = True
        self._tick += 1
        since, last_ev, want_plan = self._frame_index, self._last_event, self._plan_created
        cur_valid = (self._plan or {}).get("valid")
        rbin = self._remote_bin

        def job(invoke):
            out = {"status": invoke("get_status"),
                   "frame": invoke("get_frame", since, rbin),
                   "events": invoke("get_events", last_ev)}
            st = out["status"]
            if st["ions"].get("n_spots") or st["ions"].get("n_clouds"):
                out["detection"] = invoke("get_detection")
            if st["readout"]["has_plan"]:
                plan = invoke("get_readout_plan")
                if plan and (plan["created"] != want_plan or plan.get("valid") != cur_valid):
                    out["plan"] = plan
            return out
        self.backend.run_in_poll(job, self._polled).add_done_callback(
            lambda f: setattr(self, "_polling", False))

    def _polled(self, out):
        st = out["status"]
        self._status = st
        self.banner.update_status(st)
        fr = out.get("frame")
        if fr is not None:
            self._frame_index = fr["index"]
            self.image.set_frame(fr["image"], fr["roi"], fr["hbin"], fr["vbin"])
            if self.backend.remote:
                self._remote_bin = 2 if fr["image"].size * fr["hbin"] * fr["vbin"] > 300000 else 1
        det = out.get("detection") or {}
        sites = st.get("sites", [])
        self.image.set_sites(sites, det.get("spots", []), det.get("clouds", []))
        upd = st["ions"].get("last_update")
        if sites and upd and upd != getattr(self, "_last_tracker_update", None):
            self._last_tracker_update = upd
            self.trace.add(upd, sites)
        evs = out.get("events") or []
        if evs:
            self._last_event = evs[-1]["id"]
            self.events.add_events(evs)
            if any(e["severity"] == "alarm" for e in evs):
                self.banner.flash()
                QtWidgets.QApplication.beep()
        if "plan" in out:
            self._got_plan(out["plan"])
        if self.tabs.currentIndex() == 2:
            self.sites.set_sites(sites)
            ions = st["ions"]
            chain = ions.get("chain") or {}
            self.crystal_label.setText(
                "state: {}  |  bright {}  dark {}  |  chain scale {} px, angle {} deg  |  "
                "focus (PSF sigma) {} px  |  background {}  noise {}".format(
                    ions.get("state"), ions.get("n_bright"), ions.get("n_dark"),
                    chain.get("scale_px", "-"), chain.get("angle_deg", "-"),
                    st["detection"].get("focus", "-"), st["detection"].get("background", "-"),
                    st["detection"].get("noise", "-")))

    def _slow_tick(self):
        self.trace.redraw()
        if self.tabs.currentIndex() == 1 and self._status.get("readout", {}).get("shots"):
            def job(invoke):
                return invoke("get_readout_counts", 0, 3000), invoke("get_readout_stats", 3000)
            self.backend.run_in_poll(job, self._got_stats)

    def _got_stats(self, res):
        counts, stats = res
        i = self.hist_ion.value()
        sums = counts.get("sums") or []
        vals = [s[i] for s in sums if len(s) > i]
        th = None
        rois = stats.get("rois") or []
        if i < len(rois) and rois[i].get("threshold", {}).get("ok"):
            t = rois[i]["threshold"]
            th = t["threshold"]
            self.hist_label.setText("threshold {:.0f}, fidelity {:.3f}, bright {:.0%}".format(
                th, t["fidelity"], rois[i].get("bright_fraction", 0)))
        else:
            self.hist_label.setText("{} shots".format(len(vals)))
        self.hist.set_data(vals, th)

    def closeEvent(self, ev):
        self.timer.stop()
        self.slow.stop()
        super().closeEvent(ev)


class _Display:
    """Display settings from the config dict (with defaults)."""

    def __init__(self, d):
        self.rotation = int(d.get("rotation", 90))
        self.flip_x = bool(d.get("flip_x", False))
        self.flip_y = bool(d.get("flip_y", False))
        self.colormap = d.get("colormap", "inferno")
        self.level_percentiles = tuple(d.get("level_percentiles", (1.0, 99.95)))
        self.max_fps = float(d.get("max_fps", 25.0))
