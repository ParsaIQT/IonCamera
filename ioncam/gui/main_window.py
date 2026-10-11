"""The ioncam main window."""

import time

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ioncam.gui.widgets import (EventList, ImagePanel, ShotHistogramPlot, SiteTable,
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


# simulator knobs shown in the Simulation tab: (parameter, label, min, max, step, decimals)
SIM_FIELDS = [
    ("photon_rate", "ion fluorescence into the camera [photons/s]", 100, 1e6, 1000, 0),
    ("laser_power", "laser power (relative)", 0.0, 5.0, 0.1, 2),
    ("psf_sigma", "PSF core sigma [px]", 0.5, 6.0, 0.1, 2),
    ("psf_halo_fraction", "PSF halo: fraction of the light", 0.0, 0.8, 0.05, 2),
    ("psf_halo_sigma", "PSF halo sigma [px]", 0.5, 30.0, 0.5, 1),
    ("length_scale_px", "chain length scale [px]", 1.0, 100.0, 0.2, 2),
    ("trap_angle_deg", "trap rotation in the image [deg]", -180.0, 180.0, 15.0, 0),
    ("background_rate", "uniform stray light [photons/s/px]", 0.0, 1e4, 1.0, 1),
    ("scatter_rate", "laser scatter at the beam centre [photons/s/px]", 0.0, 1e5, 5.0, 1),
    ("scatter_sigma_along", "scatter width along the beam [px]", 1.0, 2000.0, 10.0, 0),
    ("scatter_sigma_across", "scatter width across the beam [px]", 1.0, 2000.0, 5.0, 0),
    ("beam_angle_deg", "beam angle [deg]", -180.0, 180.0, 5.0, 0),
    ("defects", "surface defects / dust grains", 0, 5000, 10, 0),
    ("defect_rate", "defect brightness [photons/s]", 0.0, 1e6, 500.0, 0),
    ("defect_max_height_um", "dust up to this high above the surface [um]", 0.0, 500.0, 5.0, 0),
    ("scratches", "scratches", 0, 100, 1, 0),
    ("gap_scatter", "electrode-gap scatter [photons/s per px]", 0.0, 1e4, 1.0, 1),
    ("numerical_aperture", "collection NA", 0.01, 0.95, 0.01, 2),
    ("ion_height_um", "ion height above the surface [um]", 0.0, 2000.0, 5.0, 0),
]


def _background_text(bg):
    if not bg:
        return "background: none learned (surface structure not subtracted)"
    how = "ions masked ({})".format(bg.get("masked_sites", 0)) if bg.get("mask_ions") \
        else "ions dark / no ions"
    return ("background: {} x {:.3g} s at EM gain {}, {}; brightness x{:.2f}".format(
        bg.get("n_frames"), bg.get("exposure", 0.0), bg.get("em_gain"), how,
        bg.get("scale", 1.0)))


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
    off = plan.get("manual_offset") or [0, 0]
    if any(off):
        lines.append("  moved by hand by ({:+d}, {:+d}) px from the computed plan".format(*off))
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
            if any("fidelity" in i for i in ions_v):
                lines.append("  quick fidelity (ions bright, background as dark): {}".format(
                    ", ".join("{:.1%} (bright if > {})".format(i["fidelity"], i["threshold"])
                              if "fidelity" in i else "-" for i in ions_v)))
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


def fidelity_table(res):
    """The quick fidelity estimate of ``get_readout_histograms`` as an HTML table."""
    rows = []
    for r in (res or {}).get("ions") or []:
        name = '<b style="color:#{:02x}{:02x}{:02x}">{}</b>'.format(
            *ShotHistogramPlot.color(r["label"]), r["label"])
        if not r.get("ok"):
            rows.append("<tr><td>{}</td><td colspan=5>{}</td></tr>".format(
                name, r.get("reason", "")))
            continue
        nb = r.get("dark_error_neighbours_bright")
        rows.append("<tr><td>{}</td><td align=right>&gt; {}</td><td align=right><b>{:.2%}</b></td>"
                    "<td align=right>{:.2%} ({}/{})</td><td align=right>{:.2%} ({}/{})</td>"
                    "<td align=right>{}</td></tr>".format(
                        name, r["threshold"], r["fidelity"], r["bright_error"],
                        r["bright_errors"], r["n_bright"], r["dark_error"], r["dark_errors"],
                        r["n_dark"], "" if nb is None else "{:.2%}".format(nb)))
    if not rows:
        return ""
    head = ("<tr><th align=left>ROI</th><th>bright if</th><th>fidelity</th>"
            "<th>bright&#8594;dark</th><th>bg&#8594;bright</th>"
            "<th>neighbours bright</th></tr>")
    return "<table cellspacing=0 cellpadding=2>{}{}</table>".format(head, "".join(rows))


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
        self.setWindowTitle("ioncam {} - {} ({})".format(
            info.get("ioncam_version", ""), info.get("model", "camera"), backend.describe()))
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
        self.auto_levels.setToolTip("brightness levels follow every frame; off (or after "
                                    "dragging the histogram handles) they stay as set")
        self.auto_levels.toggled.connect(self.image.set_auto_levels)
        self.image.autoLevelsChanged.connect(self._auto_levels_changed)
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
        self.view = QtWidgets.QComboBox()
        self.view.addItems(["raw image", "background removed"])
        self.view.setToolTip("background removed: what the ion detection sees - the learned "
                             "static background, the scatter profile and the bias subtracted")
        self._view = "raw"
        self.view.activated.connect(lambda i: setattr(self, "_view", ("raw", "corrected")[i]))
        display = _group("Display (orientation is display-only)", [
            _row(self.auto_levels, self.overlays, "colormap", self.cmap),
            _row("rotate (clockwise)", self.rotation, self.flipx, self.flipy),
            _row("show", self.view),
        ])
        self.hold_btn = QtWidgets.QPushButton("Hold tracking (moving ions)")
        self.hold_btn.setCheckable(True)
        self.hold_btn.toggled.connect(self._hold)
        self.bg_frames = QtWidgets.QSpinBox()
        self.bg_frames.setRange(5, 1000)
        self.bg_frames.setValue(50)
        self.bg_frames.setToolTip("frames (full sensor, live exposure and gain) - more frames, "
                                  "less noise in the learned structure")
        self.bg_dark = QtWidgets.QCheckBox("ions dark / trap empty")
        self.bg_dark.setToolTip("repumper off (ions shelved, the laser still scatters) or no "
                                "ions: nothing needs masking - the most complete background. "
                                "Unchecked: the tracked ions are modelled and masked.")
        self.bg_label = QtWidgets.QLabel("background: none learned")
        self.bg_label.setWordWrap(True)
        analysis = _group("Ion detection", [
            _row(_button("Learn background", self._learn_background,
                         "learn the static light of the trap surface - laser scatter, "
                         "defects and dust (which can look like ions or clouds), hot pixels - "
                         "and subtract it from now on"),
                 self.bg_frames, "frames", self.bg_dark),
            _row(self.bg_label, _button("Clear background",
                                        lambda: self._cmd("clear_background"))),
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
        rs = (self.config.get("calibration", {}).get("roi_size") or [1, 1])
        self.roi_w, self.roi_h = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for sp, v in ((self.roi_w, rs[0]), (self.roi_h, rs[1])):
            sp.setRange(0, 12)
            sp.setValue(int(v))
            sp.setSpecialValueText("auto")
            sp.setToolTip("Grabber ROI per ion, in super-pixels.  1 x 1: one super-pixel per "
                          "ion, the grid placed so the ions sit at their centres - the fewest "
                          "pixels and so the fewest dark counts in each sum, usually the best "
                          "SPAM fidelity.  auto: as much of each ion's light as the grid "
                          "allows (bigger ROIs, more dark counts).")
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
        box = _group("Grabber readout", [
            _row("ions", self.n_ions, "binning", self.hbin, "x", self.vbin, self.auto_bin,
                 self.verify),
            _row("ROI per ion", self.roi_w, "x", self.roi_h, "super-pixels"),
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
        lay.addWidget(self._manual_box())
        top = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.plan_text, 1)
        tl.addLayout(_row(copy))
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(self._shots_box())
        split.setSizes([150, 550])
        lay.addWidget(split, 1)
        return w

    def _shots_box(self):
        """Histograms of the shots - the verification frames or the externally
        triggered readout - of every ion ROI and the background, with a quick
        fidelity estimate: prepare the ions bright in every shot; the
        background pixels of the same shots are the dark reference."""
        self.hist = ShotHistogramPlot()
        self.hist_source = QtWidgets.QComboBox()
        self.hist_source.addItems(["latest", "external triggers", "verify"])
        self.hist_source.setToolTip("latest: the newer of the externally triggered shots "
                                    "(since arming) and the last verification")
        self.hist_source.activated.connect(lambda _: self._refresh_hist(force=True))
        self.hist_log = QtWidgets.QCheckBox("log scale")
        self.hist_log.setToolTip("the tails - where bright and dark overlap - decide the "
                                 "fidelity")
        self.hist_log.toggled.connect(self.hist.set_log)
        self.hist_log.setChecked(True)
        self.hist_info = QtWidgets.QLabel("no shots yet")
        self.hist_info.setWordWrap(True)
        self.hist_table = QtWidgets.QLabel("")
        self.hist_table.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.hist_table.setToolTip(
            "Every shot taken as prepared bright: an ion ROI's sums are its bright "
            "distribution, the background pixels of the same shots (summed to the ROI's "
            "size) the dark one.  bright if: the threshold with the fewest errors;  "
            "bright->dark: bright shots at or below it;  bg->bright: dark-reference "
            "samples above it;  fidelity: 1 - the mean of the two.  neighbours bright: "
            "the dark error once the other ions' light in this ROI (planned crosstalk) is "
            "added.")
        self._hist_key = None
        self._hist_busy = False
        g = QtWidgets.QGroupBox("Shots (prepare bright): histograms and a quick fidelity "
                                "estimate")
        gl = QtWidgets.QVBoxLayout(g)
        gl.addLayout(_row("shots", self.hist_source, self.hist_log))
        gl.addWidget(self.hist_info)
        gl.addWidget(self.hist, 1)
        gl.addWidget(self.hist_table)
        return g

    def _manual_box(self):
        """Adjusting the readout by hand: a D-pad moves the AQROI - and every
        Grabber ROI with it - on screen, re-planning with another binning or
        inside a given AQROI keeps the located ion positions."""
        self.nudge_step = QtWidgets.QSpinBox()
        self.nudge_step.setRange(1, 64)
        self.nudge_step.setValue(1)
        self.nudge_step.setSuffix(" px")
        self.nudge_step.setToolTip("step of the D-pad [sensor px]")
        pad = QtWidgets.QGridLayout()
        pad.setSpacing(2)
        arrows = QtCore.Qt.ArrowType
        for r, c, arrow, right, up, name in ((0, 1, arrows.UpArrow, 0, 1, "up"),
                                             (1, 0, arrows.LeftArrow, -1, 0, "left"),
                                             (1, 2, arrows.RightArrow, 1, 0, "right"),
                                             (2, 1, arrows.DownArrow, 0, -1, "down")):
            b = QtWidgets.QToolButton()
            b.setArrowType(arrow)
            b.setAutoRepeat(True)
            b.setAutoRepeatDelay(400)
            b.setAutoRepeatInterval(150)
            b.setMinimumSize(34, 34)
            b.setToolTip("move the AQROI - and all ROI boxes with it - {} on screen by one "
                         "step: the ions move the other way inside their ROIs".format(name))
            b.clicked.connect(lambda _=False, r_=right, u_=up: self._nudge(r_, u_))
            pad.addWidget(b, r, c)
        pad.addWidget(self.nudge_step, 1, 1)
        self.offset_label = QtWidgets.QLabel("no plan yet")
        self.offset_label.setWordWrap(True)
        self.aq_edit = []
        for name in ("x0", "y0", "x1", "y1"):
            sp = QtWidgets.QSpinBox()
            sp.setRange(0, 8192)
            sp.setToolTip("AQROI {} [sensor px; x1, y1 exclusive]".format(name))
            self.aq_edit.append(sp)
        self._nudge_pending = [0, 0]
        self._nudge_busy = False
        right = QtWidgets.QVBoxLayout()
        right.addWidget(self.offset_label)
        right.addLayout(_row(
            _button("Re-plan", self._replan, "plan again with the binning set above, from the "
                    "ions already located (no new locate)"),
            _button("Verify", lambda: self._cmd("verify_readout", callback=self._got_plan),
                    "test frames with this plan: signal of every ion in its ROI, and the "
                    "shot histograms below with a quick fidelity estimate")))
        right.addLayout(_row("AQROI", *self.aq_edit))
        right.addLayout(_row(_button("Use this AQROI", self._use_aqroi,
                                     "place the ROIs inside this AQROI (sensor px, x1/y1 "
                                     "exclusive; its size a multiple of the binning)")))
        g = QtWidgets.QGroupBox("Adjust by hand (D-pad moves the AQROI and all ROIs)")
        gl = QtWidgets.QHBoxLayout(g)
        gl.addLayout(pad)
        gl.addLayout(right, 1)
        return g

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
            _row("rotate the trap", *[_button(
                "{} deg".format(a), lambda _=False, a_=a: self._set_sim(trap_angle_deg=a_),
                "turn the whole trap in the image (ions, electrodes, surface, laser beam) - "
                "then learn the background again") for a in (0, 45, 90)]),
        ]))
        lay.addWidget(_group("Lab", [
            _row(_button("Lasers off", lambda: c("sim_set_lasers", False)),
                 _button("Lasers on", lambda: c("sim_set_lasers", True))),
            _row(_button("Repumper off (ions dark)", lambda: c("sim_set_repumper", False)),
                 _button("Repumper on", lambda: c("sim_set_repumper", True))),
            _row(_button("Oven on (loading)", lambda: c("sim_set_oven", True, 0.3)),
                 _button("Oven off", lambda: c("sim_set_oven", False))),
            _row("move camera", _button("+5 px x", lambda: c("sim_move_camera", 5.0, 0.0)),
                 _button("-5 px x", lambda: c("sim_move_camera", -5.0, 0.0)),
                 _button("+5 px y", lambda: c("sim_move_camera", 0.0, 5.0))),
            _row(_button("100 external triggers", lambda: c("sim_trigger", 100, 0.005))),
        ]))

        # surface, optics and light: the realism knobs
        self.sim_preset = QtWidgets.QComboBox()
        self.sim_preset.addItems(["clean", "typical", "dirty", "horrendous"])
        self.sim_preset.activated.connect(lambda _: self._set_sim(
            surface=self.sim_preset.currentText()))
        self.sim_fields = {}
        form = QtWidgets.QFormLayout()
        form.addRow("surface preset", _row(self.sim_preset, _button(
            "New surface", lambda: self._set_sim(
                surface_seed=int(self._sim_params.get("surface_seed", 1)) + 1),
            "same kind of surface, different defects")))
        for name, label, lo, hi, step, dec in SIM_FIELDS:
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            sp.setDecimals(dec)
            sp.setKeyboardTracking(False)
            sp.editingFinished.connect(lambda n=name, w_=sp: self._set_sim(**{n: w_.value()}))
            self.sim_fields[name] = sp
            form.addRow(label, sp)
        self.sim_offset = []
        for axis in ("x", "y"):
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(-500.0, 500.0)
            sp.setSingleStep(5.0)
            sp.setDecimals(0)
            sp.setKeyboardTracking(False)
            sp.editingFinished.connect(lambda: self._set_sim(scatter_offset=[
                self.sim_offset[0].value(), self.sim_offset[1].value()]))
            self.sim_offset.append(sp)
        form.addRow("scatter centre offset x, y [px]", _row(*self.sim_offset))
        self.sim_info = QtWidgets.QLabel("")
        self.sim_info.setWordWrap(True)
        optics = QtWidgets.QGroupBox("Surface, optics and light")
        ol = QtWidgets.QVBoxLayout(optics)
        ol.addLayout(form)
        ol.addWidget(self.sim_info)
        lay.addWidget(optics)
        lay.addStretch(1)
        self._sim_params = {}
        self.backend.call("sim_get_params", callback=self._got_sim_params)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(w)
        return scroll

    def _set_sim(self, **params):
        self._cmd("sim_set_params", callback=self._got_sim_params, **params)

    def _got_sim_params(self, p):
        self._sim_params = dict(p or {})
        for name, sp in self.sim_fields.items():
            if name in p:
                sp.blockSignals(True)
                sp.setValue(float(p[name]))
                sp.blockSignals(False)
        off = p.get("scatter_offset") or (0.0, 0.0)
        for sp, v in zip(self.sim_offset, off):
            sp.blockSignals(True)
            sp.setValue(float(v))
            sp.blockSignals(False)
        if p.get("surface") in [self.sim_preset.itemText(i) for i in range(self.sim_preset.count())]:
            self.sim_preset.setCurrentText(p["surface"])
        self.sim_info.setText(
            "A point on the surface is imaged as a disk of radius {:.1f} px (ions in focus at "
            "{:.0f} um height, NA {:.2f}); dust closer to focus looks like ions.  After changing "
            "the surface or moving the camera, learn the background again.".format(
                p.get("defocus_radius_px", 0.0), p.get("ion_height_um", 0.0),
                p.get("numerical_aperture", 0.0)))

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

    def _learn_background(self):
        self._cmd("learn_background", self.bg_frames.value(), not self.bg_dark.isChecked())

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
                  roi_size=self._roi_size(), auto_binning=self.auto_bin.isChecked(),
                  verify=self.verify.isChecked(), exposure=self.ro_exp.value(),
                  em_gain=self.ro_gain.value())
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
        off = plan.get("manual_offset") or [0, 0]
        self.offset_label.setText("AQROI {} (Solis L,R,B,T {}), {}x{} binning{}".format(
            plan["aqroi"], plan["aqroi_solis"], plan["binning"][0], plan["binning"][1],
            ", moved by hand ({:+d}, {:+d}) px".format(*off) if any(off) else ", as planned"))
        for sp, v in zip(self.aq_edit, plan["aqroi"]):
            sp.setValue(int(v))

    def _auto_levels_changed(self, on):
        self.auto_levels.blockSignals(True)
        self.auto_levels.setChecked(on)
        self.auto_levels.blockSignals(False)

    def _nudge(self, right, up):
        """D-pad: move the AQROI one step in a screen direction (steps pressed
        while a move is being applied are added up and sent together)."""
        if not self._plan:
            self.msg_label.setText("no readout plan to adjust - calibrate first")
            return
        dx, dy = self.image.screen_step(right, up)
        k = self.nudge_step.value()
        self._nudge_pending[0] += dx * k
        self._nudge_pending[1] += dy * k
        self._send_nudge()

    def _send_nudge(self):
        if self._nudge_busy or self._nudge_pending == [0, 0]:
            return
        dx, dy = self._nudge_pending
        self._nudge_pending = [0, 0]
        self._nudge_busy = True
        self.msg_label.setText("moving the AQROI by ({:+d}, {:+d}) px...".format(dx, dy))

        def done(plan):
            self._nudge_busy = False
            self.msg_label.setText("AQROI moved by ({:+d}, {:+d}) px".format(dx, dy))
            self._got_plan(plan)
            self._send_nudge()

        def failed(name, exc):
            self._nudge_busy = False
            self._nudge_pending = [0, 0]
            self.msg_label.setText("cannot move the AQROI: {}".format(exc))
        self.backend.call("nudge_readout", dx, dy, callback=done, error=failed)

    def _roi_size(self):
        """[w, h] super-pixels; [0, 0] = auto (either one "auto" makes both)."""
        w, h = self.roi_w.value(), self.roi_h.value()
        return [0, 0] if w == 0 or h == 0 else [w, h]

    def _replan(self):
        self._cmd("replan_readout", self.hbin.value(), self.vbin.value(), None,
                  self._roi_size(), callback=self._got_plan)

    def _use_aqroi(self):
        self._cmd("replan_readout", self.hbin.value(), self.vbin.value(),
                  [sp.value() for sp in self.aq_edit], self._roi_size(), callback=self._got_plan)

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
        view = self._view

        def job(invoke):
            out = {"status": invoke("get_status"),
                   "frame": invoke("get_frame", since, rbin, None, view),
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
        self.bg_label.setText(_background_text(st.get("background")))
        fr = out.get("frame")
        if fr is not None:
            self._frame_index = fr["index"]
            self.image.set_frame(fr["image"], fr["roi"], fr["hbin"], fr["vbin"])
            if self.backend.remote:
                self._remote_bin = 2 if fr["image"].size * fr["hbin"] * fr["vbin"] > 300000 else 1
        det = out.get("detection") or {}
        sites = st.get("sites", [])
        self.image.set_sites(sites, det.get("spots", []), det.get("clouds", []),
                             st["ions"].get("marker_radius"))
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
        if self.tabs.currentIndex() == 1:
            self._refresh_hist()

    def _refresh_hist(self, force=False):
        """Fetch the shot histograms when there is something new."""
        ro = self._status.get("readout") or {}
        src = ("auto", "trigger", "verify")[self.hist_source.currentIndex()]
        key = (self._plan_created, ro.get("shots"), ro.get("verified"), src)
        if self._hist_busy or not ro.get("has_plan") or (key == self._hist_key and not force):
            return
        self._hist_key = key
        self._hist_busy = True

        def job(invoke):
            return invoke("get_readout_histograms", src, 20000)

        def done(f):
            self._hist_busy = False
        self.backend.run_in_poll(job, self._got_hist).add_done_callback(done)

    def _got_hist(self, res):
        self.hist.set_data(res)
        self.hist_table.setText(fidelity_table(res))
        av = res.get("available") or {}
        if not res.get("shots"):
            self.hist_info.setText(
                "no {}shots for this plan yet: Verify, or arm and trigger (prepare the ions "
                "bright in every shot)".format({"trigger": "externally triggered ",
                                                "verify": "verification "}.get(
                                                    res.get("source"), "")))
            return
        self.hist_info.setText(
            "{} {} (verification {}, triggered {});  dark reference: {}, {} super-pixel{} "
            "per shot".format(res["shots"], {"trigger": "externally triggered shots",
                                             "verify": "verification shots"}[res["source"]],
                              av.get("verify", 0), av.get("trigger", 0),
                              res.get("dark_reference", ""), res.get("dark_pixels", 0),
                              "s" * (res.get("dark_pixels", 0) != 1)))

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
