"""The ioncam main window.

Layout: the crystal status banner on top; the image (with its view bar) over
the per-ion brightness trace on the left; tabs on the right, in the order of
the work - Live (camera and display), Ions (crystal, tracking, background),
Readout (calibrate, adjust by hand, run, shots), Events, Camera (cooling,
control, configuration) and, simulated, Simulation.  The toolbar and the
menus hold the actions used from everywhere, with keyboard shortcuts
(Help > Keyboard shortcuts).
"""

import json
import os
import time

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ioncam.gui import style
from ioncam.gui.widgets import (EventList, ImagePanel, ShotHistogramPlot, SiteTable,
                                StatTile, StatusBanner, TracePlot)


def _button(text, slot, tip=None, icon=None, kind=None):
    b = QtWidgets.QPushButton(text)
    b.clicked.connect(slot)
    if tip:
        b.setToolTip(tip)
    if icon:
        b.setProperty("icon_name", icon)
        b.setIcon(style.icon(icon, _icon_color(kind)))
    if kind:
        style.kind(b, kind)
    return b


def _icon_color(kind):
    """Icon colour of a push button: white on the coloured ones."""
    t = style.current()
    return t["accent_text"] if kind in ("primary", "danger") else t["text"]


def _action_icon_color(kind):
    """Icon colour of an action (tool buttons, menus): the danger colour for
    the danger ones (drawn on a tint, not on red)."""
    t = style.current()
    return {"primary": t["accent_text"], "danger": t["danger"]}.get(kind, t["text"])


def _group(title, widgets):
    g = QtWidgets.QGroupBox(title)
    lay = QtWidgets.QVBoxLayout(g)
    lay.setSpacing(7)
    for w in widgets:
        if isinstance(w, QtWidgets.QLayout):
            lay.addLayout(w)
        else:
            lay.addWidget(w)
    return g


def _row(*widgets):
    """A row: widgets, "text" (a label), None (stretch) or an int (spacing)."""
    lay = QtWidgets.QHBoxLayout()
    lay.setSpacing(6)
    for w in widgets:
        if w is None:
            lay.addStretch(1)
        elif isinstance(w, int):
            lay.addSpacing(w)
        elif isinstance(w, str):
            lay.addWidget(QtWidgets.QLabel(w))
        elif isinstance(w, QtWidgets.QLayout):
            lay.addLayout(w)
        else:
            lay.addWidget(w)
    return lay


def _grid(rows, stretch_last=True):
    """A grid of label / widget cells; a str cell is a muted label."""
    g = QtWidgets.QGridLayout()
    g.setHorizontalSpacing(8)
    g.setVerticalSpacing(6)
    for r, cells in enumerate(rows):
        for c, w in enumerate(cells):
            if w is None:
                continue
            if isinstance(w, int):
                g.setColumnMinimumWidth(c, w)
                continue
            if isinstance(w, str):
                w = _muted(w)
            if isinstance(w, QtWidgets.QLayout):
                g.addLayout(w, r, c)
            else:
                g.addWidget(w, r, c)
    if stretch_last:
        g.setColumnStretch(g.columnCount(), 1)
    return g


def _muted(text, wrap=False):
    lab = QtWidgets.QLabel(text)
    lab.setProperty("role", "muted")
    lab.setWordWrap(wrap)
    return lab


def _scroll(widget):
    sa = QtWidgets.QScrollArea()
    sa.setWidgetResizable(True)
    sa.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    sa.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    sa.setWidget(widget)
    return sa


def _page(*items):
    """A tab page: groups and layouts stacked from the top."""
    w = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(w)
    lay.setContentsMargins(8, 8, 8, 8)
    lay.setSpacing(8)
    for it in items:
        if isinstance(it, QtWidgets.QLayout):
            lay.addLayout(it)
        elif it is None:
            lay.addStretch(1)
        else:
            lay.addWidget(it)
    return w


def _spin(lo, hi, value, step=1, suffix="", decimals=None, tip=None, special=None,
          width=None):
    sp = QtWidgets.QDoubleSpinBox() if decimals is not None else QtWidgets.QSpinBox()
    if width:
        sp.setMaximumWidth(width)
    if decimals is not None:
        sp.setDecimals(decimals)
    sp.setRange(lo, hi)
    sp.setSingleStep(step)
    sp.setValue(value)
    if suffix:
        sp.setSuffix(suffix)
    if tip:
        sp.setToolTip(tip)
    if special:
        sp.setSpecialValueText(special)
    return sp


def _tool(action, text_beside=True, flat=False):
    """A button bound to ``action`` (shares its state, icon and tooltip);
    ``flat`` for bars, else drawn like the push buttons."""
    b = QtWidgets.QToolButton()
    b.setDefaultAction(action)
    if not flat:
        b.setProperty("role", "button")
    b.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon if text_beside
                         else QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
    return b


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
    def __init__(self, backend, config, info, settings=None):
        """``settings``: a QSettings to keep the theme and the window layout
        between sessions (None: nothing is kept)."""
        super().__init__()
        self.backend = backend
        self.config = config            # dict of sections (from get_config)
        self.info = info
        self.settings = settings
        self.simulated = bool(info.get("simulated"))
        backend.on_error = self._error
        disp = _Display(config.get("display", {}))
        theme = str(settings.value("theme", "dark")) if settings is not None else "dark"
        style.apply_theme(QtWidgets.QApplication.instance(), theme)
        self.setWindowTitle("ioncam {} - {} ({})".format(
            info.get("ioncam_version", ""), info.get("model", "camera"), backend.describe()))
        self.resize(1560, 960)
        self._themed = []                   # (widget or action, icon name, kind)

        self._frame_index = 0
        self._frame_meta = {}
        self._last_event = 0
        self._plan_created = None
        self._plan = None
        self._polling = False
        self._status = {}
        self._tick = 0
        self._stats = None
        self._remote_bin = 1
        self._busy = None

        self.banner = StatusBanner()
        self.image = ImagePanel(disp)
        self.image.hovered.connect(lambda s: self.hover_label.setText(s))
        self.trace = TracePlot()
        self.events = EventList()
        self._make_actions()
        self._make_menus()
        self._make_toolbar()

        self.left = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.left.addWidget(self._image_box())
        self.left.addWidget(self._trace_box())
        self.left.setSizes([680, 210])
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(False)
        self.live_page = self._live_tab(disp)
        self.ions_page = self._ions_tab()
        self.readout_page = self._readout_tab()
        self.events_page = self._events_tab()
        self.camera_page = self._camera_tab()
        self.tabs.addTab(self.live_page, "Live")
        self.tabs.addTab(self.ions_page, "Ions")
        self.tabs.addTab(self.readout_page, "Readout")
        self.tabs.addTab(self.events_page, "Events")
        self.tabs.addTab(self.camera_page, "Camera")
        if self.simulated:
            self.sim_page = self._sim_tab()
            self.tabs.addTab(self.sim_page, "Simulation")
        self.split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.split.addWidget(self.left)
        self.split.addWidget(self.tabs)
        self.split.setStretchFactor(0, 1)
        self.split.setSizes([1010, 540])
        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.setContentsMargins(8, 8, 8, 6)
        lay.setSpacing(8)
        lay.addWidget(self.banner)
        lay.addWidget(self.split, 1)
        self.setCentralWidget(central)
        self.hover_label = QtWidgets.QLabel("")
        self.msg_label = QtWidgets.QLabel("")
        self.busy_bar = QtWidgets.QProgressBar()
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setFixedWidth(120)
        self.busy_bar.setTextVisible(False)
        self.busy_bar.hide()
        self.statusBar().addWidget(self.hover_label, 1)
        self.statusBar().addPermanentWidget(self.busy_bar)
        self.statusBar().addPermanentWidget(self.msg_label)
        self._restore_layout()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(int(1000 / max(1.0, disp.max_fps)))
        self.slow = QtCore.QTimer(self)
        self.slow.timeout.connect(self._slow_tick)
        self.slow.start(500)

    # ------------------------------------------------------------ actions
    def _action(self, text, slot, shortcut=None, icon=None, tip=None, checkable=False,
                kind=None):
        a = QtGui.QAction(text, self)
        if shortcut:
            a.setShortcut(QtGui.QKeySequence(shortcut))
            a.setShortcutContext(QtCore.Qt.ShortcutContext.WindowShortcut)
        if tip or shortcut:
            a.setToolTip("{}{}".format(tip or text, "  ({})".format(shortcut) if shortcut
                                       else ""))
            a.setStatusTip(tip or text)
        if checkable:
            a.setCheckable(True)
            a.toggled.connect(slot)
        else:
            a.triggered.connect(lambda _=False: slot())
        if icon:
            a.setIcon(style.icon(icon, _action_icon_color(kind)))
            self._themed.append((a, icon, kind))
        self.addAction(a)
        return a

    def _make_actions(self):
        A = self._action
        c = self._cmd
        self.act = {
            "live": A("Live", lambda: c("start_live"), "F5", "play",
                      "start live video (internal trigger)"),
            "stop": A("Stop", lambda: c("stop"), "F6", "stop",
                      "stop acquiring (also disarms the readout)"),
            "snap": A("Snap...", self._snap_save, "Ctrl+Shift+S", "camera",
                      "average frames with the live settings and save them"),
            "save_image": A("Save image...", self._save_image, "Ctrl+S", "save",
                            "save the image on screen (raw data .npy / .csv, or the view "
                            "as .png)"),
            "learn_bg": A("Learn background", self._learn_background, "Ctrl+B", "layers",
                          "learn the static light of the trap surface and subtract it from "
                          "now on"),
            "clear_bg": A("Clear background", lambda: c("clear_background"), None, "trash",
                          "forget the learned background"),
            "hold": A("Hold", self._hold, "Ctrl+H", "pause",
                      "hold tracking: the ions will move on purpose (shuttling, moving the "
                      "camera) - no lost / trapped events until resumed", checkable=True),
            "accept": A("Accept crystal as reference", lambda: c("relearn"), None, "check",
                        "take the crystal as it is now as the reference (brightness levels, "
                        "positions)"),
            "forget": A("Forget crystal", lambda: c("reset_tracking"), None, "refresh",
                        "start tracking afresh, as if the ions had just appeared"),
            "locate": A("Locate", self._locate, "Ctrl+L", "target",
                        "long exposure + precise fit of every ion (does not arm)"),
            "calibrate": A("Calibrate + arm", self._calibrate, "Ctrl+K", "check",
                           "locate -> plan AQROI/ROIs -> verify -> external trigger + Camera "
                           "Link", kind="primary"),
            "verify": A("Verify", self._verify, "Ctrl+Shift+V", "verify",
                        "test frames with the current plan: every ion in its ROI, and the "
                        "shot histograms with a quick fidelity estimate"),
            "arm": A("Arm", lambda: c("arm_readout"), "Ctrl+R", "bolt",
                     "external trigger + Camera Link with the current plan"),
            "disarm": A("Disarm", lambda: c("disarm_readout"), "Ctrl+D", "power",
                        "end the readout and go back to live video"),
            "abort": A("Abort", lambda: c("abort"), "Esc", "abort",
                       "abort the running procedure (locate, calibrate, verify, snap, "
                       "learn background)", kind="danger"),
            "fit": A("Fit image", self.image.fit_view, "Ctrl+0", "fit",
                     "zoom out to the whole image"),
            "zoom_ions": A("Zoom to ions", self._zoom_ions, "Ctrl+I", "zoom",
                           "zoom to the tracked ions"),
            "overlays": A("Overlays", self._toggle_overlays, None, None,
                          "markers, labels, Grabber ROIs and spots on the image",
                          checkable=True),
            "settings": A("Settings...", self._settings_dialog, "Ctrl+,", "gear",
                          "every configuration option: change it now, or write a "
                          "configuration file"),
            "save_config": A("Save configuration...", self._save_config, None, "save",
                             "write the configuration as it is now (with comments) to a "
                             "file"),
            "theme": A("Light theme", lambda on: self.set_theme("light" if on else "dark"),
                       "Ctrl+T", "theme", "switch between the dark and the light theme",
                       checkable=True),
            "fullscreen": A("Full screen", self._fullscreen, "F11", None,
                            "full screen on / off", checkable=True),
            "copy_snippet": A("Copy ARTIQ snippet", self._copy_snippet, None, "copy",
                              "the Grabber ROIs and masks as Python, for an experiment"),
            "copy_plan": A("Copy plan as JSON", self._copy_plan_json, None, "copy",
                           "the whole readout plan as JSON"),
            "full_sensor": A("Full sensor", lambda: c("reset_roi"), None, None,
                             "live video on the whole sensor"),
            "crop_view": A("Crop to view", self._crop_view, None, None,
                           "acquire only the area visible on screen"),
            "crop_ions": A("Crop around ions", self._crop_ions, None, None,
                           "acquire only the area around the tracked ions"),
            "cooler_on": A("Cooler on", lambda: c("set_cooler", True), None, None),
            "cooler_off": A("Cooler off", lambda: c("set_cooler", False), None, None),
            "take": A("Take control", lambda: c("take_control", "gui", "manual operation",
                                                1800), None, None,
                      "advisory lease shown to every client (experiments check it)"),
            "release": A("Release control", lambda: c("release_control", None, True), None,
                         None),
            "save_events": A("Save events...", self._save_events, None, "save",
                             "write the event log to a text file"),
            "shortcuts": A("Keyboard shortcuts", self._shortcuts_dialog, "F1", None),
            "about": A("About ioncam", self._about, None, None),
            "quit": A("Quit", self.close, "Ctrl+Q", None),
        }
        for key, on in (("overlays", True), ("theme", style.theme_name() == "light")):
            self.act[key].blockSignals(True)
            self.act[key].setChecked(on)
            self.act[key].blockSignals(False)

    def _make_menus(self):
        mb = self.menuBar()
        a = self.act

        def menu(title, items):
            m = mb.addMenu(title)
            for it in items:
                if it is None:
                    m.addSeparator()
                elif isinstance(it, QtWidgets.QMenu):
                    m.addMenu(it)
                else:
                    m.addAction(it)
            return m
        menu("&File", [a["save_image"], a["snap"], a["save_events"], None, a["settings"],
                       a["save_config"], None, a["quit"]])
        self.overlay_menu = QtWidgets.QMenu("Show on the image")
        self.overlay_actions = {}
        for key, text in (("markers", "Ion markers"), ("labels", "Ion numbers"),
                          ("rois", "Grabber ROIs and AQROI"), ("spots", "Detected spots")):
            act = QtGui.QAction(text, self)
            act.setCheckable(True)
            act.setChecked(True)
            act.toggled.connect(lambda on, k=key: self._set_overlay(k, on))
            self.overlay_menu.addAction(act)
            self.overlay_actions[key] = act
        menu("&View", [a["fit"], a["zoom_ions"], None, a["overlays"], self.overlay_menu, None,
                       a["theme"], a["fullscreen"]])
        menu("&Camera", [a["live"], a["stop"], a["snap"], None, a["full_sensor"],
                         a["crop_view"], a["crop_ions"], None, a["cooler_on"], a["cooler_off"],
                         None, a["take"], a["release"]])
        menu("&Ions", [a["learn_bg"], a["clear_bg"], None, a["accept"], a["forget"], a["hold"],
                       None, a["locate"]])
        menu("&Readout", [a["calibrate"], a["verify"], a["arm"], a["disarm"], a["abort"], None,
                          a["copy_snippet"], a["copy_plan"]])
        menu("&Help", [a["shortcuts"], a["about"]])

    def _make_toolbar(self):
        tb = QtWidgets.QToolBar("Main")
        tb.setObjectName("main_toolbar")
        tb.setMovable(False)
        tb.setIconSize(QtCore.QSize(18, 18))
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        a = self.act
        for item in (a["live"], a["stop"], a["snap"], None, a["learn_bg"], a["hold"], None,
                     a["locate"], a["calibrate"], a["verify"], a["arm"], a["disarm"], None,
                     a["abort"]):
            if item is None:
                tb.addSeparator()
            else:
                tb.addAction(item)
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                             QtWidgets.QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)
        for item in (a["save_image"], a["settings"], a["theme"]):
            tb.addAction(item)
            tb.widgetForAction(item).setToolButtonStyle(
                QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
        style.kind(tb.widgetForAction(a["calibrate"]), "primary")
        style.kind(tb.widgetForAction(a["abort"]), "danger")
        self.addToolBar(tb)
        self.toolbar = tb

    # -------------------------------------------------------------- left side
    def _image_box(self):
        """The image with a bar: zoom, what to show, overlays, save."""
        self.view = QtWidgets.QComboBox()
        self.view.addItems(["raw image", "background removed", "learned background"])
        self.view.setToolTip("raw: the camera frame.  background removed: what the ion "
                             "detection sees - the learned static background, the scatter "
                             "profile and the bias subtracted.  learned background: the "
                             "static light that is subtracted")
        self._view = "raw"
        self.view.activated.connect(lambda i: setattr(
            self, "_view", ("raw", "corrected", "background")[i]))
        bar = QtWidgets.QFrame()
        bar.setProperty("role", "bar")
        bl = QtWidgets.QHBoxLayout(bar)
        bl.setContentsMargins(6, 3, 6, 3)
        bl.setSpacing(4)
        bl.addWidget(_tool(self.act["fit"], False, True))
        bl.addWidget(_tool(self.act["zoom_ions"], False, True))
        bl.addSpacing(8)
        bl.addWidget(_muted("show"))
        bl.addWidget(self.view)
        bl.addSpacing(8)
        bl.addWidget(_muted("overlays"))
        self.overlay_buttons = {}
        for key, text, tip in (("markers", "markers", "circles holding 80 % of each ion's "
                                "light (green: bright, red dashed: dark)"),
                               ("labels", "numbers", "ion numbers (chain order)"),
                               ("rois", "ROIs", "Grabber ROIs, background ROI, AQROI and the "
                                "binning grid"),
                               ("spots", "spots", "what the detector found (+: spot, orange "
                                "circle: cloud)")):
            b = QtWidgets.QToolButton()
            b.setText(text)
            b.setCheckable(True)
            b.setChecked(True)
            b.setToolTip(tip)
            b.toggled.connect(lambda on, k=key: self._set_overlay(k, on))
            bl.addWidget(b)
            self.overlay_buttons[key] = b
        bl.addStretch(1)
        bl.addWidget(_tool(self.act["save_image"], False, True))
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(bar)
        lay.addWidget(self.image, 1)
        return w

    def _trace_box(self):
        self.trace_window = QtWidgets.QComboBox()
        self.trace_window.addItems(["30 s", "2 min", "10 min"])
        self.trace_window.setCurrentIndex(1)
        self.trace_window.activated.connect(lambda i: setattr(
            self.trace, "seconds", (30.0, 120.0, 600.0)[i]))
        pause = QtWidgets.QToolButton()
        pause.setText("Pause")
        pause.setCheckable(True)
        pause.setToolTip("freeze the plot (the data keep coming in)")
        pause.toggled.connect(lambda on: setattr(self.trace, "paused", on))
        clear = QtWidgets.QToolButton()
        clear.setText("Clear")
        clear.clicked.connect(self.trace.clear_history)
        bar = QtWidgets.QFrame()
        bar.setProperty("role", "bar")
        bl = QtWidgets.QHBoxLayout(bar)
        bl.setContentsMargins(8, 3, 6, 3)
        head = QtWidgets.QLabel("Ion brightness")
        head.setProperty("role", "heading")
        bl.addWidget(head)
        bl.addWidget(_muted("relative to each ion's reference; a dark ion drops to ~0"))
        bl.addStretch(1)
        bl.addWidget(_muted("last"))
        bl.addWidget(self.trace_window)
        bl.addWidget(pause)
        bl.addWidget(clear)
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(bar)
        lay.addWidget(self.trace, 1)
        return w

    # -------------------------------------------------------------- tabs
    def _live_tab(self, disp):
        live = self.config.get("live", {})
        self.exp = _spin(1e-4, 60.0, live.get("exposure", 0.05), 0.01, " s", 4, width=110)
        self.exp.editingFinished.connect(lambda: self._cmd("set_live", exposure=self.exp.value()))
        self.gain = _spin(0, 1000, live.get("em_gain", 50), width=80)
        self.gain.editingFinished.connect(lambda: self._cmd("set_live", em_gain=self.gain.value()))
        self.binning = QtWidgets.QComboBox()
        self.binning.addItems(["1", "2", "4", "8"])
        self.binning.setCurrentText(str(live.get("hbin", 1)))
        self.binning.activated.connect(lambda _: self._cmd(
            "set_live", hbin=int(self.binning.currentText()), vbin=int(self.binning.currentText())))
        self.snap_n = _spin(1, 10000, 10, tip="frames averaged by Snap", width=80)
        acq = _group("Live video", [
            _row(_button("Start live", lambda: self._cmd("start_live"), "start live video "
                         "(F5)", "play", "primary"),
                 _button("Stop", lambda: self._cmd("stop"), "stop acquiring (F6)", "stop"),
                 None,
                 _button("Snap...", self._snap_save, "average this many frames with the live "
                         "settings and save them (Ctrl+Shift+S)", "camera"), self.snap_n,
                 "frames"),
            _grid([["exposure", self.exp, "EM gain", self.gain, "binning", self.binning]]),
        ])
        self.roi_label = _muted("full sensor")
        self.crop_margin = _spin(5, 500, 40, 5, " px", tip="margin around the ions", width=80)
        region = _group("Camera region (ROI)", [
            self.roi_label,
            _row(_tool(self.act["crop_view"]), _tool(self.act["crop_ions"]), self.crop_margin,
                 None, _tool(self.act["full_sensor"])),
        ])
        self.auto_levels = QtWidgets.QCheckBox("auto levels")
        self.auto_levels.setChecked(True)
        self.auto_levels.setToolTip("brightness levels follow every frame; off (or after "
                                    "dragging the histogram handles) they stay as set")
        self.auto_levels.toggled.connect(self.image.set_auto_levels)
        self.image.autoLevelsChanged.connect(self._auto_levels_changed)
        self.overlays = QtWidgets.QCheckBox("overlays")
        self.overlays.setChecked(True)
        self.overlays.setToolTip("everything drawn on the image (each kind in the bar above "
                                 "the image)")
        self.overlays.toggled.connect(self.act["overlays"].setChecked)
        self.rotation = QtWidgets.QComboBox()
        self.rotation.addItems(["0", "90", "180", "270"])
        self.rotation.setCurrentText(str(disp.rotation))
        self.flipx = QtWidgets.QCheckBox("flip x")
        self.flipx.setChecked(disp.flip_x)
        self.flipy = QtWidgets.QCheckBox("flip y")
        self.flipy.setChecked(disp.flip_y)
        self.rotation.activated.connect(lambda _: self._orientation())
        for w in (self.flipx, self.flipy):
            w.toggled.connect(lambda _: self._orientation())
        self.cmap = QtWidgets.QComboBox()
        self.cmap.addItems(["inferno", "viridis", "magma", "plasma", "cividis", "CET-L1",
                            "CET-L8", "CET-L16"])
        self.cmap.setCurrentText(disp.colormap)
        self.cmap.activated.connect(lambda _: self.image.set_colormap(self.cmap.currentText()))
        fit_levels = _button("Fit levels", self.image.fit_levels_once,
                             "set the levels to this frame once, then keep them")
        display = _group("Display (orientation is display-only)", [
            _grid([["colormap", self.cmap, self.auto_levels, fit_levels],
                   ["rotate (clockwise)", self.rotation, self.flipx, self.flipy],
                   [None, self.overlays]]),
        ])
        self._caps = self.info.get("capabilities") or {}
        self.amp = QtWidgets.QComboBox()
        self.amp.addItems(["em", "conventional"])
        self.amp.setCurrentText(live.get("amplifier", "em"))
        self.hs = QtWidgets.QComboBox()
        self.vs = QtWidgets.QComboBox()
        self.preamp = QtWidgets.QComboBox()
        self.ft = QtWidgets.QCheckBox("frame transfer")
        self.ft.setChecked(bool(live.get("frame_transfer", True)))
        self.ft.setToolTip("no dead time between frames (internal trigger)")
        self._fill_speeds(live)
        self.amp.activated.connect(lambda _: self._fill_speeds(None))
        adv = _group("Readout electronics (live)", [
            _grid([["amplifier", self.amp, "horizontal", self.hs],
                   ["pre-amp", self.preamp, "vertical shift", self.vs],
                   [None, self.ft]]),
            _row(None, _button("Apply", self._apply_electronics,
                               "restart live video with these settings")),
        ])
        return _scroll(_page(acq, region, display, adv, None))

    def _fill_speeds(self, live):
        """Speed / gain choices from the camera's capabilities (as reported)."""
        caps = self._caps
        amp = self.amp.currentText()
        cur_hs = self.hs.currentData() if live is None else live.get("hs_speed_mhz")
        self.hs.clear()
        for v in (caps.get("hs_speeds_mhz") or {}).get(amp, []) or [10.0, 5.0, 3.0, 1.0]:
            self.hs.addItem("{:g} MHz".format(v), float(v))
        if cur_hs is not None:
            i = min(range(self.hs.count()), key=lambda k: abs(self.hs.itemData(k) - cur_hs),
                    default=0)
            self.hs.setCurrentIndex(i)
        if live is not None:
            self.vs.clear()
            self.vs.addItem("fastest recommended", 0.0)
            for v in caps.get("vs_speeds_us") or []:
                self.vs.addItem("{:g} us".format(v), float(v))
            vs = live.get("vs_speed_us", 0.0) or 0.0
            self.vs.setCurrentIndex(max(0, self.vs.findData(float(vs))))
            self.preamp.clear()
            texts = caps.get("preamp_gain_texts") or []
            for k, g in enumerate(caps.get("preamp_gains") or [1.0, 2.0, 4.0]):
                self.preamp.addItem("{} ({:g}x)".format(texts[k], g) if k < len(texts)
                                    else "{:g}x".format(g), k)
            self.preamp.setCurrentIndex(min(int(live.get("preamp_gain_index", 0)),
                                            self.preamp.count() - 1))

    def _ions_tab(self):
        self.sites = SiteTable()
        self.tiles = {}
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(6)
        for k, (key, caption) in enumerate((
                ("bright", "bright ions"), ("dark", "dark ions"), ("state", "crystal"),
                ("stable", "stable for"), ("scale", "chain scale [px]"),
                ("angle", "chain angle [deg]"), ("focus", "PSF sigma [px]"),
                ("noise", "noise [counts]"))):
            t = StatTile(caption)
            self.tiles[key] = t
            grid.addWidget(t, k // 4, k % 4)
        self.crystal_label = _muted("", wrap=True)
        tracking = _group("Tracking", [
            _row(_tool(self.act["accept"]), _tool(self.act["forget"]), None),
            _row(_tool(self.act["hold"]), _tool(self.act["locate"]), None),
        ])
        self.bg_frames = _spin(5, 1000, 50, tip="frames (full sensor, live exposure and gain) "
                               "- more frames, less noise in the learned structure", width=80)
        self.bg_dark = QtWidgets.QCheckBox("ions dark / trap empty")
        self.bg_dark.setToolTip("repumper off (ions shelved, the laser still scatters) or no "
                                "ions: nothing needs masking - the most complete background. "
                                "Unchecked: the tracked ions are modelled and masked.")
        self.bg_label = _muted("background: none learned", wrap=True)
        background = _group("Learned background (static light of the trap surface)", [
            self.bg_label,
            _row(_button("Learn background", self._learn_background,
                         "learn the static light of the trap surface - laser scatter, "
                         "defects and dust (which can look like ions or clouds), hot pixels - "
                         "and subtract it from now on (Ctrl+B)", "layers", "primary"),
                 self.bg_frames, "frames", self.bg_dark, None,
                 _button("Clear", lambda: self._cmd("clear_background"),
                         "forget the learned background", "trash")),
        ])
        self.wait_n = _spin(1, 64, 1, width=64)
        loading = _group("Loading", [
            _row("alert when", self.wait_n, "ion(s) are trapped",
                 _button("Arm alert", self._wait_ions,
                         "beeps and pops up when this many ions are trapped"), None),
        ])
        w = _page(grid, self.crystal_label, self.sites, tracking, background, loading)
        w.layout().setStretchFactor(self.sites, 1)
        return w

    def _readout_tab(self):
        ro = self.config.get("readout", {})
        self.ro_chips = {k: style.chip(QtWidgets.QLabel(""))
                         for k in ("plan", "armed", "shots", "verified")}
        chips = _row(*self.ro_chips.values(), None)
        self.n_ions = _spin(0, 64, 0, special="any", width=70,
                            tip="how many ions to calibrate for (any: those found)")
        self.hbin = _spin(1, 16, ro.get("hbin", 4), width=56)
        self.vbin = _spin(1, 16, ro.get("vbin", 4), width=56)
        rs = (self.config.get("calibration", {}).get("roi_size") or [1, 1])
        self.roi_w = _spin(0, 12, int(rs[0]), width=64)
        self.roi_h = _spin(0, 12, int(rs[1]), width=64)
        for sp in (self.roi_w, self.roi_h):
            sp.setSpecialValueText("auto")
            sp.setToolTip("Grabber ROI per ion, in super-pixels.  1 x 1: one super-pixel per "
                          "ion, the grid placed so the ions sit at their centres - the fewest "
                          "pixels and so the fewest dark counts in each sum, usually the best "
                          "SPAM fidelity.  auto: as much of each ion's light as the grid "
                          "allows (bigger ROIs, more dark counts).")
        self.auto_bin = QtWidgets.QCheckBox("auto binning")
        self.auto_bin.setToolTip("the cheapest binning that meets the containment and "
                                 "crosstalk targets")
        self.verify = QtWidgets.QCheckBox("verify")
        self.verify.setChecked(True)
        self.verify.setToolTip("test frames with the final settings before arming")
        self.ro_exp = _spin(1e-5, 1.0, ro.get("exposure", 0.0015), 0.0005, " s", 5, width=110)
        self.ro_gain = _spin(0, 1000, ro.get("em_gain", 200), width=80)
        calibrate = _group("Calibrate and run", [
            _grid([["ions", self.n_ions, 12, "exposure", self.ro_exp],
                   ["binning", _row(self.hbin, "x", self.vbin, None), 12, "EM gain",
                    self.ro_gain],
                   ["ROI per ion", _row(self.roi_w, "x", self.roi_h, None), 12, self.auto_bin,
                    self.verify]]),
            _row(_button("Calibrate + arm", self._calibrate, "locate -> plan AQROI/ROIs -> "
                         "verify -> external trigger + Camera Link (Ctrl+K)", "check",
                         "primary"), _tool(self.act["locate"]), None,
                 _tool(self.act["arm"]), _tool(self.act["disarm"])),
        ])
        top = _page(chips, calibrate, self._manual_box())
        self.plan_text = QtWidgets.QPlainTextEdit()
        self.plan_text.setReadOnly(True)
        f = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        self.plan_text.setFont(f)
        self.plan_text.setPlainText(format_plan(None))
        self.snippet_text = QtWidgets.QPlainTextEdit()
        self.snippet_text.setReadOnly(True)
        self.snippet_text.setFont(f)
        plan_page = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(plan_page)
        pl.setContentsMargins(6, 6, 6, 6)
        pl.addWidget(self.plan_text, 1)
        pl.addLayout(_row(None, _tool(self.act["copy_plan"])))
        artiq_page = QtWidgets.QWidget()
        al = QtWidgets.QVBoxLayout(artiq_page)
        al.setContentsMargins(6, 6, 6, 6)
        al.addWidget(self.snippet_text, 1)
        al.addLayout(_row(_muted("grabber0.setup_roi(i, ROI_X0[i], ROI_Y0[i], ROI_X1[i], "
                                 "ROI_Y1[i]) for every ROI"), None,
                          _tool(self.act["copy_snippet"])))
        self.results = QtWidgets.QTabWidget()
        self.results.addTab(self._shots_box(), "Shots && fidelity")
        self.results.addTab(plan_page, "Plan")
        self.results.addTab(artiq_page, "ARTIQ")
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        split.addWidget(_scroll(top))
        split.addWidget(self.results)
        split.setStretchFactor(1, 1)
        split.setSizes([440, 480])
        self.readout_split = split
        return split

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
        self.hist_info = _muted("no shots yet", wrap=True)
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
        w = QtWidgets.QWidget()
        gl = QtWidgets.QVBoxLayout(w)
        gl.setContentsMargins(6, 6, 6, 6)
        bar = [None, _muted("prepare the ions bright in every shot")]
        if self.simulated:
            self.sim_shots = _spin(1, 100000, 300, 100, width=90,
                                   tip="simulated external trigger pulses")
            bar = [None, _button("Trigger", lambda: self._cmd(
                "sim_trigger", self.sim_shots.value(), 0.005), "simulated TTL pulses, 5 ms "
                "apart (the ions are bright in every shot)", "trigger"), self.sim_shots,
                "shots"]
        gl.addLayout(_row("shots", self.hist_source, self.hist_log, *bar))
        gl.addWidget(self.hist_info)
        gl.addWidget(self.hist, 1)
        gl.addWidget(self.hist_table)
        return w

    def _manual_box(self):
        """Adjusting the readout by hand: a D-pad moves the AQROI - and every
        Grabber ROI with it - on screen, re-planning with another binning or
        inside a given AQROI keeps the located ion positions."""
        self.nudge_step = _spin(1, 64, 1, suffix=" px", tip="step of the D-pad [sensor px]",
                                width=64)
        self.nudge_step.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        pad = QtWidgets.QGridLayout()
        pad.setSpacing(4)
        for r, c, right, up, name in ((0, 1, 0, 1, "up"), (1, 0, -1, 0, "left"),
                                      (1, 2, 1, 0, "right"), (2, 1, 0, -1, "down")):
            b = QtWidgets.QToolButton()
            b.setProperty("role", "dpad")
            b.setProperty("icon_name", name)
            b.setIcon(style.icon(name))
            b.setIconSize(QtCore.QSize(20, 20))
            b.setAutoRepeat(True)
            b.setAutoRepeatDelay(400)
            b.setAutoRepeatInterval(150)
            b.setFixedSize(36, 36)
            b.setToolTip("move the AQROI - and all ROI boxes with it - {} on screen by one "
                         "step: the ions move the other way inside their ROIs".format(name))
            b.clicked.connect(lambda _=False, r_=right, u_=up: self._nudge(r_, u_))
            pad.addWidget(b, r, c)
        pad.addWidget(self.nudge_step, 1, 1)
        self.offset_label = _muted("no plan yet", wrap=True)
        self.aq_edit = []
        for name in ("x0", "y0", "x1", "y1"):
            sp = _spin(0, 8192, 0, tip="AQROI {} [sensor px; x1, y1 exclusive]".format(name),
                       width=72)
            self.aq_edit.append(sp)
        self._nudge_pending = [0, 0]
        self._nudge_busy = False
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(6)
        right.addWidget(self.offset_label)
        right.addLayout(_row(
            _button("Re-plan", self._replan, "plan again with the binning and ROI size set "
                    "above, from the ions already located (no new locate)", "refresh"),
            _button("Verify", self._verify, "test frames with this plan: signal of every ion "
                    "in its ROI, and the shot histograms below with a quick fidelity "
                    "estimate (Ctrl+Shift+V)", "verify"),
            _button("Undo moves", self._undo_moves, "move the AQROI back to where the plan "
                    "put it", "undo")))
        right.addStretch(1)
        g = QtWidgets.QGroupBox("Adjust by hand (the D-pad moves the AQROI and all ROIs)")
        gl = QtWidgets.QVBoxLayout(g)
        top = QtWidgets.QHBoxLayout()
        top.addLayout(pad)
        top.addSpacing(8)
        top.addLayout(right, 1)
        gl.addLayout(top)
        gl.addLayout(_row("AQROI", *self.aq_edit, _button(
            "Use this AQROI", self._use_aqroi, "place the ROIs inside this AQROI (sensor px, "
            "x1/y1 exclusive; its size a multiple of the binning)"), None))
        return g

    def _events_tab(self):
        self.ev_filter = QtWidgets.QComboBox()
        self.ev_filter.addItems(["all", "warnings and alarms", "alarms"])
        self.ev_filter.activated.connect(lambda i: self.events.set_filter(min_severity=i))
        self.ev_search = QtWidgets.QLineEdit()
        self.ev_search.setPlaceholderText("search...")
        self.ev_search.setClearButtonEnabled(True)
        self.ev_search.textChanged.connect(lambda t: self.events.set_filter(text=t))
        clear = _button("Clear", self.events.clear_all, "clear the list (the server keeps "
                        "its log)", "trash")
        w = _page(_row("show", self.ev_filter, self.ev_search, None, clear,
                       _tool(self.act["save_events"], False)), self.events)
        w.layout().setStretchFactor(self.events, 1)
        return w

    def _camera_tab(self):
        self.info_text = QtWidgets.QPlainTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setPlainText(_info_text(self.info))
        self.setpoint = _spin(-100, 20, int(self.config.get("camera", {}).get(
            "target_temperature", -60)), suffix=" °C", width=90)
        self.temp_tile = StatTile("sensor temperature")
        cooling = _group("Sensor cooling", [
            _row(self.temp_tile, None),
            _row("set point", self.setpoint,
                 _button("Set", lambda: self._cmd("set_temperature", self.setpoint.value())),
                 None, _tool(self.act["cooler_on"]), _tool(self.act["cooler_off"])),
        ])
        self.lease_label = _muted("nobody has taken control", wrap=True)
        control = _group("Control (advisory lease shown to all clients)", [
            self.lease_label,
            _row(_tool(self.act["take"]), _tool(self.act["release"]), None),
        ])
        configuration = _group("Configuration", [
            _muted("Every option can be changed at run time; write a file to keep them "
                   "(start with -c <file>).", wrap=True),
            _row(_tool(self.act["settings"]), _tool(self.act["save_config"]), None),
        ])
        w = _page(cooling, control, configuration,
                  _group("Camera", [self.info_text]))
        return _scroll(w)

    def _sim_tab(self):
        self.sim_idx = _spin(0, 63, 0, width=64)
        self.sim_n = _spin(1, 20, 1, width=64)
        c = self._cmd
        ions = _group("Ions", [
            _row(_button("Load", lambda: c("sim_load_ion", self.sim_n.value(), 1.0),
                         "load ions (a hot cloud first)"), self.sim_n, "ion(s)", None,
                 _button("Lose random ion", lambda: c("sim_lose_ion", -1))),
            _row("ion", self.sim_idx,
                 _button("dark", lambda: c("sim_set_dark", self.sim_idx.value(), True)),
                 _button("bright", lambda: c("sim_set_dark", self.sim_idx.value(), False)),
                 _button("lose", lambda: c("sim_lose_ion", self.sim_idx.value())), None),
            _row(_button("Melt 1 s", lambda: c("sim_melt", 1.0)), None,
                 "shift the chain",
                 _button("-5 px", lambda: c("sim_shift", -5.0, 0.0)),
                 _button("+5 px", lambda: c("sim_shift", 5.0, 0.0))),
            _row("rotate the trap", *[_button(
                "{} deg".format(a), lambda _=False, a_=a: self._set_sim(trap_angle_deg=a_),
                "turn the whole trap in the image (ions, electrodes, surface, laser beam) - "
                "then learn the background again") for a in (0, 45, 90)], None),
        ])
        self.trig_n = _spin(1, 100000, 100, 100, width=90)
        self.trig_dt = _spin(0.0, 1.0, 0.005, 0.001, " s", 3, tip="time between pulses",
                             width=90)
        lab = _group("Lab", [
            _row(_button("Lasers off", lambda: c("sim_set_lasers", False)),
                 _button("Lasers on", lambda: c("sim_set_lasers", True)), None,
                 _button("Repumper off (ions dark)", lambda: c("sim_set_repumper", False)),
                 _button("Repumper on", lambda: c("sim_set_repumper", True))),
            _row(_button("Oven on (loading)", lambda: c("sim_set_oven", True, 0.3)),
                 _button("Oven off", lambda: c("sim_set_oven", False)), None),
            _row("move camera",
                 _button("-5 px x", lambda: c("sim_move_camera", -5.0, 0.0)),
                 _button("+5 px x", lambda: c("sim_move_camera", 5.0, 0.0)),
                 _button("-5 px y", lambda: c("sim_move_camera", 0.0, -5.0)),
                 _button("+5 px y", lambda: c("sim_move_camera", 0.0, 5.0)), None),
            _row(_button("External triggers", lambda: c("sim_trigger", self.trig_n.value(),
                                                         self.trig_dt.value()),
                         "TTL pulses as ARTIQ would send them", "trigger"),
                 self.trig_n, "every", self.trig_dt, None),
        ])

        # surface, optics and light: the realism knobs
        self.sim_preset = QtWidgets.QComboBox()
        self.sim_preset.addItems(["clean", "typical", "dirty", "horrendous"])
        self.sim_preset.activated.connect(lambda _: self._set_sim(
            surface=self.sim_preset.currentText()))
        self.sim_fields = {}
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        form.addRow("surface preset", _row(self.sim_preset, _button(
            "New surface", lambda: self._set_sim(
                surface_seed=int(self._sim_params.get("surface_seed", 1)) + 1),
            "same kind of surface, different defects"), None))
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
        self.sim_info = _muted("", wrap=True)
        optics = QtWidgets.QGroupBox("Surface, optics and light")
        ol = QtWidgets.QVBoxLayout(optics)
        ol.addLayout(form)
        ol.addWidget(self.sim_info)
        self._sim_params = {}
        self.backend.call("sim_get_params", callback=self._got_sim_params)
        return _scroll(_page(_muted("Training / test: change the simulated trap and watch how "
                                    "ioncam reacts.", wrap=True), ions, lab, optics, None))

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

    # ------------------------------------------------------------ appearance
    def set_theme(self, name):
        """Switch between the "dark" and the "light" theme."""
        style.apply_theme(QtWidgets.QApplication.instance(), name)
        for obj, icon_name, kind in self._themed:
            obj.setIcon(style.icon(icon_name, _action_icon_color(kind)))
        for b in self.findChildren(QtWidgets.QAbstractButton):
            name_ = b.property("icon_name")
            if name_:
                b.setIcon(style.icon(name_, _icon_color(b.property("kind"))))
        self.image.apply_theme()
        self.trace.apply_theme()
        self.hist.apply_theme()
        self.events.refill()
        self.act["theme"].blockSignals(True)
        self.act["theme"].setChecked(name == "light")
        self.act["theme"].blockSignals(False)
        if self._status:
            self.sites.set_sites(self._status.get("sites", []))
        if self._hist_key is not None and self.hist._last is not None:
            self.hist_table.setText(fidelity_table(self.hist._last))
        if self.settings is not None:
            self.settings.setValue("theme", name)

    def _restore_layout(self):
        s = self.settings
        if s is None:
            return
        try:
            geo = s.value("geometry")
            if geo is not None:
                self.restoreGeometry(geo)
            for key, sp in (("split", self.split), ("left", self.left),
                            ("readout_split", self.readout_split)):
                v = s.value(key)
                if v is not None:
                    sp.restoreState(v)
            self.tabs.setCurrentIndex(int(s.value("tab", 0)))
        except Exception:  # pragma: no cover - a stale settings file
            pass

    def _save_layout(self):
        s = self.settings
        if s is None:
            return
        s.setValue("geometry", self.saveGeometry())
        s.setValue("split", self.split.saveState())
        s.setValue("left", self.left.saveState())
        s.setValue("readout_split", self.readout_split.saveState())
        s.setValue("tab", self.tabs.currentIndex())

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
            self.msg_label.setText("no ions to crop around")
            return
        xs = [s["x"] for s in sites]
        ys = [s["y"] for s in sites]
        m = self.crop_margin.value()
        self._cmd("set_roi", int(min(xs) - m), int(min(ys) - m), int(max(xs) + m) + 1,
                  int(max(ys) + m) + 1)

    def _apply_electronics(self):
        self._cmd("set_live", amplifier=self.amp.currentText(),
                  hs_speed_mhz=float(self.hs.currentData() or 10.0),
                  vs_speed_us=float(self.vs.currentData() or 0.0),
                  preamp_gain_index=int(self.preamp.currentData() or 0),
                  frame_transfer=self.ft.isChecked())

    def _toggle_overlays(self, on):
        self.image.show_overlays = on
        for w in (self.overlays,):
            if w.isChecked() != on:
                w.blockSignals(True)
                w.setChecked(on)
                w.blockSignals(False)
        self.image.set_sites(*self.image._sites)
        self.image.set_plan(self._plan)

    def _set_overlay(self, key, on):
        for w in (self.overlay_buttons.get(key), self.overlay_actions.get(key)):
            if w is not None and w.isChecked() != on:
                w.blockSignals(True)
                w.setChecked(on)
                w.blockSignals(False)
        self.image.set_overlay(key, on)

    def _zoom_ions(self):
        sites = self._status.get("sites") or []
        if not self.image.zoom_to([(s["x"], s["y"]) for s in sites]):
            self.msg_label.setText("no ions to zoom to")

    def _fullscreen(self, on):
        if on:
            self.showFullScreen()
        else:
            self.showNormal()

    def _orientation(self):
        self.image.set_orientation(int(self.rotation.currentText()), self.flipx.isChecked(),
                                   self.flipy.isChecked())

    def _hold(self, on):
        if on:
            self._cmd("hold_tracking", "moved from GUI", 600)
        else:
            self._cmd("resume_tracking")

    def _locate(self):
        self._cmd("locate_ions", self.n_ions.value(), callback=self._located)

    def _calibrate(self):
        kw = dict(n_ions=self.n_ions.value(), hbin=self.hbin.value(), vbin=self.vbin.value(),
                  roi_size=self._roi_size(), auto_binning=self.auto_bin.isChecked(),
                  verify=self.verify.isChecked(), exposure=self.ro_exp.value(),
                  em_gain=self.ro_gain.value())
        self._cmd("calibrate_readout", callback=self._got_plan, **kw)

    def _verify(self):
        self._cmd("verify_readout", callback=self._got_plan)

    def _located(self, res):
        warn = "".join("\n! " + w for w in res.get("warnings", []))
        QtWidgets.QMessageBox.information(
            self, "ioncam", "Located {} ion(s)\nPSF sigma {} px, halo {}\npositions:\n{}{}".format(
                res["n_ions"], res["psf_sigma"], res.get("psf_halo"),
                "\n".join("  {:.3f}, {:.3f}  ({})".format(p[0], p[1], s)
                          for p, s in zip(res["positions"], res["states"])), warn))

    def _got_plan(self, plan):
        self._plan = plan
        self._plan_created = plan.get("created")
        self.plan_text.setPlainText(format_plan(plan))
        self.snippet_text.setPlainText(artiq_snippet(plan))
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

    def _undo_moves(self):
        off = (self._plan or {}).get("manual_offset") or [0, 0]
        if not any(off):
            self.msg_label.setText("the AQROI is where the plan put it")
            return
        self._nudge_pending[0] -= int(off[0])
        self._nudge_pending[1] -= int(off[1])
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
        if not self._plan:
            self.msg_label.setText("no readout plan yet")
            return
        QtWidgets.QApplication.clipboard().setText(artiq_snippet(self._plan))
        self.msg_label.setText("ARTIQ snippet copied")

    def _copy_plan_json(self):
        if not self._plan:
            self.msg_label.setText("no readout plan yet")
            return
        QtWidgets.QApplication.clipboard().setText(json.dumps(self._plan, indent=1, default=str))
        self.msg_label.setText("readout plan copied as JSON")

    def _wait_ions(self):
        n = self.wait_n.value()

        def done(res):
            if res.get("ok"):
                QtWidgets.QApplication.beep()
                QtWidgets.QMessageBox.information(self, "ioncam",
                                                  "{} ion(s) trapped!".format(res["n_bright"]))
        self.backend.call("wait_for_ions", n, 3600.0, True, callback=done, channel="wait")
        self.msg_label.setText("waiting for {} ion(s)...".format(n))

    # -- files -------------------------------------------------------------------------
    def _ask_save(self, caption, name, filters):
        path, flt = QtWidgets.QFileDialog.getSaveFileName(self, caption, name, filters)
        return path, flt

    def _save_image(self):
        if self.image._last is None:
            self.msg_label.setText("no image yet")
            return
        path, flt = self._ask_save("Save image", "ioncam_{}.npy".format(
            time.strftime("%Y%m%d-%H%M%S")), "NumPy array (*.npy);;CSV table (*.csv);;"
            "PNG of the view (*.png)")
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".png" if "PNG" in flt else ".csv" if "CSV" in flt else ".npy"
        try:
            self.msg_label.setText("saved {}".format(self.save_image(path)))
        except OSError as e:
            self._error("save image", e)

    def save_image(self, path):
        """Save the image on screen: ``.npy`` / ``.csv`` the raw data (with a
        ``.json`` of its geometry and camera settings beside it), ``.png`` the
        view as shown.  Returns the path."""
        path = str(path)
        img, roi, hb, vb = self.image._last
        ext = os.path.splitext(path)[1].lower()
        if ext == ".png":
            self.image.plot.grab().save(path)
            return path
        if ext == ".csv":
            np.savetxt(path, np.asarray(img), fmt="%.6g", delimiter=",")
        else:
            np.save(path, np.asarray(img))
        meta = dict(self._frame_meta, roi=list(roi), hbin=hb, vbin=vb,
                    saved=time.strftime("%Y-%m-%d %H:%M:%S"),
                    orientation="raw sensor (rows = y, columns = x)")
        with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1, default=str)
        return path

    def _snap_save(self):
        n = self.snap_n.value()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path, _ = self._ask_save("Snap {} frame(s) and save the mean".format(n),
                                 "ioncam_snap_{}.npy".format(stamp), "NumPy array (*.npy)")
        if not path:
            return
        if not path.lower().endswith(".npy"):
            path += ".npy"

        def done(res):
            try:
                np.save(path, np.asarray(res["image"]))
                meta = {k: v for k, v in res.items() if k != "image"}
                meta["saved"] = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(path[:-4] + ".json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=1, default=str)
                self.msg_label.setText("saved the mean of {} frame(s) to {}".format(
                    res.get("n_frames"), path))
            except OSError as e:
                self._error("snap", e)
        self._cmd("snap", n, callback=done)

    def _save_events(self):
        path, _ = self._ask_save("Save events", "ioncam_events_{}.txt".format(
            time.strftime("%Y%m%d-%H%M%S")), "Text (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.events.as_text() + "\n")
            self.msg_label.setText("saved {}".format(path))
        except OSError as e:
            self._error("save events", e)

    def _save_config(self):
        if self.backend.remote:
            path, ok = QtWidgets.QInputDialog.getText(
                self, "Save configuration", "File on the camera PC:", text="ioncam.toml")
            if not ok or not path:
                return
        else:
            path, _ = self._ask_save("Save configuration", "ioncam.toml", "TOML (*.toml)")
            if not path:
                return
        self._cmd("save_config", path, callback=lambda _: self.msg_label.setText(
            "configuration written to {}".format(path)))

    # -- dialogs -----------------------------------------------------------------------
    def _settings_dialog(self):
        def show(cfg_help):
            def got(cfg):
                self.config = cfg
                ConfigDialog(self, cfg, cfg_help).exec()
            self.backend.call("get_config", callback=got)

        # an older server has no help texts: open the dialog without them
        self.backend.call("get_config_help", callback=show, error=lambda n, e: show({}))

    def _shortcuts_dialog(self):
        rows = [(a.text().replace("&", "").replace("...", ""), a.shortcut().toString())
                for a in self.act.values() if not a.shortcut().isEmpty()]
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Keyboard shortcuts")
        table = QtWidgets.QTableWidget(len(rows), 2)
        table.setHorizontalHeaderLabels(["action", "shortcut"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        for r, (name, key) in enumerate(rows):
            table.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
            table.setItem(r, 1, QtWidgets.QTableWidgetItem(key))
        table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(table)
        dlg.resize(420, 520)
        dlg.exec()

    def _about(self):
        QtWidgets.QMessageBox.about(
            self, "About ioncam",
            "<b>ioncam</b> {}<br>the camera brain of an ion trap<br><br>camera: {}<br>{}".format(
                self.info.get("ioncam_version", ""), self.info.get("model", "?"),
                self.backend.describe()))

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
            self._frame_meta = {k: fr.get(k) for k in ("index", "t", "tag", "exposure",
                                                       "em_gain", "view")}
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
        self._update_controls(st)
        page = self.tabs.currentWidget()
        if page is self.ions_page:
            self._update_ions(st, sites)
        elif page is self.live_page:
            self._update_live(st)
        elif page is self.camera_page:
            self._update_camera(st)

    def _update_controls(self, st):
        """Which actions make sense now; the readout's state chips; the hold."""
        busy = st.get("procedure")
        ro = st.get("readout") or {}
        armed, has_plan = bool(ro.get("armed")), bool(ro.get("has_plan"))
        state = (bool(busy), armed, has_plan, bool((st.get("ions") or {}).get("hold")))
        if state != self._busy:
            self._busy = state
            a = self.act
            for k in ("locate", "calibrate", "snap", "learn_bg", "live"):
                a[k].setEnabled(not busy)
            for k in ("verify", "arm"):
                a[k].setEnabled(has_plan and not busy)
            a["disarm"].setEnabled(armed)
            a["abort"].setEnabled(bool(busy))
            a["hold"].blockSignals(True)
            a["hold"].setChecked(state[3])
            a["hold"].blockSignals(False)
            self.busy_bar.setVisible(bool(busy))
        if self.tabs.currentWidget() is self.readout_page or self._tick % 10 == 0:
            ch = self.ro_chips
            valid = ro.get("plan_valid")
            style.chip(ch["plan"], "good" if valid else ("bad" if has_plan else ""))
            ch["plan"].setText("plan valid" if valid else
                               ("plan outdated - recalibrate" if has_plan else "no plan"))
            style.chip(ch["armed"], "accent" if armed else "")
            ch["armed"].setText("armed: external trigger" if armed else "not armed")
            ch["shots"].setText("{} shots".format(ro.get("shots", 0)))
            ver = ro.get("verified")
            ch["verified"].setText("verified {}".format(time.strftime(
                "%H:%M:%S", time.localtime(ver))) if ver else "not verified")
            style.chip(ch["verified"], "good" if ver else "")

    def _update_ions(self, st, sites):
        self.sites.set_sites(sites)
        ions = st["ions"]
        chain = ions.get("chain") or {}
        det = st.get("detection") or {}
        t = self.tiles
        t["bright"].set(ions.get("n_bright", 0))
        t["dark"].set(ions.get("n_dark", 0))
        t["state"].set(ions.get("state") or "-")
        t["stable"].set("{:.0f} s".format(ions.get("stable_for") or 0) if ions.get("stable")
                        else "-")
        def num(v, fmt):
            return fmt.format(v) if isinstance(v, (int, float)) else "-"
        t["scale"].set(num(chain.get("scale_px"), "{:.2f}"))
        t["angle"].set(num(chain.get("angle_deg"), "{:.1f}"))
        t["focus"].set(num(det.get("focus"), "{:.2f}"))
        t["noise"].set(num(det.get("noise"), "{:.1f}"))
        self.crystal_label.setText("background {} counts · {} spot(s), {} cloud(s) in the "
                                   "last frame · analysis {} ms".format(
                                       det.get("background", "-"), ions.get("n_spots", 0),
                                       ions.get("n_clouds", 0), det.get("analysis_ms", "-")))

    def _update_live(self, st):
        ls = st.get("live_settings") or {}
        roi = ls.get("roi")
        shape = (self.info.get("sensor_shape") or [512, 512])
        if roi:
            self.roi_label.setText("x {}-{}, y {}-{}  ({} x {} px)".format(
                roi[0], roi[2], roi[1], roi[3], roi[2] - roi[0], roi[3] - roi[1]))
        else:
            self.roi_label.setText("full sensor ({} x {} px)".format(shape[1], shape[0]))

    def _update_camera(self, st):
        t, ts = st.get("temperature"), st.get("temperature_status", "")
        self.temp_tile.set("{} °C".format("--" if t is None else "{:.1f}".format(t)))
        self.temp_tile.caption.setText("sensor temperature · {} (set point {} °C)".format(
            ts.replace("_", " "), st.get("target_temperature")))
        c = st.get("control")
        self.lease_label.setText("controlled by '{}': {}".format(c["owner"], c.get("reason", ""))
                                 if c else "nobody has taken control")

    def _slow_tick(self):
        self.trace.redraw()
        if self.tabs.currentWidget() is self.readout_page:
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
        self._save_layout()
        self.timer.stop()
        self.slow.stop()
        super().closeEvent(ev)


def _info_text(info):
    """Camera information, one item a line (capabilities spelled out)."""
    lines = []
    for k, v in info.items():
        if k == "capabilities" and isinstance(v, dict):
            for ck, cv in v.items():
                if isinstance(cv, dict):
                    for sub, vals in cv.items():
                        lines.append("{} ({}): {}".format(ck.replace("_", " "), sub, vals))
                else:
                    lines.append("{}: {}".format(ck.replace("_", " "), cv))
        else:
            lines.append("{}: {}".format(k.replace("_", " "), v))
    return "\n".join(lines)


def _format_value(v):
    if isinstance(v, str):
        return v
    return json.dumps(v)


class ConfigDialog(QtWidgets.QDialog):
    """Every configuration option, with its help: change options at run time
    (Apply - not saved) or write them all to a configuration file."""

    def __init__(self, win, config, cfg_help):
        super().__init__(win)
        self.win = win
        self.config = config
        self.setWindowTitle("ioncam settings")
        self.resize(760, 680)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("find an option...")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["option", "value"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree.itemDoubleClicked.connect(self._edit)
        self._loading = True
        self.tree.itemChanged.connect(self._changed)
        bold = QtGui.QFont()
        bold.setBold(True)
        for section, values in config.items():
            h = cfg_help.get(section, {})
            top = QtWidgets.QTreeWidgetItem([section, ""])
            top.setFont(0, bold)
            top.setToolTip(0, h.get("", ""))
            top.setFirstColumnSpanned(False)
            self.tree.addTopLevelItem(top)
            for key, value in values.items():
                it = QtWidgets.QTreeWidgetItem([key, _format_value(value)])
                it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
                it.setData(0, QtCore.Qt.ItemDataRole.UserRole, (section, key, value))
                for col in (0, 1):
                    it.setToolTip(col, h.get(key, ""))
                top.addChild(it)
        self._loading = False
        self.tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.info = _muted("Double-click a value to change it; Apply sets the changed options "
                           "now (not saved - Save to file... writes them all).", wrap=True)
        apply_ = _button("Apply changes", self._apply, icon="check", kind="primary")
        save = _button("Save to file...", win._save_config, icon="save")
        close = _button("Close", self.reject)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.search)
        lay.addWidget(self.tree, 1)
        lay.addWidget(self.info)
        lay.addLayout(_row(None, save, apply_, close))

    def _edit(self, item, column):
        if item.parent() is not None:
            self.tree.editItem(item, 1)

    def _changed(self, item, column):
        if self._loading or column != 1 or item.parent() is None:
            return
        section, key, value = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        changed = item.text(1) != _format_value(value)
        f = item.font(1)
        f.setBold(changed)
        item.setFont(1, f)

    def _filter(self, text):
        text = text.lower()
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            any_shown = False
            for k in range(top.childCount()):
                it = top.child(k)
                shown = not text or text in it.text(0).lower() or \
                    text in it.toolTip(0).lower() or text in top.text(0).lower()
                it.setHidden(not shown)
                any_shown |= shown
            top.setHidden(not any_shown)
            top.setExpanded(bool(text) and any_shown)

    def _parse(self, text, old):
        if isinstance(old, str):
            return text
        try:
            return json.loads(text)
        except ValueError:
            raise ValueError("{!r} is not a valid value (numbers, true / false, [lists], "
                             "null)".format(text))

    def _apply(self):
        todo, errors = [], []
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            for k in range(top.childCount()):
                it = top.child(k)
                section, key, old = it.data(0, QtCore.Qt.ItemDataRole.UserRole)
                if it.text(1) == _format_value(old):
                    continue
                try:
                    todo.append((it, section, key, self._parse(it.text(1), old)))
                except ValueError as e:
                    errors.append("{}.{}: {}".format(section, key, e))
        if errors:
            QtWidgets.QMessageBox.warning(self, "ioncam", "\n".join(errors))
            return
        if not todo:
            self.info.setText("nothing changed")
            return

        def ok(it, section, key):
            def done(stored):
                it.setData(0, QtCore.Qt.ItemDataRole.UserRole, (section, key, stored))
                self._loading = True
                it.setText(1, _format_value(stored))
                f = it.font(1)
                f.setBold(False)
                it.setFont(1, f)
                self._loading = False
                self.info.setText("set {}.{} = {}".format(section, key, _format_value(stored)))
            return done

        def failed(name, exc, section=None, key=None):
            QtWidgets.QMessageBox.warning(self, "ioncam", "{}: {}".format(name, exc))
        for it, section, key, value in todo:
            self.win.backend.call("set_config", section, key, value,
                                  callback=ok(it, section, key), error=failed)


class _Display:
    """Display settings from the config dict (with defaults)."""

    def __init__(self, d):
        self.rotation = int(d.get("rotation", 90))
        self.flip_x = bool(d.get("flip_x", False))
        self.flip_y = bool(d.get("flip_y", False))
        self.colormap = d.get("colormap", "inferno")
        self.level_percentiles = tuple(d.get("level_percentiles", (1.0, 99.95)))
        self.max_fps = float(d.get("max_fps", 25.0))
