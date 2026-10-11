"""GUI widgets (pyqtgraph; works with PyQt5, PyQt6, PySide6)."""

import math
import time
from collections import deque

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

STATE_COLORS = {
    "stable": "#1b8a3a", "empty": "#555555", "changing": "#c77700", "melted": "#b03a2e",
    "no_fluorescence": "#b03a2e", "hold": "#2e5fa8",
}
SEVERITY_COLORS = {"info": "#1b6b2f", "warning": "#a65f00", "alarm": "#b0201a"}
TEMP_COLORS = {"stabilized": "#1b8a3a", "not_stabilized": "#c77700", "not_reached": "#c77700",
               "drift": "#c77700", "off": "#777777"}
ION_COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
              (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]


class _SensorGroup(pg.ItemGroup):
    """Group in sensor coordinates; its bounding rect is the image area so the
    ViewBox auto-range (incl. the 'A' button) works through the display transform."""

    def __init__(self):
        super().__init__()
        self._rect = QtCore.QRectF()
        # ItemGroup is flagged "no contents", which makes ViewBox ignore it when
        # auto-ranging; it has contents (the image), so clear the flag.
        self.setFlag(self.GraphicsItemFlag.ItemHasNoContents, False)

    def set_rect(self, rect):
        self.prepareGeometryChange()
        self._rect = QtCore.QRectF(rect)

    def boundingRect(self):
        return self._rect


def _lighter(color, f=0.35):
    """``color`` mixed with white (labels stay readable on a dark image)."""
    return tuple(int(c + (255 - c) * f) for c in color[:3])


def _pen(color, width=1.5, style=None):
    p = pg.mkPen(color, width=width)
    p.setCosmetic(True)
    if style is not None:
        p.setStyle(style)
    return p


class ImagePanel(QtWidgets.QWidget):
    """Live image with ion overlays, drawn in raw sensor coordinates.

    Everything is a child of one group item whose transform implements the
    *display* orientation (rotation/flips) - overlays and ROIs can never get
    out of sync with the image, and the underlying geometry stays raw."""

    hovered = QtCore.Signal(str)
    autoLevelsChanged = QtCore.Signal(bool)

    def __init__(self, display_cfg, parent=None):
        super().__init__(parent)
        self.cfg = display_cfg
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.setMenuEnabled(True)
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")
        self.vb = self.plot.getViewBox()
        self.vb.invertY(False)
        self.group = _SensorGroup()
        self.plot.addItem(self.group)
        self.img = pg.ImageItem()
        self.img.setParentItem(self.group)
        self.hist = pg.HistogramLUTWidget(image=self.img)
        self.hist.setMaximumWidth(110)
        lay.addWidget(self.plot, 1)
        # brightness levels: automatic (follow every frame) or set by hand -
        # dragging the histogram handles switches to "by hand" and the levels
        # then stay exactly where they were put
        self.auto_btn = QtWidgets.QToolButton()
        self.auto_btn.setText("Auto")
        self.auto_btn.setCheckable(True)
        self.auto_btn.setChecked(True)
        self.auto_btn.setToolTip("Auto: the brightness levels follow every frame.\n"
                                 "Off: they stay where you put them (dragging the histogram "
                                 "handles switches Auto off).")
        self.auto_btn.toggled.connect(self.set_auto_levels)
        fit = QtWidgets.QToolButton()
        fit.setText("Fit")
        fit.setToolTip("fit the levels to the current frame once, then keep them")
        fit.clicked.connect(self.fit_levels_once)
        side = QtWidgets.QVBoxLayout()
        side.setContentsMargins(0, 0, 0, 0)
        side.addWidget(self.hist, 1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.auto_btn)
        row.addWidget(fit)
        side.addLayout(row)
        lay.addLayout(side)
        self._levels_busy = False          # the levels are being set by this code
        self.hist.item.sigLevelsChanged.connect(self._levels_by_hand)
        self.hist.item.sigLevelChangeFinished.connect(self._levels_by_hand)
        menu = self.hist.item.vb.menu      # right click on the histogram
        if menu is not None:
            menu.addSeparator()
            self.auto_action = menu.addAction("Auto levels (follow every frame)")
            self.auto_action.setCheckable(True)
            self.auto_action.setChecked(True)
            self.auto_action.toggled.connect(self.set_auto_levels)
            menu.addAction("Fit levels to this frame once").triggered.connect(
                self.fit_levels_once)
        else:  # pragma: no cover
            self.auto_action = None
        self.set_colormap(display_cfg.colormap)

        self.sc_bright = pg.ScatterPlotItem(symbol="o", size=16, pen=_pen("#3cff6e", 2),
                                            brush=None, pxMode=True)
        self.sc_dark = pg.ScatterPlotItem(symbol="o", size=16,
                                          pen=_pen("#ff4040", 2, QtCore.Qt.PenStyle.DashLine),
                                          brush=None, pxMode=True)
        self.sc_spots = pg.ScatterPlotItem(symbol="+", size=7, pen=_pen("#fff200", 1),
                                           brush=None, pxMode=True)
        self.sc_clouds = pg.ScatterPlotItem(symbol="o", size=34, pen=_pen("#ff9f1a", 2),
                                            brush=None, pxMode=True)
        for it in (self.sc_bright, self.sc_dark, self.sc_spots, self.sc_clouds):
            it.setParentItem(self.group)
        self._labels = []
        self._rects = []
        self._roi_items = []
        self._plan = None
        self.auto_levels = True
        self.show_overlays = True
        self._geometry = None
        self._last = None
        self.set_orientation(display_cfg.rotation, display_cfg.flip_x, display_cfg.flip_y)
        self.vb.sigResized.connect(self._update_limits)
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)

    # -- appearance --------------------------------------------------------------------
    def set_colormap(self, name):
        try:
            cmap = pg.colormap.get(name)
        except Exception:
            cmap = pg.colormap.get("viridis")
        self.hist.gradient.setColorMap(cmap)

    def set_orientation(self, rotation, flip_x=False, flip_y=False):
        """Display-only rotation (degrees clockwise on screen) and flips."""
        t = QtGui.QTransform()
        t.rotate(-float(rotation))          # view has y up: negative = clockwise
        t.scale(-1.0 if flip_x else 1.0, -1.0 if flip_y else 1.0)
        self.group.setTransform(t)
        self.rotation, self.flip_x, self.flip_y = rotation, flip_x, flip_y
        for label in getattr(self, "_labels", []):
            label.setAnchor(self._label_anchor())
        if getattr(self, "_plan", None):
            self.set_plan(self._plan)          # ROI labels sit at the on-screen corner
        self._update_limits()
        self.vb.autoRange()

    def _update_limits(self, *_):
        """Zooming out and panning stay around the image (at most ~1.3x the
        view that just fits it, for the widget's current aspect ratio)."""
        rect = self.group.boundingRect()
        if rect.isEmpty():
            return
        r = self.group.mapRectToParent(rect)          # view coordinates (display transform)
        vw, vh = max(self.vb.width(), 1.0), max(self.vb.height(), 1.0)
        per_px = 1.3 * max(r.width() / vw, r.height() / vh)
        xr, yr = per_px * vw + 8.0, per_px * vh + 8.0
        c = r.center()
        self.vb.setLimits(xMin=c.x() - xr / 2, xMax=c.x() + xr / 2, yMin=c.y() - yr / 2,
                          yMax=c.y() + yr / 2, maxXRange=xr, maxYRange=yr,
                          minXRange=4.0, minYRange=4.0)

    # -- brightness levels -----------------------------------------------------------------
    def set_auto_levels(self, on):
        """Levels follow every frame (on) or stay as set by hand (off)."""
        on = bool(on)
        for w in (self.auto_btn, self.auto_action):
            if w is not None and w.isChecked() != on:
                w.blockSignals(True)
                w.setChecked(on)
                w.blockSignals(False)
        if on == self.auto_levels:
            return
        self.auto_levels = on
        if on and self._last is not None:
            self._apply_auto_levels(self._last[0])
        self.autoLevelsChanged.emit(on)

    def fit_levels_once(self):
        """Fit the levels to the current frame, then keep them."""
        self.set_auto_levels(False)
        if self._last is not None:
            self._apply_auto_levels(self._last[0])

    def _levels_by_hand(self, *_):
        if not self._levels_busy and self.auto_levels:
            self.set_auto_levels(False)        # the user moved the levels: keep them

    def _auto_levels_for(self, img):
        lo_p, hi_p = self.cfg.level_percentiles
        sample = img[::max(1, img.shape[0] // 256), ::max(1, img.shape[1] // 256)]
        lo, hi = np.percentile(sample, [lo_p, hi_p])
        hi = max(hi, float(img.max()) * 0.8) if hi_p >= 99.9 else hi
        if hi <= lo:
            hi = lo + 1
        return float(lo), float(hi)

    def _apply_auto_levels(self, img):
        if not np.asarray(img).size:
            return
        lo, hi = self._auto_levels_for(img)
        self._levels_busy = True
        try:
            self.img.setLevels((lo, hi))
            self.hist.item.setLevels(lo, hi)
        finally:
            self._levels_busy = False

    # -- data ------------------------------------------------------------------------------
    def set_frame(self, image, roi, hbin, vbin):
        img = np.asarray(image)
        geom = (tuple(roi), hbin, vbin, img.shape)
        # levels set by hand stay put: only automatic levels are computed here
        levels = self._auto_levels_for(img) if img.size and (
            self.auto_levels or self.img.levels is None) else None
        self._levels_busy = True
        try:
            if levels is None:
                self.img.setImage(img, autoLevels=False)
            else:
                self.img.setImage(img, autoLevels=False, levels=levels)
        finally:
            self._levels_busy = False
        if geom != self._geometry:
            rect = QtCore.QRectF(roi[0] - 0.5, roi[1] - 0.5, img.shape[1] * hbin,
                                 img.shape[0] * vbin)
            self.img.setRect(rect)
            self.group.set_rect(rect)
            self._update_limits()
            if self._geometry is None or geom[0] != self._geometry[0]:
                self.vb.autoRange(padding=0.02)
            self._geometry = geom
        self._last = (img, roi, hbin, vbin)

    def set_sites(self, sites, spots=(), clouds=()):
        vis = self.show_overlays
        b = [(s["x"], s["y"]) for s in sites if s["state"] == "bright"]
        d = [(s["x"], s["y"]) for s in sites if s["state"] != "bright"]
        self.sc_bright.setData(pos=np.array(b).reshape(-1, 2) if vis else np.zeros((0, 2)))
        self.sc_dark.setData(pos=np.array(d).reshape(-1, 2) if vis else np.zeros((0, 2)))
        self.sc_spots.setData(pos=np.array([(s["x"], s["y"]) for s in spots]).reshape(-1, 2)
                              if vis else np.zeros((0, 2)))
        self.sc_clouds.setData(pos=np.array([(s["x"], s["y"]) for s in clouds]).reshape(-1, 2)
                               if vis else np.zeros((0, 2)))
        n = len(sites) if vis and len(sites) <= 40 else 0
        while len(self._labels) < n:
            t = pg.TextItem("", color="#e8ffe8", anchor=self._label_anchor())
            t.setParentItem(self.group)
            self._labels.append(t)
        for i, t in enumerate(self._labels):
            if i < n:
                t.setText(str(sites[i]["index"]))
                t.setPos(sites[i]["x"], sites[i]["y"])
                t.setVisible(True)
            else:
                t.setVisible(False)

    def _label_anchor(self):
        # beside a (screen-)vertical chain, above a horizontal one
        return (-0.5, 0.5) if int(self.rotation) % 180 == 90 else (0.5, 1.6)

    def set_plan(self, plan):
        self._plan = plan
        for r in self._rects:
            r.setParentItem(None)
            if r.scene() is not None:
                r.scene().removeItem(r)
        self._rects = []
        if not plan or not self.show_overlays:
            return
        x0, y0, x1, y1 = plan["aqroi"]
        hb, vb = plan["binning"]
        # the binning grid: Grabber ROI edges can only lie on these lines, which
        # is why an ion is generally not in the middle of its ROI
        w, h = (x1 - x0) // hb, (y1 - y0) // vb
        if 1 < max(w, h) <= 128:
            path = QtGui.QPainterPath()
            for k in range(1, w):
                path.moveTo(x0 + k * hb - 0.5, y0 - 0.5)
                path.lineTo(x0 + k * hb - 0.5, y1 - 0.5)
            for k in range(1, h):
                path.moveTo(x0 - 0.5, y0 + k * vb - 0.5)
                path.lineTo(x1 - 0.5, y0 + k * vb - 0.5)
            grid = QtWidgets.QGraphicsPathItem(path)
            grid.setPen(_pen((42, 212, 255, 90), 1.0, QtCore.Qt.PenStyle.DotLine))
            grid.setParentItem(self.group)
            self._rects.append(grid)
        aq_rect = (x0 - 0.5, y0 - 0.5, x1 - x0, y1 - y0)
        self._add_rect(*aq_rect, "#2ad4ff", QtCore.Qt.PenStyle.DashLine)
        # every Grabber ROI with its own colour and a label saying what it
        # measures - "ion 0" is Grabber ROI 0 (chain order, as in the plan
        # table) with the share of that ion's light inside it.  The labels sit
        # outside the AQROI beside their ROI (on screen, whatever the display
        # rotation): the ions' on one side, the background ROI's on its side.
        rois = plan["ion_rois"] + ([plan["background_roi"]] if plan.get("background_roi") else [])
        cont = plan.get("containment") or []
        boxes = []
        for i, (u0, v0, u1, v1) in enumerate(rois):
            rect = (x0 + u0 * hb - 0.5, y0 + v0 * vb - 0.5, (u1 - u0) * hb, (v1 - v0) * vb)
            if i < plan["n_ions"]:
                color = ION_COLORS[i % len(ION_COLORS)]
                text = "ion {}".format(i) + (" {:.0%}".format(cont[i]) if i < len(cont) else "")
            else:
                color, text = (170, 170, 170), "bg"
            self._add_rect(*rect, color, QtCore.Qt.PenStyle.SolidLine)
            boxes.append((text, _lighter(color), self.group.mapRectToParent(QtCore.QRectF(*rect))))
        aq = self.group.mapRectToParent(QtCore.QRectF(*aq_rect))      # view coords, y up
        along_y = aq.height() >= aq.width()           # chain runs up/down on screen
        bg = boxes[plan["n_ions"]][2] if len(boxes) > plan["n_ions"] else None
        if along_y:
            bg_low = bg is not None and bg.center().x() < aq.center().x()
            ion_side, bg_side = ("right", "left") if bg_low else ("left", "right")
        else:
            bg_low = bg is not None and bg.center().y() < aq.center().y()
            ion_side, bg_side = ("top", "bottom") if bg_low else ("bottom", "top")
        for i, (text, color, box) in enumerate(boxes):
            self._add_label(text, color, aq, box, ion_side if i < plan["n_ions"] else bg_side)
        self._add_label("AQROI", "#2ad4ff", aq, aq, "bottom" if along_y else "left")

    def _add_rect(self, x, y, w, h, color, style):
        r = QtWidgets.QGraphicsRectItem(x, y, w, h)
        r.setPen(_pen(color, 1.5, style))
        r.setParentItem(self.group)
        self._rects.append(r)

    def _add_label(self, text, color, aq, box, side):
        """Text just outside the AQROI ``aq`` on ``side`` (as seen on screen),
        level with ``box`` (both rects in view coordinates, y up)."""
        gap = 0.4
        if side == "left":
            pos, anchor = (aq.left() - gap, box.center().y()), (1.0, 0.5)
        elif side == "right":
            pos, anchor = (aq.right() + gap, box.center().y()), (0.0, 0.5)
        elif side == "top":
            pos, anchor = (box.center().x(), aq.bottom() + gap), (0.5, 1.0)
        else:
            pos, anchor = (box.center().x(), aq.top() - gap), (0.5, 0.0)
        t = pg.TextItem(text, color=color, anchor=anchor)
        t.setParentItem(self.group)
        t.setPos(self.group.mapFromParent(QtCore.QPointF(*pos)))
        self._rects.append(t)

    def screen_step(self, right, up):
        """Sensor-pixel step (dx, dy) that moves things ``right``/``up`` on
        screen through the display rotation and flips."""
        t = self.group.transform()
        inv, ok = t.inverted()
        if not ok:  # pragma: no cover
            return int(right), int(-up)
        a = inv.map(QtCore.QPointF(0.0, 0.0))
        b = inv.map(QtCore.QPointF(float(right), float(up)))         # view y points up
        return int(round(b.x() - a.x())), int(round(b.y() - a.y()))

    def visible_sensor_rect(self):
        """Visible area as a sensor ROI (x0, y0, x1, y1)."""
        vr = self.vb.viewRect()
        poly = self.group.mapFromParent(QtGui.QPolygonF(vr))
        br = poly.boundingRect()
        return (int(math.floor(br.left() + 0.5)), int(math.floor(br.top() + 0.5)),
                int(math.ceil(br.right() + 0.5)), int(math.ceil(br.bottom() + 0.5)))

    def _mouse_moved(self, pos):
        if self._last is None:
            return
        p = self.group.mapFromScene(pos)
        x, y = p.x(), p.y()
        img, roi, hb, vb = self._last
        u = int(math.floor((x - roi[0] + 0.5) / hb))
        v = int(math.floor((y - roi[1] + 0.5) / vb))
        val = ""
        if 0 <= v < img.shape[0] and 0 <= u < img.shape[1]:
            val = "  value {}".format(int(img[v, u]))
        self.hovered.emit("sensor x {:.1f}  y {:.1f}{}".format(x, y, val))


class StatusBanner(QtWidgets.QFrame):
    """Big, colour-coded crystal status for across-the-lab readability."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        lay = QtWidgets.QHBoxLayout(self)
        self.count = QtWidgets.QLabel("- ions")
        f = self.count.font()
        f.setPointSize(26)
        f.setBold(True)
        self.count.setFont(f)
        self.state = QtWidgets.QLabel("")
        f2 = self.state.font()
        f2.setPointSize(13)
        self.state.setFont(f2)
        self.mode = QtWidgets.QLabel("")
        self.temp = QtWidgets.QLabel("")
        self.fps = QtWidgets.QLabel("")
        self.control = QtWidgets.QLabel("")
        lay.addWidget(self.count)
        lay.addSpacing(12)
        lay.addWidget(self.state, 1)
        right = QtWidgets.QVBoxLayout()
        for w in (self.mode, self.temp, self.fps, self.control):
            w.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            right.addWidget(w)
        lay.addLayout(right)
        self._flash_until = 0.0
        self._color = "#555555"

    def flash(self, seconds=6.0):
        self._flash_until = time.time() + seconds

    def update_status(self, st):
        ions = st.get("ions") or {}
        nb, nd = ions.get("n_bright") or 0, ions.get("n_dark") or 0
        state = ions.get("state") or "empty"
        text = "{} ion{}".format(nb, "s" * (nb != 1))
        if nd:
            text += " + {} dark".format(nd)
        self.count.setText(text)
        desc = {"stable": "stable crystal", "empty": "no ions", "changing": "changing...",
                "melted": "MELTED", "no_fluorescence": "NO FLUORESCENCE",
                "hold": "tracking on hold"}.get(state, state)
        if ions.get("stable") and state == "stable":
            desc += " ({:.0f} s)".format(ions.get("stable_for") or 0)
        if ions.get("n_clouds"):
            desc += "  + hot cloud"
        lvl = ions.get("fluorescence_level")
        if lvl is not None and state == "stable":
            desc += "  |  fluorescence {:.0%}".format(lvl)
        self.state.setText(desc)
        color = STATE_COLORS.get(state, "#555555")
        if time.time() < self._flash_until:
            color = "#d0211a"
        if color != self._color:
            self.setStyleSheet("StatusBanner {{ background: {}; }} QLabel {{ color: white; }}"
                               .format(color))
            self._color = color
        proc = st.get("procedure")
        mode = (st.get("mode") or "").upper()
        if proc:
            mode += ": " + proc.get("step", proc.get("name", ""))
        self.mode.setText(mode)
        t, ts = st.get("temperature"), st.get("temperature_status", "")
        self.temp.setText("{} C  ({})".format("--" if t is None else "{:.1f}".format(t),
                                              ts.replace("_", " ")))
        self.fps.setText("{:.1f} fps  frame {}".format(st.get("fps") or 0, st.get("frame_index")))
        c = st.get("control")
        self.control.setText("controlled by {}".format(c["owner"]) if c else "")


class TracePlot(pg.PlotWidget):
    """Per-ion brightness (relative to each ion's reference) over time."""

    def __init__(self, seconds=120.0, parent=None):
        super().__init__(parent)
        self.seconds = seconds
        self.setLabel("left", "ion brightness (rel.)")
        self.setLabel("bottom", "time [s]")
        self.showGrid(x=True, y=True, alpha=0.3)
        self.addLegend(offset=(5, 5))
        self.hist = deque(maxlen=20000)
        self.curves = []
        self.setYRange(-0.1, 1.6)

    def add(self, t, sites):
        self.hist.append((t, [s.get("level", 0.0) for s in sites]))

    def redraw(self):
        if not self.hist:
            return
        t_end = self.hist[-1][0]
        pts = [(t - t_end, lv) for t, lv in self.hist if t > t_end - self.seconds]
        n = max((len(lv) for _, lv in pts), default=0)
        n = min(n, len(ION_COLORS))
        while len(self.curves) < n:
            i = len(self.curves)
            self.curves.append(self.plot([], [], pen=_pen(ION_COLORS[i], 1.5),
                                         name="ion {}".format(i)))
        for i, c in enumerate(self.curves):
            xs = [t for t, lv in pts if len(lv) > i]
            ys = [lv[i] for t, lv in pts if len(lv) > i]
            c.setData(xs, ys)


class EventList(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        self.setWordWrap(True)

    def add_events(self, events):
        for e in events:
            item = QtWidgets.QListWidgetItem("{}  {}".format(e.get("time", "")[-8:],
                                                             e.get("message", e.get("kind"))))
            item.setForeground(QtGui.QBrush(QtGui.QColor(
                SEVERITY_COLORS.get(e.get("severity"), "#000000"))))
            item.setToolTip("{} ({})".format(e.get("kind"), e.get("severity")))
            self.insertItem(0, item)
        while self.count() > 1000:
            self.takeItem(self.count() - 1)


class SiteTable(QtWidgets.QTableWidget):
    COLS = ["#", "x", "y", "state", "brightness", "SNR"]

    def __init__(self, parent=None):
        super().__init__(0, len(self.COLS), parent)
        self.setHorizontalHeaderLabels(self.COLS)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)

    def set_sites(self, sites):
        self.setRowCount(len(sites))
        for r, s in enumerate(sites):
            vals = [s["index"], "{:.2f}".format(s["x"]), "{:.2f}".format(s["y"]),
                    s["state"] + (" (inferred)" if s.get("inferred") else ""),
                    "{:.0%}".format(s.get("level", 0)), "{:.0f}".format(s.get("snr", 0))]
            for c, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(str(v))
                if c == 3:
                    it.setForeground(QtGui.QBrush(QtGui.QColor(
                        "#1b8a3a" if s["state"] == "bright" else "#b0201a")))
                self.setItem(r, c, it)


class HistogramPlot(pg.PlotWidget):
    """Histogram of one ion's Grabber-ROI sums with the fitted threshold."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLabel("bottom", "ROI sum [counts]")
        self.setLabel("left", "shots")
        self.curve = self.plot([], [], stepMode="center", fillLevel=0,
                               brush=(80, 140, 220, 120), pen=_pen((40, 90, 180), 1))
        self.line = pg.InfiniteLine(angle=90, pen=_pen("#d0211a", 2, QtCore.Qt.PenStyle.DashLine))
        self.addItem(self.line)
        self.line.setVisible(False)

    def set_data(self, values, threshold=None):
        v = np.asarray(values, float)
        if v.size < 2:
            self.curve.setData([], [])
            self.line.setVisible(False)
            return
        y, x = np.histogram(v, bins=min(80, max(10, v.size // 20)))
        self.curve.setData(x, y)
        if threshold is not None:
            self.line.setValue(threshold)
            self.line.setVisible(True)
        else:
            self.line.setVisible(False)
