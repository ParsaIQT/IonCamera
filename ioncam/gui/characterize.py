"""The camera characterization dialog (Camera tab -> Characterize camera...):
what to measure, progress, every setting's dark-frame numbers and
histogram, and the SPAM ranking with "Use for readout"."""

import math

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ioncam.characterize import ORDER, STAGES
from ioncam.gui import style
from ioncam.gui.main_window import _button, _group, _muted, _row, _spin

_NUM = QtCore.Qt.ItemDataRole.DisplayRole
_KEY = QtCore.Qt.ItemDataRole.UserRole


def _fmt(v, f="{:.3g}", none="-"):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return none
    return f.format(v)


def _minutes(s):
    if s is None:
        return "-"
    s = float(s)
    if s < 90:
        return "{:.0f} s".format(s)
    if s < 5400:
        return "{:.0f} min".format(s / 60.0)
    return "{:.1f} h".format(s / 3600.0)


def _item(value, text=None, key=None):
    """A table cell that sorts by ``value`` (numbers numerically)."""
    it = QtWidgets.QTableWidgetItem()
    if isinstance(value, (int, float)) and value is not None and math.isfinite(value):
        it.setData(_NUM, float(value))
        if text is not None:
            it.setText(text)
    else:
        it.setText(text if text is not None else ("" if value is None else str(value)))
    if key is not None:
        it.setData(_KEY, key)
    return it


def _table(cols):
    t = QtWidgets.QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(cols)
    t.verticalHeader().setVisible(False)
    t.setAlternatingRowColors(True)
    t.setShowGrid(False)
    t.setSortingEnabled(True)
    t.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    t.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
    t.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
    t.horizontalHeader().setStretchLastSection(True)
    # (Qt's default sort indicator is the first column, descending: the worst first)
    t.horizontalHeader().setSortIndicator(0, QtCore.Qt.SortOrder.AscendingOrder)
    return t


def _resort(t):
    h = t.horizontalHeader()
    t.setSortingEnabled(True)
    t.sortItems(h.sortIndicatorSection(), h.sortIndicatorOrder())


class DarkHistogramPlot(pg.PlotWidget):
    """A setting's dark counts (above bias) with the fitted model, log scale."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLabel("bottom", "counts above bias")
        self.setLabel("left", "samples")
        self.getAxis("left").enableAutoSIPrefix(False)
        self.getAxis("bottom").enableAutoSIPrefix(False)
        self.showGrid(x=True, y=True, alpha=0.2)
        self.setLogMode(y=True)
        self.legend = self.addLegend(offset=(-10, 10))
        t = style.current()
        self.data = self.plot([], [], stepMode="center", name="measured",
                              pen=pg.mkPen(t["accent"], width=1.5))
        self.model = self.plot([], [], name="model", pen=pg.mkPen((230, 120, 60), width=1.5,
                                                                   style=QtCore.Qt.PenStyle.DashLine))
        style.style_plot(self)

    def apply_theme(self):
        style.style_plot(self)

    def set_point(self, detail):
        counts = detail.get("counts") if detail else None
        if counts is None or detail.get("lo") is None:
            self.data.setData([], [])
            self.model.setData([], [])
            return
        counts = np.asarray(counts, float)
        lo = int(detail["lo"])
        x = np.arange(lo, lo + counts.size)
        # empty bins cannot be drawn on a log scale: a floor below one sample
        self.data.setData(np.append(x - 0.5, x[-1] + 0.5), np.maximum(counts, 0.3))
        m = detail.get("model")
        if m is not None:
            m = np.asarray(m, float)
            keep = m > 0.05
            self.model.setData(x[keep], m[keep])
        else:
            self.model.setData([], [])
        self.setTitle(detail.get("point", {}).get("label", ""))


class CharacterizeDialog(QtWidgets.QDialog):
    """Plan, run and read a camera characterization."""

    SETTING_COLS = ["#", "stage", "setting", "bias", "read noise [e⁻]", "EM gain meas.",
                    "±", "CIC parallel", "CIC serial", "fit", "events/px", "hot px", "χ²/dof",
                    "readout [ms]", "note"]

    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.backend = main.backend
        cfg = dict(main.config.get("characterize") or {})
        self.cfg = cfg
        self.setWindowTitle("Camera characterization (SPAM)")
        self.resize(1240, 860)
        self.directory = None
        self.results = None
        self._state = None
        self._estimate_pending = False
        self._ranking = []

        # --- what to measure ---------------------------------------------------
        self.stage_boxes = {}
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        for r, name in enumerate(ORDER):
            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(name in (cfg.get("stages") or []))
            cb.setToolTip(STAGES[name])
            cb.toggled.connect(self._plan_changed)
            self.stage_boxes[name] = cb
            grid.addWidget(cb, r, 0)
            grid.addWidget(_muted(STAGES[name], wrap=True), r, 1)
        grid.setColumnStretch(1, 1)
        self.samples = _spin(0.1, 100.0, float(cfg.get("samples_per_point", 3e6)) / 1e6,
                             step=0.5, decimals=1, suffix=" M",
                             tip="pixel samples per setting (frames x super-pixels): the "
                                 "tail of the dark distribution needs millions", width=110)
        self.max_s = _spin(1, 600, int(cfg.get("max_point_s", 20)), suffix=" s", width=90,
                           tip="longest acquisition per setting")
        self.temps = QtWidgets.QLineEdit(", ".join("{:g}".format(float(t)) for t in
                                                   cfg.get("temperatures") or []))
        self.temps.setPlaceholderText("e.g. -60, -70, -80")
        self.temps.setToolTip("sensor set points of the temperature stage [C]")
        for w in (self.samples, self.max_s):
            w.valueChanged.connect(self._plan_changed)
        self.temps.editingFinished.connect(self._plan_changed)
        self.estimate = _muted("", wrap=True)
        self.start_btn = _button("Start", self._start, "close the shutter and measure",
                                 "play", "primary")
        self.abort_btn = _button("Abort", self._abort, "stop; what was measured is kept",
                                 "abort", "danger")
        self.abort_btn.setEnabled(False)
        plan = _group("What to measure (shutter closed, the camera is busy meanwhile)", [
            _muted("Dark frames at many readout settings around the Readout tab's: bias, "
                   "read noise, the real EM gain, clock-induced charge (parallel and serial), "
                   "dark current, hot pixels - and the SPAM error every setting gives. "
                   "Without an internal shutter, cover the camera.", wrap=True),
            grid,
            _row("samples per setting", self.samples, "at most", self.max_s, None),
            _row("temperatures", self.temps),
            self.estimate,
            _row(self.start_btn, self.abort_btn, None),
        ])

        # --- progress ---------------------------------------------------------------
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.state_label = QtWidgets.QLabel("not started")
        self.notes = _muted("", wrap=True)
        self.folder = QtWidgets.QLineEdit()
        self.folder.setReadOnly(True)
        self.open_btn = _button("Open...", self._open, "show a characterization saved "
                                "earlier", "layers")
        progress = _group("Progress", [
            self.progress, self.state_label, self.notes,
            _row("results in", self.folder, self.open_btn),
        ])

        # --- results ----------------------------------------------------------------
        self.settings_table = _table(self.SETTING_COLS)
        self.settings_table.itemSelectionChanged.connect(self._setting_selected)
        self.hist_plot = DarkHistogramPlot()
        self.hist_plot.setMinimumHeight(220)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        split.addWidget(self.settings_table)
        split.addWidget(self.hist_plot)
        split.setSizes([420, 260])

        self.ion_rate = _spin(0, 1e7, float(cfg.get("ion_rate") or 0.0), step=500, decimals=0,
                              suffix=" e⁻/s", special="measured", width=150,
                              tip="photo-electrons/s from one bright ion (all its light); "
                                  "'measured' = the brightness measured live")
        self.roi_frac = _spin(0, 100, 100 * float(cfg.get("roi_fraction") or 0.0), step=5,
                              decimals=0, suffix=" %", special="from plan", width=110,
                              tip="the ion's light in its readout ROI; 'from plan' = the "
                                  "readout plan's, else 60 %")
        self.roi_px = _spin(1, 64, int(cfg.get("roi_pixels") or 1), width=70,
                            tip="super-pixels summed per ion")
        ro = main.config.get("readout") or {}
        self.hbin = _spin(1, 64, int(ro.get("hbin", 4)), width=64, tip="readout binning")
        self.vbin = _spin(1, 64, int(ro.get("vbin", 4)), width=64)
        self.bg_rate = _spin(0, 1e6, float(cfg.get("background_rate") or 0.0), step=10,
                             decimals=0, suffix=" e⁻/s", width=120,
                             tip="stray light in the ROI (dark state)")
        self.target = QtWidgets.QComboBox()
        for v in ("1e-2", "3e-3", "1e-3", "3e-4", "1e-4", "1e-5"):
            self.target.addItem(v, float(v))
        tv = float(cfg.get("target_error") or 1e-3)
        self.target.setCurrentIndex(min(range(self.target.count()),
                                        key=lambda i: abs(math.log(self.target.itemData(i) /
                                                                   tv))))
        self.tau_b = _spin(0, 1e5, float(cfg.get("bright_lifetime_ms") or 0.0), step=1,
                           decimals=1, suffix=" ms", special="ignore", width=100,
                           tip="bright state lifetime under the detection light")
        self.tau_d = _spin(0, 1e5, float(cfg.get("dark_lifetime_ms") or 0.0), step=1,
                           decimals=1, suffix=" ms", special="ignore", width=100,
                           tip="dark state lifetime under the detection light")
        self.rank_btn = _button("Rank", self._rank, "rank the settings for this ion",
                                "refresh")
        self.use_btn = _button("Use for readout", self._use, "the selected setting and its "
                               "detection time for the Grabber readout", "check", "primary")
        self.use_btn.setEnabled(False)
        self.det_ms = [float(v) for v in (cfg.get("detection_ms") or [0.5, 1, 2, 5])]
        cols = ["#", "setting", "t for target [ms]", "threshold (raw)"] + \
            ["error @ {:g} ms".format(t) for t in self.det_ms] + ["readout [ms]", "CIC/px",
                                                                   "read noise [e⁻]", "fit"]
        self.rank_table = _table(cols)
        self.rank_table.itemSelectionChanged.connect(
            lambda: self.use_btn.setEnabled(bool(self.rank_table.selectedItems())))
        self.spam_info = _muted("", wrap=True)
        spam = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(spam)
        sl.setContentsMargins(4, 4, 4, 4)
        sl.addLayout(_row("ion", self.ion_rate, "in ROI", self.roi_frac, "super-pixels",
                          self.roi_px, "binning", self.hbin, "x", self.vbin, None))
        sl.addLayout(_row("stray light", self.bg_rate, "target error", self.target,
                          "bright lifetime", self.tau_b, "dark lifetime", self.tau_d,
                          self.rank_btn, None))
        sl.addWidget(self.spam_info)
        sl.addWidget(self.rank_table, 1)
        sl.addLayout(_row(_muted("Threshold (raw): the ROI sum's threshold with the bias, "
                                 "for the Grabber.", wrap=True), None, self.use_btn))
        self.summary = QtWidgets.QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(spam, "SPAM ranking")
        self.tabs.addTab(split, "Settings measured")
        self.tabs.addTab(self.summary, "Summary")

        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(plan)
        ll.addWidget(progress)
        ll.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        body.addWidget(scroll)
        body.addWidget(self.tabs)
        body.setSizes([500, 740])
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(body)
        close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.close)
        lay.addWidget(close)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._poll)
        self.timer.start()
        self._poll()
        self._plan_changed()

    # -- plan --------------------------------------------------------------------------

    def options(self) -> dict:
        temps = []
        for part in self.temps.text().replace(";", ",").split(","):
            part = part.strip()
            if part:
                try:
                    temps.append(float(part))
                except ValueError:
                    pass
        stages = [n for n, cb in self.stage_boxes.items() if cb.isChecked()]
        if temps and "temperature" not in stages:
            stages.append("temperature")
        return {"stages": stages, "samples_per_point": int(self.samples.value() * 1e6),
                "max_point_s": float(self.max_s.value()), "temperatures": temps}

    def _plan_changed(self, *_):
        if self._estimate_pending:
            return
        self._estimate_pending = True
        QtCore.QTimer.singleShot(300, self._estimate)

    def _estimate(self):
        self._estimate_pending = False
        opts = self.options()
        if not opts["stages"]:
            self.estimate.setText("nothing selected")
            return

        def done(info):
            stages = ", ".join("{} {}".format(v, k) for k, v in info["stages"].items())
            self.estimate.setText("{} settings ({}), about {} - region {}".format(
                info["points"], stages, _minutes(info["estimate_s"]), info["region"]))

        self.backend.call("start_characterization", callback=done,
                          error=lambda n, e: self.estimate.setText("plan: {}".format(e)),
                          channel="wait", dry_run=True, **opts)

    def _start(self):
        opts = self.options()
        if not opts["stages"]:
            return
        if QtWidgets.QMessageBox.question(
                self, "ioncam", "Close the shutter and measure the camera now?\n\n{}\n\n"
                "The camera is busy until it is done (abort any time; what was measured is "
                "kept). Live video and the Grabber readout resume afterwards.".format(
                    self.estimate.text()),
                QtWidgets.QMessageBox.StandardButton.Yes |
                QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No) != \
                QtWidgets.QMessageBox.StandardButton.Yes:
            return

        def started(info):
            self.directory = info["directory"]
            self.folder.setText(self.directory)
            self.results = None
            self._poll()

        self.start_btn.setEnabled(False)
        self.backend.call("start_characterization", callback=started,
                          error=self._failed, **opts)

    def _abort(self):
        self.backend.call("abort")

    def _failed(self, name, exc):
        self.start_btn.setEnabled(True)
        QtWidgets.QMessageBox.warning(self, "ioncam", "{} failed:\n{}".format(name, exc))

    # -- progress ----------------------------------------------------------------------

    def _poll(self):
        if not self.isVisible() and self._state not in (None, "running", "finishing"):
            return
        self.backend.call("characterization_status", callback=self._status, channel="poll",
                          error=lambda n, e: None)

    def _status(self, st):
        state = st.get("state", "idle")
        running = state in ("running", "finishing")
        self.start_btn.setEnabled(not running)
        self.abort_btn.setEnabled(state == "running")
        if st.get("directory"):
            if self.directory is None or running:
                self.directory = st["directory"]
                self.folder.setText(self.directory)
        if state == "idle":
            self.state_label.setText("not started")
        else:
            cur = (st.get("current") or {}).get("label") or ""
            self.progress.setValue(int(1000 * float(st.get("progress") or 0.0)))
            text = "{}: {} of {} settings".format(state, st.get("done", 0), st.get("points", 0))
            if running:
                text += " - {} left - now: {}".format(_minutes(st.get("eta_s")), cur)
            elif st.get("elapsed_s"):
                text += " in {}".format(_minutes(st.get("elapsed_s")))
            if st.get("error"):
                text += " - " + st["error"]
            self.state_label.setText(text)
            self.notes.setText("\n".join(st.get("notes") or []))
            if running and st.get("recent"):
                self._fill_settings(st["recent"], append=True)
        if self._state in ("running", "finishing") and state in ("done", "aborted", "failed"):
            self._rank()
        elif self.results is None and state in ("done", "aborted") and self._state is None:
            self._rank()
        self._state = state

    # -- results ------------------------------------------------------------------------

    def spam(self) -> dict:
        return {"ion_rate": self.ion_rate.value() or None,
                "roi_fraction": self.roi_frac.value() / 100.0 or None,
                "roi_pixels": int(self.roi_px.value()),
                "binning": [int(self.hbin.value()), int(self.vbin.value())],
                "background_rate": float(self.bg_rate.value()),
                "target_error": float(self.target.currentData()),
                "bright_lifetime_ms": float(self.tau_b.value()),
                "dark_lifetime_ms": float(self.tau_d.value()),
                "detection_ms": self.det_ms}

    def _rank(self):
        self.rank_btn.setEnabled(False)
        self.spam_info.setText("ranking the settings...")

        def done(res):
            self.rank_btn.setEnabled(True)
            self.results = res
            if res.get("directory"):
                self.directory = res["directory"]
                self.folder.setText(self.directory)
            self._fill_settings(res.get("points") or [])
            self._fill_ranking(res)
            self._fill_summary(res)

        def failed(name, exc):
            self.rank_btn.setEnabled(True)
            self.spam_info.setText("no results: {}".format(exc))

        self.backend.call("characterization_results", self.directory, callback=done,
                          error=failed, channel="wait", **self.spam())

    def _open(self):
        def got(items):
            if not items:
                QtWidgets.QMessageBox.information(self, "ioncam", "No characterization saved "
                                                  "yet.")
                return
            names = ["{}  ({} settings{})".format(i["name"], i["points"], "" if i["complete"]
                                                  else ", incomplete") for i in items]
            name, ok = QtWidgets.QInputDialog.getItem(self, "ioncam", "Characterization:",
                                                      names, 0, False)
            if ok:
                self.directory = items[names.index(name)]["directory"]
                self.folder.setText(self.directory)
                self._rank()
        self.backend.call("list_characterizations", callback=got, channel="wait")

    def _fill_settings(self, points, append=False):
        t = self.settings_table
        t.setSortingEnabled(False)
        if not append:
            t.setRowCount(0)
        have = {t.item(r, 0).data(_KEY) for r in range(t.rowCount()) if t.item(r, 0)}
        for p in points:
            key = p.get("index")
            if key in have:
                continue
            r = t.rowCount()
            t.insertRow(r)
            gain = p.get("em_gain_measured")
            vals = [
                _item(key if key is not None and key >= 0 else None,
                      str(key) if key is not None and key >= 0 else "s{}".format(-(key or 0)),
                      key=key),
                _item(p.get("stage")), _item(p.get("label")),
                _item(p.get("bias"), _fmt(p.get("bias"), "{:.1f}")),
                _item(p.get("read_noise_e"), _fmt(p.get("read_noise_e"), "{:.1f}")),
                _item(gain, "" if gain is None else "{:.0f} / {}".format(gain, p.get("em_gain"))),
                _item(p.get("g_err"), _fmt(None if p.get("g_err") is None else
                                           100 * p["g_err"], "{:.1f} %")),
                _item(p.get("cic_parallel"), _fmt(p.get("cic_parallel"), "{:.2e}")),
                _item(p.get("cic_serial"), _fmt(p.get("cic_serial"), "{:.2e}")),
                _item(p.get("model_source") or ""),
                _item(p.get("events_per_px"), _fmt(p.get("events_per_px"), "{:.2e}")),
                _item(p.get("n_hot"), _fmt(p.get("n_hot"), "{:d}")),
                _item(p.get("chi2_dof"), _fmt(p.get("chi2_dof"), "{:.2f}")),
                _item(p.get("readout_time_ms"), _fmt(p.get("readout_time_ms"), "{:.2f}")),
                _item(p.get("note") or "")]
            for c, it in enumerate(vals):
                t.setItem(r, c, it)
        _resort(t)
        t.resizeColumnsToContents()

    def _fill_ranking(self, res):
        rows = res.get("ranking") or []
        self._ranking = rows
        sp = res.get("spam") or {}
        t = self.rank_table
        t.setSortingEnabled(False)
        t.setRowCount(0)
        for i, row in enumerate(rows):
            t.insertRow(i)
            errs = row.get("errors") or {}
            vals = [_item(row.get("rank"), str(row.get("rank")), key=i),
                    _item(row.get("label")),
                    _item(row.get("t_min_ms"), _fmt(row.get("t_min_ms"), "{:.2f}",
                                                    none="not reached")),
                    _item(row.get("threshold_raw"), _fmt(row.get("threshold_raw"), "{:d}"))]
            for tms in self.det_ms:
                e = errs.get("{:g}".format(tms))
                vals.append(_item(e, _fmt(e, "{:.1e}")))
            vals += [_item(row.get("readout_time_ms"), _fmt(row.get("readout_time_ms"),
                                                              "{:.2f}")),
                     _item(row.get("cic_parallel"), _fmt(row.get("cic_parallel"), "{:.2e}")),
                     _item(row.get("read_noise_e"), _fmt(row.get("read_noise_e"), "{:.1f}")),
                     _item(row.get("model_source") or "")]
            for c, it in enumerate(vals):
                t.setItem(i, c, it)
        _resort(t)
        t.resizeColumnsToContents()
        if not rows:
            self.spam_info.setText("no EM-gain settings measured yet")
            return
        best = rows[0]
        self.spam_info.setText(
            "For an ion giving {:,.0f} e⁻/s ({}) with {:.0%} of its light in {} "
            "super-pixel(s) of {}x{} binning: best {} - {} to reach a SPAM error of {:g}."
            .format(sp.get("ion_rate", 0), sp.get("ion_rate_source", ""),
                    sp.get("roi_fraction", 0), sp.get("roi_pixels", 1),
                    *(sp.get("binning") or [1, 1]), best.get("label"),
                    "{:.2f} ms".format(best["t_min_ms"]) if best.get("t_min_ms") else
                    "not reached within 30 ms", sp.get("target_error", 0)))

    def _fill_summary(self, res):
        g = res.get("global") or {}
        lines = ["Camera: {}".format((res.get("camera") or {}).get("model", "?")),
                 "State: {}   started {}   finished {}".format(res.get("state"),
                                                              res.get("started"),
                                                              res.get("finished")),
                 "Region: {}".format(res.get("region")), ""]
        lc = res.get("light_check") or {}
        if lc:
            lines.append("Light check (shutter closed, {:.0f} C): {} e⁻/px/s over the region, "
                         "{} lit pixels{}".format(
                             lc.get("temperature", float("nan")), _fmt(lc.get("e_per_px_s")),
                             lc.get("lit_pixels", 0), " - LIGHT GETS IN" if lc.get("light")
                             else " - dark"))
        if g.get("light_leak"):
            lines += ["LIGHT LEAK: " + g["light_leak"], ""]
        dc = g.get("dark_current") or {}
        if dc:
            lines.append("Dark current: {} e⁻/px/s (±{}) at {} C; CIC at zero "
                         "exposure {} /px/frame".format(_fmt(dc.get("e_per_px_s")),
                                                        _fmt(dc.get("error")),
                                                        _fmt(dc.get("temperature"), "{:.1f}"),
                                                        _fmt(dc.get("cic_at_zero"))))
        for d in g.get("dark_current_temperature") or []:
            lines.append("  at {:.1f} C: {} e⁻/px/s".format(d["temperature"],
                                                                _fmt(d["e_per_px_s"])))
        if g.get("gain_ratio"):
            lines.append("EM gain: real / set = {:.3f} ± {:.3f} (spread {:.3f})".format(
                g["gain_ratio"], g.get("gain_ratio_error") or 0.0,
                g.get("gain_ratio_spread") or 0.0))
        for b in g.get("binning") or []:
            lines.append("Binning {}x{}: CIC {} per super-pixel (pixels add up to {}), read "
                         "noise {} counts".format(b["binning"][0], b["binning"][1],
                                                  _fmt(b.get("cic_parallel")),
                                                  _fmt(b.get("cic_parallel_expected")),
                                                  _fmt(b.get("read_noise"), "{:.2f}")))
        st = g.get("stability")
        if st:
            bias = [b for b in st.get("bias") or [] if b is not None]
            if bias:
                lines.append("Stability over {:.0f} min: bias {:.1f}..{:.1f}, CIC {}..{}"
                             .format(st["times_min"][-1], min(bias), max(bias),
                                     _fmt(min(c for c in st["cic_parallel"] if c is not None)),
                                     _fmt(max(c for c in st["cic_parallel"] if c is not None))))
        ff = g.get("first_frame")
        if ff:
            lines.append("First frame after a start: offset {:.1f} counts, events x{:.2f} "
                         "(worst x{:.2f})".format(ff["offset_median"], ff["event_ratio_median"],
                                                  ff["event_ratio_max"]))
        for tr in g.get("triggered") or []:
            lines.append("Triggered shots, {:g} ms apart: CIC {} /px, bias {}".format(
                tr["spacing_ms"], _fmt(tr.get("cic_parallel")), _fmt(tr.get("bias"),
                                                                     "{:.1f}")))
        if g.get("hot_pixels") is not None:
            lines.append("Hot pixels on the sensor: {}".format(len(g["hot_pixels"])))
        if g.get("trail"):
            lines.append("Charge trails (fraction left in the next pixel): " + ", ".join(
                "{} {:.3f}".format(k, v) for k, v in g["trail"].items()))
        if res.get("notes"):
            lines += [""] + list(res["notes"])
        lines += ["", "Files: summary.json / summary.csv (everything), points.jsonl, hist/*.npz "
                  "(histograms), camera_profile.json (for simulations: "
                  "simulator.camera_profile). See README.txt there."]
        self.summary.setPlainText("\n".join(lines))

    def _setting_selected(self):
        items = self.settings_table.selectedItems()
        if not items:
            return
        row = items[0].row()
        key = self.settings_table.item(row, 0).data(_KEY)
        if key is None or self.directory is None:
            return
        self.backend.call("characterization_point", key, self.directory,
                          callback=self.hist_plot.set_point, channel="wait",
                          error=lambda n, e: None)

    def _use(self):
        items = self.rank_table.selectedItems()
        if not items:
            return
        i = self.rank_table.item(items[0].row(), 0).data(_KEY)
        row = self._ranking[i]
        s = dict(row.get("readout_settings") or {})
        t = row.get("t_min_ms") or row.get("best_time_ms")
        if t:
            s["exposure"] = math.ceil(float(t) * 10.0) / 10000.0     # up to 0.1 ms
        text = ", ".join("{} = {}".format(k, v) for k, v in s.items())
        if QtWidgets.QMessageBox.question(
                self, "ioncam", "Use for the Grabber readout:\n{}\n\nThe Readout tab's settings "
                "change; an armed readout is re-armed with them.".format(text),
                QtWidgets.QMessageBox.StandardButton.Yes |
                QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No) != \
                QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.main._cmd("set_acquisition", "readout", **s)

    def apply_theme(self):
        self.hist_plot.apply_theme()

    def closeEvent(self, ev):
        if self._state not in ("running", "finishing"):
            self.timer.stop()
        super().closeEvent(ev)

    def showEvent(self, ev):
        self.timer.start()
        super().showEvent(ev)
