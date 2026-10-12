"""Look of the GUI: dark and light themes (Fusion + palette + style sheet),
small vector icons drawn at run time (so they follow the theme) and the
colours of the pyqtgraph plots.

Buttons take a ``kind`` property for their role: ``primary`` (the main action
of a group), ``danger`` (abort, destructive) - set with :func:`kind`.
"""

import math
import os
import tempfile

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

THEMES = {
    "dark": {
        "bg": "#111318", "surface": "#181b21", "surface2": "#22262e", "input": "#0e1014",
        "border": "#2a2f38", "border2": "#3a404b", "text": "#e4e7ec", "muted": "#9199a6",
        "accent": "#3b9cf0", "accent_hover": "#5bb0ff", "accent_text": "#ffffff",
        "selection": "#24476e", "danger": "#e5534b", "danger_hover": "#f06a62",
        "danger_tint": "#3b1f1f",
        "success": "#2fae63", "warning": "#d99a2b", "plot_bg": "#14171c", "grid": "#9199a6",
        "canvas": "#08090c",
    },
    "light": {
        "bg": "#eef0f3", "surface": "#ffffff", "surface2": "#f4f5f7", "input": "#ffffff",
        "border": "#d5d9df", "border2": "#bfc5ce", "text": "#1d2127", "muted": "#5d6571",
        "accent": "#1f6feb", "accent_hover": "#3b82f6", "accent_text": "#ffffff",
        "selection": "#cfe0fb", "danger": "#cf2e2e", "danger_hover": "#e04444",
        "danger_tint": "#fde8e8",
        "success": "#1a7f37", "warning": "#9a6700", "plot_bg": "#ffffff", "grid": "#5d6571",
        "canvas": "#08090c",
    },
}

_current = {"name": "dark"}
_arrow_dir = None


def current() -> dict:
    """Colour tokens of the active theme."""
    return THEMES[_current["name"]]


def theme_name() -> str:
    return _current["name"]


def kind(widget, name):
    """Give a button a role for the style sheet ("primary", "danger")."""
    widget.setProperty("kind", name)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    return widget


# -- icons ----------------------------------------------------------------------------

def _path_for(name: str) -> QtGui.QPainterPath:
    """Icon outlines on a 24 x 24 canvas."""
    p = QtGui.QPainterPath()
    if name == "play":
        p.moveTo(7, 4.5)
        p.lineTo(19.5, 12)
        p.lineTo(7, 19.5)
        p.closeSubpath()
    elif name == "stop":
        p.addRoundedRect(QtCore.QRectF(6, 6, 12, 12), 2, 2)
    elif name == "pause":
        p.addRoundedRect(QtCore.QRectF(6, 5, 4, 14), 1, 1)
        p.addRoundedRect(QtCore.QRectF(14, 5, 4, 14), 1, 1)
    elif name == "camera":
        p.addRoundedRect(QtCore.QRectF(3, 7, 18, 13), 2.5, 2.5)
        p.addRect(QtCore.QRectF(8, 4.5, 8, 3))
        hole = QtGui.QPainterPath()
        hole.addEllipse(QtCore.QPointF(12, 13.5), 4, 4)
        p = p.subtracted(hole)
        p.addEllipse(QtCore.QPointF(12, 13.5), 2.2, 2.2)
    elif name == "save":
        p.addRoundedRect(QtCore.QRectF(4, 4, 16, 16), 2, 2)
        cut = QtGui.QPainterPath()
        cut.addRect(QtCore.QRectF(7.5, 4, 8, 5.5))
        cut.addRoundedRect(QtCore.QRectF(7, 12.5, 10, 5.5), 1, 1)
        p = p.subtracted(cut)
    return p


def _draw(name, painter, color):
    """Stroked / filled icon ``name`` on a 24 x 24 canvas."""
    pen = QtGui.QPen(QtGui.QColor(color), 2.0)
    pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    fill = QtGui.QBrush(QtGui.QColor(color))
    if name in ("play", "stop", "pause", "camera", "save"):
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        painter.drawPath(_path_for(name))
        return
    painter.setPen(pen)
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    P = QtCore.QPointF
    if name == "target":                       # locate: cross-hair
        painter.drawEllipse(P(12, 12), 6.5, 6.5)
        for a, b in (((12, 2.5), (12, 7)), ((12, 17), (12, 21.5)), ((2.5, 12), (7, 12)),
                     ((17, 12), (21.5, 12))):
            painter.drawLine(P(*a), P(*b))
        painter.setBrush(fill)
        painter.drawEllipse(P(12, 12), 1.3, 1.3)
    elif name == "check":                      # calibrate
        painter.drawPolyline(QtGui.QPolygonF([P(5, 12.5), P(10, 17.5), P(19.5, 6.5)]))
    elif name == "verify":                     # magnifier
        painter.drawEllipse(P(10.5, 10.5), 5.5, 5.5)
        painter.drawLine(P(14.6, 14.6), P(20, 20))
    elif name == "bolt":                       # arm
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        painter.drawPolygon(QtGui.QPolygonF([P(13.5, 2.5), P(5.5, 13.5), P(11, 13.5),
                                             P(9.5, 21.5), P(18.5, 9.5), P(12.8, 9.5)]))
    elif name == "power":                      # disarm
        painter.drawArc(QtCore.QRectF(5, 5.5, 14, 14), 120 * 16, 300 * 16)
        painter.drawLine(P(12, 3.5), P(12, 11))
    elif name == "abort":
        painter.drawEllipse(P(12, 12), 8.5, 8.5)
        painter.drawLine(P(8.8, 8.8), P(15.2, 15.2))
        painter.drawLine(P(15.2, 8.8), P(8.8, 15.2))
    elif name == "layers":                     # learn background
        for dy in (0.0, 4.5):
            painter.drawPolygon(QtGui.QPolygonF([P(12, 4 + dy), P(20, 8.5 + dy),
                                                 P(12, 13 + dy), P(4, 8.5 + dy)]))
    elif name == "gear":
        painter.drawEllipse(P(12, 12), 3, 3)
        painter.drawEllipse(P(12, 12), 6.5, 6.5)
        for k in range(8):
            a = k * math.pi / 4
            painter.drawLine(P(12 + 6.5 * math.cos(a), 12 + 6.5 * math.sin(a)),
                             P(12 + 9.3 * math.cos(a), 12 + 9.3 * math.sin(a)))
    elif name == "theme":                      # half-filled circle
        painter.drawEllipse(P(12, 12), 8, 8)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        painter.drawPie(QtCore.QRectF(4, 4, 16, 16), 90 * 16, 180 * 16)
    elif name == "fit":                        # four corners
        for (x, y, dx, dy) in ((4, 4, 1, 1), (20, 4, -1, 1), (4, 20, 1, -1), (20, 20, -1, -1)):
            painter.drawPolyline(QtGui.QPolygonF([P(x, y + 5 * dy), P(x, y), P(x + 5 * dx, y)]))
    elif name == "ions":                       # a chain of three
        painter.setBrush(fill)
        for x in (5.5, 12, 18.5):
            painter.drawEllipse(P(x, 12), 2.2, 2.2)
    elif name == "trash":
        painter.drawLine(P(4.5, 6.5), P(19.5, 6.5))
        painter.drawLine(P(9.5, 6.5), P(10, 3.8))
        painter.drawLine(P(14.5, 6.5), P(14, 3.8))
        painter.drawPolyline(QtGui.QPolygonF([P(6.5, 6.5), P(7.5, 20), P(16.5, 20),
                                              P(17.5, 6.5)]))
    elif name == "copy":
        painter.drawRoundedRect(QtCore.QRectF(8.5, 8.5, 11, 11), 2, 2)
        painter.drawPolyline(QtGui.QPolygonF([P(15.5, 5.5), P(15.5, 4.5), P(4.5, 4.5),
                                              P(4.5, 15.5), P(5.5, 15.5)]))
    elif name == "refresh":
        painter.drawArc(QtCore.QRectF(5, 5, 14, 14), 40 * 16, 280 * 16)
        painter.setBrush(fill)
        painter.drawPolygon(QtGui.QPolygonF([P(19.8, 3.5), P(20.2, 10), P(14, 8.5)]))
    elif name == "undo":
        painter.drawArc(QtCore.QRectF(6, 6, 13, 12), 90 * 16, -230 * 16)
        painter.setBrush(fill)
        painter.drawPolygon(QtGui.QPolygonF([P(3.5, 6), P(10, 2.5), P(10, 9.5)]))
    elif name == "trigger":                    # a TTL pulse
        painter.drawPolyline(QtGui.QPolygonF([P(3, 17), P(8, 17), P(8, 7), P(14, 7),
                                              P(14, 17), P(21, 17)]))
    elif name in ("up", "down", "left", "right"):     # D-pad chevrons
        pts = {"up": ((6, 15), (12, 9), (18, 15)), "down": ((6, 9), (12, 15), (18, 9)),
               "left": ((15, 6), (9, 12), (15, 18)), "right": ((9, 6), (15, 12), (9, 18))}[name]
        pen.setWidthF(2.6)
        painter.setPen(pen)
        painter.drawPolyline(QtGui.QPolygonF([P(*p) for p in pts]))
    elif name in ("wider", "narrower", "taller", "shorter"):   # AQROI size
        out = name in ("wider", "taller")
        pen.setWidthF(2.2)
        painter.setPen(pen)
        for sgn in (-1, 1):                       # chevrons out (<|>) or in (>|<)
            tip, tail = 12 + sgn * (9.5 if out else 2.5), 12 + sgn * 6.0
            if name in ("wider", "narrower"):
                painter.drawPolyline(QtGui.QPolygonF([P(tail, 7.5), P(tip, 12), P(tail, 16.5)]))
            else:
                painter.drawPolyline(QtGui.QPolygonF([P(7.5, tail), P(12, tip), P(16.5, tail)]))
        if name in ("wider", "narrower"):
            painter.drawLine(P(12, 6), P(12, 18))
        else:
            painter.drawLine(P(6, 12), P(18, 12))
    elif name == "place":                      # a square with a cross-hair
        painter.drawRect(QtCore.QRectF(6, 6, 12, 12))
        for a_, b_ in (((12, 2.5), (12, 8.5)), ((12, 15.5), (12, 21.5)),
                       ((2.5, 12), (8.5, 12)), ((15.5, 12), (21.5, 12))):
            painter.drawLine(P(*a_), P(*b_))
    elif name == "square":                     # a new AQROI
        painter.drawRect(QtCore.QRectF(5, 5, 14, 14))
        painter.setBrush(fill)
        painter.drawEllipse(P(12, 12), 2.2, 2.2)
    elif name == "zoom":
        painter.drawEllipse(P(10.5, 10.5), 5.5, 5.5)
        painter.drawLine(P(14.6, 14.6), P(20, 20))
        painter.drawLine(P(8, 10.5), P(13, 10.5))
        painter.drawLine(P(10.5, 8), P(10.5, 13))
    else:  # pragma: no cover - unknown name: a dot
        painter.setBrush(fill)
        painter.drawEllipse(P(12, 12), 3, 3)


def icon(name: str, color=None) -> QtGui.QIcon:
    """Icon ``name`` drawn in ``color`` (the theme's text colour by default),
    with a muted variant for disabled buttons."""
    ic = QtGui.QIcon()
    t = current()
    for mode, c in ((QtGui.QIcon.Mode.Normal, color or t["text"]),
                    (QtGui.QIcon.Mode.Disabled, t["border2"])):
        for size in (16, 24, 32, 48):
            pm = QtGui.QPixmap(size, size)
            pm.fill(QtCore.Qt.GlobalColor.transparent)
            painter = QtGui.QPainter(pm)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.scale(size / 24.0, size / 24.0)
            _draw(name, painter, c)
            painter.end()
            ic.addPixmap(pm, mode)
    return ic


def _arrow_images(t) -> dict:
    """Small arrow images for spin boxes / combo boxes (style sheets need files)."""
    global _arrow_dir
    if _arrow_dir is None:
        _arrow_dir = tempfile.mkdtemp(prefix="ioncam_gui_")
    out = {}
    pm = QtGui.QPixmap(14, 14)
    pm.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pm)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    pen = QtGui.QPen(QtGui.QColor(t["accent_text"]), 2.0)
    pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(3, 7.5), QtCore.QPointF(6, 10.5),
                                          QtCore.QPointF(11, 4)]))
    painter.end()
    path = os.path.join(_arrow_dir, "check_{}.png".format(theme_name()))
    pm.save(path)
    out["check"] = path.replace("\\", "/")
    for name, pts in (("up", ((1, 6), (5, 2), (9, 6))), ("down", ((1, 2), (5, 6), (9, 2)))):
        for state, c in (("", t["muted"]), ("_off", t["border2"])):
            pm = QtGui.QPixmap(10, 8)
            pm.fill(QtCore.Qt.GlobalColor.transparent)
            painter = QtGui.QPainter(pm)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            pen = QtGui.QPen(QtGui.QColor(c), 1.6)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(*p) for p in pts]))
            painter.end()
            path = os.path.join(_arrow_dir, "{}{}_{}.png".format(name, state, theme_name()))
            pm.save(path)
            out[name + state] = path.replace("\\", "/")
    return out


# -- style sheet ------------------------------------------------------------------------

_QSS = """
QWidget {{ color: {text}; }}
QMainWindow, QDialog {{ background: {bg}; }}
QToolTip {{ background: {surface}; color: {text}; border: 1px solid {border2}; padding: 4px 6px;
           border-radius: 4px; }}
QMenuBar {{ background: {surface}; border-bottom: 1px solid {border}; }}
QMenuBar::item {{ padding: 4px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {surface2}; border-radius: 4px; }}
QMenu {{ background: {surface}; border: 1px solid {border2}; padding: 4px; }}
QMenu::item {{ padding: 5px 24px 5px 10px; border-radius: 4px; }}
QMenu::item:selected {{ background: {selection}; }}
QMenu::item:disabled {{ color: {muted}; }}
QMenu::separator {{ height: 1px; background: {border}; margin: 4px 6px; }}
QToolBar {{ background: {surface}; border: none; border-bottom: 1px solid {border};
           spacing: 2px; padding: 3px 6px; }}
QToolBar::separator {{ background: {border}; width: 1px; margin: 4px 6px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 6px;
              padding: 4px 7px; }}
QToolButton:hover {{ background: {surface2}; border-color: {border}; }}
QToolButton:pressed {{ background: {selection}; }}
QToolButton:checked {{ background: {selection}; border-color: {accent}; }}
QToolButton:disabled {{ color: {muted}; }}
QToolButton[kind="primary"] {{ background: {accent}; color: {accent_text}; font-weight: 600; }}
QToolButton[kind="primary"]:hover {{ background: {accent_hover}; }}
QToolButton[kind="danger"] {{ color: {danger}; font-weight: 600; }}
QToolButton[kind="danger"]:hover {{ background: {danger_tint}; border-color: {danger}; }}
QToolButton[kind="danger"]:disabled {{ color: {muted}; font-weight: normal; }}
QToolButton[kind="primary"]:disabled {{ background: {surface2}; color: {muted}; }}
QToolButton[role="button"] {{ background: {surface2}; border: 1px solid {border2};
    border-radius: 6px; padding: 5px 10px; }}
QToolButton[role="button"]:hover {{ border-color: {accent}; }}
QToolButton[role="button"]:checked {{ background: {selection}; border-color: {accent}; }}
QToolButton[role="button"]:disabled {{ color: {muted}; border-color: {border}; }}
QToolButton[role="dpad"] {{ background: {surface2}; border: 1px solid {border2};
    border-radius: 8px; padding: 2px; }}
QToolButton[role="dpad"]:hover {{ border-color: {accent}; }}
QToolButton[role="dpad"]:pressed {{ background: {selection}; }}
QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border: 1px solid {border2}; border-radius: 4px;
    background: {input}; }}
QCheckBox::indicator:hover {{ border-color: {accent}; }}
QCheckBox::indicator:checked {{ background: {accent}; border-color: {accent};
    image: url({check}); }}
QCheckBox::indicator:disabled {{ background: {surface2}; border-color: {border}; }}
QPushButton {{ background: {surface2}; border: 1px solid {border2}; border-radius: 6px;
              padding: 5px 12px; min-height: 18px; }}
QPushButton:hover {{ border-color: {accent}; }}
QPushButton:pressed {{ background: {selection}; }}
QPushButton:checked {{ background: {selection}; border-color: {accent}; }}
QPushButton:disabled {{ color: {muted}; border-color: {border}; }}
QPushButton[kind="primary"] {{ background: {accent}; border-color: {accent};
                              color: {accent_text}; font-weight: 600; }}
QPushButton[kind="primary"]:hover {{ background: {accent_hover}; }}
QPushButton[kind="danger"] {{ background: {danger}; border-color: {danger}; color: #ffffff;
                             font-weight: 600; }}
QPushButton[kind="danger"]:hover {{ background: {danger_hover}; }}
QGroupBox {{ background: {surface}; border: 1px solid {border}; border-radius: 8px;
            margin-top: 16px; padding: 8px 8px 6px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 10px;
                   padding: 0 4px; color: {muted}; font-weight: 600; }}
QTabWidget::pane {{ border: 1px solid {border}; border-radius: 8px; background: {bg};
                   top: -1px; }}
QTabBar::tab {{ background: transparent; color: {muted}; padding: 7px 13px; border: none;
               border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {text}; border-bottom: 2px solid {accent}; font-weight: 600; }}
QTabBar::tab:hover {{ color: {text}; }}
QAbstractSpinBox, QLineEdit, QComboBox {{ background: {input}; border: 1px solid {border2};
    border-radius: 5px; padding: 3px 6px; min-height: 18px; selection-background-color: {accent};
    selection-color: {accent_text}; }}
QAbstractSpinBox:focus, QLineEdit:focus, QComboBox:focus {{ border-color: {accent}; }}
QAbstractSpinBox:disabled, QLineEdit:disabled, QComboBox:disabled {{ color: {muted}; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ width: 16px; border: none;
    background: transparent; }}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; }}
QAbstractSpinBox::up-arrow {{ image: url({arrow_up}); width: 10px; height: 8px; }}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off {{ image: url({arrow_up_off}); }}
QAbstractSpinBox::down-arrow {{ image: url({arrow_down}); width: 10px; height: 8px; }}
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {{
    image: url({arrow_down_off}); }}
QComboBox {{ padding-right: 20px; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox::down-arrow {{ image: url({arrow_down}); width: 10px; height: 8px; }}
QComboBox QAbstractItemView {{ background: {surface}; border: 1px solid {border2};
    selection-background-color: {selection}; selection-color: {text}; outline: 0; }}
QPlainTextEdit, QTextEdit {{ background: {input}; border: 1px solid {border}; border-radius: 6px;
    selection-background-color: {accent}; }}
QListWidget, QTableWidget, QTreeWidget {{ background: {input}; border: 1px solid {border};
    border-radius: 6px; alternate-background-color: {surface}; gridline-color: {border};
    selection-background-color: {selection}; selection-color: {text}; outline: 0; }}
QListWidget::item {{ padding: 3px 4px; }}
QTreeWidget::item {{ padding: 2px 0; }}
QHeaderView::section {{ background: {surface2}; color: {muted}; border: none;
    border-bottom: 1px solid {border}; padding: 4px 6px; font-weight: 600; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle {{ background: {border2}; border-radius: 3px; min-height: 24px;
    min-width: 24px; }}
QScrollBar::handle:hover {{ background: {muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QSplitter::handle {{ background: {bg}; }}
QSplitter::handle:hover {{ background: {border2}; }}
QStatusBar {{ background: {surface}; border-top: 1px solid {border}; color: {muted}; }}
QStatusBar QLabel {{ color: {muted}; }}
QProgressBar {{ background: {input}; border: 1px solid {border}; border-radius: 4px;
    max-height: 8px; text-align: center; }}
QProgressBar::chunk {{ background: {accent}; border-radius: 3px; }}
QLabel[role="muted"] {{ color: {muted}; }}
QLabel[role="heading"] {{ font-weight: 600; }}
QLabel[role="chip"] {{ background: {surface2}; border: 1px solid {border}; border-radius: 9px;
    padding: 1px 9px; }}
QLabel[role="chip"][tone="good"] {{ color: {success}; border-color: {success}; }}
QLabel[role="chip"][tone="warn"] {{ color: {warning}; border-color: {warning}; }}
QLabel[role="chip"][tone="bad"] {{ color: {danger}; border-color: {danger}; }}
QLabel[role="chip"][tone="accent"] {{ color: {accent}; border-color: {accent}; }}
QFrame[role="tile"] {{ background: {surface}; border: 1px solid {border}; border-radius: 8px; }}
QLabel[role="tile-value"] {{ font-size: 15pt; font-weight: 600; }}
QLabel[role="tile-label"] {{ color: {muted}; }}
QFrame[role="bar"] {{ background: {surface}; border: 1px solid {border}; border-radius: 8px; }}
"""


def _palette(t) -> QtGui.QPalette:
    C = QtGui.QColor
    R = QtGui.QPalette.ColorRole
    p = QtGui.QPalette()
    p.setColor(R.Window, C(t["bg"]))
    p.setColor(R.WindowText, C(t["text"]))
    p.setColor(R.Base, C(t["input"]))
    p.setColor(R.AlternateBase, C(t["surface"]))
    p.setColor(R.ToolTipBase, C(t["surface"]))
    p.setColor(R.ToolTipText, C(t["text"]))
    p.setColor(R.Text, C(t["text"]))
    p.setColor(R.Button, C(t["surface2"]))
    p.setColor(R.ButtonText, C(t["text"]))
    p.setColor(R.BrightText, C(t["danger"]))
    p.setColor(R.Highlight, C(t["accent"]))
    p.setColor(R.HighlightedText, C(t["accent_text"]))
    p.setColor(R.Link, C(t["accent"]))
    p.setColor(R.Mid, C(t["border2"]))
    p.setColor(R.Dark, C(t["border"]))
    p.setColor(R.Light, C(t["surface2"]))
    if hasattr(R, "PlaceholderText"):
        p.setColor(R.PlaceholderText, C(t["muted"]))
    for role in (R.WindowText, R.Text, R.ButtonText):
        p.setColor(QtGui.QPalette.ColorGroup.Disabled, role, C(t["muted"]))
    return p


def apply_theme(app, name: str = "dark") -> None:
    """Switch the whole application to theme ``name`` ("dark" or "light")."""
    if name not in THEMES:
        name = "dark"
    _current["name"] = name
    t = THEMES[name]
    app.setStyle("Fusion")
    app.setPalette(_palette(t))
    arrows = _arrow_images(t)
    app.setStyleSheet(_QSS.format(arrow_up=arrows["up"], arrow_up_off=arrows["up_off"],
                                  arrow_down=arrows["down"], arrow_down_off=arrows["down_off"],
                                  check=arrows["check"], **t))
    pg.setConfigOptions(background=t["plot_bg"], foreground=t["muted"])


def style_plot(plot) -> None:
    """Colours of a pyqtgraph PlotWidget for the current theme."""
    t = current()
    plot.setBackground(t["plot_bg"])
    item = plot.getPlotItem()
    for name in ("left", "bottom", "right", "top"):
        ax = item.getAxis(name)
        ax.setPen(pg.mkPen(t["border2"]))
        ax.setTextPen(pg.mkPen(t["muted"]))
        if ax.label is not None:
            ax.label.setDefaultTextColor(QtGui.QColor(t["muted"]))
    legend = item.legend
    if legend is not None:
        legend.setBrush(pg.mkBrush(QtGui.QColor(t["surface"])))
        legend.setPen(pg.mkPen(t["border"]))
        legend.setLabelTextColor(t["text"])
        for _, label in legend.items:           # the labels already there
            label.setText(label.text, color=t["text"])


def chip(label: QtWidgets.QLabel, tone: str = "") -> QtWidgets.QLabel:
    """Show ``label`` as a rounded chip; ``tone``: "", good, warn, bad, accent."""
    label.setProperty("role", "chip")
    if label.property("tone") != tone:
        label.setProperty("tone", tone)
        label.style().unpolish(label)
        label.style().polish(label)
    return label
