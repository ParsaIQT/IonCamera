"""Camera abstraction shared by the Andor driver and the simulator.

Coordinate conventions (used everywhere in ioncam)
--------------------------------------------------
* **Sensor coordinates** are 0-based ``(x, y)`` = ``(column, row)`` in the
  *raw* readout orientation of the sensor, i.e. exactly what the SDK returns
  with no ``SetImageRotate``/``SetImageFlip``.  The Camera Link stream that
  the ARTIQ Grabber sees is in this orientation too (Andor applies rotate and
  flip in host software, never in the camera), so keeping all internal
  geometry raw is what makes Grabber ROIs come out right.  Rotation/flip for
  humans is a display-only option in the GUI.
* A pixel ``k`` has its centre at coordinate ``k`` and spans
  ``[k - 0.5, k + 0.5)``.
* ROIs are ``(x0, y0, x1, y1)`` half-open sensor rectangles.  The SDK's
  1-based inclusive ``SetImage(hbin, vbin, hstart, hend, vstart, vend)``
  arguments are ``hstart = x0 + 1, hend = x1, vstart = y0 + 1, vend = y1``.
  Andor Solis displays the same numbers as "Left, Right, Bottom, Top".
* A frame array has shape ``(H, W)`` and is indexed ``frame[v, u]`` where
  ``(u, v)`` are (binned) frame pixel indices; ``v = 0`` is the first row read
  out.  The centre of frame pixel ``(u, v)`` is at sensor coordinate
  ``x = x0 + hbin*u + (hbin - 1)/2`` (same for y).
"""

from __future__ import annotations

import abc
import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

TRIGGER_MODES = ("internal", "external", "external_start", "external_exposure",
                 "software")
#: shutter modes (SDK SetShutter / SetShutterEx): permanently open, permanently
#: closed, fully automatic (opens and closes around every exposure - for the
#: iXon Ultra's shutter at most ~2 Hz, so not for video), open while acquiring
SHUTTER_MODES = {"open": 1, "closed": 2, "auto": 0, "series": 5}
#: fan modes (SDK SetFanMode)
FAN_MODES = {"full": 0, "low": 1, "off": 2}
ACQUISITION_MODES = ("single", "run_till_abort", "accumulate")
AMPLIFIERS = ("em", "conventional")


class CameraError(RuntimeError):
    """Raised for any camera/driver level failure."""


Roi = Tuple[int, int, int, int]


@dataclass
class AcquisitionSettings:
    """Everything needed to (re)configure the camera for one operating mode.

    The defaults are a reasonable live-view setting for an iXon Ultra 897.
    """
    exposure: float = 0.05            # s
    em_gain: int = 50                 # EM gain in the configured EM gain mode; 0 -> minimum
    amplifier: str = "em"             # "em" (electron multiplying) | "conventional"
    hs_speed_mhz: float = 10.0        # horizontal readout rate; nearest available is used
    vs_speed_us: float = 0.0          # vertical shift speed [us/row]; 0 -> fastest recommended
    vs_amplitude: int = 0             # vertical clock voltage: 0 = Normal, 1..4 = +1..+4
    preamp_gain_index: int = -1       # SDK pre-amp gain index (0-based; Solis "Gain 3" == 2);
                                      # -1 = the highest available at the readout speed
    baseline_clamp: bool = True
    hbin: int = 1
    vbin: int = 1
    roi: Optional[Roi] = None         # (x0, y0, x1, y1) sensor px, half-open; None = full sensor
    trigger: str = "internal"         # see TRIGGER_MODES
    acquisition: str = "run_till_abort"  # see ACQUISITION_MODES
    n_accumulations: int = 1          # only for acquisition == "accumulate"
    frame_transfer: bool = True       # FT on = no dead time between frames (internal trigger)
    fast_ext_trigger: bool = False    # don't wait for keep-clean cycle before accepting trigger
    trigger_invert: bool = False      # trigger on falling edge
    camera_link: bool = False         # stream every frame out of the Camera Link port too
    kinetic_cycle_time: float = 0.0   # 0 -> as fast as possible

    def replace(self, **changes) -> "AcquisitionSettings":
        unknown = set(changes) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise ValueError("unknown acquisition setting(s): {}".format(
                ", ".join(sorted(unknown))))
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["roi"] = list(self.roi) if self.roi is not None else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AcquisitionSettings":
        d = dict(d)
        if d.get("roi") is not None:
            d["roi"] = tuple(int(v) for v in d["roi"])
        return cls(**d)

    def validate(self) -> None:
        if self.trigger not in TRIGGER_MODES:
            raise ValueError("trigger must be one of {}".format(TRIGGER_MODES))
        if self.acquisition not in ACQUISITION_MODES:
            raise ValueError("acquisition must be one of {}".format(ACQUISITION_MODES))
        if self.amplifier not in AMPLIFIERS:
            raise ValueError("amplifier must be one of {}".format(AMPLIFIERS))
        if not (self.exposure >= 0.0):
            raise ValueError("exposure must be >= 0")
        if self.hbin < 1 or self.vbin < 1:
            raise ValueError("binning must be >= 1")
        if self.em_gain < 0:
            raise ValueError("em_gain must be >= 0")
        if self.n_accumulations < 1:
            raise ValueError("n_accumulations must be >= 1")


def normalize_roi(roi: Optional[Sequence[int]], sensor_shape: Tuple[int, int],
                  hbin: int = 1, vbin: int = 1) -> Roi:
    """Clamp ``roi`` to the sensor and make its size a multiple of the binning.

    ``sensor_shape`` is ``(height, width)``.  The ROI is shrunk (never moved
    off its start) so that width/height are multiples of ``hbin``/``vbin``;
    if that would make it empty it is grown instead, staying on the sensor.
    """
    h, w = sensor_shape
    if roi is None:
        x0, y0, x1, y1 = 0, 0, w, h
    else:
        x0, y0, x1, y1 = (int(round(v)) for v in roi)
    x0, x1 = sorted((max(0, min(w, x0)), max(0, min(w, x1))))
    y0, y1 = sorted((max(0, min(h, y0)), max(0, min(h, y1))))

    def fit(a, b, size, n):
        length = max(n, ((b - a) // n) * n)
        if a + length > size:
            a = max(0, size - length)
        return a, min(size, a + length)

    x0, x1 = fit(x0, x1, w, hbin)
    y0, y1 = fit(y0, y1, h, vbin)
    if (x1 - x0) % hbin or (y1 - y0) % vbin:
        raise ValueError("cannot fit binning {}x{} into the sensor".format(hbin, vbin))
    return x0, y0, x1, y1


def roi_to_sdk(roi: Roi) -> Tuple[int, int, int, int]:
    """Half-open 0-based ROI -> SDK/Solis 1-based inclusive (hstart, hend, vstart, vend)."""
    x0, y0, x1, y1 = roi
    return x0 + 1, x1, y0 + 1, y1


def roi_from_sdk(hstart: int, hend: int, vstart: int, vend: int) -> Roi:
    """SDK/Solis "Left, Right, Bottom, Top" (1-based inclusive) -> half-open ROI."""
    return hstart - 1, vstart - 1, hend, vend


def nearest_index(values, target) -> int:
    """Index of the value closest to ``target`` by ratio: readout speeds go in
    factors, so 5 MHz asked of an iXon Ultra 888 (30, 20, 10, 1 MHz) is 10 MHz,
    not 1 MHz (ten times slower); the 897 has 5 MHz itself."""
    v = np.asarray(values, float)
    t = float(target)
    if t <= 0 or np.any(v <= 0):
        return int(np.argmin(np.abs(v - t)))
    return int(np.argmin(np.abs(np.log(v / t))))


def vs_speed_index(speeds, fastest_recommended: int, requested_us: float,
                   amplitude: int) -> Tuple[int, str]:
    """The vertical shift speed to use: ``(index into speeds, warning or "")``.

    * 0: the camera's fastest recommended speed (``GetFastestRecommendedVSSpeed``).
    * One of the camera's speeds: exactly that one.  Faster than recommended
      at normal vertical clock voltage (``amplitude`` 0) it is still used, with
      a warning: an over-clocked speed may need a higher voltage to shift the
      charge cleanly (an iXon Ultra 888 at 0.6 us does not transfer it at all
      and every image is garbled).
    * A speed the camera does not have (an 897's 0.5 us on an 888): never
      guessed - the fastest recommended one, with a warning.
    """
    rec = min(max(int(fastest_recommended), 0), len(speeds) - 1)
    if requested_us <= 0:
        return rec, ""
    exact = [k for k, v in enumerate(speeds) if abs(v - requested_us) <= 0.02 * requested_us]
    if not exact:
        return rec, ("vertical shift speed {:g} us is not one of this camera's ({} us): using "
                     "{:g} us, its fastest recommended - choose one in the Readout tab".format(
                         requested_us, ", ".join("{:g}".format(v) for v in speeds),
                         speeds[rec]))
    i = exact[0]
    if amplitude == 0 and speeds[i] < speeds[rec]:
        return i, ("vertical shift speed {:g} us is faster than the {:g} us this camera "
                   "recommends at normal vertical clock voltage: if images look smeared or "
                   "garbled, raise the vertical clock voltage or choose a slower "
                   "speed".format(speeds[i], speeds[rec]))
    return i, ""


@dataclass
class AppliedSettings:
    """What the camera actually ended up with after :meth:`CameraBase.configure`."""
    settings: AcquisitionSettings
    roi: Roi                          # normalized ROI actually used
    frame_shape: Tuple[int, int]      # (H, W) of the returned frames
    exposure: float                   # actual exposure [s]
    accumulate_cycle: float = 0.0
    kinetic_cycle: float = 0.0        # actual frame period for internal trigger [s]
    readout_time: float = 0.0
    keep_clean_time: float = 0.0
    hs_speed_mhz: float = 0.0
    vs_speed_us: float = 0.0
    preamp_gain: float = 1.0
    em_gain: int = 0
    em_gain_range: Tuple[int, int] = (0, 0)
    tag: str = ""                     # operating mode this configuration belongs to
    warnings: List[str] = field(default_factory=list)
    sensitivity: float = 0.0          # electrons per count at this readout speed and pre-amp
                                      # gain (0: unknown)
    sensitivity_source: str = ""      # "camera" (GetSensitivity), "configuration", "simulator"
    em_gain_real: bool = True         # em_gain is the real multiplication (RealGain mode)

    @property
    def x0(self) -> int:
        return self.roi[0]

    @property
    def y0(self) -> int:
        return self.roi[1]

    @property
    def hbin(self) -> int:
        return self.settings.hbin

    @property
    def vbin(self) -> int:
        return self.settings.vbin

    @property
    def gain_factor(self) -> float:
        """Counts of a frame per unit of brightness, used to normalise it:
        exposure x EM gain / electrons per count - normalised brightness is
        then photoelectrons per second, the same at any exposure, EM gain,
        readout speed or pre-amp gain.  Without a known sensitivity: exposure
        x EM gain (arbitrary units)."""
        g = max(1, self.em_gain) if self.settings.amplifier == "em" else 1
        f = max(self.exposure, 1e-6) * g
        return f / self.sensitivity if self.sensitivity > 0 else f

    @property
    def pe_factor(self) -> float:
        """Photoelectrons per second per count of signal in one frame (0 if
        unknown): Andor's (counts - offset) x sensitivity / EM gain gives the
        photoelectrons of an exposure, divided by the exposure time."""
        if self.sensitivity <= 0 or not self.em_gain_real:
            return 0.0
        g = max(1, self.em_gain) if self.settings.amplifier == "em" else 1
        return self.sensitivity / (max(self.exposure, 1e-6) * g)

    @property
    def min_trigger_period(self) -> float:
        """Shortest safe spacing of external triggers (exposure + readout)."""
        return self.exposure + self.readout_time + self.keep_clean_time

    def to_dict(self) -> dict:
        return {
            "settings": self.settings.to_dict(),
            "roi": list(self.roi),
            "sdk_image": [self.settings.hbin, self.settings.vbin,
                          *roi_to_sdk(self.roi)],
            "frame_shape": list(self.frame_shape),
            "exposure": self.exposure,
            "accumulate_cycle": self.accumulate_cycle,
            "kinetic_cycle": self.kinetic_cycle,
            "readout_time": self.readout_time,
            "keep_clean_time": self.keep_clean_time,
            "min_trigger_period": self.min_trigger_period,
            "hs_speed_mhz": self.hs_speed_mhz,
            "vs_speed_us": self.vs_speed_us,
            "preamp_gain": self.preamp_gain,
            "em_gain": self.em_gain,
            "em_gain_range": list(self.em_gain_range),
            "tag": self.tag,
            "warnings": list(self.warnings),
            "sensitivity": self.sensitivity,
            "sensitivity_source": self.sensitivity_source,
            "em_gain_real": self.em_gain_real,
            "pe_factor": self.pe_factor,
        }


class Frame:
    """One image plus the geometry/settings needed to interpret it."""
    __slots__ = ("data", "index", "timestamp", "applied", "cam_index")

    def __init__(self, data: np.ndarray, index: int, timestamp: float,
                 applied: AppliedSettings, cam_index: int = 0):
        self.data = data                  # (H, W) uint16, raw orientation
        self.index = index                # global, monotonically increasing
        self.timestamp = timestamp        # unix time the frame was received
        self.applied = applied
        self.cam_index = cam_index        # SDK image index within the acquisition

    @property
    def tag(self) -> str:
        return self.applied.tag

    @property
    def x0(self) -> int:
        return self.applied.roi[0]

    @property
    def y0(self) -> int:
        return self.applied.roi[1]

    @property
    def hbin(self) -> int:
        return self.applied.settings.hbin

    @property
    def vbin(self) -> int:
        return self.applied.settings.vbin

    def to_sensor(self, u, v):
        """Frame pixel coordinates (may be fractional) -> sensor coordinates."""
        hb, vb = self.hbin, self.vbin
        return (self.x0 + hb * np.asarray(u) + 0.5 * (hb - 1),
                self.y0 + vb * np.asarray(v) + 0.5 * (vb - 1))

    def from_sensor(self, x, y):
        """Sensor coordinates -> fractional frame pixel coordinates."""
        hb, vb = self.hbin, self.vbin
        return ((np.asarray(x) - self.x0 - 0.5 * (hb - 1)) / hb,
                (np.asarray(y) - self.y0 - 0.5 * (vb - 1)) / vb)


class CameraBase(abc.ABC):
    """Minimal camera interface.

    **Threading contract:** every method except :meth:`cancel_wait` is called
    from a single thread (the engine's camera thread).  Implementations do not
    need internal locking; :meth:`cancel_wait` must be safe to call from any
    thread and must make a pending :meth:`wait_frames` return promptly.
    """

    #: (height, width) of the full sensor
    sensor_shape: Tuple[int, int] = (512, 512)

    @abc.abstractmethod
    def open(self) -> dict:
        """Connect/initialise; return an info dict (model, serial, ...)."""

    @abc.abstractmethod
    def close(self) -> None:
        """Stop acquiring and release the hardware."""

    @abc.abstractmethod
    def configure(self, settings: AcquisitionSettings, tag: str = "") -> AppliedSettings:
        """Apply ``settings`` (camera must be idle).  Returns the actual values."""

    @abc.abstractmethod
    def start(self) -> None:
        """Start the configured acquisition."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Abort any running acquisition (no error if idle)."""

    @abc.abstractmethod
    def is_acquiring(self) -> bool:
        ...

    @abc.abstractmethod
    def wait_frames(self, timeout: float) -> List[Tuple[int, np.ndarray]]:
        """Block up to ``timeout`` seconds; return all new frames (oldest first).

        Returns a list of ``(cam_index, array)`` tuples where ``array`` is a
        ``(H, W)`` uint16 image in raw sensor orientation and ``cam_index`` is
        the 1-based image number within the current acquisition.  Returns an
        empty list on timeout or after :meth:`cancel_wait`.
        """

    @abc.abstractmethod
    def cancel_wait(self) -> None:
        """Thread-safe: wake up a blocked :meth:`wait_frames`."""

    @abc.abstractmethod
    def get_temperature(self) -> Tuple[float, str]:
        """Return ``(temperature_C, status)``; status in
        {"stabilized", "not_reached", "not_stabilized", "drift", "off", "unknown"}."""

    @abc.abstractmethod
    def set_temperature(self, temperature: float) -> None:
        ...

    @abc.abstractmethod
    def set_cooler(self, on: bool) -> None:
        ...

    def send_software_trigger(self) -> None:
        raise CameraError("software trigger not supported by this camera")

    def set_shutter(self, mode: str = "open", external: str = "open", ttl_high: bool = True,
                    open_ms: int = 0, close_ms: int = 0) -> None:
        """The shutter (camera idle): ``mode`` of the internal one and
        ``external`` of the external shutter output (when the camera controls
        them separately), from :data:`SHUTTER_MODES`; ``ttl_high``: the output
        goes high to open; opening / closing times [ms]."""
        raise CameraError("this camera has no shutter control")

    def set_fan(self, mode: str) -> None:
        """The cooling fan (camera idle): "full", "low" or "off"."""
        raise CameraError("this camera has no fan control")

    def capabilities(self) -> dict:
        """Readout speeds, gains etc. (for GUIs and diagnostics)."""
        return {}
