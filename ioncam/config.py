"""Configuration: typed dataclasses with defaults, TOML load and save.

Every option has a default, so ioncam runs without any config file.  Write a
fully commented template with ``aqctl_ioncam --write-config ioncam.toml``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ioncam.camera.base import AcquisitionSettings

try:  # Python >= 3.11
    import tomllib as _toml
except ImportError:  # pragma: no cover - Python 3.10
    import tomli as _toml

logger = logging.getLogger(__name__)


def opt(default, help_text: str):
    """Dataclass field with a default value and a help string."""
    if isinstance(default, (list, dict, set)):
        return field(default_factory=lambda d=default: type(d)(d),
                     metadata={"help": help_text})
    return field(default=default, metadata={"help": help_text})


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------

@dataclass
class CameraConfig:
    backend: str = opt("andor", "'andor' (real iXon via pyAndorSDK2), 'simulator' "
                                "(physics simulator) or 'fake_sdk' (andor driver on an "
                                "emulated SDK - for testing the driver itself)")
    sdk_dir: str = opt("", "directory passed to Initialize(); '' = SDK default")
    camera_index: int = opt(0, "which camera to use if several are connected")
    target_temperature: int = opt(-60, "sensor set point [deg C]")
    cooler_on: bool = opt(True, "switch the TEC cooler on at start-up")
    keep_cooler_on_exit: bool = opt(True, "keep the sensor cold after the server exits "
                                          "(SetCoolerMode(1)); avoids thermal cycling")
    fan_mode: int = opt(0, "0 = full, 1 = low, 2 = off (only with water cooling!)")
    em_gain_mode: int = opt(3, "EM gain mode: 0 = DAC 0-255, 1 = DAC 0-4095, 2 = linear, "
                               "3 = real EM gain (what Solis shows for the iXon Ultra)")
    em_advanced: bool = opt(False, "allow EM gains above 300 (ages the sensor; photon "
                                   "counting only)")
    shutter: str = opt("open", "'open', 'auto' or 'closed' (external shutter TTL)")
    shutter_ttl_high: bool = opt(True, "shutter opens on TTL high")
    temperature_poll_s: float = opt(2.0, "how often to read the temperature [s]")


def _live_defaults() -> AcquisitionSettings:
    return AcquisitionSettings(exposure=0.05, em_gain=50, hs_speed_mhz=10.0,
                               vs_speed_us=0.0, preamp_gain_index=2,
                               frame_transfer=True, trigger="internal",
                               acquisition="run_till_abort")


def _locate_defaults() -> AcquisitionSettings:
    # Long exposure, low gain: precise positions without saturating.
    return AcquisitionSettings(exposure=0.5, em_gain=0, hs_speed_mhz=5.0,
                               vs_speed_us=0.5, preamp_gain_index=2,
                               frame_transfer=True, trigger="internal",
                               acquisition="run_till_abort")


def _readout_defaults() -> AcquisitionSettings:
    # The typical Grabber readout settings of an iXon Ultra 897 ion experiment.
    return AcquisitionSettings(exposure=0.0015, em_gain=200, amplifier="em",
                               hs_speed_mhz=5.0, vs_speed_us=0.5, vs_amplitude=0,
                               preamp_gain_index=2, baseline_clamp=True,
                               hbin=4, vbin=4, trigger="external",
                               acquisition="run_till_abort", frame_transfer=False,
                               fast_ext_trigger=False, camera_link=True)


ACQ_HELP = {
    "exposure": "exposure time [s]",
    "em_gain": "EM gain (in camera.em_gain_mode units); 0 = minimum",
    "amplifier": "'em' or 'conventional' output amplifier",
    "hs_speed_mhz": "horizontal readout rate [MHz]; nearest available is used",
    "vs_speed_us": "vertical shift speed [us]; 0 = fastest recommended",
    "vs_amplitude": "vertical clock voltage: 0 = Normal, 1..4 = +1..+4",
    "preamp_gain_index": "pre-amp gain index (0-based); Solis 'Gain 3' = 2",
    "baseline_clamp": "keep the baseline (bias) level constant",
    "hbin": "horizontal binning",
    "vbin": "vertical binning",
    "roi": "acquisition ROI [x0, y0, x1, y1] (0-based, half-open, raw sensor "
           "orientation); omit = full sensor",
    "trigger": "internal | external | external_start | external_exposure | software",
    "acquisition": "single | run_till_abort | accumulate",
    "n_accumulations": "number of accumulations (acquisition = accumulate)",
    "frame_transfer": "frame-transfer readout (no dead time between frames)",
    "fast_ext_trigger": "accept external triggers without finishing a keep-clean cycle",
    "trigger_invert": "trigger on the falling edge",
    "camera_link": "also stream frames out of the Camera Link port (Grabber)",
    "kinetic_cycle_time": "frame period for internal trigger [s]; 0 = fastest",
}


@dataclass
class CalibrationConfig:
    n_frames: int = opt(2, "long-exposure frames averaged to locate the ions")
    search_margin: int = opt(0, "if > 0 and ions are already known, only acquire this "
                                "many px around them (faster); 0 = full sensor/live ROI")
    expected_ions: int = opt(0, "if > 0, calibration fails unless exactly this many "
                                "ions are found")
    roi_size: Tuple[int, int] = opt((0, 0), "Grabber ROI size per ion in binned pixels "
                                            "[w, h]; [0, 0] = choose automatically")
    min_containment: float = opt(0.80, "required fraction of each ion's light inside its ROI")
    max_crosstalk: float = opt(0.05, "maximum fraction of an ion's light in a neighbour's ROI")
    frame_margin: int = opt(1, "binned pixels of margin around the ROIs in the AQROI")
    background_roi: bool = opt(True, "add one extra Grabber ROI on empty background")
    include_dark_sites: bool = opt(True, "also give dark ions known to the tracker a ROI")
    verify: bool = opt(True, "take test frames with the final readout settings "
                             "(internal trigger) and check every ROI before arming")
    verify_frames: int = opt(20, "number of verification frames")
    min_verify_snr: float = opt(3.0, "minimum (ion ROI - background)/noise in verification")
    fit_psf: bool = opt(True, "joint 2D Gaussian fit of all ions (sub-pixel positions, PSF)")


@dataclass
class DetectionConfig:
    psf_sigma: float = opt(1.5, "PSF Gaussian sigma [sensor px]; refined by calibration "
                                "if auto_psf")
    auto_psf: bool = opt(True, "update psf_sigma from calibration fits")
    threshold_sigma: float = opt(6.0, "detection threshold in units of the matched-filter "
                                      "noise")
    min_separation: float = opt(2.5, "minimum distance between two ions [sensor px]")
    background_block: int = opt(32, "block size for the smooth background map [sensor px]")
    analysis_roi: Optional[Tuple[int, int, int, int]] = opt(
        None, "only look for ions inside [x0, y0, x1, y1] (sensor px); omit = everywhere")
    max_spots: int = opt(100, "never report more spots than this")
    cloud_width_factor: float = opt(2.2, "a spot wider than this x PSF is reported as an "
                                         "uncrystallised cloud, not an ion")
    max_rate_hz: float = opt(40.0, "maximum full-detection rate (frames beyond are only "
                                   "used for cheap per-ion counts)")


@dataclass
class TrackerConfig:
    confirm_time: float = opt(0.5, "a change must persist this long before it is reported [s]")
    confirm_frames: int = opt(3, "... and for at least this many analysed frames")
    stable_time: float = opt(1.0, "time without changes before the chain counts as stable [s]")
    match_radius: float = opt(2.0, "max frame-to-frame motion of one ion after removing "
                                   "the common shift [px]")
    max_global_shift: float = opt(25.0, "largest common translation searched for [px]")
    reconfig_fraction: float = opt(0.25, "ions moving by more than this fraction of the "
                                         "local spacing => chain reconfigured")
    reconfig_min_px: float = opt(0.8, "... but by at least this many px")
    dark_level: float = opt(0.25, "site brightness below this fraction of its reference "
                                  "=> dark")
    bright_level: float = opt(0.55, "site brightness above this fraction => bright again")
    melt_level: float = opt(0.35, "total brightness below this fraction (and spots lost) "
                                  "=> crystal melted")
    forget_time: float = opt(30.0, "seconds without any fluorescence before the ions are "
                                   "presumed lost")
    position_smoothing: float = opt(0.15, "EMA weight used to follow slow drifts")
    use_chain_model: bool = opt(True, "use the linear-chain equilibrium model to locate "
                                      "dark ions after a reconfiguration")
    max_dark_inferred: int = opt(3, "maximum number of hidden dark ions to infer")
    hold_timeout: float = opt(120.0, "default maximum duration of hold_tracking() [s]")


@dataclass
class GrabberConfig:
    flip_x: bool = opt(False, "Grabber x runs opposite to the SDK image x (set from "
                              "check_grabber_mapping if needed)")
    flip_y: bool = opt(False, "Grabber y runs opposite to the SDK image y")
    roi_engine_count: int = opt(16, "ROI engines in the Grabber gateware")
    res_width: int = opt(12, "coordinate width of the Grabber gateware (max 2**12-1)")


@dataclass
class SafetyConfig:
    enabled: bool = opt(True, "protect the EMCCD against overexposure at high EM gain")
    saturation_counts: int = opt(60000, "pixel value treated as saturated")
    max_saturated_pixels: int = opt(25, "more saturated pixels than this trips protection")
    safe_em_gain: int = opt(0, "EM gain applied when protection trips (0 = minimum)")
    bright_counts: int = opt(0, "legacy rule: if > 0, also trip when more than "
                                "bright_fraction of all pixels exceed this value")
    bright_fraction: float = opt(0.7, "see bright_counts")


@dataclass
class StorageConfig:
    data_dir: str = opt("ioncam_data", "where calibrations and the event log are written")
    save_calibrations: bool = opt(True, "save every calibration (image, fit, plan)")
    event_log: bool = opt(True, "append all events to <data_dir>/events.jsonl")


@dataclass
class PublishConfig:
    master: str = opt("", "ARTIQ master host to publish ioncam.* datasets to "
                          "(empty = disabled)")
    port: int = opt(3251, "ARTIQ master control port")
    prefix: str = opt("ioncam.", "dataset name prefix")
    interval: float = opt(1.0, "publish interval [s]")


@dataclass
class DisplayConfig:
    rotation: int = opt(90, "display rotation, degrees clockwise (0, 90, 180, 270); "
                            "display only - never affects ROIs")
    flip_x: bool = opt(False, "mirror the display horizontally")
    flip_y: bool = opt(False, "mirror the display vertically")
    colormap: str = opt("inferno", "pyqtgraph colormap name")
    level_percentiles: Tuple[float, float] = opt((1.0, 99.95), "auto-levels percentiles")
    max_fps: float = opt(25.0, "GUI refresh limit")


@dataclass
class SimulatorConfig:
    n_ions: int = opt(3, "ions present at start")
    center: Tuple[float, float] = opt((262.0, 168.0), "chain centre [sensor px]")
    axis_angle_deg: float = opt(0.0, "chain axis angle w.r.t. sensor x [deg]")
    length_scale_px: float = opt(7.4, "chain length scale l = (e^2/(4 pi eps0 m w^2))^(1/3) "
                                      "in px; 2 ions are 1.26 l apart, 3 ions 1.08 l")
    psf_sigma: float = opt(1.3, "PSF sigma [px]")
    photon_rate: float = opt(4.0e5, "detected photons/s per bright ion")
    background_rate: float = opt(150.0, "scattered-light photons/s/px (plus a gradient)")
    bias: float = opt(100.0, "baseline counts per (binned) pixel")
    read_noise_e: float = opt(30.0, "read noise [e-] rms")
    sensitivity: float = opt(4.0, "electrons per count at the output")
    cic: float = opt(0.003, "clock-induced charge [e-/px/frame]")
    load_rate: float = opt(0.0, "ions loaded per second while the oven is on")
    loss_rate: float = opt(0.0, "loss probability per ion per second")
    dark_rate: float = opt(0.0, "go-dark probability per bright ion per second")
    bright_rate: float = opt(0.0, "return-to-bright probability per dark ion per second")
    melt_rate: float = opt(0.0, "crystal melting events per second")
    drift_px_per_s: float = opt(0.0, "random-walk drift of the trap centre [px/sqrt(s)]")
    auto_trigger_hz: float = opt(0.0, "external-trigger rate when nobody triggers "
                                      "(0 = only sim_trigger())")
    seed: int = opt(-1, "random seed (-1 = random)")


SECTIONS: Dict[str, type] = {
    "camera": CameraConfig,
    "live": AcquisitionSettings,
    "locate": AcquisitionSettings,
    "readout": AcquisitionSettings,
    "calibration": CalibrationConfig,
    "detection": DetectionConfig,
    "tracker": TrackerConfig,
    "grabber": GrabberConfig,
    "safety": SafetyConfig,
    "storage": StorageConfig,
    "publish": PublishConfig,
    "display": DisplayConfig,
    "simulator": SimulatorConfig,
}

SECTION_HELP = {
    "camera": "Camera hardware",
    "live": "Live video (trapping, monitoring): settings for internal-trigger video",
    "locate": "Long-exposure, low-gain frames used to locate the ions precisely",
    "readout": "Grabber readout (external trigger + Camera Link).  AQROI is computed "
               "automatically; the roi field here is ignored",
    "calibration": "How the readout calibration (locate -> plan -> verify -> arm) works",
    "detection": "Ion spot detection",
    "tracker": "Chain tracking and event classification (trapped / lost / dark)",
    "grabber": "ARTIQ Grabber geometry",
    "safety": "Sensor protection",
    "storage": "Files",
    "publish": "Optional publishing of live numbers to ARTIQ datasets",
    "display": "GUI defaults",
    "simulator": "Simulated trap + camera (backend = 'simulator' or --simulate)",
}


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    live: AcquisitionSettings = field(default_factory=_live_defaults)
    locate: AcquisitionSettings = field(default_factory=_locate_defaults)
    readout: AcquisitionSettings = field(default_factory=_readout_defaults)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    grabber: GrabberConfig = field(default_factory=GrabberConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    publish: PublishConfig = field(default_factory=PublishConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)

    # -- access -------------------------------------------------------------

    def section(self, name: str):
        if name not in SECTIONS:
            raise KeyError("unknown config section '{}'".format(name))
        return getattr(self, name)

    def get(self, section: str, key: str):
        sec = self.section(section)
        if key not in _field_names(sec):
            raise KeyError("unknown key '{}' in [{}]".format(key, section))
        return getattr(sec, key)

    def set(self, section: str, key: str, value) -> Any:
        """Set one option with type coercion; returns the stored value."""
        sec = self.section(section)
        fdefs = {f.name: f for f in dataclasses.fields(sec)}
        if key not in fdefs:
            raise KeyError("unknown key '{}' in [{}]".format(key, section))
        coerced = _coerce(value, _field_default(fdefs[key]), "{}.{}".format(section, key))
        if isinstance(sec, AcquisitionSettings):
            # validate a candidate first: a rejected value must not stay in the config
            dataclasses.replace(sec, **{key: coerced}).validate()
        setattr(sec, key, coerced)
        return coerced

    def to_dict(self) -> dict:
        out = {}
        for name in SECTIONS:
            sec = getattr(self, name)
            d = {}
            for f in dataclasses.fields(sec):
                v = getattr(sec, f.name)
                d[f.name] = list(v) if isinstance(v, tuple) else v
            out[name] = d
        return out

    # -- files --------------------------------------------------------------

    @classmethod
    def load(cls, path: Optional[str]) -> "Config":
        cfg = cls()
        if not path:
            return cfg
        with open(path, "rb") as f:
            data = _toml.load(f)
        cfg.update_from_dict(data, source=path)
        return cfg

    def update_from_dict(self, data: dict, source: str = "<dict>") -> List[str]:
        warnings = []
        for sec_name, values in data.items():
            if sec_name not in SECTIONS:
                warnings.append("{}: unknown section [{}] ignored".format(source, sec_name))
                continue
            if not isinstance(values, dict):
                warnings.append("{}: [{}] must be a table".format(source, sec_name))
                continue
            for key, value in values.items():
                try:
                    self.set(sec_name, key, value)
                except (KeyError, ValueError, TypeError) as e:
                    warnings.append("{}: [{}] {}: {}".format(source, sec_name, key, e))
        for w in warnings:
            logger.warning(w)
        return warnings

    def save(self, path: str, with_help: bool = True) -> None:
        text = self.to_toml(with_help=with_help)
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)

    def to_toml(self, with_help: bool = True) -> str:
        lines = ["# ioncam configuration", ""]
        for name in SECTIONS:
            sec = getattr(self, name)
            if with_help:
                lines.append("# " + SECTION_HELP.get(name, name))
            lines.append("[{}]".format(name))
            for f in dataclasses.fields(sec):
                value = getattr(sec, f.name)
                help_text = f.metadata.get("help") or (
                    ACQ_HELP.get(f.name) if isinstance(sec, AcquisitionSettings) else None)
                if with_help and help_text:
                    lines.append("# " + help_text)
                if value is None:
                    lines.append("# {} = ".format(f.name))
                else:
                    lines.append("{} = {}".format(f.name, _toml_value(value)))
            lines.append("")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _field_names(obj) -> set:
    return {f.name for f in dataclasses.fields(obj)}


def _field_default(f: dataclasses.Field):
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return f.default_factory()  # type: ignore[misc]
    return None


def _coerce(value, default, name: str):
    """Coerce ``value`` to the type of ``default`` (None default = optional int tuple)."""
    if default is None:
        if value is None or value == "" or (isinstance(value, (list, tuple)) and not value):
            return None
        if isinstance(value, (list, tuple)):
            if len(value) != 4:
                raise ValueError("{} must be [x0, y0, x1, y1]".format(name))
            return tuple(int(round(float(v))) for v in value)
        raise TypeError("{} must be a list or empty".format(name))
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("1", "true", "yes", "on"):
                return True
            if v in ("0", "false", "no", "off"):
                return False
            raise ValueError("{}: not a boolean: {!r}".format(name, value))
        if isinstance(value, (int, float)):
            return bool(value)
        raise TypeError("{}: not a boolean: {!r}".format(name, value))
    if isinstance(default, int):
        if isinstance(value, bool):
            raise TypeError("{}: expected an integer".format(name))
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError("{}: expected an integer, got {}".format(name, value))
            return int(value)
        return int(value)
    if isinstance(default, float):
        if isinstance(value, bool):
            raise TypeError("{}: expected a number".format(name))
        return float(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise TypeError("{}: expected a string".format(name))
        return value
    if isinstance(default, tuple):
        if not isinstance(value, (list, tuple)) or len(value) != len(default):
            raise ValueError("{}: expected a list of {} values".format(name, len(default)))
        return tuple(_coerce(v, d, name) for v, d in zip(value, default))
    if isinstance(default, list):
        return list(value)
    return value


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v:
            return "nan"
        if v in (float("inf"), float("-inf")):
            return "inf" if v > 0 else "-inf"
        r = repr(v)
        return r if any(c in r for c in ".en") else r + ".0"
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError("cannot write {!r} to TOML".format(v))
