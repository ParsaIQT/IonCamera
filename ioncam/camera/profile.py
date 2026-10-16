"""A camera profile: the dark-frame model of a real camera at every readout
setting a camera characterization measured (:mod:`ioncam.characterize`), in
a small JSON file (``camera_profile.json``).

Each measured setting has the model of :mod:`ioncam.analysis.darkframes`
for one (super-)pixel: ``bias`` and read noise (counts), the electrons per
count (``sensitivity``), the counts per input electron ``g`` (EM gain /
sensitivity), parallel CIC ``lam_p`` and serial CIC ``lam_s`` (events per
pixel per frame).  The camera-wide part: dark current (electrons/pixel/s at
the characterization's temperature, and against the temperature if that was
measured), the real EM gain / the set one, hot pixels.

The simulator (``simulator.camera_profile``) uses the nearest measured
setting; your own simulations can read the file directly.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional

FORMAT = "ioncam camera profile 1"


def _log_ratio(a: float, b: float) -> float:
    return abs(math.log(max(float(a), 1e-9) / max(float(b), 1e-9)))


class CameraProfile:
    """A loaded ``camera_profile.json``."""

    def __init__(self, data: dict, path: str = ""):
        if data.get("format") != FORMAT:
            raise ValueError("{}: not an ioncam camera profile".format(path or "profile"))
        self.data = data
        self.path = path
        self.points: List[dict] = [p for p in data.get("points", []) if p.get("g") is not None
                                   or p.get("amplifier") == "conventional"]

    @classmethod
    def load(cls, path: str) -> "CameraProfile":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f), path)

    @property
    def dark_current(self) -> float:
        return float(self.data.get("dark_current") or 0.0)

    @property
    def temperature(self) -> Optional[float]:
        t = self.data.get("temperature")
        return None if t is None else float(t)

    def dark_current_at(self, temperature: Optional[float]) -> float:
        """Dark current [e-/px/s] at a sensor temperature: interpolated in log
        between measured temperatures, else doubling every 6.3 C from the
        characterization's temperature."""
        d0 = self.dark_current
        table = sorted((float(t), float(d)) for t, d in self.data.get(
            "dark_current_temperature", []) if d and d > 0)
        if temperature is None:
            return d0
        if len(table) >= 2:
            ts = [t for t, _ in table]
            if temperature <= ts[0]:
                (t1, d1), (t2, d2) = table[0], table[1]
            elif temperature >= ts[-1]:
                (t1, d1), (t2, d2) = table[-2], table[-1]
            else:
                i = max(k for k in range(len(ts)) if ts[k] <= temperature)
                (t1, d1), (t2, d2) = table[i], table[i + 1]
            if t2 != t1:
                k = (math.log(d2) - math.log(d1)) / (t2 - t1)
                return float(math.exp(math.log(d1) + k * (temperature - t1)))
        t0 = self.temperature
        if t0 is None or d0 <= 0:
            return d0
        return d0 * 2.0 ** ((temperature - t0) / 6.3)

    def nearest(self, amplifier: str, hs_speed_mhz: float, preamp_gain_index: int,
                vs_speed_us: float, vs_amplitude: int, em_gain: float) -> Optional[dict]:
        """The measured setting closest to these (readout amplifier, speeds,
        pre-amp gain and clock voltage first, then EM gain), with ``g`` and
        serial CIC moved to ``em_gain``."""
        cands = [p for p in self.points if p.get("amplifier", "em") == amplifier]
        if not cands:
            return None

        def dist(p):
            return (10.0 * _log_ratio(p["hs_speed_mhz"], hs_speed_mhz) +
                    5.0 * (int(p.get("preamp_gain_index", 0)) != int(preamp_gain_index)) +
                    3.0 * _log_ratio(p["vs_speed_us"], vs_speed_us) +
                    2.0 * abs(int(p.get("vs_amplitude", 0)) - int(vs_amplitude)) +
                    1.0 * _log_ratio(max(p.get("em_gain", 1), 1), max(em_gain, 1)) +
                    0.5 * (p.get("hbin", 1) * p.get("vbin", 1) != 1))

        p = dict(min(cands, key=dist))
        if amplifier == "em" and p.get("g") and p.get("em_gain"):
            r = max(float(em_gain), 1.0) / max(float(p["em_gain"]), 1.0)
            p["g"] = float(p["g"]) * r
            p["lam_s"] = float(p.get("lam_s") or 0.0) * r
        # per pixel (a binned point's CIC is spread over its pixels)
        area = int(p.get("hbin", 1)) * int(p.get("vbin", 1))
        if area > 1:
            p["lam_p"] = float(p.get("lam_p") or 0.0) / area
        return p


def write_profile(path: str, camera: dict, points: List[dict], temperature: Optional[float],
                  dark_current: float, dark_current_temperature: List[List[float]],
                  gain_ratio: Optional[float], hot_pixels: List[list], created: str,
                  notes: Optional[Dict] = None) -> None:
    data = {"format": FORMAT, "created": created, "camera": camera,
            "temperature": temperature, "dark_current": dark_current,
            "dark_current_temperature": dark_current_temperature,
            "gain_ratio": gain_ratio, "hot_pixels": hot_pixels,
            "model": "x = bias + N(0, read_noise) + EM output of Poisson(lam_p) electrons "
                     "(Gamma(n, g) counts) + Poisson(lam_s) serial-CIC electrons with "
                     "log-uniform gain between 1 electron (g/em_gain counts) and g "
                     "(Lantz et al. 2008, appendix A); per (super-)pixel and frame",
            "points": points}
    if notes:
        data["notes"] = notes
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
