"""A physically sensible stand-in for "iXon Ultra 897 looking at a trapped ion chain".

Used for development, automated tests, demos and operator training.  It is
also how the whole ioncam stack (including the RPC API used by ARTIQ) can be
exercised without hardware: run ``aqctl_ioncam --simulate``.

Physics modelled
----------------
* Linear chain in a harmonic potential (James equilibrium positions), any
  axis angle; ions keep their order and identity.
* Bright/dark ions (a dark ion keeps its place), loading through a hot,
  uncrystallised cloud, losses (the chain re-arranges), melting events
  (broadened, dim, possibly re-ordered or lost), drifts, shuttling (centre /
  scale changes), repumper and lasers on/off.
* Realistic brightness: ~13k photons/s per ion reach the camera, times the
  quantum efficiency; a pixel-integrated PSF with a Gaussian core *and* a
  broad halo (what makes neighbouring ROIs see each other's ions).
* The trap surface, out of focus by the ion height: diffuse laser scatter
  with an adjustable Gaussian profile along the beam, defects / dust and
  electrode gaps blurred into disks of radius h tan(asin NA) M / pixel (with
  small NA x h they look just like ions), scratches, hot pixels.  Presets
  from a clean to a horrendous surface (``SURFACE_PRESETS``).
* Laser intensity noise (ions and scatter together).
* EMCCD: Poisson photo-electrons + CIC, Gamma-distributed EM register
  output, read noise, bias, EM-register and 16-bit ADC saturation, on-chip
  binning, ROI, frame-transfer/external-trigger timing.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import erfc, ndtr

from ioncam.analysis.chain import equilibrium_positions
from ioncam.camera.base import (AcquisitionSettings, AppliedSettings, CameraBase,
                                CameraError, normalize_roi)

SENSOR_SHAPE = (512, 512)
EM_HS_SPEEDS = [17.0, 10.0, 5.0, 1.0]       # MHz, EM amplifier (iXon Ultra 897)
CONV_HS_SPEEDS = [3.0, 1.0, 0.08]           # MHz, conventional amplifier
VS_SPEEDS = [0.3, 0.5, 0.9, 1.7, 3.3]       # us
VS_FASTEST_RECOMMENDED = 1                  # 0.5 us at Normal amplitude
PREAMP_GAINS = [1.0, 2.0, 3.0]
EM_GAIN_RANGE = (1, 300)
EM_REGISTER_FULL_WELL = 8.0e5               # e-

# Trap surfaces, from freshly made to "riddled with defects".  Rates are at the
# beam centre; ``defects`` counts the whole sensor (only those in the beam shine).
SURFACE_PRESETS = {
    "clean": dict(scatter_rate=8.0, defects=6, defect_rate=1500.0, defect_spread=0.8,
                  defect_max_height_um=0.0, scratches=0, gap_scatter=1.0, hot_pixels=3),
    "typical": dict(scatter_rate=25.0, defects=25, defect_rate=3000.0, defect_spread=1.2,
                    defect_max_height_um=0.0, scratches=0, gap_scatter=4.0, hot_pixels=8),
    "dirty": dict(scatter_rate=80.0, defects=150, defect_rate=6000.0, defect_spread=1.4,
                  defect_max_height_um=10.0, scratches=3, gap_scatter=15.0, hot_pixels=15),
    "horrendous": dict(scatter_rate=250.0, defects=600, defect_rate=10000.0,
                       defect_spread=1.6, defect_max_height_um=30.0, scratches=8,
                       gap_scatter=40.0, hot_pixels=30),
}

# parameters that change the static (surface) part of the scene
_SURFACE_KEYS = {"background_rate", "scatter_rate", "scatter_sigma_along",
                 "scatter_sigma_across", "scatter_offset", "beam_angle_deg", "defects",
                 "defect_rate", "defect_spread", "defect_max_height_um", "scratches",
                 "gap_scatter", "gap_offset_px", "hot_pixels", "hot_pixel_rate",
                 "surface_seed", "magnification", "pixel_size_um", "numerical_aperture",
                 "ion_height_um", "psf_sigma"}


def defocus_radius_px(height_um: float, na: float, magnification: float,
                      pixel_um: float) -> float:
    """Radius of the blur disk of a point ``height_um`` out of focus (geometric optics)."""
    na = min(max(na, 0.0), 0.99)
    return max(height_um, 0.0) * math.tan(math.asin(na)) * magnification / pixel_um


def blur_kernel(radius: float, sigma: float) -> np.ndarray:
    """Image of a defocused point: a disk of ``radius`` softened by a Gaussian PSF."""
    sigma = max(sigma, 0.3)
    half = int(math.ceil(radius + 4.0 * sigma)) + 1
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    d = np.hypot(xx, yy)
    if radius < 1.5 * sigma:
        s2 = sigma ** 2 + 0.25 * radius ** 2          # disk variance R^2/4 per axis
        k = np.exp(-0.5 * d * d / s2)
    else:
        k = 0.5 * erfc((d - radius) / (math.sqrt(2.0) * sigma))
    return k / k.sum()


class _Ion:
    __slots__ = ("bright", "cloud_until", "cloud_xy", "uid")

    def __init__(self, uid: int, bright: bool = True, cloud_until: float = 0.0,
                 cloud_xy=(0.0, 0.0)):
        self.uid = uid
        self.bright = bright
        self.cloud_until = cloud_until
        self.cloud_xy = cloud_xy


class IonTrapScene:
    """Thread-safe model of the trapped ions as seen through the imaging system."""

    def __init__(self, cfg, sensor_shape=SENSOR_SHAPE, rng=None, clock=time.monotonic):
        self.cfg = cfg
        self.sensor_shape = sensor_shape
        self.rng = rng if rng is not None else np.random.default_rng()
        self.clock = clock
        self._lock = threading.RLock()
        self._uid = 0
        self.ions: List[_Ion] = []
        self.center = np.array(cfg.center, float)
        angle = math.radians(cfg.axis_angle_deg)
        self.axis = np.array([math.cos(angle), math.sin(angle)])
        self.scale = float(cfg.length_scale_px)
        self.psf_sigma = float(cfg.psf_sigma)
        self.photon_rate = float(cfg.photon_rate)
        self.background_rate = float(cfg.background_rate)
        self.lasers_on = True
        self.repumper_on = True
        self.oven_on = False
        self.melt_until = 0.0
        self._was_melted = False
        self._t_last: Optional[float] = None
        self._view = np.zeros(2)                 # camera displacement [px]
        if cfg.surface:
            self._apply_preset(cfg.surface)
        self._build_surface()
        for _ in range(cfg.n_ions):
            self._add_ion(crystallized=True)

    # -- the static part: trap surface, scatter, hot pixels --------------------

    def _apply_preset(self, name: str) -> None:
        if name not in SURFACE_PRESETS:
            raise ValueError("unknown surface preset {!r} (one of {})".format(
                name, ", ".join(SURFACE_PRESETS)))
        for k, v in SURFACE_PRESETS[name].items():
            setattr(self.cfg, k, type(getattr(self.cfg, k))(v))

    def _beam_profile(self, x, y) -> np.ndarray:
        """Relative laser intensity on the surface at sensor (x, y)."""
        cfg = self.cfg
        a = math.radians(cfg.beam_angle_deg)
        dx, dy = np.asarray(x) - self._beam_center[0], np.asarray(y) - self._beam_center[1]
        along = dx * math.cos(a) + dy * math.sin(a)
        across = -dx * math.sin(a) + dy * math.cos(a)
        return np.exp(-0.5 * (along / max(cfg.scatter_sigma_along, 1e-3)) ** 2
                      - 0.5 * (across / max(cfg.scatter_sigma_across, 1e-3)) ** 2)

    def _build_surface(self) -> None:
        """Pre-compute the laser-lit surface (photons/s/px at full laser power) and
        the hot-pixel dark current (e-/s/px) on the whole sensor."""
        cfg = self.cfg
        h, w = self.sensor_shape
        seed = int(cfg.surface_seed)
        # the surface is fixed in the lab: it only moves in the image if the camera does
        vx, vy = self._view
        trap_center = np.array(cfg.center, float) + self._view
        self._beam_center = trap_center + np.asarray(cfg.scatter_offset, float)
        yy, xx = np.mgrid[0:h, 0:w].astype(float)
        profile = self._beam_profile(xx, yy)
        r_surface = defocus_radius_px(cfg.ion_height_um, cfg.numerical_aperture,
                                      cfg.magnification, cfg.pixel_size_um)
        # point sources on the surface plane (splatted to the nearest pixel)
        surface = cfg.scatter_rate * profile
        pts_x, pts_y, pts_f, pts_z = [], [], [], []
        n = int(cfg.defects)
        rng = np.random.default_rng((seed, 1))
        if n > 0:
            px, py = rng.uniform(-0.5, w - 0.5, n), rng.uniform(-0.5, h - 0.5, n)
            pts_x.append(px + vx)
            pts_y.append(py + vy)
            pts_f.append(cfg.defect_rate * np.exp(cfg.defect_spread * rng.standard_normal(n)))
            pts_z.append(rng.uniform(0.0, 1.0, n) ** 2 * cfg.defect_max_height_um)
        rng = np.random.default_rng((seed, 2))
        for _ in range(int(cfg.scratches)):
            length = rng.uniform(40.0, 220.0)
            ang = rng.uniform(0.0, math.pi)
            cx, cy = rng.uniform(0, w) + vx, rng.uniform(0, h) + vy
            t = np.arange(-0.5 * length, 0.5 * length, 1.0)
            pts_x.append(cx + t * math.cos(ang))
            pts_y.append(cy + t * math.sin(ang))
            pts_f.append(np.full(t.size, 0.08 * cfg.defect_rate) *
                         np.exp(0.5 * rng.standard_normal(t.size)))
            pts_z.append(np.zeros(t.size))
        if cfg.gap_scatter > 0:
            ax = math.radians(cfg.axis_angle_deg)
            ux, uy = math.cos(ax), math.sin(ax)
            t = np.arange(-1.5 * max(h, w), 1.5 * max(h, w), 1.0)
            for side in (-1.0, 1.0):
                gx = trap_center[0] - side * cfg.gap_offset_px * uy + t * ux
                gy = trap_center[1] + side * cfg.gap_offset_px * ux + t * uy
                pts_x.append(gx)
                pts_y.append(gy)
                pts_f.append(np.full(t.size, cfg.gap_scatter))
                pts_z.append(np.zeros(t.size))
        lit = np.zeros((h, w))
        if pts_x:
            x = np.concatenate(pts_x)
            y = np.concatenate(pts_y)
            f = np.concatenate(pts_f) * self._beam_profile(x, y)
            z = np.concatenate(pts_z)
            ix, iy = np.rint(x).astype(int), np.rint(y).astype(int)
            on = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h) & (f > 0)
            # dust grains standing up from the surface are less defocused
            height = np.clip(cfg.ion_height_um - z[on], 0.0, None)
            groups = np.round(height / max(cfg.ion_height_um, 1e-9) * 6.0) / 6.0
            for g in np.unique(groups):
                sel = groups == g
                img = np.zeros((h, w))
                np.add.at(img, (iy[on][sel], ix[on][sel]), f[on][sel])
                if g == 1.0:
                    surface = surface + img            # blurred together below
                else:
                    r = r_surface * g
                    lit += fftconvolve(img, blur_kernel(r, cfg.psf_sigma), mode="same")
        lit += fftconvolve(surface, blur_kernel(r_surface, cfg.psf_sigma), mode="same")
        self.laser_map = np.clip(lit, 0.0, None).astype(np.float32)
        dark = np.zeros((h, w), np.float32)
        nh = int(cfg.hot_pixels)
        rng = np.random.default_rng((seed, 3))       # on the sensor, not the trap
        if nh > 0:
            hx, hy = rng.integers(0, w, nh), rng.integers(0, h, nh)
            dark[hy, hx] = cfg.hot_pixel_rate * np.exp(1.0 * rng.standard_normal(nh))
        self.dark_map = dark
        self.defocus_px = r_surface

    def set_params(self, **params) -> dict:
        """Change simulation parameters (``SimulatorConfig`` names) at run time;
        ``surface="dirty"`` applies a preset first."""
        cfg = self.cfg
        with self._lock:
            names = {f for f in vars(cfg)}
            bad = set(params) - names
            if bad:
                raise ValueError("unknown simulator parameter(s): {}".format(
                    ", ".join(sorted(bad))))
            rebuild = False
            if params.get("surface"):
                self._apply_preset(params["surface"])
                rebuild = True
            for k, v in params.items():
                cur = getattr(cfg, k)
                if isinstance(cur, tuple):
                    v = tuple(float(a) for a in v)
                elif isinstance(cur, bool):
                    v = bool(v)
                elif isinstance(cur, (int, float, str)):
                    v = type(cur)(v)
                setattr(cfg, k, v)
                rebuild |= k in _SURFACE_KEYS
            self.psf_sigma = float(cfg.psf_sigma)
            self.photon_rate = float(cfg.photon_rate)
            self.background_rate = float(cfg.background_rate)
            if "length_scale_px" in params:
                self.scale = float(cfg.length_scale_px)
            if rebuild:
                self._build_surface()
            return self.parameters()

    def parameters(self) -> dict:
        d = {k: v for k, v in vars(self.cfg).items()}
        d["defocus_radius_px"] = round(self.defocus_px, 2)
        d["presets"] = sorted(SURFACE_PRESETS)
        return d

    # -- helpers ------------------------------------------------------------

    def _add_ion(self, crystallized: bool, now: float = 0.0, cloud_time: float = 0.8):
        self._uid += 1
        ion = _Ion(self._uid)
        if not crystallized:
            ion.cloud_until = now + cloud_time
            jitter = self.rng.normal(0.0, 6.0, 2)
            ion.cloud_xy = tuple(self.center + jitter)
        pos = int(self.rng.integers(0, len(self.crystallized()) + 1)) if self.ions else 0
        self.ions.insert(pos, ion)
        return ion

    def crystallized(self, now: Optional[float] = None) -> List[_Ion]:
        now = self.clock() if now is None else now
        return [i for i in self.ions if i.cloud_until <= now]

    # -- public control (any thread) -------------------------------------------

    def load_ion(self, n: int = 1, cloud_time: float = 0.8):
        """Load ``n`` ions; each spends ``cloud_time`` s as a hot cloud first."""
        with self._lock:
            now = self.clock()
            for _ in range(n):
                self._add_ion(crystallized=cloud_time <= 0, now=now, cloud_time=cloud_time)

    def lose_ion(self, index: Optional[int] = None):
        """Remove ion ``index`` (chain order, includes dark ions); random if None."""
        with self._lock:
            chain = self.crystallized()
            if not chain:
                return False
            ion = chain[int(self.rng.integers(len(chain)))] if index is None else chain[index]
            self.ions.remove(ion)
            return True

    def set_dark(self, index: int, dark: bool = True):
        with self._lock:
            self.crystallized()[index].bright = not dark

    def set_n_ions(self, n: int):
        with self._lock:
            self.ions = []
            for _ in range(n):
                self._add_ion(crystallized=True)

    def shift(self, dx: float, dy: float):
        """Move the ions rigidly (shuttling); the trap surface stays put."""
        with self._lock:
            self.center = self.center + np.array([dx, dy], float)

    def move_camera(self, dx: float, dy: float):
        """The camera moved: ions *and* the trap surface move in the image (hot
        pixels, being on the sensor, do not)."""
        with self._lock:
            d = np.array([dx, dy], float)
            self.center = self.center + d
            self._view = self._view + d
            self._build_surface()

    def set_center(self, x: float, y: float):
        with self._lock:
            self.center = np.array([x, y], float)

    def set_scale(self, scale_px: float):
        """Change the axial confinement (DAC voltages): l ~ w_z^(-2/3)."""
        with self._lock:
            self.scale = float(scale_px)

    def melt(self, duration: float = 1.0):
        with self._lock:
            self.melt_until = self.clock() + duration

    def set_lasers(self, on: bool):
        """Cooling/detection lasers (ion fluorescence *and* surface scatter)."""
        with self._lock:
            self.lasers_on = bool(on)

    def set_repumper(self, on: bool):
        """Repumper off: the ions go dark (shelved), the laser scatter stays."""
        with self._lock:
            self.repumper_on = bool(on)

    def set_oven(self, on: bool):
        with self._lock:
            self.oven_on = bool(on)

    def state(self) -> dict:
        with self._lock:
            now = self.clock()
            chain = self.crystallized(now)
            pos = self._chain_positions(len(chain))
            return {
                "n_ions": len(chain),
                "n_bright": sum(1 for i in chain if i.bright),
                "n_clouds": len(self.ions) - len(chain),
                "positions": pos.tolist(),
                "clouds": [list(i.cloud_xy) for i in self.ions if i.cloud_until > now],
                "bright": [i.bright for i in chain],
                "center": self.center.tolist(),
                "scale": self.scale,
                "lasers_on": self.lasers_on,
                "repumper_on": self.repumper_on,
                "oven_on": self.oven_on,
                "melted": now < self.melt_until,
                "surface_defocus_px": round(self.defocus_px, 2),
            }

    # -- dynamics -----------------------------------------------------------

    def _chain_positions(self, n: int) -> np.ndarray:
        if n == 0:
            return np.zeros((0, 2))
        u = equilibrium_positions(n)
        return self.center[None, :] + self.scale * u[:, None] * self.axis[None, :]

    def advance(self, now: float):
        """Apply the random processes up to time ``now`` (monotonic seconds)."""
        cfg = self.cfg
        with self._lock:
            if self._t_last is None:
                self._t_last = now
                return
            dt = min(max(now - self._t_last, 0.0), 1.0)
            self._t_last = now
            if dt == 0.0:
                return
            rng = self.rng
            if self.oven_on and cfg.load_rate > 0 and rng.random() < cfg.load_rate * dt:
                self._add_ion(crystallized=False, now=now)
            for ion in list(self.ions):
                if cfg.loss_rate > 0 and rng.random() < cfg.loss_rate * dt:
                    self.ions.remove(ion)
                    continue
                if ion.bright and cfg.dark_rate > 0 and rng.random() < cfg.dark_rate * dt:
                    ion.bright = False
                elif not ion.bright and cfg.bright_rate > 0 and rng.random() < cfg.bright_rate * dt:
                    ion.bright = True
            if cfg.melt_rate > 0 and now >= self.melt_until and rng.random() < cfg.melt_rate * dt:
                self.melt_until = now + rng.uniform(0.5, 2.0)
            if cfg.drift_px_per_s > 0:
                self.center = self.center + rng.normal(0.0, cfg.drift_px_per_s * math.sqrt(dt), 2)
            was_melted = self._was_melted
            melted = now < self.melt_until
            if was_melted and not melted and len(self.ions) > 1:
                # recrystallisation may re-order dark and bright ions
                if rng.random() < 0.5:
                    order = rng.permutation(len(self.ions))
                    self.ions = [self.ions[i] for i in order]
            self._was_melted = melted

    # -- rendering --------------------------------------------------------------

    @staticmethod
    def _bin(m: np.ndarray, roi, hbin: int, vbin: int) -> np.ndarray:
        x0, y0, x1, y1 = roi
        h, w = (y1 - y0) // vbin, (x1 - x0) // hbin
        sub = m[y0:y0 + h * vbin, x0:x0 + w * hbin]
        if hbin == 1 and vbin == 1:
            return sub.astype(np.float64)
        return sub.reshape(h, vbin, w, hbin).sum(axis=(1, 3), dtype=np.float64)

    def expected_electrons(self, roi, hbin: int, vbin: int, exposure: float,
                           now: Optional[float] = None) -> np.ndarray:
        """Mean photo-electrons (and dark electrons) per (binned) pixel for one exposure."""
        now = self.clock() if now is None else now
        cfg = self.cfg
        x0, y0, x1, y1 = roi
        w, h = (x1 - x0) // hbin, (y1 - y0) // vbin
        # edges of every binned pixel in sensor coordinates
        xe = x0 - 0.5 + hbin * np.arange(w + 1)
        ye = y0 - 0.5 + vbin * np.arange(h + 1)
        qe = cfg.quantum_efficiency
        with self._lock:
            laser = 0.0
            if self.lasers_on:
                laser = max(0.0, cfg.laser_power *
                            (1.0 + cfg.laser_noise * float(self.rng.standard_normal())))
            img = self._bin(self.dark_map, roi, hbin, vbin) * exposure
            img += self.background_rate * exposure * qe * hbin * vbin
            if laser > 0:
                img += self._bin(self.laser_map, roi, hbin, vbin) * (laser * exposure * qe)
            if laser == 0.0:
                return img
            chain = self.crystallized(now)
            melted = now < self.melt_until
            pos = self._chain_positions(len(chain))
            sigma = np.full(len(chain), self.psf_sigma)
            rate = self.photon_rate * exposure * qe * laser
            flux = np.array([rate if (i.bright and self.repumper_on) else 0.0 for i in chain])
            if melted and len(chain):
                pos = pos + self.rng.normal(0.0, 1.5, pos.shape)
                sigma[:] = self.psf_sigma + 3.0
                flux *= 0.3
            clouds = [i for i in self.ions if i.cloud_until > now]
            if clouds:
                pos = np.vstack([pos, np.array([c.cloud_xy for c in clouds])])
                sigma = np.concatenate([sigma, np.full(len(clouds), 5.0)])
                flux = np.concatenate([flux, np.full(len(clouds), 0.4 * rate if
                                                     self.repumper_on else 0.0)])
            keep = flux > 0
            if np.any(keep):
                pos, sigma, flux = pos[keep], sigma[keep], flux[keep]
                f_halo = min(max(cfg.psf_halo_fraction, 0.0), 1.0)
                parts = [(1.0 - f_halo, sigma)]
                if f_halo > 0:
                    parts.append((f_halo, np.sqrt(sigma ** 2 + max(cfg.psf_halo_sigma ** 2 -
                                                                   self.psf_sigma ** 2, 0.0))))
                for weight, sg in parts:
                    ex = (ndtr((xe[None, 1:] - pos[:, :1]) / sg[:, None]) -
                          ndtr((xe[None, :-1] - pos[:, :1]) / sg[:, None]))
                    ey = (ndtr((ye[None, 1:] - pos[:, 1:2]) / sg[:, None]) -
                          ndtr((ye[None, :-1] - pos[:, 1:2]) / sg[:, None]))
                    img = img + (ey.T * (weight * flux)[None, :]) @ ex
        return img


def emccd_counts(electrons: np.ndarray, em_gain: float, use_em: bool, sensitivity: float,
                 read_noise_e: float, bias: float, cic: float, rng) -> np.ndarray:
    """Convert mean photo-electrons into a noisy uint16 EMCCD image."""
    n = rng.poisson(np.maximum(electrons, 0.0) + cic)
    if use_em and em_gain > 1.0:
        out = rng.gamma(n, em_gain)
        np.minimum(out, EM_REGISTER_FULL_WELL, out=out)
    else:
        out = n.astype(np.float64)
    counts = bias + out / sensitivity + rng.normal(0.0, read_noise_e / sensitivity, n.shape)
    return np.clip(np.rint(counts), 0, 65535).astype(np.uint16)


def readout_time(roi, hbin: int, vbin: int, hs_mhz: float, vs_us: float) -> float:
    """Approximate iXon image readout time [s]."""
    x0, y0, x1, y1 = roi
    rows = (y1 - y0) // vbin
    return y1 * vs_us * 1e-6 + rows * (x1 + 20) / (hs_mhz * 1e6) + 2e-4


def nearest_index(values, target) -> int:
    return int(np.argmin(np.abs(np.asarray(values, float) - float(target))))


class SimulatedCamera(CameraBase):
    """Simulated iXon Ultra 897 with an ion trap in front of it."""

    sensor_shape: Tuple[int, int] = SENSOR_SHAPE

    def __init__(self, cfg):
        self.cfg = cfg
        seed = None if cfg.seed < 0 else cfg.seed
        self.rng = np.random.default_rng(seed)
        self.scene = IonTrapScene(cfg, self.sensor_shape,
                                  rng=np.random.default_rng(None if seed is None else seed + 1))
        self._applied: Optional[AppliedSettings] = None
        self._acquiring = False
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._triggers: deque = deque()
        self._next_auto_trigger = 0.0
        self._busy_until = 0.0
        self.missed_triggers = 0
        self._t0 = 0.0
        self._n_done = 0
        self._frame_counter = 0
        self._cycle = 0.0
        self._exposure = 0.0
        self._ro = 0.0
        self._em_gain = 1
        self._sensitivity = cfg.sensitivity
        self._read_noise = cfg.read_noise_e
        self.last_frame: Optional[np.ndarray] = None
        self.camera_link_enabled = False
        # thermal model
        self._temp = 20.0
        self._target = -60.0
        self._cooler = False
        self._t_temp = time.monotonic()
        self._near_since: Optional[float] = None

    # -- CameraBase ------------------------------------------------------------

    def open(self) -> dict:
        return {"model": "Simulated iXon Ultra 897 (DU-897U)", "serial": 0,
                "sensor_shape": list(self.sensor_shape), "pixel_size_um": 16.0,
                "simulated": True}

    def close(self) -> None:
        self.stop()

    def capabilities(self) -> dict:
        return {"hs_speeds_mhz": {"em": EM_HS_SPEEDS, "conventional": CONV_HS_SPEEDS},
                "vs_speeds_us": VS_SPEEDS, "vs_fastest_recommended": VS_SPEEDS[VS_FASTEST_RECOMMENDED],
                "preamp_gains": PREAMP_GAINS, "em_gain_range": list(EM_GAIN_RANGE),
                "temperature_range": [-100, 20]}

    def configure(self, settings: AcquisitionSettings, tag: str = "") -> AppliedSettings:
        settings.validate()
        if self._acquiring:
            self.stop()
        roi = normalize_roi(settings.roi, self.sensor_shape, settings.hbin, settings.vbin)
        em = settings.amplifier == "em"
        speeds = EM_HS_SPEEDS if em else CONV_HS_SPEEDS
        hs = speeds[nearest_index(speeds, settings.hs_speed_mhz)]
        vs = (VS_SPEEDS[VS_FASTEST_RECOMMENDED] if settings.vs_speed_us <= 0
              else VS_SPEEDS[nearest_index(VS_SPEEDS, settings.vs_speed_us)])
        warnings = []
        pa_index = min(max(settings.preamp_gain_index, 0), len(PREAMP_GAINS) - 1)
        if pa_index != settings.preamp_gain_index:
            warnings.append("pre-amp gain index {} not available, using {}".format(
                settings.preamp_gain_index, pa_index))
        gain = 1
        if em:
            gain = int(min(max(settings.em_gain, EM_GAIN_RANGE[0]), EM_GAIN_RANGE[1]))
        exposure = max(float(settings.exposure), 1e-5)
        ro = readout_time(roi, settings.hbin, settings.vbin, hs, vs)
        keep_clean = 1.0e-3
        internal = settings.trigger == "internal"
        if settings.frame_transfer and internal and settings.acquisition == "run_till_abort":
            cycle = max(exposure, ro) + 1e-5
        else:
            cycle = exposure + ro + keep_clean
        cycle = max(cycle, settings.kinetic_cycle_time)
        if settings.acquisition == "accumulate":
            cycle *= settings.n_accumulations
        shape = ((roi[3] - roi[1]) // settings.vbin, (roi[2] - roi[0]) // settings.hbin)
        self._exposure, self._ro, self._cycle, self._em_gain = exposure, ro, cycle, gain
        self._use_em = em
        self._sensitivity = self.cfg.sensitivity * PREAMP_GAINS[-1] / PREAMP_GAINS[pa_index]
        self._read_noise = self.cfg.read_noise_e * math.sqrt(max(hs, 0.08) / 5.0)
        self.camera_link_enabled = settings.camera_link
        self._applied = AppliedSettings(
            settings=settings, roi=roi, frame_shape=shape, exposure=exposure,
            accumulate_cycle=cycle, kinetic_cycle=cycle, readout_time=ro,
            keep_clean_time=0.0 if (settings.frame_transfer and internal) else keep_clean,
            hs_speed_mhz=hs, vs_speed_us=vs, preamp_gain=PREAMP_GAINS[pa_index],
            em_gain=gain, em_gain_range=EM_GAIN_RANGE, tag=tag, warnings=warnings)
        return self._applied

    def start(self) -> None:
        if self._applied is None:
            raise CameraError("configure() first")
        with self._lock:
            self._t0 = time.monotonic()
            self._n_done = 0
            self._triggers.clear()
            self._busy_until = 0.0
            self._next_auto_trigger = self._t0
            self._frame_counter = 0
            self._acquiring = True
        self._wake.clear()

    def stop(self) -> None:
        with self._lock:
            self._acquiring = False
            self._triggers.clear()
        self._wake.set()

    def is_acquiring(self) -> bool:
        return self._acquiring

    def cancel_wait(self) -> None:
        self._wake.set()

    def trigger(self, n: int = 1, spacing: float = 0.0) -> int:
        """Inject ``n`` external trigger pulses (thread-safe).  Returns how many
        were accepted (triggers arriving while the camera is busy are missed)."""
        accepted = 0
        with self._lock:
            if not self._acquiring or self._applied is None:
                return 0
            if self._applied.settings.trigger == "internal":
                return 0
            t = time.monotonic()
            for k in range(n):
                tk = t + k * spacing
                if tk < self._busy_until:
                    self.missed_triggers += 1
                    continue
                self._busy_until = tk + self._exposure + self._ro
                self._triggers.append(tk)
                accepted += 1
        self._wake.set()
        return accepted

    def send_software_trigger(self) -> None:
        if self._applied is None or self._applied.settings.trigger != "software":
            raise CameraError("not in software trigger mode")
        self.trigger(1)

    def wait_frames(self, timeout: float) -> List[Tuple[int, np.ndarray]]:
        deadline = time.monotonic() + max(timeout, 0.0)
        while True:
            frames, next_due = self._collect_due()
            if frames:
                return frames
            now = time.monotonic()
            if not self._acquiring:
                self._wake.wait(max(0.0, min(deadline - now, 0.05)))
                self._wake.clear()
                return []
            wait = deadline - now if next_due is None else min(deadline, next_due) - now
            if wait <= 0:
                if now >= deadline:
                    return []
                continue
            woke = self._wake.wait(wait)
            self._wake.clear()
            if woke and time.monotonic() < deadline and not self._has_pending():
                return []  # cancelled

    def _has_pending(self) -> bool:
        with self._lock:
            return bool(self._triggers)

    def _collect_due(self):
        """Return (frames ready now, monotonic time of the next frame or None)."""
        with self._lock:
            if not self._acquiring or self._applied is None:
                return [], None
            s = self._applied.settings
            now = time.monotonic()
            due_times = []
            next_due = None
            if s.trigger == "internal":
                n_due = int((now - self._t0) / self._cycle)
                if s.acquisition in ("single", "accumulate"):
                    n_due = min(n_due, 1)
                if n_due > self._n_done:
                    first = max(self._n_done, n_due - 32)
                    due_times = [self._t0 + (k + 1) * self._cycle for k in range(first, n_due)]
                    self._n_done = n_due
                if s.acquisition in ("single", "accumulate") and self._n_done >= 1:
                    self._acquiring = False
                else:
                    next_due = self._t0 + (self._n_done + 1) * self._cycle
            else:
                if self.cfg.auto_trigger_hz > 0 and s.trigger == "external":
                    period = 1.0 / self.cfg.auto_trigger_hz
                    while self._next_auto_trigger <= now:
                        tk = self._next_auto_trigger
                        self._next_auto_trigger += period
                        if tk >= self._busy_until:
                            self._busy_until = tk + self._exposure + self._ro
                            self._triggers.append(tk)
                    next_due = self._next_auto_trigger
                lat = self._exposure + self._ro
                while self._triggers and self._triggers[0] + lat <= now:
                    due_times.append(self._triggers.popleft() + lat)
                if self._triggers:
                    t_ready = self._triggers[0] + lat
                    next_due = t_ready if next_due is None else min(next_due, t_ready)
                if due_times and s.acquisition in ("single", "accumulate"):
                    due_times = due_times[:1]
                    self._acquiring = False
            start_index = self._frame_counter
            self._frame_counter += len(due_times)
            applied = self._applied
        frames = []
        for k, t in enumerate(due_times):
            frames.append((start_index + k + 1, self._render(applied, t)))
        return frames, next_due

    def _render(self, applied: AppliedSettings, t: float) -> np.ndarray:
        s = applied.settings
        self.scene.advance(t)
        el = self.scene.expected_electrons(applied.roi, s.hbin, s.vbin, applied.exposure, now=t)
        cic = self.cfg.cic * s.hbin * s.vbin
        img = emccd_counts(el, self._em_gain, self._use_em, self._sensitivity, self._read_noise,
                           self.cfg.bias, cic, self.rng)
        self.last_frame = img
        return img

    # -- temperature -----------------------------------------------------------

    def _update_temp(self):
        now = time.monotonic()
        dt, self._t_temp = now - self._t_temp, now
        goal = self._target if self._cooler else 20.0
        self._temp += (goal - self._temp) * (1.0 - math.exp(-dt / 8.0))
        if abs(self._temp - goal) < 0.5:
            self._near_since = self._near_since or now
        else:
            self._near_since = None

    def get_temperature(self) -> Tuple[float, str]:
        self._update_temp()
        if not self._cooler:
            return self._temp, "off"
        if self._near_since is None:
            return self._temp, "not_reached"
        if time.monotonic() - self._near_since < 3.0:
            return self._temp, "not_stabilized"
        return self._temp, "stabilized"

    def set_temperature(self, temperature: float) -> None:
        self._update_temp()
        self._target = float(temperature)

    def set_cooler(self, on: bool) -> None:
        self._update_temp()
        self._cooler = bool(on)
