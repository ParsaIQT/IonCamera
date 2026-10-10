"""Linear ion chain physics.

Equilibrium positions of ``N`` ions in a harmonic axial potential are
``z_m = l * u_m`` where ``l = (Z^2 e^2 / (4 pi eps0 m w_z^2))^(1/3)`` and the
dimensionless ``u_m`` solve (D. F. V. James, Appl. Phys. B 66, 181 (1998))::

    u_m - sum_{n<m} 1/(u_m - u_n)^2 + sum_{n>m} 1/(u_m - u_n)^2 = 0

The positions depend only on the charge, not on the mass, so a dark ion of
any species (shelved, or a molecular ion after a chemical reaction) keeps
its place while a *lost* ion makes the remaining ones re-arrange.  The
tracker uses this fingerprint to tell "dark" from "lost" and to infer where
hidden (dark) ions sit after a reconfiguration.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence, Tuple

import numpy as np


@lru_cache(maxsize=512)
def _equilibrium(n: int) -> Tuple[float, ...]:
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return (0.0,)
    # Initial guess: uniform chain with the right overall length scale.
    half = 0.75 * (n - 1) ** 0.62 + 0.15
    u = np.linspace(-half, half, n)
    idx = np.arange(n)
    for _ in range(200):
        d = u[:, None] - u[None, :]
        np.fill_diagonal(d, np.inf)
        inv2 = np.sign(d) / d ** 2
        f = u - inv2.sum(axis=1)
        if np.max(np.abs(f)) < 1e-13:
            break
        inv3 = 2.0 / np.abs(d) ** 3
        jac = -inv3
        jac[idx, idx] = 1.0 + inv3.sum(axis=1)
        step = np.linalg.solve(jac, -f)
        # Damp the step so that the ordering of ions is preserved.
        gaps = np.diff(u)
        dstep = np.diff(step)
        shrink = dstep < 0
        lam = 1.0
        if np.any(shrink):
            lam = min(1.0, 0.5 * np.min(gaps[shrink] / -dstep[shrink]))
        u = u + lam * step
    u = 0.5 * (u - u[::-1])  # enforce exact symmetry
    return tuple(float(v) for v in u)


def equilibrium_positions(n: int) -> np.ndarray:
    """Dimensionless equilibrium positions ``u_m`` (sorted, symmetric) for ``n`` ions."""
    return np.array(_equilibrium(int(n)))


def min_spacing(n: int) -> float:
    """Smallest dimensionless spacing (between the central ions)."""
    if n < 2:
        return float("inf")
    return float(np.min(np.diff(equilibrium_positions(n))))


@dataclass
class ChainFit:
    """Best linear-chain description of a set of 2D ion positions."""
    n: int                      # ions in the model chain (bright + hidden)
    center: np.ndarray          # (x, y) of the chain centre [px]
    axis: np.ndarray            # unit vector along the chain
    scale: float                # length scale l [px]
    rms_axial: float            # rms residual along the axis [px]
    rms_radial: float           # rms distance from the axis [px] (zig-zag indicator)
    bright_sites: Tuple[int, ...]   # which model sites are occupied by the given ions
    hidden_sites: Tuple[int, ...]   # model sites without a visible ion

    def site_positions(self) -> np.ndarray:
        """(n, 2) positions of all model sites, sorted along the axis."""
        u = equilibrium_positions(self.n)
        return self.center[None, :] + self.scale * u[:, None] * self.axis[None, :]

    def hidden_positions(self) -> np.ndarray:
        return self.site_positions()[list(self.hidden_sites)]

    @property
    def is_linear(self) -> bool:
        if self.n < 2:
            return True
        return self.rms_radial < 0.25 * self.scale * min_spacing(self.n)


def chain_axis(positions: np.ndarray, hint: Optional[np.ndarray] = None) -> np.ndarray:
    """Principal axis of the positions (unit vector; sign fixed to +x, or to ``hint``)."""
    p = np.asarray(positions, float).reshape(-1, 2)
    ref = np.array([1.0, 0.0]) if hint is None else np.asarray(hint, float)
    if len(p) < 2 or np.allclose(p, p[0]):
        axis = ref
    else:
        c = p - p.mean(axis=0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        axis = vt[0]
    if np.dot(axis, ref) < 0:
        axis = -axis
    return axis / np.linalg.norm(axis)


def fit_chain(positions: Sequence[Sequence[float]], max_hidden: int = 0,
              scale_prior: Optional[float] = None,
              center_prior: Optional[Sequence[float]] = None,
              axis_hint: Optional[np.ndarray] = None,
              pos_sigma: float = 0.25, scale_tol: float = 0.04,
              center_tol: float = 0.75, hidden_cost: float = 9.0) -> Optional[ChainFit]:
    """Fit visible ion ``positions`` [px] with a linear chain that may contain
    hidden (dark) ions.

    The fit minimises a chi-square: axial residuals / ``pos_sigma`` plus
    Gaussian priors on the length scale (relative width ``scale_tol``) and on
    the chain centre (``center_tol`` px).  Both priors are physical: losing or
    darkening an ion changes neither the trap frequency nor the trap centre.
    Every hidden ion costs ``hidden_cost`` so it must be clearly justified.

    :param max_hidden: how many hidden ions to consider (0 = plain fit).
    :param scale_prior: known length scale ``l`` [px] (e.g. from before).
    :param center_prior: known chain centre ``(x, y)`` [px].
    """
    p = np.asarray(positions, float).reshape(-1, 2)
    nb = len(p)
    if nb == 0:
        return None
    axis = chain_axis(p, axis_hint)
    perp = np.array([-axis[1], axis[0]])
    centroid = p.mean(axis=0)
    s = np.sort((p - centroid) @ axis)
    t = (p - centroid) @ perp
    rms_radial = float(np.sqrt(np.mean(t ** 2)))
    s0_prior = None
    if center_prior is not None:
        s0_prior = float((np.asarray(center_prior, float) - centroid) @ axis)

    best = None
    for k in range(0, max_hidden + 1):
        n = nb + k
        u_all = equilibrium_positions(n)
        for hidden in itertools.combinations(range(n), k):
            bright = [i for i in range(n) if i not in hidden]
            u = u_all[bright]
            rows = [np.column_stack([np.ones(nb), u]) / pos_sigma]
            rhs = [s / pos_sigma]
            n_constraints = nb
            if scale_prior is not None:
                w = 1.0 / (scale_tol * scale_prior)
                rows.append(np.array([[0.0, w]]))
                rhs.append(np.array([scale_prior * w]))
                n_constraints += 1
            if s0_prior is not None:
                w = 1.0 / center_tol
                rows.append(np.array([[w, 0.0]]))
                rhs.append(np.array([s0_prior * w]))
                n_constraints += 1
            a = np.vstack(rows)
            b = np.concatenate(rhs)
            if n_constraints < 2 or np.linalg.matrix_rank(a) < 2:
                # Not enough information to fix both offset and scale: only the
                # trivial interpretation (no hidden ions) is allowed.
                if k > 0:
                    continue
                scale = scale_prior if scale_prior is not None else 1.0
                s0 = float(np.mean(s - scale * u))
                chi2 = 0.0
            else:
                sol, *_ = np.linalg.lstsq(a, b, rcond=None)
                s0, scale = float(sol[0]), float(sol[1])
                if scale <= 0:
                    continue
                chi2 = float(np.sum((a @ sol - b) ** 2))
            cost = chi2 + hidden_cost * k
            if best is None or cost < best[0]:
                res = s - (s0 + scale * u)
                rms = float(np.sqrt(np.mean(res ** 2)))
                best = (cost, n, s0, scale, rms, tuple(bright), tuple(hidden))
    if best is None:
        return None
    _, n, s0, scale, rms, bright, hidden = best
    return ChainFit(n=n, center=centroid + s0 * axis, axis=axis, scale=scale,
                    rms_axial=rms, rms_radial=rms_radial, bright_sites=bright,
                    hidden_sites=hidden)
