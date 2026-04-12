"""Core shell-model primitives.

Everything here is in Hartree atomic units (length = a0, energy = Ha).
In these units the Coulomb prefactor 1/(4 pi eps0) = 1, the electron
charge magnitude = 1, the electron mass = 1.

A "shell electron" is a thin spherical shell of radius R carrying total
charge -1 and total mass 1.

----------------------------------------------------------------------------
IMPORTANT: this file deliberately avoids any use of hbar or the Heisenberg
uncertainty principle as a *given*. The only physics assumed for free is:

  1. Newton's laws of motion
  2. Classical electromagnetism (Coulomb's law / Gauss's law)
  3. Energy conservation

Any additional stabilizing mechanism (rigid rotation, surface tension,
postulated action quantum, virial-theorem postulate, etc.) is named
explicitly so that each experiment can try a different one and we can see
which mechanisms fit the data.
----------------------------------------------------------------------------

Key electrostatic facts (from Gauss / shell theorem) used throughout:

  - A thin shell of charge q at radius R produces a potential:
        V(r) = q / r      if  r > R
        V(r) = q / R      if  r <= R    (constant inside)

  - Interaction between a point nucleus +Z at origin and a shell of charge
    q at radius R:
        U_nuc_shell = Z * q / R

  - Interaction between two concentric shells (q1 at R1, q2 at R2):
        U_12 = q1 * q2 / max(R1, R2)

  - Electromagnetic self-energy of a thin uniform shell of charge q:
        U_self = q^2 / (2 R)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Electrostatic primitives (pure classical EM)
# ---------------------------------------------------------------------------

def nucleus_shell_energy(Z: float, q_shell: float, R: float) -> float:
    return Z * q_shell / R


def two_shell_energy(q1: float, R1: float, q2: float, R2: float) -> float:
    return q1 * q2 / max(R1, R2)


def shell_self_energy(q: float, R: float) -> float:
    return 0.5 * q * q / R


# ---------------------------------------------------------------------------
# Candidate stabilizing "kinetic / confinement" mechanisms
# ---------------------------------------------------------------------------
# Each function below represents a DIFFERENT physical hypothesis for what
# holds the electron shell up against the Coulomb attraction of the nucleus.
# None of them is taken as true a priori; experiments compare them.
# ---------------------------------------------------------------------------

def T_zero(R: float) -> float:
    """No stabilization.  (Pure static shell.)"""
    return 0.0


def T_rigid_rotation(R: float, L: float = 1.0) -> float:
    """Rigid rotation of a thin spherical shell about a fixed axis.

    Moment of inertia of a thin sphere: I = (2/3) m R^2.
    Rotational kinetic energy: T = L^2 / (2 I) = 3 L^2 / (4 R^2)  (m=1).

    L is a free classical parameter (units: angular momentum).  In atomic
    units, L = 1 corresponds to one "reduced Planck constant", but nothing
    in classical mechanics forces L to that value: this is a hypothesis.
    """
    return 3.0 * L * L / (4.0 * R * R)


def T_ring(R: float, L: float = 1.0) -> float:
    """Rigid rotation of a thin ring (all mass on the equator).

    Moment of inertia I = m R^2, so T = L^2 / (2 R^2).
    With L = 1 (atomic units) this reproduces exactly the Bohr result, so
    this is the minimal geometric model for which L=1 yields hydrogen.
    """
    return 0.5 * L * L / (R * R)


def T_point_orbit(R: float, L: float = 1.0) -> float:
    """Point electron on a circular orbit of radius R, angular momentum L.

    I = m R^2, so T_rot = L^2 / (2 R^2).  This is the Bohr model, rewritten
    to fit the "barrier" interface.  The electron here is NOT a shell.
    """
    return 0.5 * L * L / (R * R)


def T_alpha_over_R2(R: float, alpha: float = 0.5) -> float:
    """Generic 1/R^2 confinement barrier with free strength alpha.

    We treat alpha as an empirical parameter to be fit to hydrogen, and
    then test whether the same alpha carries over to higher Z and to
    multi-electron atoms.  This is the *most neutral* parameterization:
    it does not commit to any particular physical origin for the barrier.
    """
    return alpha / (R * R)


def T_surface_tension(R: float, sigma: float = 1.0) -> float:
    """Classical surface-tension energy of the shell, U = sigma * 4 pi R^2."""
    return sigma * 4.0 * math.pi * R * R


def T_internal_pressure(R: float, A: float = 1.0, n: float = 3.0) -> float:
    """Adiabatic internal gas in the shell: U = A / R^n."""
    return A / (R ** n)


# ---------------------------------------------------------------------------
# Energy minimization helpers
# ---------------------------------------------------------------------------

def one_electron_energy(
    R: float,
    Z: int,
    T_func: Callable[[float], float],
    include_self_energy: bool = False,
) -> float:
    """Total energy of a single electron shell + point nucleus."""
    E = T_func(R) + nucleus_shell_energy(Z, -1.0, R)
    if include_self_energy:
        E += shell_self_energy(-1.0, R)
    return E


def minimize_1d(
    f: Callable[[float], float],
    R_lo: float = 1e-3,
    R_hi: float = 1e3,
    n_grid: int = 400,
) -> tuple[float, float]:
    """Find the (R, f(R)) minimum on a log grid + local refinement."""
    from scipy.optimize import minimize_scalar

    # Log-grid scan for a reasonable initial bracket
    Rs = np.geomspace(R_lo, R_hi, n_grid)
    Fs = np.array([f(R) for R in Rs])
    if not np.isfinite(Fs).any():
        return float("nan"), float("nan")
    i = int(np.nanargmin(Fs))
    lo = Rs[max(i - 1, 0)]
    hi = Rs[min(i + 1, len(Rs) - 1)]
    if lo == hi:
        lo *= 0.9
        hi *= 1.1
    res = minimize_scalar(
        f, bracket=(lo, Rs[i], hi) if lo < Rs[i] < hi else None,
        bounds=(R_lo, R_hi), method="bounded",
        options={"xatol": 1e-10},
    )
    return float(res.x), float(res.fun)


# ---------------------------------------------------------------------------
# Many-electron baseline: one shell per electron, concentric
# ---------------------------------------------------------------------------

def concentric_energy(
    radii: Sequence[float],
    Z: int,
    T_func: Callable[[float], float],
    same_shell_lambda: float = 1.0,
    include_self_energy: bool = False,
) -> float:
    """Total energy of N concentric electron shells (plus point nucleus).

    `same_shell_lambda` multiplies the repulsion between any two shells
    that sit at (essentially) the same radius.  lambda = 1 means no
    correlation / full repulsion; lambda = 0 means the two shells are
    perfectly anticorrelated and do not feel each other.
    """
    N = len(radii)
    E = 0.0
    for r in radii:
        E += T_func(r) + nucleus_shell_energy(Z, -1.0, r)
        if include_self_energy:
            E += shell_self_energy(-1.0, r)
    for i in range(N):
        for j in range(i + 1, N):
            same = math.isclose(radii[i], radii[j], rel_tol=1e-9)
            factor = same_shell_lambda if same else 1.0
            E += factor * two_shell_energy(-1.0, radii[i], -1.0, radii[j])
    return E


def minimize_shell_atom(
    Z: int,
    shell_pattern: Sequence[int],
    T_func: Callable[[float], float],
    same_shell_lambda: float = 1.0,
    include_self_energy: bool = False,
    R0: Sequence[float] | None = None,
) -> dict:
    """Minimize total energy of an atom partitioned into shells.

    shell_pattern is a tuple like (2,) for He, (2, 1) for Li, (2, 2) for Be,
    etc.  Each entry is the number of electrons that share a common radius.
    """
    from scipy.optimize import minimize

    N = int(sum(shell_pattern))
    K = len(shell_pattern)

    def unpack(r_vec: np.ndarray) -> List[float]:
        radii: List[float] = []
        for k, n_k in enumerate(shell_pattern):
            radii.extend([float(r_vec[k])] * int(n_k))
        return radii

    def E_of(r_vec: np.ndarray) -> float:
        if np.any(r_vec <= 1e-6) or np.any(~np.isfinite(r_vec)):
            return 1e6
        radii = unpack(r_vec)
        return concentric_energy(
            radii, Z, T_func=T_func,
            same_shell_lambda=same_shell_lambda,
            include_self_energy=include_self_energy,
        )

    if R0 is None:
        R0 = np.array(
            [1.0 / max(Z - 2 * k, 1) for k in range(K)], dtype=float
        )
    else:
        R0 = np.asarray(R0, dtype=float)

    res = minimize(
        E_of, R0, method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 50000},
    )
    return {
        "Z": Z, "N": N,
        "shell_pattern": tuple(int(x) for x in shell_pattern),
        "shell_radii": [float(x) for x in res.x],
        "E_hartree": float(res.fun),
        "success": bool(res.success),
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def compare_row(label: str, E_pred_ha: float, E_ref_ha: float) -> str:
    from constants import HARTREE_EV
    ev_pred = E_pred_ha * HARTREE_EV
    ev_ref = E_ref_ha * HARTREE_EV
    denom = abs(E_ref_ha) if E_ref_ha != 0 else 1.0
    err_pct = 100.0 * abs(E_pred_ha - E_ref_ha) / denom
    return (
        f"{label:<18} E_pred={E_pred_ha:10.4f} Ha ({ev_pred:9.2f} eV)   "
        f"E_exp={E_ref_ha:10.4f} Ha ({ev_ref:9.2f} eV)   "
        f"err={err_pct:6.2f}%"
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

import json
import os

LOG_PATH = os.path.join(os.path.dirname(__file__), "results", "log.jsonl")


def log_experiment(record: dict) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
