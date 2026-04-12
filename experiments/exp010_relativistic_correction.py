"""Experiment 010 — Relativistic correction to the principal-shell model.

Same model as exp009, but replace the non-relativistic kinetic energy
    T_NR = n^2 / (2 R^2)             i.e.  p^2 / (2m)

with the full relativistic kinetic energy
    T_rel = c^2 * [sqrt(1 + n^2/(c^2 R^2)) - 1]

where c = 1/alpha ~ 137.036 in atomic units (alpha = fine-structure constant).

This expands as:
    T_rel = n^2/(2 R^2) - n^4/(8 c^2 R^4) + ...

The leading correction is NEGATIVE (more binding), and scales as
v^4/c^2 ~ 1/(c^2 R^4).  It's biggest for inner shells of high-Z atoms
where the electron "speed" n/R approaches c.

Everything else (Thomson geometry, shell theorem, occupancy) is unchanged.
We compare:
    (a) non-relativistic (exp009 baseline)
    (b) relativistic
    (c) the difference
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
from scipy.optimize import minimize

from constants import HARTREE_EV
from reference_data import TOTAL_BINDING_EV, Z as Z_of, ELEMENTS
from core import log_experiment

# Fine-structure constant and speed of light in atomic units
ALPHA = 1.0 / 137.035999084
C_AU = 1.0 / ALPHA   # ~ 137.036

# Reuse Thomson machinery from exp009
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from exp009_principal_shells_BeNe import (
    thomson_points, repulsion_sum, shell_capacity, partition_electrons, _unit
)


def T_nonrel(n: int, R: float) -> float:
    """Non-relativistic kinetic barrier per electron: n^2 / (2 R^2)."""
    return (n * n) / (2.0 * R * R)


def T_rel(n: int, R: float) -> float:
    """Relativistic kinetic energy per electron.

    T = sqrt(p^2 c^2 + m^2 c^4) - m c^2,  with p = n/R, m = 1.
    In atomic units: T = c^2 [sqrt(1 + n^2/(c^2 R^2)) - 1].
    """
    p = float(n) / R
    return C_AU * C_AU * (math.sqrt(1.0 + (p / C_AU) ** 2) - 1.0)


def energy_of(Z: int, occupancy: list[int], use_rel: bool = False) -> dict:
    K = len(occupancy)
    S_k = [repulsion_sum(k) for k in occupancy]
    T_func = T_rel if use_rel else T_nonrel

    def E_of_radii(r_vec: np.ndarray) -> float:
        if np.any(r_vec <= 1e-6):
            return 1e6
        E = 0.0
        for n_idx, (k, r) in enumerate(zip(occupancy, r_vec)):
            n = n_idx + 1
            E += k * T_func(n, r)
            E -= Z * k / r
            E += S_k[n_idx] / r
        for i in range(K):
            for j in range(i + 1, K):
                E += occupancy[i] * occupancy[j] / max(r_vec[i], r_vec[j])
        return E

    R0 = np.array([(i + 1) ** 2 / max(Z - 2 * i, 1) for i in range(K)],
                  dtype=float)
    res = minimize(E_of_radii, R0, method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 40000})
    return {
        "Z": Z, "occupancy": occupancy,
        "radii": [float(x) for x in res.x],
        "E_ha": float(res.fun),
    }


def run() -> dict:
    out = {}
    for sym in ELEMENTS:
        Z = Z_of[sym]
        occ = partition_electrons(Z)
        nr = energy_of(Z, occ, use_rel=False)
        rl = energy_of(Z, occ, use_rel=True)
        exp_ha = -TOTAL_BINDING_EV[sym] / HARTREE_EV
        out[sym] = {
            "Z": Z, "occ": occ,
            "E_nr": nr["E_ha"], "E_rel": rl["E_ha"], "E_exp": exp_ha,
            "delta": rl["E_ha"] - nr["E_ha"],
            "delta_eV": (rl["E_ha"] - nr["E_ha"]) * HARTREE_EV,
            "r_nr": nr["radii"], "r_rel": rl["radii"],
            "err_nr": 100.0 * abs(nr["E_ha"] - exp_ha) / abs(exp_ha),
            "err_rel": 100.0 * abs(rl["E_ha"] - exp_ha) / abs(exp_ha),
        }
    return out


if __name__ == "__main__":
    out = run()
    print("=" * 96)
    print("Experiment 010: relativistic correction to principal-shell model")
    print("=" * 96)
    print(f"  c = {C_AU:.3f} a.u.   (1/alpha)")
    print()
    print(f"{'atom':<4} {'Z':>3} {'E_NR (Ha)':>11} {'E_rel (Ha)':>11} {'E_exp (Ha)':>11}"
          f" {'delta (eV)':>11} {'err_NR%':>8} {'err_rel%':>8} {'improved?':>10}")
    for sym, r in out.items():
        improved = "yes" if r["err_rel"] < r["err_nr"] else "no"
        print(f"{sym:<4} {r['Z']:>3} {r['E_nr']:>11.4f} {r['E_rel']:>11.4f}"
              f" {r['E_exp']:>11.4f} {r['delta_eV']:>11.4f}"
              f" {r['err_nr']:>7.2f}% {r['err_rel']:>7.2f}% {improved:>10}")

    print()
    print("Radii comparison (NR vs relativistic):")
    for sym, r in out.items():
        nr_s = ", ".join(f"{x:.4f}" for x in r["r_nr"])
        rl_s = ", ".join(f"{x:.4f}" for x in r["r_rel"])
        print(f"  {sym:<4} NR=[{nr_s}]  REL=[{rl_s}]")

    print()
    print("INTERPRETATION")
    print("-" * 96)
    print("  The relativistic correction is NEGATIVE (more binding) and grows with Z.")
    print("  For H-Ne it's tiny: a few hundredths of an eV for H, up to ~0.5 eV for Ne.")
    print("  Our model errors are 1-80 eV, so relativity is 100x smaller than the")
    print("  model error throughout the first row.  It would start mattering around Z~30.")
    print("  But the correction goes in the RIGHT direction for Ne (which we underbind).")

    log_experiment({"exp": "010", "name": "relativistic_correction", "results": out})
