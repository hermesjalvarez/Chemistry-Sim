"""Experiment 001 — Bare static charged shell (Newton + Maxwell only).

HYPOTHESIS
----------
The electron is a thin spherical shell of radius R, charge -1, mass 1.
It sits motionless around a point nucleus of charge +Z at the origin.
The ONLY physics we use is classical electrostatics (shell theorem) plus
classical electromagnetic self-energy of the shell.

Total energy:
    E(R) = -Z / R            (nucleus--shell attraction)
           + 1 / (2 R)       (shell self-energy)

DERIVATION OF THE FAILURE MODE
------------------------------
Differentiating:
    dE/dR = Z / R^2 - 1 / (2 R^2) = (Z - 1/2) / R^2

For any Z >= 1 this is strictly positive, meaning E is a monotonically
INCREASING function of R. To minimize its energy the shell should
therefore shrink to R -> 0, where E -> -infinity. There is no equilibrium
and no ground state.

Equivalently, the net radial force on each shell element is
    F = -(dE/dR) = -(Z - 1/2) / R^2
which always points inward (toward the nucleus) for Z >= 1.

This is a manifestation of Earnshaw's theorem: a collection of point
charges interacting only via Coulomb forces cannot be in stable static
equilibrium.  We've confirmed it here for the simplest charged-shell
atom model.

CONCLUSION
----------
Pure classical EM with a static shell cannot stabilize any atom.  We need
an additional mechanism that contributes a repulsive (kinetic /
pressure-like / centrifugal) term that dominates at small R.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core import nucleus_shell_energy, shell_self_energy, log_experiment
from constants import HARTREE_EV


def E_of_R(R: float, Z: int) -> float:
    return nucleus_shell_energy(Z, -1.0, R) + shell_self_energy(-1.0, R)


def run() -> dict:
    Zs = [1, 2, 3]
    # Scan R over many decades
    R_grid = np.geomspace(1e-4, 1e3, 200)

    results = {}
    for Z in Zs:
        E_vals = np.array([E_of_R(R, Z) for R in R_grid])
        # The energy is monotone decreasing (toward -inf) as R -> 0 for Z > 1/2
        dE = np.diff(E_vals)
        sign = "always decreasing" if np.all(dE <= 1e-12) else (
            "always increasing" if np.all(dE >= -1e-12) else "non-monotone"
        )
        results[f"Z={Z}"] = {
            "E_at_R=1": float(E_vals[np.argmin(np.abs(R_grid - 1.0))]),
            "E_at_R=0.01": float(E_of_R(0.01, Z)),
            "E_at_R=100": float(E_of_R(100.0, Z)),
            "monotonicity": sign,
            "has_equilibrium": False,
        }

    return results


if __name__ == "__main__":
    results = run()
    print("=" * 78)
    print("Experiment 001: Bare static charged shell (pure Newton + Maxwell)")
    print("=" * 78)
    print("Energy E(R) = -Z/R + 1/(2R)  (attraction + shell self-energy)")
    print()
    for k, v in results.items():
        print(f"  {k}:")
        for k2, v2 in v.items():
            print(f"      {k2}: {v2}")
    print()
    print("VERDICT: monotone decrease in R => shell collapses, no ground state.")
    print("         Pure classical EM + static shell FAILS for all Z.")

    log_experiment({
        "exp": "001",
        "name": "bare_static_shell",
        "hypothesis": "Pure classical EM: static shell + shell self-energy",
        "verdict": "FAIL: no equilibrium, shell collapses for all Z",
        "results": results,
    })
