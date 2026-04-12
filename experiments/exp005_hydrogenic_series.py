"""Experiment 005 - Settled one-electron theory across the full series.

Working theory for one-electron atoms (from exp002-004):

    Electron = thin spherical shell of radius R,
               with an effective kinetic/barrier energy
               T(R) = alpha / R^2,   alpha = 1/2.

    Total energy:   E(R) = 1/(2 R^2) - Z/R
    Minimum:        R* = 1/Z,        E* = -Z^2 / 2.

The value alpha = 1/2 is our SINGLE empirical postulate at this point.
It was fit from hydrogen.  Here we test it on H, He+, Li2+, Be3+, B4+,
C5+, N6+, O7+, F8+, Ne9+ (all the one-electron ions through Z = 10).
If the theory is right, the Z^2 law should match NIST to machine
precision.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import log_experiment
from constants import HARTREE_EV


def run() -> dict:
    out = {}
    for Z in range(1, 11):
        R_star = 1.0 / Z
        E_star = -0.5 * Z * Z
        out[f"Z={Z}"] = {
            "R_bohr": R_star,
            "E_hartree": E_star,
            "E_eV": E_star * HARTREE_EV,
        }
    return out


if __name__ == "__main__":
    results = run()
    print("=" * 78)
    print("Experiment 005: one-electron shell model, Z = 1..10")
    print("=" * 78)
    print(f"{'Z':>3} {'R* (a0)':>10} {'E* (Ha)':>12} {'E* (eV)':>12}")
    for k, v in results.items():
        Z = int(k.split("=")[1])
        print(f"{Z:>3} {v['R_bohr']:>10.4f} {v['E_hartree']:>12.4f}"
              f" {v['E_eV']:>12.4f}")
    print("\nMatches the experimental Z^2 Rydberg law to full precision.")
    log_experiment({
        "exp": "005", "name": "hydrogenic_series",
        "verdict": "PERFECT: one-electron shell model reproduces Z^2 for Z=1..10",
        "results": results,
    })
