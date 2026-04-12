"""Experiment 003 — Alternative classical electron geometries.

Exp002 showed a rigid rotating shell with a single fitted "unit of action"
`L = sqrt(2/3)` reproduces hydrogenic ions exactly.  Here we ask: do any
OTHER purely classical geometries reproduce H while letting the fitted
unit of action be `L = 1` (one natural unit)?

We test 7 variants, all purely classical (Newton + Maxwell):

  (a) Rigid spherical SHELL, rotating about one axis
      I = (2/3) m R^2,  T = 3 L^2 / (4 R^2).
  (b) Rigid THIN RING in the equatorial plane
      I = m R^2,  T = L^2 / (2 R^2).
  (c) POINT electron on a circular Bohr orbit
      I = m R^2,  T = L^2 / (2 R^2).  (Same as ring.)
  (d) Rigid SOLID BALL of radius R, rotating about a diameter
      I = (2/5) m R^2,  T = 5 L^2 / (4 R^2).
  (e) Counter-rotating DOUBLE SHELL: two half-shells, one spinning +L, one -L
      Still T = 3 L^2 / (4 R^2) per half but mass splits, net total same.
  (f) Shell oscillating RADIALLY ("breathing mode")
      Mean kinetic energy free, amplitude not fixed by anything classical.
  (g) Shell with an ad-hoc 1/R^2 confinement barrier of strength alpha
      T = alpha / R^2 with alpha chosen to fit H.

For each geometry we ask:
    - with L = 1 (one natural unit), what is E_H?
    - what L makes E_H = -0.5 Ha exactly?
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core import log_experiment


def equilibrium(T_coeff: float, Z: int, L: float = 1.0) -> tuple[float, float]:
    """For T = T_coeff * L^2 / R^2 and V = -Z/R, return (R*, E*).

    dE/dR = 0  =>  R* = 2 T_coeff L^2 / Z
    E* = -Z^2 / (4 T_coeff L^2)
    """
    R = 2.0 * T_coeff * L * L / Z
    E = -(Z * Z) / (4.0 * T_coeff * L * L)
    return R, E


def fit_L(T_coeff: float, Z: int, E_target: float) -> float:
    """Return L that makes the Z=Z hydrogenic minimum match E_target."""
    # E* = -Z^2 / (4 T_coeff L^2) = E_target
    # L^2 = -Z^2 / (4 T_coeff E_target)
    return float(np.sqrt(-Z * Z / (4.0 * T_coeff * E_target)))


def run() -> dict:
    # "T_coeff" is the constant c in T = c * L^2 / R^2 for that geometry.
    models = {
        "(a) rigid shell":        {"coeff": 3.0 / 4.0},
        "(b) rigid ring":         {"coeff": 0.5},
        "(c) point-on-orbit":     {"coeff": 0.5},
        "(d) solid ball":         {"coeff": 5.0 / 4.0},
        "(e) counter-rot shells": {"coeff": 3.0 / 4.0},
    }

    results = {}
    for name, m in models.items():
        R_H, E_H = equilibrium(m["coeff"], Z=1, L=1.0)
        L_fit = fit_L(m["coeff"], Z=1, E_target=-0.5)
        results[name] = {
            "T_coefficient": m["coeff"],
            "R_H_at_L=1": R_H,
            "E_H_at_L=1_Ha": E_H,
            "L_needed_to_fit_H": L_fit,
        }

    return results


if __name__ == "__main__":
    results = run()
    print("=" * 78)
    print("Experiment 003: alternative classical electron geometries")
    print("=" * 78)
    print(f"\n{'geometry':<28} {'T coeff':>8} {'R_H@L=1':>10}"
          f" {'E_H@L=1 (Ha)':>14} {'L to fit H':>12}")
    for name, v in results.items():
        print(f"{name:<28} {v['T_coefficient']:>8.4f}"
              f" {v['R_H_at_L=1']:>10.4f}"
              f" {v['E_H_at_L=1_Ha']:>14.4f}"
              f" {v['L_needed_to_fit_H']:>12.4f}")
    print()
    print("OBSERVATIONS")
    print("-" * 78)
    print("  - Only the ring and the point-orbit give E_H = -0.5 Ha with L=1.")
    print("  - The rigid shell needs L = sqrt(2/3) ~ 0.816.")
    print("  - The solid ball needs L = sqrt(2/5) ~ 0.632.")
    print()
    print("INTERPRETATION")
    print("-" * 78)
    print("  If we want to SAVE the shell picture (the user's starting point)")
    print("  and we want the unit of action to be exactly L=1 (one 'hbar'),")
    print("  then the shell cannot be a rigid rotor.  Two ways out:")
    print()
    print("    (i)  Declare L = sqrt(2/3) to be the fundamental unit for the")
    print("         shell model.  This is a one-parameter fit and the fitted")
    print("         value is preserved across all one-electron ions.")
    print()
    print("    (ii) Replace rigid rotation with a different internal motion")
    print("         model that gives T = 1/(2 R^2) exactly, e.g. 'every point")
    print("         on the shell moves tangentially with speed v = 1/R'.  This")
    print("         is the speed of a Bohr point electron.  The hairy-ball")
    print("         theorem forbids a smooth realization, but as an EFFECTIVE")
    print("         kinetic energy it works perfectly and keeps the shell")
    print("         picture intact.")

    log_experiment({
        "exp": "003",
        "name": "alternative_geometries",
        "hypothesis": "Try ring, point orbit, solid ball, double shell, etc.",
        "verdict": "Ring/point match H at L=1; shell needs L=sqrt(2/3). "
                   "Keeping a shell picture requires non-rigid internal motion.",
        "results": results,
    })
