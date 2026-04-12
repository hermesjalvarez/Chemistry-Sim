"""Experiment 002 — Rigid rotating shell with free angular momentum.

HYPOTHESIS
----------
Same shell as in exp001, but now the shell is allowed to rigidly rotate
about an axis through its center with angular momentum L.  Nothing in
classical mechanics prevents this, and classically L is a free real
parameter.

This gives an additional rotational kinetic energy
    T_rot = L^2 / (2 I)
where the moment of inertia of a thin spherical shell is I = (2/3) m R^2.
With m = 1 (atomic units) we have
    T_rot(R) = 3 L^2 / (4 R^2).

Total energy (no shell self-energy for now):
    E(R; L) = 3 L^2 / (4 R^2) - Z / R

This is finally bounded below!  Setting dE/dR = 0:
    -3 L^2 / (2 R^3) + Z / R^2 = 0
    R*  = 3 L^2 / (2 Z)
    E*  = -Z^2 / (3 L^2)

Equilibrium radius scales as 1/Z, and equilibrium energy as Z^2 --- same
SCALING as the famous hydrogen result, but with a different coefficient.

QUESTIONS THIS EXPERIMENT ANSWERS
---------------------------------
1. Does any value of L reproduce the experimental hydrogen result?
2. If yes, does that same L also reproduce He+, Li2+, ...?

NOTE ON "IS ROTATION ALLOWED WITHOUT RADIATION?"
------------------------------------------------
A uniformly charged, rigidly rotating spherical shell is a TIME-INDEPENDENT
charge distribution in the lab frame (the charge density looks the same at
every instant).  Its electric multipole moments are constant in time, so
by Larmor / classical electrodynamics it does NOT radiate.  So this
configuration can in principle persist indefinitely without contradicting
Maxwell.  That's already a non-trivial virtue of the shell model.

CONCLUSION
----------
Classical rigid rotation gives a stable atom of ANY size (one-parameter
family in L).  To turn this into a theory with predictive content we need
an extra principle that picks out L.  We will see in later experiments
that the most minimal such principle is: "there is a fundamental unit of
action L_0, and the ground state has L = L_0".  This is effectively one
Bohr-like postulate.  We will then have to check whether the SHELL model
with that postulate reproduces hydrogen -- it turns out it does not,
because for a rigid shell L_0 = 1 atomic unit gives the wrong coefficient.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core import log_experiment
from constants import HARTREE_EV
from reference_data import TOTAL_BINDING_EV


def equilibrium(Z: int, L: float) -> tuple[float, float]:
    R = 3.0 * L * L / (2.0 * Z)
    E = -Z * Z / (3.0 * L * L)
    return R, E


def run() -> dict:
    results: dict = {}

    # Question 1: does L = 1 reproduce hydrogen?
    R_H, E_H = equilibrium(Z=1, L=1.0)
    H_ref_ha = -TOTAL_BINDING_EV["H"] / HARTREE_EV
    results["H_at_L=1"] = {
        "R_pred": R_H, "E_pred_ha": E_H,
        "E_pred_eV": E_H * HARTREE_EV,
        "E_ref_ha": H_ref_ha, "E_ref_eV": -TOTAL_BINDING_EV["H"],
        "ratio_pred_to_ref": E_H / H_ref_ha,
    }

    # Question 2: what L would fit hydrogen exactly?
    # E* = -1/(3 L^2) = -0.5  =>  L^2 = 2/3, L = sqrt(2/3) ~ 0.8165
    L_fit = float(np.sqrt(2.0 / 3.0))
    results["L_that_fits_H"] = {
        "L": L_fit,
        "note": "Not 1, so 'one unit of action = 1 a.u.' does not fit the shell model",
    }

    # Question 3: if we lock L = L_fit, does it also fit He+ and Li2+?
    results["with_L_fit_for_H"] = {}
    for Z, ion in [(1, "H"), (2, "He+"), (3, "Li2+")]:
        R, E = equilibrium(Z=Z, L=L_fit)
        # Exact hydrogenic experiment: E_ion = -Z^2 / 2 Ha
        E_ref = -0.5 * Z * Z
        results["with_L_fit_for_H"][ion] = {
            "Z": Z, "R_pred": R, "E_pred_ha": E,
            "E_ref_ha": E_ref,
            "err_pct": 100.0 * abs(E - E_ref) / abs(E_ref),
        }

    return results


if __name__ == "__main__":
    results = run()
    print("=" * 78)
    print("Experiment 002: Rigid rotating shell with free angular momentum")
    print("=" * 78)

    r = results["H_at_L=1"]
    print(f"\nH with L=1 (one atomic unit of action):")
    print(f"    R* = {r['R_pred']:.4f} a0   E* = {r['E_pred_ha']:.4f} Ha"
          f" = {r['E_pred_eV']:.4f} eV")
    print(f"    (exp -13.6058 eV)   ratio predicted/experiment ="
          f" {r['ratio_pred_to_ref']:.4f}")
    print()
    print(f"L that would fit H exactly = sqrt(2/3) =  {results['L_that_fits_H']['L']:.6f}")
    print(f"    note: {results['L_that_fits_H']['note']}")
    print()
    print("With that L fixed, one-electron ions:")
    for ion, v in results["with_L_fit_for_H"].items():
        print(f"    {ion:5s}: R={v['R_pred']:.4f}  E={v['E_pred_ha']:.4f} Ha"
              f"  (ref {v['E_ref_ha']:.4f})  err {v['err_pct']:.2f}%")

    print()
    print("VERDICT:")
    print("  * Rigid rotation stabilizes the shell for ANY L > 0 (good).")
    print("  * Scaling  R ~ 1/Z,  E ~ -Z^2  is correct (good).")
    print("  * But L = 1 atomic unit gives H off by a factor of 2/3.")
    print("  * A refit L = sqrt(2/3) reproduces H and CONSISTENTLY reproduces")
    print("    all hydrogenic ions H, He+, Li2+, ... exactly.")
    print("  * So: the rigid rotating shell can match one-electron atoms")
    print("    once the 'unit of angular momentum' is chosen empirically.")
    print("    That single empirical parameter is the first hint of a")
    print("    fundamental quantum of action.")

    log_experiment({
        "exp": "002",
        "name": "rigid_rotating_shell",
        "hypothesis": "Shell rigidly rotates with free angular momentum L",
        "verdict": "PARTIAL: stable for any L>0, Z^2 scaling right, "
                   "but L=1 atomic unit gives wrong constant (need L=sqrt(2/3))",
        "results": results,
    })
