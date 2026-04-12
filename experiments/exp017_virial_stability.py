"""Experiment 017 — Virial stability and natural shell boundaries.

MOTIVATION:
Exp015-016 test whether shells and quantization EMERGE from free optimization.
This experiment takes a different approach: instead of optimizing, it ANALYZES
why certain configurations are stable and others aren't.

THE VIRIAL THEOREM AS A STABILITY CRITERION:
For a system with T ∝ 1/R² and V ∝ 1/R, the virial theorem gives:
    2<T> + <V> = 0   at equilibrium.

For our shell model:
    T_total = Σ L_i² / (2 r_i²)
    V_total = -Σ Z/r_i + Σ_{i<j} 1/max(r_i, r_j)

At the energy minimum, 2T + V = 0 holds automatically (since T ∝ 1/R²
and V ∝ 1/R for each shell).  But what about the INDIVIDUAL shells?
The virial ratio for shell j:
    η_j = -2 T_j / V_j
where V_j includes both nuclear attraction and repulsion.

If η_j ≠ 1, the shell is not in its own internal virial equilibrium —
it's being held in place by the coupling to other shells.

THIS TELLS US: which electrons are "happy" (η ≈ 1) and which are
"stressed" (η far from 1)?  Stressed electrons might be the ones that
the model gets wrong for ionization energies.

ADDITIONAL TEST: "Shell breathing mode stability"
For a multi-shell atom, perturb each shell radius independently and
compute the restoring force.  Shells with weak restoring force are
loosely bound → easy to ionize.  This gives us a per-shell "binding
stiffness" that should correlate with ionization energy.

ADDITIONAL TEST: "Scaling analysis"
For a single shell with k electrons, L angular momentum, nuclear charge Z_eff:
    E(R) = k L²/(2R²) - k Z_eff/R + S_k/R
    Minimum at R* = k L² / (k Z_eff - S_k)
    E* = -(k Z_eff - S_k)² / (2 k L²)

The ionization energy of the outermost electron is approximately:
    IE ≈ dE/dk|_{outer shell}
This partial derivative tells us how the energy changes when we add/remove
one electron from the outermost shell.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
from scipy.optimize import minimize

from constants import HARTREE_EV
from reference_data import TOTAL_BINDING_EV, FIRST_IE_EV, Z as Z_of, ELEMENTS
from core import log_experiment

sys.path.insert(0, os.path.dirname(__file__))
from exp009_principal_shells_BeNe import repulsion_sum as thomson_S
from exp013_electron_clouds import (
    cloud_repulsion_C, screen_penetration, fill_octet
)


A_BEST = 0.36
DELTA_BEST = 0.83


def energy_and_decompose(Z: int, occupancy: list[int],
                          a: float, delta: float) -> dict:
    """Compute energy AND decompose it per-shell (kinetic, potential, etc.)."""
    K = len(occupancy)
    S_k = [cloud_repulsion_C(k, a) for k in occupancy]
    scr = screen_penetration(delta)

    def E_of_radii(r_vec):
        if np.any(r_vec <= 1e-6):
            return 1e6
        E = 0.0
        for j in range(K):
            n_j = j + 1
            k_j = occupancy[j]
            r_j = r_vec[j]
            E += k_j * (n_j**2) / (2.0 * r_j**2)
            Z_eff_j = float(Z)
            for i in range(K):
                if i == j:
                    continue
                if r_vec[i] < r_j * (1 - 1e-9):
                    s = scr(i+1, n_j, r_vec[i], r_j)
                    Z_eff_j -= s * occupancy[i]
            E -= Z_eff_j * k_j / r_j
            E += S_k[j] / r_j
        return E

    R0 = np.array([(j+1)**2 / max(Z - sum(occupancy[:j]), 1)
                    for j in range(K)], dtype=float)
    R0 = np.clip(R0, 0.01, 100.0)
    res = minimize(E_of_radii, R0, method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 60000})
    r_opt = res.x

    # Decompose energy per shell
    shells = []
    for j in range(K):
        n_j = j + 1
        k_j = occupancy[j]
        r_j = r_opt[j]

        T_j = k_j * (n_j**2) / (2.0 * r_j**2)

        Z_eff_j = float(Z)
        for i in range(K):
            if i == j:
                continue
            if r_opt[i] < r_j * (1 - 1e-9):
                s = scr(i+1, n_j, r_opt[i], r_j)
                Z_eff_j -= s * occupancy[i]

        V_nuc_j = -Z_eff_j * k_j / r_j
        V_rep_j = S_k[j] / r_j
        E_j = T_j + V_nuc_j + V_rep_j

        # Virial ratio for this shell (should be 1 if in own equilibrium)
        V_total_j = V_nuc_j + V_rep_j
        virial_j = -2 * T_j / V_total_j if abs(V_total_j) > 1e-10 else float('inf')

        # Breathing mode stiffness: d²E/dr² at r_opt[j]
        dr = 1e-5
        E_plus = float(E_of_radii(np.array([r_opt[i] if i != j else r_j + dr
                                             for i in range(K)])))
        E_minus = float(E_of_radii(np.array([r_opt[i] if i != j else r_j - dr
                                              for i in range(K)])))
        E_center = float(E_of_radii(r_opt))
        stiffness_j = (E_plus - 2*E_center + E_minus) / (dr**2)

        shells.append({
            "n": n_j, "k": k_j, "r": float(r_j),
            "T": float(T_j), "V_nuc": float(V_nuc_j), "V_rep": float(V_rep_j),
            "E_shell": float(E_j),
            "Z_eff": float(Z_eff_j),
            "virial": float(virial_j),
            "stiffness": float(stiffness_j),
        })

    return {
        "Z": Z, "occupancy": occupancy,
        "E_ha": float(res.fun),
        "radii": [float(x) for x in r_opt],
        "shells": shells,
    }


def analytical_IE_estimate(shell_info: dict) -> float:
    """Estimate IE from the outermost shell's analytical formula.

    For the outermost shell with k electrons, L=n, Z_eff, S_k:
        E_shell(k) = k n² / (2R²) - Z_eff k / R + S_k / R
    At optimal R:
        R* = k n² / (k Z_eff - S_k)
        E* = -(k Z_eff - S_k)² / (2 k n²)

    IE ≈ E*(k-1) - E*(k)  [removing one electron from outer shell]
    This is an ANALYTICAL estimate, no re-optimization needed.
    """
    n = shell_info["n"]
    k = shell_info["k"]
    Z_eff = shell_info["Z_eff"]

    # Energy with k electrons (already computed)
    S_k = cloud_repulsion_C(k, A_BEST)
    denom_k = k * Z_eff - S_k
    if denom_k <= 0:
        return 0.0
    E_k = -(denom_k)**2 / (2.0 * k * n**2)

    # Energy with k-1 electrons (approximate: same Z_eff, different S)
    if k <= 1:
        return -E_k  # removing the only electron
    S_km1 = cloud_repulsion_C(k - 1, A_BEST)
    denom_km1 = (k-1) * Z_eff - S_km1
    if denom_km1 <= 0:
        return -E_k
    E_km1 = -(denom_km1)**2 / (2.0 * (k-1) * n**2)

    return E_km1 - E_k  # positive = costs energy to remove


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

ATOMS_SHORT = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]


def run() -> dict:
    out = {}

    for sym in ATOMS_SHORT:
        Z = Z_of[sym]
        occ = fill_octet(Z)
        res = energy_and_decompose(Z, occ, A_BEST, DELTA_BEST)

        exp_ha = -TOTAL_BINDING_EV[sym] / HARTREE_EV
        res["E_exp"] = exp_ha
        res["err_pct"] = 100.0 * abs(res["E_ha"] - exp_ha) / abs(exp_ha)

        # Analytical IE estimate from outermost shell
        outer = res["shells"][-1]
        IE_analytic_ha = analytical_IE_estimate(outer)
        IE_analytic_eV = IE_analytic_ha * HARTREE_EV
        IE_exp = FIRST_IE_EV[sym]

        res["IE_analytic_eV"] = IE_analytic_eV
        res["IE_exp_eV"] = IE_exp
        res["IE_analytic_err_pct"] = 100.0 * abs(IE_analytic_eV - IE_exp) / IE_exp

        out[sym] = res

    return out


if __name__ == "__main__":
    out = run()
    print("=" * 120)
    print("Experiment 017: virial stability and shell diagnostics")
    print("  Using best model: cloud a=0.36 + screening δ=0.83")
    print("=" * 120)

    # Per-shell decomposition
    print("\nPER-SHELL ENERGY DECOMPOSITION:")
    print("-" * 120)
    for sym in ATOMS_SHORT:
        r = out[sym]
        print(f"\n  {sym} (Z={r['Z']}, occ={r['occupancy']}, E={r['E_ha']:.4f} Ha,"
              f" err={r['err_pct']:.2f}%)")
        print(f"    {'n':>3} {'k':>3} {'radius':>8} {'T':>10} {'V_nuc':>10} {'V_rep':>10}"
              f" {'E_shell':>10} {'Z_eff':>6} {'virial':>7} {'stiff':>8}")
        for s in r["shells"]:
            print(f"    {s['n']:>3} {s['k']:>3} {s['r']:>8.4f} {s['T']:>10.4f}"
                  f" {s['V_nuc']:>10.4f} {s['V_rep']:>10.4f} {s['E_shell']:>10.4f}"
                  f" {s['Z_eff']:>6.2f} {s['virial']:>7.3f} {s['stiffness']:>8.2f}")

    # Analytical IE estimates
    print("\n\nANALYTICAL IONIZATION ENERGY ESTIMATES (from outermost shell):")
    print("-" * 80)
    print(f"  {'atom':<4} {'IE analytic':>12} {'IE exp':>10} {'err%':>8}"
          f"  {'outer n':>8} {'outer k':>8} {'Z_eff':>6} {'radius':>8}")
    for sym in ATOMS_SHORT:
        r = out[sym]
        outer = r["shells"][-1]
        print(f"  {sym:<4} {r['IE_analytic_eV']:>12.3f} {r['IE_exp_eV']:>10.3f}"
              f" {r['IE_analytic_err_pct']:>7.1f}%"
              f"  {outer['n']:>8} {outer['k']:>8} {outer['Z_eff']:>6.2f}"
              f" {outer['r']:>8.4f}")

    # Virial analysis
    print("\n\nVIRIAL ANALYSIS: η = -2T/V (should be 1.0 at equilibrium)")
    print("-" * 80)
    for sym in ATOMS_SHORT:
        r = out[sym]
        vs = [f"n{s['n']}:{s['virial']:.3f}" for s in r["shells"]]
        print(f"  {sym:<4} {', '.join(vs)}")

    print("\n\nSTIFFNESS ANALYSIS: d²E/dr² (higher = more tightly bound)")
    print("-" * 80)
    for sym in ATOMS_SHORT:
        r = out[sym]
        ss = [f"n{s['n']}:{s['stiffness']:.1f}" for s in r["shells"]]
        print(f"  {sym:<4} {', '.join(ss)}")

    print()
    print("INTERPRETATION")
    print("=" * 120)
    print("  The virial ratio η tells us whether each shell is in its own")
    print("  internal equilibrium (η=1) or being 'held' by coupling to other shells.")
    print()
    print("  The stiffness tells us how tightly bound each shell is.")
    print("  The outermost shell should have the LOWEST stiffness — it's the")
    print("  easiest to perturb, hence the first to ionize.")
    print()
    print("  The analytical IE estimate avoids the 'catastrophic differencing'")
    print("  problem of exp014 (subtracting two large numbers).  Instead it")
    print("  estimates IE directly from the outermost shell's properties.")

    log_experiment({"exp": "017", "name": "virial_stability",
                    "results": {sym: {"E_ha": v["E_ha"],
                                      "IE_analytic_eV": v["IE_analytic_eV"]}
                                for sym, v in out.items()}})
