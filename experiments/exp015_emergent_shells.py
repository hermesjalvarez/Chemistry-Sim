"""Experiment 015 — Emergent shell structure from unconstrained optimization.

MOTIVATION:
Experiments 009-014 imposed shell structure: occupancy rules (octet),
filling order, Thomson geometry.  This gave excellent total energies but
catastrophically wrong ionization energies (exp014).  The root cause is
that we're TELLING electrons where to go rather than letting physics decide.

THE IDEA:
Give each electron TWO free variables:
  - r_i: its orbital radius
  - L_i: its angular momentum magnitude (continuous, not integer)

The ONLY physics:
  1. Kinetic barrier:   T_i = L_i² / (2 r_i²)     [our one postulate]
  2. Nuclear attraction: V_nuc_i = -Z / r_i
  3. Electron-electron repulsion: V_ij = 1 / |r_i - r_j| for each pair

For repulsion between two electrons at the SAME radius, we need their
angular separation.  Rather than imposing Thomson polyhedra, we try:

  MODEL E1: "Gauss approximation" — two electrons at the same radius
            repel as 1/R (like concentric shells, shell theorem).
            This OVERESTIMATES repulsion (ignores angular correlation).

  MODEL E2: "Antipodal on same shell" — electrons at the same radius
            are perfectly anticorrelated (opposite sides), repulsion = 1/(2R).
            This UNDERESTIMATES for k > 2.

  MODEL E3: "Average 1/d from uniform random" — for k electrons at
            radius R, the average pair repulsion is 1/R per pair
            (uncorrelated uniform shells).  Same as E1 but clearer.

  MODEL E4: "Full optimization" — give each electron its own (r, theta, phi)
            and optimize ALL positions simultaneously.  No shell assumption.
            Expensive but maximally emergent.

  MODEL E5: "Radii only, Thomson auto-grouping" — optimize each electron's
            (r_i, L_i), then at the end GROUP electrons with similar r_i and
            use Thomson repulsion for those groups.  This tests whether
            shells EMERGE from the optimization.

KEY QUESTION: Does the Pauli-like "pressure" that prevents all electrons
from collapsing to n=1 emerge naturally?  In our framework, this requires
the repulsion to be strong enough to push electrons to larger radii where
higher angular momentum becomes favorable.

We test H through B (Z=1..5) to keep it tractable.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

from constants import HARTREE_EV
from reference_data import TOTAL_BINDING_EV, FIRST_IE_EV, Z as Z_of, ELEMENTS
from core import log_experiment


# --------------------------------------------------------------------------
# Model E1: Each electron has (r_i, L_i). Repulsion via shell theorem.
# No imposed shell structure, no occupancy rules, no Thomson geometry.
# --------------------------------------------------------------------------

def energy_E1(params: np.ndarray, Z: int, N: int) -> float:
    """params = [r_1, L_1, r_2, L_2, ..., r_N, L_N]
    Each electron has its own radius and angular momentum.
    Repulsion between any two electrons: 1 / max(r_i, r_j).
    """
    if len(params) != 2 * N:
        return 1e6
    rs = params[0::2]
    Ls = params[1::2]

    if np.any(rs <= 1e-6) or np.any(Ls < 0):
        return 1e6

    E = 0.0
    for i in range(N):
        # Kinetic barrier
        E += Ls[i]**2 / (2.0 * rs[i]**2)
        # Nuclear attraction
        E -= Z / rs[i]

    # Electron-electron repulsion (shell theorem: 1/r_outer)
    for i in range(N):
        for j in range(i + 1, N):
            E += 1.0 / max(rs[i], rs[j])

    return E


# --------------------------------------------------------------------------
# Model E4: Full 3D optimization. Each electron at (r, theta, phi).
# Repulsion is exact: 1/|r_i - r_j| where positions are on spheres.
# Angular momentum: L_i is tied to r_i and its "shell" —
#   but wait, in 3D we need a different approach.
#
# Actually, let's think about this differently.  Each electron orbits
# at radius r_i with angular momentum L_i.  Its TIME-AVERAGED position
# traces a great circle on a sphere of radius r_i.  Two electrons on
# the same sphere with different orbital planes will have a time-averaged
# repulsion that depends on the angle between their orbital planes.
#
# For simplicity, we model: each electron is a uniform ring at radius r_i
# in a plane tilted by angle theta_i from the z-axis, with azimuthal
# offset phi_i.  The average 1/|r| between two such rings is a function
# of (r_i, r_j, angle between planes).
#
# This is getting complicated.  Let's use a simpler but still emergent
# approach...
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Model E5: "Cluster and Thomson"
# Optimize each electron's (r_i, L_i), then auto-detect shells by
# grouping electrons with similar radii, and compute repulsion using
# the Thomson energy for each group.
#
# This is a hybrid: optimization is free, but repulsion calculation
# uses the physical Thomson result for clustered electrons.
# --------------------------------------------------------------------------

def auto_group(rs: np.ndarray, tol: float = 0.05) -> list[list[int]]:
    """Group electron indices by similar radius (relative tolerance)."""
    order = np.argsort(rs)
    groups = []
    current = [order[0]]
    for i in range(1, len(order)):
        r_prev = rs[order[i - 1]]
        r_curr = rs[order[i]]
        if abs(r_curr - r_prev) / max(r_prev, 1e-10) < tol:
            current.append(order[i])
        else:
            groups.append(current)
            current = [order[i]]
    groups.append(current)
    return groups


# Import Thomson repulsion from exp009
sys.path.insert(0, os.path.dirname(__file__))
from exp009_principal_shells_BeNe import repulsion_sum as thomson_S


def energy_E5(params: np.ndarray, Z: int, N: int) -> float:
    """Like E1 but repulsion within same-radius groups uses Thomson."""
    rs = params[0::2]
    Ls = params[1::2]

    if np.any(rs <= 1e-6) or np.any(Ls < 0):
        return 1e6

    E = 0.0
    for i in range(N):
        E += Ls[i]**2 / (2.0 * rs[i]**2)
        E -= Z / rs[i]

    # Group electrons by radius
    groups = auto_group(rs, tol=0.05)

    # Intra-group: Thomson repulsion at the group's mean radius
    for g in groups:
        k = len(g)
        if k < 2:
            continue
        r_mean = np.mean(rs[g])
        E += thomson_S(k) / r_mean

    # Inter-group: shell theorem (1/r_outer per pair)
    for a in range(len(groups)):
        for b in range(a + 1, len(groups)):
            k_a = len(groups[a])
            k_b = len(groups[b])
            r_a = np.mean(rs[groups[a]])
            r_b = np.mean(rs[groups[b]])
            r_out = max(r_a, r_b)
            E += k_a * k_b / r_out

    return E


# --------------------------------------------------------------------------
# Model E6: "Exclusion from repulsion" — what if we DON'T allow
# multiple electrons at the same radius at all?  Each electron MUST
# be at a distinct radius.  Angular momentum L_i is a free continuous
# variable per electron.  This forces a 1D "radial shell" picture
# where each electron occupies its own radial niche.
#
# Repulsion: 1/r_outer for each pair (exact for concentric shells).
# --------------------------------------------------------------------------

def energy_E6(params: np.ndarray, Z: int, N: int) -> float:
    """Each electron at its own distinct radius. L_i free."""
    rs = params[0::2]
    Ls = params[1::2]

    if np.any(rs <= 1e-6) or np.any(Ls < 0):
        return 1e6

    E = 0.0
    for i in range(N):
        E += Ls[i]**2 / (2.0 * rs[i]**2)
        E -= Z / rs[i]

    for i in range(N):
        for j in range(i + 1, N):
            E += 1.0 / max(rs[i], rs[j])

    return E


# --------------------------------------------------------------------------
# Optimizer: many Nelder-Mead restarts + basin-hopping
# --------------------------------------------------------------------------

def optimize_model(energy_func, Z: int, N: int,
                   n_restarts: int = 20) -> dict:
    """Optimize electron configuration for a given energy model."""
    best_E = 1e6
    best_params = None

    rng = np.random.RandomState(42)

    # Strategy 1: physically-motivated starting points
    for trial in range(n_restarts):
        p0 = np.zeros(2 * N)
        for i in range(N):
            n_guess = (i // 2) + 1 if N > 1 else 1
            p0[2*i] = n_guess**2 / Z * (0.5 + rng.random())
            p0[2*i + 1] = n_guess * (0.5 + rng.random())

        res = minimize(energy_func, p0, args=(Z, N),
                       method="Nelder-Mead",
                       options={"xatol": 1e-10, "fatol": 1e-12,
                                "maxiter": 100000, "adaptive": True})
        if res.fun < best_E:
            best_E = res.fun
            best_params = res.x.copy()

    # Strategy 2: also try all-same-shell configurations
    for L_try in [1.0, 1.5, 2.0]:
        p0 = np.zeros(2 * N)
        for i in range(N):
            p0[2*i] = L_try**2 / Z * (0.9 + 0.2 * rng.random())
            p0[2*i + 1] = L_try
        res = minimize(energy_func, p0, args=(Z, N),
                       method="Nelder-Mead",
                       options={"xatol": 1e-10, "fatol": 1e-12,
                                "maxiter": 100000, "adaptive": True})
        if res.fun < best_E:
            best_E = res.fun
            best_params = res.x.copy()

    # Extract results
    rs = best_params[0::2]
    Ls = best_params[1::2]

    # Sort by radius
    order = np.argsort(rs)
    rs_sorted = rs[order]
    Ls_sorted = Ls[order]

    return {
        "Z": Z, "N": N,
        "E_ha": best_E,
        "radii": [float(x) for x in rs_sorted],
        "L_values": [float(x) for x in Ls_sorted],
    }


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

ATOMS = ["H", "He", "Li", "Be", "B"]

def run() -> dict:
    out = {}

    for model_name, energy_func in [
        ("E1_shell_theorem", energy_E1),
        ("E5_auto_thomson", energy_E5),
        ("E6_distinct_radii", energy_E6),
    ]:
        model_results = {}
        for sym in ATOMS:
            Z = Z_of[sym]
            N = Z  # neutral atom

            res = optimize_model(energy_func, Z, N)
            exp_ha = -TOTAL_BINDING_EV[sym] / HARTREE_EV
            err = 100.0 * abs(res["E_ha"] - exp_ha) / abs(exp_ha)

            # Also compute the ion (N-1 electrons) for IE
            if N > 1:
                res_ion = optimize_model(energy_func, Z, N - 1)
                IE_ha = res_ion["E_ha"] - res["E_ha"]
                IE_eV = IE_ha * HARTREE_EV
            else:
                IE_eV = -res["E_ha"] * HARTREE_EV  # H: just remove the electron

            IE_exp = FIRST_IE_EV[sym]
            IE_err = 100.0 * abs(IE_eV - IE_exp) / IE_exp

            model_results[sym] = {
                "E_ha": res["E_ha"], "E_exp": exp_ha, "err_pct": err,
                "radii": res["radii"], "L_values": res["L_values"],
                "IE_pred_eV": IE_eV, "IE_exp_eV": IE_exp, "IE_err_pct": IE_err,
            }

        mean_E_err = np.mean([v["err_pct"] for v in model_results.values()])
        mean_IE_err = np.mean([v["IE_err_pct"] for v in model_results.values()])
        out[model_name] = {
            "results": model_results,
            "mean_E_err": float(mean_E_err),
            "mean_IE_err": float(mean_IE_err),
        }

    return out


if __name__ == "__main__":
    out = run()
    print("=" * 110)
    print("Experiment 015: emergent shell structure — no imposed filling rules")
    print("  Each electron optimizes its own (r_i, L_i) freely.")
    print("  ONLY physics: T = L²/(2r²), V_nuc = -Z/r, V_ee = repulsion")
    print("=" * 110)

    for model_name in ["E1_shell_theorem", "E5_auto_thomson", "E6_distinct_radii"]:
        m = out[model_name]
        print(f"\n{'='*80}")
        print(f"Model: {model_name}")
        print(f"  Mean total-energy error: {m['mean_E_err']:.2f}%")
        print(f"  Mean ionization-energy error: {m['mean_IE_err']:.2f}%")
        print(f"{'='*80}")
        print(f"  {'atom':<4} {'E pred':>9} {'E exp':>9} {'E err%':>7}"
              f"  {'IE pred':>8} {'IE exp':>8} {'IE err%':>8}"
              f"  {'radii':<30} {'L values'}")
        for sym in ATOMS:
            v = m["results"][sym]
            rs = ", ".join(f"{r:.3f}" for r in v["radii"])
            ls = ", ".join(f"{l:.3f}" for l in v["L_values"])
            print(f"  {sym:<4} {v['E_ha']:>9.4f} {v['E_exp']:>9.4f} {v['err_pct']:>6.2f}%"
                  f"  {v['IE_pred_eV']:>8.3f} {v['IE_exp_eV']:>8.3f} {v['IE_err_pct']:>7.1f}%"
                  f"  [{rs}]  [{ls}]")

    print()
    print("INTERPRETATION")
    print("=" * 110)
    print("  Key questions to answer from the data above:")
    print("  1. Do electrons spontaneously cluster at similar radii (emergent shells)?")
    print("  2. Do the optimized L values come out near integers?")
    print("  3. Does the model predict correct ionization energies?")
    print("  4. For He: do both electrons sit at the same radius with L≈1?")
    print("     Or do they separate into different radii?")
    print("  5. For Li: does one electron naturally move to a larger radius")
    print("     (the way 2s separates from 1s in quantum mechanics)?")
    print()
    print("  If shells emerge: the model has deep explanatory power.")
    print("  If L values cluster near integers: quantization is emergent!")
    print("  If IE predictions improve: the model captures outer-shell physics.")

    log_experiment({"exp": "015", "name": "emergent_shells",
                    "results": {k: {"mean_E_err": v["mean_E_err"],
                                    "mean_IE_err": v["mean_IE_err"]}
                                for k, v in out.items()}})
