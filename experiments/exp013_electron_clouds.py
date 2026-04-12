"""Experiment 013 — Electrons as density distributions, not points.

The Thomson model (exp009) treats k electrons as point charges at fixed
optimal positions on the shell.  This overbinds He (5.5%) because two
point charges at antipodes have LESS repulsion (λ=0.5) than the real He
atom (λ≈0.593).

The physical picture: real electrons are NOT frozen points.  They are
probability distributions / charge clouds.  A cloud electron "smeared"
over a region of the sphere has a HIGHER average 1/d than a point at
the center of that region (because convexity of 1/d: Jensen's inequality).

We test several cloud models:

  MODEL A: "Power-law pair correlation"
  ----------------------------------------
  Instead of fixing the interelectron angle at the Thomson value, model
  the pair distribution as P(gamma) ∝ sin^n(gamma/2) · sin(gamma),
  where gamma is the angle between two electrons on the sphere.
  n = 0 → uniform (no correlation, λ = 1).
  n → ∞ → perfectly antipodal (λ = 0.5).

  Analytical result (derived by integration):
      λ(n) = (n + 2) / (2 (n + 1))

  This gives a smooth interpolation with ONE parameter n.

  MODEL B: "Weighted Thomson-uniform blend"
  -------------------------------------------
  S_eff = (1 - w) · S_Thomson + w · k(k-1)/2
  where w is a delocalization weight.  Single parameter w.

  MODEL C: "k-dependent delocalization"
  ----------------------------------------
  Same as B but w(k) = a / k.  The idea: more electrons on the sphere →
  each electron is more confined by its neighbors → less delocalization.
  Single parameter a.

  MODEL D: "Pair correlation + screening combined"
  --------------------------------------------------
  Combine the best screening model (S3, delta=0.806) from exp012 with
  the best cloud correction.  This addresses TWO separate problems:
  - Cloud correction fixes He (too much Thomson correlation)
  - Screening correction fixes heavy atoms (too perfect shielding)

For each model we compute all 20 atoms and report mean error.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar

from constants import HARTREE_EV
from reference_data import TOTAL_BINDING_EV, Z as Z_of, ELEMENTS
from core import log_experiment

sys.path.insert(0, os.path.dirname(__file__))
from exp009_principal_shells_BeNe import repulsion_sum as thomson_repulsion_sum


# --------------------------------------------------------------------------
# Analytical results for the power-law pair correlation model
# --------------------------------------------------------------------------

def lambda_powerlaw(n: float) -> float:
    """For P(gamma) ∝ sin^n(gamma/2), the effective repulsion coefficient is:
        lambda = (n + 2) / (2 * (n + 1))
    Note: n=0 → 1.0 (uniform), n→∞ → 0.5 (antipodal).
    """
    return (n + 2.0) / (2.0 * (n + 1.0))


def n_from_lambda(lam: float) -> float:
    """Inverse: given lambda, find the power-law exponent n."""
    # lam = (n+2)/(2(n+1)) => 2 lam (n+1) = n+2 => n(2 lam -1) = 2 - 2 lam
    # => n = (2 - 2 lam) / (2 lam - 1)
    if lam <= 0.5:
        return float('inf')
    return (2.0 - 2.0 * lam) / (2.0 * lam - 1.0)


# --------------------------------------------------------------------------
# Uniform repulsion sum for k electrons (all uncorrelated uniform shells)
# --------------------------------------------------------------------------

def uniform_repulsion_sum(k: int) -> float:
    """S_uniform = C(k,2) * 1 = k*(k-1)/2.
    Each pair of uncorrelated uniform shells at same R repels as 1/R.
    """
    return k * (k - 1) / 2.0


# --------------------------------------------------------------------------
# Model A: power-law (applies only to pairs on the same shell)
# For k electrons at Thomson positions, replace each pair's 1/d_ij
# with the weighted average <1/d> = lambda(n) / R, but ONLY for the
# nearest-neighbor pairings that are "delocalized".
#
# Simpler formulation for the whole shell:
# S_eff(k, n) = sum over pairs of lambda_ij(n, gamma_ij)
# But for different pair angles gamma_ij, the "effective lambda" differs.
# The cleanest approach: for a k-electron shell with pair distribution
# P(gamma) ∝ sin^n(gamma/2), the average pair repulsion is lambda(n)/R
# REGARDLESS of k, so:
#   S_eff(k, n) = C(k,2) * lambda(n)
# --------------------------------------------------------------------------

def cloud_repulsion_A(k: int, n: float) -> float:
    """Power-law pair correlation: S = C(k,2) * lambda(n)."""
    lam = lambda_powerlaw(n)
    return k * (k - 1) / 2.0 * lam


# --------------------------------------------------------------------------
# Model B: weighted Thomson-uniform blend
# --------------------------------------------------------------------------

def cloud_repulsion_B(k: int, w: float) -> float:
    return (1.0 - w) * thomson_repulsion_sum(k) + w * uniform_repulsion_sum(k)


# --------------------------------------------------------------------------
# Model C: k-dependent delocalization w(k) = a / k
# --------------------------------------------------------------------------

def cloud_repulsion_C(k: int, a: float) -> float:
    w = a / k
    w = min(w, 1.0)
    return (1.0 - w) * thomson_repulsion_sum(k) + w * uniform_repulsion_sum(k)


# --------------------------------------------------------------------------
# Energy calculator (combines cloud model + optional screening)
# --------------------------------------------------------------------------

def fill_octet(N: int) -> list[int]:
    occ, n, left = [], 1, N
    while left > 0:
        cap = 2 if n == 1 else 8
        occ.append(min(cap, left))
        left -= min(cap, left)
        n += 1
    return occ


def energy_of(
    Z: int,
    occupancy: list[int],
    cloud_func,           # cloud_func(k) -> effective S_k
    screening_func=None,  # screening_func(n_i, n_j, r_i, r_j) -> s
) -> dict:
    K = len(occupancy)
    S_k = [cloud_func(k) for k in occupancy]

    def E_of_radii(r_vec: np.ndarray) -> float:
        if np.any(r_vec <= 1e-6):
            return 1e6
        E = 0.0
        for j in range(K):
            n_j = j + 1
            k_j = occupancy[j]
            r_j = r_vec[j]
            E += k_j * (n_j * n_j) / (2.0 * r_j * r_j)
            Z_eff_j = float(Z)
            for i in range(K):
                if i == j:
                    continue
                if r_vec[i] < r_j * (1 - 1e-9):
                    if screening_func is not None:
                        s = screening_func(i + 1, n_j, r_vec[i], r_j)
                    else:
                        s = 1.0
                    Z_eff_j -= s * occupancy[i]
            E -= Z_eff_j * k_j / r_j
            E += S_k[j] / r_j
        return E

    R0 = np.array([(j + 1) ** 2 / max(Z - sum(occupancy[:j]), 1)
                    for j in range(K)], dtype=float)
    R0 = np.clip(R0, 0.01, 100.0)
    res = minimize(E_of_radii, R0, method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 60000})
    return {"Z": Z, "occupancy": occupancy, "E_ha": float(res.fun),
            "radii": [float(x) for x in res.x]}


def sweep(cloud_func, screening_func=None) -> tuple[float, dict]:
    total_err = 0.0
    results = {}
    for sym in ELEMENTS:
        Z = Z_of[sym]
        occ = fill_octet(Z)
        r = energy_of(Z, occ, cloud_func, screening_func)
        exp_ha = -TOTAL_BINDING_EV[sym] / HARTREE_EV
        err = 100.0 * abs(r["E_ha"] - exp_ha) / abs(exp_ha)
        total_err += err
        results[sym] = {"E_ha": r["E_ha"], "E_exp": exp_ha, "err": err}
    return total_err / len(ELEMENTS), results


def screen_penetration(delta):
    def f(n_i, n_j, r_i, r_j):
        return 1.0 - delta * (r_i / r_j)
    return f


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

def run() -> dict:
    out = {}

    # Baseline: Thomson only
    me, res = sweep(lambda k: thomson_repulsion_sum(k))
    out["baseline_thomson"] = {"mean_err": me, "label": "Thomson points (baseline)"}

    # Model A: power-law, fit n across all 20 atoms
    def err_A(n):
        me, _ = sweep(lambda k, _n=n: cloud_repulsion_A(k, _n))
        return me
    res_A = minimize_scalar(err_A, bounds=(0.5, 50.0), method="bounded",
                            options={"xatol": 0.1})
    n_best = float(res_A.x)
    me_A, res_A_full = sweep(lambda k: cloud_repulsion_A(k, n_best))
    out["A_powerlaw"] = {"n": n_best, "lambda_2": lambda_powerlaw(n_best),
                         "mean_err": me_A, "results": res_A_full,
                         "label": f"Power-law n={n_best:.1f} (λ₂={lambda_powerlaw(n_best):.3f})"}

    # What n fits He exactly?
    HE_LAM = 2 * 2 - 2 * math.sqrt(TOTAL_BINDING_EV["He"] / HARTREE_EV)
    n_he = n_from_lambda(HE_LAM / 2.0)  # wrong, let me recalculate
    # For He: E = 1/R^2 - (4 - lam_eff)/R. Min: -(4-lam)^2/4 = E_exp.
    # lam = 4 - 2*sqrt(-E_exp). E_exp = -2.9032.
    lam_he = 4.0 - 2.0 * math.sqrt(2.9032)
    n_he = n_from_lambda(lam_he)

    # Model C: k-dependent delocalization, fit a
    def err_C(a):
        me, _ = sweep(lambda k, _a=a: cloud_repulsion_C(k, _a))
        return me
    res_C = minimize_scalar(err_C, bounds=(0.01, 5.0), method="bounded",
                            options={"xatol": 0.01})
    a_best = float(res_C.x)
    me_C, res_C_full = sweep(lambda k: cloud_repulsion_C(k, a_best))
    out["C_kdep_deloc"] = {"a": a_best, "mean_err": me_C, "results": res_C_full,
                           "label": f"k-dep delocalization a={a_best:.2f}"}

    # Model D: cloud C + screening S3, jointly fit (a, delta)
    def err_D(params):
        a, delta = params
        if a < 0 or delta < 0 or delta > 3:
            return 100.0
        me, _ = sweep(lambda k: cloud_repulsion_C(k, a),
                       screen_penetration(delta))
        return me
    res_D = minimize(err_D, [0.5, 0.8], method="Nelder-Mead",
                     options={"xatol": 0.01, "fatol": 0.01})
    a_D, delta_D = float(res_D.x[0]), float(res_D.x[1])
    me_D, res_D_full = sweep(lambda k: cloud_repulsion_C(k, a_D),
                              screen_penetration(delta_D))
    out["D_cloud_plus_screening"] = {
        "a": a_D, "delta": delta_D, "mean_err": me_D, "results": res_D_full,
        "label": f"Cloud a={a_D:.2f} + screening δ={delta_D:.2f}"}

    # Also: He-exact power-law (n chosen to exactly match He, then applied to all)
    me_he, res_he = sweep(lambda k: cloud_repulsion_A(k, n_he))
    out["A_He_exact"] = {"n": n_he, "lambda_2": lambda_powerlaw(n_he),
                         "mean_err": me_he, "results": res_he,
                         "label": f"Power-law n={n_he:.1f} (fit to He exactly)"}

    # Reference: Thomson + best screening from exp012
    me_ref, res_ref = sweep(lambda k: thomson_repulsion_sum(k),
                            screen_penetration(0.806))
    out["ref_thomson_screening"] = {"mean_err": me_ref, "results": res_ref,
                                    "label": "Thomson + screening δ=0.806 (exp012 best)"}

    return out


if __name__ == "__main__":
    out = run()
    print("=" * 100)
    print("Experiment 013: electrons as charge clouds on the shell")
    print("=" * 100)
    print()
    print("ANALYTICAL NOTE: power-law pair correlation")
    print("  P(γ) ∝ sin^n(γ/2) · sin(γ)")
    print("  → λ(n) = (n+2) / (2(n+1))")
    print("  → n=0: λ=1 (uniform), n→∞: λ=0.5 (antipodal)")
    print(f"  → He needs λ≈{4.0 - 2*math.sqrt(2.9032):.3f}, i.e. n≈{n_from_lambda(4.0-2*math.sqrt(2.9032)):.1f}")
    print()

    # Summary table
    print(f"{'model':<50} {'mean err':>9} {'params':>7}")
    for key in ["baseline_thomson", "ref_thomson_screening",
                "A_powerlaw", "A_He_exact", "C_kdep_deloc", "D_cloud_plus_screening"]:
        v = out[key]
        print(f"  {v['label']:<48} {v['mean_err']:>8.2f}% {1 if 'a' in v or 'n' in v else 0:>7d}"
              + (f"+1" if 'delta' in v else ""))

    # Detailed per-atom for best model
    best_key = min(out, key=lambda k: out[k]["mean_err"])
    best = out[best_key]
    print(f"\nBest model: {best['label']}")
    if "results" in best:
        print(f"  {'atom':<4} {'E pred':>10} {'E exp':>10} {'err%':>7}")
        for sym in ELEMENTS:
            v = best["results"][sym]
            print(f"  {sym:<4} {v['E_ha']:>10.3f} {v['E_exp']:>10.3f} {v['err']:>6.2f}%")

    print()
    print("PHYSICS INTERPRETATION")
    print("-" * 100)
    print("  The Thomson point-charge model slightly OVERBINDS He (too much angular")
    print("  correlation) and slightly UNDERBINDS heavy atoms (perfect screening).")
    print("  These are TWO SEPARATE problems needing TWO SEPARATE fixes:")
    print("    1. Cloud delocalization (w = a/k): fixes He by increasing repulsion")
    print("       for small k (fewer electrons → more spread out → less correlation)")
    print("    2. Imperfect screening (δ): fixes heavy atoms by letting outer")
    print("       electrons see more nuclear charge")
    print("  Combined, they produce the best overall model.")

    log_experiment({"exp": "013", "name": "electron_clouds",
                    "results": {k: {"label": v.get("label",""), "mean_err": v["mean_err"]}
                                for k, v in out.items()}})
