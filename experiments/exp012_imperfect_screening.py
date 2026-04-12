"""Experiment 012 — Imperfect screening corrections.

Problem identified in exp011: the model systematically UNDERBINDS for
Z >= 10, with error growing to ~7% by Ca.  The root cause is that the
shell theorem gives PERFECT screening: an inner shell of k electrons
reduces the effective nuclear charge seen by an outer shell by exactly k.
But in reality, outer electrons "penetrate" inner shells and see more
nuclear charge than the model predicts.

We test several imperfect-screening hypotheses:

  (S0) Perfect screening (baseline from exp009/011).
       Z_eff_outer = Z - sum(inner electrons).

  (S1) Uniform fractional screening: each inner electron screens a
       fraction s < 1 of its charge.  One free parameter s, fit to
       minimize total error across all 20 atoms.

  (S2) Shell-dependent screening: electrons in shell n_i screen an
       outer shell n_j by a factor s(n_i, n_j) that depends on WHICH
       shells are involved.  Inspired by Slater's rules:
         - same shell (n_i = n_j):  s = 0.35
         - one shell below (n_i = n_j - 1):  s = 0.85
         - two or more shells below:  s = 1.00
       These are Slater's actual numbers.  We test them as-is (no fit).

  (S3) Single-parameter penetration: each inner electron at radius r_i
       screens an outer electron at r_j by a factor
           s = 1 - delta * (r_i / r_j)
       where delta is a free parameter.  This models the idea that
       screening gets worse when inner and outer shells are close in
       radius (more penetration).

  (S4) Fit a single universal screening fraction s to minimize error
       across Z=1..20, but ONLY for screening between DIFFERENT shells
       (same-shell repulsion stays at Thomson value).

Physics note: in the shell-theorem model, the "screening" enters as
the effective nuclear charge felt by each shell.  If shell j has
n_inner electrons interior to it, each screening by fraction s, then
Z_eff_j = Z - s * n_inner.  This increases Z_eff (more attraction)
and therefore increases binding.
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
from exp009_principal_shells_BeNe import repulsion_sum


# --------------------------------------------------------------------------
# Generalized energy function with screening model
# --------------------------------------------------------------------------

def energy_of(
    Z: int,
    occupancy: list[int],
    screening_func=None,
) -> dict:
    """Compute atom energy with a general screening model.

    screening_func(n_inner, n_outer, r_inner, r_outer) -> fraction of
    charge screened per inner electron.  If None, perfect screening (=1).
    """
    K = len(occupancy)
    S_k = [repulsion_sum(k) for k in occupancy]

    def E_of_radii(r_vec: np.ndarray) -> float:
        if np.any(r_vec <= 1e-6):
            return 1e6
        E = 0.0
        # Per-shell: kinetic + nuclear attraction + same-shell repulsion
        for n_idx, (k, r) in enumerate(zip(occupancy, r_vec)):
            n = n_idx + 1
            E += k * (n * n) / (2.0 * r * r)   # kinetic
            E -= Z * k / r                       # nuclear attraction
            E += S_k[n_idx] / r                  # same-shell Thomson repulsion

        # Inter-shell repulsion WITH screening correction
        for i in range(K):
            for j in range(i + 1, K):
                k_i, k_j = occupancy[i], occupancy[j]
                r_i, r_j = r_vec[i], r_vec[j]
                # Determine which is inner, which is outer
                if r_i <= r_j:
                    r_in, r_out = r_i, r_j
                    n_in, n_out = i + 1, j + 1
                else:
                    r_in, r_out = r_j, r_i
                    n_in, n_out = j + 1, i + 1

                # Full Coulomb repulsion between shells
                E += k_i * k_j / r_out

                # But inner shell also screens the nucleus for the outer.
                # With perfect screening, outer feels Z_eff = Z - k_inner.
                # We've already applied full Z above, so no adjustment needed
                # for perfect screening (it's implicit in the shell-shell term).
                # For IMPERFECT screening, we add back a fraction of the
                # screening that we're "removing":
                #   Each inner electron screens by factor s instead of 1,
                #   so the outer shell sees an EXTRA (1-s)*k_inner / r_out
                #   of nuclear attraction.
                # Wait - let me think about this more carefully.
                # In the baseline, shell j feels:
                #   V_nuc = -Z / r_j
                #   V_inner = +k_i / r_j  (repulsion from inner shell)
                #   Net = -(Z - k_i) / r_j  = -Z_eff / r_j with Z_eff = Z - k_i
                # With imperfect screening (s < 1), the inner shell only
                # "blocks" s * k_i of the nuclear charge:
                #   V_repulsion_effective = s * k_i / r_j
                #   Net = -(Z - s * k_i) / r_j
                # So we need to ADD (1-s) * k_i * k_j / r_out to the energy
                # (i.e., reduce the repulsion, which increases binding).
                # Actually no: the shell-shell Coulomb IS physical repulsion.
                # Screening is the CANCELLATION between nuclear attraction
                # and shell repulsion.  If screening is imperfect, the outer
                # shell sees MORE attraction.
                #
                # Cleanest formulation: the effective nuclear charge felt by
                # each electron in shell j due to the presence of shell i is:
                #   Z_eff contribution from i = -s * k_i
                # The actual Coulomb repulsion between i and j is still k_i*k_j/r_out.
                # So the net effect on shell j's k_j electrons is:
                #   k_j * (-Z + sum_i(k_i/r_out)) / r_j ... this is getting circular.
                #
                # Simpler: I'll rebuild the energy from scratch with Z_eff.
                pass

        return E

    # Actually, let me just build the energy directly with Z_eff per shell.
    def E_of_radii_v2(r_vec: np.ndarray) -> float:
        if np.any(r_vec <= 1e-6):
            return 1e6
        E = 0.0
        for j in range(K):
            n_j = j + 1
            k_j = occupancy[j]
            r_j = r_vec[j]

            # Kinetic
            E += k_j * (n_j * n_j) / (2.0 * r_j * r_j)

            # Nuclear attraction: each electron in shell j feels Z_eff_j
            # Z_eff_j = Z - sum over inner shells i of (s_ij * k_i)
            Z_eff_j = float(Z)
            for i in range(K):
                if i == j:
                    continue
                r_i = r_vec[i]
                if r_i < r_j * (1 - 1e-9):  # shell i is interior
                    n_i = i + 1
                    if screening_func is not None:
                        s = screening_func(n_i, n_j, r_i, r_j)
                    else:
                        s = 1.0
                    Z_eff_j -= s * occupancy[i]
            E -= Z_eff_j * k_j / r_j

            # Same-shell Thomson repulsion
            E += S_k[j] / r_j

        return E

    R0 = np.array([(j + 1) ** 2 / max(Z - sum(occupancy[:j]), 1)
                    for j in range(K)], dtype=float)
    R0 = np.clip(R0, 0.01, 100.0)
    res = minimize(E_of_radii_v2, R0, method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 60000})
    return {
        "Z": Z, "occupancy": occupancy,
        "radii": [float(x) for x in res.x],
        "E_ha": float(res.fun),
    }


# --------------------------------------------------------------------------
# Filling rule (use "energy decides" from exp011 for simplicity, but
# since we're changing the energy function, just use octet rule)
# --------------------------------------------------------------------------

def fill_octet(N: int) -> list[int]:
    occ = []
    n = 1
    left = N
    while left > 0:
        cap = 2 if n == 1 else 8
        occ.append(min(cap, left))
        left -= min(cap, left)
        n += 1
    return occ


# --------------------------------------------------------------------------
# Screening models
# --------------------------------------------------------------------------

def screen_perfect(n_i, n_j, r_i, r_j):
    return 1.0

def make_screen_uniform(s):
    def f(n_i, n_j, r_i, r_j):
        return s
    return f

def screen_slater(n_i, n_j, r_i, r_j):
    """Slater's empirical screening constants."""
    if n_i == n_j:
        return 0.35
    elif n_i == n_j - 1:
        return 0.85
    else:
        return 1.00

def make_screen_penetration(delta):
    def f(n_i, n_j, r_i, r_j):
        return 1.0 - delta * (r_i / r_j)
    return f


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------

def compute_errors(screening_func, label: str) -> dict:
    total_err = 0.0
    results = {}
    for sym in ELEMENTS:
        Z = Z_of[sym]
        occ = fill_octet(Z)
        r = energy_of(Z, occ, screening_func=screening_func)
        exp_ha = -TOTAL_BINDING_EV[sym] / HARTREE_EV
        err = 100.0 * abs(r["E_ha"] - exp_ha) / abs(exp_ha)
        total_err += err
        results[sym] = {
            "E_ha": r["E_ha"], "E_exp": exp_ha, "err": err,
            "radii": r["radii"], "occ": occ,
        }
    results["_mean_err"] = total_err / len(ELEMENTS)
    results["_label"] = label
    return results


def run() -> dict:
    out = {}

    # S0: perfect screening (baseline)
    out["S0_perfect"] = compute_errors(screen_perfect, "S0: perfect (s=1)")

    # S1: fit uniform s
    def mean_err_of_s(s):
        r = compute_errors(make_screen_uniform(s), f"s={s:.3f}")
        return r["_mean_err"]

    res = minimize_scalar(mean_err_of_s, bounds=(0.5, 1.0), method="bounded",
                          options={"xatol": 0.001})
    s_best = float(res.x)
    out["S1_uniform_fit"] = compute_errors(
        make_screen_uniform(s_best), f"S1: uniform s={s_best:.3f}")

    # S2: Slater's rules (no fit)
    out["S2_slater"] = compute_errors(screen_slater, "S2: Slater rules")

    # S3: penetration model, fit delta
    def mean_err_of_delta(delta):
        r = compute_errors(make_screen_penetration(delta), f"d={delta:.3f}")
        return r["_mean_err"]
    res = minimize_scalar(mean_err_of_delta, bounds=(0.0, 2.0), method="bounded",
                          options={"xatol": 0.001})
    delta_best = float(res.x)
    out["S3_penetration_fit"] = compute_errors(
        make_screen_penetration(delta_best),
        f"S3: penetration delta={delta_best:.3f}")

    return out


if __name__ == "__main__":
    out = run()
    print("=" * 110)
    print("Experiment 012: imperfect screening corrections")
    print("=" * 110)

    for model_key in ["S0_perfect", "S1_uniform_fit", "S2_slater", "S3_penetration_fit"]:
        r = out[model_key]
        print(f"\n{r['_label']}   (mean err = {r['_mean_err']:.2f}%)")
        print(f"  {'atom':<4} {'occ':>14} {'E pred':>10} {'E exp':>10} {'err%':>7}  {'radii'}")
        for sym in ELEMENTS:
            v = r[sym]
            rs = ", ".join(f"{x:.3f}" for x in v["radii"])
            print(f"  {sym:<4} {str(v['occ']):>14} {v['E_ha']:>10.3f} {v['E_exp']:>10.3f}"
                  f" {v['err']:>6.2f}%  [{rs}]")

    print()
    print("SUMMARY")
    print("-" * 110)
    for mk in ["S0_perfect", "S1_uniform_fit", "S2_slater", "S3_penetration_fit"]:
        r = out[mk]
        print(f"  {r['_label']:<45} mean err = {r['_mean_err']:.2f}%")
    print()
    print("INTERPRETATION")
    print("-" * 110)
    print("  - S0 (perfect screening) is our baseline: good for Z<=10, degrades to ~7% by Ca.")
    print("  - S1 (uniform fractional screening) asks: what single s minimizes error?")
    print("  - S2 (Slater rules) uses the historically known empirical screening constants.")
    print("    These were derived from spectroscopic data, not from our model, so this is a")
    print("    test of whether the shell model + Slater screening improves things.")
    print("  - S3 (penetration) models screening as worse when inner/outer radii are close.")

    log_experiment({"exp": "012", "name": "imperfect_screening",
                    "results": {k: {"label": v["_label"], "mean_err": v["_mean_err"]}
                                for k, v in out.items()}})
