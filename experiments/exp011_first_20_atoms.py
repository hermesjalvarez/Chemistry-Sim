"""Experiment 011 — First 20 atoms with competing filling rules.

The model from exp009 worked beautifully for H-Ne (row 1-2).  Now we push
to Z = 11..20 (Na through Ca), which crosses two critical boundaries:

  * Z = 11..18 (Na..Ar): third row, filling n = 3.
  * Z = 19..20 (K, Ca): START of fourth row.  In reality these atoms put
    electrons into n = 4 BEFORE n = 3 is full (Aufbau anomaly).

We test THREE filling rules:

  RULE A ("2n^2"):  shell n holds at most 2 n^2 electrons.
      n=1: 2, n=2: 8, n=3: 18, n=4: 32.
      K would be [2, 8, 9],  Ca [2, 8, 10].

  RULE B ("octet"):  every shell holds at most 8 electrons (except n=1: 2).
      This is the "noble gas octet" rule.
      K would be [2, 8, 8, 1],  Ca [2, 8, 8, 2].

  RULE C ("Madelung"): fill in order of n + l, using subshell capacities
      2(2l+1).  In our model we don't have explicit l, so we approximate
      this as: n=1 holds 2, n=2 holds 8, n=3 holds 8 (3s+3p only),
      then n=4 starts (4s before 3d).  3d fills later.
      For Z <= 20 this is IDENTICAL to Rule B.

  RULE D ("energy decides"): try BOTH [2,8,k] and [2,8,k-1,1] for each
      atom from Z=11 onward, and pick whichever has LOWER total energy.
      This lets the model itself decide the filling order with no imposed
      rule.

The physics is otherwise identical to exp009: kinetic barrier T_n = n^2/(2R^2),
Thomson polyhedra, Gauss shell-shell repulsion.
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

sys.path.insert(0, os.path.dirname(__file__))
from exp009_principal_shells_BeNe import (
    repulsion_sum, thomson_points, _unit
)


# --------------------------------------------------------------------------
# Energy calculator (same as exp009 but accepts arbitrary occupancy)
# --------------------------------------------------------------------------

def energy_of(Z: int, occupancy: list[int]) -> dict:
    K = len(occupancy)
    S_k = [repulsion_sum(k) for k in occupancy]

    def E_of_radii(r_vec: np.ndarray) -> float:
        if np.any(r_vec <= 1e-6):
            return 1e6
        E = 0.0
        for n_idx, (k, r) in enumerate(zip(occupancy, r_vec)):
            n = n_idx + 1
            E += k * (n * n) / (2.0 * r * r)
            E -= Z * k / r
            E += S_k[n_idx] / r
        for i in range(K):
            for j in range(i + 1, K):
                E += occupancy[i] * occupancy[j] / max(r_vec[i], r_vec[j])
        return E

    R0 = np.array([(i + 1) ** 2 / max(Z - sum(occupancy[:i]), 1)
                    for i in range(K)], dtype=float)
    R0 = np.clip(R0, 0.01, 100.0)
    res = minimize(E_of_radii, R0, method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 60000})
    return {
        "Z": Z, "occupancy": occupancy,
        "radii": [float(x) for x in res.x],
        "E_ha": float(res.fun),
    }


# --------------------------------------------------------------------------
# Filling rules
# --------------------------------------------------------------------------

def fill_2n2(N: int) -> list[int]:
    """Rule A: shell n holds up to 2n^2."""
    occ = []
    n = 1
    left = N
    while left > 0:
        cap = 2 * n * n
        take = min(cap, left)
        occ.append(take)
        left -= take
        n += 1
    return occ


def fill_octet(N: int) -> list[int]:
    """Rule B: shell n=1 holds 2; all higher shells hold up to 8."""
    occ = []
    n = 1
    left = N
    while left > 0:
        cap = 2 if n == 1 else 8
        take = min(cap, left)
        occ.append(take)
        left -= take
        n += 1
    return occ


def fill_energy_decides(Z: int) -> tuple[list[int], float]:
    """Rule D: try multiple fillings and pick the lowest energy."""
    N = Z
    candidates = []

    # Always try 2n^2 and octet
    candidates.append(fill_2n2(N))
    candidates.append(fill_octet(N))

    # For Z > 10, also try splitting the outermost shell:
    # e.g. for Z=19, try [2,8,9], [2,8,8,1], [2,8,7,2], [2,8,6,3]
    if N > 10:
        base = [2, 8]
        remaining = N - 10
        for split in range(remaining + 1):
            n3 = remaining - split
            if n3 < 0:
                continue
            if split == 0:
                if n3 > 0:
                    candidates.append(base + [n3])
            else:
                if n3 > 0:
                    candidates.append(base + [n3, split])
                else:
                    candidates.append(base + [split])

    # For Z > 18, also try [2,8,8,k]
    if N > 18:
        k = N - 18
        candidates.append([2, 8, 8, k])

    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)

    best_occ = None
    best_E = 1e6
    for occ in unique:
        r = energy_of(Z, occ)
        if r["E_ha"] < best_E:
            best_E = r["E_ha"]
            best_occ = occ
    return best_occ, best_E


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

def run() -> dict:
    out = {}
    for sym in ELEMENTS:
        Z = Z_of[sym]
        N = Z
        exp_ha = -TOTAL_BINDING_EV[sym] / HARTREE_EV

        # Rule A
        occ_a = fill_2n2(N)
        res_a = energy_of(Z, occ_a)

        # Rule B
        occ_b = fill_octet(N)
        res_b = energy_of(Z, occ_b)

        # Rule D (energy decides)
        occ_d, E_d = fill_energy_decides(Z)
        res_d = energy_of(Z, occ_d)

        out[sym] = {
            "Z": Z, "E_exp_ha": exp_ha,
            "A_occ": occ_a, "A_E": res_a["E_ha"], "A_r": res_a["radii"],
            "A_err": 100.0 * abs(res_a["E_ha"] - exp_ha) / abs(exp_ha),
            "B_occ": occ_b, "B_E": res_b["E_ha"], "B_r": res_b["radii"],
            "B_err": 100.0 * abs(res_b["E_ha"] - exp_ha) / abs(exp_ha),
            "D_occ": occ_d, "D_E": res_d["E_ha"], "D_r": res_d["radii"],
            "D_err": 100.0 * abs(res_d["E_ha"] - exp_ha) / abs(exp_ha),
        }
    return out


if __name__ == "__main__":
    out = run()
    print("=" * 110)
    print("Experiment 011: first 20 atoms — three filling rules compared")
    print("=" * 110)
    print(f"{'':4} {'':3}  {'--- Rule A (2n^2) ---':^30}  {'--- Rule B (octet) ---':^30}  {'--- Rule D (energy) ---':^30}")
    print(f"{'atom':<4} {'Z':>3}  {'occ':>12} {'E(Ha)':>9} {'err%':>6}  {'occ':>12} {'E(Ha)':>9} {'err%':>6}  {'occ':>12} {'E(Ha)':>9} {'err%':>6}  {'E_exp':>9}")
    for sym, r in out.items():
        def fmt_occ(o):
            return str(o)
        star_a = "*" if r["A_err"] == min(r["A_err"], r["B_err"], r["D_err"]) else " "
        star_b = "*" if r["B_err"] == min(r["A_err"], r["B_err"], r["D_err"]) else " "
        star_d = "*" if r["D_err"] == min(r["A_err"], r["B_err"], r["D_err"]) else " "
        print(f"{sym:<4} {r['Z']:>3}  {fmt_occ(r['A_occ']):>12} {r['A_E']:>9.3f} {r['A_err']:>5.1f}%{star_a}"
              f" {fmt_occ(r['B_occ']):>12} {r['B_E']:>9.3f} {r['B_err']:>5.1f}%{star_b}"
              f" {fmt_occ(r['D_occ']):>12} {r['D_E']:>9.3f} {r['D_err']:>5.1f}%{star_d}"
              f" {r['E_exp_ha']:>9.3f}")

    print()
    print("* = best rule for that atom")
    print()
    print("Radii for Rule D (energy-chosen filling):")
    for sym, r in out.items():
        rs = ", ".join(f"{x:.3f}" for x in r["D_r"])
        print(f"  {sym:<4} Z={r['Z']:>2}  occ={str(r['D_occ']):<16}  radii=[{rs}]")
    print()
    print("KEY OBSERVATIONS")
    print("-" * 110)
    print("  1. For Z=1..10 (rows 1-2), all three rules are identical: [2] or [2, k].")
    print("  2. For Z=11..18 (row 3), Rules A and B may differ. Check which fits better.")
    print("  3. For Z=19..20 (K, Ca), Rule A says [2,8,9] / [2,8,10] (all in n=3).")
    print("     Rule B says [2,8,8,1] / [2,8,8,2] (start n=4 at 8 in n=3).")
    print("     Rule D lets energy decide. Does the model DISCOVER the Aufbau anomaly?")

    log_experiment({"exp": "011", "name": "first_20_atoms", "results": out})
