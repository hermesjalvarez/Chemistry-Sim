"""Experiment 009 - Principal-quantum shells through the first row.

Lesson from exp008: we need *some* rule that prevents many electrons
from collecting into a single innermost shell.  We now test a specific
classical-ish postulate that generates the periodic table's row
structure without explicit quantum mechanics:

  POSTULATE (S): Each electron carries an angular-momentum-like
                 integer label n = 1, 2, 3, ...  Shells are filled
                 in order of increasing n.  An electron in shell n has
                 kinetic-barrier energy T_n(R) = n^2 / (2 R^2).
                 (This is equivalent to "L = n" in the rigid-ring model.)

  POSTULATE (P): Each n-shell holds at most 2 n^2 electrons (so n=1
                 holds 2, n=2 holds 8, etc.)

  POSTULATE (C): Inside a given n-shell all electrons sit at a common
                 radius r_n, distributed as the regular / optimal
                 (Thomson) polyhedron with k vertices where k is the
                 current occupancy of that shell.

Then:

  * The first row of the periodic table (H..He) lives on n=1: 1 or 2 electrons.
  * The second row (Li..Ne) adds 1..8 electrons on n=2.
  * The electron-electron repulsion inside a shell is computed exactly
    from the Thomson polyhedron (fully deterministic geometry, NO free
    parameter).
  * Between shells (n_i < n_j), Gauss's law gives 1/r_j per pair.

We compute Be..Ne under this postulate and compare to experiment.
The goal is to see how far we can push a purely geometric, no-fit
theory.

Thomson polyhedra used
----------------------
k=1: single point
k=2: antipodal   (d = 2)
k=3: equilateral triangle on equator   (d = sqrt(3))
k=4: regular tetrahedron                 (d = sqrt(8/3))
k=5: triangular bipyramid
k=6: octahedron                           (d_edge = sqrt(2), d_diag = 2)
k=7: pentagonal bipyramid + apex (approx)
k=8: square antiprism
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
from scipy.optimize import minimize

from core import log_experiment
from constants import HARTREE_EV
from reference_data import TOTAL_BINDING_EV, Z as Z_of, ELEMENTS

# --------------------------------------------------------------------------
# Thomson polyhedra on a unit sphere
# --------------------------------------------------------------------------
# Each array is shape (k, 3).  We then compute sum_{i<j} 1/|p_i-p_j|.
# For k up to 4 there are closed-form regular polyhedra; for larger k
# we use known Thomson minima.
# --------------------------------------------------------------------------

def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)

def thomson_points(k: int) -> np.ndarray:
    if k == 1:
        return np.array([[0.0, 0.0, 1.0]])
    if k == 2:
        return np.array([[0, 0, 1], [0, 0, -1]], dtype=float)
    if k == 3:
        return np.array([
            [1, 0, 0],
            [-0.5, math.sqrt(3)/2, 0],
            [-0.5, -math.sqrt(3)/2, 0],
        ])
    if k == 4:
        # Regular tetrahedron
        pts = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                       dtype=float)
        return pts / math.sqrt(3)
    if k == 5:
        # Triangular bipyramid: two poles + equilateral triangle on equator
        return np.vstack([
            [[0, 0, 1], [0, 0, -1]],
            [[math.cos(a), math.sin(a), 0] for a in
             np.linspace(0, 2*math.pi, 4)[:-1]]
        ])
    if k == 6:
        # Regular octahedron
        return np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
        ], dtype=float)
    if k == 7:
        # Pentagonal bipyramid: two poles + pentagon at equator
        top = [[0, 0, 1], [0, 0, -1]]
        pent = [[math.cos(a), math.sin(a), 0] for a in
                np.linspace(0, 2*math.pi, 6)[:-1]]
        return np.array(top + pent, dtype=float)
    if k == 8:
        # Square antiprism: two squares rotated 45 deg, z = +/- 0.707
        z = 1.0 / math.sqrt(2.0)
        top = [[math.cos(a), math.sin(a), z] for a in
               [0, math.pi/2, math.pi, 3*math.pi/2]]
        bot = [[math.cos(a + math.pi/4), math.sin(a + math.pi/4), -z]
               for a in [0, math.pi/2, math.pi, 3*math.pi/2]]
        raw = np.array(top + bot, dtype=float)
        return raw / np.linalg.norm(raw[0])
    raise ValueError(f"k={k} not implemented")


def repulsion_sum(k: int) -> float:
    """S_k = sum_{i<j} 1 / |p_i - p_j|  for k unit-sphere Thomson points."""
    if k <= 1:
        return 0.0
    pts = thomson_points(k)
    pts = np.array([_unit(p) for p in pts])
    total = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            total += 1.0 / np.linalg.norm(pts[i] - pts[j])
    return float(total)


# --------------------------------------------------------------------------
# Shell-model energy under postulates (S, P, C)
# --------------------------------------------------------------------------

def shell_capacity(n: int) -> int:
    return 2 * n * n   # n=1:2, n=2:8, n=3:18, ...


def partition_electrons(N: int) -> list[int]:
    """Fill shells n=1,2,3,... in order, at most 2 n^2 electrons each."""
    occ = []
    n = 1
    remaining = N
    while remaining > 0:
        cap = shell_capacity(n)
        take = min(cap, remaining)
        occ.append(take)
        remaining -= take
        n += 1
    return occ


def energy_of(Z: int, occupancy: list[int]) -> dict:
    """Compute total energy given the occupancy pattern, under (S,P,C).

    Variables: one radius r_n per occupied shell.
    By shell theorem, inner shells decouple radially from outer shells'
    positions, but the shell-shell Coulomb energy q1 q2 / r_outer still
    depends on r_outer.  We therefore minimize over ALL radii jointly.
    """
    K = len(occupancy)

    # precompute repulsion sums
    S_k = [repulsion_sum(k) for k in occupancy]

    def E_of_radii(r_vec: np.ndarray) -> float:
        if np.any(r_vec <= 1e-6):
            return 1e6
        E = 0.0
        # Per-shell kinetic and nuclear attraction
        for n_idx, (k, r) in enumerate(zip(occupancy, r_vec)):
            n = n_idx + 1
            # Kinetic: each electron carries L=n -> T = n^2 / (2 r^2).
            E += k * (n * n) / (2.0 * r * r)
            # Nuclear attraction: -Z*k / r
            E -= Z * k / r
            # Same-shell repulsion: sum_{i<j} 1/|p_i-p_j| / r
            E += S_k[n_idx] / r
        # Inter-shell repulsion (Gauss): q_inner*q_outer / r_outer
        for i in range(K):
            for j in range(i + 1, K):
                k_i, k_j = occupancy[i], occupancy[j]
                r_i, r_j = r_vec[i], r_vec[j]
                r_out = max(r_i, r_j)
                E += k_i * k_j / r_out
        return E

    # Reasonable starting radii: ~ n^2 / (Z - 2*(n-1))
    R0 = np.array([(i + 1) ** 2 / max(Z - 2 * i, 1) for i in range(K)],
                  dtype=float)
    res = minimize(E_of_radii, R0, method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-12,
                            "maxiter": 40000})
    return {
        "Z": Z,
        "occupancy": occupancy,
        "radii": [float(x) for x in res.x],
        "S_k": S_k,
        "E_ha": float(res.fun),
    }


# --------------------------------------------------------------------------
# Run across the first row
# --------------------------------------------------------------------------

def run() -> dict:
    out = {}
    for sym in ELEMENTS:
        Z = Z_of[sym]
        N = Z
        occ = partition_electrons(N)
        r = energy_of(Z, occ)
        exp_ha = -TOTAL_BINDING_EV[sym] / HARTREE_EV
        r["E_exp_ha"] = exp_ha
        r["E_exp_eV"] = -TOTAL_BINDING_EV[sym]
        r["E_pred_eV"] = r["E_ha"] * HARTREE_EV
        r["err_pct"] = 100.0 * abs(r["E_ha"] - exp_ha) / abs(exp_ha)
        out[sym] = r
    return out


if __name__ == "__main__":
    out = run()
    print("=" * 82)
    print("Experiment 009: principal-quantum shells, no same-shell correction")
    print("=" * 82)
    print(f"{'atom':<4} {'Z':>3} {'occ':>10} {'E pred':>12} {'E exp':>12}"
          f" {'pred eV':>10} {'exp eV':>10} {'err%':>7}")
    for sym, r in out.items():
        print(f"{sym:<4} {r['Z']:>3} {str(r['occupancy']):>10}"
              f" {r['E_ha']:>12.4f} {r['E_exp_ha']:>12.4f}"
              f" {r['E_pred_eV']:>10.2f} {r['E_exp_eV']:>10.2f}"
              f" {r['err_pct']:>6.2f}%")
    print()
    print("Radii (Bohr):")
    for sym, r in out.items():
        rs = ", ".join(f"{x:.3f}" for x in r["radii"])
        print(f"  {sym:<4} occ={r['occupancy']}  S_k={[round(s,3) for s in r['S_k']]}  radii=[{rs}]")
    print()
    print("Physics notes:")
    print("  - Kinetic barrier T_n = n^2 / (2 R^2) per electron in shell n.")
    print("  - Same-shell repulsion = S_k / R where S_k is the Thomson sum")
    print("    of 1/|p_i - p_j| for k unit-sphere vertices.")
    print("  - Shell-shell repulsion via Gauss = q1 q2 / r_outer.")
    print("  - No free parameter!  One postulate: L_n = n.")

    log_experiment({"exp": "009", "name": "principal_shells_BeNe",
                    "results": out})
