"""Experiment 016 — Can angular momentum quantization emerge?

THE DEEPEST QUESTION:
Our model's one postulate is T = L²/(2R²).  We've always IMPOSED L = n
(integer).  But why should L be quantized?  Can we derive it?

APPROACH 1: "Orbit closure / resonance condition"
  A classical orbiting charge radiates.  The ONLY non-radiating orbits
  are standing waves: the electron's de Broglie wavelength must fit an
  integer number of times around its orbit.  For a circular orbit of
  radius R:
      2πR = n λ  →  2πR = n (2πℏ/p)  →  p R = nℏ  →  L = n

  But this smuggles in de Broglie!  Can we get the same result without it?

APPROACH 2: "Minimum-action postulate"
  Postulate that the electron's orbit has a minimum action per revolution:
      S = ∮ p · dq = L · 2π ≥ 2π α₀
  where α₀ is the smallest allowed action quantum.  If α₀ = 1 (in atomic
  units, i.e., ℏ), this gives L ≥ 1 and L = integer at each energy
  minimum.  This is Bohr-Sommerfeld quantization — still a postulate.

APPROACH 3: "Stability under perturbation"
  What if we don't impose L = integer, but instead ask: for a given atom,
  what continuous L values are STABLE equilibria?  If L is treated as a
  dynamical variable that can change slowly, stable points are where
  dE/dL = 0 with d²E/dL² > 0.

  For hydrogen:  E(R, L) = L²/(2R²) - Z/R
  Minimize over R:  R*(L) = L²/Z,  E*(L) = -Z²/(2L²)
  dE*/dL = Z²/L³ > 0  →  E* is monotonically increasing in L.
  So the MINIMUM energy is at L → 0, which is the collapse!

  This means continuous L doesn't give stable shells.  We NEED some
  quantization condition.  The model can't generate it internally.

APPROACH 4: "Exclusion from angular momentum"
  What if there's a constraint that no two electrons can share the same
  (n, orientation) state?  In our classical model, a ring-orbit electron
  has L_z as a continuous parameter.  But if we postulate that each
  orientation "slot" is exclusive, we get 2n+1 orientations for angular
  momentum n (m = -n..+n), times 2 for spin → 2(2n+1) electrons per n.

  n=0: 2 electrons  (but L=0 means collapse — need L≥1)
  n=1: 6 electrons
  n=2: 10 electrons

  This gives subshell structure (s, p, d) rather than principal shells!

  Actually, if we combine this with a principal quantum number framework:
  shell n allows L = 0, 1, ..., n-1 (this is the hydrogen constraint).
  Each L allows 2(2L+1) electrons.
  n=1: L=0 only → 2 electrons
  n=2: L=0,1 → 2+6 = 8
  n=3: L=0,1,2 → 2+6+10 = 18

  This IS the quantum mechanical result, but let's see if we can
  MOTIVATE it from classical stability rather than just imposing it.

APPROACH 5: "What the data tells us"
  Instead of deriving from theory, let's work BACKWARDS from experiment:
  - Optimize each electron's L freely (continuous)
  - See what L values the optimizer finds
  - Check if they cluster near integers or half-integers
  - This is what exp015 does — here we analyze it deeper

WHAT THIS EXPERIMENT ACTUALLY DOES:
  For H through B (Z=1..5), solve for EACH electron's optimal (r, L)
  with L as a continuous variable, for several different L-constraints:

  (a) L free, no minimum  (L ≥ 0.01)
  (b) L ≥ 1  (minimum angular momentum postulate)
  (c) L = integer only  (explicit quantization, for comparison)
  (d) L free but with EXCLUSION: at most 2 electrons can share
      the same L value (crude Pauli-like constraint)

  For each, report the optimized L values and whether they look quantized.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import product

from constants import HARTREE_EV
from reference_data import TOTAL_BINDING_EV, FIRST_IE_EV, Z as Z_of

sys.path.insert(0, os.path.dirname(__file__))
from exp009_principal_shells_BeNe import repulsion_sum as thomson_S


# --------------------------------------------------------------------------
# Energy function: each electron has (r_i, L_i), repulsion via shell theorem
# --------------------------------------------------------------------------

def atom_energy(rs: np.ndarray, Ls: np.ndarray, Z: int) -> float:
    """Total energy given electron radii and angular momenta."""
    N = len(rs)
    E = 0.0
    for i in range(N):
        E += Ls[i]**2 / (2.0 * rs[i]**2)   # kinetic
        E -= Z / rs[i]                        # nuclear attraction

    # Repulsion: shell theorem for different radii
    # For same radius: need angular info. Use Thomson if clustered.
    # Simple version: 1/max(r_i, r_j) for all pairs
    for i in range(N):
        for j in range(i + 1, N):
            E += 1.0 / max(rs[i], rs[j])

    return E


# --------------------------------------------------------------------------
# (a) L free, no minimum
# --------------------------------------------------------------------------

def optimize_free_L(Z: int, N: int, L_min: float = 0.01,
                    n_restarts: int = 10) -> dict:
    """Each electron optimizes (r, L) with L ≥ L_min."""
    best_E = 1e6
    best_rs = None
    best_Ls = None

    rng = np.random.RandomState(42)

    for trial in range(n_restarts):
        p0 = np.zeros(2 * N)
        for i in range(N):
            n_g = (i // 2) + 1
            p0[2*i] = n_g**2 / Z * (0.7 + 0.6 * rng.random())
            p0[2*i + 1] = max(L_min, n_g * (0.5 + rng.random()))

        bounds = []
        for i in range(N):
            bounds.append((0.01, 100.0))  # r
            bounds.append((L_min, 15.0))  # L

        def objective(params):
            rs = params[0::2]
            Ls = params[1::2]
            if np.any(rs <= 0) or np.any(Ls < L_min):
                return 1e6
            return atom_energy(rs, Ls, Z)

        res = minimize(objective, p0, method="Nelder-Mead",
                       options={"xatol": 1e-11, "fatol": 1e-13,
                                "maxiter": 200000, "adaptive": True})
        if res.fun < best_E:
            best_E = res.fun
            best_rs = res.x[0::2].copy()
            best_Ls = res.x[1::2].copy()

    # Also try differential evolution
    bounds = []
    for i in range(N):
        bounds.append((0.01, 50.0))
        bounds.append((L_min, 10.0))

    def obj_de(params):
        rs = params[0::2]
        Ls = params[1::2]
        return atom_energy(rs, Ls, Z)

    try:
        res_de = differential_evolution(obj_de, bounds, seed=42,
                                         maxiter=2000, tol=1e-12,
                                         polish=True)
        if res_de.fun < best_E:
            best_E = res_de.fun
            best_rs = res_de.x[0::2].copy()
            best_Ls = res_de.x[1::2].copy()
    except Exception:
        pass

    order = np.argsort(best_rs)
    return {
        "E_ha": best_E,
        "radii": [float(best_rs[i]) for i in order],
        "L_values": [float(best_Ls[i]) for i in order],
    }


# --------------------------------------------------------------------------
# (c) L = integer: try all integer assignments and pick lowest energy
# --------------------------------------------------------------------------

def optimize_integer_L(Z: int, N: int) -> dict:
    """Try all possible integer L assignments (L=1,2,3,...) for N electrons."""
    max_L = min(N + 2, 5)
    L_options = list(range(1, max_L + 1))

    best_E = 1e6
    best_rs = None
    best_Ls = None

    for L_combo in product(L_options, repeat=N):
        Ls = np.array(L_combo, dtype=float)

        # Optimize radii given fixed L
        def E_of_r(r_vec):
            if np.any(r_vec <= 1e-6):
                return 1e6
            return atom_energy(r_vec, Ls, Z)

        # Start radii at L²/Z
        r0 = np.array([l**2 / Z for l in Ls], dtype=float)
        r0 = np.clip(r0, 0.01, 100.0)

        res = minimize(E_of_r, r0, method="Nelder-Mead",
                       options={"xatol": 1e-11, "fatol": 1e-13,
                                "maxiter": 50000})
        if res.fun < best_E:
            best_E = res.fun
            best_rs = res.x.copy()
            best_Ls = Ls.copy()

    order = np.argsort(best_rs)
    return {
        "E_ha": best_E,
        "radii": [float(best_rs[i]) for i in order],
        "L_values": [float(best_Ls[i]) for i in order],
    }


# --------------------------------------------------------------------------
# (d) L free with Pauli-like exclusion: at most 2 electrons per L value
# --------------------------------------------------------------------------

def optimize_pauli_L(Z: int, N: int, L_min: float = 0.5) -> dict:
    """Free L optimization but with a penalty if >2 electrons share a
    similar L value (within 0.3 of each other)."""
    rng = np.random.RandomState(42)
    best_E = 1e6
    best_rs = None
    best_Ls = None

    def objective(params):
        rs = params[0::2]
        Ls = params[1::2]
        if np.any(rs <= 1e-6) or np.any(Ls < L_min):
            return 1e6
        E = atom_energy(rs, Ls, Z)

        # Exclusion penalty: for each pair of electrons with similar L,
        # add a large penalty if more than 2 share the same L
        # Actually, simpler: sort Ls, check for triples within tolerance
        Ls_sorted = np.sort(Ls)
        for i in range(len(Ls_sorted) - 2):
            if Ls_sorted[i+2] - Ls_sorted[i] < 0.3:
                E += 100.0  # large penalty for 3+ at same L
        return E

    for trial in range(12):
        p0 = np.zeros(2 * N)
        for i in range(N):
            n_g = (i // 2) + 1
            p0[2*i] = n_g**2 / Z * (0.7 + 0.6 * rng.random())
            # Spread L values to avoid collisions
            p0[2*i + 1] = max(L_min, 0.5 + i * 0.6 + 0.5 * rng.random())

        res = minimize(objective, p0, method="Nelder-Mead",
                       options={"xatol": 1e-11, "fatol": 1e-13,
                                "maxiter": 200000, "adaptive": True})
        if res.fun < best_E:
            best_E = res.fun
            best_rs = res.x[0::2].copy()
            best_Ls = res.x[1::2].copy()

    # Recompute actual energy without penalty
    actual_E = atom_energy(best_rs, best_Ls, Z)

    order = np.argsort(best_rs)
    return {
        "E_ha": actual_E,
        "radii": [float(best_rs[i]) for i in order],
        "L_values": [float(best_Ls[i]) for i in order],
    }


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

ATOMS = ["H", "He", "Li", "Be", "B"]


def run() -> dict:
    out = {}

    for label, opt_func in [
        ("(a) L_free_no_min", lambda Z, N: optimize_free_L(Z, N, L_min=0.01)),
        ("(b) L_free_min_1", lambda Z, N: optimize_free_L(Z, N, L_min=1.0)),
        ("(c) L_integer", lambda Z, N: optimize_integer_L(Z, N)),
        ("(d) L_pauli_excl", lambda Z, N: optimize_pauli_L(Z, N)),
    ]:
        results = {}
        for sym in ATOMS:
            Z = Z_of[sym]
            N = Z

            res = opt_func(Z, N)
            exp_ha = -TOTAL_BINDING_EV[sym] / HARTREE_EV

            # IE: compute ion
            if N > 1:
                res_ion = opt_func(Z, N - 1)
                IE_ha = res_ion["E_ha"] - res["E_ha"]
                IE_eV = IE_ha * HARTREE_EV
            else:
                IE_eV = -res["E_ha"] * HARTREE_EV

            err = 100.0 * abs(res["E_ha"] - exp_ha) / abs(exp_ha)
            IE_exp = FIRST_IE_EV[sym]
            IE_err = 100.0 * abs(IE_eV - IE_exp) / IE_exp

            results[sym] = {
                "E_ha": res["E_ha"], "E_exp": exp_ha, "err_pct": err,
                "IE_pred_eV": IE_eV, "IE_exp_eV": IE_exp, "IE_err_pct": IE_err,
                "radii": res["radii"], "L_values": res["L_values"],
            }

        out[label] = results

    return out


if __name__ == "__main__":
    out = run()
    print("=" * 120)
    print("Experiment 016: angular momentum emergence")
    print("  Can we derive L = integer rather than imposing it?")
    print("=" * 120)

    for label in ["(a) L_free_no_min", "(b) L_free_min_1",
                   "(c) L_integer", "(d) L_pauli_excl"]:
        results = out[label]
        print(f"\n{'='*100}")
        print(f"  {label}")
        print(f"{'='*100}")
        print(f"  {'atom':<4} {'E pred':>9} {'E exp':>9} {'E err%':>7}"
              f"  {'IE pred':>8} {'IE exp':>8} {'IE err%':>8}"
              f"  {'radii':<28} {'L values'}")
        for sym in ATOMS:
            v = results[sym]
            rs = ", ".join(f"{r:.3f}" for r in v["radii"])
            ls = ", ".join(f"{l:.3f}" for l in v["L_values"])
            print(f"  {sym:<4} {v['E_ha']:>9.4f} {v['E_exp']:>9.4f} {v['err_pct']:>6.2f}%"
                  f"  {v['IE_pred_eV']:>8.3f} {v['IE_exp_eV']:>8.3f} {v['IE_err_pct']:>7.1f}%"
                  f"  [{rs}]  [{ls}]")

    # Analysis
    print()
    print("=" * 120)
    print("ANALYSIS: What do the optimal L values look like?")
    print("=" * 120)

    for label in ["(a) L_free_no_min", "(b) L_free_min_1"]:
        results = out[label]
        print(f"\n  {label}:")
        for sym in ATOMS:
            Ls = results[sym]["L_values"]
            near_int = [abs(l - round(l)) < 0.15 for l in Ls]
            print(f"    {sym}: L = {Ls}")
            print(f"         Near integer? {near_int}")
            print(f"         Rounded:      {[round(l) for l in Ls]}")

    print()
    print("KEY FINDINGS")
    print("=" * 120)
    print("  1. With L free and no minimum: L → 0 (collapse). Need SOME minimum.")
    print("  2. With L ≥ 1: do optimal Ls cluster near integers?")
    print("     If yes → quantization is energetically favored (emergent).")
    print("     If no  → it's imposed, not derived.")
    print("  3. With integer L: how does this compare to free L?")
    print("     The GAP between free-L and integer-L energies tells us the")
    print("     'cost of quantization' — how much energy we pay for discreteness.")
    print("  4. With Pauli exclusion: does the L-spectrum spread out like")
    print("     real subshells (s, p, d)?")

    log_experiment({"exp": "016", "name": "angular_momentum_emergence",
                    "results": {k: {sym: {"E_ha": v["E_ha"], "L": v["L_values"]}
                                    for sym, v in results.items()}
                                for k, results in out.items()}})
