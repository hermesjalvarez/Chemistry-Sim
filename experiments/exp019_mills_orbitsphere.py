"""Experiment 019 — Mills's orbitsphere: testing the assumptions one by one.

Mills proposes a specific physical picture of the electron:
  - A 2D spherical shell of charge at radius R
  - Every point moves at the same speed v = 1/R (atomic units)
  - Total angular momentum = ℏ = 1 (atomic units), always
  - Kinetic energy T = 1/(2R²)  [NOT rigid rotation 3/(4R²)]
  - No electrostatic self-energy
  - Non-radiating (static multipole moments)

We DON'T impose multi-electron rules.  Instead, we test:

LEVEL 0: The five minimal assumptions on hydrogen (Z=1, 1 electron).
         This MUST give E = -0.5 Ha exactly.

LEVEL 1: Add a second electron (helium).  The five assumptions say
         nothing about electron-electron interaction.  So we test
         EVERY reasonable repulsion model and see which one(s) the
         DATA selects:

         (a) Two uniform shells at same R: V_ee = 1/R (shell theorem limit)
         (b) Antipodal points on same R: V_ee = 1/(2R)
         (c) "Orbitsphere overlap" — two charge distributions on the same
             sphere.  What's the actual electrostatic interaction energy
             of two uniform hemispheres?
         (d) Let the data decide: what V_ee coefficient λ gives the right
             He energy?  Is that λ physically interpretable?
         (e) What if the two electrons DON'T share a shell?  Let each
             electron have its own radius.

LEVEL 2: Three electrons (lithium).  Now we need shell structure.
         Does the model naturally want 2 electrons close + 1 far out?
         Or does it prefer all 3 at the same radius?
         We let the optimizer decide — no imposed filling rules.

LEVEL 3: Extend to Be, B (4-5 electrons).  Same approach: optimize
         all radii freely, see what structure emerges.

For each level, we ask:
  - What does the energy say?
  - What does the structure look like?
  - How close is it to experiment?
  - What additional assumption would we need to get closer?
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar

from constants import HARTREE_EV
from reference_data import TOTAL_BINDING_EV, FIRST_IE_EV, Z as Z_of
from core import log_experiment


# =========================================================================
# LEVEL 0: Hydrogen — the five minimal assumptions
# =========================================================================

def hydrogen_orbitsphere(Z: int = 1) -> dict:
    """Single electron orbitsphere.
    T = 1/(2R²), V = -Z/R, no self-energy.
    Analytical: R* = 1/Z, E* = -Z²/2.
    """
    # Analytical solution
    R_star = 1.0 / Z
    E_star = -Z**2 / 2.0

    # Verify numerically
    def E(R):
        return 1.0 / (2.0 * R**2) - Z / R

    from scipy.optimize import minimize_scalar
    res = minimize_scalar(E, bounds=(0.001, 100.0), method="bounded")

    return {
        "R_analytical": R_star,
        "E_analytical": E_star,
        "R_numerical": float(res.x),
        "E_numerical": float(res.fun),
        "E_eV": E_star * HARTREE_EV,
    }


# =========================================================================
# LEVEL 1: Helium — what repulsion model does the data select?
# =========================================================================

def helium_scan(Z: int = 2) -> dict:
    """Two electrons at radius R around nucleus Z.
    T_total = 2 × 1/(2R²) = 1/R²
    V_nuc   = -2Z/R
    V_ee    = λ/R  where λ is the unknown repulsion coefficient.

    E(R) = 1/R² - (2Z - λ)/R
    R* = 2/(2Z - λ), E* = -(2Z - λ)²/4

    What λ reproduces the experimental He energy?
    """
    E_exp = -TOTAL_BINDING_EV["He"] / HARTREE_EV   # ≈ -2.9037 Ha

    # Solve for λ: -(2Z - λ)²/4 = E_exp → (2Z-λ)² = -4 E_exp → 2Z-λ = √(-4E_exp)
    lam_exact = 2*Z - math.sqrt(-4 * E_exp)

    results = {}

    # Test specific physical models for λ
    models = {
        "shell_theorem":  1.0,    # Two uniform shells at same R
        "antipodal":      0.5,    # Two point charges at opposite poles
        "data_selected":  lam_exact,  # What the data requires
    }

    # Also: what if we compute the actual electrostatic energy of two
    # uniform hemispheres on a sphere of radius R?
    # Hemisphere 1: charge -1 over upper hemisphere
    # Hemisphere 2: charge -1 over lower hemisphere
    # Their mutual interaction energy (not self-energy of each) is:
    # V = ∫∫ dq₁ dq₂ / |r₁ - r₂| integrated over hemisphere 1 × hemisphere 2
    # For uniform charge on a sphere of radius R, two hemispheres interact as:
    # V_hemi = (1/R) × [numerical factor]
    # The numerical factor for two hemispheres: we can compute this.
    #
    # Actually: the full interaction of two uniform spherical shells at
    # the same radius R, each with charge q, is V = q²/R (shell theorem).
    # But this is the TOTAL interaction including same-hemisphere pairs.
    # If we split into two hemispheres, the interaction between them
    # is a fraction of the total.

    # Let me compute the hemisphere-hemisphere interaction numerically
    lam_hemisphere = compute_hemisphere_interaction()
    models["hemisphere"] = lam_hemisphere

    # Also test: what if each electron is a "cap" (not hemisphere) that
    # subtends a solid angle less than 2π?  For antipodal caps of half-angle θ:
    # As θ→0: approaches antipodal points, λ→0.5
    # As θ→π/2: hemispheres
    # As θ→π: full shells, λ→1.0

    for name, lam in models.items():
        Zeff = 2*Z - lam
        R_star = 2.0 / Zeff
        E_star = -Zeff**2 / 4.0
        E_eV = E_star * HARTREE_EV
        err_pct = 100.0 * (E_star - E_exp) / abs(E_exp)

        results[name] = {
            "lambda": lam,
            "R_star": R_star,
            "E_star": E_star,
            "E_eV": E_eV,
            "err_pct": err_pct,
        }

    results["_E_exp"] = E_exp
    results["_lambda_exact"] = lam_exact
    return results


def compute_hemisphere_interaction(n_pts: int = 2000) -> float:
    """Compute λ for two uniform hemispheres on a unit sphere.

    Each hemisphere has charge -1 spread uniformly.
    λ = R × V_interaction, where V = ∫∫ σ₁ σ₂ / |r₁-r₂| dA₁ dA₂.

    For a unit sphere with total charge 1 per hemisphere:
    σ = 1/(2π) per hemisphere.
    """
    rng = np.random.RandomState(42)

    # Monte Carlo: sample points on upper hemisphere (hemisphere 1)
    # and lower hemisphere (hemisphere 2)
    def sample_hemisphere(n, sign=1):
        """Sample n points uniformly on upper (sign=1) or lower (sign=-1) hemisphere."""
        pts = []
        while len(pts) < n:
            x = rng.randn(3)
            x /= np.linalg.norm(x)
            if sign * x[2] > 0:
                pts.append(x)
        return np.array(pts)

    pts1 = sample_hemisphere(n_pts, sign=1)
    pts2 = sample_hemisphere(n_pts, sign=-1)

    # V = (1/N²) × Σᵢ Σⱼ 1/|r₁ᵢ - r₂ⱼ| × (area₁ × area₂) × σ₁ × σ₂
    # Area of each hemisphere = 2π (for unit sphere)
    # σ = 1/(2π) for charge 1 spread over hemisphere
    # V = (2π)² × (1/(2π))² × (1/N²) × sum = (1/N²) × sum

    total = 0.0
    for i in range(n_pts):
        diffs = pts2 - pts1[i]  # (n_pts, 3)
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        total += np.sum(1.0 / dists)

    lam = total / (n_pts * n_pts)
    return float(lam)


# =========================================================================
# LEVEL 2-3: Multi-electron — let structure emerge
# =========================================================================

def multi_electron_free(Z: int, N: int, lam: float) -> dict:
    """N electrons around nucleus Z.  Each electron has its own radius r_i.
    T_i = 1/(2 r_i²)    [Mills: every electron has L=ℏ, T=ℏ²/(2mR²)]
    V_nuc_i = -Z / r_i
    V_ee_ij = depends on whether electrons are at same or different radii:
      - Same radius (|r_i - r_j| < tol): use λ/r  (the pair correlation)
      - Different radii: 1/max(r_i, r_j)  (shell theorem)
    No self-energy.

    We optimize ALL N radii freely.  No filling rules.
    """
    def energy(radii):
        if np.any(radii <= 1e-6):
            return 1e6
        E = 0.0
        for i in range(N):
            E += 1.0 / (2.0 * radii[i]**2)   # kinetic
            E -= Z / radii[i]                   # nuclear

        # Electron-electron repulsion
        for i in range(N):
            for j in range(i + 1, N):
                ri, rj = radii[i], radii[j]
                # Are they at the "same" radius?
                if abs(ri - rj) / max(ri, rj) < 0.01:
                    # Same shell: use λ
                    E += lam / max(ri, rj)
                else:
                    # Different shells: shell theorem
                    E += 1.0 / max(ri, rj)
        return E

    # Try many starting configurations
    best_E = 1e6
    best_r = None
    rng = np.random.RandomState(42)

    for trial in range(40):
        r0 = np.zeros(N)
        for i in range(N):
            # Various strategies
            if trial < 10:
                # All at same radius
                r0[i] = (1.0 / Z) * (0.5 + rng.random())
            elif trial < 20:
                # Paired: first 2 close, rest further
                if i < 2:
                    r0[i] = 1.0 / Z * (0.8 + 0.4 * rng.random())
                else:
                    r0[i] = 4.0 / Z * (0.5 + rng.random())
            elif trial < 30:
                # Each at different radius
                r0[i] = (i + 1)**2 / Z * (0.5 + rng.random())
            else:
                # Random
                r0[i] = rng.uniform(0.1, 10.0)

        res = minimize(energy, r0, method="Nelder-Mead",
                       options={"xatol": 1e-11, "fatol": 1e-13,
                                "maxiter": 100000, "adaptive": True})
        if res.fun < best_E:
            best_E = res.fun
            best_r = res.x.copy()

    # Sort radii and identify shells
    radii_sorted = np.sort(best_r)
    shells = identify_shells(radii_sorted)

    return {
        "Z": Z, "N": N,
        "E_ha": best_E,
        "E_eV": best_E * HARTREE_EV,
        "radii": [float(r) for r in radii_sorted],
        "shells": shells,
    }


def identify_shells(radii: np.ndarray, tol: float = 0.02) -> list[dict]:
    """Group radii into shells."""
    shells = []
    current_r = [radii[0]]
    for i in range(1, len(radii)):
        if abs(radii[i] - current_r[-1]) / current_r[-1] < tol:
            current_r.append(radii[i])
        else:
            shells.append({"k": len(current_r), "r": float(np.mean(current_r))})
            current_r = [radii[i]]
    shells.append({"k": len(current_r), "r": float(np.mean(current_r))})
    return shells


# =========================================================================
# Run
# =========================================================================

def run() -> dict:
    out = {}

    # --- LEVEL 0: Hydrogen ---
    print("LEVEL 0: Hydrogen (single electron)")
    print("-" * 70)
    h_results = {}
    for Z in [1, 2, 3, 4, 5]:
        r = hydrogen_orbitsphere(Z)
        ion_name = {1: "H", 2: "He+", 3: "Li2+", 4: "Be3+", 5: "B4+"}[Z]
        h_results[ion_name] = r
        print(f"  {ion_name:<6} R* = {r['R_analytical']:.4f}  "
              f"E* = {r['E_analytical']:.4f} Ha = {r['E_eV']:.3f} eV")
    out["level0"] = h_results
    print()
    print("  Verdict: Mills's 5 assumptions → EXACT hydrogen-like energies.")
    print("  E = -Z²/2, R = 1/Z.  This is the Bohr result from a shell picture.")
    print()

    # --- LEVEL 1: Helium ---
    print("LEVEL 1: Helium (two electrons, what repulsion?)")
    print("-" * 70)
    he = helium_scan(Z=2)
    out["level1"] = he
    print(f"  Experimental He energy: {he['_E_exp']:.4f} Ha")
    print(f"  Repulsion coefficient that fits exactly: λ = {he['_lambda_exact']:.4f}")
    print()
    print(f"  {'model':<20} {'λ':>6} {'R*':>8} {'E* (Ha)':>10} {'err%':>8}")
    for name in ["shell_theorem", "antipodal", "hemisphere", "data_selected"]:
        v = he[name]
        print(f"  {name:<20} {v['lambda']:>6.3f} {v['R_star']:>8.4f}"
              f" {v['E_star']:>10.4f} {v['err_pct']:>+7.1f}%")

    print()
    print(f"  The data requires λ = {he['_lambda_exact']:.4f}.")
    print(f"  Antipodal points give λ = 0.500 (5.5% overbinding).")
    print(f"  Shell theorem gives λ = 1.000 (22% underbinding).")
    print(f"  Hemispheres give λ = {he['hemisphere']['lambda']:.3f}.")
    print(f"  The exact λ is between antipodal and shell-theorem.")
    print(f"  Physical interpretation: the two electrons are PARTIALLY correlated.")
    print()

    # --- LEVEL 2: Li, Be, B — emergent structure ---
    print("LEVEL 2-3: Li through B — does shell structure emerge?")
    print("-" * 70)
    print(f"  Using λ = {he['_lambda_exact']:.4f} (He-exact) for same-shell pairs.")
    print()

    lam = he["_lambda_exact"]
    multi_results = {}

    for sym, Z in [("Li", 3), ("Be", 4), ("B", 5)]:
        N = Z
        r = multi_electron_free(Z, N, lam)
        E_exp = -TOTAL_BINDING_EV[sym] / HARTREE_EV
        err = 100.0 * (r["E_ha"] - E_exp) / abs(E_exp)

        # IE: remove outermost electron
        r_ion = multi_electron_free(Z, N - 1, lam)
        IE_ha = r_ion["E_ha"] - r["E_ha"]
        IE_eV = IE_ha * HARTREE_EV
        IE_exp = FIRST_IE_EV[sym]
        IE_err = 100.0 * (IE_eV - IE_exp) / IE_exp

        r["E_exp"] = E_exp
        r["err_pct"] = err
        r["IE_pred_eV"] = IE_eV
        r["IE_exp_eV"] = IE_exp
        r["IE_err_pct"] = IE_err
        multi_results[sym] = r

        shell_str = "  ".join(f"[{s['k']}e @ r={s['r']:.3f}]" for s in r["shells"])
        print(f"  {sym} (Z={Z}): E = {r['E_ha']:.4f} Ha (exp {E_exp:.4f},"
              f" err {err:+.1f}%)")
        print(f"    IE = {IE_eV:.3f} eV (exp {IE_exp:.3f}, err {IE_err:+.1f}%)")
        print(f"    Structure: {shell_str}")
        print(f"    Radii: {[f'{x:.4f}' for x in r['radii']]}")
        print()

    out["level2"] = multi_results

    # --- What if λ is different? Scan λ for Li ---
    print("SENSITIVITY: How does Li depend on the choice of λ?")
    print("-" * 70)
    E_exp_Li = -TOTAL_BINDING_EV["Li"] / HARTREE_EV
    for lam_test in [0.3, 0.4, 0.5, 0.593, 0.7, 0.8, 1.0]:
        r = multi_electron_free(3, 3, lam_test)
        err = 100.0 * (r["E_ha"] - E_exp_Li) / abs(E_exp_Li)
        shells = identify_shells(np.array(r["radii"]))
        shell_str = " + ".join(f"{s['k']}e@{s['r']:.2f}" for s in shells)
        print(f"  λ={lam_test:.3f}: E={r['E_ha']:.4f} Ha (err {err:+.1f}%)"
              f"  structure: {shell_str}")

    out["li_lambda_scan"] = True  # just a flag

    return out


if __name__ == "__main__":
    out = run()

    print()
    print("=" * 90)
    print("SUMMARY: What emerges vs. what must be assumed")
    print("=" * 90)
    print()
    print("  EMERGES NATURALLY:")
    print("    - Hydrogen energy and radius (from T = 1/(2R²), no self-energy)")
    print("    - Hydrogenic scaling E ∝ Z²")
    print("    - Shell structure in Li+ (two electrons cluster together)")
    print()
    print("  MUST BE ASSUMED OR FIT:")
    print("    - T = 1/(2R²) specifically (not 3/(4R²) or other coefficient)")
    print("    - L = 1 always (why not L = 0 or L = 2?)")
    print("    - No self-energy (why not?)")
    print("    - Electron-electron correlation λ (different models give very")
    print("      different answers for He)")
    print()
    print("  KEY OPEN QUESTION:")
    print("    Can the same λ that fits He also work for Li, Be, B?")
    print("    Or does each atom need its own λ?  If a single universal λ works,")
    print("    that's strong evidence for a real physical mechanism.")

    log_experiment({"exp": "019", "name": "mills_orbitsphere",
                    "lambda_He_exact": out["level1"]["_lambda_exact"]})
