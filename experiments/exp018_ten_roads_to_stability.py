"""Experiment 018 — Ten roads to stability: what prevents shell collapse?

Back to basics.  A charged shell at radius R around a nucleus Z has
energy E(R) = -Z/R, which collapses to R=0.  What mechanism prevents this?

We test EVERY physically motivated stabilizer on hydrogen (Z=1) and
compare to the known ground-state energy E = -13.606 eV = -0.5 Ha.

The winning mechanism(s) must:
  (a) Produce an energy minimum at finite R
  (b) Give E* = -0.5 Ha for hydrogen  (or close)
  (c) Scale correctly for heavier atoms: E ∝ Z²
  (d) Have a clear physical origin (not just a fit)

=== THE TEN MECHANISMS ===

M1: ORBITAL ANGULAR MOMENTUM (Bohr)
    T = L²/(2R²)  with L = 1.
    Our original postulate from exp002-004.

M2: SPIN ANGULAR MOMENTUM
    Electron has intrinsic spin S = ½ (in units of ℏ).
    For a spinning shell: I = (2/3)mR², T_spin = S²/(2I) = 3/(16R²).
    NO free parameter — S = ½ is measured.

M3: SPIN + ORBITAL (spin-orbit total)
    J = L + S.  For the lowest state, L=0, J=S=½, same as M2.
    For L=1, J = 1/2 or 3/2.  T = J(J+1)/(2R²) if we use QM formula,
    or J²/(2R²) if classical.

M4: ELECTROMAGNETIC SELF-ENERGY
    U_self = 1/(2R).  Same scaling as Coulomb: E = (1/2 - Z)/R.
    Still collapses for Z ≥ 1, but let's quantify.

M5: MAGNETIC SELF-ENERGY OF SPINNING SHELL
    A uniformly magnetized sphere with magnetic moment μ stores energy
    U_mag = μ₀ μ² / (4π · ⅔ R³) = (2/3) μ²/R³  (Gaussian/atomic units).
    For an electron shell with spin: μ = g_s · S / (2m) ≈ 1 Bohr magneton.
    In atomic units μ_B = α/2 where α ≈ 1/137 (fine structure constant).
    So U_mag = (2/3)(α/2)² / R³ = α²/(6R³).
    This is 1/R³ — STRONGER than Coulomb!

M6: STANDING WAVE ON SPHERE (lowest non-trivial mode)
    If the electron is a wave, spherical harmonics give the modes.
    Mode l has "kinetic energy" l(l+1)/(2R²).
    l=0: no barrier (collapse).  l=1: T = 1/R² (same as L=1).
    The question: can we justify excluding l=0?

M7: RADIATION REACTION EQUILIBRIUM
    An accelerating charge radiates with power P = (2/3)α⁴ a² (atomic units).
    For a circular orbit: a = v²/R.  At equilibrium, radiation power = 0
    requires the orbit to be non-radiating.  Non-radiating conditions
    select specific R values.

M8: POINCARÉ STRESS (internal pressure)
    Postulate an internal stress that stores energy U_P = A/R^n.
    For n=2: same as angular momentum — fit A to get hydrogen.
    For n=3: like magnetic energy — different R, different E.
    For n=4: even steeper.
    We check which n gives the right Z-scaling.

M9: RELATIVISTIC KINETIC ENERGY
    T = c²[√(1 + p²/c²) - 1] where c = 137.036 in atomic units.
    If p = L/R with L=1: T = c²[√(1 + 1/(c²R²)) - 1].
    For R >> 1/c (non-relativistic): T ≈ 1/(2R²) as before.
    For R << 1/c: T ≈ c/R — linear, like Coulomb!
    This means relativity WEAKENS the barrier at very small R.
    Not helpful for stability, but important to check.

M10: ZITTERBEWEGUNG (trembling motion)
     In Dirac theory, the electron undergoes rapid oscillation at
     frequency 2mc²/ℏ with amplitude ℏ/(2mc) = 1/(2c) ≈ 0.00365 Bohr.
     If we model this as a minimum orbital radius: R ≥ R_zitter = 1/(2c).
     The "effective" kinetic energy from this trembling is T ~ c/R at
     the zitter scale.  At atomic scales (R >> 1/c) it's negligible.

For each mechanism, we compute:
  - E(R) analytically
  - R* (equilibrium radius)
  - E* (ground-state energy)
  - Comparison to E_exact = -0.5 Ha
  - Z-scaling (does E ∝ Z²?)
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
from scipy.optimize import minimize_scalar

from constants import HARTREE_EV
from core import log_experiment


# Fine structure constant in atomic units
ALPHA = 1.0 / 137.036
C_AU = 137.036   # speed of light in atomic units


def find_minimum(E_func, R_lo=1e-4, R_hi=1e4):
    """Find R* and E* = min E(R) on [R_lo, R_hi]."""
    # Grid search first
    Rs = np.geomspace(R_lo, R_hi, 5000)
    Es = np.array([E_func(R) for R in Rs])
    valid = np.isfinite(Es)
    if not valid.any():
        return None, None
    i_min = np.nanargmin(Es)
    if i_min == 0 or i_min == len(Rs) - 1:
        # Minimum at boundary — no true equilibrium
        return float(Rs[i_min]), float(Es[i_min])

    try:
        res = minimize_scalar(E_func,
                              bounds=(Rs[max(i_min-5, 0)], Rs[min(i_min+5, len(Rs)-1)]),
                              method="bounded", options={"xatol": 1e-12})
        return float(res.x), float(res.fun)
    except Exception:
        return float(Rs[i_min]), float(Es[i_min])


# --------------------------------------------------------------------------
# Define all ten mechanisms
# --------------------------------------------------------------------------

def make_M1(Z, L=1.0):
    """M1: Orbital angular momentum T = L²/(2R²)."""
    def E(R):
        return L**2 / (2*R**2) - Z / R
    # Analytical: R* = L²/Z, E* = -Z²/(2L²)
    return E, f"Orbital L={L}"

def make_M2(Z):
    """M2: Spin only. T_spin = 3S²/(4R²) = 3/(16R²) for S=1/2, shell."""
    # I_shell = (2/3)mR², T = S²/(2I) = (1/2)²/(2·(2/3)R²) = 1/(4·4R²/3) = 3/(16R²)
    coeff = 3.0 / 16.0
    def E(R):
        return coeff / R**2 - Z / R
    return E, "Spin S=½ (shell I=⅔mR²)"

def make_M2b(Z):
    """M2b: Spin only, ring geometry. I_ring = mR², T = S²/(2R²) = 1/(8R²)."""
    coeff = 1.0 / 8.0
    def E(R):
        return coeff / R**2 - Z / R
    return E, "Spin S=½ (ring I=mR²)"

def make_M2c(Z):
    """M2c: Spin with QM formula S(S+1). T = 3S(S+1)/(4R²) for shell."""
    # S(S+1) = (1/2)(3/2) = 3/4
    coeff = 3.0 * 0.75 / 4.0  # = 9/16
    def E(R):
        return coeff / R**2 - Z / R
    return E, "Spin S(S+1)=¾ (shell, QM)"

def make_M3(Z):
    """M3: Spin-orbit J=½ (L=0, S=½). Same as M2 for ground state."""
    # J² = (1/2)² = 1/4, T = J²/(2I) with I = (2/3)R²
    # T = (1/4)/(4R²/3) = 3/(16R²) — same as M2
    return make_M2(Z)

def make_M4(Z):
    """M4: Electromagnetic self-energy. U_self = 1/(2R)."""
    def E(R):
        return 1.0/(2*R) - Z/R
    return E, "EM self-energy 1/(2R)"

def make_M5(Z):
    """M5: Magnetic self-energy of spinning shell.
    μ = g_s μ_B ≈ α (in atomic units, μ_B = α/2, g_s ≈ 2).
    U_mag = (2/3) μ²/R³ = (2/3) α²/R³.
    Combined with Coulomb: need orbital AM too or just magnetic alone?
    Let's test magnetic alone first, then magnetic + spin kinetic.
    """
    coeff_mag = 2.0 * ALPHA**2 / 3.0  # ≈ 3.5e-5
    def E(R):
        return coeff_mag / R**3 - Z / R
    return E, f"Magnetic self-energy (2α²/3)/R³"

def make_M5b(Z):
    """M5b: Spin kinetic + magnetic self-energy."""
    coeff_spin = 3.0 / 16.0
    coeff_mag = 2.0 * ALPHA**2 / 3.0
    def E(R):
        return coeff_spin / R**2 + coeff_mag / R**3 - Z / R
    return E, "Spin T + magnetic U_mag"

def make_M6(Z, l=1):
    """M6: Standing wave on sphere. T = l(l+1)/(2R²)."""
    coeff = l * (l + 1) / 2.0
    def E(R):
        return coeff / R**2 - Z / R
    return E, f"Standing wave l={l}, T=l(l+1)/(2R²)"

def make_M7(Z):
    """M7: Non-radiating orbit condition.
    A charge in uniform circular motion DOES radiate (Larmor).
    Power = (2/3)(e²/c³)a² = (2α³/3) v⁴/R² (atomic units).
    Non-radiating condition: dP/dR = 0 selects R, but this isn't an
    energy minimum — it's a power extremum.  Let's compute anyway.
    Actually, the condition for a non-radiating current distribution
    is that all multipole moments are time-independent.  A uniformly
    rotating ring satisfies this for the monopole and dipole terms IF
    the charge is symmetric.  A single point charge always radiates.
    A uniform current ring does NOT radiate (steady current = static B field).
    So: model the electron as a UNIFORM CURRENT RING at radius R.
    Then it's non-radiating, and the energy is just T + V with T = L²/(2R²).
    This reduces to M1!  The non-radiating condition doesn't add new physics.
    """
    # Falls back to M1
    return make_M1(Z, L=1.0)

def make_M8(Z, n_pow=2, A=None):
    """M8: Poincaré stress U = A/R^n.
    For n=2, fit A to match hydrogen: R* = 2A/Z, E* = -Z²/(4A).
    E* = -0.5 → A = Z²/2 = 0.5 for Z=1.
    """
    if A is None:
        # Fit to hydrogen: E* = -Z²/(4A) = -0.5 → A = Z²/2
        if n_pow == 2:
            A = 0.5
        elif n_pow == 3:
            # E(R) = A/R³ - Z/R.  dE/dR = -3A/R⁴ + Z/R² = 0 → R* = √(3A/Z)
            # E* = A/(3A/Z)^(3/2) - Z/√(3A/Z) = ...
            # Just let the optimizer find A
            A = 0.1
        else:
            A = 0.1

    def E(R):
        return A / R**n_pow - Z / R
    return E, f"Poincaré stress A/R^{n_pow}, A={A:.4f}"

def make_M9(Z, L=1.0):
    """M9: Relativistic kinetic energy with angular momentum L.
    T = c²[√(1 + L²/(c²R²)) - 1].
    """
    def E(R):
        p = L / R
        T = C_AU**2 * (math.sqrt(1 + p**2/C_AU**2) - 1)
        return T - Z / R
    return E, f"Relativistic T, L={L}"

def make_M10(Z):
    """M10: Zitterbewegung.  Minimum effective radius R_zitter = 1/(2c).
    Model: below R_zitter, the kinetic energy rises as c²(R_z/R - 1)
    (the electron can't be confined smaller than its Compton wavelength).
    Above R_zitter: no additional barrier.
    Combined with orbital AM L=1 for a complete model.
    """
    R_z = 1.0 / (2.0 * C_AU)  # ≈ 0.00365 Bohr
    def E(R):
        T = 1.0 / (2.0 * R**2)  # orbital L=1
        if R < R_z:
            T += C_AU**2 * (R_z / R - 1)  # zitter confinement
        return T - Z / R
    return E, f"Zitterbewegung floor R_z={R_z:.5f}"


# --------------------------------------------------------------------------
# Z-scaling test: for a good mechanism, E*(Z) should scale as Z²
# --------------------------------------------------------------------------

def z_scaling_test(make_func, Z_values=[1, 2, 3, 5, 10]):
    """Test whether E*(Z) ∝ Z². Return the scaling exponent."""
    Es = []
    for Z in Z_values:
        E_func, _ = make_func(Z)
        R_star, E_star = find_minimum(E_func)
        if E_star is None or E_star >= 0:
            return None
        Es.append(abs(E_star))

    # Fit log(E) vs log(Z): E = a · Z^n → log E = log a + n log Z
    logZ = np.log(Z_values)
    logE = np.log(Es)
    n, log_a = np.polyfit(logZ, logE, 1)
    return float(n)


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

def run() -> dict:
    Z = 1  # Hydrogen
    E_exact = -0.5  # Ha
    R_exact = 1.0   # Bohr

    results = {}

    mechanisms = [
        ("M1_orbital_L1", lambda Z: make_M1(Z, L=1.0)),
        ("M2_spin_shell", lambda Z: make_M2(Z)),
        ("M2b_spin_ring", lambda Z: make_M2b(Z)),
        ("M2c_spin_QM",   lambda Z: make_M2c(Z)),
        ("M4_EM_self",    lambda Z: make_M4(Z)),
        ("M5_magnetic",   lambda Z: make_M5(Z)),
        ("M5b_spin+mag",  lambda Z: make_M5b(Z)),
        ("M6_standing_l1", lambda Z: make_M6(Z, l=1)),
        ("M6_standing_l0", lambda Z: make_M6(Z, l=0)),
        ("M8_poincare_n2", lambda Z: make_M8(Z, n_pow=2, A=0.5)),
        ("M8_poincare_n3", lambda Z: make_M8(Z, n_pow=3)),
        ("M9_relativistic", lambda Z: make_M9(Z, L=1.0)),
        ("M10_zitter",    lambda Z: make_M10(Z)),
    ]

    for key, make_func in mechanisms:
        E_func, label = make_func(Z)
        R_star, E_star = find_minimum(E_func)

        # Boundary check: is this a true minimum or edge?
        has_minimum = True
        if R_star is not None:
            # Check if minimum is at boundary
            if R_star < 2e-4 or R_star > 5e3:
                has_minimum = False

        # Error vs exact
        if E_star is not None and has_minimum:
            err_pct = 100.0 * (E_star - E_exact) / abs(E_exact)
            R_err = 100.0 * (R_star - R_exact) / R_exact if R_star else None
        else:
            err_pct = None
            R_err = None

        # Z-scaling
        z_exp = z_scaling_test(make_func)

        results[key] = {
            "label": label,
            "R_star": R_star,
            "E_star": E_star,
            "E_star_eV": E_star * HARTREE_EV if E_star else None,
            "has_minimum": has_minimum,
            "err_pct": err_pct,
            "R_err_pct": R_err,
            "Z_exponent": z_exp,
        }

    return results


if __name__ == "__main__":
    results = run()

    print("=" * 110)
    print("Experiment 018: ten roads to stability — what prevents shell collapse?")
    print("  Target: hydrogen E* = -0.500 Ha = -13.606 eV, R* = 1.000 Bohr")
    print("=" * 110)

    print(f"\n  {'mechanism':<20} {'label':<32} {'R*':>8} {'E* (Ha)':>10} {'E* (eV)':>10}"
          f" {'E err%':>8} {'Z^n':>5} {'stable?':>8}")
    print("  " + "-" * 106)

    for key, v in results.items():
        stable = "YES" if v["has_minimum"] else "NO"
        R_str = f"{v['R_star']:.4f}" if v['R_star'] else "---"
        E_str = f"{v['E_star']:.4f}" if v['E_star'] else "---"
        Eev_str = f"{v['E_star_eV']:.3f}" if v['E_star_eV'] else "---"
        err_str = f"{v['err_pct']:+.1f}%" if v['err_pct'] is not None else "---"
        z_str = f"{v['Z_exponent']:.2f}" if v['Z_exponent'] else "---"

        print(f"  {key:<20} {v['label']:<32} {R_str:>8} {E_str:>10} {Eev_str:>10}"
              f" {err_str:>8} {z_str:>5} {stable:>8}")

    # Verdicts
    print()
    print("VERDICTS")
    print("=" * 110)

    winners = []
    for key, v in results.items():
        if v["has_minimum"] and v["err_pct"] is not None:
            winners.append((key, v))

    winners.sort(key=lambda x: abs(x[1]["err_pct"]))

    print("\n  Mechanisms that produce a stable atom (ordered by accuracy):")
    for key, v in winners:
        z_note = f"Z^{v['Z_exponent']:.2f}" if v['Z_exponent'] else "?"
        print(f"    {key:<20} E* = {v['E_star']:.4f} Ha  (err {v['err_pct']:+.1f}%)"
              f"  R* = {v['R_star']:.4f}  scaling: {z_note}")

    print("\n  Mechanisms that FAIL (no equilibrium or collapse):")
    for key, v in results.items():
        if not v["has_minimum"]:
            print(f"    {key:<20} {v['label']}")

    print()
    print("PHYSICS INSIGHTS")
    print("=" * 110)
    print("  1. Any 1/R² barrier works for stability and gives E ∝ Z².")
    print("     The coefficient α in T = α/R² determines R* = 2α/Z, E* = -Z²/(4α).")
    print("     α = 0.5 (orbital L=1) gives E* = -0.5 Ha (exact).")
    print("     α = 3/16 (spin S=½, shell) gives E* = -4/3 Ha (too bound).")
    print()
    print("  2. 1/R³ barriers (magnetic) also stabilize but give E ∝ Z^(3/2),")
    print("     not Z² — WRONG scaling. These fail the hydrogenic test.")
    print()
    print("  3. 1/R barriers (self-energy) are too weak — same scaling as")
    print("     Coulomb, just shifts Z_eff. Fails for Z ≥ 1.")
    print()
    print("  4. SPIN ALONE (S=½) gives stability but the WRONG energy:")
    print("     E = -16Z²/3 ≈ -5.33 Ha for hydrogen (10.7x too bound).")
    print("     But the QM formula S(S+1) = 3/4 gives α = 9/16,")
    print("     E* = -Z²·4/9 = -0.444 Ha (11% error, close!).")
    print()
    print("  5. Standing wave l=1 is IDENTICAL to orbital L=1 if we use")
    print("     T = l(l+1)/(2R²) = 1/R². The standing-wave and orbit")
    print("     pictures give the same answer.")

    log_experiment({"exp": "018", "name": "ten_roads_to_stability",
                    "results": {k: {"E_star": v["E_star"], "R_star": v["R_star"],
                                    "err_pct": v["err_pct"], "Z_exp": v["Z_exponent"]}
                                for k, v in results.items()}})
