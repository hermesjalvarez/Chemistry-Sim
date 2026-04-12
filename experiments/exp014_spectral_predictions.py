"""Experiment 014 — Spectral predictions from the shell model.

Our best model (exp013 Model D: cloud a + screening δ) accurately predicts
total binding energies to ~0.85% mean error.  But total binding is only ONE
observable.  Can the same model predict other measurable quantities?

We test THREE classes of predictions:

  1. FIRST IONIZATION ENERGIES (IE₁)
     IE₁ = E(Z, N-1 electrons) - E(Z, N electrons)
     We compute the atom and its singly-charged ion, take the difference,
     and compare to NIST first-ionization-energy data (already in
     reference_data.py).

  2. HYDROGEN EMISSION SPECTRUM
     For hydrogen, the model gives E_n = -1/(2n²) exactly (Bohr formula).
     We compute the full Lyman (n→1), Balmer (n→2), and Paschen (n→3)
     series and compare to the exact Rydberg formula.

  3. SECOND AND THIRD IONIZATION ENERGIES
     IE₂ = E(Z, N-2) - E(Z, N-1) for selected atoms (He, Li, Be, Na, Mg).
     These test whether the model correctly predicts the LARGE jump in IE
     when you break into a closed shell.

Physics: the key question is whether a model optimized for TOTAL binding
also captures ENERGY DIFFERENCES accurately.  Total binding is dominated by
inner-shell electrons; ionization energies probe the outermost shell.  If
the model gets the outer-shell physics wrong, it could still match total E
(by cancelling errors) while failing badly on IE.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar

from constants import HARTREE_EV
from reference_data import TOTAL_BINDING_EV, FIRST_IE_EV, Z as Z_of, ELEMENTS
from core import log_experiment

sys.path.insert(0, os.path.dirname(__file__))
from exp009_principal_shells_BeNe import repulsion_sum as thomson_repulsion_sum


# --------------------------------------------------------------------------
# Best model from exp013: cloud delocalization + screening penetration
# --------------------------------------------------------------------------

def cloud_repulsion_C(k: int, a: float) -> float:
    """k-dependent delocalization: w(k) = a/k."""
    if k <= 1:
        return 0.0
    w = min(a / k, 1.0)
    uniform = k * (k - 1) / 2.0
    return (1.0 - w) * thomson_repulsion_sum(k) + w * uniform


def screen_penetration(delta: float):
    def f(n_i, n_j, r_i, r_j):
        return 1.0 - delta * (r_i / r_j)
    return f


def fill_octet(N: int) -> list[int]:
    """Shell filling: n=1 holds 2, higher shells hold up to 8."""
    occ, n, left = [], 1, N
    while left > 0:
        cap = 2 if n == 1 else 8
        occ.append(min(cap, left))
        left -= min(cap, left)
        n += 1
    return occ


# --------------------------------------------------------------------------
# Energy calculator (from exp013)
# --------------------------------------------------------------------------

def energy_of(
    Z: int,
    occupancy: list[int],
    a: float,
    delta: float,
) -> dict:
    """Compute atom energy with cloud + screening model."""
    K = len(occupancy)
    S_k = [cloud_repulsion_C(k, a) for k in occupancy]
    scr = screen_penetration(delta)

    def E_of_radii(r_vec: np.ndarray) -> float:
        if np.any(r_vec <= 1e-6):
            return 1e6
        E = 0.0
        for j in range(K):
            n_j = j + 1
            k_j = occupancy[j]
            r_j = r_vec[j]
            # Kinetic barrier
            E += k_j * (n_j * n_j) / (2.0 * r_j * r_j)
            # Effective nuclear charge
            Z_eff_j = float(Z)
            for i in range(K):
                if i == j:
                    continue
                if r_vec[i] < r_j * (1 - 1e-9):
                    s = scr(i + 1, n_j, r_vec[i], r_j)
                    Z_eff_j -= s * occupancy[i]
            E -= Z_eff_j * k_j / r_j
            # Same-shell repulsion
            E += S_k[j] / r_j
        return E

    R0 = np.array([(j + 1) ** 2 / max(Z - sum(occupancy[:j]), 1)
                    for j in range(K)], dtype=float)
    R0 = np.clip(R0, 0.01, 100.0)
    res = minimize(E_of_radii, R0, method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 60000})
    return {"Z": Z, "occupancy": occupancy, "E_ha": float(res.fun),
            "radii": [float(x) for x in res.x]}


# --------------------------------------------------------------------------
# Best-fit parameters from exp013
# --------------------------------------------------------------------------
A_BEST = 0.36
DELTA_BEST = 0.83


# --------------------------------------------------------------------------
# 1. First ionization energies
# --------------------------------------------------------------------------

def compute_first_IE(a: float, delta: float) -> dict:
    """IE₁ = E(Z, Z-1 electrons) - E(Z, Z electrons)."""
    results = {}
    for sym in ELEMENTS:
        Z = Z_of[sym]

        # Neutral atom: Z electrons
        occ_neutral = fill_octet(Z)
        res_neutral = energy_of(Z, occ_neutral, a, delta)

        # Singly-ionized: Z-1 electrons
        if Z == 1:
            # H⁺ has no electrons: E = 0
            E_ion = 0.0
        else:
            occ_ion = fill_octet(Z - 1)
            res_ion = energy_of(Z, occ_ion, a, delta)
            E_ion = res_ion["E_ha"]

        IE_ha = E_ion - res_neutral["E_ha"]   # positive (costs energy to remove)
        IE_eV = IE_ha * HARTREE_EV

        IE_exp_eV = FIRST_IE_EV[sym]
        err_eV = IE_eV - IE_exp_eV
        err_pct = 100.0 * abs(err_eV) / IE_exp_eV

        results[sym] = {
            "IE_pred_eV": IE_eV,
            "IE_exp_eV": IE_exp_eV,
            "err_eV": err_eV,
            "err_pct": err_pct,
            "E_neutral_ha": res_neutral["E_ha"],
            "E_ion_ha": E_ion,
        }

    mean_err = np.mean([v["err_pct"] for v in results.values()])
    results["_mean_err_pct"] = float(mean_err)
    return results


# --------------------------------------------------------------------------
# 2. Hydrogen emission spectrum
# --------------------------------------------------------------------------

def hydrogen_spectrum() -> dict:
    """Compute hydrogen transition energies from our model.

    For a single electron in shell n:
    E_n = T_n + V_n = n²/(2R²) - Z/R, minimized over R.
    dE/dR = 0  →  R* = n²/Z, E* = -Z²/(2n²).
    For hydrogen (Z=1): E_n = -1/(2n²).

    Transition energy: ΔE = E_upper - E_lower = 1/(2n_lo²) - 1/(2n_hi²).
    """
    results = {}

    # Exact energies from our model
    def E_n(n: int) -> float:
        """Energy of hydrogen with electron in shell n."""
        return -1.0 / (2.0 * n * n)

    # Series
    series = {
        "Lyman":   (1, [2, 3, 4, 5, 6, 7]),
        "Balmer":  (2, [3, 4, 5, 6, 7]),
        "Paschen": (3, [4, 5, 6, 7]),
    }

    # Exact Rydberg formula wavelengths (nm)
    # ΔE (Ha) = 1/(2 n_lo²) - 1/(2 n_hi²)
    # ΔE (eV) = ΔE_Ha * 27.2114
    # λ (nm)  = 1239.84 / ΔE_eV

    for series_name, (n_lo, n_hi_list) in series.items():
        lines = []
        for n_hi in n_hi_list:
            dE_ha = E_n(n_hi) - E_n(n_lo)  # negative (emission)
            dE_eV = abs(dE_ha) * HARTREE_EV
            wavelength_nm = 1239.84 / dE_eV if dE_eV > 0 else float('inf')

            # Exact Rydberg: 1/λ = R_inf * (1/n_lo² - 1/n_hi²)
            # R_inf = 1.0974e7 m⁻¹ = 0.010974 nm⁻¹
            R_inf = 1.09737e-2  # nm⁻¹
            inv_lam_exact = R_inf * (1.0/n_lo**2 - 1.0/n_hi**2)
            lam_exact = 1.0 / inv_lam_exact if inv_lam_exact > 0 else float('inf')

            lines.append({
                "n_hi": n_hi, "n_lo": n_lo,
                "dE_eV": dE_eV,
                "wavelength_nm": wavelength_nm,
                "wavelength_exact_nm": lam_exact,
                "err_nm": wavelength_nm - lam_exact,
                "err_pct": 100.0 * abs(wavelength_nm - lam_exact) / lam_exact,
            })
        results[series_name] = lines

    return results


# --------------------------------------------------------------------------
# 3. Higher ionization energies
# --------------------------------------------------------------------------

# Experimental successive ionization energies (eV) for selected atoms
# Source: NIST Atomic Spectra Database
IE_DATA = {
    "He": [24.587, 54.418],
    "Li": [5.392, 75.640, 122.454],
    "Be": [9.323, 18.211, 153.897, 217.718],
    "B":  [8.298, 25.155, 37.930, 259.375, 340.226],
    "C":  [11.260, 24.384, 47.888, 64.494, 392.087, 489.993],
    "N":  [14.534, 29.601, 47.449, 77.474, 97.890, 552.068, 667.046],
    "O":  [13.618, 35.117, 54.936, 77.414, 113.899, 138.120, 739.327, 871.410],
    "Na": [5.139, 47.286, 71.620, 98.91, 138.40, 172.18, 208.50, 264.25, 299.87, 1465.1, 1648.7],
    "Mg": [7.646, 15.035, 80.144, 109.265, 141.27, 186.76, 225.02, 265.96, 328.06, 367.50, 1761.8, 1962.7],
    "K":  [4.341, 31.63, 45.81, 60.91, 82.66, 99.4, 117.56, 154.88, 175.82, 503.8, 564.7],
    "Ca": [6.113, 11.872, 50.913, 67.27, 84.50, 108.78, 127.2, 147.24, 188.54, 211.28, 591.9, 657.2],
}


def compute_successive_IE(a: float, delta: float) -> dict:
    """Compute IE₁, IE₂, IE₃, ... for selected atoms."""
    results = {}

    for sym, ie_exp_list in IE_DATA.items():
        Z = Z_of[sym]
        ie_pred = []
        ie_err = []

        # Compute energies for N, N-1, N-2, ... electrons
        energies = {}
        for N in range(Z, max(Z - len(ie_exp_list) - 1, -1), -1):
            if N == 0:
                energies[0] = 0.0
            else:
                occ = fill_octet(N)
                res = energy_of(Z, occ, a, delta)
                energies[N] = res["E_ha"]

        for i, ie_exp in enumerate(ie_exp_list):
            N = Z - i
            if N not in energies or (N - 1) not in energies:
                break
            ie_ha = energies[N - 1] - energies[N]
            ie_eV = ie_ha * HARTREE_EV
            err_pct = 100.0 * abs(ie_eV - ie_exp) / ie_exp
            ie_pred.append({
                "ie_num": i + 1,
                "N_before": N,
                "IE_pred_eV": ie_eV,
                "IE_exp_eV": ie_exp,
                "err_pct": err_pct,
            })

        results[sym] = ie_pred

    return results


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

def run() -> dict:
    out = {}

    # 1. First ionization energies
    out["first_IE"] = compute_first_IE(A_BEST, DELTA_BEST)

    # 2. Hydrogen spectrum
    out["H_spectrum"] = hydrogen_spectrum()

    # 3. Successive ionization energies
    out["successive_IE"] = compute_successive_IE(A_BEST, DELTA_BEST)

    return out


if __name__ == "__main__":
    out = run()

    print("=" * 100)
    print("Experiment 014: spectral predictions from the shell model")
    print("  Model: cloud a=0.36 + screening δ=0.83 (best from exp013)")
    print("=" * 100)

    # --- Section 1: First ionization energies ---
    print("\n1. FIRST IONIZATION ENERGIES")
    print("-" * 80)
    print(f"  {'atom':<4} {'IE pred (eV)':>12} {'IE exp (eV)':>12} {'err (eV)':>10} {'err%':>7}")
    ie_data = out["first_IE"]
    for sym in ELEMENTS:
        v = ie_data[sym]
        print(f"  {sym:<4} {v['IE_pred_eV']:>12.3f} {v['IE_exp_eV']:>12.3f}"
              f" {v['err_eV']:>+10.3f} {v['err_pct']:>6.1f}%")
    print(f"\n  Mean IE₁ error: {ie_data['_mean_err_pct']:.1f}%")

    # --- Section 2: Hydrogen spectrum ---
    print("\n2. HYDROGEN EMISSION SPECTRUM")
    print("-" * 80)
    h_spec = out["H_spectrum"]
    for series_name in ["Lyman", "Balmer", "Paschen"]:
        print(f"\n  {series_name} series (n → {h_spec[series_name][0]['n_lo']}):")
        print(f"    {'transition':<12} {'ΔE (eV)':>9} {'λ pred (nm)':>12} {'λ exact (nm)':>13} {'err%':>7}")
        for line in h_spec[series_name]:
            trans = f"{line['n_hi']}→{line['n_lo']}"
            print(f"    {trans:<12} {line['dE_eV']:>9.3f} {line['wavelength_nm']:>12.2f}"
                  f" {line['wavelength_exact_nm']:>13.2f} {line['err_pct']:>6.2f}%")

    # --- Section 3: Successive ionization energies ---
    print("\n3. SUCCESSIVE IONIZATION ENERGIES")
    print("-" * 80)
    succ = out["successive_IE"]
    for sym in ["He", "Li", "Be", "Na", "Mg", "K", "Ca"]:
        if sym not in succ:
            continue
        print(f"\n  {sym} (Z={Z_of[sym]}):")
        print(f"    {'IE#':<5} {'N→N-1':<8} {'pred (eV)':>10} {'exp (eV)':>10} {'err%':>7}  {'note'}")
        for entry in succ[sym]:
            # Detect shell-breaking transitions
            occ_before = fill_octet(entry["N_before"])
            occ_after = fill_octet(entry["N_before"] - 1)
            note = ""
            if len(occ_after) < len(occ_before):
                note = "← closes shell"
            elif len(occ_before) > 1 and occ_before[-1] == 1:
                note = "← last in outer shell"
            print(f"    IE{entry['ie_num']:<3} {entry['N_before']:>2}→{entry['N_before']-1:<2}"
                  f"  {entry['IE_pred_eV']:>10.2f} {entry['IE_exp_eV']:>10.2f}"
                  f" {entry['err_pct']:>6.1f}%  {note}")

    # --- Summary ---
    print("\n" + "=" * 100)
    print("SUMMARY & INTERPRETATION")
    print("=" * 100)
    print()
    print("  1. FIRST IONIZATION ENERGIES:")
    print(f"     Mean error = {ie_data['_mean_err_pct']:.1f}% across {len(ELEMENTS)} atoms.")
    print("     This tests whether the model gets the OUTERMOST shell right,")
    print("     not just the total (which is dominated by inner shells).")
    print()
    print("  2. HYDROGEN SPECTRUM:")
    print("     Our model gives E_n = -1/(2n²), EXACTLY the Bohr/Rydberg formula.")
    print("     All transition wavelengths match to numerical precision.")
    print("     Lyman-α: 121.50 nm (UV), Balmer-α: 656.11 nm (red), etc.")
    print()
    print("  3. SUCCESSIVE IONIZATION ENERGIES:")
    print("     The model should predict the LARGE jump in IE when you break")
    print("     into a closed shell (e.g., IE₃ of Li >> IE₂, because you're")
    print("     pulling from the n=1 core). This is a stringent test of the")
    print("     shell structure and screening model.")

    log_experiment({"exp": "014", "name": "spectral_predictions",
                    "first_IE_mean_err": ie_data["_mean_err_pct"]})
