"""Experiment 006 - Helium, many configurations (single-shell model, no corrections).

Given our one-electron theory:  T(R) = 1/(2 R^2),  V_nuc = -Z/R,
we now add a second electron shell.  The electrostatic interaction
between two concentric shells (Gauss / shell theorem) is

    U_12 = (-1)(-1) / max(R1, R2) = 1 / max(R1, R2).

We consider several configurations, all built ONLY from the assumptions
we have made so far:

    (A) Two shells at independent radii R1, R2 (they'll screen each other)
    (B) Two shells forced to the same radius R (no correlation, lambda=1)
    (C) One electron modeled as a full shell, the other as a point at the
        same radius (i.e. in the potential from the first shell)
    (D) Two point electrons on opposite ends of a diameter (R1=R2=R,
        distance 2R, repulsion 1/(2R))   -- an "antipodal" configuration
    (E) Two point electrons 90 degrees apart on the equator
        (distance R*sqrt(2), repulsion 1/(R*sqrt(2)))
    (F) Two point electrons on *perpendicular* great circles, each with
        uniform angular distribution (integrated).  This gives a
        parameter-free geometric reduction factor lambda.
    (G) Two electrons each modeled as a half-shell (hemisphere), their
        centroids antipodal.

For each, we minimize the predicted total energy and compare to the
experimental He binding energy (-2.9032 Ha).
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
from scipy.integrate import dblquad
from scipy.optimize import minimize

from core import log_experiment, minimize_shell_atom, T_alpha_over_R2
from constants import HARTREE_EV
from reference_data import TOTAL_BINDING_EV

HE_EXP_HA = -TOTAL_BINDING_EV["He"] / HARTREE_EV

# ---------------------------------------------------------------------------
# Helper: minimize an energy function E(r1, r2) over both radii
# ---------------------------------------------------------------------------

def minimize2(E_func, r0=(0.6, 1.2)):
    res = minimize(lambda x: E_func(x[0], x[1]) if x[0] > 1e-5 and x[1] > 1e-5
                   else 1e6, np.array(r0, dtype=float), method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 30000})
    return float(res.x[0]), float(res.x[1]), float(res.fun)


def minimize1(E_func, r0=0.6):
    res = minimize(lambda x: E_func(x[0]) if x[0] > 1e-5 else 1e6,
                   np.array([r0], dtype=float), method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-12})
    return float(res.x[0]), float(res.fun)


# ---------------------------------------------------------------------------
# Geometric reduction factor from TWO POINT ELECTRONS on PERPENDICULAR rings
# ---------------------------------------------------------------------------
# Electron 1 at (R cosA, R sinA, 0),   A uniform on [0, 2 pi]
# Electron 2 at (R cosB, 0, R sinB),   B uniform on [0, 2 pi]
# Distance:  R * sqrt(2 - 2 cosA cosB)
#
# Time-averaged repulsion = (1/(2 pi)^2) * int int dA dB / distance
#                         = (1/R) * lambda_perp
# where lambda_perp = (1/(2 pi)^2) * int int dA dB / sqrt(2 - 2 cosA cosB).
# This integral is diverges at (A,B) such that cosA=cosB=1 (same point),
# so physically we compute it as a principal value; easier: parameterize
# and integrate.
# ---------------------------------------------------------------------------
def lambda_perpendicular_rings() -> float:
    def integrand(A, B):
        d = math.sqrt(max(2.0 - 2.0 * math.cos(A) * math.cos(B), 1e-12))
        return 1.0 / d
    val, _ = dblquad(integrand, 0.0, 2 * math.pi, 0.0, 2 * math.pi,
                     epsabs=1e-6, epsrel=1e-6)
    return val / (4.0 * math.pi * math.pi)

# SAME-PLANE rings: two electrons on the same great circle at angular
# separation free => they will naturally settle at antipodes. Average
# repulsion is 1/(2R). lambda = 1/2.


def run() -> dict:
    Z = 2
    results = {}

    # ---------- (A) two shells, independent radii ----------
    def E_A(r1, r2):
        T = 0.5 / r1**2 + 0.5 / r2**2
        V = -Z / r1 - Z / r2
        U12 = 1.0 / max(r1, r2)
        return T + V + U12
    r1A, r2A, EA = minimize2(E_A, (0.5, 1.0))
    results["A_two_shells_indep"] = {
        "r1": r1A, "r2": r2A, "E": EA,
        "E_eV": EA * HARTREE_EV,
        "err_pct": 100.0 * abs(EA - HE_EXP_HA) / abs(HE_EXP_HA),
    }

    # ---------- (B) two shells same radius, lambda = 1 ----------
    def E_B(r):
        T = 2 * (0.5 / r**2)
        V = -2 * Z / r
        U = 1.0 / r
        return T + V + U
    rB, EB = minimize1(E_B, 0.7)
    results["B_two_shells_same_radius"] = {
        "r": rB, "E": EB,
        "E_eV": EB * HARTREE_EV,
        "err_pct": 100.0 * abs(EB - HE_EXP_HA) / abs(HE_EXP_HA),
    }

    # ---------- (D) antipodal point electrons on common sphere ----------
    def E_D(r):
        T = 2 * (0.5 / r**2)
        V = -2 * Z / r
        U = 1.0 / (2 * r)      # repulsion lambda = 1/2
        return T + V + U
    rD, ED = minimize1(E_D, 0.7)
    results["D_antipodal_points"] = {
        "r": rD, "E": ED, "E_eV": ED * HARTREE_EV,
        "err_pct": 100.0 * abs(ED - HE_EXP_HA) / abs(HE_EXP_HA),
        "lambda": 0.5,
    }

    # ---------- (E) 90-deg-apart point electrons ----------
    def E_E(r):
        T = 2 * (0.5 / r**2)
        V = -2 * Z / r
        U = 1.0 / (r * math.sqrt(2.0))   # lambda = 1/sqrt(2)
        return T + V + U
    rE, EE = minimize1(E_E, 0.7)
    results["E_90deg_points"] = {
        "r": rE, "E": EE, "E_eV": EE * HARTREE_EV,
        "err_pct": 100.0 * abs(EE - HE_EXP_HA) / abs(HE_EXP_HA),
        "lambda": 1.0 / math.sqrt(2.0),
    }

    # ---------- (F) perpendicular rings with uniform angle averaging ----------
    lam_perp = lambda_perpendicular_rings()
    def E_F(r):
        T = 2 * (0.5 / r**2)
        V = -2 * Z / r
        U = lam_perp / r
        return T + V + U
    rF, EF = minimize1(E_F, 0.7)
    results["F_perp_rings_uniform"] = {
        "r": rF, "E": EF, "E_eV": EF * HARTREE_EV,
        "err_pct": 100.0 * abs(EF - HE_EXP_HA) / abs(HE_EXP_HA),
        "lambda": lam_perp,
    }

    # ---------- (G) two half-shells / hemispheres at antipodes ----------
    # Each hemisphere carries charge -1 spread uniformly over 2 pi R^2.
    # Exact Coulomb energy between two such antipodal hemispheres is a
    # double surface integral; we compute it numerically.  By symmetry
    # the result is lambda / R for some dimensionless lambda we compute.
    #
    # Using azimuthal symmetry around the axis connecting the two hemi-
    # centroids, we parameterize a point on hemi 1 at polar angle theta1
    # from the +z axis (theta1 in [0, pi/2]) and azimuth phi1; on hemi 2
    # at theta2 in [pi/2, pi] and azimuth phi2.  Distance between two
    # unit-sphere points (r1=r2=R) is
    #   d = R * sqrt(2 - 2 (sin t1 sin t2 cos(phi1-phi2) + cos t1 cos t2))
    def integrand_halves(t1, t2, dphi):
        cos_ang = (math.sin(t1) * math.sin(t2) * math.cos(dphi)
                   + math.cos(t1) * math.cos(t2))
        d2 = max(2.0 - 2.0 * cos_ang, 1e-12)
        return (math.sin(t1) * math.sin(t2)) / math.sqrt(d2)

    # Triple integral: integrate over t1 in [0, pi/2], t2 in [pi/2, pi],
    # dphi in [0, 2 pi]; divide by (2 pi)^2 (area normalization per hemi).
    from scipy.integrate import tplquad
    val, _ = tplquad(
        lambda dphi, t2, t1: integrand_halves(t1, t2, dphi),
        0.0, math.pi / 2,
        lambda t1: math.pi / 2, lambda t1: math.pi,
        lambda t1, t2: 0.0, lambda t1, t2: 2 * math.pi,
        epsabs=1e-4, epsrel=1e-4,
    )
    # Each hemisphere surface area is 2 pi R^2 and charge -1 over that area,
    # so sigma = -1 / (2 pi R^2). Energy of two such charge distributions at
    # unit sphere is (1/(4pi eps0))* int (sigma1 sigma2 / d) dA1 dA2 which in
    # AU becomes:  U = sigma1 * sigma2 * R^4 * val / R = sigma1*sigma2*R^3*val.
    # With sigma = -1/(2 pi R^2):
    #   U = (1/(2 pi R^2))^2 * R^3 * val = val / (4 pi^2 R)
    # so lambda_hemispheres = val / (4 pi^2).
    lam_hemi = val / (4.0 * math.pi * math.pi)

    def E_G(r):
        T = 2 * (0.5 / r**2)
        V = -2 * Z / r
        U = lam_hemi / r
        return T + V + U
    rG, EG = minimize1(E_G, 0.7)
    results["G_antipodal_hemispheres"] = {
        "r": rG, "E": EG, "E_eV": EG * HARTREE_EV,
        "err_pct": 100.0 * abs(EG - HE_EXP_HA) / abs(HE_EXP_HA),
        "lambda": lam_hemi,
    }

    results["_experiment_He_Ha"] = HE_EXP_HA
    return results


if __name__ == "__main__":
    results = run()
    print("=" * 78)
    print("Experiment 006: helium - many shell/point configurations")
    print("=" * 78)
    print(f"Experimental He ground state:  E = {HE_EXP_HA:.4f} Ha"
          f" = {HE_EXP_HA*HARTREE_EV:.2f} eV")
    print()
    print(f"{'config':<36} {'r':>8} {'E (Ha)':>10} {'E (eV)':>10}"
          f" {'err%':>7} {'lambda':>8}")
    for key, v in results.items():
        if key.startswith("_"):
            continue
        r = v.get("r") or v.get("r1")
        lam = v.get("lambda", 1.0)
        if isinstance(v.get("r"), float):
            rshow = f"{v['r']:.4f}"
        else:
            rshow = f"{v.get('r1',0):.3f}/{v.get('r2',0):.3f}"
        print(f"{key:<36} {rshow:>8} {v['E']:>10.4f} {v['E_eV']:>10.2f}"
              f" {v['err_pct']:>6.2f}% {lam:>8.4f}")

    print()
    print("OBSERVATIONS")
    print("-" * 78)
    print("  - (A) lets the two shells split into an inner and outer layer,")
    print("    discovers screening automatically, and is moderately accurate.")
    print("  - (B) forces same-radius and is the WORST: full repulsion hurts.")
    print("  - (D) antipodal points: much better; repulsion halved.")
    print("  - (E/F) intermediate.")
    print("  - (G) antipodal hemispheres: computes a geometric lambda from the")
    print("    full Coulomb integral; no free parameter.")
    print()
    print("We'll pick the best geometric theory and carry it forward.")

    log_experiment({
        "exp": "006", "name": "helium_configurations",
        "results": results,
    })
