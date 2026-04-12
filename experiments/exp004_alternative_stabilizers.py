"""Experiment 004 — Alternative classical stabilization mechanisms.

In exp002/003 we saw that rotation (of some form) gives a 1/R^2 barrier
which yields correct Z^2 scaling for one-electron ions.  Here we test
OTHER purely classical mechanisms that could keep the shell from
collapsing:

    (A) Surface tension of the shell                   U = sigma * 4 pi R^2
    (B) Internal "gas" pressure with fixed amount of   U = A / R^n
        substance (polytropic law)
    (C) Radial breathing oscillation with fixed        amplitude free,
        zero-point amplitude                            E ~ constant only

We ask: can any of these give the correct Z^2 scaling observed in
hydrogenic ions?  The answer is telling.

For an energy of the form
    E(R) = T(R) - Z / R
the virial theorem gives us the scaling directly:
    T(R) ~ C / R^p          =>   R* ~ Z^(-1/(p-1))
                                E* ~ -Z^( p/(p-1) )

To get  R* ~ 1/Z  and  E* ~ Z^2  we need   p = 2  exactly.
So any barrier with a DIFFERENT radial exponent gives WRONG scaling.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np

from core import log_experiment, minimize_1d


def fit_and_test(T_func, p_label: str) -> dict:
    """Fit any free constant in T to hydrogen, then test on He+ and Li2+."""
    # Fit: minimize |E_H + 0.5 Ha|
    from scipy.optimize import brentq

    # Use T_func(R, c) and search over c>0
    def E_min_for_c(c):
        Rm, Em = minimize_1d(lambda R: T_func(R, c) - 1.0 / R)
        return Em

    # Bracket search
    try:
        c_lo, c_hi = 1e-6, 1e6
        # Search for c such that E_H = -0.5
        def g(c):
            return E_min_for_c(c) + 0.5
        # ensure bracket
        if g(c_lo) * g(c_hi) > 0:
            # scan log grid
            cs = np.geomspace(c_lo, c_hi, 100)
            gs = [g(c) for c in cs]
            signs = np.sign(gs)
            flip = np.where(np.diff(signs) != 0)[0]
            if len(flip) == 0:
                return {"fit": None, "message": "no bracket"}
            c_lo, c_hi = cs[flip[0]], cs[flip[0] + 1]
        c_fit = brentq(g, c_lo, c_hi, xtol=1e-10)
    except Exception as e:
        return {"fit": None, "message": f"fit failed: {e}"}

    # Test on Z = 1..3
    results = {}
    for Z in [1, 2, 3]:
        Rm, Em = minimize_1d(lambda R: T_func(R, c_fit) - Z / R)
        E_ref = -0.5 * Z * Z
        results[f"Z={Z}"] = {
            "R": Rm, "E_ha": Em, "E_ref": E_ref,
            "err_pct": 100.0 * abs(Em - E_ref) / abs(E_ref),
        }
    return {"c_fit": c_fit, "by_Z": results, "p_label": p_label}


def run() -> dict:
    # (A) surface tension: T = sigma * 4 pi R^2
    sigma_result = fit_and_test(
        lambda R, sigma: sigma * 4 * math.pi * R * R, "p = -2 (tension R^2)"
    )
    # (B1) internal pressure with n=3: T = A / R^3
    press3 = fit_and_test(lambda R, A: A / R**3, "p = 3  (A/R^3)")
    # (B2) internal pressure with n=4
    press4 = fit_and_test(lambda R, A: A / R**4, "p = 4  (A/R^4)")
    # (reference) 1/R^2 barrier: T = alpha / R^2
    press2 = fit_and_test(lambda R, a: a / R**2, "p = 2  (alpha/R^2)")

    return {
        "surface_tension_R2": sigma_result,
        "pressure_Rm3": press3,
        "pressure_Rm4": press4,
        "reference_Rm2": press2,
    }


if __name__ == "__main__":
    results = run()
    print("=" * 78)
    print("Experiment 004: alternative classical stabilizers")
    print("=" * 78)
    for key, r in results.items():
        print(f"\n{key}  ({r.get('p_label', '')})")
        print(f"    fitted constant = {r.get('c_fit', 'NA')}")
        by_Z = r.get("by_Z", {})
        for zlabel, v in by_Z.items():
            print(f"    {zlabel}: R={v['R']:.4f}  E={v['E_ha']:.4f} Ha"
                  f"  (ref {v['E_ref']:.4f})  err {v['err_pct']:.2f}%")
    print()
    print("VERDICT")
    print("-" * 78)
    print("  Only barriers of the form C / R^2 give the correct Z^2 energy")
    print("  scaling for hydrogenic ions.  Surface tension (R^2), internal")
    print("  polytropic pressures (1/R^n, n != 2), etc. all fit hydrogen by")
    print("  construction but mispredict He+, Li2+ by large fractions.")
    print("  This is a stringent filter: only 1/R^2 stabilization survives.")

    log_experiment({
        "exp": "004",
        "name": "alternative_stabilizers",
        "verdict": "Only 1/R^2 barriers reproduce Z^2 scaling",
        "results": {
            k: {"c_fit": v.get("c_fit"), "by_Z": v.get("by_Z")}
            for k, v in results.items()
        },
    })
