"""Physical constants used throughout the project.

All experiments work in Hartree atomic units (ħ = m_e = e = 4πε₀ = 1).
SI constants are kept here only for conversion and sanity checks.
"""

import math

# --- SI constants (CODATA 2018) -----------------------------------------
HBAR_SI   = 1.054_571_817e-34          # J·s
ME_SI     = 9.109_383_7015e-31          # kg
E_SI      = 1.602_176_634e-19           # C
EPS0_SI   = 8.854_187_8128e-12          # F/m
K_E_SI    = 1.0 / (4.0 * math.pi * EPS0_SI)  # N·m²/C²
C_SI      = 2.997_924_58e8              # m/s

# --- Derived atomic-unit scales (in SI) --------------------------------
BOHR_M    = HBAR_SI**2 / (ME_SI * K_E_SI * E_SI**2)     # ≈ 5.291e-11 m
HARTREE_J = K_E_SI * E_SI**2 / BOHR_M                     # ≈ 4.3597e-18 J
HARTREE_EV = HARTREE_J / E_SI                             # ≈ 27.2114 eV
BOHR_PM   = BOHR_M * 1e12                                 # ≈ 52.918 pm

# --- Conversions ---
def hartree_to_eV(E_ha: float) -> float:
    return E_ha * HARTREE_EV

def eV_to_hartree(E_eV: float) -> float:
    return E_eV / HARTREE_EV

def bohr_to_pm(R_bohr: float) -> float:
    return R_bohr * BOHR_PM

def pm_to_bohr(R_pm: float) -> float:
    return R_pm / BOHR_PM


if __name__ == "__main__":
    print(f"Bohr radius     : {BOHR_PM:.4f} pm")
    print(f"Hartree energy  : {HARTREE_EV:.4f} eV")
    print(f"1 Ha = 2 Ry     : {HARTREE_EV/2:.4f} eV   (Rydberg)")
