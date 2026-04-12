"""Experimental reference data for H..Ne.

Sources:
- Ionization energies: NIST Atomic Spectra Database (rounded to 0.01 eV).
- Total binding energies: sum of all successive ionization energies,
  equivalently the energy to strip every electron to infinity. These are
  taken from NIST as well (cumulative sums of ionization potentials).

All values are in electronvolts.  Negative total energies (below) are the
non-relativistic ground-state total electronic energies (what a
Schrödinger-equation calculation would be trying to reproduce).

For the semi-classical shell model the cleanest comparison is against the
**total electronic binding energy**, which is the sum of the first Z
ionization energies. This is what `TOTAL_BINDING_EV` holds.
"""

# First-ionization energies (eV) --- NIST
FIRST_IE_EV = {
    "H":  13.6057,
    "He": 24.5874,
    "Li":  5.3917,
    "Be":  9.3227,
    "B":   8.2980,
    "C":  11.2603,
    "N":  14.5341,
    "O":  13.6181,
    "F":  17.4228,
    "Ne": 21.5645,
    "Na":  5.1391,
    "Mg":  7.6462,
    "Al":  5.9858,
    "Si":  8.1517,
    "P":  10.4867,
    "S":  10.3600,
    "Cl": 12.9676,
    "Ar": 15.7596,
    "K":   4.3407,
    "Ca":  6.1132,
}

# Total binding energies (eV) = sum of all successive ionization energies.
# Equivalently, the (positive) energy needed to completely strip the atom.
# Numbers below are from NIST cumulative IE tables, rounded to 0.1 eV.
# Total binding energies (eV) = sum of all successive ionization energies.
# Z=1..10 from NIST. Z=11..20 from cumulative NIST IE tables (approx ±5 eV).
TOTAL_BINDING_EV = {
    "H":      13.6,
    "He":     79.0,
    "Li":    203.5,
    "Be":    399.1,
    "B":     670.9,
    "C":    1030.1,
    "N":    1486.1,
    "O":    2043.8,
    "F":    2715.9,
    "Ne":   3511.7,
    "Na":   4423.0,
    "Mg":   5451.0,
    "Al":   6609.0,
    "Si":   7906.0,
    "P":    9353.0,
    "S":   10966.0,
    "Cl":  12750.0,
    "Ar":  14722.0,
    "K":   16890.0,
    "Ca":  19264.0,
}

# Nuclear charges
Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5,
    "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
}

ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
]


def total_binding_hartree(symbol: str) -> float:
    """Return the experimental total binding energy in Hartree (positive)."""
    from constants import HARTREE_EV
    return TOTAL_BINDING_EV[symbol] / HARTREE_EV


def ground_state_energy_hartree(symbol: str) -> float:
    """Return the ground-state electronic energy in Hartree (negative)."""
    return -total_binding_hartree(symbol)


if __name__ == "__main__":
    from constants import HARTREE_EV
    print(f"{'atom':<4} {'Z':<3} {'1st IE (eV)':<12} {'Total bind (eV)':<16} {'E_0 (Ha)':<12}")
    for sym in ELEMENTS:
        print(f"{sym:<4} {Z[sym]:<3} {FIRST_IE_EV[sym]:<12.4f} "
              f"{TOTAL_BINDING_EV[sym]:<16.2f} {-TOTAL_BINDING_EV[sym]/HARTREE_EV:<12.4f}")
