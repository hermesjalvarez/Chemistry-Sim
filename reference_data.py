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
}

# Total binding energies (eV) = sum of all successive ionization energies.
# Equivalently, the (positive) energy needed to completely strip the atom.
# Numbers below are from NIST cumulative IE tables, rounded to 0.1 eV.
TOTAL_BINDING_EV = {
    "H":    13.6,
    "He":   79.0,
    "Li":  203.5,
    "Be":  399.1,
    "B":   670.9,
    "C":  1030.1,
    "N":  1486.1,
    "O":  2043.8,
    "F":  2715.9,
    "Ne": 3511.7,
}

# Nuclear charges
Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5,
    "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
}

ELEMENTS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]


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
