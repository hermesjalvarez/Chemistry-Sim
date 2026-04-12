# Hypotheses tried (keep every one alive unless decisively falsified)

## Hypothesis family H1: how to keep a charged shell from collapsing

- **H1.0** Pure static shell (no kinetic).  Status: **FAILS** (Earnshaw).
- **H1.1** Shell rigidly rotates; angular momentum L is a free parameter.  Status: stable but underdetermined.
- **H1.2** Shell rigidly rotates with L = "one unit of action" a priori.  Status: **FAILS** for hydrogen (gives -9.07 eV instead of -13.6).
- **H1.3** Replace shell by rigid ring / point on orbit; L = 1 a.u.  Status: PASSES for hydrogen exactly.
- **H1.4** Solid ball, rigid rotation, L = 1.  Status: **FAILS** (gives -5.4 eV for H).
- **H1.5** Shell with surface tension σ.  Status: **FAILS** (doesn't stabilize; force all inward).
- **H1.6** Shell with polytropic pressure U ∝ 1/R^n, n ≠ 2.  Status: **FAILS** Z² scaling.
- **H1.7** Shell with empirical barrier T = α/R², α = 1/2.  Status: **PASSES** one-electron ions perfectly and is the minimum-parameter working theory.
- **H1.8** Shell with Bohr-type postulate L = n·ℏ, non-rigid internal motion, T_n = n²/(2R²).  Status: **PASSES** for hydrogen (n=1 case coincides with H1.7), and generalizes naturally to multi-shell atoms.

## Hypothesis family H2: how to model electron-electron correlation on a shared shell

- **H2.0** No correlation, two concentric shells at independent radii.  Status: gives 14% error on He (undershoots).
- **H2.1** Same radius, full repulsion (λ=1).  Status: 22% error on He.
- **H2.2** Same radius, point electrons at antipodes (λ=1/2).  Status: 5.5% (overshoots).
- **H2.3** Same radius, point electrons at 120° (λ=1/√3 ≈ 0.577).  Status: 0.88% error on He with ZERO fit parameters.
- **H2.4** Variational effective nuclear charge Z_eff.  Status: fits exactly, one free parameter, equivalent to H2.5.
- **H2.5** Attractive pairing term -K/r on same shell.  Status: fits exactly, equivalent to H2.4.
- **H2.6** Kinetic discount γ on same-shell pair.  Status: fits exactly, different math.
- **H2.7** Hard-avoidance radius a_c.  Status: **FAILS** (can't lower energy enough).
- **H2.8** Thomson polyhedra (k electrons on sphere arranged at classical min).  Status: generalizes H2.3 to k>2; **used in exp009 for best first-row result**.

## Hypothesis family H3: how to handle shell occupancy

- **H3.0** No rule; all electrons fall into the innermost shell.  Status: **FAILS** for Li — classical minimum disagrees with experiment.
- **H3.1** "At most 2 electrons per shell" (Pauli-like rule).  Status: needed to kick Li's third electron out, but motivated only empirically.
- **H3.2** "Shell n holds at most 2n² electrons" (periodic-table row structure).  Status: used in exp009, **matches first row to ~5%**.

## Hypothesis family H4: kinetic term scaling with n

- **H4.0** Every shell has the same barrier T = 1/(2R²).  Status: partial; works for one-electron atoms and leads to tight inner shell for heavy atoms, but can't represent large outer orbits.
- **H4.1** Shell n has barrier T = n²/(2R²).  Status: **used in exp009**, the key that makes the outer shells large enough to match real atomic radii.

## Open / not yet tried
- H5.1 Derivation of the n² kinetic scaling from a classical internal-motion model.
- H5.2 Magnetic-dipole interaction between shells (corrections of order α² ~ 10^-4 Ha).
- H5.3 Subshell splitting: n=2 split into (s,p) with different radii.
- H5.4 Relaxing Thomson polyhedra to allow distortion in heavy atoms.
