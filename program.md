# Program: Semi-Classical Shell Model of the Atom

This project follows the "autoresearch" philosophy: a single AI agent iteratively
proposes, tests, and documents physical hypotheses in an attempt to reproduce the
first 10 atoms of the periodic table from first principles.

## Core Assumption (starting point)

- The nucleus is a point charge `+Ze` at the origin.
- Each electron is modeled as a **thin spherical shell** of radius `R`,
  total charge `-e`, total mass `m_e`.
- Only classical mechanics and classical electromagnetism (Coulomb / Gauss /
  shell theorem) are available a priori. Whenever that is not enough, a new
  assumption must be explicitly added and justified.

## Units

All calculations use **Hartree atomic units**:
`ħ = m_e = e = 4πε₀ = 1`.
- Length unit: Bohr radius `a₀ ≈ 52.918 pm`.
- Energy unit: Hartree `Ha ≈ 27.2114 eV`.

## Workflow

1. **Hypothesis**: stated in the experiment file's docstring.
2. **Derivation**: closed-form expressions in the docstring / comments.
3. **Code**: a `run()` function returns a dict of predictions.
4. **Comparison**: predictions are compared against `reference_data.py`.
5. **Logging**: every experiment appends a row to `results/log.jsonl` and
   updates `wiki/log.md`.
6. **Lessons**: after each experiment, a one-paragraph lesson is added to
   `wiki/lessons.md`.

## Success Criteria

- Tier 1: reproduce hydrogen exactly (`R = a₀`, `E = -0.5 Ha = -13.6 eV`).
- Tier 2: match helium total binding energy within 5% (`-2.903 Ha`).
- Tier 3: reproduce the Li/Be/... total binding energies within ~10% with
  a **single** consistent theory (same parameters, no per-atom refitting).
- Stretch: get all of H → Ne within 10% and explain the noble-gas / alkali
  pattern from model structure alone.

## Rules for Adding Hypotheses

- Each new assumption must be the **smallest** extension needed.
- The new assumption must be explicit, stated in plain English in the
  experiment docstring.
- If a new assumption breaks an earlier success, that must be reported in
  `wiki/lessons.md`, not swept under the rug.
- Parameter-fitting is allowed, but must be clearly flagged, and any fitted
  constant must be tested on at least one *different* atom before being
  trusted.

## File Layout

```
Chemistry-Sim/
├── program.md                 # this file
├── constants.py               # physical constants (both SI and atomic units)
├── reference_data.py          # experimental energies for H..Ne
├── core.py                    # shell-model primitives (shell theorem, etc.)
├── run_all.py                 # runs every experiment and updates the wiki
├── experiments/               # one file per hypothesis
│   ├── exp001_bare_classical_shell.py
│   ├── exp002_no_self_repulsion.py
│   ├── exp003_uncertainty_barrier_H.py
│   ├── ...
├── results/
│   └── log.jsonl              # append-only machine-readable log
└── wiki/
    ├── index.md               # catalog of experiments
    ├── log.md                 # human-readable run log
    ├── hypotheses.md          # every hypothesis considered
    └── lessons.md             # what each experiment taught us
```
