# Project index

## Hypotheses and experiments tested so far

| # | Experiment | Hypothesis | Verdict |
|---|---|---|---|
| 001 | `exp001_bare_static_shell.py` | Pure classical EM with static shell (no kinetic term) | FAIL: shell collapses for all Z (Earnshaw) |
| 002 | `exp002_rigid_rotating_shell.py` | Rigid rotation with free angular momentum L | PARTIAL: any L gives a stable atom; L=1 a.u. gives wrong constant for H; fit L=sqrt(2/3) matches all hydrogenic ions exactly |
| 003 | `exp003_alternative_geometries.py` | Ring, point orbit, solid ball, double shell as alternative electron geometries | Ring/point give H exactly at L=1; shell needs L=sqrt(2/3); solid ball needs L=sqrt(2/5) |
| 004 | `exp004_alternative_stabilizers.py` | Surface tension (R^2), polytropic pressure (1/R^n, n≠2), breathing modes | FAIL for all except n=2: only 1/R^2 barriers give correct Z^2 scaling |
| 005 | `exp005_hydrogenic_series.py` | Settled 1/R^2 shell model applied to all hydrogenic ions Z=1..10 | PERFECT: reproduces Z^2 Rydberg law exactly |
| 006 | `exp006_helium_configurations.py` | Helium with many electron-pair geometries | First survey; antipodal points overshoot, shell-theorem screening undershoots |
| 007 | `exp007_helium_many_branches.py` | 10+ theories for the helium same-shell problem | Branch D (120-deg, λ=1/√3, zero fit): 0.88% error. Several fit-based branches match exactly |
| 008 | `exp008_lithium_branches.py` | Lithium with multiple structural hypotheses | L2 (2 paired + 1 outer) with He-fit λ: 4.45%. L1 (all three in one shell) dramatically overbinds — pure classical energetics prefers trigonal, so we need an occupancy rule |
| 009 | `exp009_principal_shells_BeNe.py` | Postulate L = n·ℏ per shell + shell capacity 2n² + Thomson polyhedra | **Best result so far**: all 10 atoms within 5.5%, C nails it (0.12%), no fitting |

## Running every experiment

```
python3 experiments/exp001_bare_static_shell.py
...
python3 experiments/exp009_principal_shells_BeNe.py
```

or all at once:

```
python3 run_all.py
```

## Results log

Machine-readable JSONL: `results/log.jsonl`.
Human-readable log: `wiki/log.md`.
