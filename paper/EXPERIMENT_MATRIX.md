# Frozen experiment matrix

## Central hypothesis

For a fixed-capacity finite-time atomic transition network, conditioning the
refinement stage on relations computed from a learned future-state proposal
improves endpoint and autonomous dynamics relative to using only current
relations or ballistic future relations.

## Execution order

| Stage | Experiment | Model selection | Primary evidence |
|---|---|---|---|
| A | Freeze Quartz evidence; paper draft; R1 independent evaluation | Existing validation selections | Complete: endpoint test and two 100 ps NVE runs |
| B | Water, 64 flexible molecules, 5 fs | Endpoint training/test complete | Multistep + 100 ps NVE complete; substantial heating; full physical analysis/NVT pending |
| C | Current-Only and Ballistic-Geometry controls on Quartz and Water | Same split and validation rule as Full | Mechanism benefit versus geometric change |
| D | Quartz repeats, 250/350 K, 1,296/4,374 atoms | No reselection of frozen 300 K checkpoint | Repeatability, transfer limits, throughput |
| E | Diverse 25K Quartz training set | Same validation set and matched optimizer updates | Data efficiency and physical-statistics gain |

## Matched mechanism controls

All three models keep the two-stage depth, width, heads, training data,
optimizer schedule and endpoint objectives fixed.

| Variant | Refinement relation |
|---|---|
| Full DSRT | Current relation plus relation from the learned endpoint proposal |
| Current-Only | Current relation reused in the refinement stage |
| Ballistic-Geometry | Relation from `R_t + h V_t` and ballistic velocity |

The control implementation must not remove the proposal blocks or correction
head. Test and physical trajectories cannot select checkpoints.

## Fixed physical protocols

- Quartz: 162 atoms, 30 fs transition, 300 K; 5 × 100 ps NVE and 3 ×
  (1.5 ns thermalization + 15 ns NVT production).
- Water: 64 flexible molecules, 5 fs transition; 3 independent 2.5 ns NVT
  production trajectories plus independent short NVE trajectories.
- Quartz temperature transfer: frozen 300 K checkpoint on independently
  equilibrated 250 and 350 K states.
- Size transfer: independently thermalized 1,296- and approximately 4,374-atom
  cells; no replicated velocity field presented as independent transfer.
- VDOS comparisons use identical sampling and common resolvable bandwidth.
- Diffusion fitting windows and block lengths are frozen before inspecting
  model results.

## Stop and interpretation rules

- If Full does not beat Current-Only reproducibly, future geometry is not a
  supported core mechanism claim.
- If endpoint error improves but physical statistics do not, report the split
  result rather than redefining success.
- Failed temperature or size transfer defines the empirical scope and does not
  trigger architecture changes in this paper cycle.
