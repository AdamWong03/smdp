# NMI research plan: one mechanism, accountable evidence

Date: 2026-11-13. Status: research decision proposal, **not** a newly frozen
architecture/training protocol. No new training or MD production was launched
while preparing this plan. Preserve all previous code, weights, negative
results and reference data. The old manuscript remains a historical draft.

## 1. Objective and ownership

Target a substantial original research Article at Nature Machine Intelligence.
Acceptance cannot be guaranteed. The target determines the quality of the
question, evidence and communication; it does not predetermine positive results.
Operational responsibility means selecting a narrow question, implementing it
correctly, controlling expenditure and reporting failures, rather than producing
more architectural names or post-hoc explanations.

NMI states that editorial assessment considers advance in understanding,
soundness, evidential support and relevance to its readership. Its criteria are
not a fixed requirement for a particular number of modules, systems or seeds.
The experiment gates below are our project decisions, not journal rules.

## 2. Evidence we actually have

- Quartz R3-Opt reproduces useful NVT statistics in two long autonomous runs,
  with residual structure/energy errors. It does not strictly conserve energy.
- Water R3-Opt has small endpoint errors but severe NVT motion/structure/
  diffusion errors. This is the most consequential unresolved application gap.
- The light R1 has lower endpoint errors but worse Quartz autonomous physics:
  lower regression error alone is not an adequate selection principle.
- Stable residual reparameterization, multiple motion modes, two-step training,
  and conditional-successor aggregation have not established a robust remedy.
- The latest conditional-successor comparison completed eight 25k-update arms.
  It is a negative result for that fixed protocol, not proof that all dataset
  aggregation or all endpoint supervision is ineffective.
- There is still no matched, successfully trained mature direct propagator on
  our own Water split. Historical official checkpoint numbers do not provide
  this control. The necessity of learned future-relation feedback also remains
  unestablished.

Sources: `RESULTS_REGISTRY.md`,
`../reports/dynamics_diagnosis_20261109/DIAGNOSIS.md`, and
`../reports/conditional_transition_20261109/RESULTS.md`. Do not average their
different initial conditions, horizons, training budgets or endpoint/test scopes.

## 3. A factual correction about spatial information

On 32 evenly spaced **training transition** indices (0 to 4999, inclusive),
Vesin radius graphs give the following averages. Reach is the fraction of all
other atoms connected by at most the stated number of unweighted graph hops.
These are topology measurements, not trained-model influence or expressivity.

| System | Radius (Å) | Directed edges / atom | 1-hop reach | 2-hop reach | 4-hop reach |
|---|---:|---:|---:|---:|---:|
| Water | 4.5 | 37.7907 | 0.197857 | 0.883854 | 1.000000 |
| Water | 6.0 | 89.8464 | 0.470400 | 1.000000 | 1.000000 |
| Quartz | 4.5 | 28.6964 | 0.178238 | 0.812200 | 1.000000 |
| Quartz | 6.0 | 64.6682 | 0.385146 | 0.999468 | 1.000000 |

Thus, the observed four-layer small-cell graphs are not strictly isolated from
atoms outside the first radius. A larger cutoff changes direct geometrical
relations, path lengths, aggregation and computational cost. It may improve
learning, but the claim that 4.5 Å alone makes the required state unobservable
has not been demonstrated. Reachability is equally not proof of effective use:
messages still pass through finite-dimensional aggregated representations.
These small-cell measurements must not be extrapolated to large supercells.

Reproduce this read-only measurement in the existing Python environment:

```python
import h5py, numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from periodic import PeriodicGraphBuilder
from pathlib import Path
root = Path('/data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1')
for system, h in [('water', 5), ('quartz', 30)]:
    with h5py.File(root/system/'transitions'/f'{h}fs/train.h5') as f:
        ids = np.linspace(0, 4999, 32, dtype=int)
        qs, cells = f['positions_initial_A'][ids], f['cell_A'][ids]
    for radius in [4.5, 6.0]:
        degrees, reach = [], []
        for q, cell in zip(qs, cells):
            edges, _ = PeriodicGraphBuilder(radius).one(q, cell)
            n = len(q)
            graph = csr_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(n,n))
            distances = shortest_path(graph, directed=False, unweighted=True)
            degrees.append(edges.shape[1]/n)
            reach.append([np.sum((distances <= k)&(distances > 0))/(n*(n-1)) for k in [1,2,4]])
        print(system, radius, np.mean(degrees), np.mean(reach, axis=0))
```

## 4. Paper question and limits of novelty

The principal candidate question is:

> Does making predicted relative motion an explicit evolving condition of
> neural interaction processing improve finite-time atomistic propagation
> at comparable computational cost, including autonomous physical statistics?

This is a computational representation hypothesis. Predicted future relations
are a deterministic function of current input and introduce no new observations.
Their possible benefit is a more effective computation of the same transition.
Current-only models are not theoretically incapable of predicting the future.

Three levels must remain distinct:

1. Adequate base representation: symmetry, neighborhood information, velocity
   encoding, numerical precision and optimization. Necessary engineering.
2. Proposed mechanism: persistent interaction processing conditioned on the
   model's own motion estimate. A hypothesis requiring matched controls.
3. Useful outcome: accurate trajectories/statistics at a measured cost. Neither
   endpoint accuracy nor a sophisticated architecture implies this outcome.

Do not claim relative velocity inputs, l=2 tensors, neighbor attention,
predictor-corrector organization, large strides, or force-free inference as new
by themselves. MDNet, TrajCast, ChronoCast, MDtrajNet and geometric GNNs already
occupy substantial parts of this space. Exact overlap with iterative geometric
networks must be documented before a method contribution is frozen.

If the future-relation mechanism has no independent value, the old DSRT paper
claim is not protected. Retain its results and revise the scientific question;
do not add experts or losses merely to preserve a method name.

## 5. One architecture candidate, not a component collection

Use one coherent equivariant interaction family as the base. Compare complete
information paths rather than borrowing a mixing layer from every paper.
The candidate computational outline is:

```
current phase-space relations
    -> equivariant interaction processing
    -> provisional relative-motion estimate
    -> update relation conditions, retaining learned interaction state
    -> equivariant refinement
    -> one future q/v state
```

This outline is close to the existing DSRT hypothesis, intentionally. No new
name or claimed discovery is assigned to it. The implementation must specify
exactly what the relation update changes beyond an ordinary deeper network.
The two-stage requirement itself is revisable if a simpler computation wins.

Candidate base operators are evaluated on relevant coupling paths, numerical
behavior, edge/node activation shapes and measured cost. A high-order path
needs explicit coupling back into the output vectors, not just l=2 storage.
Do not combine full CG message passing, GotenNet refinement, Equiformer S2
nonlinearity and EFA global attention into one first candidate.

Keep direct neural endpoint inference, variable atom count, per-graph momentum
handling and periodic image consistency. No inference force/energy model,
Hessian/mode propagator, physical substep integrator or externally corrected
trajectory is introduced by this plan. Dense internal trajectories are not a
mandatory first capability. Thermostats remain external and fully specified.

## 6. Execution stages and decisions

### A. Establish a successful comparison on the actual task

Start with Water, where the material discrepancy is largest. The first formal
comparisons should be: native TrajCast on our 5K split, and unchanged R3-Opt
with Water radius 6 Å, trained from random initialization. Keep the historical
4.5 Å R3-Opt results intact. This is not evidence of the new mechanism yet.

Before training, freeze the actual optimizer budgets and fair comparison view.
Use the mature baseline's appropriate native training, not an arbitrary
optimizer transplantation. Record differences in updates, target evaluations,
precision, compute and validation-based tuning. If budgets differ, report them;
do not label the experiment perfectly matched. Our new architecture/ablation
must have matched training schedules and data.

Reuse existing verified reference MD and source splits. Necessary geometry,
units, gradients and output checks are part of development, not a separate
small-data fitting experiment. Do not touch test data for model selection.

Decision:
- If a mature model also fails on our protocol, resolve the task/data/protocol
  discrepancy before selling a new architecture as a solution.
- If 6 Å alone fixes DSRT Water, radius is an important engineering result;
  it does not by itself create an NMI method contribution.
- If the mature model succeeds and DSRT remains poor, the representation/
  optimization gap becomes experimentally grounded rather than speculative.

### B. Test one central mechanism on an adequate base

After choosing one coherent base interaction operator, freeze one candidate
and its same-budget **current-only** control. Both retain the same depth,
readouts, persistent relation state, parameter scale and training labels. Only
the access to predicted future relation conditions differs. If the full model
has extra auxiliary heads/supervision, retain them in the control.

Use full formal training, random initialization, endpoint q/v supervision and
validation selection. No simultaneous on-policy data, rollout training,
physical penalties or thermostat change. Start on Water; only promote to the
two-system study if it predicts and simulates usefully. Keep Quartz as a guard
against a Water-specific gain obtained at the expense of prior capability.

Decision:
- Better endpoint error with wrong motion statistics is insufficient.
- A difference within seed/initial-state variation is insufficient.
- No benefit over current-only means no future-feedback mechanism claim.
- An improvement bought by more computation must be compared on an accuracy/
  runtime frontier, not called an efficiency advance by parameter count.

These stages are not yet an authorization to launch unspecified new runs.
Concrete architecture and resource configs must first be reviewable and fixed.

### C. Assemble paper evidence only after the mechanism survives

1. Repeat the decisive full/control comparison with three training seeds;
   retain paired initial states, per-run outputs and effect distributions.
2. Evaluate Quartz and flexible Water with fixed checkpoint selection and
   matched-reference NVE/NVT protocols. Use temperature, molecular/element
   kinetic partition, raw potential distribution, RDF, VACF/VDOS, Water MSD/
   diffusion, failures and measured production throughput.
3. Test dependence on stride, rather than only one favorable stride per
   material: choose a small predeclared set from existing high-resolution MD.
   The expected mechanism need not produce a monotonic effect; report failures.
4. Test one meaningful size and temperature transfer using independently
   prepared states. Do not spend first on another set of long replicas of a
   physically failing model.
5. If claiming broad atomistic applicability, add a held-out material class
   or molecular system after defining its data and budget. Quartz/Water alone
   can establish a focused result, not an automatic universal claim or an
   automatic NMI qualification.
6. External comparisons need at least a genuinely run close direct method;
   train/predict cost and any thermostat/energy correction must be visible.
   Do not substitute published scores on other data for our own runs.

Before C, freeze tolerances/effect measures against reference block variability
and seed variation. Do not invent numerical acceptance thresholds retrospectively
or claim a small p-value from correlated MD frames. NVE drift is an essential
limitation metric; exact 100 ps conservation is not the definition of success
already established by every direct MD paper.

## 7. What would support the submission

The intended evidence chain is an important limitation, one justified
computational mechanism, reproducible improvement over a controlled alternative,
and a useful physical/computational consequence beyond one state point.

The possible contribution is a result about how to compute learned finite-time
interactions, supported by a practical propagator. It is not a claim that all
neural MD failures share a cause, nor that adding high-order tensors is novel.

If only basic parity with TrajCast is achieved after copying its configurations,
report a successful engineering milestone, not a completed NMI paper. If the
core controlled gain is absent, stop that contribution path instead of escalating
module count. Existing negative evidence can inform a separate analysis, but
one failed model family alone does not establish a field-wide limitation.

## 8. Deliverables and expenditure discipline

Next deliverable: a code-level candidate specification with one primary
interaction operator, exact irreps/couplings, velocity treatment, q/v readout,
graph/image conventions, parameter and activation accounting, baseline configs,
and the corresponding control. No speculative final title/positive abstract.

Track experiment state separately as planned, running, numerically failed,
training completed, trajectory completed and analysis completed. Each new
scientific decision records the preceding evidence, budget and stop rule.
Keep old experiments unchanged. Do not launch all horizons, controls, seeds,
supercells and long trajectories before the first decisive comparison.

The current manuscript must not continue to say Water is merely pending or
future feedback is already supported. It is retained for historical context;
update the main narrative only when the new mechanism has evidence.

## Verified sources

- NMI editorial process: https://www.nature.com/natmachintell/submission-guidelines/editorial-process
- NMI reviewer criteria: https://www.nature.com/natmachintell/for-reviewers/writing-your-report
- Multi-ACE/design space (NMI 2025): https://doi.org/10.1038/s42256-024-00956-x
  Full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC11769842/
- Euclidean Fast Attention (NMI 2026): https://doi.org/10.1038/s42256-026-01195-y
  Code: https://github.com/thorben-frank/euclidean_fast_attention
- TrajCast (NMI 2026): https://doi.org/10.1038/s42256-026-01227-7
  Code: https://github.com/IBM/trajcast
- ChronoCast (PRE 2026): https://doi.org/10.1103/ccl7-lx2b
  Authors' research summary: https://nanophononics.tongji.edu.cn/show/173
- MDNet: https://arxiv.org/abs/2111.15176
- MDtrajNet (JCTC 2026): https://doi.org/10.1021/acs.jctc.5c01689
  Code: https://github.com/dralgroup/mlatom/blob/main/mlatom/EEEDDDD.py
- GotenNet (ICLR 2025): https://proceedings.iclr.cc/paper_files/paper/2025/hash/64d4ff4fff788cdffe236f9ce8b09400-Abstract-Conference.html
  Code: https://github.com/sarpaykent/GotenNet/blob/main/gotennet/models/representation/gotennet.py

NMI publication status and abstracts/design-space full text were verified.
The APS full PDF could not be retrieved in this session; do not represent its
complete code/method as audited solely from the abstract or an indexed excerpt.
