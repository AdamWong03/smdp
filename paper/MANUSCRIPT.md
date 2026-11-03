# Beyond Current-State Geometry: Future-Conditioned Relational Refinement for Learned Molecular Dynamics

> Working manuscript. Numerical statements are restricted to completed
> artifacts in `RESULTS_REGISTRY.md`. Bracketed result placeholders are frozen
> experiments, not projected outcomes.

## Abstract

Direct neural propagation offers an alternative to repeatedly evaluating
interatomic forces and integrating molecular dynamics (MD) with small time
steps. Existing direct propagators can learn finite-time updates from the
current atomic state, but must internally infer how pair geometry changes over
the prediction interval. We study whether making this intermediate estimate
explicit improves learned state transitions. We introduce the Dual-State
Relational Transition network (DSRT), an equivariant finite-horizon propagator
that first proposes a future phase-space state, constructs predicted future
relations on the current periodic neighbor graph, and uses both current and
predicted relations to refine the endpoint. DSRT does not evaluate forces or
energies during propagation. On 162-atom alpha-quartz with a 30 fs transition,
the frozen 2.36M-parameter model obtains test root-mean-square errors of
0.00180 Å in position and 0.000122 Å fs^-1 in velocity. Autonomous 100 ps NVE
trajectories remain structurally stable but exhibit measurable reference-energy
drift. A 15 ns NVT production trajectory reproduces low-frequency element-wise
vibrational spectra with overlap scores of 0.946–0.950 and runs at 106 ns/day on
one NVIDIA L40S, while retaining systematic errors in the Si–O radial
distribution and potential-energy variance. The remaining study tests the
mechanism with matched current-only and ballistic-geometry controls and extends
the same architecture to flexible liquid water. These experiments distinguish
endpoint accuracy, autonomous stability and equilibrium statistical fidelity,
and determine whether predicted future geometry—not merely two-stage network
capacity—accounts for any observed gain.

## 1. Introduction

Molecular dynamics resolves atomistic motion by repeatedly evaluating
interactions and advancing the state with a numerical integrator. Machine-
learned interatomic potentials can reduce the cost of an individual force
evaluation, but they retain the sequential small-step integration procedure.
Direct state propagators instead learn a finite-time map from positions,
velocities and chemical identity to a future phase-space state. Recent work has
shown that equivariant autoregressive models can generate long trajectories at
prediction intervals one or more orders of magnitude larger than the reference
integration step. This capability is established by methods such as TrajCast,
ChronoCast and MDtrajNet; it is therefore the context, not the novelty, of this
work.

Let the state at time (t) be

\[
X_t = (R_t,V_t,Z,m,\mathcal C),
\]

where (R_t) and (V_t) are atomic positions and velocities, (Z) and (m)
are species and masses, and (\mathcal C) is the periodic cell. A typical
direct predictor constructs a graph from the current configuration, propagates
features on that graph, and decodes (X_{t+h}). During a finite interval (h),
however, relative positions and velocities change. A sufficiently expressive
current-state network can in principle infer these changes implicitly. Our
hypothesis is narrower: explicitly exposing the model's own estimate of future
pair geometry to a refinement stage may make the finite-time transition easier
to learn and may improve autonomous propagation.

The true future state is unavailable at inference. DSRT therefore first
predicts a proposal (\widehat X^p_{t+h}), converts it into predicted pair
relations (\widehat{\mathcal R}^{p}_{t+h}), and refines the endpoint using
both current and proposed relations:

\[
\mathcal R_t \rightarrow \widehat X^p_{t+h}
\rightarrow \widehat{\mathcal R}^{p}_{t+h}
\rightarrow \widehat X_{t+h}.
\]

This is a deterministic, end-to-end neural transition map rather than a
learned force field or physical substep integrator. The proposal and refinement
operate on scalar, vector and persistent edge features. Translation,
orthogonal-coordinate covariance, atom permutation and per-system center-of-
mass handling are built into the representation and decoder. The architecture
does not enforce total-energy conservation.

The paper tests three levels of evidence. First, does future-conditioned
refinement improve endpoint and short autonomous prediction under a matched
model budget? Second, does any improvement persist in NVE and NVT structural,
dynamical and energetic observables? Third, does the same architecture work for
both crystalline quartz and flexible liquid water and transfer across system
size and temperature? The central mechanism is tested against two-stage
current-only and ballistic-geometry controls, so that an improvement cannot be
attributed merely to extra layers or a correction head.

## 2. Related work

### 2.1 Force-based and direct atomistic propagation

Machine-learned interatomic potentials approximate energies and forces while
retaining conventional integration. Direct propagators solve a different
learning problem: they approximate a finite-time state transition and need not
evaluate a position gradient during rollout. TrajCast uses equivariant message
passing to predict atomic displacement and future velocity and evaluates long
rollouts through structure, energy distributions and element-wise vibrational
density of states. It also demonstrates transfer from small training cells to
larger systems. DSRT adopts the same stringent reference-MD evaluation
principle, but asks whether a predicted future relation should enter the neural
message-passing computation itself.

ChronoCast frames molecular simulation as equivariant autoregressive time-
series prediction and evaluates crystalline dynamics using observables such as
RDF, MSD and VDOS. MDtrajNet combines equivariant representations with a
transformer architecture for direct prediction across molecular chemical
space. These works motivate comparing complete physical trajectories rather
than reducing performance to endpoint error, but neither makes DSRT's matched
future-relation conditioning test.

### 2.2 Trajectory operators and predictive correction

EGNO models a trajectory as a time-dependent function and alternates
equivariant spatial processing with Fourier temporal convolutions. DSRT instead
learns one fixed-horizon endpoint; it does not claim arbitrary-time trajectory
generation. LiFlow separates a generative displacement propagator from a
geometry corrector for atomic transport. This establishes that a broad
predictor–corrector narrative is not unique. DSRT's testable distinction is
that a deterministic full phase-space proposal is transformed into predicted
pair relations that condition a capacity-matched relational refinement stage.

Learned graph simulators have also shown that small one-step errors may compound
under autoregressive distribution shift. We therefore treat endpoint error,
short trajectory error, NVE behavior and NVT statistics as separate outcomes.
The present model is trained on reference transitions without rollout or energy
losses; this keeps the mechanism comparison focused but does not eliminate
autoregressive error accumulation.

## 3. Method

### 3.1 Problem formulation

For a fixed horizon (h), DSRT learns

\[
F_\theta: X_t \mapsto \widehat X_{t+h}.
\]

Training pairs come from reference NVE trajectories and contain the endpoint
displacement and velocity update. During autonomous simulation, every predicted
endpoint becomes the next input and the periodic graph is rebuilt. No reference
force, energy or future state is available to the model. NVT simulation applies
a separately specified CSVR thermostat after each learned NVE transition.

### 3.2 Current-state relational encoding

A periodic radius graph is constructed at (R_t). Each atom carries invariant
scalar features (s_i) and equivariant vector features (v_i); each directed
pair carries a persistent edge state (e_{ij}). Inputs include element type,
mass-aware kinematics, current distance and direction, and relative velocity.
Radial basis functions encode distance while all learned vector maps mix
channels but not Cartesian axes.

Two proposal relation blocks update edge, scalar and vector states. Edge updates
use both endpoint node features and invariant contractions of learned vector
features with pair directions. Messages consist of learned invariant scalar
coefficients multiplying pair directions, relative velocity and sender vector
features. Neighbor aggregation is degree normalized. Node-local mixing feeds
vector norms and invariant vector correlations back into the scalar state,
which in turn gates vector updates.

### 3.3 Future proposal

The proposal head predicts mass-centered residual position and velocity
updates. With training-set scales (\sigma_R) and (\sigma_V),

\[
\widehat R^p_{t+h}
= R_t + hV_t + \sigma_R\,\mathcal C_m(A_p),
\]

\[
\widehat V^p_{t+h}
= V_t + \sigma_V\,\mathcal C_m(B_p),
\]

where (\mathcal C_m) subtracts the mass-weighted mean separately for every
physical graph in a batch. The free-drift term parameterizes displacement but
does not constitute numerical substepping.

### 3.4 Future-conditioned relational refinement

For every edge ((i,j)) in the current graph, DSRT forms a continuous-image
proposal relation

\[
\widehat d^p_{ij}
= d^t_{ij}
+ (\widehat{\Delta r}^p_j-\widehat{\Delta r}^p_i)
\]

with the sign matching the directed-edge convention, together with proposal
relative velocity. Current and proposed distances, directions, radial features,
longitudinal velocities, speed invariants and direction agreement condition two
refinement blocks. The final head predicts a correction using the refined
states, and the same mass-centering rule produces the final endpoint.

The topology is deliberately fixed within one forward pass: proposed geometry
updates features on current edges but does not add newly entering neighbors or
remove exiting ones. At the next autoregressive step, the graph is rebuilt from
the predicted configuration. This boundary controls cost and must be considered
when interpreting rapidly changing local environments.

### 3.5 Objective and optimization

Position and velocity errors are normalized by scales computed only from the
training set. The endpoint objective is

\[
\mathcal L = \mathcal L_{\mathrm{final}}
+0.02\,\mathcal L_{\mathrm{proposal}},
\]

where each term is the sum of normalized position and velocity mean-square
errors. The proposal loss anchors the intermediate state to an actual future
estimate; it is not a dense trajectory loss.

The frozen R3-Opt configuration uses 192 scalar, 48 vector and 96 edge channels,
two proposal and two refinement blocks, FP32, batch size 8 and AdamW with
AMSGrad. Learning rate warms linearly for 5,000 optimizer steps to (3\times
10^{-4}), then decays cosinusoidally toward (10^{-5}). Validation endpoint
loss selects the checkpoint. No test or long-rollout observable is used for
selection.

## 4. Experimental protocol

### 4.1 Quartz reference data

The primary system is periodic 162-atom alpha-quartz at 300 K under the BKS
reference interaction used to generate the project dataset. Independent NVE
source trajectories define train, validation and test sets; no source
trajectory crosses a split. The frozen 30 fs dataset contains 5,000 training,
1,250 validation and 1,000 test transitions derived from 1 fs reference data.
Positions are handled with continuous displacement conventions and velocities
are expressed in Å/fs.

Offline energy evaluation of neural trajectories uses exactly the reference
BKS convention. BKS is never called during a DSRT forward or autonomous step.
The long NVT comparison uses the same CSVR degrees of freedom and relaxation
protocol as the reference setup. VDOS is compared only over the common
30 fs-sampled band, whose Nyquist frequency is 16.67 THz.

### 4.2 Endpoint and autonomous metrics

Endpoint reporting includes position and velocity MAE/RMSE on the complete test
set. Short autonomous prediction reports 1, 2, 4 and 8 learned steps without
reference correction. NVE evaluation reports temperature, element kinetic
energy, minimum pair distance, offline potential and total energy, total-energy
change and fitted drift. NVT evaluation reports block-aware temperature and
energy statistics, Si–O/O–O RDF, element VDOS and the kinetic-energy exchange
from the neural transition and thermostat separately.

### 4.3 Mechanism controls

Full DSRT is compared with two models that preserve width, two-stage depth,
heads, data and optimization. Current-Only Refinement reuses current geometry
in both stages. Ballistic-Geometry Refinement conditions the second stage on
(R_t+hV_t) rather than the learned proposal. Analysis groups test examples by
the true change in current-edge geometry and measures whether Full DSRT's gain
increases when relations evolve more strongly. True future geometry is used
only for post hoc analysis.

### 4.4 Flexible water and generalization

The second system contains 64 flexible water molecules. The same architecture
is trained from scratch at a 5 fs horizon on the independent 5,000/1,250/1,000
split. Physical evaluation includes O–O, O–H and H–H RDF, element VACF/VDOS,
MSD, a predeclared diffusion fit, potential-energy distribution and short NVE
behavior. Separate Quartz experiments apply the frozen 300 K model to
independently equilibrated 250 and 350 K states and to independently thermalized
1,296- and approximately 4,374-atom cells.

### 4.5 Statistical treatment and efficiency

Independent trajectories, rather than individual correlated frames, are the
primary replicates. Long NVT trajectories are divided into blocks longer than
the measured correlation scale; uncertainty is reported across blocks and
independent starts. Efficiency includes model execution, neighbor rebuilding,
thermostat work and production I/O. Compute-only timing is reported separately.
Step-size ratio is not treated as a measured speedup.

## 5. Current Quartz results

### 5.1 Endpoint and short autonomous prediction

The frozen checkpoint contains 2,357,616 trainable parameters. On 1,000
independent test transitions, position RMSE/MAE are 0.001800/0.001430 Å and
velocity RMSE/MAE are 0.0001221/0.0000970 Å/fs. Across 999 starts with complete
reference continuations, position RMSE grows from 0.00180 Å at 30 fs to
0.01899 Å at 240 fs, while temperature RMSE grows from 1.10 to 5.06 K.

### 5.2 NVE behavior

Both existing 100 ps NVE trajectories complete without collision or non-finite
state. Mean temperatures are 298.6 and 304.6 K. Nevertheless, offline BKS total
energy rises by 1.302 and 0.689 eV, compared with 0.0078 and 0.0019 eV in the
reference trajectories. Fitted DSRT drifts are 0.0579 and 0.0886 meV atom^-1
ps^-1. Structural stability therefore does not imply reference-energy
conservation.

### 5.3 Long NVT statistics

Two independent experiments each contain 1.5 ns thermalization and 15 ns
production (500,000 learned steps). Mean temperatures are 300.23 and 300.60 K
versus 299.45 K for the reference. Mean potential energies differ by +0.1184 eV
in both runs, while their standard deviations are 0.435 and 0.430 versus 0.397
eV. O–O structure is close (RDF RMSE 0.024 and 0.022), whereas Si–O structure
retains a larger repeatable error (RDF RMSE 0.180 and 0.179, with first-peak
heights 5.4–5.6% below reference). O VDOS overlaps are 0.9465 and 0.9466 and Si
overlaps are 0.9497 and 0.9501 in the common frequency band. Reference
split-half overlaps exceed 0.999, confirming that the residual spectral
discrepancy is larger than sampling repeatability.

The neural transition adds 0.000961 eV of kinetic energy per 30 fs step on
average and CSVR removes approximately the same amount. Temperature agreement
must therefore be interpreted jointly with thermostat exchange, not as NVE
energy conservation. End-to-end throughput is 106.1 ns/day on one L40S; a
compute-only measurement reaches approximately 291 ns/day.

## 6. Discussion

The current Quartz evidence shows that a learned 30 fs transition can sustain a
long canonical trajectory with useful structural and dynamical statistics.
It also exposes non-negligible limitations: NVE energy drift, a systematic
Si–O peak error, element kinetic-energy imbalance and excess potential-energy
variance. These observations motivate mechanism controls and cross-system
evaluation rather than another architecture iteration.

The present results do not yet isolate future geometry. The two-stage network,
geometry-aware message passing and optimization schedule all contribute to the
observed accuracy. Only matched Current-Only and Ballistic-Geometry experiments
can attribute an independent benefit to the learned proposal relations. The
lightweight earlier DSRT-R1 checkpoint is also retained as an architecture-
capacity reference because its validation endpoint error is lower than R3-Opt.

On the 1,000-transition independent test split, R1 obtains position and
velocity RMSE of 0.001654 Å and 0.00011294 Å/fs, respectively, compared with
0.001800 Å and 0.00012208 Å/fs for R3-Opt. Thus the lightweight R1 model is the
stronger endpoint predictor. This comparison does not establish that R1 has
better autonomous physical statistics, which require the same rollout and
offline BKS evaluation protocol.

Several limits are structural. The model learns one fixed horizon and does not
provide arbitrary-time states. Future relations are evaluated only for current
neighbors within a forward pass. Linear-momentum handling does not imply exact
energy or angular-momentum conservation. Canonical agreement under CSVR does
not establish complete unthermostatted dynamics, chemical reactivity, phase
transition accuracy or transfer beyond evaluated conditions.

## 7. Conclusion

DSRT turns its own future-state proposal into a geometric conditioning signal
for relational refinement. The frozen Quartz model already supports a
long-running, computationally efficient learned trajectory with encouraging
low-frequency dynamics and equilibrium statistics, while its NVE and local-
structure errors establish a clear boundary. The paper's main conclusion will
depend on whether capacity-matched controls and flexible-water experiments show
that predicted future relations provide a reproducible benefit. If not, the
future-geometry mechanism will not be claimed as the source of the long-time
results.

## References used in this draft

1. F. L. Thiemann *et al.*, “Force-free molecular dynamics through
   autoregressive equivariant networks,” *Nature Machine Intelligence* 8,
   764–776 (2026), https://doi.org/10.1038/s42256-026-01227-7.
2. “ChronoCast: A time-series paradigm for molecular dynamics simulation using
   equivariant graph neural networks,” *Physical Review E* 113, 065305 (2026),
   https://pubmed.ncbi.nlm.nih.gov/42493906/.
3. F. Ge, Y. Chen and P. O. Dral, “Artificial Intelligence for Direct
   Prediction of Molecular Dynamics across Chemical Space,” *J. Chem. Theory
   Comput.* 22, 1549–1562 (2026), https://doi.org/10.1021/acs.jctc.5c01689.
4. M. Xu *et al.*, “Equivariant Graph Neural Operator for Modeling 3D
   Dynamics,” *ICML* (2024), https://proceedings.mlr.press/v235/xu24j.html.
5. J. Nam *et al.*, “Flow Matching for Accelerated Simulation of Atomic
   Transport in Materials,” arXiv:2410.01464, https://arxiv.org/abs/2410.01464.

Additional bibliography will be added only after primary-source verification.
