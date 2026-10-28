# SMDP: learned motion modes on TrajCast infrastructure, 30 fs

This is the 2026-10-28 experiment requested by the research lead. **Train only
one complete SMDP, from random initialization**, on our own Quartz 30 fs data.
No original TrajCast or Direct Segment is trained. No published checkpoint is
loaded. Earlier code, weights and negative results remain untouched.

## Network

The pinned IBM TrajCast encoder (four MP layers, lmax=2, 64 channels, 4.5 Å,
float64), endpoint readout, periodic graph construction and ConservationLayer
are reused. Between the last MP layer and readout, four independent equivariant
projections with learned scalar mode embeddings construct motion tokens.
Within a bounded latent irrep space, four shared-parameter recurrent updates
are gated by the mode state, embedding and update phase. Each update includes
off-diagonal, bidirectional mode attention and a separate gated residual.
All attention weights are rotational invariants; values and gates respect
irrep copies. No scalar MLP emits laboratory xyz coordinates.

Mode outputs add to the shared endpoint representation without a competitive
mode softmax. The expansion is zero-initialized; all its inputs and the original
trunk/readout are nevertheless **random**, not pretrained. The residual shared
path remains present; measured mode contributions must establish whether the
network actually uses the extra mechanism. We do not guarantee or enforce a
particular physical interpretation of the modes.

The Temporal Operator produces one set of polar-vector time-response
coefficients per evolved mode. Cubic endpoint interpolation enforces predicted
end position/velocity and the exact initial state. Learned
`u²(1-u)² P_k(2u-1)` corrections and their analytic derivatives change only the
interior. These are time basis functions, not oscillator dynamics or a force
integrator. Every query shares the same graph encoding and latent evolution.
The position derivative equals velocity by construction. Response coefficients
are mass-centered separately per graph; COM displacement follows input COM
velocity, including nonzero COM inputs. The official endpoint momentum
correction is retained. This additional COM displacement consistency is an
explicit SMDP extension, not an unmodified original model prediction.

## Data and frozen training plan

Source: `../data/smdp_reference_md_v1/quartz/transitions/30fs/{train,val,test}.h5`.
All 5000/1250/1000 samples retain independent-trajectory splits. Actual 1 fs
frames and continuations to 240 fs are read using recorded trajectory/frame
indices. No interpolated training labels and no new MD. Source data are read
only. The reused official graph-cache adapter checks all endpoints, cells,
masses, momenta and displacement units against raw trajectories.

A train-only, no-network endpoint-constrained fit compares 6 and 8 response
functions before training. Select the lower count with RMS error <0.0002 Å and
<0.00002 Å/fs; if neither fits, stop rather than silently changing architecture.
The chosen count and numerical results are frozen in `data_preparation.json`.

Official Trainer infrastructure supplies training-set RMS normalization,
Adam/AMSGrad, validation-loss scheduling, checkpointing and logging. SMDP
extends its batch objective and validation callback for true segment labels
and autoregressive training. The previously verified epoch-count scheduler
adapter is reused. Batch size 2, float64, LR 0.001, plateau factor 0.8/patience20,
gradient clipping10, seed20261028. Unlike the paper's endpoint-only LR0.01,
this lower initial LR is a fixed pre-experiment setting for the new segment
objective, not a result-dependent tuning choice.

Stage1: all samples, endpoint and real interior q/v, up to400 epochs,
early stop60. Fixed anchors plus random interior times cover1–30fs; the30fs
endpoint is always included. Stage2/3: 4/8-step pushforward, at most100 epochs
each, early stop30, LR0.0003. Every fourth batch adds rollout loss; only the last
two steps carry gradient. Original supervision is retained. No thermostat,
energy loss, load balancing or artificial mode labels are used in training.

All1250 validation endpoints/segments and a fixed128 valid validation starts
at240fs determine a consistent selection score across every stage. The best
validation checkpoint across **all** stages wins, not automatically the latest
long-rollout checkpoint. Independent tests are evaluated once after selection.

## Execution and physical evaluation

```bash
PY=/data1/wyd/envs/trajcast-c52f107/bin/python
$PY -m trajcast_smdp.data
$PY -m trajcast_smdp.check
bash scripts/run_smdp_trajcast30.sh
```

Run the last command under a persistent user service with an available GPU.
The pipeline refuses to overwrite a started training or existing trajectory.
It publishes failures as failures and never starts another architecture.

Forecast `_make_timestep` is upstream code: network output, position wrapping,
optional original CSVR, neighbor rebuild. Our wrapper adds only unwrapped/HDF5
output, dense queries, physical-stop reporting and timing. It does not call
BKS. The 100ps reference admits3333 complete30fs steps, so the two NVE runs
actually target **99.99ps**, not a fictitious exact100ps endpoint. NVT uses the
reference's100fs CSVR damping,300K,483DOF,1.5ns thermalization plus15ns production.
Dense NVT output contains the real thermostat velocity jump at boundaries;
it must not be described as a smooth purely neural NVE trajectory.

Offline analysis uses the same BKS parameter implementation and existing
VACF/VDOS/RDF/statistics functions as earlier experiments, in the validated
MPICH environment. Report macro30fs (Nyquist16.67THz) and dense1fs spectra
separately against identically sampled own MD. The earlier7.47eV discrepancy
with the published source distribution does not license shifting these data.
Our reference BKS, cells and masses are unchanged. Reference data metadata
contain an obsolete Structured Action research description; these immutable
metadata are not an instruction to restore that model.

Fixed safety stops: nonfinite state, distance<0.65Å, or pre/post-temperature
outside1–3000K. Save the failed state and report actual duration; do not rescale
or repair it. Endpoint and dense-query compute times, neighbor costs and full
production I/O are reported separately. No high-frequency fidelity claim is
made solely because dense output exists.

## Paths and attribution

Server: `/data1/wyd/research/foundation_mlip_adaptation/experiments/trajcast_smdp/quartz30_20261028`.
Main code: `model.py`, `data.py`, `train.py`, `evaluate.py`, `rollout.py`, `analyze.py`.
Metrics, reports and plots are exported to `reports/smdp_trajcast_quartz30_20261028`.
Weights, full trajectories and raw logs remain on the server.

Upstream commit `c52f107a02176671b21a420b2d4c67e0327b04f7`, Apache-2.0:
[retained license](../trajcast_based/LICENSE.upstream).
F. L. Thiemann et al., *Force-free molecular dynamics through autoregressive
equivariant networks*, Nature Machine Intelligence (2026),
https://doi.org/10.1038/s42256-026-01227-7.
Our additions are explicitly named above; no upstream code is relabeled as
SMDP's original contribution. Numerical checks do not establish long-time
stability, improved physical statistics or actual acceleration.
