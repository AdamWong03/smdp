# Composed CSVR Rollout Training — 2026-11-21

**Goal:** constrain the stationary measure of the *composed* propagator + thermostat map,
instead of the one-step conditional mean or NVE energy.

---

## 1. Method

Training rollout:

1. Start from states of the base model's own NVT trajectory (`self_pairs.h5`, 2,000 states,
   5 fs pairs).
2. Roll out K = 5 steps. At each step: model forward → global thermostat rescaling to 300 K.
3. Loss:
   ```
   L = λ_ep · endpoint(base step-0 prediction vs the model's own next state)
     + λ_ens · [ Σ_mode ((KE_mode - 0.5·dof_mode·kT)/target)²
               + ((T - 300)/300)² ]
   ```
4. Stability mask: exclude rollouts with final T > 450 K.
5. Gradient checkpointing for memory.

The thermostat is a deterministic global rescaling in training (a mean-field proxy for CSVR);
inference uses the standard stochastic CSVR.

## 2. First run (full fine-tune): partition fixed, endpoint collapsed

Run `csrv_rollout_finetune_500_20261121`, λ_ep=1, λ_ens=10, lr=1e-5, 500 steps (stopped at
~step 110):

| Step | endpoint | val moment | val T | mode KE (trans/rot/int) |
|---:|---:|---:|---:|---|
| 1 | 7.8e-5 | 2.34 | 322 K | 0.52 / 0.83 / 6.15 |
| 50 | — | 0.162 | 307.6 K | — |
| 100 | 0.041 | **0.028** | 307.4 K | 2.63 / 2.71 / 2.79 |

The ensemble moment dropped by ~80× and the partition moved to near-equipartition, but the
one-step endpoint degraded from 7.8e-5 to 0.041 (~500×).

**Interpretation:** the base model's one-step map and its wrong stationary measure are
properties of the same map. Changing the stationary measure requires changing the map; the
one-step accuracy on the original data degrades. The question is whether the endpoint is the
right constraint at all — the published model has a ~15× larger pointwise error than the base
yet passes NVT.

## 3. Frozen-head and mode-aware variants

- **Frozen backbone, train proposal/correction heads (115,776 params):** endpoint still
  degraded to 0.05–0.06 at λ_ep=10/λ_ens=10. The heads can change the map enough to break the
  endpoint.
- **Mode-aware correction (83,722 params, zero-initialized):** endpoint degraded to 0.18–0.26;
  val moment 0.35–0.80. A zero-init correction does not stay small under the ensemble gradient.

The endpoint/ensemble trade-off is structural, not a head-capacity issue.

## 4. Decisive test

Run a 100 ps NVT screen on the step-100 checkpoint (`val moment 0.028`, endpoint 0.041):

- `experiments/nvt_screen_csrv_full_s100_20261121`

If the NVT partition is at equipartition and RDF/VDOS are good, the endpoint collapse is
acceptable and the method works (consistent with the published model's generative behavior).
If the NVT is unstable or structurally wrong, the endpoint loss is protecting something real,
and the next step is a trust-region / architecture change.

## 5. Step-100 screen: failed at 4.3 ps

The step-100 checkpoint (val moment 0.028, endpoint 0.041) was run under the 100 ps NVT screen:

- **failed at step 855 (4.3 ps)**: min distance **0.647 Å**, pre-thermostat T **1473 K**.

The 5-step ensemble moment did **not** predict the 4 ps blow-up. The endpoint degradation from
7.8e-5 to 0.041 was not merely "pointwise accuracy"; it signaled that the map had left the
stable basin. The short-rollout moment loss found a map that looks partition-correct locally
but is dynamically invalid.

**Method verdict:** the composed 5-step CSVR rollout loss, as implemented, is **negative**. It
does not preserve stability. The next route must either:

1. **Constrain stability explicitly**: longer rollout, minimum-distance/energy/contact penalty,
   and a hard trust region on the map (e.g., correction norm bounded to a few % of velocity).
2. **Change the architecture/inductive bias**: the published model is stable and
   partition-correct; our 250k RelationalTransition may be the limiting factor.

## 6. Open issues

- K=5 is short; the moment loss may fix the mean partition without the correct long-time
  dynamics.
- Deterministic training thermostat vs stochastic CSVR at inference.
- Self-states from the base model: no reference labels, but the model can drift to a different
  ensemble; validation on the independent NVT start is required.
- The endpoint anchor may need to be a **trust region on the output**, not an MSE to the base.
- 100 ps → 2.5 ns, seeds, and quartz remain.
