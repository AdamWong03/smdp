# Root-Cause Audit Plan

**Date:** 2026-11-20
**Why:** all stable 250k models fail Water NVT in the same way, despite different operators and
conditioning paths. Training-time/early stopping is ruled out (H-C unstable). The next step is
to find whether the shared data/training/evaluation pipeline is the cause, or whether the
architecture family is fundamentally mismatched.

---

## 1. The evidence that triggers the audit

| Model | Stable NVT? | Translation | Rotation | Internal | VDOS H/O |
|---|---|---:|---:|---:|---:|
| R3-Opt 4.5 Å (2.5 ns) | yes | 0.152 | 0.212 | 7.330 | 0.215/0.233 |
| 250k Full 6 Å (100 ps) | yes | 0.538 | 0.786 | 6.044 | 0.406/0.487 |
| Current-Only 250k (100 ps) | yes | 0.646 | 0.535 | 6.384 | 0.348/0.600 |
| v0 250k (2.5 ns) | yes | 0.111 | 0.808 | 6.338 | 0.356/0.236 |
| H-C 20k | **no** | — | — | — | — |
| Reference | — | 2.37–2.46 | 2.43–2.50 | 2.45–2.61 | — |

Three independent 20 ps NVE “improvements” were false signals. The failure is robust across
architectures and conditioning paths.

---

## 2. Audit tracks

### A. Data and reference integrity (P0)

- Verify transition HDF5 positions/velocities/cell/molecule_ids against raw NVE at the stored
  source frames (the cache builder already does this, but re-run on the 6 Å cache).
- Verify `displacements` = final − initial and `update_velocities` = final velocity; check for
  unit or frame-offset errors.
- Verify molecule IDs and OHH ordering; check the partition code on a known synthetic molecule.
- Verify the reference NVT: `reference_verified` in the NVT energy analysis, thermostat
  metadata, time origin, common sampling.
- Compare the project's water data protocol to TrajCast: 64 molecules, 5 fs, flexible SPC,
  6 Å, 5,000 samples. Identify any protocol difference (potential, cutoff, timestep,
  thermostat, reference package).

### B. Model input/target contract (P0)

- Trace one validation sample through: raw HDF5 → graph cache → model input → output → loss.
- Check COM removal, global velocity RMS, `velocity_scale`, absolute vs relative velocity,
  displacement parameterization, and periodic wrapping.
- Check that the model's one-step validation partition is near-perfect (it is) and that the
  failure only appears in rollout; identify the first rollout step where the partition drifts.
- Check whether the model's predicted positions/velocities are consistent with a single
  Hamiltonian/energy surface or whether the map is dissipative/injecting energy. The NVT
  thermostat balance shows the network adds/removes energy; quantify.

### C. Training recipe (P1)

- Compare to TrajCast: equal Δr/v MSE (no proposal term), batch 2 condensed, double precision,
  plateau scheduler, up to 1,500 epochs; ours: q/v MSE + 0.02 proposal, batch 8, FP32,
  warmup-cosine, 250k updates.
- Check train/validation loss curves for overfitting or a shift in the velocity distribution.
- Check whether the model learns the **conditional mean** but not the state distribution;
  compare predicted vs reference velocity distributions on validation and autonomous states.

### D. Matched native TrajCast NVT control (P0)

Use the existing native TrajCast validation-best checkpoint (epoch 145, 362,500 updates,
val pos 0.000144 Å, vel 0.000063 Å/fs) on the same val_04 start and 100 ps NVT protocol.

- Adapt `water_reference/diagnose_native_nve.py` to NVT with the same CSVR 300 K, τ=50 fs.
- If native TrajCast preserves partition → the gap is architecture/training, and the method
  should target the specific difference.
- If native TrajCast also fails → the data/protocol/task is the issue; do not claim an
  architecture problem.

This is a **root-cause control**, not a SOTA claim. The checkpoint is validation-selected but
the native protocol was stopped before completion; label it as diagnostic.

### E. Evaluation audit (P1)

- Confirm 100 ps NVT screen ranking matches 2.5 ns NVT for at least two models.
- Confirm the partition definitions and reference windows are identical across models.
- Check thermostat balance and stationarity in the first 100 ps.

---

## 3. Decision tree

```
Audit A/B finds a data/contract bug?
  yes → fix, re-run the 100 ps NVT screens; do not change architecture yet.
  no  → run native TrajCast NVT control (D).

Native TrajCast preserves NVT partition?
  yes → architecture/training gap; compare operators/loss and design a targeted method.
  no  → data/protocol/task gap; revisit reference/data/protocol before any method claim.

Both fail and no bug?
  → the task may be harder than the published water benchmark under our protocol;
    report the protocol difference honestly; consider a different evaluation/training target.
```

---

## 4. What not to do

- Do not launch another architecture/loss change before A/B/D are resolved.
- Do not use 20 ps NVE or 500 fs rollout as a selection metric.
- Do not interpret a model that blows up as “better partition.”
- Do not claim NMI readiness until a stable, NVT-faithful method exists at the full budget.

---

## 5. Audit findings (progress 2026-11-20)

### A. Data contract: PASS (sampled validation)

For validation samples 0/1/17/100 of the 6 Å cache, compared against the transition HDF5:

- initial velocities: exact match (`0.0` max abs difference);
- final velocities (`update_velocities`): exact match;
- displacements modulo cell: ≤3.7e-9 Å;
- molecule IDs: exact match;
- cell: ≤8.9e-8 Å.

This does **not** prove the full pipeline is bug-free, but the obvious frame/unit/ID/velocity
contract is correct on the sampled transitions.

### B. One-step model contract: PASS

The 250k 6 Å Full model on validation states:

| Component | Predicted | Reference |
|---|---:|---:|
| Translation | 2.2757 eV | 2.2747 eV |
| Rotation | 1.9906 eV | 1.9923 eV |
| Internal | 2.5718 eV | 2.5724 eV |

One-step partition is almost perfect. The failure appears only in autonomous rollout/NVT, so
it is not a simple input/target/partition-code bug on reference states. It is an
autonomous-state/error-accumulation problem.

### D. Native TrajCast NVT: completed — stable, opposite bias

Using the validation-best epoch-145 checkpoint (not a completed native protocol; native
training stopped at epoch 149/1500) on the same val_04 start and 100 ps NVT screen:

| 100 ps NVT | Native TrajCast | 250k Full | Reference |
|---|---:|---:|---:|
| Translation | 3.256 eV | 0.538 eV | 2.373 eV |
| Rotation | 3.105 eV | 0.786 eV | 2.430 eV |
| Internal | 1.041 eV | 6.044 eV | 2.609 eV |
| Total | 7.401 eV | 7.368 eV | 7.411 eV |
| O–O RDF RMSE | 0.180 | 0.646 | — |
| H/O VDOS | 0.683/0.905 | 0.406/0.487 | — |

Native is stable and total KE is close to reference, but its partition bias is **opposite**:
collective motion too hot, internal motion too cold. Our models over-excite internal and
suppress collective.

This is **not** a completed native success: the native protocol stopped at epoch 149/1500, and
its O–H RDF is poor. It is a diagnostic control, not a SOTA baseline.

**Conclusion so far:**

- data contract and one-step contract pass;
- the task is not universally forcing one failure direction;
- architecture/training controls the **direction** of the partition bias;
- native is under-trained, so we cannot attribute the opposite bias to architecture alone;
- a method that balances collective/internal motion is still needed.

### C. Training recipe audit: in progress

Native uses equal Δr/v MSE, no proposal loss, velocity tensor product, momentum correction;
ours uses global q/v MSE + 0.02 proposal, endpoint selection. The next diagnostic should test
one of these differences at a stable budget (250k or fine-tune from 250k).

Compare equal Δr/v MSE vs our proposal loss, batch size, precision, epochs; only after D.

## 6. Immediate next actions

1. Run the data/contract audit on one validation sample and the 6 Å cache.
2. Implement the native TrajCast NVT screen (100 ps) and run it.
3. Compare native TrajCast vs 250k Full / Current-Only / v0 on the same NVT panel.
4. Only then choose the next method change.
