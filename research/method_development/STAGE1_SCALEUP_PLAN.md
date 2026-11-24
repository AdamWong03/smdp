# Stage 1 scale-up — statistics and 2.5 ns quartz

**Status:** prepared 2026-11-23; run after the R8 package and the D-feasibility gate.
**Principle:** scale the evidence, not the method. No new variants.

## 1. What is already locked

- Water: fixed GLR, molecular modes, AR(10), deterministic variance rescale, \(\lambda=0.1\),
  \(q'=q+\Delta q\). Two starts, one 2.5 ns each, three 100 ps seeds.
- Quartz: fixed GLR, phonon-band partition, memory off, stochastic rescale, 5 fs, two 100 ps seeds.
- Open limitations: water \(D\) 26% low (translation kernel); stochastic water GK/MSD trade-off.

## 2. Stage 1 matrix

### Water 2.5 ns deterministic

| Start | Seeds | Runs | Purpose |
|---|---|---|---:|---|
| val_04 NVE | 2026110401, 02, 03 | 3 | mean ± 95% CI for partition/RDF/VDOS/D |
| independent NVT | 2026110402, 03, 04 | 3 | initial-condition robustness |

Existing runs: val_04 seed 01 and independent seed 02 (2.5 ns). Add four runs.

Command template:

```bash
python rollout_relational.py \
  --checkpoint water6_reference_20261113/dsrt_radius6/best.pt \
  --initial <val_04 or independent_nvt_500ps_20261121/initial.h5> \
  --initial-frame 0 --output <run>/trajectory.h5 \
  --ensemble nve --analysis-ensemble nvt \
  --steps 500000 --thermalization-steps 0 \
  --glr-coefficients experiments/glr_fit_20261122/glr_coefficients.json \
  --fd-blend 0.1 --fd-rescale --seed <seed>
```

### Quartz 2.5 ns phonon-band stochastic

| Seeds | Runs | Purpose |
|---|---:|---|
| 2026110401, 02, 03, 04 | 4 | 100 ps result survives 2.5 ns; mean ± range |

Existing runs: seeds 01 and 02 at 100 ps. Add four 2.5 ns runs.

Command template:

```bash
python rollout_relational.py \
  --checkpoint quartz5_relational_20261122/best.pt \
  --initial quartz/raw_nve/test_09/trajectory.h5 \
  --initial-frame 0 --output <run>/trajectory.h5 \
  --ensemble nve --analysis-ensemble nvt \
  --steps 500000 --thermalization-steps 0 \
  --phonon-glr-basis /home/wyd/quartz_phonon/phonon_basis.npz \
  --phonon-memory-off --phonon-rescale-partition bands \
  --fd-rescale --fd-stochastic-rescale --fd-blend 0.0 --seed <seed>
```

## 3. Compute estimate

| Job | Unit cost | Runs | GPU-h |
|---|---:|---:|---:|
| Water 2.5 ns | 3–5 h | 4 new (+2 existing) | 12–20 |
| Quartz 2.5 ns | 3–5 h | 4 | 12–20 |
| **Total** | | **8 new** | **24–40** |

4×A100/H100 80GB, 1–2 days; rental cost roughly \$300–800 at \$2–4/GPU-h.

## 4. Gates

Water:
- mean partition within 0.05 eV of the 2.5 ns reference on both starts;
- RDF/VDOS within the existing seed spread;
- \(D\) mean in 1.3–1.5×10⁻⁹ with 95% CI; the 26% deficit is reported, not hidden.

Quartz:
- \(T=300\pm3\) K at 2.5 ns, stable;
- VDOS O/Si ≥ 0.90;
- Si–O RDF ≤ 0.08;
- no drift in the mode KE/ACF between 100 ps and 2.5 ns.

If quartz 2.5 ns drifts, stop and report the 100 ps result; do not tune.

## 5. Analysis

- Water: `run_glr_2p5ns_analysis.sh <run> "<label>"`, `quick_msd.py --fit-start-ps 10
  --fit-stop-ps 2000`, `green_kubo.py --duration-ps 100`.
- Quartz: `quartz_analysis.py`, `phonon_diagnostics.py`.
- Aggregate seeds with mean ± 95% CI (water) and mean ± range (quartz).

## 6. Not in Stage 1

- Temperature transfer (needs new reference NVT + ACF refit).
- Larger box (backbone may not transfer).
- Third material (new reference + backbone).
- Action-map baseline (separate implementation).
- Stochastic water as production (GK/MSD trade-off unresolved).
