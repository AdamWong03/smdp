# D provenance and calibration cost (2026-11-27)

All D values in **10⁻⁹ m²/s**. The numbers below are deliberately not merged into one
"GLR D": they come from different run lengths, initial states, block selections and estimators.

**Definitions**

- `MSD-std`: molecular-COM MSD, global COM removed, standard `analyze_water.py` diffusion
  analysis; 8 blocks of 312.5 ps for a 2.5 ns trajectory, fit window 10–100 ps, 50 fs sampling.
- `MSD-audit`: same MSD definition, but 8 evenly spaced 100 ps blocks at 0, 300, …, 2100 ps;
  fit window 10–100 ps.
- `GK-audit`: Green–Kubo from the molecular-COM velocity ACF, 0–5 ps, on the same 8 × 100 ps
  blocks.
- `1 block`: one 100 ps trajectory, fit window 10–100 ps.
- Single-block `green_kubo` values are first/later 100 ps segments, not 2.5 ns means.

## Table

| D | run | initial | length | estimator | window / blocks | seed | source |
|---:|---|---|---:|---|---|---:|---|
| **1.406 ± 0.155** | deterministic GLR 2.5 ns | val_04 frame 0 | 2.5 ns | MSD-std | 10–100 ps, 8 × 312.5 ps | 2026110401 | `experiments/glr_2p5ns_20261122/analysis/diffusion.json` |
| **1.476 ± 0.228** | same run | val_04 | 2.5 ns | MSD-audit | 10–100 ps, 8 × 100 ps | 2026110401 | `experiments/position_velocity_audit_20261127.json` |
| **1.392 ± 0.127** | same run | val_04 | 2.5 ns | GK-audit | 0–5 ps, 8 × 100 ps | 2026110401 | same |
| 1.420 | same run, first 100 ps | val_04 | 100 ps | GK | 0–5 ps, 1 block | 2026110401 | `experiments/glr_2p5ns_20261122/green_kubo.json` |
| 1.450 | same run, later 100 ps | val_04 | 100 ps | GK | 0–5 ps, 1 block | 2026110401 | `experiments/glr_2p5ns_20261122/green_kubo_start500.json` |
| **1.711** | `glr_100ps` deterministic | val_04 | 100 ps | `block_metrics` MSD | 10–100 ps, 1 block | 2026110401 | `experiments/glr_ablation_audit_20261127.json` |
| **1.623** | `glr_noise_off` deterministic | val_04 | 100 ps | `block_metrics` MSD | 10–100 ps, 1 block | 2026110401 | same |
| 1.257 | `glr_seed2` deterministic | val_04 | 100 ps | `block_metrics` MSD | 10–100 ps, 1 block | 2026110402 | same |
| 1.252 | `glr_seed3` deterministic | val_04 | 100 ps | `block_metrics` MSD | 10–100 ps, 1 block | 2026110403 | same |
| **1.658** | deterministic GLR independent 100 ps | independent frame 0 | 100 ps | `block_metrics` MSD | 10–100 ps, 1 block | 2026110402 | `experiments/glr_independent_100ps_20261122/` |
| 1.494 | same | independent | 100 ps | `block_metrics` GK | 0–5 ps, 1 block | 2026110402 | same |
| **1.393 ± 0.079** | deterministic GLR independent 2.5 ns | independent | 2.5 ns | MSD-std | 10–100 ps, 8 × 312.5 ps | 2026110402 | `experiments/glr_independent_2p5ns_20261122/analysis/diffusion.json` |
| 1.444 ± 0.151 | same | independent | 2.5 ns | MSD-audit | 10–100 ps, 8 × 100 ps | 2026110402 | `experiments/independent_2p5ns_block_audit_20261127.json` |
| 1.419 ± 0.078 | same | independent | 2.5 ns | GK-audit | 0–5 ps, 8 × 100 ps | 2026110402 | same |
| **1.435 ± 0.130** | stochastic GLR 2.5 ns | val_04 | 2.5 ns | MSD-std | 10–100 ps, 8 × 312.5 ps | 2026110401 | `experiments/glr_stoch_2p5ns_20261122/analysis/diffusion.json` |
| 1.350 ± 0.243 | same | val_04 | 2.5 ns | MSD-audit | 10–100 ps, 8 × 100 ps | 2026110401 | `experiments/position_velocity_audit_20261127.json` |
| 1.403 ± 0.136 | same | val_04 | 2.5 ns | GK-audit | 0–5 ps, 8 × 100 ps | 2026110401 | same |
| 1.206 | same, first 100 ps | val_04 | 100 ps | GK | 0–5 ps, 1 block | 2026110401 | `experiments/glr_stoch_2p5ns_20261122/green_kubo_100ps.json` |
| 1.157 | same, later 100 ps | val_04 | 100 ps | GK | 0–5 ps, 1 block | 2026110401 | `.../green_kubo_start500.json` |
| **1.387 ± 0.125** | stochastic GLR independent 2.5 ns | independent | 2.5 ns | MSD-std | 10–100 ps, 8 × 312.5 ps | 2026110402 | `experiments/glr_stoch_independent_2p5ns_20261122/analysis/diffusion.json` |
| 1.377 ± 0.214 | same | independent | 2.5 ns | MSD-audit | 10–100 ps, 8 × 100 ps | 2026110402 | `experiments/independent_2p5ns_block_audit_20261127.json` |
| 1.343 ± 0.031 | same | independent | 2.5 ns | GK-audit | 0–5 ps, 8 × 100 ps | 2026110402 | same |
| **1.417** | stochastic full 100 ps | val_04 | 100 ps | `block_metrics` MSD | 10–100 ps, 1 block | 2026110401 | `experiments/glr_ablation_audit_20261127.json` |
| **1.883 ± 0.306** | reference 2.5 ns | N/A | 2.5 ns | MSD-std | 10–100 ps, 8 × 312.5 ps | N/A | `experiments/glr_2p5ns_20261122/analysis/diffusion.json` |
| 1.890 ± 0.410 | reference 2.5 ns | N/A | 2.5 ns | MSD-audit | 10–100 ps, 8 × 100 ps | N/A | `experiments/position_velocity_audit_20261127.json` |
| 1.943 ± 0.253 | reference 2.5 ns | N/A | 2.5 ns | GK-audit | 0–5 ps, 8 × 100 ps | N/A | same |
| 1.836 | reference block 500–600 ps | N/A | 100 ps | GK | 0–5 ps, 1 block | N/A | `experiments/glr_2p5ns_20261122/green_kubo_reference.json` |

**Not D:** the 5 ps feedback controls report early MSD slopes (full 1.333, no_feedback 0.721,
thermostat_only 2.224); these are 1–5 ps non-diffusive indicators, not long-time D. The paired
200-step run is a 1 ps mechanism test and its COM-MSD field was not usable.

**Why the numbers differ**

- 1.406 vs 1.476: same trajectory, same seed; different block selection (standard 8 × 312.5 ps
  vs 8 evenly spaced 100 ps) and therefore different MSD fits. Both are valid estimators; the
  paper should use one pre-registered protocol.
- 1.406 vs 1.711: 2.5 ns vs 100 ps. The 100 ps block has a larger D; the 2.5 ns mean is lower.
  Do not compare a 100 ps D to a 2.5 ns D as if they were the same measurement.
- 1.711 vs 1.623: two deterministic 100 ps runs/flags at the same seed/initial; the 5% difference
  is within the 100 ps/block/seed spread (1.25–1.71). Use the explicitly noise-off run for the
  deterministic production comparison.
- 1.435 vs 1.350: stochastic 2.5 ns, standard vs audit block selection.
- The single-block 1.206/1.157 (stochastic) and 1.420/1.450 (deterministic) are **not** 2.5 ns
  means; they are 100 ps GK segments.

## Calibration cost and independence

- Production GLR coefficients: a **100 ps reference NVT** block; the artifact has no reference
  path/start/duration metadata. The fit script's Yule–Walker system uses **0–50 fs** (order 10,
  stride 5 fs); `--maxlag 40` only extends the reported ACF to 200 fs.
- Likely production window: `fit_glr.py` defaults start 500 ps, duration 100 ps → block 5
  (500–600 ps) of the 2.5 ns test trajectory. This is **not recorded**; calibration/evaluation
  overlap is unresolved.
- Independent `train_nvt_200ps`: 100 ps block 0, 100 ps block 1 and the full 200 ps give the same
  AR(10) (radius 0.853–0.855) and the same 32.6–33.5 fs AR integral. The production coefficients
  are reproduced. Therefore the missing metadata/overlap is a provenance issue, not evidence that
  leakage drives the low D; nevertheless the paper must use and record a held-out calibration
  trajectory.
- Extra MD needed by the GLR: **100–200 ps reference NVT** for the AR coefficients; **≥500 ps**
  (preferably the full 2.5 ns) for a stable D target. No reference forces at inference. The GLR is
  training-free for the backbone but **not data-free**.
- The simple rescale-only baseline uses the same mode basis and canonical per-mode target but
  **does not need the reference ACF**. If it matches the full GLR, the AR memory's extra
  calibration data is not justified.
- Official TrajCast requires large-scale training data/compute; the 100 ps control is the
  completed official comparison, and the 2.5 ns result is pending.
