# Unified comparison — water, 2026-11-27

All rows use the same frozen R3-Opt checkpoint (`water6_reference_20261113/dsrt_radius6/best.pt`),
the same 5 fs horizon and the same `analyze_water.py` protocol. D definitions, windows, blocks and
seeds are in `experiments/D_PROVENANCE_20261127.md`; do not compare a 100 ps D to a 2.5 ns D as
the same measurement.

## 2.5 ns

| run | initial | seed | T (K) | total KE (eV) | translation / rotation / internal KE (eV) | RDF RMSE O–O / O–H / H–H | VDOS O / H | D_MSD (10⁻⁹ m²/s) | D_GK (10⁻⁹ m²/s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| reference | N/A | N/A | 300.169 ± 17.81 | 7.4108 | 2.4629 / 2.5007 / 2.4472 | — | — | **1.883 ± 0.306** | **1.943 ± 0.253** |
| deterministic GLR | val_04 frame 0 | 2026110401 | 299.994 ± 0.391* | 7.4064 | 2.4818 / 2.5259 / 2.3988 | 0.0307 / 0.0376 / 0.0282 | 0.953 / 0.847 | **1.406 ± 0.155** | **1.392 ± 0.127** |
| stochastic GLR | val_04 frame 0 | 2026110401 | 299.969 ± 17.84 | 7.4058 | 2.4830 / 2.5252 / 2.3977 | 0.0276 / 0.0540 / 0.0237 | 0.949 / 0.852 | **1.435 ± 0.130** | **1.403 ± 0.136** |
| deterministic GLR | independent frame 0 | 2026110402 | 300.0005 | 7.4066 | 2.4818 / 2.5271 / 2.3977 | 0.0281 / 0.0437 / 0.0265 | 0.953 / 0.847 | **1.393 ± 0.079** | 1.419 ± 0.078 (audit) |
| stochastic GLR | independent frame 0 | 2026110402 | 300.084 | 7.4087 | 2.4821 / 2.5284 / 2.3981 | 0.0280 / 0.0483 / 0.0266 | 0.950 / 0.853 | **1.387 ± 0.125** | 1.343 ± 0.031 (audit) |
| official TrajCast 2.5 ns | frame 0 | 2026110401 | **pending** | pending | pending | pending | pending | **pending** | **pending** |

\* The deterministic `motion.json` T std is the instantaneous 0.391 K; the block-mean std is
0.016 K. The paired 1–5 ps full/thermostat-only runs have T std ~1e-5 K. The deterministic
rescalse fixes the per-mode target; it is not a stochastic thermostat. See the T/rescale audit
below.

The 100 ps deterministic D is **not** 1.406; the 2.5 ns number is lower. The 100 ps D is given in
the table below.

## 100 ps (same length, same fit window 10–100 ps)

| run | initial | seed | T (K) | translation / rotation / internal KE (eV) | RDF RMSE O–O / O–H / H–H | VDOS O / H | D_MSD (10⁻⁹ m²/s) | D_GK (10⁻⁹ m²/s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| reference block 0 | N/A | N/A | 300.19 ± 17.89† | 2.373 / 2.430 / 2.609 | — | — | **1.692** (one block) | **1.502** (one block) |
| deterministic GLR (`glr_100ps`) | val_04 | 2026110401 | 299.99 ± 0.553* | 2.482 / 2.526 / 2.399 | 0.0322 / 0.1043 / 0.0256 | 0.964 / 0.848 | **1.711** | 1.528 |
| deterministic GLR (`glr_noise_off`) | val_04 | 2026110401 | 299.99 ± 0.553* | 2.482 / 2.526 / 2.399 | 0.0336 / 0.0979 / 0.0255 | 0.962 / 0.849 | **1.623** | 1.473 |
| deterministic GLR | independent | 2026110402 | pending analysis | pending | pending | pending | **1.659** | **1.494** |
| rescale-only baseline | val_04 | 2026110401 | 299.988 ± 0.553* | 2.482 / 2.543 / 2.382 | 0.0383 / 0.1215 / 0.0232 | 0.972 / 0.941 | **2.818** | **2.392** |
| rescale-only baseline | independent | 2026110401 | 300.001 ± 0.041* | 2.482 / 2.544 / 2.381 | 0.0415 / 0.1314 / 0.0233 | 0.975 / 0.941 | **2.669** | **2.422** |
| official TrajCast 100 ps | frame 0 | 2026110401 | 300.19 ± 17.68 | 2.505 / 2.463 / 2.444 | 0.0431 / 0.1180 / 0.0957 | 0.937 / 0.624 | not reported‡ | not reported‡ |

† Reference block 0 T/partition from the first 100 ps of the 2.5 ns reference; the 2.5 ns mean T
std is 17.81 K. The block-0 D/GK are one 100 ps block, not the 2.5 ns mean.
‡ The 100 ps common block is below the frozen MSD fit limit; the 2.5 ns run will produce D.

The `glr_100ps` and `glr_noise_off` are both deterministic at the same seed/initial; the 5% D
difference (1.711 vs 1.623) is within the 100 ps/block/seed spread (1.25–1.71). The reference
block-0 D (1.692) is close to the deterministic 100 ps values; the 2.5 ns reference mean 1.883 is
higher. Do not claim a 25% deficit from a 100 ps comparison.

## Translation ACF (same 100 ps window)

| run | 5 fs | 25 fs | 100 fs | 200 fs | 500 fs | integral 0–5 ps (fs) |
|---|---:|---:|---:|---:|---:|---:|
| reference block 0 | 0.9814 | 0.6533 | **0.0311** | −0.0798 | −0.0217 | **11.341** |
| deterministic GLR val_04 | 0.9806 | 0.6405 | **0.0143** | −0.0812 | −0.0299 | **11.036** |
| rescale-only val_04 | 0.9819 | 0.6622 | **0.0620** | −0.0621 | −0.0258 | **17.278** |
| deterministic GLR independent | 0.9807 | 0.6421 | **0.0110** | −0.0810 | −0.0242 | **10.790** |
| rescale-only independent | 0.9819 | 0.6622 | **0.0548** | −0.0615 | −0.0258 | **17.496** |

The full GLR matches the reference block-0 ACF and its 0–5 ps integral (10.8–11.0 vs 11.34 fs).
The rescale-only baseline has a 100 fs shoulder that is too high (0.055–0.062 vs 0.031) and an
integral 17.3–17.5 fs, ~1.5× the reference block 0. This is the same over-diffusion seen in the
100 ps D (2.67–2.82 vs 1.66–1.71). **AR memory is necessary for the 100 ps ACF/D**; the simple
rescale cannot reproduce the reference time correlation. It is not uniformly better: the simple
baseline has a better H VDOS (0.941 vs 0.848) and lower H VACF error (0.024 vs 0.073–0.076).

## T and rescale audit

- Deterministic GLR: `stochastic_rescale=False`; each mode is rescaled to a fixed
  `0.5 dof k_B T` target every step. The paired full/thermostat-only 1–5 ps T std is ~1e-5 K; the
  100 ps `motion.json` instantaneous T std is 0.553 K and the 2.5 ns 0.391 K (block-mean std
  0.033/0.016 K). This is a narrow fixed target, not the canonical Gamma distribution.
- Stochastic GLR: `stochastic_rescale=True`; the per-mode target is Gamma-distributed. The 100 ps
  T std is 17.84 K, partition translation std 0.253 eV, matching the reference (17.89 K,
  0.239 eV). The AR memory does not create the KE variance; the stochastic rescale does.
- The simple rescale-only baseline is deterministic, so its KE distribution should be compared to
  the deterministic GLR, not to the stochastic GLR. A stochastic rescale-only baseline was not
  run this round.

## Calibration cost / independence

See `experiments/D_PROVENANCE_20261127.md`:

- production GLR: 100 ps reference NVT ACF; 0–50 fs Yule–Walker fit; no D constraint; no recorded
  reference/start metadata; likely block 5 (500–600 ps) of the 2.5 ns test trajectory.
- independent `train_nvt_200ps`: 100 ps/200 ps fits reproduce the same AR(10) and 32.6–33.5 fs
  integral; the coefficients are not sensitive to the calibration block, but the paper must use a
  held-out trajectory and record the window.
- extra MD: 100–200 ps reference NVT for AR; ≥500 ps for a stable D target; no reference forces
  at inference. The rescale-only baseline needs no ACF calibration; if it matches the full GLR,
  the AR memory's extra calibration cost is not justified.
- official TrajCast: large-scale training data/compute; 100 ps control complete, 2.5 ns pending.

## Bottom-line comparison

- At 2.5 ns, deterministic/stochastic GLR D is 1.39–1.44 vs reference 1.88; ~24–26% low, with
  seed/block spread of similar size. GK/MSD agree (ratios ~0.96–1.06); this is a common low-D
  bias, not a GK/MSD break.
- At 100 ps, deterministic GLR D 1.62–1.71 is close to the reference block-0 D 1.69; the 2.5 ns
  deficit emerges at longer times. Do not use the 100 ps D to claim or refute the 2.5 ns deficit.
- **The 100 ps no-AR baseline is not equivalent:** rescale-only D_MSD is 2.67–2.82 (vs full
  1.62–1.71 and reference block 0 1.69), and its translation ACF integral is 17.3–17.5 fs (vs
  full 10.8–11.0 and reference block 0 11.34). AR memory is necessary for the 100 ps ACF/D.
- **But the AR memory is not uniformly better:** rescale-only has a better H VDOS (0.941 vs
  0.848) and lower H VACF error (0.024 vs 0.073–0.076); its RDF O–H is worse (0.122–0.131 vs
  0.098–0.104). The method is a trade-off, not a free improvement.
- The matching 2.5 ns rescale-only baseline is running on val_04 (one long start; the two 100 ps
  starts cover initial-state independence). It will decide whether the simple scaling's
  over-diffusion persists at long time or converges toward the reference. Do not fill a 2.5 ns
  baseline number before it exists.
- Official TrajCast 2.5 ns is still running; no 2.5 ns D/VDOS number is available yet.
