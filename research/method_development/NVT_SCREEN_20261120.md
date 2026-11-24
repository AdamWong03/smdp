# NVT Screen — 2026-11-20

**Purpose:** replace the misleading 20 ps NVE proxy with a short NVT screen.
**Protocol:** 50 ps thermalization + 100 ps production, CSVR 300 K, τ=50 fs, val_04 frame 0.
**Reference:** `physical_reference/nvt_2p5ns/production.h5`, first 100 ps window.
**Note:** diffusion is not analyzed at 100 ps because the frozen fit window needs ≥100 ps of
stationary MSD; use 2.5 ns for diffusion.

---

## 1. Results

| Model | Translation KE | Rotation KE | Internal KE | Total KE | O–O RDF RMSE | O–H RDF RMSE | H VDOS | O VDOS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Reference (first 100 ps) | 2.373 | 2.430 | 2.609 | 7.411 | — | — | — | — |
| 250k Full 6 Å | 0.538 | 0.786 | 6.044 | 7.368 | 0.646 | 1.312 | 0.406 | 0.487 |
| v0 (2.5 ns, full production) | 0.111 | 0.808 | 6.338 | 7.257 | 0.944 | 1.409 | 0.356 | 0.236 |
| Current-Only 250k | 0.646 | 0.535 | 6.384 | 7.566 | 0.631 | 1.345 | 0.348 | 0.600 |
| Native TrajCast ep145 | 3.256 | 3.105 | 1.041 | 7.401 | 0.180 | 1.584 | 0.683 | 0.905 |
| H-C 20k prefix | **invalid** | **invalid** | **invalid** | **invalid** | **invalid** | **invalid** | **invalid** | **invalid** |

**Native is stable but not a success:** collective motion too hot, internal too cold, O–H RDF
poor; the native protocol stopped at epoch 149/1500. It is a diagnostic control, not a
completed baseline.

**H-C is not a 100 ps screen.** It stopped at step 20,154 (10,154 production steps ≈ 50.8 ps)
with a physical safety stop: min distance 0.852 Å and pre-thermostat 3338 K. The partial-window
numbers (translation 3.06, rotation 2.31, internal 2.45, VDOS 0.906/0.864) look “better” only
because the trajectory is blowing up; they are not a valid equilibrium comparison.

The 250k Full and v0 both fail the same way: translation/rotation suppressed, internal motion
over-excited, RDF over-sharpened, VDOS low. Total KE is close to reference only because the
partition is redistributed, not because the dynamics are correct.

---

## 2. Why 100 ps NVT is the right screen

v0's 20 ps NVE looked better than the 250k Full NVE (internal 3.13 vs 4.42 eV), but its 2.5 ns
NVT failed (internal 6.34 vs 2.45 eV). The NVT bias is already visible in the first 100 ps:

| v0 NVT window | Translation | Rotation | Internal | Reference |
|---|---:|---:|---:|---:|
| 0–100 ps | 0.286 | 1.308 | 5.524 | 2.298 / 2.356 / 2.755 |
| 100–250 ps | 0.104 | 0.846 | 6.505 | 2.413 / 2.358 / 2.392 |
| 250–500 ps | 0.107 | 0.816 | 6.383 | 2.393 / 2.390 / 2.607 |
| 1000–2500 ps | 0.106 | 0.802 | 6.412 | 2.483 / 2.516 / 2.388 |

Therefore:

- **Do not use 20 ps NVE or 500 fs rollout for equilibrium selection.**
- Use 100 ps NVT for method iteration; reserve 2.5 ns NVT for the decisive result.
- H1's null at 20k is not surprising: its proxy was not predictive of NVT.

---

## 3. What this means for the method

No current version passes NVT:

| Version | Endpoint | 20 ps NVE | 100 ps/2.5 ns NVT |
|---|---|---|---|
| R3-Opt 4.5 Å | good on Quartz | Water fails | Water fails (2.5 ns) |
| 6 Å Full 250k | best Water endpoint | internal 4.42 vs 2.34 | **fails** (100 ps: internal 6.04 vs 2.61) |
| Current-Only 250k | comparable endpoint | translation/rotation near ref; internal 4.51 | not run |
| v0 | worse endpoint | internal 3.13 (looked better) | **fails** (2.5 ns: internal 6.34 vs 2.45) |

The failure is robust. A method that only improves endpoint or 20 ps NVE is not enough.

---

## 4. H-C conclusion: training-time/early-stopping line is not supported

H-C (20k prefix, same seed as 250k Full) was expected to test whether the 20k model is
physically better. It is **unstable** at ~50 ps production (min 0.852 Å, preT 3338 K), so it
cannot be compared as an equilibrium model. The 20 ps NVE “better partition” signal was
misleading again.

This rules out the simple “train less / early stop” interpretation:

- 20k prefix: unstable NVT;
- 250k Full: stable but physically wrong;
- v0: stable but physically wrong and worse;
- R3-Opt 4.5 Å: stable but physically wrong.

A viable method must be **stable at the full training budget and physically faithful in NVT**.
Stability is now a first-class requirement, not an afterthought.

## 5. Current-Only 250k result: also fails

Current-Only is stable for the full 100 ps NVT screen, but its partition fails:

| 100 ps NVT | Current-Only | Reference |
|---|---:|---:|
| Translation | 0.646 eV | 2.373 eV |
| Rotation | 0.535 eV | 2.430 eV |
| Internal | 6.384 eV | 2.609 eV |
| Total | 7.566 eV | 7.411 eV |
| O–O RDF RMSE | 0.631 | — |
| O–H RDF RMSE | 1.345 | — |
| H/O VDOS | 0.348 / 0.600 | — |

**Removing future conditioning does not fix NVT.** The failure is deeper than the
future-conditioning path. Current-Only's 20 ps NVE translation/rotation looked close to
reference; NVT shows they are not preserved. This is the third independent short-NVE false
signal.

## 6. Conclusion: shared pipeline is the suspect

All stable 250k models fail the same way:

| Model | Stable NVT? | Translation | Rotation | Internal |
|---|---|---:|---:|---:|
| 250k Full | yes | 0.538 | 0.786 | 6.044 |
| Current-Only | yes | 0.646 | 0.535 | 6.384 |
| v0 | yes | 0.111 | 0.808 | 6.338 |
| R3-Opt 4.5 Å | yes (2.5 ns) | 0.152 | 0.212 | 7.330 |
| H-C 20k | **no** | — | — | — |

The common factor is not one operator or one conditioning path. It is the shared **data,
training objective, input/target normalization, and evaluation pipeline**. The next step should
be a root-cause audit + a matched native TrajCast NVT control, not another blind architecture
change.
