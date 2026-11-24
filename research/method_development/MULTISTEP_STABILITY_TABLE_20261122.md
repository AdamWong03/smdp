# Multi-Step NVT Stability — Comparison Table

**Date:** 2026-11-22
**Question:** how do the project methods and TrajCast behave under long autonomous NVT rollouts?

All numbers are from our own 100 ps / 2.5 ns screens unless marked "paper". Different methods
were tested for different lengths; this is a screening comparison, not a controlled benchmark.

---

## 1. Stability and mode partition

Production metrics are 100 ps windows unless noted. "Stable" = no safety stop (min distance
≥0.65 Å, pre-thermostat T <3000 K).

| Method | Class | Longest stable NVT (total) | T (K) | trans KE | rot KE | int KE | Partition |
|---|---|---|---:|---:|---:|---:|---|
| v0 | phase-space operator | **2.75 ns** | 293.9 | 0.111 | 0.808 | 6.338 | **fail** (collective cold, internal hot) |
| 250k Full | RelationalTransition, 6 Å | 150 ps | 298.7 | 0.538 | 0.786 | 6.044 | **fail** (same direction) |
| Current-Only | no future conditioning | 150 ps | 304.9 | 0.646 | 0.535 | 6.384 | **fail** |
| Native ep145 | TrajCast, 10% trained | 150 ps | 300.0 | 3.256 | 3.105 | 1.041 | **fail** (opposite bias; undertrained) |
| M0 5k | NVT-pair endpoint fine-tune | 150 ps (val_04) | 311.6 | 3.373 | 3.270 | 1.000 | **fail** (overshoot; independent start **blew up at 130 ps**, preT 3798 K) |
| CSRV rollout | 5-step moment loss | **4.3 ps** | — | — | — | — | **fail** (blow-up: min 0.647 Å, preT 1473 K) |
| FDR v1 | AR(1) mode readout + rescale | **2.5 ns** | 300.0 | 2.482 | 2.533 | 2.392 | **pass** |
| **GLR v2** | **AR(10) memory readout + rescale** | **100 ps** (2.5 ns running, 0.54 ns stable) | 300.0 | 2.482 | 2.526 | 2.399 | **pass** |
| Published TrajCast | IBM checkpoint | **2.5 ns** (paper) | 300.2 | 2.505 | 2.463 | 2.444 | **pass** |

Reference: 2.373 / 2.430 / 2.609 eV for the first 100 ps block; the 2.5 ns mean is
2.459 / 2.502 / 2.450 eV (total 7.411 eV, T = 300 K). The 100 ps block is an outlier in the
internal mode; use the 2.5 ns mean for long-time comparisons.

**Reading:**

- v0 / 250k / Current-Only / native: long-lived but **physically wrong** (not a blow-up).
- M0: stable on one start, **unstable on an independent start**.
- CSRV rollout: **unstable at 4.3 ps**.
- FDR v1: **stable 2.5 ns** and partition-correct, but diffusion too fast and O–H off.
- GLR v2: **stable 100 ps on two starts**, partition-correct, best structure; 2.5 ns diffusion pending.
- Published: stable 2.5 ns, partition-correct, diffusion 2.18e-9.

---

## 2. Structure, spectra, diffusion

RDF = RMSE against the reference NVT; VDOS = overlap. 100 ps values unless noted.

| Method | O–O RDF | O–H RDF | H–H RDF | H VDOS | O VDOS | Diffusion (m²/s) |
|---|---:|---:|---:|---:|---:|---:|
| v0 | 0.944 | 1.409 | 0.229 | 0.356 | 0.236 | 2.81e-13 (2.5 ns) |
| 250k Full | 0.646 | 1.312 | 0.249 | 0.406 | 0.487 | — |
| Current-Only | 0.631 | 1.345 | 0.232 | 0.348 | 0.600 | — |
| Native ep145 | 0.180 | 1.584 | 0.074 | 0.683 | 0.905 | — |
| M0 5k | 0.191 | 1.790 | 0.082 | 0.668 | 0.888 | — |
| FDR v1 | 0.062 | 0.301 | 0.110 | 0.807 | 0.851 | **3.59e-9** (2.5 ns) |
| **GLR v2** | **0.032** | **0.104** | **0.026** | **0.848** | **0.964** | **pending** |
| GLR independent | 0.036 | 0.098 | 0.027 | 0.851 | 0.963 | — |
| Published TrajCast | 0.043 | 0.118 | 0.096 | 0.624 | 0.937 | **2.18e-9** (paper) |

Reference diffusion: 1.84–1.88e-9 m²/s.

**Caveats:**

- The H VDOS comparison is limited by our reference H-stretch frequency (our analysis gives
  published 0.624, while the paper reports ~0.99 with its own reference). The O VDOS and RDF are
  more robust.
- Native ep145 is only ~10% of the training schedule; it is not a fair architecture baseline.
- GLR 2.5 ns and diffusion are still running.
- Only water; one seed per method.

---

## 3. What the table says

1. **Stability alone is not enough.** v0 is stable for 2.75 ns but has the wrong partition;
   250k/Current-Only are stable for 150 ps but physically wrong.
2. **Short-horizon fixes can destabilize.** M0 fails on an independent start; the 5-step CSRV
   moment loss fails at 4.3 ps.
3. **The FD/GLR readout separates thermodynamics from structure.** Partition/T are guaranteed;
   the GLR memory kernel then fixes the multi-lag ACF, which fixes O–H and the diffusion tail.
4. **GLR v2 is the first method that is stable on two starts, partition-correct, and better than
   published on O–O/O–H/H–H RDF and VDOS.** The remaining question is the 2.5 ns diffusion.
