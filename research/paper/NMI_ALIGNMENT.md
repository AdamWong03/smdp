# NMI Alignment Audit

**Question:** did we use the skill and reference NMI papers to calibrate decisions/design?
**Answer:** we used the `academic-paper` phase structure, the `academic-paper-reviewer` 5-seat
protocol, and the closest NMI method paper (TrajCast, *Nat. Mach. Intell.* **8**, 764–776, 2026)
as a standard-setting anchor. We did **not** copy its method or claims; we extracted the
structural and evidentiary bar and compared our manuscript against it.

---

## 1. Anchor paper: TrajCast (NMI 2026)

Why this anchor: it is the closest NMI method paper (force-free learned propagator, water/quartz/
paracetamol, NVT/NVE, RDF/VDOS/diffusion). It is the paper our work must differentiate from and
be judged against.

### What TrajCast does structurally

| Element | TrajCast | Our manuscript |
|---|---|---|
| Systems | **3 positive**: paracetamol (gas, 7 fs), quartz (solid, 30 fs), water (liquid, 5 fs); plus zero-shot glassy water | water (2.5 ns, two starts); quartz (30 fs, zero-memory limit); two backbones |
| Main figures | 5–6 multi-panel figures | 4 multi-panel figures + 4 tables |
| Abstract | ~180 words; broad framework claim, supported by three systems | ~200 words; calibrated claim, two backbones + quartz limit + D limitation |
| Baselines | MD reference; horizon/data-efficiency sweep; zero-shot | published checkpoint; 250k; v0; FDR; M0; CSRV; AR order; rescale/noise |
| Error bars | 3 models / 4 blocks; shaded 3σ | 3 seeds; 8 blocks; 2.5 ns two starts |
| Generalization | zero-shot glassy water | two architectures; quartz zero-memory; no zero-shot |
| Data/code | GitHub + Hugging Face + source data | pending repository/DOI |
| SI | Methods, benchmarks, scalability, data efficiency, zero-shot | outline only |

### What this implies for NMI

1. **A method paper needs more than one positive system.** TrajCast's breadth is the bar.
2. **A “framework” claim needs a generality test** (different systems, horizons, or zero-shot).
3. **Source data and code are part of the submission**, not an afterthought.
4. **The abstract can be broad, but each clause is backed by a system.**
5. **Figures are multi-panel and carry the evidence**; tables are supplementary or compact.

---

## 2. Where our manuscript stands

### Stronger than the anchor in one respect

- The **diagnostic** (pointwise vs ensemble, NVE red herring, mode partition) is a conceptual
  contribution TrajCast does not have.
- The **two-backbone rescue** (250k relational and v0 phase-space) is a stronger
  architecture-agnostic test than a single model.
- The **negative controls** (rescale off, FD noise off, displacement crash) are unusually complete.

### Weaker than the anchor

- **One positive condensed-phase system** (water), plus quartz only in the zero-memory limit.
- **No zero-shot / temperature-transfer test.**
- **No scalability sweep** (system size, horizon, training-set size).
- **No code/data repository yet.**
- **Diffusion 26% low**; TrajCast reports 2.18 vs 1.84 (16% high).

---

## 3. Alignment actions taken

| Action | Status |
|---|---|
| Add second backbone (v0 phase-space) | ✅ done; partition/RDF/VDOS rescued |
| Add cross-material test (quartz species basis) | ✅ zero-memory limit positive; full memory negative boundary |
| Calibrate abstract claims to evidence | ✅ done |
| Length-matched RDF/VDOS comparison | ✅ done |
| Held-out reference-fit check | ✅ done (r = 0.999) |
| FD-noise negative result reported | ✅ done |
| Prior-art table (TrajCast/Bigi–Ceriotti/MAMD/GLE) | ✅ done |
| 100 ps seed study + 2.5 ns two starts | ✅ done |
| Cost/throughput reporting | ✅ partial (15.4 ns/day; no scaling sweep) |
| Source-data files for figures | ⬜ pending |
| Code/data repository + DOI | ⬜ pending |
| SI assembly | 🟡 outline only |
| Author contributions / AI disclosure | ⬜ pending |

---

## 4. Remaining NMI-standard gaps

### Must

1. **A second positive system.** Quartz zero-memory is a positive cross-material limit, but the
   full memory readout is not validated there. The strongest next options:
   - **5 fs quartz backbone** (training in progress, ETA ~6.5 h): true cross-material, memory
     kernel resolved; 5 fs species AR(10) prediction errors 0.0137 (O) / 0.0150 (Si) vs 0.91 at
     30 fs. Transitions/cache built; training run `quartz5_relational_20261122`.
   - **SPC/E water** (2–3 days): cross-force-field, rigid translation/rotation-only basis.
   - **Paracetamol** (2–4 days): cross-molecule/elements; the HF data are one-step pairs, so a
     physical NVT reference must be generated/reconstructed first.
2. **Source data** for Figs. 1–4 and Tables 1–4.
3. **Code/data repository** with the fit/rollout/analysis scripts and trajectories.
4. **SI** with methods, benchmarks, additional ACF/RDF/VDOS, seed tables.

### Suggested

5. **Scalability sweep**: ns/day vs system size, horizon, and training-set size (TrajCast Fig. 6
   analogue).
6. **Zero-shot / temperature-transfer test**: e.g., run the water GLR at 280/320 K with the same
   coefficients, or the glassy-water cooling protocol.
7. **Figure consolidation**: 4 figures + 4 tables is acceptable, but NMI often prefers 5–6
   multi-panel figures with source data.

---

## 5. Bottom line

We are **aligned in structure and evidentiary style** (multi-panel figures, baselines, error bars,
SI, data/code), and **ahead on the diagnostic**. We are **behind on system breadth**: one positive
condensed-phase system plus a zero-memory solid limit is not yet TrajCast-level breadth. The
paper is a credible NMI method paper if we (i) keep the calibrated claims, (ii) add the second
positive system or explicitly frame the work as a water method paper with a cross-material limit,
and (iii) provide source data/code/SI.

The next recommended move is a **5 fs quartz backbone** (true cross-material) or **SPC/E water**
(faster, cross-force-field). Paracetamol is attractive chemically but needs a physical NVT
reference that is not in the local HF download.
