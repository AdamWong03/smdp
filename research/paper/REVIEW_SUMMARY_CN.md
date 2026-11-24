# Phase 6 Mock Review — 中文摘要

**稿件:** *Thermodynamically consistent force-free molecular dynamics via a mode-resolved
generalized Langevin readout*
**Decision:** **Major Revision**
**审稿席:** Journal-Fit (EIC) + Methodology (R1) + Domain (R2) + Perspective (R3) + Devil's Advocate

---

## 1. 五个 reviewer 的共识

1. **Diagnostic 是全文最强贡献。** pointwise-vs-ensemble inversion、NVE red herring、mode
   partition heat map 都清楚且有价值。
2. **结构结果很强且可重复。** 两个 2.5 ns 起点的 partition/RDF/VDOS 几乎一致；100 ps
   length-matched 比较下，O–H 0.104 vs published 0.118，O VDOS 0.964 vs 0.937；2.5 ns
   O–H 进一步到 0.038。
3. **Diffusion 26% low 是核心限制。** 两个 2.5 ns 起点都是 1.39–1.41 vs reference 1.88。
   不能声称 exact recovery。
4. **FD noise 在 λ=0.1 下不是 necessary。** noise-off 对 partition/RDF/VDOS/D 基本无影响；
   memory kernel + rescale 才是 load-bearing components。
5. **单 water model、单 backbone。** “general recipe / backbone-agnostic” 需要 soften 或补
   第二个 backbone/system。

## 2. 两个 CRITICAL（均已 adjudicate）

### DA-C1: “thermodynamically consistent” overclaim

- **Status: VALIDATED.**
- 处理：abstract/Results 2.2 明确定义为 **mean mode kinetic partition + multi-lag ACF**；
  明确不保证 full joint distribution、mode-KE fluctuations、exact transport；D 26% low 作为
  transport limitation 写进 abstract/Results/Discussion。
- Title 可以保留，但定义必须在 abstract 里可见。

### DA-C2: RDF/VDOS 比较没有 length-matched

- **Status: VALIDATED.**
- 处理：abstract 改为 **100 ps GLR vs 100 ps published**（O–H 0.104 vs 0.118；O VDOS
  0.964 vs 0.937）；Table 1 增加 100 ps GLR 行；2.5 ns GLR 单独一行，作为 long-run
  convergence，不再与 100 ps published 直接对比。

## 3. 主要 MAJOR 及处理

| Issue | 处理 | 状态 |
|---|---|---|
| Reference ACF 用于 fit 和 validation 的 circularity | 增加 0–100 ps vs 500–600 ps held-out fit：系数相关 0.999；Results 明确 ACF match is by construction，RDF/VDOS/D 是独立 test | ✅ |
| FD noise not necessary | abstract 不再把 noise 列为核心组件；Results/Discussion 报告 negative ablation | ✅ |
| Single system/backbone | “general recipe” 改为 “demonstrated here on water”；“backbone-agnostic” 改为 “architecture-agnostic in principle; one backbone tested” | ✅ |
| Prior art differentiation | 增加 Table 2（TrajCast / Bigi–Ceriotti / MAMD / GLE / GLR） | ✅ |
| Seed-to-seed D | 100 ps seeds 1.20/1.21/1.45；2.5 ns 1.39–1.41 | ✅ |
| Length-mismatched comparison | 同 DA-C2 | ✅ |

## 4. 还需要作者提供 / 决定

1. **Authors / affiliations / corresponding author / CRediT。**
2. **Code/data repository + DOI + license。**
3. **Supplementary Information**（outline 已有：Notes S1–S7、Figs S1–S7、Tables S1–S4）。
4. **AI-use disclosure**（`paper/AI_DISCLOSURE_DRAFT.md`，需确认工具与用途）。
5. **可选实验**（会增强，但不是 water-scoped claim 的硬性要求）：
   - second backbone（native ep145 + GLR）；
   - second system（SPC/E、TIP4P/2005 或简单液体）；
   - published checkpoint 的 2.5 ns RDF/VDOS（我们停在 3k/510k，太慢）；
   - position-consistent COM head / stochastic rescale，尝试修 26% D；
   - Bigi–Ceriotti action map 的 water 对比。

## 5. 当前判断

论文的核心科学结果足够强：**diagnostic + mode-resolved thermodynamic readout + water
structure/spectrum**。Diffusion 26% low 是明确的 transport limitation，不是致命缺陷，但必须
在 title/abstract/claim 里保持边界。

按 `Major Revision` 的修订路径，**R1–R7 的文本修订已经完成**；剩下的是 R8（authors/code/data/
SI/AI disclosure）和可选的实验增强。完成 R8 后，这篇论文可以作为 NMI method paper 提交。

## 6. Post-review 新结果：second backbone（2026-11-22）

审稿后补了一个关键实验：把**同一套 GLR 系数**直接挂到 **v0 phase-space backbone**
（3.04M 参数，完全不同的架构）。v0 本身 NVT 失败非常严重；加 GLR 后：

| Start | T (K) | partition (eV) | RDF O–O/O–H/H–H | VDOS H/O |
|---|---:|---|---:|---:|
| v0 baseline | 293.9 | 0.111 / 0.808 / 6.338 | 0.944 / 1.409 / 0.229 | 0.356 / 0.236 |
| v0 + GLR (val_04) | 299.99 | 2.482 / 2.529 / 2.396 | 0.042 / 0.109 / 0.019 | 0.855 / 0.969 |
| v0 + GLR (independent) | 300.00 | 2.482 / 2.527 / 2.398 | 0.050 / 0.131 / 0.023 | 0.857 / 0.977 |

这直接支撑 **backbone-agnostic**：同一个 readout 能救回两个完全不同的架构。已写入
Results 2.4 + Table 3 + abstract/discussion。

同时试了 **quartz species basis**（30 fs horizon）作为 cross-material 扩展：

- **full species GLR（memory + noise）是负结果**：30 fs ACF 几乎是 white（prediction error
  0.91），phonon VDOS 从 baseline 0.82 降到 0.48（λ=0.02）。
- **zero-memory limit（只做 species variance rescale）是正结果**：T=301.9 K，Si/O KE 精确
  canonical，phonon VDOS 0.838/0.835（baseline 0.837/0.819），Si–O RDF 0.149（baseline 0.175）。
- 解释：**variance-rescale 组件可以跨材料 generalize；memory kernel 需要 horizon 短于相关
  振动周期**。quartz 需要 5 fs backbone 或 phonon-mode basis，已写成 Discussion 的 design
  principle/limitation。

**当前判断：** two-backbone 结果让论文明显更强；剩下的核心缺口是 **second system**（不是
second backbone）和 26% D。下一步优先级：

1. R8 投稿包（authors/code/data/SI/AI disclosure）；
2. second molecular system（paracetamol 或 SPC/E）——最能补 NMI 的 cross-system 证据；
3. 5 fs quartz backbone（如果要做真正的 cross-material claim，这是必经之路）；
4. position-consistent COM head（如果要修 26% D）。
