# SMDP/DSRT — ICML 2027 paper-oriented status audit

**Date:** 2026-11-26  
**Mode:** research-status audit, not open topic selection, not new architecture exploration.  
**Scope:** all formal SMDP/DSRT experiments available on the server and GitHub snapshot; no new
large training launched.  
**Bottom line:** the paper's real method is the **frozen R3-Opt backbone + mode-resolved GLR
readout**, not the raw DSRT/phase-space/minimal-joint network. The raw architecture is a plug-in;
the contribution is the readout and the pointwise-vs-ensemble diagnostic. The best water result is
the deterministic GLR at 2.5 ns (partition/RDF/VDOS strong, **D 25–26% low**, KE variance
suppressed); the stochastic GLR restores the KE variance but breaks GK/MSD consistency. The single
priority optimization is a **self-consistent stochastic GLE readout** (GK-constrained memory kernel
+ FD-consistent colored noise) that removes the ad-hoc variance rescale and targets the
variance/GK–MSD/diffusion trade-off. The first step is an offline kernel fit; no backbone training.

**Execution update (2026-11-26):** the offline study shows that **3 DHO modes + slow mode are
sufficient** on the 25-block mean ACF (integral 14.34 vs 14.45 fs); 4–6 modes improve the ACF
shape but not the integral. The dominant problem is the **calibration window**: the 100 ps
reference blocks have ACF integrals **11.34–18.00 fs** (mean 14.45, std 1.86), and the current GLR
AR(10) is fitted on a 100 ps block with only a 200 fs ACF window. The GLR's 25% D deficit is
therefore at least partly a **calibration/transport-window bias**, not proof that the readout
cannot recover D. A naive rank-1 AR(2) discretization does not reproduce the DHO ACF; the correct
discrete implementation needs an exact complex-OU/2×2-noise realization. The single optimization is
now: **full-reference (2.5 ns mean) calibration + exact discrete complex-OU implementation of the
3-mode DHO+slow readout**, then a two-start 100 ps test. No 100 ps GLE rollout was started because
the discrete offline gate has not passed.

---

## 0. 中文摘要（给研究负责人）

- **最佳候选不是 raw DSRT，也不是 phase-space v0 / minimal-joint。** 论文的方法是：
  **frozen R3-Opt（2,357,616 params）+ mode-resolved GLR readout**。R3-Opt 只是 backbone，
  可以被替换（v0 + GLR 也能被救活）；GLR 才是方法贡献。
- **Water 最佳结果**：R3-Opt 6 Å / 5 fs + deterministic GLR（AR(10), λ=0.1, per-group
  rescale, COM projection, decoupled positions），2.5 ns、两个 start：
  partition 2.482/2.526/2.399 eV，RDF 0.031/0.038/0.028，VDOS 0.847/0.953，**D 1.406/1.393
  (25–26% low)**，GK≈MSD。问题：deterministic rescale 把 total KE std 压到 0.0097 eV
  （reference 0.440）；stochastic GLR 把 KE std 恢复到 0.437/0.429，但 GK 1.16–1.21 与
  MSD 1.39–1.44 不一致。
- **Quartz 最佳结果**：R3-Opt 30 fs / 4.5 Å，15 ns NVT，T 300.23/300.60 K，VDOS 0.946/0.950，
  O–O RDF 0.024，Si–O 0.180；5 fs + phonon-band stochastic rescale（memory off），100 ps、
  两个 seed：T 299.8/299.9，KE std 0.397/0.399，VDOS 0.950/0.955，RDF 0.028/0.016/0.033。
- **Water 失败机制**：不是实现 bug，不是 thermostat bug，不是 NVE 能量漂移，也不是
  pointwise 误差或 signed-KE 偏差本身。Published TrajCast 的 pointwise error 大约是我们的
  15×，signed KE +0.083 eV/step（我们 +0.0062），却通过 100 ps NVT；我们的模型 pointwise
  更准、signed KE 更小，却失败。**决定性变量是 ensemble/conditional-mean mismatch：
  模型学了条件均值，而不是条件分布。** 这直接否定了 A1/A2、on-policy signed-KE、
  post-hoc energy rescale 作为主修复。
- **唯一优先优化（执行后修正）**：离线研究显示 **3 个 DHO + slow mode 已足够**（mean ACF
  积分 14.34 vs 14.45 fs），4–6 mode 只改善形状、不改善积分；真正的问题是 **校准窗口**：
  100 ps reference blocks 的 ACF 积分在 **11.34–18.00 fs** 之间（mean 14.45，std 1.86），
  当前 GLR AR(10) 只用了 100 ps block + 200 fs ACF 窗。因此 25% D deficit 至少部分是
  **calibration/transport-window bias**，不能直接归因于 readout 无能。下一步不是加模态，
  而是：**用 2.5 ns mean ACF 重新校准 + 实现 exact complex-OU/2×2-noise 离散化**，
  再跑 two-start 100 ps。naive rank-1 AR(2) 离散化在 100 fs 不匹配，**未通过离线 gate，
  所以没有启动 100 ps GLE rollout**。
- **ICML 缺口**：官方 TrajCast 2.5 ns 与 converged native TrajCast；one-rule memory
  diagnostic；reference 8-frame energy check（frame 28,571 失败）；2.5 ns quartz；
  external baselines（FlashMD 视协议适用性，Bigi/MAMD 若有代码）；claim 只能写
  mean partition + structure/VDOS，不能写 exact diffusion、full KE variance、one rule 或
  NVE accuracy。

**ICML 2027 decision:** **GO, conditional**. The paper has a defensible ML method and a strong
2.5 ns water result, but only if (i) the self-consistent GLE kernel passes the offline fit and
the 100 ps gate, **or** the paper explicitly scopes D 25% low and mean-only KE variance as
limitations; (ii) the one-rule memory diagnostic is validated **or** the claim is scoped to a
framework with physics-defined bases; (iii) the official TrajCast 2.5 ns baseline and a converged
native TrajCast baseline are run; (iv) the reference 8-frame energy check is fixed. If (i) fails,
the deterministic GLR is still submittable as a methods paper with quantified limitations, but
the claim must be narrowed. If (ii) fails, the paper is a two-system framework paper, not a
one-rule method paper.

---

## 1. Latest experiment inventory (server, 2026-11-26)

No SMDP/DSRT training or rollout is currently running. `systemctl --user` shows failed
`dsrt-current-only-water6-20261117` and `dsrt-phase-water5-pipeline-20261120` units; no active
`train_relational`/`rollout_relational`/`gate2` process. The latest completed formal run is the
R0 response study (2026-11-26). The new open-topic Stage 1 response-uncertainty work is separate
and is not part of this project.

### 1.1 Synced vs unsynced

- **GitHub:** `AdamWong03/smdp`.
- **Latest synced method snapshot:** branch `research/h1h2-selection-20261120` at
  `4399ccd`, tag `research-snapshot-20261124` (pushed 2026-11-24 05:42 UTC).
- **Synced:** `research/` docs/evidence up to 2026-11-24, including GLR/quartz evidence
  (`research/evidence/water_det_val04_2p5ns_report.json`,
  `water_stoch_*_2p5ns_report.json`, `quartz_band_seed{1,2}_report.json`), `reports/`,
  `configs/`, code. The remote tree has 774 files, including an older 8-file `paper/`
  (`MANUSCRIPT.md`, `RESULTS_REGISTRY.md`, etc.).
- **Unsynced / local only:** Gate 2 pilot and postmortem (2026-11-25), CAM Gate 0 (2026-11-25),
  R0 response study (2026-11-26), the current `paper/` submission artifacts
  (`CLAIM_BOUNDARY.md`, `NOVELTY_MATRIX.md`, `EVIDENCE_MAP.md`, `PHONON_GLR_DECISION.md`, etc.),
  `experiments/` artifacts, and many analysis outputs. The local
  `/data2/.../foundation_mlip_adaptation` is **not a git repo**.
- **Action before ICML:** freeze a new tag containing the GLR code, coefficients, analysis
  scripts, reference data pointers, and the submission claim boundary. Do not rent GPUs against
  an uncommitted method.

### 1.2 Formal experiment campaigns

Status key: ✅ positive · ❌ negative · ⚠️ mixed/near-miss · 🚫 invalid/aborted · ⏸ not run.

| Campaign | Representative runs | Status | Key result |
|---|---|---|---|
| R3-Opt water 5 fs 4.5 Å | `water5_20261103_r3_opt` | ❌ | 100 ps NVT: trans 0.152, rot 0.212, int 7.330 eV; O–O 11.46; VDOS 0.215/0.233; D plateau. Raw backbone fails. |
| R3-Opt water 6 Å 5 fs | `water6_reference_20261113/dsrt_radius6` | ❌ | 100 ps NVT: trans 1.366, rot 1.538, int 4.418 eV; still wrong partition; radius not sufficient. |
| Current-Only 250k | `nvt_screen_current_only_250k_20261120` | ❌ | trans 0.646, rot 0.535, int 6.384 eV; future conditioning is not the sole cause. |
| R3-Opt water 1 fs 6 Å | `water6_1fs_20261123` | ❌ | one-step pos 3.16e-5 Å, vel 4.58e-5 Å/fs; raw + CSVR fails both starts (trans 3.24/3.19, rot 3.16/3.17, int 1.18/1.22; O–H 1.56/1.53; D 5–8× high). |
| R3-Opt quartz 30 fs 4.5 Å | `relational_opt_quartz30_20261103` | ✅ | 15 ns NVT: T 300.23/300.60; VDOS 0.946/0.950; O–O 0.024; Si–O 0.180; NVE drift +1.30/+0.69 eV/100 ps. |
| R3-Opt quartz 5 fs 4.5 Å | `quartz5_relational_20261122` | ✅/⚠️ | test pos 7.7e-5 Å, vel 2.8e-5 Å/fs; baseline NVT cold (235 K); phonon-band rescale fixes it (below). |
| Phase-space v0 | `phase_space_20261120` | ❌ | water 2.5 ns NVT: trans 0.111, rot 0.808, int 6.338; VDOS 0.356/0.236; D 2.81e-13. Quartz mixed/worse; 1.9–2.6× slower. |
| GLR/FDR readout | `glr_2p5ns_20261122`, `glr_independent_2p5ns_20261122` | ✅/⚠️ | water 2.5 ns deterministic: partition 2.482/2.526/2.399; RDF 0.031/0.038/0.028; VDOS 0.847/0.953; D 1.406/1.393 (25–26% low); total KE std 0.0097. |
| Stochastic GLR | `glr_stoch_2p5ns_20261122`, `glr_stoch_independent_2p5ns_20261122` | ⚠️ | KE std 0.437/0.429; partition/RDF/VDOS good; MSD D 1.435/1.387; GK 1.206/1.16 → GK/MSD inconsistent. |
| Slow-mode / rescale variants | `glr_slow_*_100ps_20261123` | ⚠️/❌ | slow-stoch 100 ps: KE std 0.4405, D 1.480 (21.4% low), GK 1.386 vs MSD 1.480, O–H 0.1253; slow-global D 1.246; slow-weak AR1 total KE std 0.318. |
| Quartz phonon-band rescale | `quartz5_phonon_bands_rescale_only_100ps_20261123`, `...seed2...` | ✅ | T 299.8/299.9; KE std 0.397/0.399; VDOS 0.950/0.955; RDF 0.028/0.016/0.033. |
| Quartz full phonon memory | `quartz5_phonon_norescale_100ps_20261123`, `...species...`, `...glr...` | ❌ | VDOS 0.594/0.686 (memory only), 0.734/0.838 (memory+species), 0.209/0.253 (per-mode). Memory hurts because the 5 fs backbone ACF is already correct. |
| Minimal-joint Gate 1 | `gate1_minimal_joint_100ps_20261124` | ❌ | partition 3.301/3.187/1.046; O–H 1.808; D 9.82e-9; failure within 5 ps. |
| Minimal-joint Gate 2 A1/A2 | `gate2_pilot_A1_20261125`, `gate2_pilot_A2_20261125` | ❌ | A1 endpoint improves 1.08e-3→0.813e-3 but 1 ps rollout collapses by step 50; A2 diverges by step 39–40; pilot stopped. |
| CAM | `cam_gate0_toy_20261125` | ❌ | exact Gibbs invariance false in general; symplectic Euler+momentum q var 0.571; NO-GO. |
| R0 response | `r0_water_*_20261126`, `r0_quartz_*_20261126` | ❌ | no stable separation of success/failure; published +0.019/−0.237/+0.485 but correct ensemble; R3-Opt +0.017/+0.015/+0.010 but wrong; NO-GO. |
| Published TrajCast control | `nvt_screen_published_trajcast_20261121` | ✅ | 100 ps: trans 2.505, rot 2.463, int 2.444; T 300.2; O–O 0.043; O–H 0.118; VDOS O 0.937; D 2.18e-9 (16% high). |
| Native TrajCast control | `nvt_screen_native_trajcast_20261120` | ⚠️ | ep145/1500 only: trans 3.256, rot 3.105, int 1.041; O–O 0.180; not a fair baseline. |
| Training/reference NVT | `gate2_training_nvt_20261125`, `independent_nvt_500ps_20261121` | ✅/⚠️ | 200 ps training NVT and 500 ps independent NVT completed; 8-frame energy check fails at frame 28,571; reference verification incomplete. |
| H1 selection proxy | `h1h2_endpoint_20k_20261120`, `h1h2_rollout_20k_20261120` | ❌ | both arms selected epoch 32; null at 20k; rollout proxy noisy. |
| E1 motion loss | `e1_baseline_20k_20261120`, `e1_motion_20k_20261120` | ❌ | motion arm heats to 400–420 K; internal 5.396 vs 2.873; worse physics. |
| M0 ensemble-matched | `m0_nvt_finetune_5k_20261121` | ❌ | bias flips and overshoots (3.37/3.27/1.00); independent start fails at 130 ps. |
| Composed CSVR rollout | `csrv_rollout_finetune_500_20261121` | ❌ | failed at 4.3 ps; short-rollout moment loss does not constrain stability. |
| Covariance matching | `covariance_match_20261123_v2.json` | ❌ | static covariance matched, KE std 0.105 vs 0.240; GK unchanged. |
| 5 fs/1 fs GLE fits | `velocity_gle_fit_20261123`, `...1fs_m4/m5...` | ❌ | 3-mode 5 fs misses negative dip/long tail; 4–5 mode 1 fs ACF@5 fs 0.315/0.346 vs 0.981. |

### 1.3 Running / aborted

- **Running:** none.
- **Aborted:** `v0_stoch_*_100ps_20261123` (not needed; water production deterministic);
  `slow-weak-100ps-20261123` (invalid EMA Gamma variance; rerun as `slow-weak-ar1`);
  `gate2_pilot_A2` stopped at step 40; native TrajCast stopped at ep145/1500.
- **Not run:** self-consistent GLE kernel (the proposed optimization); published TrajCast 2.5 ns;
  converged native TrajCast; quartz 2.5 ns; one-rule memory diagnostic; reference verification
  fix.

---

## 2. Best candidate: evidence

### 2.1 Core network and training configuration

**Backbone (best water):**

- DSRT `RelationalTransition` / `geometry_gated`, **2,357,616 parameters**.
- scalar 192, vector 48, edge 96, radial 16; cutoff **6 Å**; horizon **5 fs**; proposal 2 layers,
  refinement 2, `velocity_scale=0.01`.
- batch 8, FP32, AdamW lr 3e-4, **250,000 optimizer steps**, proposal weight 0.02, grad clip 10,
  warmup-cosine (5k warmup), **endpoint-only selection** (normalized final position/velocity MSE).
- Water training cache: `water6_reference_20261113`; 5,000/1,250/1,000 train/val/test.

**Backbone (best quartz):**

- Same architecture; cutoff **4.5 Å**; horizon **30 fs** (15 ns NVT result) or **5 fs** (the
  phonon-band rescale result); same 250k budget/config.
- Quartz 30 fs config: `code/h1h2/configs/quartz30_r3_opt.json`; 5 fs config:
  `experiments/quartz5_relational_20261122`.

**Readout (the method):**

- Physics-defined mode basis: molecular translation/rotation/internal for water; phonon bands
  for quartz.
- AR(10) Yule–Walker memory, fitted from the reference mode ACF; blend **λ=0.1**; COM projection;
  per-group variance rescale to \(k_BT/2\) per dof; **decoupled positions** \(q'=q+\Delta q\);
  no force at inference.
- Water production: molecular modes + memory + **deterministic** rescale (GK/MSD consistent;
  mean-only KE variance).
- Quartz: phonon-band partition + **stochastic** rescale + **memory off** (the 5 fs backbone ACF
  is already correct; full memory is negative).

### 2.2 Water evidence table

| Model / protocol | trans KE | rot KE | int KE | total KE std | O–O RDF | O–H RDF | H–H RDF | VDOS H/O | D (MSD) | GK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Reference NVT 2.5 ns | 2.459 | 2.502 | 2.450 | 0.440 | 0 | 0 | 0 | 1/1 | 1.883 | 1.836 |
| Published TrajCast 100 ps | 2.505 | 2.463 | 2.444 | — | 0.043 | 0.118 | 0.096 | 0.624/0.937 | 2.18 (paper) | — |
| Raw R3-Opt 5 fs 4.5 Å 100 ps | 0.152 | 0.212 | 7.330 | — | 11.46 | — | — | 0.215/0.233 | plateau | — |
| Raw R3-Opt 6 Å 100 ps | 1.366 | 1.538 | 4.418 | — | — | — | — | — | — | — |
| Current-Only 250k 100 ps | 0.646 | 0.535 | 6.384 | — | — | — | — | 0.348/0.600 | — | — |
| Phase-space v0 2.5 ns | 0.111 | 0.808 | 6.338 | — | 0.944 | 1.409 | 0.229 | 0.356/0.236 | 2.81e-13 | — |
| **GLR deterministic 2.5 ns** | **2.482** | **2.526** | **2.399** | **0.0097** | **0.031** | **0.038** | **0.028** | **0.847/0.953** | **1.406** | **1.420** |
| GLR independent 2.5 ns | 2.482 | 2.527 | 2.398 | — | 0.028 | 0.044 | 0.027 | 0.847/0.953 | 1.393 | — |
| Stochastic GLR 2.5 ns | 2.483 | 2.525 | 2.398 | **0.437** | 0.028 | 0.054 | 0.024 | 0.852/0.949 | 1.435 | **1.206** |
| Stochastic GLR independent | 2.482 | 2.528 | 2.398 | **0.429** | 0.028 | 0.048 | 0.027 | 0.853/0.950 | 1.387 | **1.16** |
| Slow-stoch 100 ps | 2.483 | 2.529 | 2.399 | 0.4405 | 0.0355 | 0.1253 | 0.0218 | 0.858/0.970 | 1.480 | 1.386 |

The deterministic GLR is the best **mean-stationary** result: mean partition, RDF, VDOS, and
initial-condition independence are strong; D is 25–26% low; total KE std is 45× too small. The
stochastic GLR fixes the variance but breaks GK/MSD consistency. The slow-stoch 100 ps is the
closest full-gate near-miss (D 21.4% low, GK/MSD 6.3%, O–H 0.1253) but is water-specific.

### 2.3 Quartz evidence table

| Model / protocol | T (K) | total KE std | Si/O KE per atom | RDF Si–O/O–O/Si–Si | VDOS O/Si |
|---|---:|---:|---:|---:|---:|
| R3-Opt 30 fs 15 ns | 300.23/300.60 | 19.95/19.18 (K std) | 4.212/2.036 (total O/Si) | 0.180/0.024/— | 0.946/0.950 |
| 5 fs baseline 100 ps | 235.3 | 0.4355 | 0.0298/0.0305 | 0.271/0.096/0.133 | 0.918/0.944 |
| **5 fs phonon-band stochastic, seed 1** | **299.8** | **0.397** | 0.03846/0.03855 | **0.028/0.016/0.033** | **0.953/0.949** |
| 5 fs phonon-band stochastic, seed 2 | 299.9 | 0.399 | 0.03846/0.03855 | 0.031/0.018/0.040 | 0.950/0.955 |
| Species stochastic | 301.68 | 0.4012 | exact | 0.0245/0.0169/0.0380 | 0.898/0.955 |
| Deterministic species | 301.86 | 0.0036 | exact | 0.078/0.022/0.055 | 0.866/0.914 |
| Full phonon memory | 301.9 | 0.4024 | — | 0.197 | 0.734/0.838 |
| Per-mode rescale | 299.6/299.8 | 0.398/0.401 | exact | 0.027/0.016/0.027 | 0.217/0.269 |

**Quartz is a rescale-only system** at 5 fs. Memory hurts because the 5 fs backbone's mode ACF is
already correct (100 fs error 0.03–0.04), while water's is not. This is both a positive result
and the source of the one-rule gap.

### 2.4 Necessary components vs unsupported claims

| Component / claim | Verdict | Evidence |
|---|---|---|
| Physics-defined mode basis | **necessary** | molecular modes for water; phonon bands for quartz; Cartesian/species controls worse |
| Per-group variance rescale | **necessary** | rescale-off: T 172 K, partition 0.76/0.65/2.84, RDF 0.398/0.474/0.279 |
| AR(p) memory for water | **necessary** for internal/translation ACF | p1 O–H 0.301 vs p10 0.104; p10 2.5 ns good |
| Memory for quartz 5 fs | **not necessary / harmful** | full phonon memory VDOS 0.594/0.686; 0.734/0.838 with rescale |
| FD noise | **not necessary at λ=0.1** | noise-off: partition/RDF/VDOS/D unchanged; theoretically consistent but not empirically load-bearing |
| Blend λ=0.1 | **necessary** | λ=1 FDR v1 internal spectrum fails (O–H 0.957) |
| Decoupled positions | **necessary** | displacement correction crashes at 24.8 ps |
| Frozen backbone / training-free readout | **supported** | v0 phase-space backbone + GLR rescued; no backbone gradients |
| **DSRT architecture as the method** | **not supported** | R3-Opt raw fails water; v0 raw fails; published TrajCast passes with a different architecture |
| **Phase-space operator / motion modes** | **not supported** | v0 worse endpoint, mixed physics, slower; GLR rescue is backbone-agnostic, not v0-specific |
| **Minimal-joint q/v consistency** | **falsified** | Gate 1 100 ps fails; Gate 2 A1/A2 fail |
| **Future conditioning** | **not the cause** | Current-Only fails similarly |
| **Graph radius 6 Å** | **partial** | improves rotation/multistep; translation still wrong |
| **NVE energy stability** | **red herring** | published TrajCast is least NVE-stable but NVT-correct; base 250k most stable but fails |
| **Pointwise one-step accuracy** | **neither necessary nor sufficient** | published 15× larger pointwise error, correct ensemble; ours smaller, wrong |
| **Signed mode-KE bias as the main fix** | **falsified** | published +0.083 eV/step passes; ours +0.0062 fails; on-policy signed-KE is not the controlling variable |
| **One rule across water/quartz** | **not established** | water: molecular modes + memory + deterministic rescale; quartz: phonon bands + no memory + stochastic rescale |
| **Canonical partition guarantee** | **mean-only** | deterministic GLR total KE std 0.0097 vs 0.440; stochastic fixes variance but breaks GK/MSD |

---

## 3. Water failure attribution

### 3.1 Excluded implementation issues

- State transition, CSVR, PBC, graph rebuild, q/v timing: **no confirmed bug** (Gate 2
  postmortem, matched A0/A1).
- CSVR: standalone unit test matches stationary T/KE; no thermostat bug.
- Safety stop: not a false positive; A1 bond breaking is real.
- Checkpoint metadata `epoch` bug: minor, not dynamics.
- Reference/transition alignment: positions/velocities match source frames; no time offset.
- Training data partition: raw NVE is mode-equilibrated; no inherited bias.

### 3.2 Failed optimization schemes

- Endpoint-only / A1: lower endpoint, closed-loop collapse.
- A2: mode-KE std + ACF loss diverges; std term noisy/dominant.
- M0 ensemble-matched endpoint: bias flips/overshoots; independent start fails.
- Composed CSVR short-rollout loss: fails at 4.3 ps.
- Covariance matching: static covariance matched, temporal variance/D not.
- Slow-mode/rescale variants: near-miss or fail; per-mode/global/weak-coupling trade-offs.
- PSD GLE kernel fits: 3-mode 5 fs and 4–5 mode 1 fs cannot fit short decay + negative dip + long
  tail.
- Langevin+CSVR: destroys partition/structure.
- E1 motion-balanced loss: heats; proposal inflation/equal weights.
- Mode-aware/rollout correction: correction too small / 2-step blind.
- Minimal-joint: kinematic consistency alone does not fix closed-loop partition.
- Phase-space v0: raw water fails; GLR rescue is not v0-specific.
- CAM: exact Gibbs invariance false; no independent advantage.
- R0 mode response: no stable success/failure separation.
- **On-policy signed-KE**: not tested as training, but **falsified as the main hypothesis** by the
  published TrajCast control (larger signed error, correct ensemble).

### 3.3 Remaining possible causes

1. **Ensemble / conditional-mean mismatch (strongest).** The model learns the conditional mean;
   the published TrajCast behaves generatively (different trajectory, same ensemble). Pointwise
   and signed-KE errors are not the controlling variables. This is supported by the 15× pointwise
   paradox and the +0.083 vs +0.0062 signed-KE paradox.
2. **Readout distribution correction is incomplete.** The GLR fixes the mean partition and
   structure, but the deterministic rescale suppresses the KE variance; the stochastic rescale
   restores variance but changes the translation ACF/GK. The remaining object is the **full
   mode-resolved stationary distribution**, not just its mean.
3. **Translation ACF integral / diffusion.** The GLR translation ACF at 100 fs is too small
   (0.011–0.014 vs 0.031–0.039), so D is 25% low; GK and MSD agree, so the position update is
   consistent; the bottleneck is the memory kernel's long-lag tail.
4. **Training objective/model selection.** Endpoint MSE + 0.02 proposal, endpoint-only selection;
   H1 proxy null at 20k; E1 negative. The objective is blind to the stationary measure.
5. **Reference/native baseline incomplete.** 8-frame energy check fails at frame 28,571; native
   TrajCast ep145/1500 is not a fair architecture baseline; published 2.5 ns not run.
6. **Architecture/backbone inductive bias.** Raw R3-Opt and v0 fail, but published TrajCast shows
   an offline direct propagator can pass; this is a training/ensemble issue, not a fundamental
   limitation of force-free propagation.

### 3.4 What the published TrajCast control proves

- A direct, force-free, offline 5 fs propagator **can** reproduce the correct water NVT
  partition/structure under our protocol.
- Pointwise accuracy is **neither necessary nor sufficient**: published has ~15× larger pointwise
  error, +0.083 eV/step signed KE, yet passes; 250k has smaller errors, yet fails.
- The failure is **not** a fundamental long-time limitation, not a thermostat bug, not a
  representation bug; it is an **ensemble/training-objective** problem.
- The paper's defensible claim is **ensemble fidelity, not trajectory accuracy**.
- The **published checkpoint is the external baseline**; the native ep145 is not.

---

## 4. Single priority optimization: self-consistent stochastic GLE readout

### 4.1 Decision

**Optimize the GLR readout, not the backbone.** Replace the current ad-hoc variance rescale with
a **self-consistent mode-resolved GLE**:

\[
\dot v_m(t) = -\int_0^t K_m(t-s)v_m(s)\,ds + \eta_m(t),
\]

with a positive spectral density \(K_m(t)\) fitted to the reference mode ACF and constrained to
the Green–Kubo diffusion coefficient; \(\eta_m\) satisfies the fluctuation–dissipation relation.
The stationary distribution is canonical **by construction**, so no post-hoc variance rescale is
needed. The readout is blended with the frozen backbone at the existing λ=0.1; COM projection and
decoupled positions are unchanged.

This targets the **two coupled failures** of the best candidate:

- deterministic GLR: KE variance suppressed (total KE std 0.0097 vs 0.440);
- stochastic GLR: KE variance correct but GK/MSD inconsistent (1.16–1.21 vs 1.39–1.44);
- both: D 25–26% low due to the translation ACF integral.

It is **not** A1/A2 (no backbone training, no endpoint/magnitude loss), **not** a post-hoc energy
rescale (the variance follows from the GLE stationary distribution, not a rescale step), and
**not** a new architecture (same mode basis, same frozen backbone, same blend).

### 4.2 Why this mechanism is the right one

- The GLR already has the correct **mean** partition and structure; the remaining error is the
  **distribution** (variance, ACF tail, D). The GLE is the minimal object that fixes the
  distribution by construction.
- The current rescale is the source of the trade-off: deterministic rescale removes the variance;
  independent stochastic rescale adds white noise that lowers the GK integral. A colored noise
  tied to the memory kernel preserves the ACF while giving the correct variance.
- The translation ACF is the measured bottleneck (100 fs 0.011 vs 0.031–0.039; 5 ps integral
  10.258 vs 11.341 fs). A GK-constrained kernel is the principled fit.
- The DHO kernel is already validated for quartz phonon linewidths (1–5 THz ACF error 0.035–0.042
  vs AR(10) 0.14–0.20), so the representation is physically appropriate; the water translation
  ACF has a negative dip/long tail that a 3-mode fit missed. Use 4–6 modes + slow mode + GK
  constraint, not a 3-mode fit.
- The optimization is **backbone-agnostic**: it applies to R3-Opt and, if needed, v0; it does not
  depend on the raw architecture.

### 4.3 Implementation

1. **Kernel fit (CPU, first gate).**
   - Reference: 5 fs water NVT translation/rotation/internal mode ACFs, 2.5 ns.
   - Kernel: \(K_m(t)=\sum_{j=1}^{J} A_j e^{-t/\tau_j}\cos(\omega_j t+\phi_j)\) with PSD check, plus
     one slow OU mode; \(J=4\)–6.
   - Constraints: match ACF at 5/25/100/200/500/1000/2000/5000 fs; match the GK integral
     \(D=\int_0^\infty C(t)dt\); zero-frequency constraint; positive spectral density.
   - Output: Markovian embedding coefficients (auxiliary variables) for the readout, not a
     post-hoc rescale table.
2. **Readout integration.**
   - Replace the deterministic/stochastic rescale in `glr_readout.py` with the GLE auxiliary
     variables and FD-consistent noise.
   - Same mode basis, λ=0.1, COM projection, \(q'=q+\Delta q\), frozen R3-Opt 6 Å checkpoint.
   - No new hyperparameter sweep; the kernel is fitted, not tuned on the rollout.
3. **Formal comparison (100 ps, two starts).**
   - Arms: deterministic GLR (current production), stochastic GLR (current ablation), slow-stoch
     (near-miss control), self-consistent GLE (new), published TrajCast (external).
   - Metrics: partition mean/std, RDF, VDOS, MSD D, GK D, translation/rotation/internal ACF,
     stability, cost.
   - Pre-registered pass: T 300±3 K; total KE std within reference block scatter; mode KE means
     within 5%; D within 20% of reference and GK/MSD within 20%; RDF/VDOS no worse than
     deterministic GLR.
   - If 100 ps passes on two starts, run 2.5 ns on two starts; compare to deterministic GLR and
     published TrajCast.
4. **Quartz.**
   - The kernel fit is water-specific; the one-rule diagnostic (§5) decides whether quartz uses
     memory. The existing phonon-band stochastic rescale remains the quartz production result
     until the diagnostic says otherwise.
   - Do not force the water kernel onto quartz; that would violate the physics-defined basis.

### 4.4 Minimal decisive experiment

**Offline kernel fit only, CPU.** Before any GPU rollout:

- Fit the 4–6 mode DHO + slow-mode kernel to the 5 fs water translation ACF with the GK/D
  constraint.
- Check ACF at 5/25/100/200/500/1000/2000/5000 fs; check \(D\) and the 5 ps ACF integral.
- Compare to the current AR(10) fit (ACF 0.981/0.640/0.014/−0.030 vs reference
  0.982/0.658/0.039/−0.019; 5 ps integral 10.258 vs 11.341 fs) and to the failed 3-mode fit
  (200 fs −0.041 vs −0.080; 500 fs +0.006 vs −0.022).
- **Kill:** if the kernel cannot match the negative dip/long tail and the D constraint
  simultaneously, stop; the paper reports D 25% low and the mean-only KE variance as limitations.
- **Go:** if the fit passes, integrate and run the 100 ps two-start comparison; 2.5 ns only if
  100 ps passes.

This is the decisive evidence: whether a self-consistent GLE can represent the water translation
ACF with the correct D. If it cannot, no amount of rollout tuning will fix the readout; if it
can, the variance/GK–MSD trade-off is addressable without a new backbone.

### 4.5 Why not the alternatives

| Alternative | Why not the single priority |
|---|---|
| On-policy signed mode-KE training | Falsified as the main hypothesis by published TrajCast: larger signed error (+0.083) yet correct ensemble; our smaller (+0.0062) yet wrong. |
| A1/A2 retraining | Already failed; endpoint/magnitude losses blind to the ensemble; do not rename. |
| Post-hoc energy/rescale variants | The current trade-off is exactly the rescale problem; another rescale would repeat it. |
| M0 ensemble-matched endpoint | Failed: bias flips/overshoots, unstable. |
| Mode-resolved thermostat only | A control, not a method; it masks the state-distribution problem. |
| New generative/stochastic head | The right long-term direction but a new architecture; not the short-term convergence, and not needed until the GLE readout is tested. |
| One-rule memory diagnostic | Paper-critical, but a protocol/claim fix; it does not fix the water variance/D trade-off. Do it as the ICML experiment gap, not the single optimization. |
| Longer/better native TrajCast | Baseline completion, not our method. |

### 4.6 Expected outcome and risk

- **Success:** water 2.5 ns with mean+std partition, RDF/VDOS, D within 20%, GK/MSD consistent;
  the paper's main physical limitation is removed; the method claim becomes "self-consistent
  mode-resolved GLE readout for force-free propagators."
- **Partial:** variance fixed but D still low, or D fixed but variance still off; report the
  trade-off with the kernel fit; the paper can still submit with the deterministic GLR as main
  and the GLE as an ablation/limitation.
- **Failure:** kernel cannot fit the ACF/D; stop the optimization; submit the deterministic GLR
  with D 25% and mean-only variance as explicit limitations. Do not invent another rescale.

---

## 5. ICML 2027 experiment gaps

### 5.1 External method baselines

| Baseline | Status | Required action |
|---|---|---|
| **Official TrajCast (published checkpoint)** | 100 ps PASS under our protocol; 2.5 ns not run | **Mandatory.** Run 2.5 ns water under the same protocol; report partition/RDF/VDOS/D/ACF. This is the external baseline the paper must beat or match. |
| **Native TrajCast (our reproduction)** | ep145/1500 only; opposite bias; not fair | **Required for architecture/training separation.** Complete to convergence (float32 resume, 5–8× faster) or explicitly state it is incomplete and use the published checkpoint as the only TrajCast baseline. |
| **FlashMD** | NeurIPS 2025; long-stride direct predictor + action-map correction; code/checkpoint availability not audited | **Conditional.** Include only if the released model supports the same water/quartz systems and 5/30 fs protocol; otherwise cite/discuss, do not force a reproduction. |
| **Bigi/Spies/Ceriotti action map (PRL 2026)** | closest structure-preserving correction; code availability not audited | Cite; run on water if code is available; otherwise conceptual comparison. |
| **MAMD (AIP Advances 2026)** | closest learned velocity-update correction; code availability not audited | Cite; run if code is available; otherwise conceptual comparison. |
| **GLE/GLE4MD thermostats** | physics prior art | Cite/distinguish; not a learned-propagator baseline. |
| **TITO / KC-Maps / Timewarp / MDGen** | generative alternatives | Cite/discuss; not required for the main comparison. |

The paper must not claim SOTA over TrajCast until the 2.5 ns published baseline is run under the
same protocol. The 100 ps PASS is necessary but not sufficient.

### 5.2 Paper experiments

**Water (main):**

1. Deterministic GLR 2.5 ns, two starts — **done** (partition/RDF/VDOS strong; D 25–26% low;
   KE variance suppressed).
2. Stochastic GLR 2.5 ns, two starts — **done** (KE variance correct; GK/MSD inconsistent).
3. Self-consistent GLE readout — **proposed** (offline fit → 100 ps → 2.5 ns).
4. Published TrajCast 2.5 ns — **missing**.
5. Native TrajCast converged — **missing/incomplete**.
6. Seeds/error bars: deterministic 2 starts; stochastic 2; add 100 ps seeds for the GLE.
7. Ablations: rescale off, FD noise off, AR order p1/p5/p10/p20, blend 0.1/1.0, mode basis,
   slow mode, GLE kernel — mostly done; GLE kernel new.

**Quartz (scope / second system):**

1. R3-Opt 30 fs 15 ns — **done** (T/VDOS/O–O strong; Si–O 0.180; NVE drift +1.30/+0.69 eV;
   element KE imbalance).
2. 5 fs phonon-band stochastic rescale 100 ps, two seeds — **done** (T 299.8/299.9; VDOS
   0.950/0.955; RDF 0.028/0.016/0.033).
3. 5 fs 2.5 ns — **missing**; required if the paper claims long-time quartz, not just 100 ps.
4. Full phonon memory / per-mode / species controls — **done, negative**; keep as ablations.
5. One-rule memory diagnostic — **missing**; see §5.3.

**Reference / protocol:**

1. Fix the 8-frame energy check at frame 28,571 before any NVE/energy claim.
2. Standardize the 50 ps vs 250 ps equilibration and initial-condition protocol; report both if
   they matter.
3. Keep the published TrajCast 100 ps PASS artifact; do not rerun with a different protocol
   without reporting.

**Model selection / training:**

1. Endpoint-only selection is a known weakness; H1 proxy was null at 20k. Report the null and the
   physical-selection rationale; do not claim H1 works.
2. E1 motion loss negative; report as an ablation/negative control.
3. M0/composed CSVR negative; report as negative controls.

### 5.3 The one-rule gap (required ICML experiment, not the single optimization)

The current method is **not one rule**:

- water: molecular modes + AR(10) memory + deterministic rescale;
- quartz: phonon bands + no memory + stochastic rescale.

The paper can either:

- **A. Scope the claim** to "a framework with physics-defined mode bases; the memory component is
  selected by the backbone ACF." This is honest but weaker; reviewers may still ask for a rule.
- **B. Implement a pre-registered memory-selection diagnostic**:
  - build the physics-defined mode basis;
  - compute the backbone mode ACF error at 100/500 fs relative to the reference;
  - enable AR(p) memory only if the error exceeds a **pre-registered reference-scatter threshold**
    (e.g., 2× the reference block-to-block scatter);
  - same blend, COM projection, variance rescale.
  - Minimal experiment: compute the diagnostic on the existing water R3-Opt 5 fs and quartz
    R3-Opt 5 fs trajectories; the rule must select memory for water and no memory for quartz
    **without per-system switches**; then run the auto-selected 100 ps arms.
  - If the threshold cannot separate them, the one-rule claim fails; scope the paper to A.

This diagnostic is cheap (existing trajectories + 100 ps validation) and is the highest-value
paper-strategy experiment. It is not the single physical optimization because it does not fix the
water variance/D trade-off.

### 5.4 Claim boundary for the ICML submission

**Can claim:**

- Pointwise one-step accuracy is neither necessary nor sufficient for long-time ensemble fidelity
  (published TrajCast paradox; 250k counterexample).
- NVE energy stability is a red herring for NVT fidelity.
- A frozen force-free backbone + mode-resolved GLR readout restores the **mean** canonical mode
  partition, RDF, and VDOS at 2.5 ns; initial-condition independent; backbone-agnostic.
- Water RDF/VDOS at 2.5 ns are better than the published TrajCast checkpoint in our analysis.
- Quartz 5 fs phonon-band stochastic rescale passes T/KE std/VDOS/RDF at 100 ps, two seeds.
- The readout is training-free for the backbone; the memory kernel and rescale are the load-bearing
  components; FD noise is not empirically necessary at λ=0.1.

**Cannot claim (without new evidence):**

- Exact diffusion recovery (D 25–26% low).
- Full canonical KE variance for the deterministic GLR (std 0.0097 vs 0.440).
- One rule across water/quartz (memory on/off differs).
- NVE energy accuracy (quartz drift +1.30/+0.69 eV/100 ps; published TrajCast less NVE-stable).
- SOTA over TrajCast until the official 2.5 ns baseline is run.
- "First" / "only" / universal backbone-agnosticism (two architectures only).
- That the raw DSRT/phase-space/minimal-joint architecture is the method; it is a plug-in.
- That signed-KE/on-policy correction is the main fix; the published control falsifies it.

### 5.5 ICML framing vs current NMI framing

The current `paper/` artifacts are NMI-aligned (`NMI_ALIGNMENT.md`, target NMI). For ICML 2027:

- Lead with the **ML method**: mode-resolved GLE readout, stationary-measure correction,
  pointwise-vs-ensemble diagnostic, backbone-agnostic readout.
- Treat the water/quartz physics as the **evaluation**, not the contribution.
- The closest ML baselines are TrajCast, FlashMD/action map, MAMD, GLE thermostats, and
  generative transition kernels; the paper must position against them.
- The one-rule/generative gap is the main reviewer risk; the self-consistent GLE and the memory
  diagnostic are the two experiments that address it.
- Do not submit the current NMI draft as-is to ICML; rewrite the claim/experiments around the
  method and the external baselines.

---

## 6. What not to do

- Do not train the raw R3-Opt / phase-space / minimal-joint again on the old endpoint loss.
- Do not propose A1/A2, on-policy signed-KE, post-hoc energy rescaling, or a renamed rescale
  variant as the main fix.
- Do not use the native TrajCast ep145 as a SOTA baseline; it is 10% trained.
- Do not claim exact diffusion, full KE variance, one rule, or NVE accuracy.
- Do not start a new architecture search; the paper's method is the readout.
- Do not run a large hyperparameter sweep before the offline kernel fit and the one-rule
  diagnostic.
- Do not submit without a frozen code/data/coefficient tag and the reference verification fix.

---

## 7. Artifacts

**Project root:** `/data2/wyd/research/foundation_mlip_adaptation`

**Key documents:**

- `RESEARCH.md`, `RESEARCH_CONTRACT.md`, `DECISION.md`, `EXPERIMENTS.md`
- `GATE0_REPORT.md`, `GATE1_NUMERICAL_REPORT.md`, `GATE2_PILOT_REPORT.md`,
  `GATE2_FAILURE_POSTMORTEM.md`
- `R0_RESPONSE_STUDY.md`
- `method_development/CURRENT_METHOD_STATUS.md`
- `method_development/LONG_TIME_TRAINING_LITERATURE.md`
- `method_development/PUBLISHED_TRAJCAST_CONTROL.md`
- `method_development/H1_SELECTION_PROXY.md`
- `method_development/E1_MOTION_LOSS_RESULT.md`
- `paper/CLAIM_BOUNDARY.md`, `paper/NOVELTY_MATRIX.md`, `paper/EVIDENCE_MAP.md`,
  `paper/PHONON_GLR_DECISION.md`, `paper/PAPER_CONFIG.md`

**Key experiment artifacts:**

- Water GLR 2.5 ns: `experiments/glr_2p5ns_20261122/analysis`,
  `experiments/glr_independent_2p5ns_20261122/analysis`
- Stochastic GLR 2.5 ns: `experiments/glr_stoch_2p5ns_20261122/analysis`,
  `experiments/glr_stoch_independent_2p5ns_20261122/analysis`
- Quartz 5 fs phonon-band: `experiments/quartz5_phonon_bands_rescale_only_100ps_20261123/analysis_5fs`,
  `experiments/quartz5_phonon_bands_rescale_seed2_100ps_20261123/analysis_5fs`
- Quartz 30 fs 15 ns: `code/h1h2/reports/stable_geometry_optimization/R3_OPT_PHYSICS_RESULTS.md`
- Phase-space v0: `experiments/phase_space_20261120/water5_nvt_replica_01/analysis`
- Minimal-joint: `experiments/gate1_minimal_joint_100ps_20261124`,
  `experiments/gate2_pilot_A1_20261125`, `experiments/gate2_pilot_A2_20261125`
- Published TrajCast: `experiments/nvt_screen_published_trajcast_20261121/analysis`
- Native TrajCast: `experiments/nvt_screen_native_trajcast_20261120/analysis`
- Reference NVT: `water/physical_reference/train_nvt_200ps`,
  `experiments/independent_nvt_500ps_20261121`

**GitHub:**

- `AdamWong03/smdp`, branch `research/h1h2-selection-20261120`, commit `4399ccd`,
  tag `research-snapshot-20261124`.
- Unsynced: Gate 2, CAM, R0, `paper/`, `experiments/`, GLR code/coefficients.

---

## 8. Execution addendum (2026-11-26)

### 8.1 Official TrajCast 2.5 ns — running

- Checkpoint: `reference/trajcast_published_water/published_checkpoint.pt` (IBM published water
  e3nn state, 2,258,856 parameters).
- Protocol: same initial state (`nvt_2p5ns/production.h5`, frame 0), CSVR 300 K / τ=50 fs,
  5 fs horizon, 50 ps thermalization + **2.5 ns production** (510,000 steps).
- Output: `experiments/published_trajcast_2p5ns_20261126/trajectory.h5`.
- PID 3840811 on GPU 3; at 2026-11-27 00:xx it was at step ~75,000/510,000, ~3.5 h elapsed,
  ETA ~20 h. The 100 ps control already PASSed; the 2.5 ns result is pending. Do not report a
  2.5 ns number before it completes. `analyze_when_done.sh` will run the analysis automatically
  and write `analysis/trajcast_2p5ns_report.json`.

### 8.2 Reference ACF / D statistical uncertainty

25 consecutive 100 ps blocks of the 2.5 ns reference NVT, translation ACF integral (0–5 ps):

| quantity | value |
|---|---:|
| mean | **14.4479 fs** |
| std (ddof=1) | **1.8636 fs** |
| SEM | 0.3727 fs |
| min | **10.9356 fs** (block 4, 400–500 ps) |
| max | **18.0034 fs** (block 2, 200–300 ps) |
| block 0 (0–100 ps) | **11.3405 fs** |
| block 5 (500–600 ps) | 13.3812 fs |
| block 1 (100–200 ps) | 14.4674 fs |

The first 100 ps is **~22% below** the 2.5 ns mean; the production GLR `glr_fit_20261122` used a
100 ps block (default `fit_glr.py --start-ps 500`) and only a 200 fs ACF window. The GLR's
**25% D deficit is therefore at least partly calibration/transport-window bias**, not proof that
the readout cannot recover D. The reference D itself has ~13% block-to-block CV; any claim of
"exact D" or "25% failure" must use the full 2.5 ns mean with error bars and a held-out reference.

### 8.3 GLE offline fit study

**Continuous 3–6 mode DHO + slow on the 25-block mean ACF:**

| modes | integral (fs) | mean integral | RMSE (mean ACF) | params | condition |
|---:|---:|---:|---:|---:|---:|
| 3 | **14.3378** | 14.4479 | **0.01283** | 10 | 2.39e11 |
| 4 | 14.6418 | 14.4479 | 0.04097 | 13 | 7.74e8 |
| 5 | 14.4216 | 14.4479 | 0.01018 | 16 | 7.03e11 |
| 6 | 14.4432 | 14.4479 | 0.00708 | 19 | 2.01e10 |

**The continuous formula is misleading at a 5 fs rollout cadence.** The 3-mode continuous ACF
contains a mode with ω≈1.2475 rad/fs (period 5.04 fs), which is aliased by the 5 fs sampling: its
**discrete 0–5 ps integral is 27.4596 fs (1.90× the 14.4479 fs reference)**, even though the
continuous Green–Kubo formula gives 14.3378 fs. The 5/6-mode fits also contain ω≈1.27 modes and
have discrete integrals 25.31/21.93 fs. The 4-mode fit has no near-Nyquist mode and a correct
discrete integral (14.642 fs), but its 5/25 fs ACF is 0.909/0.074 vs the reference 0.982/0.659.
Thus the earlier "3-mode is sufficient" conclusion was a **continuous-formula/aliasing artifact**,
not a usable 5 fs kernel. Parameter identifiability is poor (condition ~1e8–1e11); more modes do
not fix the 5 fs shape.

**Direct discrete-time rank-1 AR(2)+slow fit:**

| modes | integral (fs) | ratio | RMSE mean ACF | min eigenvalue gap | min innovation variance |
|---:|---:|---:|---:|---:|---:|
| 3 | 14.591 | 1.0099 | **0.312** | 0.035 | 3.9e-9 |
| 4 | 14.586 | 1.0096 | 0.303 | 0.034 | 8.6e-8 |
| 5 | 14.579 | 1.0091 | 0.262 | 0.010 | 7.9e-7 |

The rank-1 AR(2) state-space is **not** the exact complex-OU realization; it cannot reproduce the
DHO ACF (RMSE 0.26–0.31). The exact 2×2 realization (`A_d=exp(AΔt)`, `Q_d=Σ−A_dΣA_dᵀ`) reproduces
the continuous ACF to machine precision; Monte Carlo sampling of the 5-mode state-space matches
the analytic ACF at 5–5000 fs with **RMSE 0.0025**. This is an implementation check, not a pass:
the kernel's 5 fs ACF itself is wrong.

**Long-window AR(p) Yule–Walker:** p10/p20/p40/p80 integrals 292.7/339.1/293.9/244.6 fs (20×
too large); the long-lag AR recursion is unstable. Do not use high-order AR fitted directly to
5 ps.

**Conclusion (superseded 2026-11-27):** the exact 2×2 discrete implementation is correct, but no
3–6 mode DHO+slow kernel fits both the 5/25 fs ACF and the discrete D. The unconstrained 3-mode
matches the ACF but has 1.90× D; the frequency-bounded 4/5/6-mode fits restore D but miss 5 fs by
0.25–0.30. The failure is **kernel identification at the 5 fs cadence**, not discrete
implementation. No 100 ps GLE rollout was started. See §9 for the final gate result.

### 8.4 GLR calibration provenance audit

- `experiments/glr_fit_20261122/glr_coefficients.json` contains **no reference path, start time,
  or fit-window metadata**. `fit_glr.py` defaults: `--start-ps 500`, `--duration-ps 100`,
  `--stride-fs 5`, `--maxlag 40` (200 fs), order 10. `glr_fit_heldout0_20261122` is the block-0
  fit; the production fit is a later block (likely 500–600 ps) but this is not recorded.
- The AR(10) fit uses only a **200 fs ACF window** and has **no D/integral constraint**; the
  long-lag translation tail (500 fs–5 ps), which controls D, is unconstrained. This is a
  calibration/fit-window gap, not an AR-order issue.
- The GLR readout needs a **reference NVT trajectory** for the mode ACF (at least ~100 ps, better
  2.5 ns), the physics-defined mode basis (molecular topology for water; phonon Hessian for
  quartz), and the variance target \(k_BT/2\). It needs **no reference forces at inference**.
  It is "training-free for the backbone" but **not data-free**.
- Data roles are currently blurred: calibration = a 100 ps reference NVT block; validation =
  independent NVT 500 ps / disjoint blocks; **test = none**. The evaluation reference is the same
  2.5 ns trajectory used for calibration. Before a paper claim, use a held-out reference block or
  the independent NVT trajectory for validation, and report the GLR coefficients' provenance.

### 8.5 8-frame reference verification fix

- Root cause: `verify_reference` reconstructed images from **float32 unwrapped coordinates**;
  frame 28,571 lost precision and gave −1.17e-3 eV, just over the 1e-3 tolerance.
- Fix: use `positions_wrapped_A` + stored `image_flags` in `WaterReference.evaluate` /
  `verify_reference`. The tolerance remains **1e-3 eV / 0.02 eV/Å**.
- Result for `e1_baseline` and `e1_motion`: **accepted=True**, max energy error **0.913e-3 eV**,
  max force error **0.00445 eV/Å**. Old `reference_verification.json` files are preserved; new
  `reference_verification_fixed.json` files record the fix. The E1 negative result is unchanged.

### 8.6 Revised single priority optimization

**Full-reference calibration + exact discrete complex-OU implementation of the 3-mode DHO+slow
readout**, then a two-start 100 ps water test.

- Calibrate the mode ACF on the **2.5 ns mean** (or ≥500 ps multi-block mean), not a single 100 ps
  block; report the block std/SEM.
- Implement the 3-mode DHO+slow readout with an exact complex-OU/2×2-noise state-space (not a
  rank-1 AR(2)); verify the sampled ACF at 5/25/100/200/500/1000/2000/5000 fs and the D integral
  on the mean and a held-out independent block.
- Then run 100 ps NVT, two independent starts, against deterministic GLR, stochastic GLR, and
  official TrajCast. If KE mean/std, ACF, GK/MSD, RDF, and VDOS improve without structure
  degradation, run 2.5 ns. If not, classify kernel identification vs discrete implementation vs
  neural-propagator coupling; do not automatically add a new kernel/rescale.

**Executed 2026-11-27: the offline gate failed. The 100 ps test was not run.** See §9.

### 8.7 What is ready for the ICML main text

**Ready:**

- Pointwise one-step accuracy is neither necessary nor sufficient for long-time ensemble fidelity
  (published TrajCast paradox; 250k counterexample).
- NVE energy stability is a red herring for NVT fidelity.
- GLR mean partition/RDF/VDOS at 2.5 ns, two starts; backbone-agnostic v0+GLR 100 ps; quartz 5 fs
  phonon-band stochastic rescale, 100 ps two seeds.
- Component ablations: rescale necessary, FD noise not necessary at λ=0.1, memory needed for water
  but not quartz 5 fs.
- **New:** reference ACF/D has large 100 ps block uncertainty; short calibration biases D; this is
  a methodological diagnostic, not a method failure.

**Not ready:**

- Exact D recovery / a "25% low" claim without full-reference uncertainty and a recalibrated
  readout.
- Full KE variance for deterministic GLR (mean-only); stochastic GLR GK/MSD inconsistency.
- One-rule cross-material claim (water memory vs quartz no memory).
- Official TrajCast 2.5 ns (running) and converged native TrajCast (ep145/1500 is not fair).
- 2.5 ns quartz; external action-map/FlashMD comparisons; a clean held-out reference test.
- The new GLE readout: the offline gate failed at the 5 fs cadence; no 100 ps or 2.5 ns GLE rollout
  was run.

### 8.8 Artifacts from this round

- `experiments/published_trajcast_2p5ns_20261126/` (running; `run.log`, `trajectory.h5`)
- `experiments/reference_acf_blocks_20261126.json`, `...npz`, `...log`
- `experiments/gle_kernel_study_20261126.json`, `...log`
- `experiments/gle_kernel_study_mean_20261126.json`, `...log`
- `experiments/fit_glr_long_20261126.json`, `...log`
- `experiments/fit_gle_discrete_20261126.json`, `...log`
- `experiments/e1_baseline_20k_20261120/physics/nve_val_20ps/analysis/reference_verification_fixed.json`
- `experiments/e1_motion_20k_20261120/physics/nve_val_20ps/analysis/reference_verification_fixed.json`
- Code: `code/h1h2/physics/spcfw.py`, `code/h1h2/scripts/native_nvt_screen.py`,
  `code/h1h2/gle_kernel_study.py`, `code/h1h2/gle_kernel_study_mean.py`,
  `code/h1h2/fit_glr_long.py`, `code/h1h2/fit_gle_discrete.py`,
  `code/h1h2/reference_acf_blocks.py`.

---

## 9. Final GLE gate and execution update (2026-11-27)

### 9.1 Exact discrete implementation: PASS

The exact 2×2 complex-OU realization is:

- per DHO mode: `A = [[-γ/2, -ω_d], [ω_d, -γ/2]]`, `A_d = exp(A·5 fs)`;
- stationary covariance `Σ = [[w, -w(γ/2)/ω_d], [-w(γ/2)/ω_d, w(1 + 2((γ/2)/ω_d)²)]]`;
- innovation covariance `Q_d = Σ - A_d Σ A_dᵀ` (full 2×2, not rank-1);
- observation `x = e1ᵀ state`; slow mode is a 1D exact OU.

The sampled first component is exactly the DHO velocity ACF at integer lags. Monte Carlo of the
5-mode generalized fit (10⁶ steps, 5 fs) matches the analytic ACF at 5–5000 fs with **RMSE
0.00248** and max ACF error **0.0067**. The analytic discrete integral is **14.5914 fs**
(reference 14.4479, +1.0%). The discrete implementation is not the failure.

### 9.2 Kernel family: FAIL at 5 fs

Generalized DHO+slow fits allow underdamped and overdamped modes, use the sampled 5 fs ACF, and
constrain the discrete Green–Kubo integral. Best candidates:

| fit | modes | params | discrete integral (fs) | ratio | 5 fs | 25 fs | key RMSE | max key error | long-lag RMSE | condition |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unconstrained 3-mode | 3 | 10 | **27.4596** | **1.90×** | 0.968 | 0.572 | 0.038 | 0.088 | 0.041 | — |
| unconstrained 6-mode | 6 | 19 | 21.9303 | 1.52× | 0.972 | 0.610 | 0.019 | 0.049 | 0.020 | — |
| generalized 3-mode | 3 | 10 | 14.547 | 1.007 | 0.606 | 0.528 | 0.141 | 0.375 | 0.050 | 1.8e10 |
| generalized 4-mode | 4 | 13 | 14.566 | 1.008 | 0.662 | 0.510 | 0.115 | 0.296 | 0.057 | 2.2e9 |
| generalized 5-mode | 5 | 16 | **14.591** | **1.010** | 0.726 | 0.536 | **0.100** | **0.255** | **0.047** | 1.4e9 |
| generalized 6-mode | 6 | 19 | 14.592 | 1.010 | 0.734 | 0.539 | 0.097 | 0.247 | 0.046 | 5.5e8 |
| reference (25-block mean) | — | — | 14.4479 | 1 | **0.982** | **0.659** | — | — | — | — |

- The unconstrained 3/5/6-mode fits fit the 5/25 fs ACF by putting weight on a mode with
  ω≈1.25–1.27 rad/fs (period ≈5 fs). At the 5 fs rollout cadence that mode is aliased; the
  discrete integral is 1.5–1.9× the reference even though the continuous formula is ≈14.3–14.4 fs.
- The frequency-bounded generalized fits restore the discrete D but cannot reproduce the 5/25 fs
  shoulder. The 5-mode 5 fs ACF is 0.726 vs 0.982 (error 0.255); the 25 fs is 0.536 vs 0.659
  (error 0.123). The 6-mode adds 3 parameters and improves the key RMSE by only 0.003.
- **Mode count is not the bottleneck; the 3–6 mode DHO+slow family is not identifiable at 5 fs.**

### 9.3 Independent-reference confirmation

`train_nvt_200ps/production.h5` (independent NVT trajectory), two 100 ps blocks:

| lag | block 0 | block 100–200 ps | 2.5 ns 25-block mean |
|---:|---:|---:|---:|
| 5 fs | 0.9816 | 0.9818 | 0.9817 |
| 25 fs | 0.6573 | 0.6606 | 0.6594 |
| 100 fs | 0.0314 | 0.0408 | 0.0393 |
| 200 fs | −0.0822 | −0.0728 | −0.0752 |
| 500 fs | −0.0286 | −0.0254 | −0.0229 |
| integral | 12.9489 fs | 14.6705 fs | 14.4479 fs |

The ACF shape and the 5 fs failure are systematic, not a calibration-block artifact.

### 9.4 GLR provenance and AR audit

- `experiments/glr_fit_20261122/glr_coefficients.json` still has **no reference path, start,
  duration, or ACF-window metadata**. Defaults are start 500 ps, duration 100 ps, stride 5 fs,
  maxlag 40 frames (200 fs), order 10. The production fit is inferred to be block 5 (500–600 ps),
  not recorded. `glr_fit_heldout0_20261122` is block 0.
- The translation AR(10) is stable (spectral radius **0.855**) and matches 5/25 fs, but misses the
  100–500 fs negative dip: AR 100 fs −0.0316, 200 fs +0.0016; reference +0.0391, −0.0781.
- Extrapolating the AR recursion to 5 ps gives a translation integral **33.24 fs**, **2.30×** the
  25-block mean. The 200 fs fit window cannot constrain D. This is a fit-window/provenance gap,
  not proof that the readout cannot recover D.
- The 25-block ACF integral has mean 14.4479, std 1.8636, SEM 0.3727, min 10.9356, max 18.0034.
  The deterministic GLR D is 1.406 ± 0.155 ×10⁻⁹ m²/s vs reference 1.883 ± 0.306; the 25% deficit
  is at least partly calibration-window bias. A ≥500 ps multi-block mean (preferably the full
  2.5 ns) and a D-constrained fit are needed.

### 9.5 Official TrajCast control and 2.5 ns status

**100 ps control** (official IBM published checkpoint, same initial state / CSVR / 5 fs):

| metric | TrajCast | reference |
|---|---:|---:|
| temperature (K) | 300.19 ± 17.68 | 300.19 ± 17.89 |
| total KE (eV) | 7.4113 ± 0.4366 | 7.4114 ± 0.4417 |
| translation KE (eV) | 2.5050 | 2.3725 |
| rotation KE (eV) | 2.4626 | 2.4302 |
| internal KE (eV) | 2.4437 | 2.6087 |
| RDF RMSE O–O / O–H / H–H | 0.0431 / 0.1180 / 0.0957 | — |
| VDOS overlap O / H | 0.9370 / 0.6241 | — |
| D | not reported (100 ps < frozen fit limit) | — |

The temperature, total KE, and fluctuations are excellent. The mode partition differences are
~0.5–0.7 block σ at 100 ps; the H VDOS is the weakest metric. These are 100 ps context numbers,
not a 2.5 ns result.

**2.5 ns:** `experiments/published_trajcast_2p5ns_20261126/`, PID 3840811 on GPU 3, 50 ps
thermalization + 500,000 production steps. At 2026-11-27 00:xx it was at step ~75,000/510,000,
ETA ~20 h. `analyze_when_done.sh` waits for `trajectory.summary.json`, runs `analyze_water.py`
with the same protocol, and writes `analysis/trajcast_2p5ns_report.json`. Do not report a 2.5 ns
number before that file exists.

### 9.6 GLR vs stochastic GLR 2.5 ns

| run | T (K) | total KE (eV) | translation KE std (eV) | D (10⁻⁹ m²/s) | RDF RMSE O–O | VDOS O/H |
|---|---:|---:|---:|---:|---:|---:|
| reference | 300.17 | 7.4108 | 0.2534 | 1.883 ± 0.306 | — | — |
| deterministic GLR | 299.99 | 7.4064 | **0.0017** | **1.406 ± 0.155** | 0.0307 | 0.953/0.847 |
| stochastic GLR | 299.97 | 7.4058 | **0.2520** | **1.435 ± 0.130** | 0.0276 | 0.949/0.852 |

Deterministic GLR recovers the mean partition but suppresses the KE variance. Stochastic GLR
restores the variance but does **not** fix D (still ~24% low) and has a known GK/MSD consistency
problem. The GLE was intended to fix both; its offline gate failed.

### 9.7 Final decision

- **The GLE offline gate failed.** No 100 ps GLE rollout was started; no 2.5 ns GLE run.
- **Classification:** kernel identification at the 5 fs cadence (plus 5 fs aliasing of the
  near-Nyquist DHO modes), **not** discrete implementation (exact embedding verified by Monte
  Carlo) and **not** neural-propagator coupling (the coupled process was not exercised).
- No new rescale, kernel family, mode count, or backbone variant was started after the failure.
- A different kernel family (e.g. Gaussian/continuous-spectrum memory) or a 1 fs rollout cadence
  would be required to revisit the question. Both are outside the authorized scope.

### 9.8 Reproducible snapshot

`reproducibility/snapshot_20261127/` contains selected code, configs, GLR/GLE parameters,
reference ACF blocks, fit/audit results, logs, `MANIFEST.json`, `CHECKSUMS.sha256`, and a local
git branch `research/glr-gle-20261127` / tag `research-snapshot-20261127`. Large trajectories are
referenced, not copied.

### 9.9 Updated readiness

**Ready:** the same list as §8.7, plus the new negative/methodological results: reference ACF/D
has large 100 ps block uncertainty; short calibration biases D; the 3-mode continuous fit's
apparent success is a 5 fs aliasing artifact; the exact 2×2 discrete implementation is verified.

**Not ready:** exact D recovery; full KE variance for deterministic GLR; stochastic GLR GK/MSD
consistency; one-rule cross-material claim; official TrajCast 2.5 ns (running); converged native
TrajCast; 2.5 ns quartz; external action-map/FlashMD comparisons; clean held-out test; the new
GLE readout (offline gate failed).
