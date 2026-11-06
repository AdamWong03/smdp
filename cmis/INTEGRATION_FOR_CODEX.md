# CMIS 正式集成与实验执行说明（Codex 可直接使用）

目标：将独立研究原型 `cmis_model.py`、`vesin_adapter.py`、`train_cmis.py`、`rollout_nve.py` 集成进 `AdamWong03/smdp`，作为新方法 **Coupled Modal Impulse Segments（CMIS）**。研究问题是：**学习耦合的有限时间动量交换模式，能否通过相空间一致的轨迹段预测，同时改善 Quartz 与 Water 的长期物理保真？**

## 严格边界

- 在 `experiment/cmis-coupled-impulse-segments` 新分支工作。保留所有原有 Full SMDP、DSRT R3-Opt、TrajCast 基线配置与结果，不能覆盖 checkpoint、参考 MD 和历史物理评估。
- **不要**把旧 Full SMDP 的 Softmax Motion Allocation、Recurrent/Direct 双专家路由或单独 q/v 解码移植进 CMIS。也不要偷偷加入 PET、额外势函数或隐式积分器。算法数学定义与 `README.md` 一致。
- 本目录的 toy/单元结果**绝不是 MD 物理证据**。没有运行过正式 Quartz/Water 训练，不得宣称任何精度提升。
- `smdp_reference_md_v1` 的参考数据必须原样保存；禁止复制整个约 212GB 数据集。先读 metadata、HDF5 schema、轨迹 ID 与单位，再开发 lazy HDF5 index/reader。
- 不使用 test 数据选 checkpoint，不使用 BKS/SPC-Fw forces 作为模型输入、teacher 或动力学传播。势能只用于离线评估或预先声明的 validation 模型选择。

## 1. 审查与接入

1. 从本目录代码复制到新分支，先运行 `pytest -q test_cmis.py`。预期 13 项软件测试通过，包括 time-zero、平移/旋转/PBC/Galilean、COM、qdot=v、独立 batch、mode intervention 和梯度。
2. 对比项目现有 `periodic.py`，检查 Vesin 的 `ijS` 定义：`edge_index=(sender=j, receiver=i)`，`shift=(image_shift - image[j]+image[i]) @ cell`，以及 `dr=q[sender]+shift-q[receiver]`；在随机 triclinic 盒、原子镜像重置、同种和异种原子上逐项测试。生产必须是高效邻居列表，不能使用小盒 O(N²) 辅助函数。
3. 对同一个真实初始状态，检查单独 endpoint 查询与 dense query 的终点完全一致；检查 `qdot=v` 数值导数、不同 frame 的 mass-weighted COM、近 cutoff 平滑性、有限梯度。
4. 复用现有参考 NVE 轨迹索引和独立 train/val/test trajectory split。确认原始位置 unwrapped Å、速度 Å/fs、mass amu、cell row lattice vectors；真实内部时刻标签不能由 endpoints 人工插值。

## 2. 正式网络定义（不要另行临时发明版本）

- `CMISConfig(width=96,modes=4,encoder_layers=3,cutoff=4.5,rbf_dim=20,horizon_fs=5或30)`。
- 初始 invariant scalar encoder 加可训练 mode embeddings，**不使用跨 mode softmax allocation**。
- 两次 `ModeCoupling` 对 K 个 mode 进行注意力交互，第一次在初始状态，第二次在模型预测的候选中点。
- 两次 `PairModeStep` 对周期性无向成对边输出反对称 vector impulse：第 1 次是主冲量，第 2 次是小初始化的未来几何修正。
- 每个 mode、每条边学习 `1, cos(ω_e,k τ), sin(ω_e,k τ)` 三个局部时间基的幅值与正频率。使用 **精确解析 P/Q 积分** 从同一冲量场同时得到 q(τ) 与 v(τ)，保证 `dq/dτ=v`。K=4 mode 的频率可学习且依赖局部状态。
- 不声称能量、角动量、时间可逆性、辛性或半群组合自动成立；在物理评估中单独报告。
- 在大体系中首先 profile 时间查询和 E×K×width 的临时显存，再考虑保持数学定义不变的 kernel 融合/分块优化。不能因为速度不够便临时换架构。

## 3. 训练数据与训练目标

- **Quartz** `h=30fs`，原子和质量严格与参考轨迹一致。实际 `times_fs` 优先 [3,6,...,30]fs，要求每个采样点都有原始参考帧。
- **Water** `h=5fs`，优先 [1,2,3,4,5]fs；若真实保存有 0.5fs 全帧，可选 [0.5,1,...,5]fs。Water 与 Quartz **分别从随机初始化训练同一架构**，不能假定两者共享权重。
- 只用独立训练 NVE 轨迹计算 scale_q、scale_v。Loss 是所有真实参考时间点上质量加权的 normalized q/v MSE，终点权重 2，其余 1；NVE 轨迹用随机时间反演增强。不要将温度均分写成固定 target loss。
- `train_cmis.py` 已提供 batch disjoint-graph 训练入口、train-only normalization、时反演与 best val checkpoint；真正运行前将 NPZ reader 换成 server 现有 HDF5/lazy reader，保持同样张量契约，不重新下载或复制 reference MD。
- 第一次正式训练参数：batch 8，AdamW 2e-4，width96，mode4，最大 350 epochs，使用固定种子 20261106 以及 validation checkpoint 选择。按现有硬件调整 batch/梯度累积只能改变工程预算，不得改变模型定义与数据协议。
- 如在 validation 单步误差接近但物理明显失真，先检查频率范围、train/test state coverage 和 mode 能量/频谱；不要立刻叠加 pushforward、额外专家、能量投影等模块。

## 4. 两级 validation 与正式 test

- **第一次筛选**：所有 val 轨迹 1/2/4/8-step q/v error，单步按 H/O 或 Si/O 分解，mode 有效秩、每模态冲量能量、频率分布、冻结 mode 删除干预、时间导数一致性。
- **第二次筛选**：在严格独立的 val NVE 状态上做 20–50ps 早期物理测试，并与旧 DSRT R3-Opt、Full SMDP 共享同一评估器；输出真实 Kinetic/PE/Total energy 与温度曲线，注意 reference potential 仅为 *offline* evaluation。
- checkpoint 在 val 上锁定后才跑正式 test。Water 至少 3 条独立 100ps NVE、3 条 2.5ns NVT；Quartz 至少 3 条 100ps NVE、3 条 15ns NVT，视实际计算预算可先跑一个并如实标记未完成。计入 graph rebuild、thermostat、I/O 的真实 ns/day，报告数据和硬件条件。
- Water 需评估 H/O VDOS、O–O/O–H/H–H RDF、molecular COM MSD、扩散系数（同时报告统一的 1–10ps 与 10–100ps 拟合窗口）、平动/转动/内部能量分配、thermostat 每步能量收支。
- Quartz 需评估 Si/O VDOS、Si–O/O–O RDF、参考 BKS 总能量漂移和温度；对高频 VDOS 区分 h-步采样频带与 segment dense query 频带。
- 第三体系优先使用有完整参考轨迹的气相 paracetamol（与 TrajCast 可对照）或其他能提供新的运动机制且可比的体系；在前两者有效后扩展，不随意增加一个仅报告 MD17 MSE 的数据集冒充物理泛化。

## 5. 必须提供的外部基线与内部对照

- 外部强基线：**官方 TrajCast**（同 Water/Quartz 物理条件、统一 evaluator、相同时长）；FlashMD/MDtrajNet 仅在不同任务、势面、体系可以对应时加入，不做不公平的生硬数值比较。
- 内部对照：冻结 DSRT R3-Opt、Full SMDP R3-B。
- 关键机制验证：同目标预算下的单冲量场（K=1）、不进行中点几何修正、禁用跨 mode 耦合、禁用自适应频率改为固定公共频率，以及无中间时间监督。它们是 **ablations，不是 external baselines**。无需在首轮正式物理数据跑完前把所有消融训练完。
- 如果最终不同 mode 仍然高度相关或 mask 一个 mode 不影响输出，就拒绝“成功学习到互补运动模式”的主张，不使用负载均衡正则来制造表面差异。

## 6. 交付与报告

- 提交新分支代码、全部 unit test 结果、真实 HDF5 adapter 及单位验证、同配置训练脚本、超参、训练曲线、验证和物理 summary JSON、mode 分析工具。
- 明确 `commit SHA`, model parameter count, GPU dtype/peak VRAM and full-production end-to-end ns/day。
- 不要自行因为结果“不好看”修改模型和替换 checkpoint；先把真实报告回传，讨论后再决定下一轮。这轮冻结网络、训练基础和评价协议，不冻结无关 bug fixes。