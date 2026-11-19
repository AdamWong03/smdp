# 神经状态传播器：实现与训练流程审查

开始：2026-11-19；更新：2026-11-20。自身代码基点：`ce5c6ee`。本文依据固定版本源码、实际配置与计算路径，而不是历史实验排名来决定设计。

**本次做的是算子审查，不是模型性能比较。** 原模型权重仅用于确认张量形状、数值尺度与梯度路径；没有参数更新，没有重新生成 MD，没有重新启动参考模型训练。CPU 数值核对使用 16 个固定 TRAIN 样本及其中 8 个组成的一次反向传播。它不能代替全数据正确性验证，也不能证明某个算子导致长期失真。

源码版本、具体文件与审查范围见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。E2Former-V2 与 EquiformerV3 的逐算子审查见 [FRONTIER_OPERATORS.md](FRONTIER_OPERATORS.md)。复现见 [REPRODUCE.md](REPRODUCE.md)，数值输出见 [representation_contracts.json](representation_contracts.json)。外部文件保留在服务器，本文不把未经运行的参考实现写成已复现结果。

## 1. 先纠正方法名称掩盖的实现差异

| 实现 | 实际预测对象和计算路径 | 不应混淆的事情 |
|---|---|---|
| TrajCast，固定官方代码 | 当前状态编码；边方向张量积；邻域聚合；接收原子速度张量积；直接输出位移与绝对终点速度 | 不是只使用静态几何的普通 GNN；也不是对速度简单拼接一次 |
| MDtrajNet，MLatom 当前公开实现 | 时间条件化等变注意力反复更新几何/运动表示，核心网络返回位置；公开预测接口通过时间差分或时间导数获得速度 | 内部名为 `veloc` 的表示不是直接返回的物理速度头；查询 q/v 的成本不能按一次位置 forward 计算 |
| PaiNN，SchNetPack 实现 | 径向滤波的标量—向量消息，加节点内部的向量投影、模长、内积及双向混合 | 不含速度输入；性质/能量/力任务上的算子不能直接当作有限时间传播器 |
| GotenNet，官方 QM9 配置及实现 | 空间滤波消息与注意力消息相加；多阶方向特征传播；节点更新后更新几何边状态；EQFF 混合 | 不是只有 softmax 注意力；类中出现 LayerNorm 不代表实际配置启用了它 |
| GNS，官方实现 | 多帧位置构造速度历史；预测归一化离散加速度；固定离散更新；噪声输入对应专门调整后的加速度标签 | 不等于当前状态输入的 q/v 双终点回归；其噪声训练不能只搬“加噪声”三字 |
| FlashMD | 论文复现指定的 v0.1.2 推演器与当前 metatrain 架构分别核查 | 当前代码不自动等于论文版本；旧版默认能量重标定需要势能计算，不能归入无外部能量计算的运行设置 |
| E2Former-V2 | 节点 Q/K/V 投影、SO(2) 分解、Triton 稀疏读取和聚合 | 减少高维边中间量，不等于所有计算都变成 O(N)；有效配置与备用编码器必须分开 |
| EquiformerV3 | 跨阶不变量归一化、S²乘性非线性、带平滑截断的图注意力 | 不等于普通 LayerNorm + SwiGLU；势能/力模型不是直接状态传播器 |

### MDtrajNet 的具体纠正

`EEEDDDD.py:73–100` 的 attention block 使用相对位置、相对速度、元素、时间和边球谐函数。`160–193` 在最初的邻居索引上，用各层更新后的坐标和速度表示重新计算几何；每层输出加回原始输入，不是简单沿上一层状态做恒等残差。网络最终只返回坐标。

`625–637` 的预测接口用两个时间点的位置差分，或对时间求 Jacobian，计算速度。训练中的速度监督也沿该路径计算；`velocity_weight` 是显式配置项，库默认值不是每个论文实验的实际值。`propagate()` 还提供默认关闭的势能辅助速度重标定；模型集合分支也有额外逻辑。因此不能把“八层等变 Transformer、独立 q/v 头、一次 forward、完全相同的 MD 推演”作为已核实描述。

来源：[网络与预测接口](https://github.com/dralgroup/mlatom/blob/c34839eb354c7a8fb264b34b82c6b90043f6572e/mlatom/EEEDDDD.py#L73)。本次审查的是该固定代码版本，未取得与论文每个实验一一对应的训练配置，不填补这些缺项。

## 2. 自身网络：从物理输入走到输出

以下对应 [relational_transition.py](../../relational_transition.py)。这里的“不同”是计算差异，不是错误认定。

### 2.1 输入与边描述

物理位置单位 Å，速度 Å/fs，质量 amu。`450–476` 按图减去输入速度的质量加权 COM，再除以固定 `velocity_scale=0.01`。节点标量是元素 embedding 加 `MLP(log mass, |v_relative / 0.01|²)`；48 个初始向量通道都是该原子速度的元素条件化倍数，所以初始化时同一原子的向量通道共线。邻居传播随后可以打破这种共线性，不能据此说整个网络只有一个方向。

边方向统一为 sender 减 receiver 加周期 shift。`RelationEncoding:88–104` 给出当前/候选距离的两组 RBF、5 个不变量和 4 组向量基。第一阶段两份状态相同，因此部分输入重复；第二阶段用候选终点更新边几何，保持当前边的周期镜像对应。单次 forward 不建立新未来拓扑。

这是对相空间信息的一个明确编码选择。它并非没有元素、速度或角度信息；但与“每层在聚合后显式计算环境表示 × 接收原子运动表示”不是同一种运算。

### 2.2 边消息：先压缩成不变量，再生成向量系数

`GatedGeometryRelationalBlock:247–287` 的实际顺序为：

1. 对节点向量做两组可学习通道投影；形成当前/未来方向投影、横向相关及两端内积，共 `7 × 48 = 336` 个不变量。
2. 结合两端标量、旧边状态和 37 维关系输入，执行 `853 → 192 → 96` 的边 MLP，残差相加后做边 LayerNorm。
3. 从 96 维边状态生成 192 维标量消息，以及 `5 × 48` 个向量系数。
4. 向量消息为当前/候选方向、当前/候选相对速度的线性组合，加受系数控制的 sender 隐藏向量。
5. 聚合后更新标量与向量，再做一次节点内部向量—标量混合。

这条路径包含非线性、方向关联和真正的向量—标量反馈；“只剩标量 MLP，完全没有几何耦合”不准确。需要研究的是：**哪些几何交互在压缩前完成、哪些在聚合后完成、哪些表示可以跨层传递**。把宽度改大不会自动改变这些路径。

当前 `pair_mlp:62–77` 已把第一层仿射映射的两个节点投影搬到节点上计算。这是代数等价分解，不是低秩近似。不能把这一项重新包装成尚未实现的加速措施。

### 2.3 聚合与隐藏状态尺度

自身块使用

\[
\bar m_i=\frac{\sum_j c(r_{ij})m_{ij}}{\max(1,\sum_j c(r_{ij}))}.
\]

这是一种局部加权平均。分母参与每个原子的实际计算，邻域密度变化会同时改变分子和分母。重复完全相同的消息不会像求和那样扩大聚合幅度，但这不等于网络完全不能识别配位数：边状态、方向分布、多层路径仍可编码其他差异。

每块一次边 LayerNorm、两次标量 LayerNorm；向量聚合由 sigmoid 门控，节点向量混合受 `0.1 * tanh(gate)` 调节。这里的 0.1 只限制该分支的系数，不限制可学习向量投影矩阵的范数，不能把它解释为整个向量更新严格小于 10%。

这些归一化与门控必须作为一套完整算子分析，不能分别因为“残差”“归一化”“有界门控”通常有用，就断言组合必然合理。

### 2.4 终点参数化和目标

`378–387, 479–508`：两组读出均由标量生成系数，对隐藏向量线性组合。Proposal 为

\[
\Delta q^p=\Delta t\,v+\sigma_q C_m(a),\qquad
v^p=v+\sigma_v C_m(b),
\]

Final 再加两组中心化修正。q/v 共享表示，但物理终点可以分别变化；没有解析的时间导数关系。这里的尺度来自完整目标位移与绝对终点速度 RMS，并非 ballistic 残差和速度增量的 RMS。目标为

\[
L_{final}=\tfrac12\left[\operatorname{MSE}(\Delta q/\sigma_q)+
\operatorname{MSE}(v^+/\sigma_v)\right],\qquad
L=L_{final}+0.02L_{proposal},
\]

式中的 MSE 均指预测与对应真实目标之间的差，不是对预测本身取平方。

残差读出不降低函数的理论表达上限，但会改变网络实际需要拟合的量、梯度条件及读出尺度。应检查残差是否真的是小量，不能从“5 fs 很短”直接推断。

## 3. TrajCast：关键是完整的两次耦合，不只是 l=2

实际 Water 配置与 CPU 张量追踪一致：初始节点表示为 `64x0e + 64x1o + 64x2e`，576 个分量；第一层后为 l=0/1/2 的两种宇称各 64 通道，1152 个分量；最后压缩到 `16x1o`，读出 `2x1o`。这些是表示分量，不是参数数量。

`ResidualConditionedMessagePassingLayer.forward:392–444`：

```text
sender irreps × edge spherical harmonics  [distance-conditioned weights]
    → sum over neighbors / fixed training-average neighbor count
    → node-level linear contraction
    → contracted environment × receiver velocity spherical harmonics
       [species-conditioned speed-basis weights]
    → equivariant linear mixing + gated nonlinearity
    → species-conditioned residual; divide by sqrt(2)
```

第二次速度张量积在节点聚合后执行，成本随 N 而非 E 计。速度幅值也不是一个无条件的平方标量：`ElementBasedNormEncoding` 先计算速度模长基函数，再与元素编码耦合。它不等于手工按元素温度缩放速度。

因此，把 DSRT 增加一个 l=2 缓冲张量、速度 RBF 或 attention，就称为“已经补齐 TrajCast 表示”是不成立的。需要一起定义张量乘积路径、聚合尺度、节点混合、门控、残差和读出。相反，这段源码也不能证明 l=2 是好动力学的必要条件。

[消息层](https://github.com/IBM/trajcast/blob/c52f107a02176671b21a420b2d4c67e0327b04f7/trajcast/nn/_message_passing.py#L392) · [元素条件化速度编码](https://github.com/IBM/trajcast/blob/c52f107a02176671b21a420b2d4c67e0327b04f7/trajcast/nn/_encoding.py#L110) · [模型装配](https://github.com/IBM/trajcast/blob/c52f107a02176671b21a420b2d4c67e0327b04f7/trajcast/model/models.py#L109)。

## 4. PaiNN 与 GotenNet：借鉴应落实到完整运算

### PaiNN

Interaction 先对节点标量做 MLP，再将 sender 结果与径向滤波器逐通道相乘。消息包含标量、沿边方向的新向量和旧 sender 向量输运；全部按邻居相加。Mixing 将向量投影成 U/W，以 |U| 和标量输入 MLP，再分别产生标量增量、向量门控和乘在 `U·W` 上的独立系数。

与自身实现的区别不仅是有没有内积。自身已有内积，但将它拼入 MLP；PaiNN 保留了一条明确的“可学习系数 × 向量内积”标量更新路径。PaiNN 也没有我们这套每层两次标量 Post-Norm 和固定 0.1 的节点向量修正系数。这些是完整参数化差异，不应挑一个操作单独移植后沿用原来的所有尺度。

[官方 Interaction/Mixing](https://github.com/atomistic-machine-learning/schnetpack/blob/69d26f3842a012f1f1e461ad7f1984d353843c8d/src/schnetpack/representation/painn.py#L13)。其输入通常是位置和元素，向量初始化为零，默认输出任务不是 MD 终点；本文不据其力场指标推断传播效果。

### GotenNet

`GATA.message:498–560` 包含两条相加路径：边滤波 × sender 特征 × cutoff，以及按邻居 softmax 的注意力 value。方向注入和旧 steerable 特征输运分别处理，各阶可有不同系数。随后节点更新，使用**更新后的**两端 steerable 表示计算几何边更新；EQFF 再在节点上混合标量与方向信息。

特别核对了配置覆盖：QM9 实验覆盖基础配置中的 weight decay、RBF 数、scheduler monitor/patience；GATA 的 `layer_norm=""`、`steerable_norm=""` 在这组调用中为 Identity。NodeInit 内的归一化是另一处运算。不能只看到 `h=self.layernorm(h)` 就把它归类为已启用的 Pre-Norm 网络，也不能把 loss 的 EMA 字段当成参数 EMA。

[GATA](https://github.com/sarpaykent/GotenNet/blob/44c945bccf5ae2cf68d61b2fd32ae1e5be3db8d6/gotennet/models/representation/gotennet.py#L398) · [QM9 覆盖配置](https://github.com/sarpaykent/GotenNet/blob/44c945bccf5ae2cf68d61b2fd32ae1e5be3db8d6/gotennet/configs/experiment/qm9.yaml) · [优化与 loss EMA](https://github.com/sarpaykent/GotenNet/blob/44c945bccf5ae2cf68d61b2fd32ae1e5be3db8d6/gotennet/models/goten_model.py#L479)。这是性质/势能任务实现，不能原样替换周期相空间传播。

## 5. 训练流程必须核对实际任务，不能只比优化器名称

| 代码/配置 | 被优化的量 | 已核实训练设置与边界 |
|---|---|---|
| 自身 Water 6 Å 配置 | 全目标 RMS 归一化位移与绝对速度；0.5 平均；额外 0.02 proposal | FP32，batch 8，AdamW AMSGrad，wd=0，clip=10，5000-step warmup，250k-step cosine，验证 final loss 选模型 |
| 本地 native TrajCast Water 配置 | 相同物理位移和绝对速度，各自 RMS，拼接六分量 MSE | FP64，batch 2，Adam AMSGrad，lr=0.01，plateau 0.8/patience20；实际默认 grad clip=∞，非自身的10；不能视为同优化预算 |
| GotenNet QM9 配置 | 属性 MSE；任务 handler 处理标签/偏移 | batch32，lr1e-4，warmup10000，wd0，AdamW eps1e-7，plateau0.8/patience15；checkpoint与scheduler监测字段须分别核对 |
| SchNetPack 保存的 ethanol PaiNN 示例 | 能量与力，权重0.01/0.99 | 该实例为3层128、5Å、batch10、FP32、参数EMA；不是 PaiNN 全部论文任务的统一配置，也不是 q/v 训练 |
| MDtrajNet 库默认训练 | 位置 MSE + 可配置速度导数 MSE | AdamW、plateau、任务分桶/批次上限；默认参数不能冒充论文专用配置；没有核实的论文设置标缺失 |
| GNS 官方训练 | 归一化有限差分加速度 | 6帧位置历史，10个不共享参数的处理步；输入随机游走噪声与标签调整配套；Adam指数衰减 |

GNS 特别说明为什么只复制训练名词容易出错：在带噪历史上训练时，`next_position_adjusted = next_position + last_position_noise`，再用与解码器一致的逆操作得到加速度标签。它有意纠正速度噪声，并不是从噪声状态运行真实物理得到后继。这是其任务与离散表示中的训练定义，不等同于“真实条件后继”。本文不据此给 DSRT 增加噪声损失或积分器。

[GNS 标签与解码](https://github.com/google-deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/learning_to_simulate/learned_simulator.py#L188) · [训练配置与优化](https://github.com/google-deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/learning_to_simulate/train.py#L305)。GNS 的 MLP 后 LayerNorm 位于更新分支，残差在外部相加，也不能简写成与自身相同的 Post-Norm。

自身训练代码另核实三项容易误读的细节：缓存的 `update_velocities` 是绝对速度，不是速度增量；验证函数虽调用具有默认 proposal 权重的辅助函数，实际选择用返回的 `final`，没有按错误权重选 checkpoint；epoch 内在线训练损失与 epoch 末固定权重验证损失不是完全同一模型状态下的两个统计量。未发现这些地方的确定性目标错误。

## 6. FlashMD 必须先分版本，再讨论成功机制

当前官方 README 的论文复现说明指向 `flashmd==0.1.2`。该固定版本 `stepper.py` 将网络输出的质量尺度位移除以 sqrt(m)，将质量尺度动量乘以 sqrt(m)，不是当前 metatrain 接口的直接同名输出约定。

更关键的是 `ase/velocity_verlet.py:21,51–73`：`rescale_energy=True` 为默认值，每步调用两次 `get_total_energy()`，再重标定动量。Bussi/Langevin 包装继承该开关。**这是已核实的实现默认值，不是说每条论文轨迹一定开启了它。** 需要具体运行配置才能作该结论。

当前 README 明确说明论文结果使用 i-PI 接口，因此不能用 ASE 默认值推断论文中每条轨迹的设置。旧版与当前版的质量缩放、运行器选项、论文实际 XML/config 必须分别对齐；本次没有宣称完成最后这一项。
旧版 `ipi.py` 的 vv/mts 入口也有默认开启的 `rescale_energy` 参数及势能调用，但论文具体运行是否覆盖默认值仍需原配置确认。

当前 `metatrain 2026.3.1` 的 FlashMD 则包含 Cartesian Transformer、动量输入、旋转数据增强、additive 模型与 scaler；它不应直接归类为严格逐层 O(3) irreps 网络，更不能作为旧版论文网络的唯一源码证据。本次没有完成旧论文训练包到全部实验配置的追溯，因此不给旧版训练架构填入当前实现参数。

[v0.1.2 推演器](https://github.com/lab-cosmo/flashmd/blob/1189ebe1d5ebab650a2b4eb87b42f680298d1481/src/flashmd/stepper.py#L64) · [默认能量重标定](https://github.com/lab-cosmo/flashmd/blob/1189ebe1d5ebab650a2b4eb87b42f680298d1481/src/flashmd/ase/velocity_verlet.py#L51) · [当前架构](https://github.com/metatensor/metatrain/blob/f01e79535305e9c48eb39baa9f6b7cf75d9b6f07/src/metatrain/experimental/flashmd/model.py)。

这项审查用于厘清任务和接口，不授权在我们的推演器中引入势能修正。

## 7. 实际计算核对：限定用途的事实

16 个训练图，每图 192 原子；以下数值用于计算契约检查，不是模型评估或设计排名。

| 核对项 | 结果 | 能说明什么 |
|---|---:|---|
| 位移目标分量 RMS | 0.055478 Å | 当前任务实际尺度 |
| 去掉 ballistic 项后的真实位置残差 RMS | 0.050031 Å | 残差并非比目标小很多 |
| 绝对终点速度分量 RMS | 0.012735 Å/fs | 速度目标尺度 |
| 真实速度增量 RMS | 0.016482 Å/fs | 5fs不能默认理解为很小的速度修正 |
| 同存储权重 FP32/FP64 位移输出差 RMS | 1.26e-7 Å | 包含输入 cast 与前向运算差；不是双精度训练比较 |
| 同存储权重 FP32/FP64 速度输出差 RMS | 1.94e-8 Å/fs | 不能据此判定长期累积或训练精度效应 |
| 原始邻居数范围 | 74–105 | 6Å当前图真实边密度 |
| cosine 权重和范围 | 13.11–20.25 | 自身逐原子除数；不同于native固定平均原始度89.84 |
| q/v归一化 MSE 与六分量拼接MSE | 数值一致 | 核实0.5系数、绝对速度语义 |
| native标签缩放往返 | 通过1e-12容差 | forward会改写传入图中的标签，调用必须使用新batch或clone |
| 一次真实batch反向 | 各参数组梯度有限；未更新 | 不是缓存detached分数；不证明所有训练步稳定 |

自身首个训练图四层标量 RMS 约1.00–1.02、向量约1.06–1.13；这只核实该输入/权重下的尺度，不能当成归一化设计已经最优或有害的证据。质量守恒与相空间语义仍按现有接口保留；本次没有扩展为全套物理成功判据。

## 8. 审查结论的边界

已确认的是实现差异和计算语义，尚未确认哪项差异决定长期动力学质量。当前代码足以排除一些错误概括：自身不是没有几何耦合；参考网络也不是同一种网络只换层数和参数量；归一化、残差和训练标签不能脱离所处的完整更新算子讨论。

后续设计以算子的信息流、表达能力、数值参数化及实际执行成本为依据。历史实验保留为档案，不作为这轮选择架构的判据。不得把本报告中的相关差异升级为已证明的因果解释。
