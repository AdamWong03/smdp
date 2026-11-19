# 前沿实现：表达能力与执行效率如何一起设计

依据用户最新要求，本文件不使用历史实验输赢决定网络设计。只讨论固定源码中的运算、有效配置和适用边界。它与 [逐路径审查](AUDIT.md) 配套。

## 1. E2Former-V2：不是靠缩小网络获得效率

论文官方仓库 `IQuestLab/UBio-MolFM` 的 `e2formerv2` 分支固定到 `24c49d48260c252b888aacbf4a1493a074818b7a`。审查了模型、SO(2) 张量积、两个 Triton kernel 的前向/反向、wrapper、训练循环、优化器选择和配置；未在本机运行或测量其 kernel。

有效 backbone 配置使用 `attn_type=so2-first-order`、`tp_type=QK_alpha+triton`、4 个传播块。配置中同时保留了 DiT 的 1024 维/12层参数，但 `encoder=default` 时 wrapper 使用 irreps 的通道数。因此不能看 YAML 就把未启用的另一套编码器算进模型。

### 实际计算路径

`e2former.py:219–350`：Q/K/V 的线性投影先在节点上计算。稀疏索引选择邻居，径向/元素边特征生成 gate 和 attention 权重。Triton QK 路径在 kernel 内读取邻居 K，避免 PyTorch 分支的 `key[idx]` 大张量。权重仍是 `[N,K,H]`，没有把所有与边相关的内存消除。

`so2_tensor_product.py:332–374`：利用分解后的节点位置作用，先做扩展节点上的旋转与 SO(2) 运算，将两个项打包后按邻居聚合，再做接收节点上的对应运算及差值/线性混合。Triton value 聚合避免显式构造 `value[idx] * alpha` 的完整高维中间张量。

这里有两层不同的贡献：**数学分解减少逐边高维运算；kernel 融合减少中间张量的写回、读取和分配。** 只在原网络外加 attention，或只调用 `torch.compile`，不会自动获得这两项效果。

### 不应直接套用的部分

- 仍然存在边 gate、径向编码、稀疏权重与拓扑，不能把总计算量说成完全不依赖 E。
- kernel 明确支持 FP16/BF16/FP32，累加与部分梯度使用 FP32；不支持直接把整个输入设成 FP64。采用何种训练精度需和实际 kernel 绑定。
- 默认配置的 `max_neighbors=32` 与 cutoff 共同定义输入，不能只搬 cutoff 而漏掉邻居上限。
- 周期扩展节点索引、旋转矩阵及相对位置的代数约定必须一起移植；否则“等价改写”可能变成不同计算。
- 这里没有速度输入和 q/v 转移目标。其能量/力任务结果不证明它已经适用于直接 MD。
- 代码模板、任务默认值与 merged demo 的 loss 权重、学习率和 warmup 有区别。本次没有论文每项实验的 resolved config，不能把某个模板参数写成统一论文设置。
- 训练循环按 `idx % accumulation == 0` 更新；从 `idx=0` 开始时，`accumulation>1` 的首次窗口不是完整窗口。这是可定位的条件性实现风险，不代表已知论文运行采用了该设置；也说明官方 trainer 仍要逐条核对，不能直接复制。

来源：[ICML 2026 论文](https://proceedings.mlr.press/v306/huang26p.html) · [稀疏 attention 实际调用](https://github.com/IQuestLab/UBio-MolFM/blob/24c49d48260c252b888aacbf4a1493a074818b7a/src/molfm/models/e2former/e2former.py#L200) · [SO(2) 分解](https://github.com/IQuestLab/UBio-MolFM/blob/24c49d48260c252b888aacbf4a1493a074818b7a/src/molfm/models/e2former/wigner6j/so2_tensor_product.py#L332) · [kernel 数据类型与反向](https://github.com/IQuestLab/UBio-MolFM/blob/24c49d48260c252b888aacbf4a1493a074818b7a/src/molfm/models/e2former/triton_dr/triton_sparse_qk_autograd.py#L217) · [训练累积边界](https://github.com/IQuestLab/UBio-MolFM/blob/24c49d48260c252b888aacbf4a1493a074818b7a/src/molfm/pipeline/trainer.py#L364)。

## 2. EquiformerV3：归一化、非线性与边界处理是配套的

官方仓库 `atomicarchitects/equiformer_v3` 固定到 `a7300c58df683dc99cb48027d5bfd4c887486c48`。重点阅读 `experimental/models/equiformer_v3/`、官方 OC20/MPtrj 配置和 trainer 的标签归一化、累积及 EMA 路径。未执行该模型训练。

### 不是普通 LayerNorm 换个位置

`EquivariantMergeLayerNorm:194–278` 只对 l=0 标量作通道中心化，然后用所有阶的旋转不变二阶量计算共同尺度；可按每阶的分量数平衡统计。每个阶/通道有仿射缩放，bias 只加在标量上。它在保留等变表示类型的同时协调不同阶的幅值。

这与“标量单独做 LayerNorm，向量完全走另一套尺度”是不同参数化。也与把 xyz 展平后做普通 LayerNorm 不同。源码中虽然采用 Pre-Norm 残差，但其处理对象和统计量都必须一起理解。

### 非线性实际混合了什么

官方配置启用 `sep-merge_gates2_swiglu`。`activation.py:419–470` 将标量路径做 SwiGLU；另一条路径将球谐系数投影到 S² 网格，拆成两份后相乘，再投回，并用标量 gate 调节；l=0 输出合并两条路径。

关键并非激活函数名称，而是非线性使不同方向分量发生乘性交互，随后重新投回保留的表示空间。网格分辨率、lmax/mmax、投影与 dropout 都是实现的一部分；不能只提取一个 SwiGLU 代替整个运算，也不能忽略有限网格的数值等变误差。

### cutoff 在 softmax 中和外部都有作用

`GraphSoftmax` 先用 envelope 缩放指数值，再按邻居归一化。attention 层随后再乘一次 envelope。若只在 softmax 分子和分母中加入 envelope，当仅剩一个邻居时，它的归一化权重仍接近1；外部 envelope 让最终消息也趋于零。这说明“出现两次 cutoff”不能仅凭代码外观认定是重复或错误，必须分析最后一条边消失时的极限。

### 训练配置也是方法的一部分

已核实的 OC20 配置使用8层、lmax6、merge norm、energy/force不同权重、AdamW、梯度累积、cosine与参数EMA；MPtrj配置改为7层/lmax4，并启用去噪任务。trainer对物理力目标和去噪目标分开归一化、评分。不能把这些全部简写为“端点MSE+AdamW”，更不能直接把力场任务的去噪监督加入我们的方法。

EquiformerV3 的公开定位是 SE(3) 模型。迁入需要旋转和反射性质的传播器时，必须明确每个表示的宇称及非线性选择规则；有球谐函数不自动等于任意代码组合都满足 O(3)。

来源：[官方项目](https://github.com/atomicarchitects/equiformer_v3/tree/a7300c58df683dc99cb48027d5bfd4c887486c48) · [跨阶归一化](https://github.com/atomicarchitects/equiformer_v3/blob/a7300c58df683dc99cb48027d5bfd4c887486c48/experimental/models/equiformer_v3/layer_norm.py#L194) · [S²混合](https://github.com/atomicarchitects/equiformer_v3/blob/a7300c58df683dc99cb48027d5bfd4c887486c48/experimental/models/equiformer_v3/activation.py#L419) · [attention与截断](https://github.com/atomicarchitects/equiformer_v3/blob/a7300c58df683dc99cb48027d5bfd4c887486c48/experimental/models/equiformer_v3/transformer_block.py#L311) · [实际反向更新](https://github.com/atomicarchitects/equiformer_v3/blob/a7300c58df683dc99cb48027d5bfd4c887486c48/experimental/trainers/equiformer_v3_dens_trainer.py#L1143)。

## 3. 对自身代码的直接含义

审查对象应该是以下完整链条，而不是给每篇论文挑一个组件：

```text
相空间输入与单位
  → 可跨层保留的几何/运动表示
  → 边上的条件化交互
  → 邻域聚合与尺度
  → 节点上的运动条件化和非线性混合
  → 物理输出参数化与对应训练目标
```

表达和效率需要在同一张计算图上定义。自身 `pair_mlp` 已把节点仿射投影从 E 次缩为 N 次，但仍显式保存边几何和消息。例如本次首个训练图 E=17,276，单个 FP32 `[E,336]` 几何张量约22.14MiB，`[E,469]` 拼接输入约30.91MiB，`[E,432]` 消息约28.47MiB；8个同规模图分别约177.15/247.27/227.76MiB。**这些是单张量尺寸计算，不是实测峰值显存，也不能直接求和当作峰值。** 它说明为什么参数量不大仍可能有较高训练开销。

因此下一版的计算预算应首先约束边激活、张量积路径及scatter/gather次数，而不是先缩小节点通道或一味增大batch。可提到节点上的线性运算先提到节点；能融合的读取—加权—聚合保留清楚的参考实现；涉及非线性或归一化的操作则不能随意交换次序。

速度接口同样不能只做拼接。需要明确自身运动表征在哪一层与聚合后的环境表征发生乘性交互，输出的速度是独立回归量、残差量还是坐标对时间的导数；不同选择对应不同计算成本与训练含义。

这里确定的是**设计与实现的要求**，不是已经选定“把E2Former、EquiformerV3和TrajCast合起来”的架构，更不是由论文发表级别推导某个组件一定有效。下一步的候选必须给出完整算子定义、实际启用配置、张量尺寸和前后向路径，才能进入正式训练。
