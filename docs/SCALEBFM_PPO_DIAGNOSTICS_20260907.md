# PPO 更新内部诊断

## 目的与边界

承接 [初始学习率对照](SCALEBFM_RESUME_LR_ABLATION_20260907.md)：迭代结束的 LR
不能说明前面每个 minibatch 使用了什么学习率。本次只增加观测，不改变网络、
loss、adaptive 调度公式、梯度裁剪、optimizer 更新或动作采样。

`PPO` 默认启用 `log_update_diagnostics=True`。直接构造算法时可设为 `False`，
关闭新增计算和 `PPO/*` 标量。该选项目前未添加到 IsaacLab 的结构化配置类，
不要把它当成已经支持的 Hydra 命令行覆盖参数。

每次 `update()` 开始清空 `update_diagnostics`，成功结束后才公布本次结果。
返回值仍只有 `value_function`、`surrogate`、`entropy`，原 `Loss/*` 标签保留。
runner 将诊断单独写入 TensorBoard 的 `PPO/*`，使用相同的训练 iteration。
循环中只暂存脱离梯度图的设备端标量，更新结束后统一传回 CPU；不在每个
minibatch 增加 `.item()` / CUDA 标量同步。

## 指标含义

| TensorBoard 标签（`PPO/` 前缀） | 含义 |
| --- | --- |
| `minibatches` | 该次更新实际完成的 minibatch 数 |
| `policy_kl_mean/max/first/last` | 精确高斯 KL(old policy ∥ 当前 policy)，先对动作维求和、再对样本求均值 |
| `scheduler_kl_mean/max/first/last` | 上游 adaptive 调度器实际使用的 KL；保留其 `1e-5` 偏置，仅 adaptive 且 desired_kl 非空时存在 |
| `ratio_clip_fraction_mean/max` | 动作概率比满足 `abs(ratio - 1) > clip_param` 的样本比例，不是损失函数实际选择 clipped 分支的比例 |
| `actor_lr_first/last/min/max` | 调度完成后、Adam step 之前 actor 参数组的实际 LR |
| `critic_lr_first/last/min/max` | 同上，critic 的实际 LR |

`mean` 是本次更新内等大小 minibatch 均值的平均，`max` 是这些 minibatch 均值的
最大值；它们不是逐样本最大 KL，也不是最终更新后策略对全部 rollout 的 KL。
`first/last` 对应首次与末次梯度更新**之前**，不是末次更新之后的重新评估。

当前 actor/critic 各有一个 Adam 参数组，LR 从该实际组读取。精确 KL 和裁剪比例
是当前 rank 的局部统计；调度 KL 沿用原算法已有的跨卡平均。没有增加集体通信，
也不能把这些局部指标宣称为多 GPU 全局指标。此次验证范围是单 GPU 环境与 CPU 单测。

当新旧 29 维分布完全相同时，精确 KL 为 0，而原调度器由于 epsilon 仍约为
`2.9e-4`，可能将下一次 LR 放大 1.5 倍。两个标签并存是为了显式区分这一机制，
不是在日志中悄悄替换调度器公式。

## 验证记录

先增加真实 CPU PPO 测试并确认新增行为缺失（6 失败、1 通过），再实现诊断。
经独立审查修正浮点容差和循环内同步开销后，新增 **11 项测试**全部通过，覆盖：

- 已知 KL=0.5 和全部样本概率比为 2 的数值例子；
- 29 动作维、不等方差的解析对照，明确验证 KL 的 old → current 方向；
- adaptive LR 上升/下降，记录的是调度之后实际使用的 LR；
- 两个确定性种子覆盖 float32 舍入边界，解析 float64 对照使用 `1e-6` 绝对容差；
- 连续更新及失败更新不残留旧统计；
- 真实非零梯度下，启用与禁用诊断的模型、两个 Adam 状态、loss 和 RNG 逐项完全相同；
- 真实 TensorBoard writer 写入与重新读取，保留旧 Loss 标签并独立记录 PPO 标签。
- 开启诊断不增加 minibatch 循环中的标量 host reads。

2026-09-07 使用既有 IsaacLab Python 的完整相关回归：**205 passed、62 warnings**。
警告来自既有 IsaacLab 弃用 API / TorchScript，未借本次任务升级依赖。

```bash
cd /home/sw/bfm
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/sw/isaaclab_ws/env_isaaclab_sim6_newton/bin/python -m pytest \
  ScaleTrack/tests tests/test_gui_launcher.py tests/test_ci_workflow.py \
  tests/test_mask_report_compare.py -q --tb=short
```

独立代码复审通过；轻量 Python 3.11 CI 测试在本机另有 **101 passed**，不声称
已经跑过 GitHub 托管 CI。单测通过不代表策略质量提升。

## 实际 IsaacLab 更新验收

KIT 批处理结束后，使用新 v2 训练池 7174 条，从官方低初始 LR 派生副本执行
1 次更新：128 环境 × 64 步，seed 42，adaptive schedule，entropy 0.001，
2 epochs × 32 minibatches。与旧 v1 实验的数据池不同，因此不是它们的单变量质量对照。
实际采样包括 55 条 KIT、3520 个 KIT 环境步，详见 [KIT 接入验收](KIT_INGESTION_20260907.md)。

| 单次更新内统计 | 实测值 |
| --- | ---: |
| 完成 minibatches | 64 |
| 精确 policy KL first / mean / max / last | 0 / 0.018635 / 0.038954 / 0.015507 |
| scheduler KL first | 0.0002903938 |
| ratio outside-clip fraction mean / max | 0.241211 / 0.378906 |
| actor/critic LR first | 0.000015 |
| actor/critic LR min / max | 0.00001 / 0.0000759375 |
| actor/critic LR last | 0.00001 |

本次实测确认新旧策略首次 KL=0，但上游调度 KL 非零并先放大 LR；即使从 `1e-5`
起步，更新内部最高 LR 仍约为它的 7.59 倍。只看迭代末的 `1e-5` 会遗漏这段变化。
这些是每次更新的摘要，不是逐样本最大 KL、完整 64 点时序或行为成功率。
clipping 比例与 KL 用于后续调参诊断，不能单凭它们断言策略已经改善或失稳。

重新读取真实 TensorBoard 后，每个 `PPO/*` 标量都与 rollout 审计 JSON 对应，
共 55 个标量标签均为有限数。141 个模型张量发生改变，actor 71/critic 70 个 Adam
状态的 step 均增加 64，证明不是只加载模型而没有执行更新。

run：`amass_full_v2_lr1e5_diag_step1_20260907`；
checkpoint：`model_22199.pt`，SHA-256
`d8e253be07cd85410c19f173ae8e08710259bc2de9a49ace15701a4e8b1b9eed`。
runner 的内部 iteration 命名沿用既有语义，不用文件编号推断新增更新次数。
验证记录：`logs/amass_training/kit_diag_step1_20260907_verification.json`。

本次实际训练服务退出码 0，所有本轮仿真/训练进程已结束。未对该新 checkpoint
执行八模式质量 gate，未替换默认模型，未据此延长之前退步的训练。
下一项训练实验可隔离比较 fixed LR 与 adaptive LR，再使用相同数据/源码协议进行
官方与候选的八模式评估；不把这项后续实验写成已经完成。
