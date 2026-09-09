# ScaleBFM 首步学习率诊断（2026-09-07）

后续开发：已增加 [PPO 更新内部诊断](SCALEBFM_PPO_DIAGNOSTICS_20260907.md)，
原始 A/B 结果保留如下；新增日志不会反向补出旧实验没有记录的 KL。

## 问题与假设

本地旧微调在八种 mask 全部退步。恢复学习率标量后，从官方出发的 5-update、
`entropy_coef=0.001` 小实验仍在部分模式退步，因此不直接延长训练。
这里只诊断初始学习率，不把退步预先归因于某个单一超参数。

现有上游 PPO 每次 minibatch 更新前用 actor 分布 KL 调整两个 optimizer 的学习率。
公式中 `log(sigma / old_sigma + 1e-5)` 在新旧分布相等时仍约为每动作维 `1e-5`，
29 维合计约 `2.9e-4`，会满足 `0 < KL < desired_kl / 2` 并放大学习率 1.5 倍。
因此官方保存的 `8.649755618534982e-4` 可能在首次梯度更新前放大至约 `1.29746e-3`。
之后多次高 KL 又能迅速把它降至 `1e-5`；迭代末 TensorBoard 的学习率不能证明
整个迭代始终使用该值。这是代码机制，不是已实测的逐 minibatch KL 轨迹。

本次不修改 PPO 的 KL 公式、源数据、mask 分布或网络结构。

## 单变量设计

- A：已有 scout 的第一个更新，
  `amass_full_v1_entropy001_scout5_20260907/model_22199.pt`。
- B：从官方 checkpoint 的派生副本出发，仅将 actor/critic Adam 的保存学习率改为
  `1e-5`，其余模型张量、Adam moments、参数组字段、内部 iteration 和 metadata
  逐项验证相等，然后同样只执行 **1 次更新**。
- 两组均使用原训练索引 4247 条、128 环境、seed 42、64 步 rollout、
  `entropy_coef=0.001`、adaptive schedule、关闭训练内评估。
- 比较 Pelvis-1 与 UMI-2，沿用 512 环境、seed 42、962 条验证片段、最多 1000 步、
  `--mask_metrics`。这个两模式筛查不是完整八模式 gate，不挑最好片段。
- GPU 物理不保证逐位确定；单次对照只能提供初步证据，不能作统计因果结论。

adaptive 在 B 首步也可能先把 `1e-5` 放大到 `1.5e-5`，所以不能将其描述为
“固定学习率训练”。使用 fixed 会同时改变另一个实验变量，本轮不这么做。

## 派生副本与保留策略

官方来源：`humanoid_transformer_m/model_22200.pt`，内部 iteration 为 22199。

- 官方 SHA-256：`88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3`
- 派生目录：`logs/rsl_rl/g1_bfm_tracking_exp/official_lr1e5_derived_20260907/`
- 派生文件：`model_22200.pt`
- 派生 SHA-256：`269f17e040ad0c27f651a25097a2650380ff1a05714dde102625aabd8d327c46`

派生文件以新目录、独占创建方式写出；重新加载后与待保存对象递归比较一致。
把派生对象的两个 LR 字段恢复原值后，又与官方对象递归比较完全相同。
它不是重新训练的策略，不能将它本身宣称为改进模型。原官方和旧微调文件均不覆盖。

## 状态与判定

派生副本、B 的单步训练与 A/B 两模式评估均已完成，所有训练/评估进程退出码为 0。
B 实际执行 8192 个环境步（128 × 64）和 64 个 optimizer minibatch 更新，
进程退出码为 0；最终模型和 TensorBoard 标量均为有限数。

- B run：`amass_full_v1_lr1e5_step1_20260907`
- B checkpoint：`model_22199.pt`（这是更新后保存，不是未更新的副本）
- B checkpoint SHA-256：`acc641356553db3e569ce4fb1993bd24a8686a1969a4bc212b81a96153641395`
- A checkpoint SHA-256：`8bb4b4ef6095248a3003717882cd02706a16beec93b8e1de084a1de0e5dfee52`
- B 迭代末 actor/critic LR 均为 `1e-5`，动作 std 均值 `0.3471492231`。
- A/B 首轮 surrogate loss 均值为 `0.08879 / -0.002825`；两者数值不同不直接等同于
  行为质量改善，仍要看独立参考跟踪结果。
- 训练日志：`logs/amass_training/lr1e5_step1_20260907.log`
- A/B 评估日志：`logs/amass_training/lr_first_step_pair_20260907.log`

若低初始 LR 仅减轻退步但仍弱于官方，也不替换官方参考、不进入长训。
若没有明显区别，则不继续把初始 LR 当成主要解释；后续需检查更新内 KL/clip fraction
轨迹与本地数据分布差异。所有实验进程使用有时限的独立进程组。

## 实测结果：初始学习率影响明显，但未达标

每份报告均为 962 条片段、342637 个有效步、81 条截断片段；与官方对应模式的
协议、数据与 Python 源码 SHA 一致，输入前后未变。以下从每条 motion 重算，
“通过数”分母均为 962，仍是单连杆最大偏差 0.5 m 的诊断，不是跌倒成功率。

| 模式 / 权重 | active 位置 cm | active 姿态 rad | active 通过数 | all-14 通过数 |
| --- | ---: | ---: | ---: | ---: |
| Pelvis-1 / 官方未更新 | 5.265398 | 0.110198 | 961 | 835 |
| Pelvis-1 / A 高初始 LR 单步 | 5.572824 | 0.116719 | 961 | 811 |
| Pelvis-1 / B 低初始 LR 单步 | 5.243236 | 0.110026 | 961 | 834 |
| UMI-2 / 官方未更新 | 6.112479 | 0.134642 | 960 | 906 |
| UMI-2 / A 高初始 LR 单步 | 6.748742 | 0.148062 | 957 | 894 |
| UMI-2 / B 低初始 LR 单步 | 6.107956 | 0.136793 | 960 | 911 |

B 相对 A 两模式的位置/姿态误差及 all-link 通过数均改善，支持“较高初始 LR 的
首步更新会造成明显偏移”这一解释。但没有记录逐 minibatch KL，且为单种子结果，
不是排除其他因素的统计因果证明。

B 相对**未更新官方**仍未同时通过四项无退步标准：Pelvis 的 all-link 少通过 1 条，
UMI-2 的姿态误差高约 0.00215 rad。位置均值的微小下降不能抵消这些差异，也不能
据此宣称稳定提升。未评其余六模式，所以本诊断不产出八模式 comparer 的 overall gate。
本轮不替换默认权重，也不延长高 LR scout。

报告路径：`logs/evaluations/20260907_mask_mode_<0,1>_<officiallr_step1,lr1e5_step1>_seed42_v4.json`。
原始报告保留完整数值；上表仅作展示而舍入。

后续优先记录 PPO 每个更新内的 KL、ratio clipping 比例、实际 LR 最小/最大值，
再做低初始 LR 的有限更新与完整八模式对照；增加种子后才判断接近基线的小差异。
不把验证难例放回训练，也不以更长训练替代诊断。

## B 的训练参数复现

从 `ScaleTrack/` 目录使用现有 IsaacLab Python，外层仍需有时限的独立进程组；
`replace-with-a-new-run-name` 必须替换为新的 run 目录名，不重用本文已有路径。

```bash
"$BFM_ISAAC_PYTHON" -u scripts/pretrain/rsl_rl/train.py \
  --task G1-BFM-Transformer-Tracking \
  --motion_file /home/sw/bfm/ScaleRetarget/retargeted_dataset/amass_full_v1_train.yaml \
  --test_motion_file /home/sw/bfm/ScaleRetarget/retargeted_dataset/amass_full_v1_validation.yaml \
  --run_name "replace-with-a-new-run-name" --logger tensorboard \
  --num_envs 128 --max_iterations 1 --seed 42 \
  --resume True --load_run official_lr1e5_derived_20260907 --checkpoint model_22200.pt \
  --device cuda:0 --headless \
  agent.algorithm.entropy_coef=0.001 agent.algorithm.schedule=adaptive \
  agent.eval_during_training=False agent.save_interval=1
```

训练索引 SHA-256：`b415332e3e998dad1f75e297d4a62b98b5f545b7cabe1787e4a6db44aa8bb792`；
验证索引 SHA-256：`2ec2fe800f6a3fc27e98af58b18037752058ef263a0064982671bc138aaf2529`。
