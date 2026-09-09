# ScaleBFM 八模式配对结果（2026-09-07）

## 结论

本页保留两组正式八模式比较，二者相对同一官方 checkpoint 的逐模式 no-regression
结果均为 **0/8**，overall gate 均为 `false`：旧本地 checkpoint 不替换官方参考；随后
从官方启动的 5-update scout 也不替换、不延长。这些结果不表示模型“达标”，也不是收敛
或统计显著性的证据。

- 官方参考：`humanoid_transformer_m/model_22200.pt`，SHA-256
  `88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3`
- 旧本地候选：`amass_full_v1_finetune_20260906/model_23197.pt`，SHA-256
  `439890f2e4c50506e3c62e2b88d501fa624a705d1f4209a24450ea5faf30cd4a`
- 正式比较：`logs/evaluations/20260907_mask_paired_comparison_v1.json`
- 比较器源码 SHA-256：
  `7f167ee206ea19be7d0658a7f496454a092cdb8e1b67cc016b7c663918abdc1b`

## 旧本地候选八模式结果

位置是 active 连杆世界坐标误差的片段等权均值，表中由米换算为厘米；姿态单位为弧度。
通过率按 clip 统计：任一有效时刻的任一对应连杆位置误差严格大于 0.5 m 即失败。
“active”只统计当前 mask；“all”统计全部 14 个配置参考连杆。官/旧分别表示官方参考和
旧本地候选。

| 模式 | active 位置 cm（官 / 旧） | active 姿态 rad（官 / 旧） | active 通过率（官 / 旧） | all-14 通过率（官 / 旧） | gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pelvis-1 | 5.265 / 6.747 | 0.1102 / 0.1558 | 99.90% / 98.86% | 86.80% / 85.45% | 未通过 |
| UMI-2 | 6.112 / 8.038 | 0.1346 / 0.2159 | 99.79% / 98.86% | 94.18% / 90.54% | 未通过 |
| VR-3 | 5.432 / 7.374 | 0.1255 / 0.1942 | 99.90% / 99.06% | 98.86% / 96.67% | 未通过 |
| UMI-4 | 6.699 / 8.389 | 0.1739 / 0.2123 | 99.38% / 98.44% | 99.38% / 98.34% | 未通过 |
| VR-5 | 6.095 / 7.711 | 0.1586 / 0.2001 | 99.69% / 98.54% | 99.69% / 98.44% | 未通过 |
| UpperBody-6 | 5.503 / 7.365 | 0.1073 / 0.1787 | 99.79% / 98.75% | 95.95% / 92.93% | 未通过 |
| UpperBody-Mobile-7 | 5.344 / 7.121 | 0.1095 / 0.1728 | 99.79% / 98.75% | 99.06% / 96.26% | 未通过 |
| WholeBody-14 | 5.679 / 7.104 | 0.1325 / 0.1792 | 99.58% / 98.02% | 99.58% / 98.02% | 未通过 |

旧本地候选的八种模式在 active 位置、active 姿态和 active/all 通过率上都弱于官方，
因此未通过 gate。该组原有 16 份报告和正式 comparison JSON 保持不变。

## entropy-0.001 scout5 八模式结果

本节的“scout”专指新 run `amass_full_v1_entropy001_scout5_20260907`，不是上节的旧本地
checkpoint。它从官方 `model_22200.pt` 恢复，实际执行 5 次 update，得到：

- scout checkpoint：`model_22203.pt`
- SHA-256：`d014253e94c1aaf3d289436a61d0b6eb74055c961fa419b8968054ab7131c8e0`
- 正式比较：`logs/evaluations/20260907_mask_scout5_comparison_v1.json`
- 训练规模：128 环境 × 64 rollout steps × 5 updates = 40,960 environment steps
- 训练配置：seed 42、`entropy_coef=0.001`、adaptive schedule

表中官/scout 分别表示官方参考和 scout5；指标口径和上表相同。

| 模式 | active 位置 cm（官 / scout） | active 姿态 rad（官 / scout） | active 通过率（官 / scout） | all-14 通过率（官 / scout） | gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pelvis-1 | 5.265 / 5.467 | 0.1102 / 0.1197 | 99.90% / 99.79% | 86.80% / 83.78% | 未通过 |
| UMI-2 | 6.112 / 6.879 | 0.1346 / 0.1480 | 99.79% / 99.48% | 94.18% / 92.72% | 未通过 |
| VR-3 | 5.432 / 6.135 | 0.1255 / 0.1365 | 99.90% / 99.79% | 98.86% / 98.34% | 未通过 |
| UMI-4 | 6.699 / 6.875 | 0.1739 / 0.1810 | 99.38% / 99.58% | 99.38% / 99.48% | 未通过 |
| VR-5 | 6.095 / 6.519 | 0.1586 / 0.1671 | 99.69% / 99.79% | 99.69% / 99.79% | 未通过 |
| UpperBody-6 | 5.503 / 6.516 | 0.1073 / 0.1167 | 99.79% / 99.90% | 95.95% / 95.84% | 未通过 |
| UpperBody-Mobile-7 | 5.344 / 6.314 | 0.1095 / 0.1186 | 99.79% / 99.79% | 99.06% / 98.75% | 未通过 |
| WholeBody-14 | 5.679 / 6.181 | 0.1325 / 0.1385 | 99.58% / 99.69% | 99.58% / 99.69% | 未通过 |

scout 在部分模式的 0.5 m 通过率持平或略高，但八种模式的 active 位置和姿态均值都高于
官方；gate 要求四项同时不退步，因此仍是 0/8。TensorBoard 在五次 update 结束时均
记录 actor/critic learning rate 为 `1e-5`。checkpoint 中策略 `std` 张量均值从官方的
`0.347084` 变为 scout 的 `0.346689`。update 末日志不能还原每个 minibatch 内的学习率，
而 `std` 变化也不能证明控制性能改善；独立八模式结果已经否决本轮延长。

## 数据量与协议边界

原有 16 份正式报告（官方/旧本地 × 8 个固定 mask）继续保留；scout 新增 8 份正式
报告。两组比较中的每份报告都已核对，均包含同一组 962 条 clip、342,637 个有效仿真步
和 81 条截断片段。评估协议为 512 环境、seed 42、50 Hz、每条
最多 1000 步；关闭观测噪声、重置扰动和周期推力，不加入物体，训练更新数为零。每条
统计 `min(source_frames - 1, 1000)` 个物理步，先在 clip 内平均，再对 962 条 clip
等权平均。失败片段不剔除。

这是固定一种 mask 后运行的、同一离线参考数据上的单种子比较，不覆盖在线任意目标、
多种子稳定性、抗扰动、跌倒检测或物体任务。max-link 0.5 m 只是参考失配诊断；模拟在
越线后继续，因此通过率不是跌倒成功率，也不是任务成功率。512 环境改变了批次分组和
启动随机化分配，所以本结果不能与此前 128 环境报告直接当作同协议前后提升。

## no-regression gate

比较器从 motion 行重新计算数值，不复制 `summary`。每个模式同时满足以下条件才通过：

1. 候选 active 位置均值不大于官方值加 `1e-6` 米；
2. 候选 active 姿态均值不大于官方值加 `1e-6` 弧度；
3. 候选 active max-link 0.5 m 通过率不低于官方；
4. 候选 all-14 max-link 0.5 m 通过率不低于官方。

八个模式全部通过才允许 overall gate 为真。比较器还拒绝缺模式、重复模式、非规范 G1
mask、混用权重、协议或软件环境不一致、数据/源码 SHA 不一致、clip 身份或步数不一致、
非有限/不可能的指标以及已有输出路径。这个 gate 只表示受控配对下没有观察到上述四项
退步，不是形式化统计检验、收敛判定或完整 BFM 验收。

## 复现比较

下面命令只使用 CPU 和 Python 标准库读取既有报告；输出必须是尚不存在的新路径。

```bash
python3 scripts/compare_mask_evaluations.py \
  --reference \
  logs/evaluations/20260907_mask_mode_0_official_seed42_v4.json \
  logs/evaluations/20260907_mask_mode_1_official_seed42_v4.json \
  logs/evaluations/20260907_mask_mode_2_official_seed42_v4.json \
  logs/evaluations/20260907_mask_mode_3_official_seed42_v4.json \
  logs/evaluations/20260907_mask_mode_4_official_seed42_v4.json \
  logs/evaluations/20260907_mask_mode_5_official_seed42_v4.json \
  logs/evaluations/20260907_mask_mode_6_official_seed42_v4.json \
  logs/evaluations/20260907_mask_mode_7_official_seed42_v4.json \
  --candidate \
  logs/evaluations/20260907_mask_mode_0_finetuned_seed42_v4.json \
  logs/evaluations/20260907_mask_mode_1_finetuned_seed42_v4.json \
  logs/evaluations/20260907_mask_mode_2_finetuned_seed42_v4.json \
  logs/evaluations/20260907_mask_mode_3_finetuned_seed42_v4.json \
  logs/evaluations/20260907_mask_mode_4_finetuned_seed42_v4.json \
  logs/evaluations/20260907_mask_mode_5_finetuned_seed42_v4.json \
  logs/evaluations/20260907_mask_mode_6_finetuned_seed42_v4.json \
  logs/evaluations/20260907_mask_mode_7_finetuned_seed42_v4.json \
  --output logs/evaluations/20260907_mask_paired_comparison_recheck.json
```

预期终端摘要为 `0/8 modes pass no-regression; overall=false`。正式归档 JSON 保留完整
小数和逐模式 candidate-minus-reference delta；上表仅为展示而舍入。

scout5 使用同一比较器；如需重算，也必须选择尚不存在的新输出路径：

```bash
python3 scripts/compare_mask_evaluations.py \
  --reference logs/evaluations/20260907_mask_mode_[0-7]_official_seed42_v4.json \
  --candidate logs/evaluations/20260907_mask_mode_[0-7]_entropy001_scout5_seed42_v4.json \
  --output logs/evaluations/20260907_mask_scout5_comparison_recheck.json
```

## scout5 训练复现

从 `/home/sw/bfm/ScaleTrack` 运行，使用现有 Isaac Python。归档 run 名已经占用；重跑时
必须将下面的 `--run_name` 换成新的唯一名称，避免覆盖 checkpoint 和 TensorBoard 事件。
下面仅列训练参数，外层仍需有时限的独立进程组，并在结束后检查无残留。

```bash
cd /home/sw/bfm/ScaleTrack
/home/sw/isaaclab_ws/env_isaaclab_sim6_newton/bin/python -u \
  scripts/pretrain/rsl_rl/train.py \
  --task G1-BFM-Transformer-Tracking \
  --motion_file /home/sw/bfm/ScaleRetarget/retargeted_dataset/amass_full_v1_train.yaml \
  --test_motion_file /home/sw/bfm/ScaleRetarget/retargeted_dataset/amass_full_v1_validation.yaml \
  --run_name amass_full_v1_entropy001_scout5_REPLACE_WITH_UNIQUE_NAME \
  --logger tensorboard --num_envs 128 --max_iterations 5 --seed 42 \
  --resume True --load_run humanoid_transformer_m --checkpoint model_22200.pt \
  --device cuda:0 --headless \
  agent.save_interval=1 agent.eval_during_training=False \
  agent.algorithm.entropy_coef=0.001 agent.algorithm.schedule=adaptive
```

归档 run 的名字是 `amass_full_v1_entropy001_scout5_20260907`。上述训练命令只生成新的
scout；完整结论仍需按前述八模式评估和比较流程重新产生，不能用训练日志代替。

## 后续状态

scout5 已完成但未通过 gate，不替换官方参考，也不继续延长。首步学习率的两模式诊断
也已完成：降低初始 LR 明显减轻退步，但相对官方仍未同时满足四项无退步要求，见
[`SCALEBFM_RESUME_LR_ABLATION_20260907.md`](SCALEBFM_RESUME_LR_ABLATION_20260907.md)；
它不是完整八模式 gate。旧 checkpoint、原始数据索引、原 16 份报告和新增 8 份 scout
报告均保留，不覆盖。
