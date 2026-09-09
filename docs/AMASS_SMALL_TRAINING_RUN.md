# AMASS 小规模多动作微调记录

当前状态（2026-09-05）：首阶段 100 次更新已完成并通过检查，其进程和控制组已全部
退出；已启动第二阶段 900 次更新。第二阶段尚未完成，状态应以实时日志为准。

## 2026-09-05 本轮配置

这是从官方 M 权重开始的 G1 多动作跟踪微调，不是从零预训练，也不代表完整 BFM 效果复现。
沿用现有 IsaacLab，未重装环境、未修改训练算法、未覆盖旧 checkpoint。

| 项目 | 本轮设置 |
| --- | --- |
| 任务 | `G1-BFM-Transformer-Tracking` |
| 起点 | `humanoid_transformer_m/model_22200.pt` |
| Python | `/home/sw/isaaclab_ws/env_isaaclab_sim6_newton/bin/python` |
| 设备 | RTX 5080 / `cuda:0`，无 GUI |
| 并行环境 | 128 |
| 随机种子 | 42 |
| 首阶段 | 100 次 PPO 迭代，检查稳定性后再决定是否继续 900 次 |
| 每环境每轮采样 | 64 步 |
| 首阶段保存、评估间隔 | 每 50 个全局迭代编号，评估每条动作最多 1,000 步 |
| 数值库线程 | OMP / MKL / OpenBLAS / NumExpr 各 1 |
| 进程保护 | 独立 systemd 用户服务，45 分钟上限，停止信号 SIGINT，20 秒后强制清理整个控制组，无自动重启 |

官方 checkpoint 带已有迭代编号，因此日志中的全局编号不是本轮从零开始的次数。
本轮实际从全局 `22199` 开始，100 次更新的最后编号为 `22298`。

评估结果沿用仓库内置协议：每条动作在评估窗口中，最大 `error_body_pos_g` 不超过
0.5 米则计为通过。该阈值较宽松，通过率不是“BFM 复现百分比”，也不是接触质量或
真实机器人安全性指标。该运行器会在相邻训练轮重复记录上次评估值，分析时只取真正
触发评估的全局步 `(step + 1) % eval_interval == 0`，不把重复值当作新评估。

## 数据与隔离

| 集合 | 动作数 | ACCAD | BMLrub | 50 Hz 帧数 | 时长 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 训练 | 128 | 37 | 91 | 46,063 | 921.26 秒 |
| 验证 | 32 | 13 | 19 | 9,930 | 198.60 秒 |

- [训练索引](../ScaleRetarget/retargeted_dataset/amass_small_locomotion_v1_train.yaml)
- [验证索引](../ScaleRetarget/retargeted_dataset/amass_small_locomotion_v1_validation.yaml)
- [筛选规则、统计和索引哈希](../logs/amass_preparation/small_locomotion_v1_manifest.json)

ACCAD 选择有明确名称的站立、重心摆动、摆臂、观察、行走、转身和侧步；排除拿箱、
跳跃、爬行、蹲起等首轮之外的动作。BMLrub 选择 `normal_walk1`、`normal_walk2`、
`circle_walk`。ACCAD 的 Male1 留作验证；BMLrub 按受试者的固定哈希分组，再按受试者
轮流选取。两组受试者标识和源文件 SHA-256 均无交叉。

启动前重新验证了全部 160 个打包文件 SHA-256，与全量来源审计吻合；检查了有限值、
v3 / 50 Hz / 29 关节 / 30 刚体契约、2–20 秒片长、骨盆高度、速度、朝向和关节速度。
351 条候选中有 17 条未通过本轮保守数值筛选；它们未被删除，仍保留在全量素材池中。

数值筛选不等于已通过碰撞、接触、完整视觉质量或物理跟踪验收。验证集仅与本轮微调
训练集隔离；官方预训练是否见过这些 AMASS 动作未知，不能把它称为严格的预训练未见
数据测试集。首轮训练后仍需检查回放与误差。

## 首阶段运行

- run：`amass_small_locomotion_v1_probe_20260905_1536`
- 服务：`bfm-amass-small-locomotion-probe-20260905-1536.service`
- [标准输出日志](../logs/amass_training/small_locomotion_v1_probe_20260905_1536.log)
- [checkpoint / TensorBoard 目录](../logs/rsl_rl/g1_bfm_tracking_exp/amass_small_locomotion_v1_probe_20260905_1536)

查看实时进度：

```bash
tail -f /home/sw/bfm/logs/amass_training/small_locomotion_v1_probe_20260905_1536.log
```

停止这一轮及其子进程（不影响其他任务）：

```bash
systemctl --user stop bfm-amass-small-locomotion-probe-20260905-1536.service
```

服务正常退出或超时后会清理控制组并自动卸载临时服务；日志和已保存权重保留。
本文件的初始记录状态为已启动，完成情况以日志、checkpoint 和后续验收记录为准。

## 首阶段验收结果

15:37:23 启动，15:45:44 正常结束，共 100 次 PPO 更新、819,200 个仿真交互步。
日志没有致命异常，全部已记录标量与 checkpoint 中的 572 个张量均为有限值。

- [最终短训权重 model_22298.pt](../logs/rsl_rl/g1_bfm_tracking_exp/amass_small_locomotion_v1_probe_20260905_1536/model_22298.pt)
- [机器可读验收记录](../logs/amass_training/small_locomotion_v1_probe_20260905_1536_verification.json)

| 指标 | 结果 |
| --- | ---: |
| 最后 20 轮平均训练 reward | 17.4664 |
| 最后 20 轮平均 episode 长度 | 230.21 步 |
| 最后 20 轮训练身体位置误差 | 0.09579 米 |
| 最后 20 轮位置越界终止日志指标 | 0.00887 |
| 首次 / 中途训练集评估身体位置误差 | 0.06186 / 0.06117 米 |
| 首次 / 中途验证集评估身体位置误差 | 0.06415 / 0.06416 米 |
| 内置评估通过率，训练 / 验证 | 100% / 100%（0.5 米阈值，含已说明的局限） |

真实评估点为全局迭代 `22199`、`22249`，不是第 100 次更新后的独立评估。
训练和验证误差计算协议不同，不能直接把两个数比较为泛化差距。
没有对该数值通过率作完整 BFM 复现、视觉质量或真实机器人安全保证。

## 第二阶段：继续 900 次更新

从短训的 `model_22298.pt` 继续微调，合计计划 1,000 次 PPO 更新。
相同数据划分、种子 42、128 个环境；保存间隔改为 100，评估间隔改为 200，
单条评估最多 1,000 步。未修改 PPO 算法或网络结构。

- run：`amass_small_locomotion_v1_finetune_20260905`
- 服务：`bfm-amass-small-locomotion-finetune-20260905.service`
- [当前训练日志](../logs/amass_training/small_locomotion_v1_finetune_20260905.log)
- [当前权重和 TensorBoard 目录](../logs/rsl_rl/g1_bfm_tracking_exp/amass_small_locomotion_v1_finetune_20260905)
- [启动命令与配置快照](../logs/amass_training/small_locomotion_v1_finetune_20260905_launch.json)

按短训速度估计约需一小时，取决于同机其他负载；服务硬上限为 120 分钟。
达到 900 次更新后正常结束；异常、主动停止或超时也会清理该服务的整个控制组，
20 秒宽限后可强制终止，无自动重启。保留所有已完成 checkpoint 和日志。

```bash
# 查看实时进度（Ctrl-C 只退出查看，不停止训练）
tail -f /home/sw/bfm/logs/amass_training/small_locomotion_v1_finetune_20260905.log

# 停止本轮训练及其所有子进程
systemctl --user stop bfm-amass-small-locomotion-finetune-20260905.service
```

运行器恢复时会复用上次的最后迭代编号，所以两阶段的编号 `22298` 会重复一次；
实际更新次数仍是 100 + 900，不应仅用全局编号差值推算。环境状态与随机数状态没有
逐位恢复，因此这不是与连续单次 1,000 更新完全相同的仿真轨迹。
