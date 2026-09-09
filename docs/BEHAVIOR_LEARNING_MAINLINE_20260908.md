# 回到行为学习主线：学习率对照与全训练池覆盖长训

用户于 2026-09-08 要求回到 ScaleBFM 主线；停止把 GUI、手写 waypoint 或步态参考桥的
开发量作为行为学习进度。本轮不使用 superpowers 或 isaaclab-api-context 技能，
直接检查仓库和已安装源码；保留现有 IsaacLab，不重装、不升级。

## 当前数据与学习状态

当前索引实查：9,487 条已打包机器人动作，按现有去重选择为 9,484 条。
这是片段数，不是行为类别数；只有合格训练数据参与参数更新。

| 来源 | 打包片段数 | 当前训练池 |
| --- | --- | --- |
| ACCAD | 252 | 197 |
| BMLmovi | 1,864 | 1,528 |
| BMLrub | 3,061 | 2,483 |
| CNRS | 79 | 39 |
| KIT | 4,231 | 2,927 |
| 合计 | 9,487 | 7,174 |

训练池 `amass_full_v2_train.yaml` 共 3,315,606 帧，50 Hz，约 18.42 小时。
旧验证 962 条 + KIT 留出 789 条 = 1,751 条；剩余去重后 559 条未进入当前平地训练/验证
划分，不删除原数据。KIT 的 515 条排除原因包括外部支撑/非平地线索、过短、标定与极端
根速度，详见 [KIT 入库审计](KIT_INGESTION_20260907.md)。

**7,174 条均可进入采样池，但每个短 rollout 只采到其中一部分。**
新增审计记录实际环境步对应的动作和 mask，不能用“已加载”冒充“已充分学习”。
旧微调和 scout 的八模式无退步门槛均未通过，仍以官方权重为基线。
最近两条 KIT 行走实验是该官方权重的零更新推理验收，不是新训练。

## 第一阶段：预先确定的学习率实验

第一阶段为小规模训练方案筛选，不是论文规模预训练；当时不自动延长长训。
该对照只测试一个变量：**fixed 与 adaptive 学习率调度**。
随后用户明确要求“开启长训练吧”，第二阶段单独立项，见下文。

- 起点相同：既有 `official_lr1e5_derived_20260907/model_22200.pt`。
  主控重新验证其模型、Adam moments、iteration 和其他字段与官方完全一致，
  仅 actor/critic 两个 Adam 参数组 LR 为 `1e-5`。不修改来源 checkpoint。
- 相同数据：v2 全部 7,174 条；不向训练进程传入验证索引，关闭训练内 evaluation。
- 相同 seed 42、128 环境、64 步 rollout、5 次 PPO update，即每候选 40,960 环境步。
- 相同原八模式随机采样、Transformer、原 motion-tracking rewards 和 domain randomization。
- 相同 entropy 0.001、2 epochs × 32 minibatches；每个 Adam 状态应增加 320 steps。
- 唯一有意差异：`agent.algorithm.schedule=fixed` 或 `adaptive`。
  恢复时真实初始 LR 以 checkpoint 为准；fixed 全程仍须由诊断核对为 `1e-5`。
- 保留两个候选独立 run 和所有 checkpoint，不覆盖官方或旧模型，不自动推广。

首步实际策略更新前的 rollout 用同种子初始化；GPU PhysX 不承诺逐位可复现。
策略更新后两组采样轨迹可以分化，这正是 on-policy 学习，不强行让后续动作一致。
LR 的 KL/clip 变化只是更新诊断，质量仍由独立物理评估决定。

## 新协议八模式评估

原始两个验证索引不修改；创建隔离的 1,751 条 union YAML，仅用于评估，
先检查训练/验证名称、解析后路径和 NPZ 精确内容不相交。
这些检查不能排除动作近重复、同一录制事件或官方预训练数据重叠。

原计划为官方未更新、fixed 和 adaptive 各跑八模式，共 24 份新报告
（实际只完成下文列出的两份评估，不能当作八模式结果）：

- 1,024 并行环境、seed 42、每片段最多 1,000 步、50 Hz。
- 每份 1,751 条片段、预计 605,664 有效步、98 条截断；不称全部片段全时长验收。
- 使用实际模型动作和机器人状态，记录 active 连杆位置/姿态及 active/all-14 最大单点
  偏差是否超过 0.5 m。这不是跌倒、足底接触或搬运任务成功率。
- 同一新协议重跑官方基线；不能将旧 512 环境结果拿来直接比较。
- 所有输入、数据 payload 与实际 Python 源码指纹锁定，运行结束再次核对未变化。
- 两个数据部分分别判定四项无退步：active 位置均值、active 姿态均值、active/all-14
  最大偏差门槛通过率。只有总体及 **两部分 × 八模式**全部满足，才具备进一步考察资格；
  不让 KIT 均值改善掩盖旧行为退步，也不自动替换权重。

本轮两个集合都是**未参与梯度更新的开发验证集**。因用于 fixed/adaptive 的选择与诊断，
KIT 789 条不能再称“从未用于模型选择的最终测试集”。更强的泛化主张仍需另外冻结最终
测试协议、更多种子和行为覆盖检查。本轮门槛不是统计显著性检验。

## 工程入口与进程边界

[主控](../scripts/run_bfm_learning_ablation.py) 默认只打印计划，实际执行必须显式
`--execute` 且位于自己命名的 systemd user cgroup 中；拒绝已有实验目录或候选 run。
训练复用原 `train.py` / runner / PPO，不增加手写腿部动作或在线目标桥。
[训练审计](../ScaleTrack/scripts/pretrain/rsl_rl/training_probe.py) 只观察实际采样和更新，
[分层比较器](../ScaleTrack/scripts/pretrain/rsl_rl/compare_learning_ablation.py) 从原始逐片段
报告重算结果，保留旧八模式比较器的协议和来源检查。

每个训练子进程上限 300 秒、每个评估上限 240 秒；外层 user cgroup 90 分钟总上限，
无自动重启，退出时清理整组子进程。逐任务日志和 `status.json` 在本次新实验目录下。
全量评估预计约 30–40 分钟，实际时间以运行日志为准，不能把后台启动写成已经完成。

## 第一阶段实际结果与中断边界

实验 `behavior_lr5_20260908a` 的 fixed/adaptive 两个训练子任务均通过执行审计：
各 5 次更新、40,960 环境步、71 个 actor 参数张量发生改变，actor/critic Adam 状态
各增加 320 steps。两组均实际采到 128 条动作，其中 KIT 55 条；不是全池覆盖。

- fixed：实际 LR 始终 `1e-5`；五次更新的平均 policy KL 约 0.004388，
  ratio 超出裁剪区间比例约 0.05573。
- adaptive：单次 update 内最高 LR 达 `1.1390625e-4`；平均 policy KL 约 0.022164，
  裁剪区间外比例约 0.24584。每次 update 最后 LR 又回到 `1e-5`，只读末值会遗漏波动。
- 这些是更新幅度诊断，不证明 fixed 的泛化质量胜出。
- 只有 `official_0`、`fixed_0` 两份 Pelvis-1 评估完成；`adaptive_0` 因用户转入长训的
  优先级变更而中断，其他模式未运行。主控 `status.json` 的 FAIL/KeyboardInterrupt
  是人为停止评估队列，不是两个已完成训练子任务失败。不作八模式无退步结论。
- 对应 systemd cgroup 已停止，训练产物、部分评估和原始日志全部保留。

## 第二阶段：显式授权的覆盖长训

关闭训练内 evaluation 后，旧路径只在构造时抽一次 128 个 motion IDs；episode reset
改变的是片段时间与 mask，不会换片段。上述对照每次更新的动作计数也证实了这一点。
所以“训练池有 7,174 条”并不等于“长时间运行就会遍历它们”。这是已确认的覆盖问题；
它是否解释历史质量退步，尚不能单独确定。

新增 `motion_sampling_strategy=coverage`、`motion_resample_interval=5`，仅显式开启时生效：

- 独立 CPU 随机数生成器按随机排列遍历训练索引，每 5 次完整 PPO update 切换 128 条。
  每次先完成 rollout、计算 returns 和 update，随后更换参考并完整 reset 机器人/历史，
  不在 rollout 中途换片段，不复用 reset 前的观测。
- 默认 `legacy` / interval 0 保持旧路径；目前覆盖模式只支持单 GPU、均匀训练采样。
  显式拒绝自适应权重、训练内 evaluation 或装载验证池，不悄悄丢弃这些设置。
- checkpoint 保存 sampler、目录顺序哈希和 interval；训练恢复时验证后推进下一批。
  恢复的是后续采样顺序，不是机器人、episode、rollout 或全部 RNG 的逐位续接；
  中断批次剩余的 update 不补齐，恢复时重新开始一个五次更新窗口。
- 实际采样审计只在真实 `env.step` 和 transition 处理成功后计数。
  第 281 次更新预计首次采遍 7,174 条；sampler 分配计数不能替代实际步计数，
  “采遍”也不代表所有帧或所有 mask/动作组合都已充分学会。

长训计划：Transformer + PPO，从原官方派生低 LR 权重独立起跑，128 环境 × 64 步，
固定 LR `1e-5`、entropy 0.001、seed 42、原八模式随机 mask，**1,000 次更新 / 8,192,000
环境步**。数据仍为 7,174 条训练片段，1,751 条开发验证不参与梯度更新。
这不是从头预训练、完整论文规模训练，也不自动更改 GUI 默认模型。

[长训主控](../scripts/run_bfm_long_training.py) 默认只打印计划，`--execute` 必须在对应
命名的 systemd user cgroup 中。1,000 次更新启动前，必须已有 **同 checkpoint、训练
索引、数据 payload 和 Python 源码指纹**的六次更新真实覆盖短测通过证明；长训不接着
短测权重训练。短测必须验证前五次采到同一批 128 条，第六次切到不重叠的第二批。

长训独立进程组设置：3 小时硬上限，训练子进程上限 9,000 秒，24 GiB 内存上限，
无自动重启，KillMode=control-group，停止时清理整组。按原 runner 的全局 iteration
每 50 保存一次，并保存结束 checkpoint；预计结束文件 `model_23198.pt`。
进度逐 update 原子写入 `progress.json`；最终还核对 actor/critic Adam 更新步数、
参数有限性、全 7,174 条真实覆盖和输入指纹。主控 COMPLETE 仅表示训练执行验收通过，
**不是策略质量达标**；独立八模式/旧数据/KIT 验证仍是后续门槛。

启动前 CPU 回归：**608 passed**（62 warnings），包含 sampler、runner 边界/恢复、
训练观察器生命周期和长训计划测试。

### 实际启动记录（2026-09-08，Asia/Shanghai）

短测 `behavior_coverage_smoke_20260908a` 已 COMPLETE，用时 43.83 秒：

- 6 次更新、49,152 实际环境步，256 条不同动作，其中 KIT 109 条。
- 前五次更新同一批 128 条，第六次新的 128 条；八种 mask 均有真实采样。
- actor 的 71 个、critic 的 70 个 Adam 状态均增加 384 steps，模型参数有限且有变化。
- `model_22204.pt` 已落盘，数据/源码/原始索引结束指纹复核通过。
- 短测 service 已回收，MainPID=0，启动长训前 GPU 无残留计算进程。

长训 run：`behavior_coverage_long_20260908a`，service：
`bfm-behavior_coverage_long_20260908a.service`。此处只记录启动，完成状态请查主控
`status.json`；`progress.json` 的 COMPLETE 还须等待主控 checkpoint/指纹复核，
不能替代主控的最终结论。

15:57:44 启动，15:58:28 实查仍 active/running：已完成 7/1,000 次更新、57,344 环境步，
实际采到 256 条（KIT 109 条），第二批轮换已进入长训，当前真实 actor/critic LR 均为
`1e-5`。首个周期 checkpoint `model_22200.pt` 已保存于新的 run（不是覆盖官方同名文件）。
主控 PID 258051、GPU 训练 PID 258077，显存约 3.1 GiB；均受该 cgroup 管理。
短测预计约 1 小时完成；硬截止为当日 18:57:44，未达完成条件会失败退出，不自动延长。

```bash
# 实时训练输出（Ctrl-C 仅退出查看，不停止训练）
tail -f /home/sw/bfm/logs/behavior_learning/behavior_coverage_long_20260908a/train.log

# 实际完成的更新和数据覆盖，不输出几千条动作明细
jq '{result, completed_updates, completed_environment_steps, unique_motion, unique_kit}' \
  /home/sw/bfm/logs/behavior_learning/behavior_coverage_long_20260908a/progress.json

# 主控最终验收状态
jq '{result, stage, coverage_verification, inputs_verified_unchanged, final_checkpoint}' \
  /home/sw/bfm/logs/behavior_learning/behavior_coverage_long_20260908a/status.json

# 停止本轮整个进程组；已保存 checkpoint 和日志保留
systemctl --user stop bfm-behavior_coverage_long_20260908a.service
```

训练结束后的下一步是同协议重跑官方基线与候选的八模式评估，并分别检查旧 962 条
和 KIT 789 条。此轮长训**没有自动启动这组质量评估，也不会自动推广 checkpoint**。
训练期间应保持已锁定的 Python 源码、数据与输入索引不变；继续改代码会使最终
来源一致性验收失败。Markdown 文档不在该 Python 源码指纹范围内。

### 长训完成与后续测评

该长训已于 2026-09-08 16:54:45 正常完成，主控 COMPLETE、train_audit PASS，用时
3,420.93 秒。1,000 次更新、819.2 万实际环境步；7,174 条训练动作全部参与采样，
其中 KIT 2,927 条，八种 mask 均有实际覆盖。全部 update 的 LR 为 `1e-5` 且诊断有限。
最终 `model_23198.pt`、双 Adam 更新增量 64,000、输入来源复核通过；保存 21 个
checkpoint，run 约 1.5 GiB。service 已退出回收，此后没有自动继续训练。

用户随后明确要求开始独立效果测评；见 [长训后八模式测评协议与状态](BEHAVIOR_LONG_EVALUATION_20260908.md)。
长训完成时还没有新模型质量结果；不得用上面训练执行结果替代该测评。

测评已于当日 19:17:34 完成，16/16 子任务执行和来源复核均通过，但质量为
`FAIL_NO_REGRESSION`：总体 0/8、旧数据/KIT 分层 0/16 通过门槛。
各模式激活连杆位置均值误差增加约 1.07%–6.15%，不推广该长训候选，官方默认不变。
结果已从逐片段原始报告独立复算；完整数值、解释限制及未启动的下一步诊断见上述链接。
