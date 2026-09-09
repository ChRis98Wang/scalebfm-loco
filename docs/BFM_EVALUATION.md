# BFM：从动作跟踪到任务控制的下一步

## 当前阶段

我们已跑通官方预训练 Transformer 的本地 PPO 微调和参考动作跟踪。
下一步首先比较官方起点与本地最终权重，确认微调的效果；之后才验证稀疏目标控制和
更具体的任务。不把动作菜单、训练 reward 或动作名称等同于任务成功。

下述历史配对轮未增加训练量、未改变训练/验证划分，也不依赖新增 CMU。
用户现已确认继续 ScaleBFM 路线；逐连杆 mask 的 schema 4 验收与后续小步优化口径见
[ScaleBFM 连杆掩码验收](SCALEBFM_MASK_VALIDATION.md)。

2026-09-07 配对评估已完成：本地最终权重在 WholeBody-14 下的平均位置误差较官方起点
增加 20.10%，需要先复查训练效果，再推进任务控制。见 [结果与失败名单](BFM_EVALUATION_20260907.md)。

## 独立评估入口

在仓库根目录，使用已有 IsaacLab Python。只加载**可信来源**的 checkpoint：

```bash
"$BFM_ISAAC_PYTHON" ScaleTrack/scripts/pretrain/rsl_rl/evaluate.py \
  --checkpoint_path logs/rsl_rl/g1_bfm_tracking_exp/humanoid_transformer_m/model_22200.pt \
  --motion_file ScaleRetarget/retargeted_dataset/amass_full_v1_validation.yaml \
  --output logs/evaluations/official_wholebody_seed42.json \
  --num_envs 128 --max_steps 1000 --seed 42 --mode_index 7 --device cuda:0
```

第二次使用同样设置，只替换 checkpoint 和新输出路径：

```bash
"$BFM_ISAAC_PYTHON" ScaleTrack/scripts/pretrain/rsl_rl/evaluate.py \
  --checkpoint_path logs/rsl_rl/g1_bfm_tracking_exp/amass_full_v1_finetune_20260906/model_23197.pt \
  --motion_file ScaleRetarget/retargeted_dataset/amass_full_v1_validation.yaml \
  --output logs/evaluations/finetuned_wholebody_seed42.json \
  --num_envs 128 --max_steps 1000 --seed 42 --mode_index 7 --device cuda:0
```

两次顺序运行，避免抢占同一 GPU。入口始终无 GUI，不调用 `learn`、不加载优化器状态、
不保存模型，也不使用会改变采样权重的训练内评估函数。已有 JSON 输出会被拒绝覆盖。
实际代理运行使用有时限的临时服务，结束时清理整个控制组。

## 协议与指标

- 数据：962 条本次本地微调未使用的验证片段；与官方预训练数据的重叠未知，因此不能
  据此声称它们对官方模型也是全新动作。
- WholeBody-14：完整 14 个身体点参考；相同种子、128 环境、相同任务默认启动随机化。
- 确定性均值动作；关闭观测噪声、重置扰动和周期推力。本轮不评价抗扰动能力。
- 每条 N 帧动作统计 `min(N - 1, 1000)` 个真实推进步，不统计动作结束自动重置后的样本。
  50 Hz 下最长 20 秒；长片段标记 `truncated`，不是整段评估。
- 沿用 command manager 的“物理步之后、参考帧推进之前”误差对齐方式。先按每条动作的
  有效样本数平均，再在动作之间等权平均；同时提供 ACCAD/BMLmovi/BMLrub/CNRS 分组。
- `error_body_pos_g`：14 个身体点在世界坐标中的平均位置距离，单位米。
- `error_body_pos`：去掉根位置后的身体点位置误差，单位米。
- 0.1 / 0.2 / 0.5 米标准：一条动作在评估窗口内，**每时刻 14 点平均误差的最大值**是否
  超过阈值；不是每个身体点的最大误差，也不是抓取成功率。0.5 米标准较宽松。
- JSON 包含每条动作的均值、最大值、有效步数和截断标记，以及数据集汇总和失败名单。
  NaN/Inf 或有效步内的意外重置会中止评估，不能产生正常完成报告。
- v2 报告在启动模拟器前保存 checkpoint/index、全部被引用 NPZ 和实际 Python 源码的
  SHA-256；结束后重新校验，不一致则拒绝写入正常完成报告。源码范围包含评估入口、
  ScaleTrack、定制 RSL-RL 和当前解释器实际使用的 IsaacLab/rl/tasks，覆盖未提交文件。
  这不是完整环境锁定：模拟器二进制、机器人资产和驱动不在该内容清单内。
- 记录种子、模式、步长、Python/Torch/CUDA、GPU 和相关包版本；相同种子仅配对启动状态，
  不能保证 GPU 物理计算逐位相同。原始日志和报告应随实验保留。
- 新入口输出 schema 3，沿用上述有效步计数与内容校验；`scene_variant` 明确区分
  无物体基线与可选目标物体场景，并记录生效物理参数和物体诊断指标。旧 v2 报告保持
  不变。带物体短程检查不是新的训练质量结论，见 [目标物体验证说明](BFM_TARGET_OBJECT_VALIDATION.md)。

路径说明：沿用上游加载器行为，索引中相对 NPZ 路径以**启动命令的工作目录**为基准，
不是以 YAML 所在目录为基准；本地验证索引使用绝对路径。

旧训练内评估跳过了重置样本但仍用原帧数作除数；本轮使用实际统计步数，
**不要把两套协议的数值直接当成训练前后的变化**。应比较本轮生成的两份报告。

单个种子的对比只是一轮基线证据，不等于收敛、统计显著改善或完整论文复现。

## 下一阶段的门槛

1. 比较两份报告的整体误差、分组误差和失败集合，选择可信的基线权重。
2. 在相同参考片段上验证已有的 Pelvis-1（索引 0）、VR-3（索引 2：骨盆和双腕）等模式，
   检查较少目标输入时的全身协调。2026-09-07 已通过真实播放中的掩码切换与重置烟测，
   但尚未完成这些稀疏模式的完整定量质量验收。
3. 再设计明确的任务接口，例如目标点行走或手腕目标跟踪；离线参考片段与在线目标
   的分布可能不同，必要时需任务适配和单独的成功标准。
4. 抓取、搬运等物体交互需要物体、接触和任务目标环境，不能只靠动作名证明完成。
