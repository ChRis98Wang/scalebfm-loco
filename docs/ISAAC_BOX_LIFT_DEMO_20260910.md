# 双臂夹箱：抬起、保持、放回、释放已跑通

2026-09-10。**原生 IsaacLab / PhysX、1 kg 动态箱子，单初态完整抬放流程通过。**
这不是行走搬运，也不是正式 D1 的 60 次验收；使用官方预训练 BFM，不是本地新训练的搬箱网络。

## 直接观看

[GIF（无需 MP4 播放器）](media/isaac-lift-single-state-20260910.gif) ·
[字幕 MP4](media/isaac-lift-single-state-20260910.mp4) ·
[无字幕 MP4 原片](media/isaac-lift-single-state-20260910-raw.mp4) ·
[归档与哈希](media/isaac-lift-single-state-20260910.json)

[![双臂实际抬起箱子，原速完整 GIF](media/isaac-lift-single-state-20260910.gif)](media/isaac-lift-single-state-20260910.gif)

原速、完整 19.42 s，971 个物理控制步全部录像；MP4 960×720 / 50 fps，GIF 640×480 / 10 fps。
原片、字幕片、GIF、PNG 共 16,179,655 bytes，随本次 `main` 更新保存在 `docs/media/`。
只排除 reset-only 初始化画面，未剪掉接近、预压、放回或释放阶段。
归档 JSON 是先前本地保存时的原始凭据，其 `publication_performed=false` 不改写；
本次后续公开以 Git 提交记录为准。完整原始实验目录没有随媒体上传。

## 实测结果

| 项目 | 原有门槛 | 实测 |
|---|---|---|
| 箱子质量 | 1 kg | 1 kg，动态刚体 |
| 预压 | 两侧 ≥18 N，连续 0.20 s | 通过，10.52 s 进入抬升 |
| 真实离台 | 最低角点 ≥8 cm | 最高 **10.19 cm** |
| 保持 | ≥2 s；倾斜 ≤15°；台面无接触 | HOLD 2.02 s，最低间隙 **9.42 cm**，倾斜 **9.36–9.73°** |
| 连续有效离台窗口 | ≥8 cm 且夹持、倾斜与支持接触均合格 | **2.58 s**，包含 HOLD 前后连续片段 |
| 实际夹持力峰值 | 每侧每个物理子步 ≤40 N | 左 **33.44 N** / 右 **27.14 N** |
| 放回位置 | 箱心到目标 XYZ ≤3 cm；完整投影落在支持面内 | 误差 **1.78 cm**，完整投影通过 |
| 释放稳定 | 双臂接触 <0.2 N、台面承重 ≥0.8 mg、稳定 ≥1 s | 双侧最终 0 N，台面约 **9.81 N**，稳定 1 s 通过 |
| 释放速度 | 线速度 ≤0.05 m/s；角速度 ≤0.10 rad/s | **0.00032 m/s / 0.00057 rad/s** |
| 总时长 | <20 s，无跌倒/滑落保护触发 | **19.42 s** |

目标箱心为世界坐标 **(0.43, 0, 0.68) m**；最终实测为
**(0.422196, 0.016044, 0.680000) m**。高度来自实际箱体几何和支持台位置，不是规划器的目标高度。

状态事件：

```text
2.00 接近 → 5.00 闭合 → 8.08 预压 → 10.52 抬升
→ 12.70 保持 → 14.72 放回 → 18.40 松手 → 19.42 成功
```

## 网络与控制：哪些是学到的，哪些是本次开发的

**这个演示没有视觉指引。** 不输入 RGB、深度图或点云，也没有检测箱子或估计位姿的视觉网络。
录像中的相机只负责渲染，不参与动作决策。物体在哪里、是否接触台面，来自 IsaacLab 暴露的
**仿真真值**，而不是机器人从画面中看出来的结果。

```text
IsaacLab 箱体位姿 / 速度 / 接触力
  → 任务规划器生成骨盆、双腕 XYZ + 姿态目标（VR-3）
  → 官方 BFM Transformer + 机器人本体状态 / 动作历史
  → 关节动作 → PD 执行器 → PhysX 接触运动 → 下一步状态反馈
```

API 用于启动环境、添加动态箱子和接触传感器、读取状态、提交目标及 `env.step(actions)`。
**API 不是策略本身**；它没有替网络给机器人逐帧设置姿态，也没有在初始化后直接抬升箱子。
当前的能力是“仿真状态反馈下的规划器 + 学到的身体控制”，不是端到端视觉搬运。
若接实物视觉，需要另外实现相机标定、三维箱体位姿估计、坐标变换和延迟/噪声下的闭环验收；本次没有完成这些。

- 官方 Transformer `model_22200.pt` 负责将机器人状态和稀疏身体目标转换为关节动作。
  权重 SHA-256 为 `88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3`，本轮 **0 次训练更新**。
- 本次开发的任务规划器读取仿真箱体/接触测量，生成 pelvis、左腕、右腕的 **世界 XYZ 与姿态（VR-3）**。
  BFM 本身没有直接输入箱子状态；不是端到端学会了任意箱子搬运任务。
- 机器人通过原有 PD/执行器和 PhysX 接触实际抬箱。箱子不绑定手、不添加附着关节，
  不写后续箱体/机器人状态，不施加额外托举力，不用参考动画代替物理。
- 只使用机器人 FK 合成站立初态，没有人体动作文件作为该箱子试验的运动轨迹。
- 物理仍为现有 PhysX，机器人资产、质量、摩擦、接触和 PD/力矩限幅未更改；接触参数未做实物标定。

本轮有效改进分为三个部分：

1. **姿态与高度一起控制**：单独增加高度补偿会更早倾斜；加强腕部俯仰反馈后，完成了实际抬起和保持。
2. **落台后卸载**：测到真实台面接触后，才把夹持力控制设定值从 24 N 降到 1 N，
   增加卸载反馈增益，但保持原 6 cm/s 目标间距速度上限。物理支持、接触、40 N 峰值和超时门槛没有降低。
3. **回到放置点再松手**：LOWER 阶段把夹持 XY 基准平滑移回预先指定的目标，
   只有台面承重和速度稳定满足条件后才进入 RELEASE。

## 证据与局限

无录像开发试验 k 和完整录制复测 l 各执行 971 个控制步 / 3884 个 PhysX 子步。
**两次是相同初态、相同 seed 42 的确定性复测，不是两个独立泛化样本。**
两份完整轨迹 JSON 的 SHA-256 完全相同：

```text
0f76dcead663c45d3abc46b6c580b3ef935317e9aef279cbeb9fc719de124dcf
```

- `local/isaac_lift_balanced_20260910k/`：开发通过的完整状态、动作、接触、源码快照、配置和报告。
- `local/isaac_lift_balanced_20260910l/`：相同配置的录像复测及逐帧图像摘要。
- 两次独立验算均 `task_success=true`，完整序列、时间、机器人保护、预压、实际保持、
  载荷阶段倾斜/力峰值、滑移、接触覆盖、落台承重、释放 XYZ 稳定共 **10/10 检查通过**。
- 审计从箱体四元数独立旋转八个角点，复算最低点和完整投影；复算 31 个接触对 × 4 个物理子步的力，
  检查轨迹、源文件、策略/原片哈希及 50 Hz / 200 Hz 时钟。
- 机器人/箱子/支持台在每次纯渲染前后的状态和物理计数保持不变。
  按 IsaacLab API 技能核对了本机接触数组与录制接口，避免把渲染或目标状态当作物理测量。
- 原片和字幕片均全量解码 971 帧；GIF 全帧解码，时长 19.4 s（10 Hz 量化）。
  人工视觉检查为首帧、1/4、1/2、3/4、末帧五个样本，**不是逐帧人工验收**。
- 49 项控制逻辑测试 + 8 项独立验收逻辑测试通过；单元测试本身不算物理任务成功。

本轮失败同样保留：h 只加高度反馈，12.04 s 倾斜超限；i 完成保持但 17.34 s 落台夹持力峰值超限；
j 以 6 N 卸载，18.74 s 尚未满足稳定台面承重窗口。不能将调参后的两次通过当成无偏成功率。
更早失败与完整源记录保留在原开发机，不在这个媒体/源码发布包中。

本页对应的 [原生入口](../scripts/run_bfm_isaac_lift_balanced.py)、
[接触场景](../ScaleTrack/scripts/pretrain/rsl_rl/lift_demo.py)、
[规划器](../ScaleTrack/scripts/pretrain/rsl_rl/lift_demo_balanced.py)、
[独立验算](../scripts/audit_bfm_isaac_lift.py) 和
[成功配置](../configs/isaac_lift_balance_unload_20260910.json) 已包含在此源码快照。

## 开发机复测命令与公开包边界

需要本机现有环境和绑定的官方权重/合成初态；不自动下载安装依赖或数据。
**以下是原开发机的复测记录，不是新 clone 后即可运行的一键命令。**
入口绑定官方权重、机器人资产、已有 FK 初态与元数据，缺少或哈希不符会拒绝启动。
公开包不含这些运行输入；先阅读 [源码范围与运行资产说明](SOURCE_SNAPSHOT.md)。
每次使用新的单元名和 `local/` 输出目录；GPU 被其他计算任务占用时会拒绝启动。

```bash
cd /home/sw/bfm
systemd-run --user --unit=bfm-isaac-lift-demo-manual-001 \
  --property=KillMode=control-group --property=Restart=no \
  --property=RuntimeMaxSec=600 --property=TimeoutStopSec=20 \
  --property=MemoryMax=16G --property=TasksMax=256 \
  --property=WorkingDirectory=/home/sw/bfm \
  --setenv=OMNI_KIT_ACCEPT_EULA=YES --setenv=OMP_NUM_THREADS=1 \
  --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
  /home/sw/isaaclab_ws/env_isaaclab_sim6_newton/bin/python -u -B \
  /home/sw/bfm/scripts/run_bfm_isaac_lift_balanced.py \
  --balance-config /home/sw/bfm/configs/isaac_lift_balance_unload_20260910.json \
  --output /home/sw/bfm/local/isaac_lift_demo_manual_001 \
  --headless --device cuda:0 --record --execute

journalctl --user -u bfm-isaac-lift-demo-manual-001.service -f
```

测试服务有限时、无自动重启；成功退出码 0，任务失败退出码 2，失败轨迹不覆盖。
本轮所有 BFM 仿真/审计/编码任务结束后均检查 MainPID=0、ControlGroup 为空；没有停止其他项目。

不启动仿真、不使用数据或权重的 57 项逻辑测试，可以在已有 NumPy 的 Python 环境中执行：

```bash
python -B -m unittest discover -s ScaleTrack/tests -p 'test_lift_demo*.py'
python -B -m unittest discover -s tests -p 'test_bfm_lift_acceptance.py'
```

这些测试与实际物理验算分开；GitHub 上配置的 CPU 流水线仍是原先的显式测试子集，
此处不宣称远端 CI 或所有 GPU 集成测试已经通过。

## 下一阶段

先完成正式 D1 的 20 组初态 × 3 seed（60 次）验收，再接走近、携带至少 1 m、放入三维目标区域的 D2。
目前**没有验证行走携带、地面深蹲拾箱、不同尺寸/质量箱子或真实硬件**。
这段视频证明当前条件下的双臂物理抬放可行，不代表完整 ScaleBFM 或所有 loco-manipulation 功能完成。
