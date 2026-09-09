# UMR v2：canonical 手形一致性机制试验

状态：执行前固定设计，事后提出的机制实验，不修改 v1 数据或质量门槛。

## 动机和边界

v1 的 17 train 和 10 development 原 source `pose_hand` 均为同一个、全时段恒定的
SMPL-X mean-hand 向量。adapter 的 `flat_hand_mean=True` 没有重复加 mean；但 canonical
rest 未传 hand pose，使用平手，moving surface 却使用该恒定 mean hand。G1 没有独立手指，
可能通过手腕补偿表面/法线差异。这是可检验机制，尚非已证明的错误或性能提升。

旧 GMR 使用 wrist 完整位姿目标（位置/姿态 cost 10/10），UMR 则拟合材料表面点和法线。
旧 wrist target 的残差仅衡量目标兼容性，不能作为人体动作真实度的唯一判据。

## 独立诊断

`analyze_umr_wrist_targets_v2.py` 对完整 17 train + 10 development 重建旧目标旋转残差。
使用已绑定 prepared source 的 joint rotations、pair clock 和已验证 FK 的 packed link quaternion，
不重新推断人体、不拟合常量 offset。特别保留 GMR pelvis 非单位旋转 offset，避免错误相对坐标系。
不据该结果重新挑选下面的四条动作，不更改 v1 comparator。

## 固定四条机制样本

从冻结 v1 manifest 的 17 `target_origins`，按 dataset 分组，各选字典序第一条。
没有 CNRS train 候选，因此不得声称五来源覆盖。列表为：

1. `ACCAD/Female1General_c3d/A2_-_Sway_stageii`
2. `BMLmovi/Subject_22_F_MoSh/Subject_22_F_9_stageii`
3. `BMLrub/rub058/0012_normal_jog4_stageii`
4. `KIT/205/walking_run06_stageii`

这些是已被几何选择的 train 短窗，不代表 7,174 条全库、完整长动作或新 test。

## 单变量设置

- control：原 v1 prepared source，重新从头训练 Stage I 并运行 Stage II。
- matched_hand：仅 canonical rest 加上原 source 恒定 hand pose；保持原材料 face/barycentric
  和所有 moving arrays、时间、body pose、wrist pose、betas、地面规则完全不变。
- 重算 canonical 点/法线和 joint frames；先重建原 flat canonical 核验，拒绝依赖或源漂移。
- 不重采样材料点、不修改 hand 动作、不改关节约束/目标权重/posture cost/tpose_offset。
- 两侧均 fresh setup，不写/复用 v1 setup cache；固定 4,096 点、Stage I 2,500 epochs、seed 0、
  Stage II 512 selected points、6 iterations、50 Hz、CUDA，使用同一个 pinned UMR commit。
- 如果 canonical actor height 改变超过 1e-6 m，停止此单变量 pilot，避免静默改变 moving scale。
- native inclusive 全窗口比较，两侧帧数和 frame_indices 必须一致；这里没有打包/PPO。
- 另外报告 fresh control 与原 UMR 产物的 qpos 差，暴露重新学习 Stage I 的重现误差。

## 结果定义

工程完成需要 4/4 原始输入绑定、4/4 hand 变体、8/8 retarget 正常返回且 receipt/NPZ 匹配、
输出有限、相同 frame clock、SHA 前后不变，以及服务退出。solve_failures 单列，不能仅凭退出码通过。

逐 origin、逐 arm 报告：

- 历史 GMR world wrist/pelvis orientation 目标残差（最短 SO(3) 角，mean/p95/max）；
- 两侧 wrist FK 位姿迁移、原 control 重现差；
- 原几何 FK 碰撞检测的脚底、全身穿地、自碰撞、关节限位；
- native 相邻帧关节速度统计；配置 max_velocity=12 尚未接入 solver 的事实不能掩盖。

不定义自动晋升门槛，不声称几何改进等于学会行为；不按单条结果跳过失败样本。
结果用于决定是否值得扩展到完整配对 pilot，再重新冻结学习协议。失败也保留全部日志。

## 进程、产物和冻结

controller 默认只读计划，`--execute` 要求有界 `bfm-umr-hand-*.service`；KillMode=control-group、
Restart=no、RuntimeMaxSec/MemoryMax/TasksMax 有限。子进程独立会话、有超时并回收。
新文件仅写 `local/umr_hand_v2_<run-id>/`，拒绝已有路径/符号链接，不触碰原 data/index/checkpoint。
执行前 controller 将本协议、实现、依赖/机器人资源、全部 v1 manifest 输入和 raw/model SHA 绑定；
完成前重核验。当前协议在实际运行之前冻结；后续技术解释和结果另写报告。

AMASS/SMPL-X 和本地派生 motion 不因实验而取得再分发许可，不上传受许可数据。
