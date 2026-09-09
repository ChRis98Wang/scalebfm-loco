# 在线稀疏目标控制（Pelvis-1 / UMI-2 / VR-3）

本功能沿用现有 IsaacLab 和 Transformer + PPO 策略，新增原生 Kit 目标面板。
这是 BFM 控制层的增量开发，不是新训练结果，也不代表自主导航、夹箱搬运或完整论文复现。
当前支持一个 G1，在 Pelvis-1、UMI-2、VR-3 三种模式间在线切换。
离线动作切换保持原入口，不在在线会话中重置/换片段。

## 启动与操作

在本机桌面终端运行：

```bash
cd /home/sw/bfm
BFM_LOAD_RUN=humanoid_transformer_m \
BFM_CHECKPOINT=model_22200.pt \
BFM_INITIAL_MOTION=ACCAD/Male2General_c3d/A1-_Stand_stageii \
bash scripts/open_motion_browser.sh --online-targets
```

这里显式选择官方基线；普通动作浏览器默认的本地权重没有被替换。
需要既有本地 AMASS 索引和 checkpoint，不会自动下载安装或训练。
启动器创建有时限、不重启的用户服务；在线与箱子交互 UI 不能同时启用。

1. 等待 `BFM Online Targets` 面板显示 `READY`。机器人处于物理暂停状态。
2. 点击 `Enable`：以当前真实姿态建立参考，开始策略/物理步进。
3. 修改激活目标的 XYZ（米）与 XYZ 欧拉角（度），点击 `Apply` 一次性提交完整目标。
   第一轮建议只改动 0.005–0.02 米；工作空间上限不等于策略能可靠完成的范围。
4. `Pause` 停止物理步进，但 GUI 继续处理事件。`Resume` 重新以当前真实姿态建立参考，
   **仍暂停物理，直到重新编辑并点击一次有效的 `Apply`**。此前排队的旧目标被丢弃。
   `awaiting_goal=True` 表示这一等待状态。再次点击 `Pause` 会取消恢复授权。
5. `Exit` 或关闭目标面板的 X 退出。停止心跳后，X 仍可关闭。

面板上的三个模式按钮选择本次控制约束：

| 模式 | 激活目标 | 启动器可选参数 |
| --- | --- | --- |
| Pelvis-1 | 骨盆 | `BFM_MODE_INDEX=0` |
| UMI-2 | 左腕、右腕 | `BFM_MODE_INDEX=1` |
| VR-3（默认） | 骨盆、左腕、右腕 | `BFM_MODE_INDEX=2` |

运行中点击模式按钮会先暂停。等 `mode` 显示新名称后，点击 `Resume`，再提交新 `Apply`。
未激活的输入行禁用，坐标轴只显示激活目标。切换只重建参考和掩码，不重置机器人、
关节或观测历史；仍用同一个 Transformer checkpoint，不是三套新训练的模型。
首次 `Enable` 前也可选模式；这时保持 `READY`，选定后点击 `Enable`。
故障状态不能靠换模式解除，需先按既有恢复规则处理当前模式。

新增 `Waypoint` 区域可在 Pelvis-1 下输入三维 XYZ（包含骨盆高度）和 yaw。
这是单独的有界任务入口，Go 不会绕过 Enable 或 Resume 后的新 Apply。
使用、到达/停稳条件与限制见 [三维目标点控制](ONLINE_WAYPOINT_CONTROL.md)。

紧急关闭此启动器拥有的进程组：

```bash
systemctl --user stop bfm-motion-browser.service
```

不要使用面向所有 Python/Isaac 进程的 `pkill`。非有限遥测、跌倒/大倾角会锁住恢复，
应退出并重新初始化；不提供“复位后冒充连续控制成功”的恢复方式。

## 技术链路与安全边界

`Kit 完整目标包 → LiveReferenceProvider → LiveMotionCommand → 原 Transformer → 29 维动作 → 真实物理`

- 目标顺序固定为 pelvis、left_wrist_yaw_link、right_wrist_yaw_link，映射到 14 点中的 0/10/13。
- 目标包始终保留三行；当前模式决定策略输入掩码以及哪些行可编辑。输入包带模式名和
  单调增加的 `mode_epoch`，避免切回同名模式时误接受之前的旧目标。
- 输入位置为环境局部坐标，姿态内部统一为 `wxyz`；在当前 IsaacLab 的 `xyzw` 边界转换一次。
  可视化添加环境原点，真实遥测减去环境原点，不把参考位置当机器人真实位置。
- Provider 生成 33 帧限速未来参考。骨盆/腕位置速度上限分别为 0.2/0.35 m/s，
  角速度上限为 0.5/1 rad/s。骨盆水平范围限制在初始参考附近，竖直范围初始值 ±0.15 m，
  并与绝对高度 [0.35, 1.25] m 取交集；腕目标距骨盆上限为 1 m。
- 提交目标只刷新 task/mode 观测，不推进 proprioception/action 历史，不写机器人/箱子状态。
  最后一次初始化 reset 后才接入在线源；会话禁用自动终止重置及 interval 扰动。
- 心跳由 Kit 事件订阅独立产生，消费者不能伪造。超过 0.5 秒未更新进入 `PAUSED_FAULT`。
  收到新心跳本身不会自动恢复。
- 原生 Play 不能绕过用户暂停、等待新目标或故障锁。`sim.play/pause` 自身会处理 Kit 事件，
  控制器在这些调用后再次检查 Pause/Close，避免批准额外一步。
- 无效完整目标保留已接受参考并显示错误；等待恢复期间的无效 Apply 绝不能释放物理权限。
  程序错误继续抛出并清理资源，不伪装成普通输入拒绝。

## 如何解读界面

`goal_positions` 是用户要求，`reference_positions` 是限速后的参考，`actual_positions`
是 articulation 的真实遥测。界面 `position_errors` / `orientation_errors` 当前分别表示
实际姿态相对原始 goal 的位置误差（米）/旋转差（弧度），不应标成限速参考跟踪误差。
`active_position_errors` 和 `active_reference_position_errors` 只统计当前模式激活点，
分别相对原始 goal 和限速 reference；未激活点的位置差不作为该模式目标跟踪误差。
物理验收报告另存实际相对 reference 与 goal 的误差。

目标坐标轴是参考标记，不是人体骨架，更不是机器人已到达目标的证明。三点控制允许
其余身体点由策略协调，不代表所有动作、所有目标分布已经达标。

## 验收

自动回归覆盖真实时间轴状态的替身边界、原生按钮意图和 Provider/适配器；这些单元测试
不能代替策略稳定性。真实 GUI 验收使用 `ScaleTrack/tests/online_robot_gui_smoke.py`，
运行现有策略和 PhysX，不替换策略、不伪造物理、不改变保护阈值。
三模式连续切换另用 `ScaleTrack/tests/online_modes_gui_smoke.py`，与原 VR-3 安全回归分开记录。

物理时间来自原生 PhysX 步事件累计的 dt，并先确认 ACTIVE 中确实增长，再检查暂停冻结。
当前本机 SDK 的通用 physics clock 返回常量、Kit timeline 又不等于积分时间，不能用它们
作为“暂停成功”的证据。实际结果和限制见 [本次运行记录](ONLINE_VR3_VALIDATION_20260907.md)：
最终 UI 已通过内部截图检查，110 个不同接受目标、两次越界拒绝、10.54 秒真实物理、
暂停恢复、失联/原生 Play 防绕过，以及初始化异常清理已有实测。

后续 [三模式连续切换验收](ONLINE_SPARSE_MODES_VALIDATION_20260907.md) 已完成：
各模式 5.02 秒真实物理、各 251 个不同接受目标，位置 ±2 cm、yaw ±2°。
切换和 Resume 等待期间实际状态/历史保持冻结，六个生产源文件指纹及 GUI 截图已记录。
这是功能验收，仍不是长时间、大范围目标的能力保证。

没有新增训练、覆盖 checkpoint、改变 AMASS/KIT 原始数据或发布远端内容。
