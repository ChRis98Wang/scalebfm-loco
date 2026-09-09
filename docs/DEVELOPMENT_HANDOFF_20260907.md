# 明日接续：三维目标点闭环

> 历史暂停记录。已于 2026-09-08 继续开发，修复回归并完成两轮真实测试（均 STALLED，
> 未通过目标任务）；最新结果见 [2026-09-08 验收](ONLINE_WAYPOINT_VALIDATION_20260908.md)。

用户要求今晚停止，明天继续。本轮开发代理已中断，测试进程已回收；
检查无 `bfm-*` 服务、BFM 播放/训练或本轮 pytest 进程。未启动本轮 GPU 验收。
没有重装环境、训练、覆盖 checkpoint、推送或操作真机。

## 最近已验收基线

Pelvis-1 / UMI-2 / VR-3 三种在线模式的 GUI 切换已实测；各约 5.02 秒 PhysX。
详情见 `ONLINE_SPARSE_MODES_VALIDATION_20260907.md`。
此前 294 项回归通过属于该基线，**不代表本轮三维 waypoint 开发也已全部通过**。

## 本轮用户确认的要求

目标必须是 **XYZ 三维位置（含骨盆高度）+ yaw 朝向**，不能只做 XY。
当前任务入口限定 Pelvis-1；Z 用于身体高度调整，不是飞行或任意台阶通行承诺。
以真实 XYZ、姿态、三维线速度和角速度判定到达与停稳。
仍保持 Pause/Close/心跳优先，不用 reset 或 state-write 模拟成功。
不用 superpowers 工作流；沿用现有 IsaacLab，按本机 API 核对。

## 已写入但未完成验收

- 新 `utils/waypoint.py` / `tests/test_waypoint.py`：纯 NumPy 三维限速/限前探参考，
  RUNNING、SETTLING、ARRIVED、TIMED_OUT、STALLED、CANCELLED 状态；代理中断前尚未交付最终测试报告。
- `online_ui.py` / `test_online_ui.py`：XYZ/yaw 输入、Go/Stop、共享递增序号、三维误差展示。
  UI 代理报告 12 项测试通过；真实 GUI 新布局未验收。
- `online_control.py` / `test_online_control.py`：任务接入、真实 body-link 速度读取、
  暂停/手动 Apply 取消任务、过期模式拒绝；只在 `after_step` 推进任务时间。
  内部参考提交留到下一 `before_step` 消费用户意图之后，避免内部序号超越排队的手动 Apply。
- `live_reference.py` / `test_live_reference.py`：新增无队列/时钟副作用的目标终点预检查，
  保留原空间和单位四元数要求。

运行中的聚焦测试已收尾：**64 passed、1 failed、59 warnings**。
失败：`test_preflight_validates_final_xyz_without_touching_pending_or_clocks`。
新测试把四元数乘 2，却期望通过；现有 Provider 明确拒绝偏离单位范数的输入。
明天先修测试假设或补拒绝断言，**不要放宽原四元数保护来迎合测试**。
当前 `git diff --check` 通过；未提交，保留整个已有脏工作区。

## 明日先检查，再继续

1. 核对 `waypoint.py` 的三维转换完整性和代理中断边界。已读代码中 `start` 的初始
   SETTLING 条件尚未包含单独 height_tolerance；应与 `step` 保持一致。
   初始速度尚无实测时不要用 0 冒充实测；终端/未启动 cancel 边界也需审查。
2. 修上述测试假设，运行纯核心、UI、controller、Provider 聚焦测试及原完整 CPU 回归。
3. 新 `ScaleTrack/tests/waypoint_gui_smoke.py` 已分派但因用户停止而中断，检查时尚无该文件。
   需要实现真实策略/PhysX 驱动：初始 Pelvis-1，站立 1 秒后向前 0.20 m 且降高 0.06 m，yaw 不变。
4. 默认门槛：三维距离 ≤0.05 m、高度差 ≤0.03 m、yaw 差 ≤5°、线速度 ≤0.05 m/s、
   角速度 ≤0.10 rad/s，连续实际物理时间 ≥0.5 秒；随后保持真实策略/物理 2 秒再检查。
   ARRIVED 不自动暂停物理。超时、停滞或失稳如实记失败，不能调宽门槛制造通过。
5. 单轮内部 165 秒 / 外层 180 秒独立 systemd 用户进程组；实际 PhysX 步事件累计 dt。
   记录每步 XYZ/参考/速度/误差、零 reset/state-write 计数、暂停前后历史/状态冻结、
   Kit 内部截图和退出清理。桌面锁屏时不解锁，使用 owned swapchain。

现有 interpreter：`/home/sw/isaaclab_ws/env_isaaclab_sim6_newton/bin/python`。
权重：`humanoid_transformer_m/model_22200.pt`；站立种子与命令参数沿用上一轮报告。
源码/API 已核对 `body_link_lin_vel_w` / `body_link_ang_vel_w` 返回 ProxyArray，
在 torch 边界展开为 `(num_envs, num_bodies, 3)`。不升级 SDK。

当前仍不是完整 ScaleBFM 复现，三维目标行走尚无真实通过证据。
