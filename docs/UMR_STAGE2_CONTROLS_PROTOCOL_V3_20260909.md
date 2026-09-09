# UMR v3：输出帧间限速与显式腕姿态

执行前固定设计；继承 v2 的同四个 train 短窗，不修改历史数据和门槛。

## 两因素对照

保留 **原 v1 flat canonical**，不将上一轮效果不一致的 matched-hand 变体混入本实验。
使用已冻结 v2 `case_0..3/control/setup` 的 Stage I bodies/correspondence，
parent status SHA `7416751c1b83c20d46f4d375123e0cc311f417b8011fbe3f0b5d6b755d0b3f0b`。
按来源固定 ACCAD A2 Sway、BMLmovi Subject22 F9、BMLrub normal_jog4、KIT walking_run06，
每条运行 `control / rate_only / wrist_only / both`，共 16 个 Stage II job。
不重新学习 Stage I，不训练 BFM PPO，不换 seed、不选最优片段、不跳过失败。

四臂共享原始 prepared source、body model/shape/scale、material samples、correspondence、
point/normal/contact/posture 权重、关节位置限制、floor/self-collision/trust-region 设置。
control 必须与原 control 输出 qpos 以绝对误差不超过 1e-9 复现；不满足则停止继续作因素归因。
Stage II 使用现有 UMR Python、50 Hz、每帧 6 次迭代、首帧 60 次 warmup；只运行 CPU IK/FK。

## 限速因素

仅限制全部 29 个 hinge joint（不声称 free root 的线/角速度受限）。
每个已输出帧的预算为 `vmax=12 rad/s × 0.02s = 0.24 rad`。
设上一个实际输出为 `q_prev`，当前内部迭代状态为 `q_cur`，QP 变量为 `delta_q`：

`G = [P; -P]`

`h = [q_prev + 0.24 - q_cur; q_cur - q_prev + 0.24]`

`P` 按模型的真实 named hinge qpos/dof 地址选取，不能混淆 nq 与 nv。
同一输出帧的所有 IK 迭代共用固定 q_prev，不重新发放预算；输出后才推进。
warmup 和第 0 个输出帧尚无前一个输出，不施加帧间限制；从 interval 0→1 起全部统计。
每帧输出及完整 motion 二次验证，不后剪切/缩放或平滑关节，不静默吞掉 QP 失败。
固定数值容差为 2e-6 rad，相当于 1e-4 rad/s；同时记录严格 >12 与 >12.0001 的计数。

## 腕姿态因素

新增两个 Mink FrameTask：`left/right_wrist_yaw_link`，frame_type=`body`，
position_cost=0、orientation_cost=10、gain=1、lm_damping=1。
目标为 prepared world joint_rotations 的 human 20/21 右乘历史 GMR offset：
左单位，右局部 Z 180°（wxyz `[0,0,0,-1]`）。不再次乘 heading，不用 COM frame。
target translation 设零且 cost 为零，不引入新的 wrist position 约束。
这是 **软目标**，不是零残差保证；保留 UMR 所有原表面和接触项，不宣称上游官方 UMR 已有它。

## 报告和执行边界

保存每个 job 的真实 warmup/每帧失败计数、实际任务/限速调用证据、motion NPZ/兼容 PKL，
world 腕/骨盆目标差、表面 point/normal 残差、帧间速度、关节位置限位/穿地/自碰撞。
工程完成不等于任务成功：即使限速满足，也必须披露姿态/位置和碰撞退步。
只报告本四条短窗，没有策略跟踪、物体操作、实物接触标定或全库提升结论。

默认只读计划；--execute/worker 均要求 `bfm-umr-stage2-<run-id>.service`，
KillMode=control-group、Restart=no、RuntimeMaxSec/MemoryMax/TasksMax 有限。
单 worker 超时 180 秒，独立进程组最终 TERM/KILL/reap；输出只写新 `local/umr_stage2_v3_<run-id>/`。
所有历史输入/输出、代码、机器人资源和所用数值依赖在 controller 前后核验 SHA；
worker 复核自身脚本、数值依赖、该条 source/setup 和启动清单。没有受许可数据上传授权。

新方法不自动替换训练数据或默认权重。v1/v2 及本协议执行后保持不变，结果另写技术报告。
