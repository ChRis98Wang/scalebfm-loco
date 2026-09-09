# AMASS → ScaleBFM 技术路线与实现说明

本文说明当前仓库如何把 AMASS 人体动作转换为 Unitree G1 可训练的
ScaleBFM/ScaleTrack 数据，并给出训练、回放、验收和继续开发语义任务的路线。
本文针对本机 `/home/sw/bfm` 的实际环境；不需要重装 IsaacLab。

2026-09-05 已完成本地四包共 5,256 条动作的重定向、50 Hz 打包和全量审计，
并提供按源文件哈希去重的 5,253 条素材索引。CMU 下载仍未完成。
数据清单、索引、验收证据和恢复边界见 [AMASS 收集与重定向进度](AMASS_COLLECTION_STATUS.md)，
下一批选择见 [补充数据建议](AMASS_DATA_RECOMMENDATIONS.md)。

2026-09-05 15:37 已启动 128 条训练动作 / 32 条验证动作的首轮多动作微调，沿用
128 个并行环境和官方 M 权重；新一轮配置、状态和停止命令见
[小规模训练记录](AMASS_SMALL_TRAINING_RUN.md)。

## 1. 先明确：当前开源代码能做什么

当前开源的 ScaleTrack 是一个通用**动作跟踪策略**预训练环境。给定未来若干帧的
身体目标，它输出 G1 的 29 个关节位置动作，使机器人在物理仿真中跟随参考动作。

因此，现阶段可直接训练和回放的是：

- 站立、走、跑、转身、舞蹈、上下肢协调等 AMASS 中存在的无物体动作；
- 只指定骨盆、双手、手脚、上半身或全身的稀疏/稠密身体目标；
- 全局轨迹跟踪，以及带骨盆模式下的本地相对跟踪；
- 在摩擦、质心、手部负载、关节零位和外力扰动下保持动作跟踪。

AMASS 本身没有任务物体状态、接触目标或“是否完成取放”的标签，所以仅把 AMASS
重定向并训练，不会自动得到开门、抓取或搬运任务。此类语义任务需要在 ScaleTrack
基础上增加场景物体、任务观测、奖励/终止条件和相应交互数据；第 11 节给出了开发路线。

## 2. 整体数据流

```text
AMASS Stage-II / 公开兼容样本
  root_orient + pose_body + trans + betas + fps
                    │
                    ▼
SMPL-X 推理 + 机器人形体匹配（6000 次 Adam）
                    │
                    ▼
GMR / Mink 逐帧 IK：人体关键点 → G1 29-DoF
                    │
                    ▼
ScaleRetarget .pkl（30 Hz）
  root_pos + root_rot(xyzw) + dof_pos + fps
                    │
                    ▼
IsaacLab 重采样 + G1 正向运动学（50 Hz）
                    │
                    ▼
ScaleTrack .npz + motions.yaml
  29 关节状态 + 30 刚体位姿/速度；v3 归档；磁盘四元数固定为 wxyz
                    │
                    ▼
IsaacLab ManagerBasedRLEnv + Transformer Actor/Critic + PPO
                    │
                    ▼
checkpoint → GUI/视频回放 → 导出 → ScaleBridge Sim2Sim/Sim2Real
```

仓库入口脚本为 [`scripts/amass_to_scalebfm.py`](../scripts/amass_to_scalebfm.py)。
它把上述数据阶段串起来，并把 ScaleRetarget 与 IsaacLab 隔离在各自 Python 解释器中。

## 3. 本机环境与边界

当前已验证的解释器：

| 用途 | 路径 | 版本/说明 |
| --- | --- | --- |
| ScaleRetarget | `/home/sw/.cache/bfm/scaleretarget-py311/bin/python` | Python 3.11.15 |
| ScaleTrack/IsaacLab | `/home/sw/isaaclab_ws/env_isaaclab_sim6_newton/bin/python` | Python 3.12.3；已有环境 |
| 实际 IsaacLab 源码 | `/home/sw/isaaclab_ws/IsaacLab-3.0-sim6-newton` | 当前提交 `ffff603` |
| SMPL-X neutral | `/home/sw/shuaiwang/.cache/ula_smplx/SMPLX_NEUTRAL_2020.npz` | 已存在 |

不要把两个 Python 环境混用。ScaleRetarget 依赖 SMPL-X、MuJoCo、Mink 和 Hydra；
ScaleTrack 需要能启动当前 Isaac Sim/IsaacLab 的解释器。入口脚本使用 subprocess 形成
清晰边界，因此不要求把两套依赖安装进同一个环境。非 dry-run 启动前，入口会分别用
指定解释器导入所需依赖；缺少模块时会在下载数据或启动昂贵计算之前报错。即使是
dry-run，也会拒绝不存在或没有执行权限的解释器。两套环境的 Python 版本、解释器、
关键模块版本、导入路径和模块文件内容会汇总成环境指纹；本地 editable 的
`isaaclab`、`isaaclab_physx` 和 `scaletrack` 还会递归哈希 Python 实现树，内部 FK 或
物理接口代码发生变化也会使缓存失效。

## 4. 各阶段数据契约

### 4.1 AMASS 输入

原生 AMASS Stage-II `.npz` 至少应包含：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `root_orient` | `(N, 3)` | 根部 axis-angle |
| `pose_body` | `(N, 63)` | 21 个身体关节 axis-angle |
| `trans` | `(N, 3)` | 人体全局平移，单位 m |
| `betas` | 通常 `(16,)` | SMPL-X 形体参数 |
| `gender` | 标量字符串 | 性别元数据 |
| `mocap_frame_rate` | 标量 | 原始采样率 |

入口脚本会递归查找 `.npz`，跳过 `_stagei.npz`，保留目录层级，并在昂贵计算前检查
形状、帧数、帧率以及 NaN/Inf。固定版本的公开 AMASS 小样使用较旧的 `poses` 和
`mocap_framerate` 字段，脚本会把其身体部分切成上述 Stage-II 兼容字段。

完整 AMASS 数据需要在 [AMASS 官网](https://amass.is.tue.mpg.de/) 注册并遵守各子数据集
许可。`--public-sample` 只下载带固定 commit 和 SHA-256 的小样，用于端到端冒烟测试，
不能替代完整训练集。

### 4.2 ScaleRetarget 输出

AMASS loader 首先用 SMPL-X 得到人体关节位置和方向，并重采样到 30 Hz。默认配置会：

1. 用匹配的人体/G1 关键点，在静止姿态上优化 16 维 SMPL-X `betas`，默认 6000 次；
2. 用优化后的形体减少人体与 G1 比例差异；
3. 通过 GMR 的 Mink 约束 IK，逐帧拟合 G1 根位姿和关节角；
4. 使用 `kinematic` formatter 做地面高度调整。

每个结果是一个 `joblib` `.pkl`：

| 字段 | 形状 | 约定 |
| --- | --- | --- |
| `root_pos` | `(N, 3)` | G1 根部全局位置 |
| `root_rot` | `(N, 4)` | **xyzw** |
| `dof_pos` | `(N, 29)` | 29 个关节角，rad |
| `fps` | 标量 | 当前默认约 30 Hz |

入口同时为每个 `.pkl` 写入两个 sidecar：`.pkl.source.sha256` 绑定对应 Stage-II
输入内容，`.pkl.pipeline.sha256` 绑定 SMPL-X 模型、重定向代码、配置、G1 MJCF 和
Python 环境。后续即使文件修改时间被恢复，或模型/代码发生变化，也不会静默复用旧
结果。没有 sidecar 的历史 `.pkl` 默认拒绝接管；只有显式传入
`--adopt-legacy-retarget` 才会标记为 `[ADOPT-UNVERIFIED]` 并绑定当前输入和流水线。

重定向结果通过同目录临时文件、`fsync` 和原子替换发布；串行与多进程路径都会把
子任务失败传播到入口。失败时不会留下半写 `.pkl`，也不会破坏上一份有效结果。
准备阶段还会原子生成 `.bfm_motion_manifest`，ScaleRetarget 只读取本次精确文件清单，
不会误扫受管目录中的历史孤儿文件。部分续跑时，已有 `.pkl` 会先逐个检查结构、时间和
双 sidecar；合法 skip 计入完成数，而 provenance 缺失/不匹配会在处理缺失项之前报错。
入口只为本次确实新生成或强制刷新的结果写 provenance，不会给被 loader 跳过的旧文件
“补认证”。

### 4.3 ScaleTrack 打包输出

打包器把 `.pkl` 插值到 50 Hz，在 IsaacLab 中写入 G1 根状态和关节状态，再读取完整
正向运动学结果。每个 `.npz` 包含：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `joint_pos` | `(T, 29)` | 关节位置 |
| `joint_vel` | `(T, 29)` | 关节速度 |
| `body_pos_w` | `(T, 30, 3)` | 30 个刚体的世界位置 |
| `body_quat_w` | `(T, 30, 4)` | 世界方向；**磁盘固定 wxyz** |
| `body_lin_vel_w` | `(T, 30, 3)` | 世界线速度 |
| `body_ang_vel_w` | `(T, 30, 3)` | 世界角速度 |
| `fps` | 标量 | 必须为 50，供当前训练加载器使用 |
| `format_version` | 标量 | 当前为 `3` |
| `quaternion_order` | 标量字符串 | 当前必须为 `wxyz` |
| `source_sha256` | 标量字符串 | 对应 `.pkl` 的内容指纹 |
| `pipeline_fingerprint` | 64 位十六进制字符串 | 打包代码、G1 资产和 IsaacLab 环境指纹 |
| `reference_root_pos` | `(T, 3)` | 从源 `.pkl` 独立重采样的根位置 |
| `reference_root_quat_w` | `(T, 4)` | 从源 `.pkl` 独立转换的根方向，`wxyz` |

打包器只把当前 simulator batch 载入内存，`--package-num-envs` 控制每批动作数，
`--package-loader-workers` 控制该批的 CPU 加载进程数，不再用 `n_jobs=-1` 一次读取
整个动作库。输入的相对目录层级会保留到输出，避免不同子数据集中同名动作相互覆盖。
每个 `.npz` 先写到同目录随机临时文件、`fsync`，再原子发布；失败不会破坏旧归档。
v3 校验还会逐帧比较 FK 得到的 pelvis 与独立参考根轨迹，因而能发现“数组形状和模长
都正常、但四元数顺序导致 FK 整体错误”的回归。

YAML 文件是稳定的 `motion_name: absolute_npz_path` 索引。入口向打包器和 YAML 生成器
传递本次预期文件的精确清单，目录里遗留的孤儿文件不会混入训练集。YAML 同样采用
原子发布。训练时会加载索引中的动作、拼接数组，并用每条动作的长度与偏移量采样。

## 5. 四元数兼容问题及本次修复

这是本机 AMASS 训练此前每一步都立即终止的根因。

项目涉及三种边界：

| 边界 | 四元数顺序 |
| --- | --- |
| ScaleRetarget `.pkl` | `xyzw` |
| ScaleTrack `.npz` 磁盘协议 | `wxyz` |
| IsaacLab 运行时 | 可能为 `wxyz` 或 `xyzw`，必须实际检测 |

官方固定版本附近的 IsaacLab 数学 API 使用 `wxyz`，而本机实际导入的
`IsaacLab-3.0-sim6-newton` 使用 `xyzw`。旧打包器无条件把 `xyzw` 改成 `wxyz` 后写入
仿真器，于是本机把 `[w,x,y,z]` 当成 `[x,y,z,w]`。随后得到的 `body_pos_w` 和速度属于
错误根方向。训练加载器即使重新排列了保存的四元数，也无法修复已经由错误正向运动学
算出的刚体位置，最终出现约 0.55 m 的全身位置误差和 100% `body_pos` 终止。

当前修复采用显式边界转换：

- [`quaternion_compat.py`](../ScaleTrack/source/scaletrack/scaletrack/utils/quaternion_compat.py)
  通过 `quat_from_euler_xyz(0,0,0)` 的返回值检测运行时顺序，不依赖版本字符串；
- 打包前执行 `ScaleRetarget xyzw → runtime`；
- `quat_slerp` 始终使用 runtime 顺序；
- 旧的角速度辅助函数明确使用 `wxyz`，只在进入该函数前转换；
- 保存前执行 `runtime → packed wxyz`；
- 训练加载时执行 `packed wxyz → runtime`；无版本元数据的旧官方动作仍按历史 `wxyz`
  协议兼容，v2/v3 文件若声明其他顺序或元数据不完整则在训练前拒绝加载。

修复后对全部 208 帧的数值校验为：

- 重定向根位置与打包 pelvis 位置最大绝对误差：`5.96e-8 m`；
- 预期磁盘根四元数与打包 pelvis 四元数最大符号不变 L2 误差：`2.17e-7`；
- 全文件四元数模长最大误差：约 `3.6e-7`。

1 个 PPO 迭代的对照探针结果：

| 指标 | 错误打包数据 | 修复后数据 |
| --- | ---: | ---: |
| `error_body_pos_g` | `0.5485 m` | `0.1559 m` |
| `Episode_Termination/body_pos` | `1.0000` | `0.1196` |
| 平均 episode 长度 | `1.00` | `29.60` |

这些指标说明数据错位问题已经闭环修复；它们不是最终策略收敛指标。

## 6. ScaleBFM 训练任务的技术解释

### 6.1 仿真与动作

- 物理步长为 `0.005 s`（200 Hz）；
- `decimation=4`，策略输出频率为 50 Hz；
- 动作为 29 维关节位置目标，乘以 G1 每关节的 action scale；
- episode 上限为 10 秒，也会在参考动作结束时终止；
- 任一跟踪刚体与目标的全局位置误差超过 `0.5 m` 时提前终止。

### 6.2 八种条件模式

训练 reset 时随机选择一种模式，仅激活对应身体链接。模式编号与回放一致：

| 编号 | 模式 | 激活链接 |
| ---: | --- | --- |
| 0 | Pelvis / Root | 骨盆 |
| 1 | Bimanual | 左、右手 |
| 2 | Root-and-Hand | 骨盆 + 左、右手 |
| 3 | End-Effector | 左、右手 + 左、右脚 |
| 4 | Root-and-End-Effector | 骨盆 + 双手 + 双脚 |
| 5 | Upper-Body | 双肩、双肘、双手 |
| 6 | Root-and-Upper-Body | 骨盆 + 双肩、双肘、双手 |
| 7 | Whole-Body | 骨盆、腿、躯干、肩、肘、手共 14 个链接 |

模式映射会屏蔽未激活链接的 actor 任务特征，并把 14 维链接 mask 一并送入策略。
固定模式回放会复用训练时的 mask 路径。`--local-tracking` 必须与包含骨盆的
0、2、4、6 或 7 模式一起使用。

### 6.3 观测

Actor 本体感知保留最近 3 帧，每帧 64 维：投影重力 3、基座角速度 3、关节位置 29、
关节速度 29。它还接收最近 3 帧动作，以及未来偏移 `[0,1,2,3,4,-1]` 的 6 个任务
token。每个任务 token 包括目标刚体的绝对/相对位置、绝对/相对旋转与时间戳。

Critic 使用 privileged observation：根高度、局部刚体位置/旋转/线速度/角速度、关节
状态，并使用 `[0,1,2,4,8,16,32]` 的 7 帧目标，包括目标速度。因此训练是非对称
Actor-Critic：部署侧不需要 critic 的特权信息。

### 6.4 网络与 PPO

当前 M 模型的 Actor 和 Critic 都使用 Humanoid Transformer：

- embedding 256，4 个 attention heads，4 层；
- 前馈维度 256，SwiGLU、RMSNorm、RoPE；
- 本体感知与历史动作交错为 context token；
- 未来动作目标投影成 task token，通过 cross-attention 注入；
- Actor 输出 29 维动作均值，训练时叠加可学习的高斯噪声；
- Critic 输出标量 value。

PPO 默认每环境收集 64 步，2 个 epoch、32 个 mini-batch，`gamma=0.99`、
`lambda=0.95`、clip `0.2`。Actor/critic 学习率分别为 `2e-5` 和 `1e-3`。

### 6.5 奖励与随机化

正奖励包括身体高度、全局身体位置、方向、线速度、角速度跟踪以及存活；惩罚包括动作
变化率和关节限位。启动时随机化摩擦、关节默认位置、躯干质心和双手附加质量，运行中
每 1–3 秒注入一次随机速度扰动。这些随机化用于提高鲁棒性，但也意味着初期误差不会是 0。

## 7. 统一入口的使用方法

先定义本机路径：

```bash
cd /home/sw/bfm

export BFM_RETARGET_PY=/home/sw/.cache/bfm/scaleretarget-py311/bin/python
export BFM_ISAAC_PY=/home/sw/isaaclab_ws/env_isaaclab_sim6_newton/bin/python
export BFM_SMPLX=/home/sw/shuaiwang/.cache/ula_smplx/SMPLX_NEUTRAL_2020.npz
```

### 7.1 只跑数据链路（安全默认）

```bash
"$BFM_RETARGET_PY" scripts/amass_to_scalebfm.py \
  --input /path/to/AMASS \
  --run-name amass_train \
  --smplx-model "$BFM_SMPLX" \
  --retarget-python "$BFM_RETARGET_PY" \
  --isaaclab-python "$BFM_ISAAC_PY" \
  --retarget-workers 4 \
  --package-num-envs 256 \
  --package-loader-workers 8 \
  --output-fps 50
```

默认只执行准备、重定向、打包和 YAML，不会意外启动长训练。完整阶段成功后再次运行会
在输入 SHA-256、重定向流水线指纹、归档版本、`.pkl` 来源 SHA-256 和打包流水线指纹
都匹配时跳过已有重定向与打包结果。
`--output-fps` 当前只接受 50；先加 `--dry-run` 可查看所有命令且不创建文件。

公开小样冒烟测试：

```bash
"$BFM_RETARGET_PY" scripts/amass_to_scalebfm.py \
  --public-sample \
  --run-name amass_public_smoke \
  --smplx-model "$BFM_SMPLX" \
  --retarget-python "$BFM_RETARGET_PY" \
  --isaaclab-python "$BFM_ISAAC_PY"
```

如果只修改了 IsaacLab 打包逻辑，使用 `--force-package` 复用 `.pkl`，不要使用会同时
重跑昂贵重定向的 `--force-data`：

```bash
"$BFM_RETARGET_PY" scripts/amass_to_scalebfm.py \
  --input /path/to/AMASS \
  --run-name amass_train \
  --smplx-model "$BFM_SMPLX" \
  --retarget-python "$BFM_RETARGET_PY" \
  --isaaclab-python "$BFM_ISAAC_PY" \
  --force-package
```

历史 `.pkl` 缺少 sidecar 时，只有在人工确认它确实对应当前输入、模型与配置后，才可
一次性加入 `--adopt-legacy-retarget`。该选项不会重新验证历史计算本身；不确定时应使用
`--force-data` 重新重定向。两个选项互斥。

### 7.2 从官方 M checkpoint 微调

训练是显式 opt-in，只有 `--train` 才启动：

```bash
"$BFM_RETARGET_PY" scripts/amass_to_scalebfm.py \
  --input /path/to/AMASS \
  --run-name amass_train \
  --smplx-model "$BFM_SMPLX" \
  --retarget-python "$BFM_RETARGET_PY" \
  --isaaclab-python "$BFM_ISAAC_PY" \
  --train \
  --train-run-name amass_train_finetune \
  --train-num-envs 256 \
  --max-iterations 1000 \
  --base-run humanoid_transformer_m \
  --base-checkpoint model_22200.pt
```

checkpoint 位于：
`ScaleTrack/logs/rsl_rl/g1_bfm_tracking_exp/<train-run-name>/model_*.pt`。
显存不足时先降 `--train-num-envs`；数据多时可以增大 `--retarget-workers` 和
`--package-num-envs`，但后二者分别受 CPU/RAM 和 GPU 显存约束。

### 7.3 GUI 回放

不传 `--play-headless` 就会打开 GUI。统一入口会显式向本机 IsaacLab 传递
`--viz kit`；这一步不能省略，因为当前 AppLauncher 在没有选择 visualizer 时默认进入
headless 模式：

```bash
"$BFM_RETARGET_PY" scripts/amass_to_scalebfm.py \
  --input /path/to/AMASS \
  --run-name amass_train \
  --smplx-model "$BFM_SMPLX" \
  --retarget-python "$BFM_RETARGET_PY" \
  --isaaclab-python "$BFM_ISAAC_PY" \
  --play \
  --play-run amass_train_finetune \
  --play-checkpoint model_XXXX.pt \
  --mode-index 7 \
  --play-num-envs 1
```

本机当前是 X11，`DISPLAY=:1`。远程终端必须接入已有图形会话；无显示服务时使用
`--play-headless --record-video --video-length 500`，视频会写入所加载 run 的
`videos/play/`。

## 8. 当前已验证产物与训练状态（2026-09-04）

| 产物 | 路径/状态 |
| --- | --- |
| AMASS 兼容 Stage-II | `ScaleRetarget/dataset/amass_smoke_stageii/amass_sample_stageii.npz` |
| G1 重定向 | `ScaleRetarget/retargeted_dataset/amass_smoke/amass_sample_stageii.pkl`；126 帧，约 30.06 Hz |
| G1 打包数据 | `ScaleRetarget/retargeted_dataset/amass_smoke_processed/amass_sample_stageii.npz`；v3，208 帧，50 Hz；来源、流水线和根 FK 已验证 |
| 动作索引 | `ScaleRetarget/retargeted_dataset/amass_smoke.yaml` |
| 参考视频 | `ScaleRetarget/videos/amass_smoke_reference.mp4` |
| 官方起点 | `ScaleTrack/logs/rsl_rl/g1_bfm_tracking_exp/humanoid_transformer_m/model_22200.pt` |
| 短探针 | `.../amass_smoke_quatfix_probe/`；1 个 PPO 迭代已通过 |
| 正式微调 | run `amass_smoke_fixed_finetune`；1000 iterations、256 envs 已完成；最终 `model_23198.pt` |
| 最终策略回放 | mode 7、1 env、250 帧已成功；`.../amass_smoke_fixed_finetune/videos/play/rl-video-step-0.mp4` |

2026-09-04 17:21 最后一轮日志：mean reward `11.71`、mean episode length `201.60`、
`error_body_pos_g=0.1065 m`、`Episode_Termination/body_pos=0.0125`；总采样步数
`16,384,000`，systemd 服务退出码为 0。这是训练末轮窗口，不等于独立测试集统计；当前
小样仍需用未见过的 AMASS subject/sequence 做固定种子评估。

后台训练服务：

```bash
systemctl --user status bfm-scaletrack-amass-fixed-train.service
journalctl --user -u bfm-scaletrack-amass-fixed-train.service -f

# 需要主动停止时
systemctl --user stop bfm-scaletrack-amass-fixed-train.service
```

当前小样只有一条约 4.16 秒动作，适合验证链路或观察单动作微调，不代表通用 BFM
训练效果。通用性必须依赖大量、多样、清洗后的动作库。

## 9. 验收门槛

建议每次扩充数据都按下面顺序验收，不要直接投入长训练：

1. **输入门槛**：Stage-II 字段、形状、fps、有限值全部通过。
2. **重定向门槛**：每条 `.pkl` 至少 3 帧；29 DoF；根四元数为单位 `xyzw`；抽查视频无严重穿地、漂浮和抖动。
3. **打包门槛**：50 Hz、29 joints、30 bodies；磁盘四元数为单位 `wxyz`；全段根位姿与独立重采样参考一致。
4. **1-iteration 门槛**：`error_body_pos_g` 明显低于 `0.5 m`；`body_pos` 终止率不能接近 1；episode 长度大于 1。
5. **短微调门槛**：先跑 50–200 iterations，检查 reward、episode length、位置/旋转误差是否改善，无 NaN/OOM。
6. **回放门槛**：分别检查模式 0、2、4、6、7 的根运动，以及模式 1、3、5 的局部协调。
7. **扩量门槛**：只有上述指标稳定后，才增加动作数量、环境数和训练时长。

本次修复后的探针已通过第 1–4 项，1000-iteration 单动作微调已正常完成且无 NaN/OOM；
尚未完成的是独立测试动作、各稀疏模式的系统评估以及真实通用性验收。

## 10. 常见故障定位

| 现象 | 首要检查 | 处理 |
| --- | --- | --- |
| episode 恒为 1，`body_pos` 终止约 1.0 | 打包时 runtime 四元数顺序和 FK 是否一致 | 用当前代码 `--force-package`，再跑 1 iteration 探针 |
| `root_rot` 非单位或姿态整体翻转 | `.pkl` 是否误当成 wxyz | `.pkl` 固定按 xyzw，不要手工重排后再交给入口 |
| 找不到 SMPL-X | `--smplx-model` 是否指向合法模型/目录 | 使用已有 neutral 模型；完整性别模型按官方许可准备 |
| CUDA OOM | env 数量和其他 GPU 进程 | 降低 `--package-num-envs` 或 `--train-num-envs` |
| GUI 不出现 | `DISPLAY`、X11 权限、是否误传 headless | 本机使用 `DISPLAY=:1`；远程优先录视频 |
| 重跑非常慢 | 是否误用了 `--force-data` | 仅改打包器时使用 `--force-package` |
| 缓存提示 content/provenance/pipeline 不匹配 | 输入、模型、代码、资产、环境或 `.pkl` 已变化 | 确认变化后用 `--force-data` 或 `--force-package` 重建对应阶段 |
| 历史 `.pkl` 提示缺少 provenance | 无法证明它对应当前输入/流水线 | 推荐 `--force-data`；只有人工确认后才用一次 `--adopt-legacy-retarget` |
| 启动前 dependency probe 失败 | 指定 Python 缺少报错中的模块 | 检查解释器路径，不要把 ScaleRetarget 与 IsaacLab 环境混用 |
| 动作有穿地/漂浮 | 重定向质量、height adjust、原始 trans | 查看参考视频，过滤坏片段或调整 correspondence/formatter |
| 单动作指标好但新动作差 | 数据覆盖不足/过拟合 | 扩充并按 subject/sequence 划分 train/test |

## 11. 从 AMASS 跟踪到“完成任务”的开发路线

### 阶段 A：把 AMASS 动作基座跑稳

- 下载有许可的完整 Stage-II 子集；
- 优先选择站立、行走、转向、蹲起、伸手、搬运动作风格，去除躺地、多人和严重接触异常片段；
- 以 subject/sequence 维度划分训练集与测试集，避免相邻帧泄漏；
- 建立自动质量报告：时长、根速度、关节限位、脚底高度、FK 连续性、失败率；
- 从官方 `humanoid_transformer_m/model_22200.pt` 微调，再逐步扩大数据规模。

完成标准：测试动作 `error_body_pos_g < 0.5 m`，回放无系统性摔倒，且加入扰动后仍能恢复。

### 阶段 B：做稀疏控制任务

利用现有八种 mode，先实现不带物体的可控任务：

- 骨盆轨迹导航（mode 0）；
- VR 头/手目标或双手遥操作（mode 2/4，可按需要增加头链接）；
- 双手到达、上半身模仿（mode 1/5/6）；
- 全身参考动作执行（mode 7）。

这一步主要修改 command 生成器与在线目标来源，不必重写底层跟踪策略。

### 阶段 C：增加语义任务层

以“走到桌边并搬箱子”为例，需要新增：

1. 带桌子、箱子和碰撞体的 IsaacLab scene；
2. 物体位姿、目标位姿、接触状态、可达性等观测；
3. 到达、抓稳、抬起、运输、放置、稳定站立等分阶段奖励；
4. 掉落、碰撞、超时和机器人跌倒等终止条件；
5. 能体现接触的 mocap/遥操作/优化轨迹，而不只是 AMASS 空手动作；
6. 高层任务策略到 BFM 身体目标的接口。

推荐先采用分层方案：高层每较低频率输出骨盆/手/脚目标，预训练 BFM 作为 50 Hz
全身低层控制器。这样能复用当前模型的协调和抗扰能力，并把任务学习集中在较低维的
目标空间。待数据充足后，再考虑端到端联合微调。

### 阶段 D：部署

- 导出策略并先在 MuJoCo 做 ScaleBridge Sim2Sim；
- 对齐关节顺序、控制频率、action scale、观测归一化和四元数约定；
- 增加状态估计延迟、通信延迟、传感器噪声和执行器限制；
- 实机从吊架、限速、小动作开始，保留急停和姿态/力矩安全边界；
- 通过仿真回放与日志逐项比较后再扩大动作幅度。

## 12. 相关实现位置

- 统一入口：[`scripts/amass_to_scalebfm.py`](../scripts/amass_to_scalebfm.py)
- 入口测试：[`tests/test_amass_to_scalebfm.py`](../tests/test_amass_to_scalebfm.py)
- AMASS loader：[`amass_loader.py`](../ScaleRetarget/scaleretarget/loader/amass_loader.py)
- GMR/Mink IK：[`gmr_retargeter.py`](../ScaleRetarget/scaleretarget/retargeter/gmr_retargeter.py)
- 原子重定向写入：[`atomic_io.py`](../ScaleRetarget/scaleretarget/utils/atomic_io.py)
- IsaacLab 打包器：[`package_motions.py`](../ScaleTrack/scripts/pretrain/data_process/package_motions.py)
- 50 Hz/分批约束：[`motion_processing.py`](../ScaleTrack/scripts/pretrain/data_process/motion_processing.py)
- 安全版本化归档：[`motion_archive.py`](../ScaleTrack/source/scaletrack/scaletrack/utils/motion_archive.py)
- 精确 YAML 索引：[`create_yaml.py`](../ScaleTrack/scripts/pretrain/data_process/create_yaml.py)
- 四元数兼容层：[`quaternion_compat.py`](../ScaleTrack/source/scaletrack/scaletrack/utils/quaternion_compat.py)
- 动作加载与命令：[`commands.py`](../ScaleTrack/source/scaletrack/scaletrack/tasks/tracking/mdp/commands.py)
- 环境/奖励/终止：[`tracking_env_cfg.py`](../ScaleTrack/source/scaletrack/scaletrack/tasks/tracking/tracking_env_cfg.py)
- G1 模式定义：[`flat_env_cfg.py`](../ScaleTrack/source/scaletrack/scaletrack/tasks/tracking/config/g1_29dof/flat_env_cfg.py)
- PPO/Transformer 配置：[`rsl_rl_ppo_cfg.py`](../ScaleTrack/source/scaletrack/scaletrack/tasks/tracking/config/g1_29dof/agents/rsl_rl_ppo_cfg.py)
- 固定模式回放：[`play.py`](../ScaleTrack/scripts/pretrain/rsl_rl/play.py)
