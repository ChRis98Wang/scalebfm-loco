# BFM 目标物体验证环境与稀疏模式菜单

本阶段在现有 IsaacLab 播放器和独立评估入口中加入**可选的动态目标方块**。
同时打通动作切换、参考点模式切换、物体单独复位与实际状态显示。
这是任务控制的环境基础，不是已经学会抓取、搬运或自主导航。

## 一条命令启动

在本机已解锁的桌面终端、仓库根目录执行；继续使用现有 IsaacLab，无需重装：

```bash
bash scripts/open_motion_browser.sh --target-object
```

不加 `--target-object` 就是原来的无物体动作浏览器。默认加载本地最终微调权重，
初始模式为 WholeBody-14；这不表示它优于官方权重，已有配对评估结果见
[2026-09-07 基线报告](BFM_EVALUATION_20260907.md)。

显式使用仓库提供的工程参数，以及 VR-3 初始模式：

```bash
BFM_TARGET_CONFIG=configs/target_objects/box_engineering.json \
BFM_MODE_INDEX=2 bash scripts/open_motion_browser.sh --target-object
```

参数文件的相对路径以调用者当前目录为准；路径有空格时加引号。其他机器仍需准备
合法取得的本地数据和可信 checkpoint，并设置 `BFM_ISAAC_PYTHON`。
`--dry-run --target-object` 仅打印命令，不启动任何仿真。

## 菜单功能

| 控件 | 实际行为 |
| --- | --- |
| Search / Clear / 动作下拉框 | 筛选 5209 条本地片段；选择候选不会立即重置 |
| Play selected / Previous / Next / Restart | 从第 0 帧切换或重播，重置机器人历史与物体 |
| Reference mode | 选择参考点集合，尚未应用；不是自由移动目标点 |
| Apply mode + restart | 应用参考点掩码，并从头重播当前片段 |
| Reset target only | 恢复物体初始位姿、零速度；不重启机器人动作 |
| Target local XYZ / root distance | 物体实际中心坐标与真实机器人骨盆的三维距离，单位米 |
| Exit player / 关闭菜单 | 退出整个播放器并关闭仿真 |

可在菜单切换 Pelvis-1、UMI-2、VR-3、UMI-4、VR-5、UpperBody-6、
UpperBody-Mobile-7、WholeBody-14。VR-3 是骨盆和双腕；WholeBody-14 使用全部参考点。
切换改变实际策略输入掩码，不只是改标签，但仍从**离线动作片段**取参考目标。
继续保持机器人单独显示，不增加人类骨架叠加。可拖动菜单标题栏到右侧，避免遮挡机器人。
通过底层 Python 入口使用 `--local_tracking` 时，菜单仅保留兼容的固定模式；物体距离
仍读取真实机器人的物理位置，不使用替换后的参考骨盆位置。

物体位置相对环境原点，不自动跟随机器人或手腕；各动作的初始根位置不同，所以
同一个方块不一定处于每段动作的可接触范围。单独复位是仿真瞬移，不是机器人把物体放回。

## 物理参数与“接近真实”的边界

配置文件：[`configs/target_objects/box_engineering.json`](../configs/target_objects/box_engineering.json)。
尺寸和位置用米、质量用千克；位置是中心，不是底面。当前默认：

| 参数 | 默认值与含义 |
| --- | --- |
| size / position | 0.30 × 0.30 × 0.30 m；中心 (0.9, -0.7, 0.16) m |
| mass | 1.0 kg，动态刚体，启用重力与实体碰撞 |
| static_friction / dynamic_friction | 0.6 / 0.5，静摩擦与动摩擦工程初值 |
| restitution | 0.05，低弹性恢复系数 |
| contact_offset / rest_offset | 0.005 / 0 m，物体碰撞形状的接触生成与静止偏移 |
| linear_damping / angular_damping | 0 / 0.02；滑动检查不靠额外线性阻尼“刹停” |
| color / roughness | 棕色、粗糙度 0.75；只影响外观，不代替物理摩擦 |

方块使用几何与质量产生的刚体惯性近似，没有指定实物测得的重心/惯量。
配置接触求解位置迭代 16 次、速度迭代 4 次，最大去穿透速度 2 m/s；沿用任务
5 ms 物理步长、20 ms 策略步长。没有安装外部材质包或下载在线 USD 环境。

物理材质与渲染材质是独立配置；接触是两个碰撞形状共同决定的，不能只看方块参数。
当前地面摩擦为 1.0/1.0，双方摩擦组合为 multiply，因此本场景方块—地面组合为
0.6/0.5。恢复系数用 max 组合，地面为 0 时该组合为 0.05。
机器人沿用任务的启动材质随机化，机器人—方块的摩擦不能只用方块参数代表。
组合规则参考 [NVIDIA 刚体与物理材质文档](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/rigid_bodies.html)。
contact/rest offset 的作用与两形状求和规则参考
[NVIDIA 碰撞形状文档](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/collision.html)。

这些是**已通过仿真行为检查、尚未实物标定**的工程值，不把棕色方块命名为某种材料的
测量模型。后续接近指定实物，需要提供尺寸、质量、重心/惯量以及实际接触表面，
用静摩擦起滑、滑动衰减、回弹测试拟合参数；若对象会显著变形，还需要相应形变模型。
“仿真能滑停”不能证明 sim-to-real 一致。

JSON 只接受 `TargetObjectSpec` 的输入字段，可以只覆盖部分值；未知字段、非有限数、
非正质量/尺寸、动摩擦大于静摩擦、非法恢复系数和初始穿地位置均在启动仿真前拒绝。
评估报告中的 `calibrated`、`shape`、组合模式等是输出元数据，不应复制成输入字段。

## 独立评估

已有 `evaluate.py` 新增 `--target_object`、`--target_config` 和
`--target_position X Y Z`。注意 Python 参数用下划线，shell 启动开关用连字符。

```bash
"$BFM_ISAAC_PYTHON" ScaleTrack/scripts/pretrain/rsl_rl/evaluate.py \
  --checkpoint_path logs/rsl_rl/g1_bfm_tracking_exp/amass_full_v1_finetune_20260906/model_23197.pt \
  --motion_file ScaleRetarget/retargeted_dataset/amass_full_v1_validation.yaml \
  --output logs/evaluations/my_target_object_smoke_v3.json \
  --num_envs 128 --max_steps 10 --seed 42 --mode_index 7 \
  --target_object --target_config configs/target_objects/box_engineering.json
```

该命令只是每段 10 步的集成检查。质量评估应使用完整的统一评估协议，并给每次运行
使用新输出路径；已有文件会拒绝覆盖，带物体结果不能与旧无物体基线直接混比。

新报告为 schema 3：`scene_variant` 区分 baseline / target_object，记录生效物理参数，
保留 checkpoint、动作文件、实际 Python 源码的前后校验。新增逐片段与汇总指标：

- `target_root_distance`：实际物体中心到真实骨盆的三维距离，m。
- `target_height`：物体中心的世界坐标 Z，m，不是离手高度。
- `target_speed`：物体线速度模长，m/s。

它们使用和跟踪指标相同的有效物理步掩码，不统计自动重置后的样本。指标是诊断量，
没有抓取成功率、抓稳接触判定或搬运目标达成判定。物体状态未新增到策略观测/奖励，
所以现有 checkpoint 保持兼容，但策略可能无意撞到方块，不能据此宣称学会物体交互。

## 本机验收证据（2026-09-07）

- 真实物理检查：两个环境中的 30 cm 方块从高处落地，中心 Z 稳定在约 0.150000 m；
  施加 1 m/s 水平初速度后约滑行 0.09945 m，最终速度约 0.000232 m/s。
  单环境复位不改变另一环境。脚本 `ScaleTrack/tests/target_object_physics_smoke.py`，
  日志 `logs/amass_training/target_physics_20260907.log`。这不是实物标定结果。
- 真实 Kit 控件检查通过：搜索/选择、模式请求、物体复位请求、HUD、空结果、退出、
  构造失败清理。脚本 `ScaleTrack/tests/playback_menu_gui_smoke.py`。
- 真实机器人全链路检查通过：`playback_robot_gui_smoke.py` 调用现有 `play.py`，
  加载最终微调权重与 5209 条动作，不替换策略或物理引擎；依次切片、应用 VR-3 / 
  Pelvis-1 / WholeBody-14，核对实际掩码分别为 3 / 1 / 14 个点。将物体移开再调用
  Reset target only，验证方块回位且动作帧继续前进，最后走 Exit player 正常退出。
  日志 `logs/amass_training/target_robot_ui_20260907.log`。
- 交互窗口另外记录到 UMI-2 / UpperBody-6 和动作切换、物体复位，截图可见实体方块与
  机器人，缩小后的地面视觉网格未再出现先前的大范围条纹。
  日志 `logs/amass_training/target_motion_browser_20260907.log`。
- 962 条验证片段的带物体短程评估已跑通；每条仅 10 步，不是完整动作质量或接触成功评估。
  最终代码与 JSON 参数文件复测报告：
  `logs/evaluations/20260907_target_object_wholebody_smoke_final_v3.json`，共 9620 个有效步，
  输入内容前后校验通过。
- 最终本地回归：105 项 ScaleTrack/启动器/CI 配置测试通过；独立 Python 3.11 环境中
  CI 精确子集 26 项通过。CI 子集与前述测试有重叠，不把两者相加宣称覆盖量。
- 本次验收完成后，测试 systemd units 已卸载，未发现本次播放器/评估进程或 GPU
  计算进程残留；模型未改写。没有替用户保留一个运行到晚上的后台仿真。

数值/配置测试、Kit 控件测试、真实物理检查、真实机器人播放检查是不同证据层次。
当前本机 SDK/驱动组合见 [环境说明](../CONTRIBUTING.md)，不能据此声称所有 IsaacLab
版本均已验证。测试日志和数据均保持本地，不随源码自动发布。

## 退出与后续能力

播放器固定使用临时 `bfm-motion-browser.service`，最多 30 分钟，不自动重启；
本次自动验收已走正常退出路径。用户紧急停止命令：

```bash
systemctl --user stop bfm-motion-browser.service
```

仍未实现：外部在线目标输入/键盘导航、目标点行走闭环、物体条件策略训练、抓取与搬运
任务及成功判定。下一阶段应先选一个明确任务、加入相应目标与观测/奖励，并为其设立
独立成功标准，再决定是否需要任务适配训练。本次没有开启新训练、改写权重或宣称
BFM 全部工作完成。开源方面已加入 CPU CI，许可证与依赖完整复现仍见
[发布清单](RELEASE_CHECKLIST.md)。
