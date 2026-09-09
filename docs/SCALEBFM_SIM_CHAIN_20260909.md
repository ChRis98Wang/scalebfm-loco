# ScaleBFM 导出、Sim2Sim 与后续任务链路

更新：2026-09-09。本轮真实执行，不是仅有配置或未来计划。

## 结论与范围

当前已打通 **官方检查点 → 可验证 TorchScript → 原生 ScaleBridge → MuJoCo 实际物理闭环**。
此前数据生产、Transformer+PPO 更新、独立验证已有运行证据；本轮补齐部署接口这一段。
这不代表新的本地策略已经优于官方，也不代表完整 BFM 性能、任意在线目标或搬箱任务已完成。

| 环节 | 本轮结果 | 不能据此声称 |
|---|---|---|
| 静态部署契约 | 29 joint/action、30 body、14目标、8 mask、history 3、future 6、50 Hz，来源 SHA 绑定 | 原官方训练环境的 runtime dump |
| 无 TensorRT 导出 | 8 mask × 2 输入 seed，16/16 数值检查通过 | TensorRT/Jetson/真机已验收 |
| MuJoCo 资产契约 | 30 link FK 一致；新 variant 的29关节 nominal力矩一致 | 接触/质量/动力学完全等同 IsaacLab 或实物 |
| 原版 XYZ-relative local | 48/48执行完成，46/48跟踪保护通过 | 局部 pelvis 位置误差0代表完美控制 |
| 保留高度的 XY-relative local | 48/48执行完成，47/48跟踪保护通过；未触发跌倒保护、无数值警告 | 所有质量门槛已过、无打滑、会搬箱 |
| 原生启动入口 | headless + 10步真实运行并退出；有界/关闭/契约测试 | 本轮做过GUI人工或实物验收 |
| UMR后续训练 | 已制定17条候选＋239条旧数据回放的配对协议，尚未执行 | 本轮已经训练或推广UMR数据 |

7,174条原训练数据、1,751条原开发验证、官方默认checkpoint都未替换；没有新增PPO更新、发布或上传。
上轮本地长训练相对官方仍为0/8、来源分层0/16不退步门槛通过。

## 为什么要新增无需 TensorRT 的路径

原Bridge强制导入TensorRT并要求CUDA，启动入口无限循环且始终开viewer。本轮复用已有
`/home/sw/.cache/bfm/scaleretarget-py311/bin/python`，没有安装依赖、重装或升级IsaacLab。
该解释器提供Python 3.11、PyTorch 2.14.0+cu130、MuJoCo 3.12.0；正式导出和本轮闭环在CPU执行。
TorchScript保留官方九Tensor输入及 `(PD joint targets, raw action)` 输出；不是CVAE，也未重新训练模型。
它是现有Bridge兼容路径，PyTorch已给出TorchScript弃用提示，不据此承诺跨版本或长期部署兼容。

新增/修复入口：

- `scripts/build_bfm_deployment_metadata.py`：受限AST解析，不执行Isaac配置；用冻结评估证明、命名archive和checkpoint核验字段。
- `ScaleBridge/scalebridge/agent/portable_policy.py`：复用原Transformer网络源码，严格载入actor权重，明确四元数、FK、mask和PD约定。
- `scripts/export_bfm_torchscript.py`：重建整个metadata profile防止错head数/错mask自洽通过；新目录导出、保存、重新加载、数值比对。
- `scripts/prepare_bfm_sim2sim_asset.py`：只创建独立nominal力矩variant，原XML/mesh不改。
- `scripts/evaluate_bfm_sim2sim.py`：保留原版global/XYZ-relative协议与已冻结结果。
- `scripts/evaluate_bfm_sim2sim_height.py`：独立版本化XY-relative/world-Z协议，避免修改历史受审计入口。
- `ScaleBridge/scalebridge/env/motion_tracking_height.py`：只对齐观测中的XY，保留实际Z反馈；不修改物理状态。
- 原生Bridge修复无GUI、有限步、SIGTERM/finally关闭、SHA/维度/单位四元数检查、RSI角速度坐标转换、默认姿态和限矩。

metadata是**nominal静态profile**。原模型未附训练配置；不从state_dict形状猜`num_heads`，而是使用
冻结本地配置。部署默认角不包含startup的±0.01rad随机化。训练future `[0,1,2,3,4,-1]`
最后一项代表动态未来；部署使用原导出入口明确设置的 `[0,1,2,3,4,5]`，不是负时间。
目前profile只接受已审计官方M权重；新训练候选导出还需绑定其独立训练/配置证明，不能绕过profile检查。

## 数值一致性证据

正式目录：`local/scalebfm_deployment_20260909a/export_v2/`。
输入官方checkpoint SHA：`88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3`。
每个mask测试seed42/314159的非零输入，CPU eager、保存重载以及独立NumPy/SciPy旋转＋MuJoCo FK构造
观测后调用原actor。**并未运行IsaacLab实时观测逐Tensor对照**，不将这部分写成已验证。

| 检查 | 实测最大绝对差 | 预设上限 |
|---|---:|---:|
| 保存重载 vs eager action/PD | 0 | 1e-4 |
| 保存重载 vs 独立观测原actor action | 3.577e-6 | 1e-4 |
| 保存重载 vs 独立观测原actor PD/rad | 1.550e-6 | 1e-4 |
| 本体观测 | 1.193e-7 | 2e-5 |
| 任务观测 | 1.133e-6 | 2e-5 |

受测导出是batch=1、history=3、future=6的固定形状；mask为运行时Tensor，不是导出时固化某一个模式。
错误输入shape/dtype/device/NaN/四元数/时间偏移和未通过验证的产物会被新Bridge入口拒绝。

## 物理与时钟契约

原Bridge双髋roll motor为±88Nm，而训练nominal与joint force limits为±139Nm。
`sim_asset/g1_29dof_nominal_torque.xml`仅把两个motor改为±139Nm，另把mesh路径指回原目录；
保留碰撞、质量、惯量、摩擦、阻尼、armature和关节限位，23组编译模型数组逐项未变。
这是仿真参数一致性修复，不是更改真机额定能力或实物标定。
Bridge还保留两个额外固定body与1g总质量差；完整30参考link的命名FK一致不等于所有动力学相同。

策略50Hz、MuJoCo200Hz，每个策略步4个物理子步。初始RSI一次，之后只有PD力矩推进，无机器人/物体瞬移。
积分后显式刷新`mj_kinematics`再量指标，避免将5ms前的link pose与新参考比较；每步核验MuJoCo时钟。
参考分别229/226/250帧，实际228/225/249控制步；不循环尾帧、不多走一个终点。

## 48组完整窗口结果

选择固定的3个已审计旧参考短窗，不使用UMR候选，不表示7174条或全部动作类型已测：

- `KIT/3/squat01_stageii`：4.56秒。
- `KIT/359/walking_run09_stageii`：4.50秒。
- `BMLrub/rub114/0013_knocking1_stageii`：4.98秒；文件名仅为动作线索，不是物体交互标签。

各自8 mask × global/local，48次独立RSI，共11,232控制步，即44,928物理子步；没有片段截断。
工程完成字段与质量门槛分开。预设跟踪保护要求全部14参考link最大位置误差≤0.5m，
pelvis高度≥0.3m、up dot≥cos75°且无MuJoCo数值警告；不是最终动作精度或接触/打滑成功判据。

| 片段 | world-XYZ | XY-relative、保留world-Z |
|---|---:|---:|
| 下蹲 | 8/8 | 8/8 |
| 行走 | 7/8 | 8/8 |
| 上肢参考 | 8/8 | 8/8 |

正式高度版：`logs/sim2sim/height_20260909a/report.json`，`result=COMPLETE`，
`tracking_guard_pass_jobs=47`，**`quality_gate_pass=false`**。全程没有触发跌倒保护、没有数值警告，
29关节力矩契约匹配。CPU单步纯策略调用的最差episode p95约1.42ms，不是硬件端到端延迟。

唯一未过是行走/Pelvis-1/global：骨盆平均误差8.58cm、最大19.90cm，未受控部位令全14link最大误差达到58.23cm。
“未受控部位偏离”不等于策略摔倒，但原先声明的全身参考保护没有全过，不能看完结果改门槛宣布通过。
WholeBody-14/global行走实际根净位移2.776m，身体均误差9.193cm；下蹲/上肢分别2.723cm/4.148cm。
这仍是离线参考，不是自由在线导航/目标到达率。

原版local会把root XYZ都替为参考值，Pelvis-only位置误差为0是数学恒等。
原版下蹲/local失败且机器人最低高度约0.770m；高度版实际降到0.468m，骨盆高度均误差2.716cm，
该组全14link最大误差39.38cm，通过同一保护门槛。新local只去XY误差，**不会去掉Z误差**；
同时另报未对齐的世界坐标误差，避免用local精度冒充绝对落区精度。
v1的46/48与v2的47/48是输入控制语义修正，不是新模型学习提升或扩大样本的独立统计实验。

## 本机重跑与进程管理

复用现有产物验收（输出目录、unit必须用新名称；脚本拒绝已有目录）：

```bash
cd /home/sw/bfm
systemd-run --user --unit=bfm-sim2sim-height-review01 \
  --property=KillMode=control-group --property=Restart=no \
  --property=RuntimeMaxSec=600 --property=TimeoutStopSec=10 \
  --property=MemoryMax=6G --property=TasksMax=128 \
  --property=WorkingDirectory=/home/sw/bfm \
  --setenv=OMP_NUM_THREADS=2 --setenv=OPENBLAS_NUM_THREADS=2 \
  /home/sw/.cache/bfm/scaleretarget-py311/bin/python -u \
  /home/sw/bfm/scripts/evaluate_bfm_sim2sim_height.py \
  --policy /home/sw/bfm/local/scalebfm_deployment_20260909a/export_v2/policy.pt \
  --packed-root /home/sw/bfm/local/umr_amass_pilot_20260908a/packed_paired_clock/baseline \
  --xml /home/sw/bfm/local/scalebfm_deployment_20260909a/sim_asset/g1_29dof_nominal_torque.xml \
  --output /home/sw/bfm/logs/sim2sim/height_review01 \
  --translations both --max-steps 250 --strict-torque-limits --device cpu
```

```bash
journalctl --user -u bfm-sim2sim-height-review01 --no-pager -n 20
systemctl --user show bfm-sim2sim-height-review01 -p ActiveState -p MainPID -p ExecMainStatus
systemctl --user stop bfm-sim2sim-height-review01
```

最后一条用于提前停止该明确unit，禁止用全局`pkill python`影响其他项目。自然结束也会清理进程；
本轮服务全部`MainPID=0`。保留原始失败smoke和原版矩阵，不删除以美化结果。

本轮测试用现有py311，不能为测试向Isaac解释器强装loguru/TensorRT：

上述6个测试模块合计58项通过；没有将测试中的mock stepping计作实际物理验收。

```bash
/home/sw/.cache/bfm/scaleretarget-py311/bin/python -m unittest \
  tests.test_bfm_deployment_metadata tests.test_scalebridge_runtime \
  tests.test_portable_bfm_policy tests.test_bfm_sim2sim_evaluation \
  tests.test_bfm_sim2sim_asset tests.test_bfm_sim2sim_height -q
```

## 下一步与demo范围

1. 主线学习：按 [版本化UMR配对训练协议](UMR_BEHAVIOR_AB_PROTOCOL_20260909.md) 跑17条候选的可学习性对照；
   通过固定旧验证/KIT、多seed后才扩大替换并晋升，保留原7174条。短窗候选不直接冒充全长生产数据。
2. 控制验收：继续解决剩余Pelvis-only全身参考偏离，补Isaac实时观测对照、长时间/多初态/任意在线XYZ目标，
   并把新checkpoint的训练证明接入导出profile；不因这3个clip跑通就把所有控制标为完成。
3. 任务扩展：依 [D0/D1/D2交付路线](LOCO_MANIPULATION_DELIVERY_PLAN.md) 做带上肢目标行走、站立双臂夹箱、
   再走近/搬1m/放下。无需手指，但需要实际接触与物体任务训练，不能把箱子绑定到手上演示。
4. 发布/实物：清洁环境安装、许可证/数据资产权利、CI与实物接触标定仍需单独完成。

本轮没有启动硬件接口、上传代码/数据/权重，也没有宣称材料接触已经接近实物标定结果。
