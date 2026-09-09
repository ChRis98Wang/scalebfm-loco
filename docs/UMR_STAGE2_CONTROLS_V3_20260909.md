# UMR v3：显式腕姿态与输出帧间限速实跑

本轮完成新模块、测试和固定四条动作的 16 组 Stage II 对照。
**两个功能在这四条短窗上验证有效；尚未验证策略跟踪，未晋升训练数据或模型。**
执行设计见 [冻结协议 v3](UMR_STAGE2_CONTROLS_PROTOCOL_V3_20260909.md)。

## 实现

- `scripts/umr_output_rate_limit_v3.py`：按真实 named joint 的 qpos/nv 地址构造 Mink QP 不等式，
  整个输出帧共享上一已输出 qpos 的 ±0.24 rad 预算；固定 12 rad/s、50 Hz、数值容差 2e-6 rad。
  不裁剪、平滑或重置内部迭代预算；frame0 显式无前帧，frame0→1 起全部计数。
- `scripts/umr_stage2_controls_v3.py`：两个 body-frame wrist 的 world orientation 软任务，
  position cost 0、orientation cost 10、LM damping 1；严格区分 warmup60 和逐帧6次 IK，
  保存预热/输出失败统计、实际目标与输出指纹，禁止后处理。
- `scripts/run_umr_stage2_pilot_v3.py`：默认只读计划；固定 control/rate_only/wrist_only/both 四臂，
  复用已绑定的原 canonical 对应点，不重新 Stage I、不运行 PPO/physics。
  启动和完成前核验来源/依赖/代码 SHA，实际解失败时保留失败收据，不输出不合格 motion。

Mink 的优化变量是 delta_q，因此本帧第 j 次迭代要使用：

`P delta_q ≤ q_previous_output + 0.24 − q_current_iterate`

`−P delta_q ≤ q_current_iterate − q_previous_output + 0.24`

普通 VelocityLimit(12) 只限制每次内部迭代的增量，六次可累加；不是同一个约束。
新增真实 Mink/Clarabel、内存构造29关节模型的三项测试：固定预算六次求解、普通限速的累加反例、
position cost 为零时改变 target translation 不改变腕姿态 QP。全部通过，无 mj_step。
12 rad/s 是本管线统一设置，不等于已标定各实物关节速度；free root 不受此 hinge 限速约束。

## 对照结果

全部使用原 v1 flat canonical 和 v2 的 control/setup，不混入上一轮 matched-hand 因素。
4 origin 帧数 251/166/82/172，每臂 671 native inclusive 帧，合计 2,684 帧。
16/16 正常完成，预热与输出 QP failures 全为 0；约 82.46 秒，不含 controller 前置验证。

| origin 等权指标 | control | 仅限速 | 仅腕姿态 | 两者同时启用 |
|---|---:|---:|---:|---:|
| 双腕旧映射目标差 ° | 45.735 | 45.657 | 0.538 | 0.540 |
| 表面点位置残差 m | 0.045781 | 0.045778 | 0.044816 | 0.044892 |
| 表面法线残差 rad | 0.714934 | 0.714885 | 0.720305 | 0.720377 |
| 最大输出关节速度 rad/s | 17.748 | ≤12 | 17.748 | ≤12 |
| 严格超过12 rad/s的 joint-interval数 | 28 | 0 | 27 | 0 |
| 最大自碰撞深度 mm | 0 | 1.783 | 1.615 | 0.497 |
| 自碰撞超过1 mm帧数 | 0 | 1 | 2 | 0 |

“两者同时”每条双腕平均目标差：ACCAD 0.557°、BMLmovi 0.340°、BMLrub 0.554°、KIT 0.708°。
四条均显著下降；8 个启用限速 job 的实际输出速度都满足严格 ≤12，而不仅是容差内通过。
全部16组无脚底 >1 mm 穿地、无检测到的全身穿地、无 joint position 限位违例。

不能概括为全部指标改善：“两者同时”的法线残差四条均增大，BMLrub 表面点位置残差
0.042964945→0.043403021 m，增加约 0.438 mm；BMLmovi 仍有最大 0.497 mm 几何重叠，不能称零碰撞。
这里只是四条已选择 train 短窗的 IK/FK 数据质量诊断，没有接触力、真实机器人或 BFM 策略 rollout。
腕目标直接加入了优化项，残差下降不是独立泛化证据，也不等于学会新的 behavior。

## 证据与进程

结果目录：`local/umr_stage2_v3_20260909a/`，约 37 MiB；未上传受许可输入或派生 motions。

- `status.json` SHA：`b08e6143ec8464eedaa2dfc4d3f3eef9cc899415a40662e9c55fa0f674c9e8fc`
- 协议 SHA：`05b9290520a01cd575067160a700b04296c0e2385e90a280da61f309814da363`
- controller SHA：`f713a0bf0f3785e34cea07fe8fa4ae91032c70319b58a6c68c5d70a9b1a47ab0`
- controls SHA：`c80ce1d1de0692fe4d5ed4505db3f95555fba8a6bcfe60589d437b35e92d7f5e`
- rate-limit SHA：`600b7206487a8f079b866e12a7d63ef883162c5b94d08aced57c3098da9fd6f5`

独立复核13,545输入文件、2,014路径别名和81最终输出，SHA均一致；四个control与原motion逐元素完全相同。
同case四臂共享prepared/setup，在线限速统计与重新np.diff得到的离线统计逐字段相同。
有界unit `bfm-umr-stage2-20260909a` 已 inactive/dead、MainPID=0、cgroup为空。
无 Stage I/PPO 更新、无 mj_step、无 GPU 计算进程，旧7174训练数据和默认官方模型保持不变。

## 回归

新增48项纯CPU测试（rate18、controls21、controller9）及3项真实Mink求解测试。
完整本地回归：

- 现有 Isaac Python：1,109 passed，1,000 subtests，65 warnings，67.92秒。
- 原独立 Python3.11 八模块：73 tests OK，5.49秒。
- 现有 UMR Python：3项真实Mink集成测试通过，0.107秒。

合计 **1,185 项主测试**；未重复计入子集，无依赖安装/升级。日志分别为
`logs/behavior_learning/umr_stage2_v3_fulltests_20260909a.log`、
`umr_stage2_v3_py311tests_20260909a.log`、`umr_v3_mink_tests_20260909a.log`。
这不是远端CI通过的声明，新的Mink/NumPy测试使用既有本地环境。

下一段：扩展到完整同源 pilot，重新打包并独立核验 FK/时钟，再做八模式策略跟踪回归。
在那之前，不启动下一轮数据替换PPO，不宣称完整ScaleBFM或loco-manipulation已经完成。
