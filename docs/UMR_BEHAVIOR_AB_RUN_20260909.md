# UMR 局部参考替换：配对学习执行记录

本页记录 2026-09-09 的实际执行；固定条件见 [执行前冻结协议 v1](UMR_BEHAVIOR_AB_PROTOCOL_20260909.md)。
它是新的学习试验，不是替换全部 7,174 条数据或宣布完整 ScaleBFM 复现。

**最终状态：数据版本、双臂 smoke、各 100 次正式 PPO 和 56 组评估全部执行完成；本版本未通过质量门槛，
不晋升模型、不替换全库、不启动 seed 43/44 确认。所有本轮服务已退出，无 GPU 计算进程残留。**

## 已完成的数据版本

新目录：`local/umr_behavior_ab_dataset_20260909a/`。数据生成服务正常退出，`MainPID=0`。

| 内容 | 数量 / 含义 |
|---|---|
| 独立训练池 | 256 个原 train origin，A/B 顺序相同 |
| 局部替换目标 | 17 个已通过几何筛查的 train 短窗；A 用旧参考，B 用 UMR 参考 |
| 原数据回放 | 239 个确定性分层选择的 train origin；A/B 文件相同 |
| 开发诊断 | 10 个原 development origin，其中几何通过 7 个；不进入梯度 |
| 固定回归验证 | 962 non-KIT + 789 KIT = 1,751 个原 origin |
| 实际输入审计 | 10,578 个文件，绑定源数据、原索引、配对时钟、几何/FK 及检查点 |

256 个 train 按来源分布：ACCAD 11、BMLmovi 56、BMLrub 86、CNRS 1、KIT 102。
其中 KIT 并未遗漏，也没有新增或移动 development 的身份。17 个目标都采用新旧成对的最多 5 秒窗口，
不能把它们当作 17 条整段已替换。原始全库索引、payload 和官方默认权重保持不变。

- `train_a.yaml` SHA：`2371b53368c28e334ed38866cdbf0aa46cf8a535e53909e7cc3c488d2bfa647c`
- `train_b.yaml` SHA：`282339cb89e3add548eaa25ea31e8b345ab396fa9f057f94418d59ecde0a5306`
- 协议 SHA：`254a70dbdda526d7fc9190f748577bc6b0095cb6d93f84a82c24e420e9afc8b6`

### 参考幅度和速度：不能只看穿地

以下按本版本 origin 清单，对已绑定的 `umr_amass_pairs_20260909a/geometry_audit.json` 逐 origin 等权汇总。
数字为 **旧参考 → UMR 参考**，是原有冻结参考统计的重汇总，不是新策略实测。
同一窗口的 packed SHA 已由新清单复核；不按新模型评估结果筛选样本。

| 固定分组 | 骨盆 XY 跨度 m | 骨盆高度范围 m | 关节平均活动范围 rad |
|---|---|---|---|
| 17 train 目标 | 0.7067 → 0.6203 | 0.0854 → 0.0749 | 0.5245 → 0.5513 |
| 全部 10 development | 2.0516 → 1.6173 | 0.1081 → 0.0912 | 0.5570 → 0.5694 |
| 几何通过 7 development | 1.8007 → 1.4357 | 0.1211 → 0.1058 | 0.5890 → 0.5982 |
| 几何拒绝 3 development | 2.6373 → 2.0412 | 0.0778 → 0.0572 | 0.4826 → 0.5021 |

| 固定分组 | 骨盆平均线速度 m/s | 骨盆平均角速度 rad/s | 关节平均绝对速度 rad/s |
|---|---|---|---|
| 17 train 目标 | 0.3163 → 0.2822 | 0.6728 → 0.7912 | 0.6079 → 0.6549 |
| 全部 10 development | 0.5422 → 0.4384 | 0.8278 → 0.9702 | 0.7083 → 0.7485 |
| 几何通过 7 development | 0.5159 → 0.4242 | 0.7569 → 0.8669 | 0.6707 → 0.6952 |
| 几何拒绝 3 development | 0.6037 → 0.4716 | 0.9932 → 1.2113 | 0.7959 → 0.8730 |

这再次说明 UMR 不只是统一上移脚底：骨盆平移/高度幅度变小，角速度和关节活动也有变化。
不能把后续学习差异完全归因于足底接触改善；策略滑移、实际碰撞力及箱子任务仍需另测。

### 事后排查：骨盆坐标系中的姿态迁移

在看到 development 结果后，新增 `scripts/analyze_umr_reference_shift.py`，并生成
`logs/behavior_learning/umr_reference_shift_20260909a.json`，SHA
`cfc5daeeda9417f3b14529c69097c1d80670f0dc404513075937a9710698b7e9`。
这是 **事后描述性排查，不属于已冻结的 v1 质量门槛**；未改变本轮数据、训练器或比较器。
40 项新测试通过，真实报告核验 10,585 个文件，服务正常退出、`MainPID=0`。

两侧各自使用骨盆完整 SO(3) 坐标系：`p_rel = R_pelvisᵀ(p_body − p_pelvis)`；
`q_rel = conjugate(q_pelvis) × q_body`。使用 link frame 而非 COM，四元数为单位 `wxyz`；
报告位置欧氏差及最短 SO(3) 姿态角，各 origin 等权。pelvis 自身差值按定义为零，另报剔除它的 13 link。

| 参考集合 | 17 train 位置差 cm | 10 dev 位置差 cm | 17 train 姿态差 ° | 10 dev 姿态差 ° |
|---|---:|---:|---:|---:|
| 非 pelvis 的 13 link | 4.881 | 5.419 | 18.728 | 20.255 |
| 双腕 | 8.094 | 7.095 | 51.168 | 56.489 |
| 双踝 | 7.854 | 10.022 | 12.040 | 12.975 |

JSON 保留每个 origin、每个 link 和集合的 mean/p95；跨 origin 的 p95 字段是“各 origin p95 的平均”，
不是混合全部帧的 p95。该坐标化去掉各自骨盆的全局平移及旋转，不用于衡量高度、heading 或全局跟踪。

腕部姿态差异是下一步检查人体→机器人目标坐标系和数据适配的线索，但不是策略误差，也不能证明 solver 有 bug。
这里比较的是不同 shape、scale、约束和重定向流程的整体结果，尚不能判断哪套腕姿态更符合原始人体动作。

## 新增可执行链路

- `scripts/build_umr_behavior_ab.py`：默认只读计划；只生成新版本索引，完整重建来源契约。
- `scripts/umr_behavior_training_probe.py`：观察真实 rollout、八模式使用、来源曝光、两侧 Adam 和有限值参数；
  成功报告在 Kit 关闭前写入并 fsync，退出码不单独作为通过证据。
- `scripts/run_umr_behavior_ab.py`：先两臂各 6-update smoke，再从共同起点分别 100-update；
  正式训练要求同输入指纹的双臂 smoke 证明。默认不执行、不自动晋升。
- `scripts/evaluate_umr_behavior_ab.py`：新旧两参考的开发诊断共 32 组；官方/A/B 原固定验证集共 24 组。
- `scripts/compare_umr_behavior_ab.py`：在同一参考内比较 B−A 和 B−官方，保留原四项门槛，
  加上 active 位置、active 旋转及全身位置三项均值门槛；逐八模式和 KIT/non-KIT 分层，禁止自动晋升。

首轮 84 项 CPU 测试及 44 个子测试通过；包含伪 runner 的真实观察器测试，不把这些测试当作 GPU 学习成功。
新比较器还只读核验了历史 16 份真实 schema-4 报告的格式兼容性。

随后完成全部分环境回归：现有 Isaac Python 下 948 项通过、617 个子测试通过；现有重定向 Python 3.11
下另 73 项通过，合计 **1,021 项主测试**。后者覆盖 Bridge 及依赖 `loguru` 的重定向模块，未为此向 Isaac
环境安装依赖。全套测试暴露并修复了一处测试顺序污染：leaf-import 检查移至独立、有超时的进程；
仅测试文件改变，训练 probe SHA 保持 `2407406b16f59ed37f8b3228fcb56991207fa27f2d346cf6f4ebea69ee719d48`。
失败的首轮收集/组合测试日志保留，不覆盖。最终日志为：

- `logs/behavior_learning/umr_behavior_cpu_tests_20260909c.log`：948 passed，617 subtests passed，65 warnings。
- `logs/behavior_learning/umr_behavior_cpu_tests_py311_20260909a.log`：73 tests，OK。

加入上述 40 项离线姿态诊断测试后再次跑完整回归，最终为 **988 passed、803 subtests passed**，
另保留未改动的 Python 3.11 八模块 **73 tests OK**，合计 **1,061 项主测试**，不重复计先前轮次。
最终日志 `logs/behavior_learning/umr_behavior_cpu_tests_20260909d.log` 的 SHA 为
`cf7f8f0337affd5fa44a9c361c0a82fef3b995254f66968dd3a8ab585f68f5cb`；所有 CPU 测试服务均已退出，`MainPID=0`。

CI 已添加新训练计划、评估计划和比较器的 28 项纯契约测试；另以 `python -S -m unittest` 验证它们不依赖
第三方 site-packages、受许可数据或模拟器。这里只验证本地执行，未声称远端 GitHub CI 已运行。

## 执行状态

双臂 smoke 于 14:30:05 启动，约 91.3 秒后完整结束，服务 `MainPID=0`、`Result=success`。
收据：`logs/behavior_learning/umr_behavior_smoke_20260909a/status.json`。

| 实际 smoke 结果 | A / B |
|---|---|
| 完整 PPO 更新 | 各 6 次 |
| 实际环境步 | 各 49,152，总计 98,304 |
| 实际使用的原 train origin | 各 256；128 个使用 320 步，另 128 个使用 64 步 |
| 两臂 cohort | 实际序列相同，前 5 update 固定，第 6 update 切换至不相交的 128 个 |
| 两侧 Adam | 每臂 actor/critic 各增加 384 步 |
| 八 mask / KIT | 全部实际进入 rollout |
| 最终 runner iteration | 各 22204；从 checkpoint 内容核验 |
| 数据、runtime 和机器人资源 | 前后指纹相同 |

正式双臂各 100-update 服务于 14:32:23 启动，重新加载共同初始状态，不使用 smoke 终点。
收据：`logs/behavior_learning/umr_behavior_train_20260909a/status.json`。
约 770.9 秒后双臂完整结束，服务 `MainPID=0`、`Result=success`。

| 正式训练验收 | 结果 |
|---|---|
| 完整更新 / 环境步 | A/B 各 100 次 / 819,200 步；合计 200 次 / 1,638,400 步 |
| 256 个 origin 的实际曝光 | 每臂每条恰好 3,200 步，共 10 轮 |
| 17 个目标 / 239 个回放 | 每臂分别 54,400 / 764,800 步 |
| 两臂真实 cohort | 完全相同；不要求物理轨迹或重置后的 mask 数量相同 |
| 参数和优化器 | 每臂 71 组 actor 参数变化；actor/critic Adam 各推进 6,400 步 |
| 最终 runner iteration | 22298；未选择中间 checkpoint |
| 前后冻结输入 | 数据、runtime、评估器/比较器及机器人资源均未变化 |
| 自动晋升 | 无，默认模型不变 |

最终检查点：

- A：`logs/rsl_rl/g1_bfm_tracking_exp/umr_behavior_train_20260909a_a/model_22298.pt`，
  SHA `828ebf000daa6927d9f9ab755643d5d54e950b516406de68dbf0fada0f21b46b`。
- B：`logs/rsl_rl/g1_bfm_tracking_exp/umr_behavior_train_20260909a_b/model_22298.pt`，
  SHA `85b2fa90b669a75104434e19ec80269c3a9a030876fe0d23375834a51504cae5`。

本轮含双臂 smoke 共 212 更新、1,736,704 个环境步。四个训练日志/检查点目录合计约 697 MiB；
新数据清单约 7.5 MiB，通过引用原数据避免整库复制。

100 次更新的数值诊断全量汇总如下，不能把 KL 小或 loss 有限直接解释为行为提升：

| 诊断 | A | B |
|---|---:|---:|
| 每 update 的 minibatch 平均 KL，再对 100 updates 等权平均 | 0.0016120 | 0.0015570 |
| 所有 minibatch 中最大 KL | 0.0119276 | 0.0119139 |
| 每 update clip fraction 均值，再对 100 updates 等权平均 | 0.0115869 | 0.0112964 |
| 所有 minibatch 中最大 clip fraction | 0.1796875 | 0.1875000 |

两臂 actor/critic 的实际 LR 始终为 `1e-5`，未重新出现历史恢复 LR 被优化器旧状态覆盖的问题。

## 56 组评估的最终结果

收据：`logs/behavior_learning/umr_behavior_eval_20260909a/status.json`，
SHA `9eec45aedc784e022bd25f450e884257f9e81bfd22b10442b4c04bad3bce3cf0`。
56/56 全部正常执行，约 2,368.7 秒；输入前后核验通过，评估 PPO 更新为 0。
控制器 `result=COMPLETE`，但质量为 **`FAIL_THIS_SEED_NO_REGRESSION`**，不能把完成状态当作质量通过。
独立审计另实读 1,771 个评估 NPZ 的帧数，核验 3,100 个唯一冻结输入、56 份报告及三份比较的 SHA；
三份比较 JSON 内存重算后与发布值逐字段完全一致。工程证据通过与质量门槛未过是两个独立结论。

- 开发诊断 32 组：两参考 × 两模型 × 八模式，每组 10 origin、1,959 有效步、0 截断。
- 原固定验证 24 组：官方/A/B × 八模式，每组 1,751 origin、605,664 有效步、98 个截断窗口。
- 原验证集是反复使用的开发/回归基准，未改称全时长验收或独立最终 test。

### 原固定验证集：保留两个不同的门槛表

下表表示八模式中通过的数量；四项门槛和三均值附加条件并不相互替代。

| 比较 | 四项门槛：总体 | 四项门槛：non-KIT | 四项门槛：KIT | 三均值：总体 | 三均值：non-KIT | 三均值：KIT |
|---|---:|---:|---:|---:|---:|---:|
| B 对 A | 2/8 | 1/8 | 1/8 | 4/8 | 4/8 | 1/8 |
| B 对官方 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |

B 对 A 的总体四项门槛通过模式是 VR-3、WholeBody-14。不能写成所有指标都退步：
B 的总体 active 位置均误差在八模式中都比 A 小，但部分模式旋转误差或逐动作最大误差通过率退步。
例如 UMI-2 的三项均值都改善，active-link 超过 0.5 m 的失败动作却由 3 条增至 4 条；
VR-5 的 active/all14 超阈值失败由各 1 条增至各 2 条。因此它们仍不通过原四项联合门槛。
不因为差值小或仅差一条动作，就在看完结果后修改冻结阈值。

全身模式的实际均误差如下。WholeBody-14 的 active 集合就是全部 14 个参考 link，
两位置指标仅有浮点归约级差别；不重复把它们当作两个独立改善证据。

| 模型 | WholeBody-14 active 位置 cm | active 旋转 rad | 全 14 link 位置 cm |
|---|---:|---:|---:|
| 官方 | 4.6626 | 0.114459 | 4.6626 |
| A：旧参考对照 | 4.8656 | 0.119723 | 4.8656 |
| B：17 个 UMR 参考替换 | 4.7869 | 0.119338 | 4.7869 |

B 相对 A 的全身位置改善约 0.788 mm，但相对官方仍增大约 1.243 mm，旋转均误差也高于官方。
这些数值是在既定参考和窗口下的跟踪误差，不是搬运成功率、实际足底穿透或接触力指标。

### 开发诊断：新旧参考分别比较 B−A

| 同参考三均值联合门槛 | 全部 10 origin | 几何通过 7 origin | 几何拒绝 3 origin |
|---|---:|---:|---:|
| 旧参考 | 4/8 | 4/8 | 5/8 |
| UMR 参考 | 0/8 | 0/8 | 0/8 |

旧参考 WholeBody-14 的 active 位置 A→B 为 5.4698→5.3012 cm，旋转 0.122398→0.121245 rad；
UMR 参考则为 5.1415→5.8235 cm，旋转 0.191960→0.197429 rad。
本预算下 B 没有展现更好的 UMR development 适应；这不是“同一旧策略不适应新数据”的零更新试验，
而是实际训练后的两个模型对照。但它也不能证明所有 UMR 数据或更完整训练都无效。

### 收据、进程和后续边界

- `heldout_comparison.json`：SHA `2a0e1a6b4e215d9f2ea121753d547bd37841cb2f56ba8d63a27c8bcc557087d7`。
- `development_baseline_comparison.json`：SHA `5428b4198715779f9a072f00d87ac47210cd1918260ebc1978098e262858729e`。
- `development_candidate_comparison.json`：SHA `b416b97079cbac9e62f5146bb5970762a3c5155530b81262ed7cc4abceada4eb`。

`confirmation_quality_condition_met=false`；按冻结协议停止本版本的晋升流程，保留全部模型与失败结论，
不继续加预算挑终点、不启动 seed 43/44，不覆盖原数据/默认模型。
训练、数据生成、离线分析及评估服务全部退出；最终 `bfm-*` running unit 为 0，GPU compute process 列表为空。

后续应另立新版本：先核对人体→机器人腕部目标坐标/语义及新旧参考保真，再确定目标数据曝光和旧策略能力保持方案。
本轮的事后腕姿态诊断是该方向的线索，尚未确定原因；任何新训练需独立冻结条件并保留本版结果。
只有数据/模型回归通过后，才扩大完整时长候选和全库替换，并继续候选部署/Sim2Sim及 loco-manipulation 验收。
目前双臂夹箱搬运仍未实现，也不能声称完整 ScaleBFM 复现已经完成。

现有 IsaacLab 环境未重装；仅使用现有 Python 和 GPU。所有执行服务限定时间、内存及任务数，
`KillMode=control-group`、`Restart=no`，子进程超时会清理并回收；不启动 GUI 或访问实机。

HTML：当前仓库没有项目 HTML 页面，也没有本轮新建网页。训练进度和验收以本页及 JSON 收据为准；
网页需求待确认具体页面，不阻塞学习主线。
