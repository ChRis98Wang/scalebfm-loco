# KIT 入库与后续开发记录

## 当前状态

2026-09-07 用户要求先完成 KIT，再继续开发并准备 loco-manipulation 演示。
沿用现有 ScaleRetarget / IsaacLab 环境，不安装或升级依赖，不覆盖既有数据和权重。

- 安全解压完成：4344 个归档成员，4288 个普通文件，10,608,829,500 字节。
- 4232 个 Stage-II 候选中，4231 个通过 CRC、核心数组、模型类型与帧率检查。
- `KIT/9/WalkingStraightBackwards08_stageii.npz` 只有 **1 帧**，不满足重定向至少
  3 帧的要求；原文件保留，排除而不伪造补帧。这不是整个归档下载失败。
- 有效源动作均为 100 Hz，共 3,971,044 源帧，约 661.84 分钟；这是源动作时长，
  不是已经训练的时长，也不是动作类别数。
- 18 个普通批次、10 个长动作批次全部成功退出，4231 条均完成 G1 29 DoF
  重定向、50 Hz v3 打包及独立全量来源/格式/FK 审计，没有截短长动作。
- 已发布新的 KIT/v2 索引；旧 v1 索引字节不变。新训练池 **7174 条**。
- 新索引已完成 **1 次实际 PPO 更新**：55 条 KIT 动作进入 3520 个环境步。
  这只是接入与诊断验收，不代表全部 KIT 已充分训练，更不代表效果达标。
  官方、旧微调和 GUI 默认权重不替换。全部本轮数据/训练进程已结束。

## 最终入库结果

| KIT 索引 | 动作条数 | 50 Hz 帧数 |
| --- | ---: | ---: |
| `amass_kit_all_v1.yaml` | 4231 | 1,984,676 |
| `amass_kit_train_v1.yaml` | 2927 | 1,512,924 |
| `amass_kit_heldout_v1.yaml` | 789 | 275,978 |
| `amass_kit_excluded_v1.yaml` | 515 | 195,774 |

索引均位于 `ScaleRetarget/retargeted_dataset/`。515 条普通平地训练排除项中：
496 条有非平地/外部支撑名称线索，15 条不足 1 秒，2 条标定，2 条源根速度超过
12 m/s；这四类在本批次不重叠。过短动作同时有源和打包标记，不能重复计数。
所有成功产物仍可供后续专门场景研究，不删除源文件，也不把排除当作下载失败。

有效 KIT 源文件全哈希无重复，与旧 5256 条源文件的哈希交集为零。合并后：

- `amass_full_v2_train.yaml`：旧 4247 + KIT 2927 = **7174 条**，3,315,606 帧；
- `amass_collected_all_v2.yaml`：五个来源共 **9487 条**成功产物；
- `amass_collected_unique_v2.yaml`：保留旧去重选择后共 **9484 条**；
- 旧 962 条验证基线仍使用 `amass_full_v1_validation.yaml`；KIT 的 789 条单独报告。

KIT 的 55 个数字目录按整组分配；筛选后训练/留出分别有 40/13 个非空目录组。
这不是已经证实的跨人员隔离，也没有做近重复/共同录制事件去重。

KIT 重定向共 1,194,298 帧、345,609,870 字节 PKL；打包文件
3,628,721,226 字节。机器人数据约 3.70 GiB，加本归档与解压内容约 **16.97 GiB**，
不含运行日志、checkpoint 和原有四个数据集。

## 保留与审计

原归档：`/home/sw/Downloads/KIT.tar.bz2`。
SHA-256：`0f27388ec6ff3af31a54ef424f4d5d567d214d8389403f0914bf52021f962fa0`。
解压前核对全部成员均在 `KIT/` 下，仅接受普通文件和目录，拒绝越界、重复路径、
符号/硬链接和特殊文件；以不存在的新 KIT 目录写入，并使用 Python data filter。
解压结束复核文件数、总字节数及源归档未变化。

原始目录：`ScaleRetarget/dataset/amass/KIT/`。
每批准备目录仅引用已审计 Stage-II 文件，不纳入其他 NPZ 或 Stage-I。
来源、分批清单及执行日志：

- `logs/amass_preparation/kit_20260907_raw_audit.json`
- `logs/amass_preparation/kit_20260907_manifest.json`
- `logs/amass_preparation/kit_20260907_progress.jsonl`
- `logs/amass_preparation/kit_controller_20260907.log`
- `logs/amass_preparation/amass_kit_batch_v1_<r/l><编号>_20260907.log`
- `logs/amass_preparation/kit_20260907_batch_results.json`
- `logs/amass_preparation/kit_20260907_postpack_audit_all.json`：最终权威全量审计；
- `logs/amass_preparation/kit_20260907_environment_recheck.json`：重新计算当前环境/代码指纹；
- `logs/amass_preparation/kit_20260907_indexes.json`：索引 SHA、划分、排除和旧索引冻结记录。

本次审计/发布脚本分别为同目录下的 `kit_20260907_postpack_audit.py` 和
`kit_20260907_publish_indexes.py`。经过独立审查，发布前重新哈希新旧源数据、
打包文件与新增 PKL，检查实际 28 批/4231 条分区、唯一 YAML 键、规范路径、目录组
和 train/heldout 文件哈希隔离。所有新输出独占创建；旧 train、validation、all、
unique 四个索引哈希固定，不覆盖。发布标记在七个索引落盘后才写出。
历史单批/阶段性审计保留，不替代最终 `*_all.json`。

复用 `scripts/amass_to_scalebfm.py`，其 57 项数据入口回归测试已在现有 retarget
Python 下通过。每批由入口执行输入校验、G1 29 DoF 重定向、来源/流水线指纹记录、
50 Hz v3 打包、29 关节/30 刚体/四元数/根部 FK 校验以及精确批次索引生成。
格式和 FK 通过不等于策略能够在物理中跟踪全部动作。

## 资源与恢复

普通动作不超过 4000 源帧；每批最多 256 条、250000 源帧，最多四批并行。
长动作按约 110000 源帧分批、单批串行，不截短原动作。
每批 2 个重定向进程、1 个打包加载进程；普通批 32 个打包环境，长批 8 个。
数值库线程数限制为 1。

首批独立运行已正常退出。其余批次受同一
`bfm-kit-controller-20260907.service` 控制，3 小时总上限、36 GiB 总内存限制，
每批有 45 分钟上限。停止时仅清理本任务进程组和 cgroup，不使用广域 `pkill`。
剩余 27 批调度实际运行 1 小时 4 分 18.972 秒，全部退出码 0；首批另约 60 秒。
调度组内存峰值达到 36.0 GiB、swap 峰值 7.5 GiB，因此没有提高长批并发或同时启动
训练。结束后已确认 MainPID=0、inactive/dead，GPU 计算进程为空；没有常驻调度器。

重定向 sidecar 在整批重定向成功后写入。被中断的批次即使已有 PKL，也不能直接
当作可信缓存。先检查日志和产物，再决定是否重跑本轮对应新批次；不对旧 v1 目录
使用 force，不默认认领缺失来源记录的文件。

## 索引验收规则

完成打包后再次逐条校验来源、形状、有限值、单位四元数和输出 SHA，再发布新索引。
旧训练索引 4247 条、旧验证索引 962 条及其划分保持字节不变。KIT 的新增 heldout
独立报告，不把其结果混入旧 962 条基线。

KIT 以 AMASS 顶层数字目录作为不可拆分分组，使用与 v1 一致的 seed 42 哈希规则
分配 train/heldout。数字目录尚未独立核实为真实人员身份；跨目录、跨数据集人员身份
及共同录制事件不明，不能宣称是严格的跨人员最终测试集。

标定命名、台阶/坡面/梁等非平地参考、过短或极端数值动作单独登记，不删除原始或
成功生成的机器人数据。训练池沿用已有数值门槛：时长至少 1 秒、根高度范围
[0.05, 2.5] m、最大根速度不超过 12 m/s、关节速度不超过 80 rad/s、身体中心不低于
-0.2 m。命名标记只是筛查线索，不是已验证的任务语义。

人体动作中带有 carry/push 等名称，并不表示包含物体状态、双侧接触或搬运监督。
后续 loco-manipulation 仍需要在线目标、实际物体状态、接触判断和任务验收；
不通过把箱子绑定到手臂或直接写箱子轨迹伪造完成。

## 实际训练接入验收

run：`amass_full_v2_lr1e5_diag_step1_20260907`。从保留模型和 Adam moments 的
官方低初始 LR 派生副本出发，128 环境 × 64 rollout 步，seed 42，
`entropy_coef=0.001`，adaptive schedule，2 epochs × 32 minibatches，
`max_iterations=1`，关闭训练内评估。未从零训练，也未把 heldout 用于更新。

在每次真实 `PPO.act` 前只读记录 CPU motion id，实际环境步数：
ACCAD 256、BMLmovi 1920、BMLrub 2432、CNRS 64、KIT **3520**，共 **8192**。
KIT 有 55 条不同片段被采样；加载全部 2927 条 KIT 不等于单次更新遍历了全部片段。
模型 141 个张量改变，actor 71/critic 70 个 Adam 状态的 step 均增加 64；
新 checkpoint 与 55 个 TensorBoard 标量标签的数值均通过有限性检查。

- checkpoint：`logs/rsl_rl/g1_bfm_tracking_exp/amass_full_v2_lr1e5_diag_step1_20260907/model_22199.pt`
- SHA-256：`d8e253be07cd85410c19f173ae8e08710259bc2de9a49ace15701a4e8b1b9eed`
- 日志：`logs/amass_training/kit_diag_step1_20260907.log`
- 实际采样：`logs/amass_training/kit_diag_step1_20260907_rollout.json`
- 数值/Adam/TensorBoard 验证：`logs/amass_training/kit_diag_step1_20260907_verification.json`

训练服务正常退出，22.455 秒、内存峰值 9.7 GiB、swap 峰值 0，随后确认无本轮
训练/仿真进程。学习率/KL 实测见 [PPO 诊断记录](SCALEBFM_PPO_DIAGNOSTICS_20260907.md)。
没有对新 checkpoint 做八模式质量 gate，所以不升为默认、不据此自动长训。
后续 demo 的参考片段见 [候选审计](LOCO_MANIPULATION_REFERENCE_CANDIDATES_20260907.md)。
