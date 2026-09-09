# UMR 局部参考替换的 Behavior 学习 A/B 协议

状态：**FROZEN v1，执行结果另记**。冻结日期：2026-09-09。

本文在新的 A/B 数据清单生成和 PPO 执行前冻结；实际执行状态以绑定本文 SHA 的新清单、训练和评估收据为准。
执行器冻结本文列出的参数、样本、代码、检查点和门槛。不得先查看本试验结果，再挑样本、改预算、换 seed、
挑中间检查点或调整通过门槛后仍称为原协议；任何变更须发布新协议版本，并保留原试验全部结果。

## 目的与当前证据

问题是：在同一批原 train origin、同一参考时间窗和同样训练条件下，用 UMR 参考替换其中 17 个旧参考，
能否产生有用的策略学习改善，并保持原固定验证集上的能力。

这 17 个 origin **均已属于现有 `amass_full_v2_train.yaml` 的 7,174 个训练 origin**，其中 4 个来自 KIT。
它们是训练池中的局部参考替换，不是新增来源，也不是将 development 改为 train。
本协议先在独立的 256-origin 训练池上试验；不会覆盖 7,174 条原数据、原索引或官方默认权重。

现有 UMR 证据见 [40-origin 正式对照报告](UMR_PAIRED_EVALUATION_20260909.md)：

- 40 个固定 origin，各数据源 6 train + 2 development，共 30 train + 10 development。
- 实际验收的是从原始第 0 帧开始、最多 5 秒的时间窗，不是 40 条完整动作。
- 24 个 UMR 窗口通过几何筛查，其中 17 train、7 development；另 16 个拒绝项仍参加冻结策略完整对照。
- 足底穿地明显改善，但新增自碰撞和参考幅度变化仍存在。几何通过不证明不滑、不摔或可完成接触任务。
- 同一官方策略的新旧参考八模式联合不退步为 0/8；该批 PPO 更新为 0。不能据此断言新策略学不会 UMR。
- 40-origin pilot 和这 10 个 development 已用于方案开发，不得称为最终 test 或未接触过的测试集。

既有训练链也已经真实运行：`behavior_coverage_long_20260908a` 完成 1,000 次 PPO 更新、8,192,000 个环境步、
全部 7,174 个 train origin 覆盖，71 组 actor 参数改变，两侧 Adam 状态各增加 64,000 步。
随后 `behavior_long_eval_20260908a` 完成正式验证，但为 `FAIL_NO_REGRESSION`：总体 0/8、来源分层 0/16。
这证明训练和评估工程链能运行，尚未证明新模型达到既有质量门槛，更不能承诺完整论文性能复现。

## 现有入口的复用边界

| 环节 | 现有入口 | 本试验需要处理的边界 |
|---|---|---|
| 来源、时间窗与 packed 契约 | `scripts/evaluate_umr_pairs.py` 的 `validate_pairs` / `bind_geometry` | 先验证完整 40-origin 原清单，再派生目标子集；不改原清单的数量或身份 |
| 整数时钟打包 | `ScaleTrack/scripts/pretrain/data_process/package_paired_motions.py` | 本轮复用已通过的 `packed_paired_clock/`；不使用早期多帧的 `packed/` |
| 独立全身 FK | `scripts/audit_umr_body_fk.py` | 绑定正式审计报告、机器人几何及实际 packed SHA；旧报告不代表未来新全长产物 |
| 真实学习观察 | `ScaleTrack/scripts/pretrain/rsl_rl/training_probe.py` | 复用 rollout、mask、origin、Adam 和参数更新观察；需要显式、受审计的新索引与 seed 契约 |
| 固定学习配置与 coverage | `scripts/run_bfm_long_training.py` | 复用 fixed LR、5-update cohort 和有界进程；需要新的 100-update 数据 A/B 控制器 |
| 原验证集逐行核验 | `scripts/run_bfm_long_evaluation.py`、`evaluate.py` | 绑定本次 A/B 检查点和训练证明，不能沿用历史候选 SHA 或硬编码 runner iteration |
| 模型回归比较 | `compare_learning_ablation.py`、`scripts/compare_mask_evaluations.py` | 在同一套固定评估参考上比较两个模型；读取质量字段，不能把退出码 0 当质量通过 |

`run_bfm_learning_ablation.py` 原本比较 fixed/adaptive 学习率，并非数据 A/B。
当前 `training_probe.py` 硬锁 canonical full_v2 路径、7,174 条和 seed 42；当前 long controller 只允许 6/1,000 更新。
因此本设计不是只换一个 YAML 参数即可执行。应新增独立数据试验控制器，并增加明确的 probe 契约，
保持原脚本默认协议和历史证据有效；不能通过改写 canonical 索引绕过检查。

当前 `UniformMotionCohortSampler` 只接受完全均匀权重，拒绝非均匀概率。
本协议通过固定训练池构成确定目标/回放比例，不新增加权采样算法，也不复制别名来增加目标权重。

## 数据版本和确定性选择

训练池共 256 个唯一 origin：17 个目标 + 239 个旧参考回放。A/B origin 集合及顺序完全相同。

| 内容 | A：旧参考对照 | B：UMR 局部替换 |
|---|---|---|
| 17 个目标 | 正式 `packed_paired_clock/` 中对应的 baseline 窗口 | 同批、同 origin、同时间窗的 candidate 窗口 |
| 239 个回放 | 原 full_v2 train 所指向的旧 packed 文件 | 与 A 完全相同的文件和 SHA |
| 7 个几何通过 development | 仅评估 | 仅评估 |
| 其余 3 个 development | 保留开发诊断，按预声明分层报告 | 保留开发诊断，按预声明分层报告 |

目标集合严格取已绑定正式几何审计的 `split=train` 且 `candidate.kinematic_pass=true` 的 17 个 origin，
不根据冻结策略跟踪成绩另挑更容易的目标。先验证它们都在 full_v2 train，且不在两个原验证索引中。

回放选择规则固定如下，实际列表须写入执行前清单：

1. 从 full_v2 train 排除这 17 个目标，得到 7,157 个候选；只使用 train 身份和来源信息，不使用验证成绩。
2. 按 origin 的数据集前缀分层，按剩余各层数量比例分配 239 个名额，使用最大余数法；余数相同按数据集名排序。
3. 各层按 `sha256("umr_behavior_ab_20260909/v1/replay/" + origin_id)` 升序选择，哈希相同按 origin 排序。
4. 合并 17 + 239 后，A/B 均按同一个明确的 origin 字符串顺序输出；记录最终顺序、分层计数及全部 payload SHA。

17 个目标在 A、B 都使用成对短窗。不能让 A 用完整动作、B 用前 5 秒，否则数据流程与时长同时改变。
239 个回放允许保留原完整参考，因为它们在两臂完全一致。本文的 256-origin 结果不代表全库分布验收。

不得改变 `amass_full_v2_train.yaml`、`amass_full_v1_validation.yaml`、`amass_kit_heldout_v1.yaml` 或其原 payload。
由新清单引用独立候选文件，保留每个 origin 的原索引映射、来源身份和裁窗身份。

## 固定训练配置与预算

| 参数 | 两臂相同的值 |
|---|---|
| 任务 | `G1-BFM-Transformer-Tracking` |
| 起点 | `official_lr1e5_derived_20260907/model_22200.pt`，SHA `269f17e040ad0c27f651a25097a2650380ff1a05714dde102625aabd8d327c46` |
| 起点身份 | 与官方检查点仅两侧 Adam LR 字段不同；模型与其余优化器状态相同 |
| 学习率 | actor/critic 均 `1e-5`，`schedule=fixed` |
| PPO | entropy `0.001`，2 epochs，32 minibatches |
| rollout | 128 环境 × 每环境 64 步 = 每更新 8,192 环境步 |
| 采样 | `coverage`，均匀、独立 CPU RNG，每 5 个更新换 128 个 origin |
| 正式预算 | 每臂 100 个完整 PPO 更新 |
| 工程 smoke | 每臂另做 6 个完整 PPO 更新；正式训练重新从共同初始检查点开始 |
| 验证和保存 | `eval_during_training=False`；save interval 50；正式只比较最终 checkpoint |
| 首轮 seed | 42；两臂同 seed，GPU 物理不承诺位级确定性 |
| 确认阶段 | 首轮协议完成后按下述固定条件运行 seed 43、44；不是结果不佳后另挑 seed |

6-update smoke 用于证明首次 5-update cohort 内 origin 固定，第 6 update 进入另一组不相交的 128 个 origin，
以及两侧优化器真实推进；它不是训练质量评估。不得查看 smoke 的 heldout 成绩决定是否保留 A 或 B。
两臂 smoke 都通过工程核验后，才启动两臂正式 100-update 试验。

256 个 origin 在每 10 个更新中完成一轮，100 个更新共 10 轮。
在现有固定 cohort 语义下，每个 origin 应实际贡献 3,200 个环境步，目标总计 54,400 步（6.640625%），
旧回放总计 764,800 步。以上必须由实际 rollout 审计重算，不以 sampler 曾发出某个 ID 代替真实使用证据。

| 预算范围 | PPO 更新 | 环境步 |
|---|---:|---:|
| 每臂 smoke | 6 | 49,152 |
| 每臂正式 | 100 | 819,200 |
| seed 42，两臂含 smoke | 212 | 1,736,704 |
| 确认阶段 seed 43/44，两臂正式合计 | 400 | 3,276,800 |
| 全部计划上限，含首轮 smoke | 612 | 5,013,504 |

确认阶段的固定启动条件：首轮两臂工程审计完整，且 B 相对 A 通过下面的原固定验证集模型门槛。
若未通过，记录本版本结果，停止该版本的晋升流程；任何继续训练或改变协议属于新版本。
确认阶段不复用 seed 42 的训练终点，每个 seed 均从共同初始权重开始。所有已运行 seed 必须完整报告。

正式每臂应有 100 次完整 rollout/update，actor 有有限值参数发生变化，两侧 Adam 各推进 `100 × 64 = 6,400` 步。
最终 runner iteration 从输入 checkpoint 内容计算，不从文件名猜测；历史起点为 22199 时，100 更新的终点为 22298。
记录每 update、origin、数据源、mask 的实际环境步，推荐补充 origin × mask 联合计数。
两臂 origin cohort 必须相同；策略导致的状态、重置和 mask 使用差异需如实记录，不要求物理轨迹相同。

进程使用独立有界 user service，`KillMode=control-group`、`Restart=no`，并冻结有限的时间、内存、任务数和 GPU 配置。
新控制器应默认只读计划；所有输出必须是新目录。质量失败也保留完整候选和报告，不覆盖已有结果。

## 数据门槛与模型门槛分开

### 数据资格与冻结参考诊断

训练目标的数据资格来自已审计 packed 文件的来源、50 Hz 时间轴、显式名字/顺序、速度一致性、独立全身 FK 和几何：

- 足底穿地不超过 1 mm；身体地面/自碰撞穿透不超过 5 mm；关节位置限位容差 `1e-6 rad`。
- 独立全身 FK 的位置、旋转误差均不超过原定 `1e-4 m` / `1e-4 rad` 数值门槛。
- 这些是数据工程阈值，不是实物或动态接触阈值；参考速度、幅度和保真相关统计继续报告。

UMR 冻结策略对照的三项门槛是 active-link 全局位置、active-link 旋转、全部参考 link 全局位置均误差，
同一官方模型比较新旧参考，要求各项 candidate−baseline ≤ `1e-6`，且逐八 mask 检查，不能漏掉第三项。
当前整体 0/8 是保留的诊断结果，不伪装成已过关；也不把这个冻结模型诊断作为禁止试验新策略学习的理由。
它比较的是参考数据变化，不是两个训练模型的晋升判断。

### 开发诊断：两个模型使用同一套参考

固定全部 10 个 development origin，分别建立旧参考和 UMR 参考的两个评估清单。
A、B 都在这两套参考上按八 mask 评估；7 个几何通过项作为预声明子集，另 3 个不静默丢弃。
逐 origin 先求均值，再等权汇总；分别报告上述三项误差以及参考幅度和速度，不跨 mask 混成一个平均数。
开发比较采用同一参考内 B−A，不能拿 A 对旧参考与 B 对 UMR 参考的各自奖励直接宣称学习改善。

这部分是已参与开发的数据上的适配诊断，不是最终 test。参考几何穿地的构造改善也不是策略实际足部接触改善。
物理滑移、跌倒、力矩或后续任务指标若尚未具备有效测量，标为未测，不从跟踪误差推断成功。

### 原固定验证集：模型回归和晋升

原固定基准仍是 `amass_full_v1_validation.yaml` 的 962 个 non-KIT origin 加
`amass_kit_heldout_v1.yaml` 的 789 个 KIT origin，共 1,751 个。
在对应 seed 下，A、B 和官方模型使用同一套原 packed 参考、同一 runtime 指纹和相同协议：
八个 canonical mask；1024 环境；50 Hz；每动作最多 1000 步；关闭观测噪声、重置扰动和周期推力；
不加入物体；评估 PPO 更新为 0。沿用当前窗口时应为 605,664 个有效步、98 条截断，不能称完整时长验收。

原模型门槛为总体和 KIT/non-KIT 两个分层的八 mask 全部满足以下四项，不能与数据诊断的三项门槛混写：

1. active-link 全局位置均误差不高于对照 + `1e-6 m`。
2. active-link 旋转均误差不高于对照 + `1e-6 rad`。
3. active-link 的每动作最大位置误差 ≤ 0.5 m 的动作通过率不下降。
4. 全部 14 个参考 link 的每动作最大位置误差 ≤ 0.5 m 的动作通过率不下降。

同时补充逐 mask、逐来源的三项均误差检查表，包括全部参考 link 的全局位置均误差，明确 B−A 和 B−官方；
不能只展示 active 两项而隐藏全身均误差退步。该补充三项均误差不退步检查亦按 `1e-6` 数值容差预先冻结，
作为本版本候选晋升的附加条件，不替代原四项门槛。

首轮是否进入确认阶段按 B 相对 A 的原四项门槛及附加三项检查共同决定。
最终候选在 seed 42/43/44 中须分别对 A 和官方模型满足这些已冻结条件；若 B 优于 A 但仍退步于官方，
可以作为后续研发线索，不能更换官方默认。三个 seed 不得只汇总一个平均数掩盖未通过的 seed。
`1e-6` 是数值容差，不是统计显著性检验；0.5 m 是既有跟踪阈值，不代表接触任务成功。

比较器退出码 0 只表示报告生成成功。必须读取 `promotion_candidate_no_regression`、新增三指标检查结果和逐模式明细。
本版本任何质量结论都不触发自动晋升；先形成绑定数据、训练和评估证明的可回退候选记录。

## 执行前冻结和防泄漏核验

至少冻结以下输入，执行前后各核验一次，并在所有 A/B/seed 报告中绑定同一实验版本：

- 原 full_v2 train 和两个原 development/validation 索引：完整内容、路径、SHA、origin 到原 payload 的映射。
- 原 40-origin 选择清单、`paired_manifest.json`、正式几何/FK 报告及其 SHA；10 个 development 的身份和两个成对参考清单。
- 17 个目标、239 个回放的确定性选择算法、实际 origin 顺序、source 身份、所有实际加载文件的 SHA。
- 目标的原始源文件、时间窗、源/packed 时钟规则、插值证明、joint/body names、尺度/高度规则及 UMR commit。
- A/B 初始检查点及模型/两侧优化器状态，训练/评估配置，seed、预算、检查点选择规则和全部门槛。
- 实际训练和评估 Python runtime、控制器、probe、comparator、机器人 XML/相关几何、依赖版本和设备配置。

验证隔离按原始 origin 继承，不能只检查新文件路径或 SHA。重定向、重打包和裁窗会改变内容哈希，
不会让原 development 变成新的训练样本。训练加载器必须只装载这 256 个 train origin，禁止 `--test_motion_file`，
并核对实际 rollout origin 都在清单中。

现有检查排除了训练/验证名字、路径及精确 payload 重叠，但明确没有排除所有近重复动作或上游数据重叠。
官方初始 checkpoint 的上游训练集是否接触过这些 origin 也未由本地 split 证明；不能声称它们对官方模型完全未见过。
原固定验证集已用于多轮开发/回归，不升级称为独立最终测试集。

新增代码须在 A/B 正式运行前冻结。不能 A 训练后修改受审计 runtime，再 B 训练/评估并沿用旧 fingerprints。
若与导出或 Sim2Sim 同时开发，使用已冻结且彼此隔离的运行输入；不混用历史结果和当前代码结果。

## 试验之后的边界

若本版本获得一致收益，下一步是为这 17 个 origin 生成完整时长的新旧候选，重新做全段契约、几何、FK、速度和幅度验收，
再建立 7,174-origin 的独立版本化 A/B 索引。不能把前 5 秒通过当作整条动作可替换。
全库均匀采样中 17/7174 仅约 0.237%；训练曝光和总预算须单独预声明，不能把本次 6.640625% 的结果冒充全库效果。

ScaleBFM 仿真主线中的导出数值一致性、global/local 控制、MuJoCo 闭环可以继续按各自证据推进。
数据可读、PPO 有更新、八 mask 回归通过、导出一致和 Sim2Sim 闭环各自证明不同事项。
这些事项全部完成前不称仿真全链路验收完成；loco-manipulation/箱子演示随后另行定义接触、成功/失败和扰动验收。
本协议没有声称新学习已达标、完整 ScaleBFM 已复现或任务策略已经掌握搬运。

## 权威证据位置

- `docs/UMR_PAIRED_EVALUATION_20260909.md`
- `logs/behavior_learning/umr_amass_pairs_20260909a/geometry_audit.json`
- `logs/behavior_learning/umr_amass_pairs_20260909a/comparison.json`
- `logs/behavior_learning/behavior_coverage_long_20260908a/status.json`
- `logs/behavior_learning/behavior_coverage_long_20260908a/train_audit.json`
- `logs/behavior_learning/behavior_long_eval_20260908a/status.json`
- `docs/DATA_REFRESH_AND_SCALEBFM_COMPLETION.md`

本文件是执行前冻结的协议 v1，不是训练收据或数据/模型晋升批准；后续进度记录不改写本协议。
