# ScaleBFM 连杆掩码验收

## 路线与边界

2026-09-07 用户明确继续当前 ScaleBFM：Transformer actor-critic + PPO、G1 29 DoF、
八组连杆目标掩码。不是原始 BFM 的 CVAE + 在线掩码蒸馏，也不新增原版的三类输入。
本阶段先验证控制器，暂不扩展箱子搬运；不重装 IsaacLab，不覆盖原权重和数据索引。

八组掩码来自 `flat_env_cfg.py`：Pelvis-1、UMI-2、VR-3、UMI-4、VR-5、UpperBody-6、
UpperBody-Mobile-7、WholeBody-14。它们选择可见的身体目标，并非八个独立模型。

## 新增评估能力

现有 `evaluate.py` 增加可选 `--mask_metrics`。关闭时保留 schema 3；开启后输出
schema 4，旧指标和旧报告不改动。新指标在 CommandTerm 原有 `_update_metrics()`
边界计算：物理步之后、参考帧推进之前。不能在 `env.step()` 之后重新取下一帧参考来
计算当前误差。临时包装仅用于本次独立评估，异常/退出时恢复，不改奖励或策略输入。

| 新指标 | 口径 |
| --- | --- |
| `error_active_body_pos_g` | 激活连杆的世界坐标位置距离均值，米 |
| `error_active_body_pos_root_relative` | 分别减去参考根/实际物理根平移后的位置误差；不消除朝向，米 |
| `error_active_body_rot` | 激活连杆的最短四元数姿态角误差均值，弧度 |
| `error_active_body_pos_g_max` | 单个有效时刻，激活连杆中的最大位置误差，米 |
| `error_all_body_pos_g_max` | 单个有效时刻，全部 14 个配置参考连杆中的最大位置误差，米 |

`summary.masked_tracking` 分别报告 active/all links 的跟踪通过率：一条片段任意有效
时刻、任意被统计连杆误差 **> 0.5 m** 即算该片段跟踪失败。此标准与旧报告的
“14 点平均误差 > 0.5 m”不同，通常更严格；旧字段仍保留。

这不是跌倒检测，也不是物体任务成功率。评估继续模拟失配后的状态，不因误差超过
阈值而重置。趴下、爬行等参考本身允许低位姿态，不能用统一根高度阈值冒充跌倒判定。

## 首轮配对协议

- 原验证索引 962 条，原训练索引 4247 条，均不修改；KIT 不在本轮数据内。
- 官方 `humanoid_transformer_m/model_22200.pt` 与本地
  `amass_full_v1_finetune_20260906/model_23197.pt`，共 8 × 2 组。
- 每组固定一种 mask；512 环境、seed 42、50 Hz、每条最多 1000 步（20 秒）。
- 关闭噪声、重置扰动和周期推力；保留同条件启动随机化，不加入物体。
- 每条实际统计 `min(N - 1, 1000)` 步；排除结束重置样本，长片段明确标截断。
- 先片段内平均，再片段等权平均；失败片段不剔除，不挑选最好的一次作为总体结论。
- 先后校验权重、索引、全部 NPZ 和执行源码；跨组核对实际输入内容和协议一致。
- 512 环境改变了分组与随机化分配，因此不能直接和此前 128 环境结果当成前后提升；
  本轮候选与官方都重新测量。

本轮是同分布离线参考的单种子比较，不代表在线任意目标、抗扰动或完整论文验收。

## 配对比较器与首轮结论

`scripts/compare_mask_evaluations.py` 只接受完整的 schema 4 八模式配对报告。它先检查
模式名和 active 连杆表必须严格对应当前 G1 ScaleBFM 八组配置，并要求同一侧八份报告
使用同一 checkpoint 内容；两侧的 task、protocol、seed、环境数、步长、设备及软件版本
必须一致。motion index、全部 motion 内容和 Python 执行源码的 SHA-256 也必须一致，且
每条 clip 的 ID、名称、源帧数、评估步数和截断状态逐一相同。报告还必须声明输入在运行
前后未变、训练更新数为零，数值有限且满足基本物理统计约束。比较结果记录比较器自身
源码 SHA-256，输出路径已存在时拒绝覆盖。

比较器不采信报告中的汇总值，而从每条 motion 重新计算片段等权均值和 0.5 m active/all
max-link 跟踪通过率。逐模式 gate 要求候选 active 位置均值和姿态均值不比官方大超过
`1e-6`，且 active/all 通过率均不降低；八种模式全部通过才得到 overall no-regression。

2026-09-07 的正式配对结果见 `SCALEBFM_MASK_RESULTS_20260907.md` 和
`logs/evaluations/20260907_mask_paired_comparison_v1.json`。旧本地模型为候选，结果是
**0/8 模式通过**，所以未通过 no-regression gate，不能替换官方参考，也不能据此宣称
ScaleBFM 已达标。每份报告均包含同一组 962 条 clip、342,637 个有效步和 81 条截断片段。
这是固定 mask、单 seed 的离线参考跟踪比较；0.5 m 通过率不是跌倒率或任务成功率。
本轮使用 512 环境重新配对测量，与旧 128 环境结果不是可直接相减的前后对照。

## 选择与后续训练

先比较每种 mask 的位置、姿态误差及 active/all 跟踪通过率，不以总体平均改善掩盖
单模式退步。一个候选只有在这些逐模式指标都不弱于官方对照时，才有资格替换基线；
仍需更多种子和完整长片段检查才能声称可靠提升。0.5 m 仅是失配诊断线，不是精度达标线。

中间权重筛选和超参数调整会利用验证集，因此其后“全新数据泛化”还需要新的独立测试集。
不把原验证失败片段挪入训练集，也不把选择后的验证结果称为完全未见的测试结果。

续训审计发现：`runner.load()` 恢复 optimizer 的学习率，但 PPO 的学习率标量仍来自
当前配置；adaptive 分支又会把标量写回 optimizer。旧 probe/final 的两个 optimizer
保存值都是 1e-5，而配置默认 actor 为 2e-5、critic 为 1e-3。这是需要控制的恢复风险，
尚未证明是此前退步的唯一原因。现已修复为恢复 optimizer 后同步 PPO 标量，并用真实
CPU PPO/Adam 保存加载测试复现失败、修复后通过；`load_optimizer=False` 的独立推理
路径不改变。新实验必须记录实际学习率、探索噪声和独立评估结果。

注意官方文件 `model_22200.pt` 的内部迭代是 22199，其两个 Adam 保存学习率均为
`0.0008649755618534982`，不是本地 probe 的 1e-5。恢复该 optimizer 后设置 fixed
schedule 会固定为这个保存值；单改配置学习率不能覆盖已恢复的 optimizer。

新训练使用独立 run，不覆盖旧 checkpoint；先有限更新，再完整对照，未通过则不替换
已验证参考。所有模拟任务使用有时限的独立进程组，结束后检查无残留。

从官方 checkpoint 启动的 5-update、`entropy_coef=0.001`、adaptive schedule scout
已经完成：实际执行 40,960 environment steps，其八模式正式比较同样为 **0/8**，因此
不替换官方参考，也不延长该 run。TensorBoard 在每次 update 结束时记录 actor/critic
learning rate 均为 `1e-5`；checkpoint 中策略 `std` 均值从官方的 `0.347084` 变为
`0.346689`。这些训练量和参数变化不能证明跟踪改善，最终仍以独立八模式评估为准。
详细结果和复现命令见 `SCALEBFM_MASK_RESULTS_20260907.md`。

首步学习率的两模式单变量诊断也已完成：低初始 LR 减轻了高 LR 单步更新的退步，
但相对未更新官方仍各有一项指标未满足无退步条件，见
[`SCALEBFM_RESUME_LR_ABLATION_20260907.md`](SCALEBFM_RESUME_LR_ABLATION_20260907.md)；
它不是完整八模式 gate。旧模型、原始数据索引和既有报告继续保留。
