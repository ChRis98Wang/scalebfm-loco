# UMR 对比与本地试验边界（2026-09-08）

本文记录 Hanyang Cao 等人的 Unified Motion Retargeting（UMR）论文、当前可见的
非官方复现，以及本仓库 ScaleRetarget 流水线之间已经确认的边界。
截至 2026-09-08 20:32 CST，已在隔离环境跑通非官方仓内 5 秒 BVH → G1 样例、
生成视频，并完成独立结构/碰撞审计。结论为 **样例流程成功，接触质量与物理策略尚未达标或验证**；
没有接入现有 AMASS 数据、训练索引或 BFM 模型。实测证据见第 7 节。

## 1. 身份与开源状态

- 论文为 Hanyang Cao 等人的 *Unified Motion Retargeting for Humanoids with Learned
  Point Cloud Correspondence*，见 [arXiv:2609.02134](https://arxiv.org/abs/2609.02134)
  和 [HTML 全文](https://arxiv.org/html/2609.02134)。arXiv 首次提交日期为
  2026-09-02。
- 截至 2026-09-08，论文页面、正文及作者/机构相关检索中均未确认到作者发布的官方
  项目页、官方代码仓、官方预训练权重或官方资产包。因此目前不能进行“官方 UMR
  代码对官方 ScaleBFM 代码”的复现实验，也不能把第三方运行结果归因给论文作者。
- arXiv 页面所示 CC BY 4.0 是论文内容的许可，不代表尚未发布的软件、模型权重或
  资产获得相同许可。
- 当前可运行候选是
  [`longchengzhuo/Unified-Motion-Retargeting`](https://github.com/longchengzhuo/Unified-Motion-Retargeting/tree/0aa1855fe4f65a73681ffbd1d9f95ab1c2bad9ca)。
  其 README 明确声明 **Unofficial**、属于独立复现，且与原作者无隶属或背书关系。
  试验和结果中必须始终称其为“非官方 UMR 复现”，不能简称为官方 UMR 实现。
- 非官方仓代码声明 [MIT License](https://github.com/longchengzhuo/Unified-Motion-Retargeting/blob/0aa1855fe4f65a73681ffbd1d9f95ab1c2bad9ca/LICENSE)；
  G1 资产另有独立的
  [BSD-3-Clause 文件](https://github.com/longchengzhuo/Unified-Motion-Retargeting/blob/0aa1855fe4f65a73681ffbd1d9f95ab1c2bad9ca/assets/robots/g1_description/LICENSE)，
  锁定 README 称其 vendored from `unitree_ros`。动作数据的来源、许可与使用限制仍需
  逐项确认，不能仅凭代码仓许可证推定动作数据和机器人资产可再分发。

非官方仓的网页搜索缓存曾与 `master` 最新 README 不一致。后续报告必须记录实际检出的
commit SHA，并以该 commit 的原始文件内容为证据，不能混用搜索摘要或其他 commit 的功能描述。

## 2. 与本地 ScaleRetarget 的定位差异

| 维度 | 本地 ScaleRetarget / ScaleBFM 流水线 | UMR 论文 / 当前非官方复现 |
| --- | --- | --- |
| 核心目标 | 将 AMASS SMPL-X 人体动作经形体匹配和约束 IK 重定向到 G1，再打包为 ScaleTrack 可训练参考 | 论文以学习到的人体—机器人点云对应建立约束，再进行逐帧运动学优化；非官方仓是对此思路的独立实现 |
| 本地输入 | AMASS Stage-II：`root_orient`、`pose_body`、`trans`、`betas`、`gender`、`mocap_frame_rate` | 论文对应阶段依赖人体网格/点云表示；非官方仓另有自己的 BVH 人体构建与预处理入口 |
| 本地中间输出 | joblib pickle：`root_pos (N,3)`、`root_rot (N,4, xyzw)`、`dof_pos (N,29)`、`fps` | 必须以锁定 commit 的实际输出结构为准，不能假定天然满足本地 pickle 契约 |
| ScaleTrack 入口 | 将 pickle 重采样并用 G1 FK 打包为 50 Hz、29 关节、30 刚体的 v3 NPZ；磁盘四元数为 `wxyz` | 非官方输出若要进入 ScaleTrack，仍需显式的格式、关节顺序、四元数、帧率、有限值和 FK 适配/校验 |
| 物理与学习 | ScaleRetarget 是参考动作生成；ScaleTrack 的 PPO 才负责物理仿真跟踪 | UMR 论文/非官方重定向输出本身不等于机器人已在物理仿真中稳定执行，也不等于下游 PPO 已训练或评估 |

两条路线都可能生成 G1 运动学参考，但它们不是可直接互换的“同一个 retargeter”。UMR 的
点云对应及其优化目标，与本地 SMPL-X 形体拟合加 GMR/Mink IK 的建模假设不同；比较时需把
输入、机器人模型、坐标系、关节定义、帧率和后处理锁定，不能只比较最终视频观感。

本地 ScaleRetarget 的默认 AMASS 路径先优化 SMPL-X 静态形体，再用稀疏加权
Mink `FrameTask` 逐帧 IK；关节限位启用，速度限位可选但默认关闭，实际 IK 中未接入
接触/自碰撞优化。这不代表机器人资产没有碰撞几何。当前非官方 UMR 则学习一次
T-pose 密集表面点对应，再逐帧优化点位置、法线和地面接触向量，另加地面/自碰撞约束。
它改变的是参考动作的生成方式，不替代 ScaleBFM 的 Transformer/PPO 行为学习。

## 3. 接入 ScaleBFM 的硬门槛

### 3.1 输入不能直接替换

本地 AMASS 原始文件是参数化 SMPL-X Stage-II 数据，不是已经采样好的 UMR 人体点云；
UMR 论文的网格/点云输入也不是本地 loader 的 `root_orient + pose_body + trans + betas`
接口。若使用论文式输入，需要先以相同 SMPL-X 模型和形体参数生成逐帧人体网格/点云，
并固定单位、坐标轴、根定义和拓扑/采样规则。

非官方仓虽然提供 BVH 路径，但 BVH 的程序化人体表示不能被当作 AMASS SMPL-X 网格的
无损替代，也不能把现有 AMASS `.npz` 直接改扩展名或字段后送入。若试验采用仓内 BVH，
它只能验证该非官方实现自身的示例路径；若要比较本地 AMASS，则需要独立、可审计的
AMASS/SMPL-X 适配，并保留原始动作与派生输入的内容哈希。

### 3.2 输出必须满足本地契约

锁定 commit 的 README 所列原生 pickle 使用 `root_trans (T,3)`、
`root_rot (T,4, xyzw)`、`dof (T,29)`、`dof_full (T,29)`、`qpos (T,36)`、`fps`
和 `dof_names` 等字段；其中 `qpos` 的 MuJoCo 四元数仍为 `wxyz`。它与本地 loader 所需的
`root_pos`、`root_rot`、`dof_pos`、`fps` 并非同一字段协议，必须经过独立 adapter，不能把
原生 pickle 直接放进 ScaleRetarget 输出目录。

任何 UMR 候选结果进入现有 `package_motions.py` 前，至少必须转换并验证：

- `root_pos`：`(N, 3)`，单位为米，坐标系和 G1 根节点语义明确；
- `root_rot`：`(N, 4)`，进入本地 pickle 时明确为 `xyzw`，单位四元数连续且无 NaN/Inf；
- `dof_pos`：`(N, 29)`，单位为 rad，严格按本地 G1 29-DoF `ROBOT_JOINT_NAMES` 排列；
- `fps`：有限正数；进入 ScaleTrack 训练前必须由现有打包器重采样为严格 50 Hz，而不是
  只修改元数据；
- 长度一致：根位置、根姿态和关节角具有相同帧数，且至少满足打包器最短长度要求；
- 机器人模型一致：URDF/MJCF 的关节轴、零位、限制、frame 和 link 定义与本地模型逐项映射；
- 打包后仍通过 v3 契约：29 关节、30 刚体、磁盘 `wxyz`、有限值、速度和独立根 FK 一致性；
- 来源可追溯：保存 UMR commit、配置、输入内容哈希、机器人资产哈希和转换器哈希，不能将
  缺少 provenance 的输出混入现有受管索引。

关节数量相同并不足以证明顺序或语义相同；四元数四列形状相同也不足以区分 `xyzw` 与
`wxyz`。这两项必须通过名称映射和独立 FK 数值检查确认。

## 4. 非官方实现当前能力边界

以实际试验锁定的 commit 为最终依据。2026-09-08 对上述锁定 README/配置的只读核对显示，
该非官方实现的任务场景仍以 `GroundPlane` 接触为主，G1 自碰撞可配置且锁定配置默认开启；
未确认物体/场景交互任务，也未包含与本仓库等价的下游 ScaleTrack PPO 训练和八种 mask
评估。不要引用旧搜索缓存中“没有 self-collision”的说法，也不要从配置开关推导尚未实际
运行验证的接触质量。

源码审查还确认以下限制：

- `retarget.max_velocity: 12.0` 只出现在配置中，未传入 Stage II 的 limits，实际不生效。
- 默认 `tpose_offset=1.0` 是此复现的偏置补偿；默认 box trust region 也不能冒称逐式等同论文。
- 自带报告没有足滑或独立自碰撞统计；`foot.contact_ratio` 实际是全身最低表面高度
  小于 2 cm 的几何比例，不是脚底真实接触力/接触占空比。
- `solve_failures` 不包含被丢弃返回值的首帧 warmup，0 不能证明 warmup 无失败。
- PD 回放只用基座高度 `<0.35 m` 判跌倒，没有平衡策略；不能用它替代 ScaleTrack 评估。

因此，非官方试验能回答的是“该 commit 是否能在其声明输入上生成有限、结构合理的 G1
运动学结果，以及这些结果能否经过本地严格适配”。它不能单独证明：

- G1 在 IsaacLab 物理仿真中能够稳定跟随；
- 在线 waypoint、物体接触或 loco-manipulation 能力；
- 对完整 AMASS/KIT 分布的泛化；
- 与论文表格一致的实现保真度或性能。

## 5. 本地试验隔离与判定边界

本次候选源码目录为 `/home/sw/bfm/external/umr_trial_20260908`，其 `origin` 为
`https://github.com/longchengzhuo/Unified-Motion-Retargeting.git`，已锁定 HEAD
`0aa1855fe4f65a73681ffbd1d9f95ab1c2bad9ca`；核对时工作树为空。独立 Python 环境为
`/home/sw/bfm/local/umr_trial_20260908/venv`。它们不得覆盖现有 ScaleRetarget 源码、环境、
数据、索引、checkpoint 或 IsaacLab 安装。实际依赖、命令、输入哈希和退出码已记录于
`local/umr_trial_20260908/receipt.json`，依赖安装和运行详情见第 7 节。

建议按以下阶段逐级判定，前一阶段失败时不把后续结果写成通过：

1. **身份与依赖**：记录 remote、commit SHA、dirty 状态、许可证文件、Python 与关键依赖；
   不使用 README 搜索缓存代替锁定源码。
2. **仓内原生样例**：只验证非官方仓自己的输入能否完成，不触碰本地 ScaleRetarget 的 MJCF；
   程序可能改写试验 clone 内的 MJCF，必须保存运行前后 SHA 和 diff。
3. **结构审计**：检查全部输出字段、shape、dtype、有限值、帧数、fps、四元数模长与关节限制。
4. **AMASS 适配**：选择少量明确许可、已在本地 ScaleRetarget 跑通的相同 Stage-II clip，
   从同一源生成 UMR 所需网格/点云；不得用仓内 BVH 样例冒充同源 AMASS 对照。
5. **G1 映射与 50 Hz 打包**：通过显式名称表和独立 FK 校验后，生成隔离的 v3 NPZ 和新 YAML；
   不写入现有训练/验证 manifest。
6. **只读回放与物理跟踪**：先做运动学可视检查，再用同一 checkpoint、mask、seed、步数和
   终止规则做 ScaleTrack 物理评估；任何 reset、截断或失败片段必须如实计入。

本轮已完成上述 1–3 的仓内样例检查，但发现接触质量问题；4–6 仍未运行。仓内示例完成
不代表 AMASS 接入、50 Hz v3 打包或物理策略评估通过。

## 6. 对比指标与不可直接比较项

UMR 的对应/优化误差、本地 ScaleRetarget 的 IK/FK 误差，以及 ScaleTrack PPO 的
`active_position_error`、姿态误差、最大 link 阈值通过率和 episode termination 属于不同
阶段、不同定义。没有统一输入、frame、采样率、机器人模型和度量代码时，数值不能直接横排，
尤其不能把 UMR 的运动学拟合误差与本地 PPO 跟踪误差直接比较。

同源候选对比至少应分别报告：

- 输入层：clip 名称、源 SHA、原始/处理后帧数、原始 fps、人体模型及形体参数；
- 运动学层：根位置/姿态连续性、关节限位违规、足滑、地面穿透、接触切换、自碰撞统计，
  以及相对于同一定义人体目标的关键点位置/姿态误差；
- 接口层：29 关节名称映射、根 frame、四元数顺序、重采样方法、50 Hz v3/FK 审计结果；
- 物理层：同 checkpoint、同八种 mask、同 seed/horizon 下的 active/all body 位置与姿态指标、
  最大 link 阈值通过率、termination、有效步数和每 motion 原始结果；
- 工程层：单 clip 耗时、峰值 CPU/GPU/内存、失败率、重试、被修改文件和输出可复现性。

视频只作为辅助材料，不能替代实际机器人状态和逐帧误差。仅选成功片段、丢弃失败 motion，
或用不同 PPO checkpoint 分别评估两套参考，均不能称为公平比较。

## 7. 实测：仓内 5 秒样例

### 7.1 环境与运行证据

- Python 3.10.20；MuJoCo 3.9.0、Mink 1.1.1、Clarabel 0.11.1、NumPy 2.2.6、
  PyTorch 2.12.0+cu130，RTX 5080。根依赖按上游 `environment.yml` 锁定，未降版本。
  `uv pip check` 检查 59 个包全部兼容；完整版本表在 `receipt.json`。
- 安装经历两次受控网络重试：首次未继承终端代理；补齐代理后安装器下载仍慢。
  随后用 curl 下载缺少的 8 个原版 wheel，按 PyPI SHA-256/大小校验，再用 uv 离线安装成功。
  下载来源、哈希在 `local/umr_trial_20260908/wheels/manifest.json`。没有修改系统代理。
- 输入 `data/walk_slow.bvh`，SHA-256
  `9afe1879561f3593ad19da5d14bcf38e760df6e4f4b8fdf8ee9f7d5e8d9d6997`。
  自动跳过 1 个合成标定帧后，从第 0 秒取 5 秒；50 Hz 含两端点得到 **251 个姿态样本**。
- 默认 Stage I 2500 epochs，GPU 学习耗时 17.89 s；setup 合计 25.88 s。
  Stage II 约 5.5 s / 46.02 FPS。retarget 子命令总墙钟 37.10 s，首次报告含渲染 4.52 s。
  两者连同 GPU 检查总计 43.27 s，不含下载/安装；完整报告补跑 2.46 s。
- trial 使用独立 systemd 用户 cgroup，10 分钟硬上限、16 GiB 内存上限、无自动重启、
  `KillMode=control-group`；关闭 viewer/多进程，数值库线程限为 4。这里的 16 GiB 是限额，
  不是实测峰值；receipt 的最大子进程 RSS 为 4,531,516 KiB，未采集连续 GPU 峰值。
- 代码 MIT、G1 资产 BSD-3-Clause；BVH 的独立数据许可仍未确认，不把样例视频/动作推送或
  宣称可无条件再分发。

实际命令参数如下（已在上述受控 cgroup 内执行，完整 argv/退出码保存在 receipt）：

```bash
cd /home/sw/bfm/external/umr_trial_20260908
/home/sw/bfm/local/umr_trial_20260908/venv/bin/python scripts/retarget.py \
  --motion_file data/walk_slow.bvh --human xsens --robot unitree_g1 \
  --tgt_fps 50 --start 0 --duration 5 --device cuda --no_viewer \
  --save_path /home/sw/bfm/local/umr_trial_20260908/smoke_a

/home/sw/bfm/local/umr_trial_20260908/venv/bin/python scripts/report.py \
  --motion_file data/walk_slow.bvh --human xsens --robot unitree_g1 \
  --save_path /home/sw/bfm/local/umr_trial_20260908/smoke_a \
  --replay --replay_seconds 5
```

以上命令继承 `PYTHONNOUSERSITE=1`、`MUJOCO_GL=egl` 和线程限额。
已存在的 retarget 输出会被上游默认跳过，不能将二次运行的 skip 算成新的重复实验。

### 7.2 结果与质量限制

| 检查 | 本机实测 | 解释 |
| --- | --- | --- |
| 原生输出结构 | float64 `qpos=(251,36)`，29 DoF，50 Hz，全有限值 | 仅原生格式，不是 ScaleTrack v3 |
| 四元数 / PKL 一致性 | 最大单位模长误差 `1.11e-15`；PKL 数组与 NPZ 一致 | NPZ qpos 为 wxyz，PKL root_rot 为 xyzw |
| 关节限位 | 29 关节，越限 0 | 不包含真实执行误差 |
| 点匹配误差 | mean 13.151 mm，median 13.332 mm，p95 17.502 mm | 不可与 BFM PPO 身体位置误差横比 |
| 法线误差 | mean 5.783° | 表面匹配，不是控制稳定性 |
| 点云对应覆盖 | 双向 Chamfer 17.78 / 17.03 mm，2 cm 覆盖率 69.8% | 一次 T-pose 对应学习 |
| QP / 地面约束 | 报告失败 0；每帧 floor_rows 为 148–348 | warmup 失败不在计数中；约束行数不等于接触数 |
| 最低表面穿地 | 此实现的几何统计为 0 mm | 不是实物接触标定，也不证明脚不滑 |
| 独立自碰撞 | **52/251 帧（20.72%），最大 0.961 mm** | 当前启用 MuJoCo 碰撞对；左髋 pitch link 与左腕 yaw link |
| 左/右足高度相关性 | 0.780 / **-0.425**；RMSE 18.68 / 17.75 mm | 右脚抬脚时序/高度未被很好保留 |
| 低足高条件下 XY 速度代理 | 左/右均值 0.0327 / 0.0550 m/s；p95 0.0320 / 0.2595 m/s | 足部 body 原点速度，含落脚/转动，不是严格足底足滑指标 |
| 无平衡器 PD 回放 | **3.68 s 触发跌倒高度门槛**，结束根高 0.341 m | 不是已训练 BFM 策略的失败率；不能证明下游一定无法跟踪 |
| 结构审计状态 | `AUDIT_COMPLETE`，issues 空，summary 未发现非有限值 | `physical_pass=null`，未授予物理质量 PASS |

首版报告曾使用 `--max_frames 250`，而产物实际为 251 帧。上游只截取部分验证数组，
点/法线误差仍取全长，因此已保留首版在 `first_report_250_frames/`，并无截断重跑全部
251 帧，独立审计验证所有 metrics 数组帧数对齐。PD 单独按 `5 s × 50 Hz = 250` 次步进
计算，这是控制步数，不是漏掉一帧结构审计。

视频已解码检查：1120×594、H.264、25 FPS、125 帧、5.0 s；抽查第 0/50/100 帧。
它展示人体与 G1 的**运动学参考**，不是物理策略播放或真实机器人演示。

### 7.3 文件、隔离与清理

- [本机视频](../local/umr_trial_20260908/smoke_a/xsens_to_unitree_g1/walk_slow.mp4)
- [上游原始报告（全 251 帧）](../local/umr_trial_20260908/smoke_a/xsens_to_unitree_g1/walk_slow/report.md)
- [独立审计 JSON](../local/umr_trial_20260908/independent_audit.json)
- [版本/哈希/命令 receipt](../local/umr_trial_20260908/receipt.json)
- [可复查审计脚本](../local/umr_trial_20260908/audit_outputs.py)

原始 XML 已备份为 `robot_xml_before.xml`。上游存在改写 XML 的代码路径，但此固定 commit
的实际运行前后 XML SHA 均为
`bd5161b43d9212266f3646fd91c7ba877ca0c5acceb2b34173f6ad8454d35e5a`，clone 仍 clean；
本次没有实际内容差异，不能把静态风险写成“已修改”。独立审计也确认输入产物/XML 哈希未变。

源码约 46 MiB，独立环境表观占用 4.9 GiB，下载 wheel 约 780 MiB，样例输出 2.7 MiB；
uv 环境可能与已有缓存硬链接，以上不是新增磁盘块的精确计量。源码与环境/输出已加入忽略范围，
没有 vendor 入本仓发布内容；未修改现有 IsaacLab、训练数据/索引或 checkpoint。

截至 20:32 CST，`bfm-umr-*` 试验 units 全部退出且已回收，无 UMR 残留计算/渲染进程。
另一个 `rl100_playground` 项目的 GPU 进程保持不动。没有启动 GUI、BFM 训练或新的八模式评估。

## 8. 下一步建议（未执行）

保留现有 ScaleBFM 主线，不用这个样例直接替换 AMASS 重定向结果。
先选少量同源 AMASS clip 建立 SMPL-X 输入适配，再完成 G1 关节/模型映射和 50 Hz v3 独立 FK
校验；随后用同一 BFM checkpoint 做分层对照。优先修查右脚接触保持和轻微自碰撞，再决定
是否扩大数据处理。本轮 **AMASS 适配、ScaleTrack 打包、物理策略跟踪和八 mask 对照均为 NOT_RUN**。

后续已完成同一 Xsens BVH 三个固定窗口的本地 ScaleRetarget 对照，见
[同源实测报告](UMR_LOCAL_RETARGET_COMPARISON_20260908.md)。该结果仍不代表 AMASS 或物理策略验收。
