# UMR 同源对照暂停交接

2026-09-08 23:09:44（Asia/Shanghai），用户要求“明天继续”。已停止本轮数据批处理及其整个进程组，未启动后台续跑。

## 实际断点

- 批次：`local/umr_amass_pilot_20260908a/`。
- 预先固定 40 条：五个数据源各 6 train + 2 development validation，源帧 0 开始、最多 5 秒、50 Hz。
- **30/40 已完成**真实 SMPL-X → UMR → 同源配对：ACCAD 8、BMLmovi 8、BMLrub 8、CNRS 6；KIT 本批尚未开始。不是全库重做完成。
- 第 31 条 `CNRS/288/-01_L_1_stageii`：表面准备已完成，UMR retarget 阶段因用户暂停中断；该条不是质量失败，不能当成完整产物。
- 剩余 10 条包括上述中断条、另 1 条 CNRS、8 条 KIT。完整 `paired_manifest.json` 尚未发布。
- 首条两侧 IsaacLab 打包 smoke 已通过：native 138 → packed 137 帧，关节/根位置插值误差 0；产物在 `packed_smoke/`，不能冒充完整 80 个正式包。
- 首条 Stage I 缓存 miss/hit 实测 qpos 最大差 0，缓存哈希与完整性复核一致。
- **本轮正式 16 组冻结策略评估尚未开始，PPO 未开始，数据/默认权重未晋升。**

## 进程与产物

以下本轮用户服务均已核对 inactive/dead、MainPID=0：

- `bfm-umr-pilot-amass-20260908a.service`
- `bfm-umr-cache-hit-20260908a.service`
- `bfm-umr-package-smoke-20260908a.service`
- `bfm-umr-code-tests-20260908a.service`

停止后无运行中的 `bfm-*` 用户服务，GPU compute-app 查询为空。未删除任何数据、日志、缓存或失败/中断产物。
主状态已从遗留的 RUNNING 标注为 `PAUSED_BY_USER`，同时记录原状态 SHA 和用户停止原因；不能声称整批 COMPLETE。

## 明天恢复顺序

1. 读 `plan.json` / `status.json`，复核原始索引、源数据、模型、7 个生产代码文件和 UMR pin 的保护哈希。
2. 对已完成 30 条检查 pair receipt、sampling proof、双方 PKL 及 UMR receipt；仅复用核验通过的文件。保留原 origin/split，不筛掉困难动作。
3. **现有 `run_umr_amass_pilot.py` 拒绝已有输出目录，不支持直接断点重入。** 需增加独立、可校验的恢复入口，不修改已冻结的生产代码；中断条的已有文件先归档到本批明确的中断尝试目录，避免覆盖或把半成品当缓存命中。完整缓存条目可复用，半成品不得视为有效。
4. 补齐全部 40 条后才发布完整配对清单；同一既有 IsaacLab 一次打包 `input/{baseline,candidate}` 的全部 80 个 PKL 到新 `packed/`。继续 `loader_workers=1`，不重装 IsaacLab。
5. `scripts/evaluate_umr_pairs.py --manifest <batch>/paired_manifest.json --packed-root <batch>/packed --output <新评估目录>` 先只读预检；实际执行加 `--execute`，必须在 `bfm-umr-eval-*` 的有界 control-group 服务内。默认几何解释器为原重定向 Python 3.11。
6. 全部 16 组完成后用 `scripts/summarize_umr_pairs.py --batch <batch> --evaluation <评估目录>` 输出经核验的汇总，再更新技术路线和 README。此工具拒绝部分结果，不会启动训练或自动替换数据。

服务都须设置 `KillMode=control-group`、`Restart=no`、时间/内存/task 上限；结束检查 MainPID=0。不要杀其他项目的进程。

## 代码与证据

- 批处理：`scripts/run_umr_amass_pilot.py`；配对：`scripts/umr_pair_dataset.py`。
- 适配器/缓存：`scripts/umr_smplx_source.py`；UMR pin：`0aa1855fe4f65a73681ffbd1d9f95ab1c2bad9ca`，非官方实现。
- 冻结评估脚本 SHA：`48d04439d4c8a4a7ccec6ce75a7b43b007fc22cdc49c23561b81cd3e8036f0dc`。
- 已运行：Isaac 环境选定测试 798 passed（不含当时仍在开发的 evaluator/summary）；native 专用测试 15 passed；evaluator 12 项通过并验收首对真实数据；summary 12 项通过。不要把未运行的合并全套数字写成实测。
- 三套现有解释器与数据许可边界见 `docs/DATA_REFRESH_AND_SCALEBFM_COMPLETION.md`。

## 判断边界

上一批 0/8 仅指 **高度修正候选**，不能当作本批真实 UMR 的结果。
本批旧流程采用优化中性 shape，UMR 采用源 shape；根平移尺度和高度归一化也不同。跟踪更容易不等于人体动作保真更高，须共同查看几何、速度和动作幅度。
官方默认权重与 7,174 条原训练动作保留。ScaleBFM 的数据升级、行为学习改进、Sim2Sim、任务及实物验收仍未全部完成。

## 2026-09-09 恢复入口

已新增独立 `scripts/resume_umr_amass_pilot.py`，没有改动原冻结生产脚本。默认只读：

```bash
/home/sw/.cache/bfm/scaleretarget-py311/bin/python scripts/resume_umr_amass_pilot.py \
  --batch local/umr_amass_pilot_20260908a --run-id 20260909a
```

实际执行需在有界 `bfm-umr-pilot-*` control-group 服务内加 `--execute`。入口检查旧服务已停止，
对整个恢复过程加排他锁，保留旧状态/计划与中断产物到 `resume_history/<run-id>/`，不删除或覆盖旧产物。
完整缓存复用；中断缓存另存，不当作命中。正式状态仍为每条两个成功阶段，中断历史不冒充第三次正式任务。
保护检查通过后才发布完整配对清单；若收到停止信号会清理子进程并记录暂停状态。

本次恢复前独立审计：134/134 原计划保护文件、30 对共每侧 5,847 native 帧、23/23 完整缓存均通过；
原批前 30 条累计 7 个 cache hit、0 次 QP failure。主恢复入口另外核验了 462 个来源/代码/产物文件。
这些是完整性和求解执行证据，不是数据质量/物理跟踪通过证据。

### 本次恢复完成与打包端点问题

2026-09-09 12:23:33，剩余 10 条完成，40/40 清单已发布，原数据/代码前后核验通过，恢复服务已退出。
首次完整 80 包使用冻结旧打包器，`packed/` 每侧实际 8,055 帧，而清单预期 8,053；严格预检拒绝进入评估。
两条动作的双侧各多一帧：BMLmovi `Subject_41_F_18` 为 202 而非 201，KIT `walking_run09` 为 227 而非 226。

原因是 binary64 终点运算，不是安装错误：`201 * (1 / 50)` 为 `4.0200000000000005`，
`201 / 50` 为 `4.02`；两套既有 Torch 均复现不同的 `arange` 长度。更一般地，即使用除法，
在 3～251 个源样本中仍有 10 种长度受浮点向上取整影响；本批 40 条恰不含这 10 种长度。

新增显式入口 `ScaleTrack/scripts/pretrain/data_process/package_paired_motions.py`：只接收本批 50 Hz
配对原生数据，按整数规定 N 个 inclusive 源样本输出 N-1 个 half-open 样本；重算速度，再由原
IsaacLab 模拟器代码生成全部 link 状态。保留冻结旧打包器与原始 `packed/`，不是事后裁剪 NPZ。
新打包指纹绑定原打包指纹、端点规则和新入口代码 SHA；输出使用独立新目录。
下一代全库配对生成器还需统一这一整数时钟协议，不能声称冻结旧 helper 已对所有长度修复。

### 本轮已收尾

正式使用 `packed_paired_clock/`，80/80 预检与全身 FK 通过；16/16 冻结策略评估正常完成，
联合不退步 0/8，未训练/晋升/全量替换。全部本轮服务已退出。此批已经 COMPLETE，
**不再调用恢复入口重入此批**；下个环节应按 [正式结果与下一步](UMR_PAIRED_EVALUATION_20260909.md)
创建新的有版本对照。原 `packed/` 仍是端点不合格的保留诊断数据，不用于训练或最终评估。
