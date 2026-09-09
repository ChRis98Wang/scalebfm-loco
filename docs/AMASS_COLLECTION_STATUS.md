# AMASS 收集与重定向进度

更新日期：2026-09-05 11:44（Asia/Shanghai）。已完整下载的四个数据集全部处理完成；
CMU 下载未完成，不计入本轮完成数量。

后续进展：2026-09-05 15:37 已从其中筛选 128 条训练动作和 32 条按受试者隔离的
验证动作，启动首轮小规模微调。配置、运行状态与停止方式见
[小规模训练记录](AMASS_SMALL_TRAINING_RUN.md)。下文保留数据准备阶段的审计快照。

2026-09-07 新数据：用户提供的 `KIT.tar.bz2` 已通过完整归档遍历，含 4232 个
Stage-II 候选动作文件，抽样为 SMPL-X 格式。压缩包约 3.39 GiB、解压内容约
9.88 GiB。随后 4231 条有效输入已全部重定向、打包与审计，1 条单帧输入保留但排除。
新 v2 全量/去重池为 9487/9484 条，训练池扩至 7174 条，其中 KIT 2927 条；
KIT 留出 789 条单独保留，旧 962 条验证基线不变。已完成一次真实训练接入验收，
其中 55 条 KIT 参与 3520 个环境步；这不是所有新动作已经充分训练或效果达标。
下方四数据集表仍是原 2026-09-05 审计快照，不包含新增 KIT。
详见 [KIT 入库记录](KIT_INGESTION_20260907.md) 和
[最初归档检查记录](KIT_DATA_AUDIT_20260907.md)。

## 已入库原始数据

| 数据集 | Stage-II 动作 | 源帧 | 时长（分钟） | 源帧率 |
| --- | ---: | ---: | ---: | ---: |
| ACCAD | 252 | 192,553 | 26.7435 | 120 Hz |
| BMLmovi | 1,864 | 1,255,584 | 174.3867 | 120 Hz |
| CNRS | 79 | 57,639 | 9.6065 | 100 Hz |
| BMLrub | 3,061 | 3,763,367 | 522.6899 | 120 Hz |
| 合计 | 5,256 | 5,269,143 | 733.4265 | — |

原始动作位于 `ScaleRetarget/dataset/amass/<数据集>/`，官方下载包位于
`ScaleRetarget/dataset/amass/_archives/`。BMLmovi 的一条上游截断文件已经从 CRC
正确的核心数组中恢复，原文件与恢复记录分别保存在 `_quarantine/` 和 `_repairs/`。

11:39 检查时，`/home/sw/Downloads/CMU.tar.bz2` 仍为零字节占位文件，
`CMU.dEflRR8-.tar.bz2.part` 为 10,288,573 字节，最后增长时间为 10:37。
本轮曾观察到临时文件重新变小后再增长，但目前下载已停止增长。Browser 技能的连接
检查未发现可连接浏览器，因此未能代为恢复；需要在 Firefox 下载列表中手动恢复。
没有读取或入库这个不完整归档，也没有留下常驻下载监听器。

## 全量处理结果

| 数据集 | 动作数 | 50 Hz 帧数 | 状态 | 输出索引 |
| --- | ---: | ---: | --- | --- |
| ACCAD | 252 | 80,246 | 全量审计通过 | [ACCAD 索引](../ScaleRetarget/retargeted_dataset/amass_accad_all_v1.yaml) |
| BMLmovi | 1,864 | 523,065 | 全量审计通过 | [BMLmovi 索引](../ScaleRetarget/retargeted_dataset/amass_bmlmovi_all_v1.yaml) |
| CNRS | 79 | 28,803 | 全量审计通过 | [CNRS 索引](../ScaleRetarget/retargeted_dataset/amass_cnrs_all_v1.yaml) |
| BMLrub | 3,061 | 1,568,221 | 全量审计通过 | [BMLrub 索引](../ScaleRetarget/retargeted_dataset/amass_bmlrub_all_v1.yaml) |
| 合计 | 5,256 | 2,200,335 | 无遗漏 | — |

全量生成 1,324,095 个重定向帧和 2,200,335 个 50 Hz 打包帧。打包文件共
4,025,244,756 字节（约 4.03 GB / 3.75 GiB），重定向 `.pkl` 共 383,393,288 字节。
两者合计约 4.41 GB，不包含原始归档、原始解压数据和日志。

- [全量索引：5,256 条](../ScaleRetarget/retargeted_dataset/amass_collected_all_v1.yaml)
- [去重素材索引：5,253 条](../ScaleRetarget/retargeted_dataset/amass_collected_unique_v1.yaml)

索引键使用 `数据集/原始相对动作名`，值为打包文件的绝对路径；不包含旧的单动作
冒烟副本。原 331 条的 `amass_collected_batch_v1.yaml` 保留为历史快照，不作为全量入口。

审计发现 ACCAD 中有三对整个源文件 SHA-256 完全相同的动作：`C4_-_run_to_walk`、
`B17_-__Walk_to_hop_to_walk`、`B21_-__put_down_box_to_walk` 各自与 `_a` 版本重复。
去重索引按完整源文件哈希分组，保留字典序最先的键，共 5,253 条、2,199,589 帧。
全部源文件和机器人产物均保留，没有删除文件。此规则不检测相近动作，去重索引也
不是已经划分好的训练/测试集。

每个批次执行 Stage-II 输入检查、G1 重定向、IsaacLab 50 Hz 打包、来源与流水线指纹
检查以及 YAML 生成。日志位于 `logs/amass_preparation/`。

11:37 的独立全量审计重新计算了当前环境、代码、配置、模型和机器人资产指纹，
逐条检查原始 Stage-II 核心数组、输入链接、来源 SHA-256、`.pkl` 的 29 DoF、单位
四元数与有限值，以及 v3 `.npz` 的 50 Hz、29 个关节、30 个刚体和根部 FK 一致性。
四个原始数据集、30 个累计批次、全部重定向文件和全部打包文件一一对应；各批索引
与实际产物完全匹配，合并索引无重复路径。

新运行的 28 批全部返回 0。调度会话已退出，所有本轮批次进程组均不存在；GPU
计算进程列表为空。数据入口相关 57 项测试、重定向恢复/原子写入相关 7 项测试通过。

审计与可追溯记录：

- [逐条验证报告](../logs/amass_preparation/all_20260905_validation.json)
- [去重选择记录](../logs/amass_preparation/all_20260905_deduplication.json)
- [分批输入清单](../logs/amass_preparation/all_20260905_manifest.json)
- [各批退出码与执行时间](../logs/amass_preparation/all_20260905_results.json)

当前仅准备数据，批量动作的视觉质量检查、按受试者划分训练/测试集、训练和策略评估
属于后续步骤；数据格式验证通过不等于机器人已掌握这些动作。

## 运行设置与恢复边界

继续使用现有环境：

- ScaleRetarget：`/home/sw/.cache/bfm/scaleretarget-py311/bin/python`
- IsaacLab：`/home/sw/isaaclab_ws/env_isaaclab_sim6_newton/bin/python`
- SMPL-X：`/home/sw/shuaiwang/.cache/ula_smplx/SMPLX_NEUTRAL_2020.npz`

本轮设置 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1`，并在 ACCAD 使用 4 个重定向进程、64 个打包环境。
CNRS 使用 2 个重定向进程、32 个打包环境。

复用已完成 ACCAD 批次的命令如下；CNRS 使用相同入口，将数据集名改为 `CNRS`、
运行名改为 `amass_cnrs_batch_v1`，重定向进程改为 2、打包环境改为 32、加载进程改为 1。

```bash
cd /home/sw/bfm
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  timeout --signal=INT --kill-after=15s 45m \
  /home/sw/.cache/bfm/scaleretarget-py311/bin/python scripts/amass_to_scalebfm.py \
  --input /home/sw/bfm/ScaleRetarget/dataset/amass/ACCAD \
  --run-name amass_accad_batch_v1 \
  --smplx-model /home/sw/shuaiwang/.cache/ula_smplx/SMPLX_NEUTRAL_2020.npz \
  --retarget-python /home/sw/.cache/bfm/scaleretarget-py311/bin/python \
  --isaaclab-python /home/sw/isaaclab_ws/env_isaaclab_sim6_newton/bin/python \
  --retarget-workers 4 --package-num-envs 64 --package-loader-workers 2 \
  --output-fps 50
```

形体拟合仍执行 6,000 次 Adam。代码目前没有跨运行的形体拟合磁盘缓存，每次新的
重定向进程会重新拟合。本轮 ACCAD 在限制数值库线程后，拟合约 35 秒。

已完成且指纹匹配的批次可通过相同命令重新运行并跳过已有结果。被中断的批次不等于
完成：当前重定向来源记录在重定向阶段成功后统一写入；若中途退出，必须先检查产物
及来源记录，不能直接认领缺少记录的输出为有效缓存。

2026-09-05 10:22:51 至 11:36:47 完成 BMLmovi、BMLrub 的全量处理，共 4,925 个动作，拆为 28 个
可验证批次。普通动作最多 4 批并行，每批 2 个重定向进程、32 个打包环境；81 条超过
4,000 源帧的动作归入长动作批次，在普通批次结束后单批执行，打包环境降为 8。
没有截短长动作。每批有 45 分钟超时及进程组清理，整个受控会话有 3 小时安全上限；
均未触发超时。会话最终回收了所有本轮子进程，没有常驻后台任务，未关闭用户 Firefox。

恢复拆分批次时，使用清单中同一条记录的 `input` 和 `run_name` 配对：输入是
`ScaleRetarget/dataset/amass_bmlmovi_batch_v1_r001` 这类准备目录，而不是将整个 BMLmovi
原始目录配给一个分片运行名。普通分片使用 2 个重定向进程、32 个打包环境、1 个加载
进程；长动作分片的打包环境数为 8。已验证批次可以复用现有来源记录，不能盲目认领
中断批次缺失来源记录的产物。

关于下一批的具体选择，见 [补充数据建议](AMASS_DATA_RECOMMENDATIONS.md)。
