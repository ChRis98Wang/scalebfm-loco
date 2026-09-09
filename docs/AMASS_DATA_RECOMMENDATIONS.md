# 下一步补充哪些动作数据

核验日期：2026-09-05。当前入库和打包数量见
[AMASS 收集与重定向进度](AMASS_COLLECTION_STATUS.md)。

## 先回答“现在够不够”

现有 ACCAD、BMLmovi、CNRS、BMLrub 共 5,256 条动作、约 12.22 小时，足以启动
首轮多动作追踪基线。这是基于本地数据规模的工程判断，不是官方最低数据量要求，
也不表示已经复现论文效果。CMU 下载完整并校验后可继续扩充。

这四包已全部重定向和打包。全量审计发现三对源文件字节级重复，另提供
[5,253 条去重素材索引](../ScaleRetarget/retargeted_dataset/amass_collected_unique_v1.yaml)。
它只排除完全相同源文件的重复引用，尚未完成视觉质量筛选或训练/测试划分。

BMLrub 约占这四包源动作总时长的 71%，所以后续训练还需要检查数据集和动作类别的
采样比例，避免数量较大的来源主导训练。本轮只准备素材，不改变训练采样策略。

应优先完成现有动作的重定向、打包和质量检查，再依据失败动作补充类别。
参考动作数量、机器人在物理仿真中的 PPO 交互量、策略训练结果是不同的指标。
论文也指出，参考动作的收益取决于多样性，而不只是重复动作的数量。
[ScaleBFM 论文 §3.3](https://arxiv.org/html/2607.15163v1)

## 建议下载顺序

以下优先级是针对当前数据组成的建议；动作类别来自相应官方目录，不保证 AMASS
某一次发布包含原始目录中的全部动作。

| 顺序 | 数据 | 补充目的 | 获取方式 |
| --- | --- | --- | --- |
| 当前 | CMU | 先完成正在下载的数据包，扩充全身动作覆盖 | AMASS，等待最终 `.tar.bz2` 完整落盘 |
| 下一批 1 | KIT | 走路、转向、变速，以及搬运、弯腰和取放动作的全身协调 | [KIT 官方动作目录](https://motion-database.humanoids.kit.edu/list/motions/)，使用 AMASS 对应发布 |
| 下一批 2 | EyesJapan | 手势、问候、坐起、投掷、运动和风格变化 | [EyesJapan 官方目录](https://mocapdata.com/)，使用 AMASS 对应发布 |
| 下一批 3 | MPI_HDM05 | 系统的动作类别覆盖，便于比较走跑跳、坐卧和运动动作的追踪表现 | [HDM05 官方目录](https://resources.mpi-inf.mpg.de/HDM05/index.html)，使用 AMASS 对应发布 |
| 跨数据源扩展 | LAFAN1 | 较长的连续动作以及失衡、恢复、跳跃等组合变化 | [Ubisoft 官方仓库](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)，需走 BVH/LAFAN 流程 |
| 按任务补充 | OMOMO | 若目标是搬箱子等人—物协调，补充带物体运动和几何的数据 | [OMOMO 作者项目页](https://lijiaman.github.io/projects/omomo/)，需专门转换和任务接入 |

KIT、EyesJapan 和 HDM05 是 AMASS 收录的数据来源，见
[AMASS 原论文表 1](https://openaccess.thecvf.com/content_ICCV_2019/papers/Mahmood_AMASS_Archive_of_Motion_Capture_As_Surface_Shapes_ICCV_2019_paper.pdf)。
无需为了开始首轮训练一次性下载上表所有数据；优先完成 CMU，然后补 KIT 和
EyesJapan，即可扩大覆盖。

## 下载时选择什么

1. 在 [AMASS 下载页](https://amass.is.tue.mpg.de/download.php) 登录后，优先选择
   **SMPL-X N / Neutral**，沿用本地已验证的模型和加载流程。
2. 官方 ScaleRetarget 明确要求 AMASS 的 **SMPL-X** 发布，排除 **SMPL+H**。
   Neutral 是本地当前流程的选择，不是官方只支持 Neutral 的声明。
   [官方 AMASS 准备说明](https://github.com/zengweishuai/ScaleBFM/tree/main/ScaleRetarget#amass-recipe)
3. 如果某个子集没有可用的 SMPL-X Neutral 下载选项，先跳过或单独核对，不把
   SMPL+H 文件改名后混入。未登录的公开页面不能核实各子集当前按钮和包大小。
4. 最终归档放在 `/home/sw/Downloads/` 即可。存在 `.part`、零字节占位文件或下载
   未结束时，不计入已收集数据。完整归档需检查压缩流、成员路径和核心动作数组。
5. 原始归档、修复记录及原始坏文件保留；训练素材不包含 Stage-I 中间结果或旧的
   冒烟测试副本。

## 想接近 BFM 效果，还缺什么

- **质量和覆盖统计**：格式正确不代表动作自然。重定向仍可能出现抖动、漂浮或
  僵硬，需要视觉抽查及物理追踪评估。
  [官方已知限制](https://github.com/zengweishuai/ScaleBFM/tree/main/ScaleRetarget)
- **独立训练/验证/测试划分**：不要把同一动作或同一受试者的相近片段随机散落到
  不同集合；批次拆分只是计算调度，不是训练划分。全量 YAML 是素材索引，不能
  直接被称为已完成测试集隔离的训练集。
- **物理训练和定量评估**：完成 50 Hz 数据包不等于机器人已经学会动作；还要测量
  追踪误差、跌倒率、不同动作类别表现和未见动作泛化。不能承诺某个数据量对应
  “论文效果的 90%”。
- **正确使用官方测试数据**：若按论文协议比较，保留其 Xsens/100STYLE 测试动作，
  不把用于测试的序列混入训练。
  [ScaleBFM 论文 §4.1](https://arxiv.org/html/2607.15163v1)
- **任务相关场景与监督**：常规重定向输出是机器人根部和关节轨迹；导入 OMOMO
  或 GRAB 的人体动作，不等于已使用物体状态、接触或抓取监督。真实搬运任务还需
  物体、场景、目标和闭环成功率评估。GRAB 原始数据提供物体姿态及接触等字段。
  [GRAB 官方数据结构](https://github.com/otaheri/GRAB#contents-of-each-sequence)

AMASS 和 SMPL-X 均有用途及再分发限制，不能默认重定向后即可商用或公开发布。
本轮数据仅在本地处理，未上传或公开分发。
[AMASS 许可](https://amass.is.tue.mpg.de/license.html)、
[SMPL-X 许可](https://smpl-x.is.tue.mpg.de/modellicense.html)
