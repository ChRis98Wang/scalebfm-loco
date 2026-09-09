# BFM 动作切换 GUI

2026-09-06：已取消参考骨架和坐标轴叠加，只显示策略驱动的 G1 机器人。
策略仍使用重定向动作参考输入，不是移除了跟踪目标，也不是实时人体骨架输入。

## 操作

- `Search`：输入 `walk`、`run`、`jump`、`ACCAD` 等动作名关键词后点击按钮。
  多个词按“全部匹配”筛选，忽略大小写。中文输入法下先切换英文输入或完成输入法提交。
- 下拉列表：浏览和选择候选，**不会立即重置机器人**。
- `Play selected`：播放选中的动作，从第 0 帧重置机器人、观测历史与动作缓存。
- `Previous` / `Next`：在当前筛选结果内切换，首尾循环。
- `Restart`：从头重播当前动作。
- `Clear`：恢复完整目录，不中断当前动作。
- `Exit player` 或关闭菜单窗口：退出整个回放程序，而非把窗口隐藏后留在后台。

当前目录合并 962 条验证动作与 4247 条训练动作，共 **5209 条动作片段**，不等于
5209 种独立语义任务。`run` 搜索得到 49 条名称匹配片段；它不是人工动作分类器。
合并使用临时 YAML，加载完成即清理，不更改训练、验证索引，不包含 44 条排除动作。
此菜单仅支持一个回放环境，不支持与 `--headless` 或 `--video` 同时使用。

2026-09-07 新增：`Reference mode` 下拉框和 `Apply mode + restart` 可现场切换
Pelvis-1、VR-3、WholeBody-14 等参考掩码。选择后需要点击 Apply 才生效并重启片段；
它仍使用离线动作参考，不是外部目标点控制。加入物体、配置材质与单独复位的入口：

```bash
bash scripts/open_motion_browser.sh --target-object
```

目标物体默认有质量、重力、碰撞和摩擦；HUD 显示真实位置与骨盆距离。参数、验收和
明确能力边界见 [目标物体验证说明](BFM_TARGET_OBJECT_VALIDATION.md)。

## 启动：沿用现有 IsaacLab

2026-09-07 起，在仓库根目录、本机已解锁的桌面终端执行：

```bash
bash scripts/open_motion_browser.sh
```

本机默认复用已有解释器；其他机器设置 `BFM_ISAAC_PYTHON` 为已有 IsaacLab Python
的绝对路径。可用 `BFM_LOAD_RUN`、`BFM_CHECKPOINT`、`BFM_INITIAL_MOTION`、
`BFM_VALIDATION_INDEX`、`BFM_TRAIN_INDEX` 覆盖本地默认值。
`--help` 查看说明，`--dry-run` 只打印命令，不启动仿真。

这是 **IsaacLab 任务运行在 Isaac Sim / Kit 窗口中**。systemd unit 只是临时进程管理，
不是另一个仿真器，也不是永久安装的服务。新入口会等待窗口退出并把日志显示在终端。
服务名固定，拒绝重复启动；最多运行 **30 分钟**，不自动重启。
用 `Exit player` 退出；紧急停止整个回放及其子进程：

```bash
systemctl --user stop bfm-motion-browser.service
```

不要使用宽泛的 `pkill python`，也不要同时开多个全动作目录回放占用显存。

## 2026-09-07：启动卡住修复

- 原因：默认 `TerrainImporter` 同步访问在线 `default_environment.usd`，导致 Kit UI
  阻塞在 `omni.client.stat`，还没有进入机器人动作播放。
- 修复：任务内使用 `OfflinePlaneTerrainImporter`，直接创建静态无限碰撞平面与本地
  PreviewSurface 材质，保留原摩擦参数和环境原点。不修改安装的 IsaacLab。
- 真实窗口已加载 `model_23197.pt` 和 5209 条片段，记录到多个跳跃、跑步、步行动作切换。
  日志 `logs/amass_training/motion_browser_20260907_acceptance.log`，截图
  `logs/amass_training/motion_browser_20260907_menu_right.png`。
- 首次验收发现超大视觉网格产生条纹，已将**视觉网格**缩为 2000 × 2000 米；碰撞平面
  仍无限。新增浮点精度回归测试；地面相关 8 项 CPU 测试通过。
- 缩小网格后的 GUI 再次到达 `Ready: 5209 clips`，但桌面已锁屏，尚未完成其最终
  画面复核，不把黑色锁屏截图当成渲染验收。
- 11:44:31 停止该次回放，PID 43018、控制组和 GPU 计算进程均已清理。
- 12:52 在已解锁的真实目标物体窗口补充画面复核：可见 G1、实体方块和正常地面，
  未再出现先前的大范围条纹。截图 `logs/amass_training/target_motion_browser_20260907_initial.png`。
  后续自动机器人 GUI 验收通过三种模式掩码、切片、物体单独复位和正常退出，日志
  `logs/amass_training/target_robot_ui_20260907.log`。

## 2026-09-06：历史验收记录

- ScaleTrack 的 45 项 CPU / 配置测试通过；59 条警告来自已有 IsaacLab/PyTorch 弃用提示。
- 真实 Kit UI 烟测覆盖搜索、筛选后 ID 映射、切换、进度、空结果、退出以及构造失败清理。
- 实际机器人窗口中完成 Previous / Next / Restart、`run` 搜索、下拉选择及 Play。
  验证下拉选择本身不重置，点击 Play 后才切换，日志明确记录 `frame=0`。
- 实际跨集合切换：验证动作 ID 17 → 训练动作 ID 985
  `ACCAD/Female1Running_c3d/C10_-__run_backwards_stop_run_forward_stageii`。
- 14:01 点击 `Exit player` 后，PID 99288 及对应控制组消失，GPU 计算进程列表为空。
  验收服务 `bfm-motion-menu-20260906-1353.service` 已卸载。
- 截图和日志：`logs/amass_training/motion_menu_20260906_1353*`。
  这轮交互验收使用稳定的 `model_22800.pt`；上面的用户启动命令使用已完成训练且
  数值检查通过的最终权重 `model_23197.pt`，不把前者的交互截图当成后者的质量评估。
- 14:03:35 按上述命令重新打开最终权重，实际窗口和日志再次确认加载 5209 条动作，
  菜单移至视口右侧，不遮挡机器人。PID 107200 属于用户查看中的有界回放，
  不是验收遗留进程；本次会在 14:18:35 达到运行时限并执行控制组清理。
  最终截图：`logs/amass_training/motion_browser_20260906_clean_gui.png`。

实现分为纯选择逻辑 `playback_menu.py`、原生窗口 `playback_ui.py` 和 `play.py`
仿真边界切换。代码审查促使本次补全启动失败路径的资源清理与小屏窗口位置。
兼容检查使用 Kit 实际 headless 配置，避免本机 IsaacLab 3 的 GUI 缓存属性误判。
没有修改 PPO、训练采样、网络结构或数据处理逻辑。
