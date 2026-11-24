# RecLab Agent

在预算和时间限制内，辅助推荐模型研究：**理解背景 → 提出假设 → 模型实验 → 分析并得出结论**。Agent 选择动作，工具修改登记源码和执行实验，固定评分器从验证集预测重算指标。

默认基线与候选使用同一个随机种子，每个方案训练一次。历史基线可核验时复用，不强制重跑。

**新版研究工作台已接入真实数据：** 首屏展示核心指标提升图和每次刷新最佳的改进，支持实验对比、证据详情、代码差异和事件筛选。评分未核验、条件不可比较或门禁未通过的高分候选不会获得提升标记。见[工作台说明与验收截图](docs/workbench-design.md)；保留的[静态原型](reclab_agent/web/static/prototype.html)全部为示例数据。

## 选择使用入口

| 想做什么 | 入口 | 需要远程 GPU |
|---|---|---|
| 查看任务、报告、代码改动和事件 | 本地 Web 工作台 | 不需要 |
| 体验证据工具、运行自动化测试 | 本地演示与测试 | 不需要 |
| 检索历史经验、查看实验账本 | 项目 CLI | 不需要 |
| 对比检索、评测诊断决策、统计 token 成本 | 评测 CLI | 不需要；真实文本模型需要本地 CPU 和模型文件 |
| 自主修改推荐模型并训练评估 | `start.cmd` | 当前 SASRec 任务需要远程训练环境 |

## 1. 本地工作台

Windows 在仓库根目录执行，启动器会复用或准备本地 Web 环境，不读取 API Key、不启动训练：

```powershell
.\start.cmd -Web
```

需要 Python 3.10+。已准备 Python 环境时，也可以手动启动：

```bash
pip install -e ".[service]"
python -m reclab_agent.cli.web
```

浏览器打开 **http://127.0.0.1:8000**。工作台可以：

- 查看 CLI 已保存的研究会话、假设、实际 Diff、独立评分、报告与 Trace。
- 运行无需 API Key 或 GPU 的本地证据演示。
- 登记任务并查看等待配置状态；用幂等键避免重复提交。
- 按事件序号重连，取消尚未执行的等待任务。

打开页面不会启动训练。当前 Web 默认不装配真实执行器；实际优化从 CLI 启动，工作台读取相同会话。运行中的远程作业需要执行端确认停止，不能仅在页面上标成已取消。

命令行查看同一份状态：

```bash
python -m reclab_agent.cli.research runs demo
python -m reclab_agent.cli.research runs list
python -m reclab_agent.cli.research runs show --run-id RUN_ID
python -m reclab_agent.cli.research runs events --run-id RUN_ID --after 0
```

## 2. 运行推荐模型优化

PowerShell 中先查看任务范围，再启动：

```powershell
.\start.cmd -Plan
.\start.cmd
```

Linux/WSL 可用 `python -m reclab_agent.start --help` 查看启动参数。默认任务由 [reclab.json](reclab.json) 指定。

| 配置位置 | 内容 |
|---|---|
| `configs/beauty-sasrec-task.json` | 模型、数据、验证指标、可修改文件和总预算 |
| `.reclab/agent.env` | 模型 API 配置，仅保存在本地 |
| `configs/autodl-ssh.json` | SSH 连接，参考同目录 example 文件 |
| 任务配置的 `task.mode` | 默认 `single`；另有 `plan_execute`、`reviewer`，以及迁移中的 `research_audit` |

原 `team` 模式已在未完成的重构中移除，正在迁移为主研究者 + 独立审计。当前默认继续使用 `single`；`research_audit` 的上下文、权限与恢复仍需完成专项验收。本轮工作台仅展示已记录的审计数据，不代表该 Agent 已验收。

每次研究保存源码快照、实验记录和 `report.json`、`report.md`、`report.html`。失败和证据不足也保留报告。恢复、基线复用与项目管理详见[使用说明](docs/research-usage.md)。

## 3. 经验记忆与检索

项目使用 SQLite 实验账本、本地 Git、Markdown Wiki 和按比较范围过滤的经验检索。默认 **BM25，无额外依赖**；数值结论仍应回读原始评分。

```bash
python -m reclab_agent.cli.research project search --config configs/beauty-sasrec-task.json --query "历史失败与诊断"
```

可选 CPU 语义检索复用同一套索引，支持 dense、hybrid 和 reranked。先安装 CPU 版 PyTorch，再安装文本模型依赖：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[retrieval]"
python -m reclab_agent.cli.evaluate retrieval --modes bm25
python -m reclab_agent.cli.evaluate retrieval --modes bm25 dense hybrid reranked --download-models
```

默认只读本地模型缓存；显式传入 `--download-models` 才允许下载。可用 `--embedding-model`、`--reranking-model` 和对应 revision 指定模型。默认 embedding 面向中文、默认 reranker 面向英文；评测报告记录所用模型、来源哈希、Recall@5、MRR 和延迟，不保证重排一定改善结果。

研究任务可在 `research.retrieval` 中指定检索模式，示例与模型版本说明见[本地功能与验收](docs/non-gpu-work.md)。Tavily 文献搜索需另配 `TAVILY_API_KEY`；缺少 Key 时本地知识仍可用。

## 4. 评测与成本

已有 30 个开发场景、30 个冻结场景，按场景组分开；它们是**合成诊断决策任务**，不代表真实训练或端到端研究能力。答案不进入模型可读的证据目录或检索语料。

默认使用匿名 ID 的 v2 冻结集。初稿的 dataset ID 含场景名称，因此旧分数仅保留为工程试跑记录，不当作 v2 成绩。以下参考检查不调用模型 API：

```bash
python -m reclab_agent.cli.evaluate diagnostics --manifest tasks/diagnostic-v2/diagnostic-dev.json --output reports/diagnostic-dev-v2-reference.json
python -m reclab_agent.cli.evaluate diagnostics --manifest tasks/diagnostic-v2/diagnostic-frozen.json --output reports/diagnostic-frozen-v2-reference.json
```

显式选择 `--mode react` 并配置 `--model`、API Key 后，才运行真实 Agent 评测并产生 API 费用。新的 `research_audit` 入口仍在迁移，不作为已验收模式。`--max-requests` 约束整批请求次数，重试和失败请求同样占额度；相同输出文件用于恢复，不重跑已记录的任务。

```bash
python -m reclab_agent.cli.evaluate costs --trace traces/latest.jsonl
```

费用只有在显式提供价格表后才估算；缺失 usage 不算免费，估算不等于服务商账单。详细命令见[本地功能与验收](docs/non-gpu-work.md)。

编码对照入口为 `python -m reclab_agent.cli.evaluate coding --help`，需要 Linux/WSL 的本地 Docker 和已核验镜像；默认 `reference` 不调用模型。选择 `react` 才运行真实模型调用；`research_audit` 仍需验收，候选代码始终在容器中执行。

## 5. 测试与当前进度

2026-11-24：GPU 阶段已归档并推送（`73e1b76`），非 GPU 对照已发布（`fbb9aed`）。当前按[四天面试准备](docs/interview-preparation.md)推进：前两天修复代码与完善评测，后两天完成演示和面试材料。[第一天失败审计](reports/day1-failure-audit-20261124.md)已完成离线复现及局部修复，原始模型成绩保留。SASRec 候选两项验证指标均下降，保留基线；GRU 完成共享接口的基线训练和独立评分，两份未通过 Smoke Test 的候选也保留失败记录。详见 [GPU 验收报告](reports/gpu-acceptance-20261123.md)与[非 GPU 对照报告](reports/non-gpu-acceptance-20261124.md)。

```bash
python -m unittest discover -s tests -q
python -m unittest experiments.strategy_comparison.test_comparison experiments.strategy_comparison.test_runner experiments.strategy_comparison.test_tools -q
python -m reclab_agent
```

基础工具与核心流程使用标准库；Web、在线模型和 CPU 文本模型分别安装可选依赖。最后一条命令执行原有 15 项证据参考调用，不能把它记为端到端成功率。

| 本轮工作 | 实际状态 |
|---|---|
| Web 页面、CLI 共用状态、报告和事件 | 已实现，HTTP 与浏览器交互验收通过 |
| 主研究者 + 独立审计 | 迁移未完成；本轮前端只适配已记录的审计数据，不计为已验收能力 |
| CPU 文本模型适配、四种检索模式和评测 | 真实 CPU 模型已运行；本次小语料重排变差，默认保留 BM25 |
| 30+30 场景、冻结校验、诊断评测与成本统计 | 开发集：单 Agent 28/30、多角色 27/30；冻结集：26/30、23/30，多角色未达 80% 目标 |
| 四类真实编码对照 | 已冻结任务、补丁工具和 CPU Docker 检查；参考配方 8/8，单 Agent 3/8，多角色 6/8，保留失败 |
| Tavily 文献搜索与正文提取 | 真实搜索、提取与缓存复用通过，共 2 次网络请求 |
| 真实 GPU 训练、验证集评分与 GRU 迁移 | 基线和 dropout 候选均有真实分数；候选未改善，保留基线 |
| 远程作业恢复 | 训练进程保留、同一作业复用已实测；首次恢复需人工核验，相关路径已修复并保留失败记录 |
| 本轮界面验收与回归 | 17 项前端逻辑、33 项报告及 HTTP 测试通过；全量 516 项中 505 通过、8 跳过，3 项旧多智能体迁移测试失败，见[验收记录](reports/workbench-acceptance-20261124.md) |

本轮合计预留 478/640 次模型请求，4 次缺少 usage，未核实账单；保留单 Agent 与 BM25 默认。[工作台演示视频](reports/workbench-demo-20261124.mp4)使用实际界面截图合成，展示真实会话，不是连续录屏。编码候选只在本地 CPU Docker 中执行，这四类合成小任务不代表生产仓库修复能力，部分输入字段说明也有歧义。仍待验证云端 Docker/GPU 隔离、所有中断点自动恢复和连续录屏；单种子验证集结果不代表普遍质量提升。历史 GPU 实验见[结果报告](reports/strategy-comparison-20261120/report.md)。

## 代码与文档

主调用关系：`cli → application → workflow → tools → experiments / execution / memory`。通用模型交互位于 `core/`，策略位于 `agents/`；数据、评分、报告和外部服务各自独立。所有 Agent 工具的参数定义与处理函数按能力放在 `tools/`。

- [项目架构与逐文件导航](docs/architecture.md)
- [四天面试准备与代码走读](docs/interview-preparation.md)
- [功能需求](docs/plan.md)与[实施方案](docs/implementation-plan.md)
- [本地功能与验收](docs/non-gpu-work.md)
- [持续优化记录](docs/optimization-log.md)
- [最初八周计划](docs/eight-week-plan.md)
