# RecLab Agent

RecLab Agent 的目标是在受控预算下自主提出、实现和验证推荐模型改进。项目将 Agent 策略与模型实验分开：Agent 选择动作，工具执行确定性操作，执行层运行候选程序，独立评分器判断模型效果。当前提供证据诊断、多种 Agent 策略，以及按任务配置绑定的优化适配层。AutoDL 使用 SSH+tmux，Docker 暂缓。

## 启动优化实验

在项目目录执行：

```powershell
.\start.cmd
```

启动器自动加载本地 API 配置、选择 Python/WSL 环境、准备 SDK，并按“理解背景 → 提出假设 → 模型实验 → 结果分析和得出结论”执行。先核验可复用的历史基线，没有合适记录时才建立基线，再开展候选实验与独立评分。结束时显示中文结论、关键指标和报告位置。启动此命令即运行当前配置的范围；当前任务最多两次训练，基线与候选共用预算，使用同一个固定随机种子（默认 2026）进行训练和验证集评估。每个方案默认训练一次；训练按 epochs、早停和时间预算执行。

每轮实验结束会在会话目录自动生成面向用户的 `report.html` 和 `report.md`，包含任务与范围、实际代码改动、基线/候选指标和变化、失败记录、最佳产物位置与结论边界；失败、无有效候选或中断也会留下报告。原始 `report.json` 保留用于程序处理，报告整理在本地完成。

项目级 SQLite 账本、本地 Git、Markdown Wiki、BM25 经验检索、任务恢复、过程指标和可选 Tavily 调研已接入标准研究入口。命令和配置见[研究使用说明](docs/research-usage.md)，实际改动与验收边界见[仓库优化记录](docs/optimization-log.md)。2026-11-23 的新链路离线与模型接口检查已执行，真实 GPU 验收受当前 SSH 连接拒绝阻断；Tavily 尚缺本地 Key，未完成联网实测。

本次真实 GPU 实验已补齐[中文结果报告](reports/strategy-comparison-20261120/report.md)和[可离线打开的 HTML 报告](reports/strategy-comparison-20261120/report.html)。

只查看任务和预算：

```powershell
.\start.cmd -Plan
```

默认任务和 API 配置文件的位置集中在 [reclab.json](reclab.json)。模型、数据、可修改文件和预算在它指向的任务配置中设置；实际 API Key 放在被 Git 忽略的 `.reclab/agent.env`，实际 SSH 连接放在 `configs/autodl-ssh.json`。更换任务时修改 `reclab.json` 的 `task` 即可。Linux/macOS 使用 `python -m reclab_agent.application.start`。

当前里程碑提供：

- 限定在仓库根目录内的只读文件沙箱；
- 证据清单、文件 SHA256 与 JSON 元信息核验；
- 基于 JSON Pointer 的指标读取和数值比较；
- Beam Search 漏召与目标频次差距诊断；
- 15 个带可执行断言的开发集任务；
- 三份来自 Semantic ID 推荐实验的脱敏证据样本。
- 统一管理步数上限、重试、消息和 JSONL 审计轨迹的运行会话；
- OpenAI Responses API 函数调用适配器，以及无需 API Key 的可复现测试后端。
- 任务级工具白名单，在模型层和执行层同时限制可调用能力；
- JSONL 轨迹离线回放，评估工具选择、证据引用、越权调用、错误率与延迟。
- 统一的 Agent、模型后端和工具接口，以及 ReAct、Simple、计划执行和 Reflection 策略；
- 可组合的工具注册表，原有证据工具与新增业务工具共用权限和审计机制；
- 按研究计划约束的源码补丁、语法检查、冻结快照与实验回调，保留原实验协议。

## 开发与证据诊断

项目仅依赖 Python 3.10+ 标准库：

```bash
python -m unittest discover -s tests -v
python -m reclab_agent
python -m reclab_agent --list-tools
```

基准命令会执行 `tasks/dev.json` 中的 15 个参考调用，并输出逐任务结果与通过率。

安装可选的 OpenAI SDK 后，可以执行自然语言任务：

```bash
pip install -e ".[openai]"
set OPENAI_API_KEY=your_key
python -m reclab_agent --model YOUR_MODEL_ID --ask "判断 Beam Search 是否为 G2 漏召的主要瓶颈，并引用证据"
```

生产任务可以按最小权限只开放必要工具：

```bash
python -m reclab_agent --model YOUR_MODEL_ID \
  --ask "判断 Beam Search 是否为主要瓶颈，并引用证据路径" \
  --allowed-tools diagnose_beam_gap
```

保存的轨迹无需再次请求模型即可复算质量指标，适合比较 Prompt、模型版本或工具实现：

```bash
python -m reclab_agent --replay traces/latest.jsonl \
  --expected-tools diagnose_beam_gap
```

每次运行会把模型回合、工具参数、结构化结果、证据路径、耗时和错误写入 `traces/latest.jsonl`。工具调用采用严格 JSON Schema；Agent 最多运行八个模型回合，瞬时模型错误默认重试两次。

默认 `--agent react` 保留原来的工具调用循环，也可以选择先规划再执行或显式反思：

```bash
python -m reclab_agent --model YOUR_MODEL_ID --agent plan_execute --max-steps 20 \
  --trace traces/plan-run.jsonl --ask "比较证据中的模型差距，并说明仍需验证的假设"
python -m reclab_agent --model YOUR_MODEL_ID --agent reflection --max-reflections 2 --max-steps 20 \
  --trace traces/reflection-run.jsonl --ask "分析已有实验，并检查结论是否得到证据支持"
```

规划、执行、反思和综合共用 `--max-steps`；反思不会授予额外工具或实验权限。`simple` 只生成一轮回答，不执行工具，适合不涉及实测数值的任务。每次运行使用独立 Trace 文件；离线 Replay 仍要求一份文件只包含一次运行。

完整架构、Python 接口和模型优化接入方式见[Agent 框架与扩展](docs/agent-framework.md)。

模型优化任务的路径、数据身份、评分器、指标和预算集中在任务配置中。先检查输入：

```bash
python -B -m reclab_agent.cli.research check --config configs/beauty-sasrec-task.json
```

`inputs_valid` 表示文件与接入合同通过检查；`ready_to_run` 表示真实执行条件是否就绪。检查入口不启动训练。通用 `run_experiment` 工具提供训练→预测→独立评分的固定流程，通过 `create_task_agent()` 按任务配置绑定，模型无需实现自己的 runner。将 [autodl-ssh.example.json](configs/autodl-ssh.example.json) 复制为被 Git 忽略的 `configs/autodl-ssh.json` 并填写实际连接信息；七个远程工具和 tmux 恢复约定见[执行层说明](docs/execution.md)。独立 Agent 的大模型 Key 和模型名配置在本地控制端环境变量中。

默认使用上方的 `start.cmd`。底层 `research_cli` 的计划查看、API 预检及显式参数用于开发集成，见[执行层说明](docs/execution.md)。每次实验使用独立的 `sessions/<时间>-<ID>/`，包含报告、计划、Agent 轨迹和实验产物；缺少实测候选时标记为 `incomplete`。

已有实验 JSON 可以直接补生成用户报告：

```bash
python -B -m reclab_agent.cli.research report --input path/to/session/report.json
```

## 项目结构

```text
reclab_agent/   按职责分层的 Agent、工具、研究流程、实验与记忆实现
evidence/       可复现的脱敏实验事实
tasks/          Agent 任务与确定性断言
tests/          工具边界和基准测试
backends/       自包含的推荐模型源码入口与统一模型/训练配置
configs/        优化任务的路径、数据/评分器选择、范围和预算
```

配套学习与面试说明见 [`docs/learning-roadmap.md`](docs/learning-roadmap.md)。

## 安全边界

默认诊断 CLI 仍只注册原有 6 个只读证据工具。可选 `OptimizationToolbox` 只允许修改登记的候选文件；补丁与实验检查匹配的研究计划审阅记录，只有宿主配置可信 runner 后才暴露实验工具。执行配置显式选择 Docker 或 SSH+tmux；原生模式适用于自有、已审阅的代码，不提供容器隔离。远端监督进程管理超时和进程组停止；AutoDL 费用及实例生命周期由用户管理。

## 下一步

当前先完成 AutoDL SSH+tmux 执行链路和训练验收；独立大模型 Agent 的完整优化仍需本地 API 配置与正式实验。训练结果、合成 Demo 和脚本回复分别记录，模型改进结论以独立验证集评估为准。Docker 后续单独处理。

本次真实 GPU 的工具验收、三轮基线/候选对比及 checkpoint 位置见[AutoDL 实测报告](reports/autodl-ssh-tmux-20261119.json)。

## 研究流程与内部结构

研究流程、工具清单、证据要求及接入边界见[Agent 工具与研究流程](docs/agent-workflow.md)。诊断与优化设想可以交替进行；实验前审阅研究计划，范围内自主修复和实验，最后交付有证据的正向、负向或不确定结论。

模型实验协议与默认诊断 CLI 的行为保持兼容，Agent 代码按以下模块组织：

```text
reclab_agent/
  cli/              命令入口
  application/      配置、环境检查与应用装配
  workflow/         四阶段协调与共享预算
  core/             通用 Agent 运行时和模型协议
  agents/           Agent 策略与优化 Agent 组合
  tools/            按能力组织的工具定义、处理函数、注册表
  domain/           任务、计划、实验与结论契约
  experiments/      工作区、基线、训练→预测→独立评分
  execution/        Worker、SSH、Docker 和远端进程监督
  memory/           实验账本、Wiki、Git 与经验检索
  data/             数据准备与数据协议
  evaluation/       验证集评分、Agent 轨迹和基准评测
  reporting/        报告模型与 Markdown/HTML 渲染
  integrations/     Tavily 等外部服务客户端
  infrastructure/   共享路径与文件操作
  examples/         合成演示
  web/              可选 Web 服务草稿
```

完整文件职责、工具位置、依赖方向和导入迁移见[项目架构与代码导航](docs/architecture.md)。

`reclab_agent.tools.tool_catalog()` 保留原默认工具目录。默认可调用列表由 `RecLabToolbox.specs` 决定；宿主显式组合 `OptimizationToolbox` 或注册自定义工具后，以组合工具箱的 `specs` 为准，目录条目本身不授予执行能力。

Python 导入按职责使用各子包，原 `research/` 和 `_execution/` 已移除。包根只保留公开导出和三个薄命令入口；原启动命令继续可用。内部导入迁移见架构文档，训练/预测协议、Worker 作业身份和已有产物格式保持一致。

`OptimizationAgent` 运行前会注入任务、研究计划及宿主登记的目标模型上下文，包括可编辑文件、候选和种子范围、基线、指标、评价协议与预算，避免让模型猜测工具参数。新增模型通过工作区、数据与可信实验 runner 接入；完成一次 Agent 会话仍不代表模型效果得到提升。
