# 项目架构与代码导航

本文描述当前代码布局。业务需求见 [plan.md](plan.md)，流程目标见 [实施方案](implementation-plan.md)，优化历史见 [优化记录](optimization-log.md)。

## 从哪里开始读

先读 `workflow/coordinator.py` 了解“背景 → 假设 → 实验 → 分析”的完整流程，再读 `tools/toolbox.py` 了解可用能力，需要深入训练、评分或历史经验时再进入对应模块。

主要调用关系：

`cli/research.py → application/launch.py → workflow/coordinator.py → tools/各能力组 → experiments/、execution/、memory/ → reporting/`

`start.cmd → scripts/start-agent.ps1 → application/start.py` 负责启动环境与默认配置。模型对话由 `agents/optimization.py` 组合具体策略，并通过 `core/` 执行模型请求、工具调用和轨迹记录。

## 仓库顶层

| 目录/文件 | 内容 |
|---|---|
| `reclab_agent/` | Agent 系统的 Python 实现 |
| `backends/` | 被优化的推荐模型源码，包含独立训练与预测入口；`sasrec/` 的共享 next-item 运行时提供 SASRec 默认模型及 GRURec 迁移示例 |
| `configs/`、`reclab.json` | 任务、模型、执行环境、预算和默认配置位置 |
| `tests/` | 单元测试、工具边界与流程集成测试 |
| `experiments/strategy_comparison/` | 多策略对比实验脚本；区别于包内负责实验生命周期的 `reclab_agent/experiments/` |
| `evidence/`、`tasks/` | 默认证据诊断任务的脱敏事实与确定性断言 |
| `reports/` | 验收和实验报告，历史报告不随模块迁移改写 |
| `scripts/`、`docker/` | 启动、环境准备、验收及镜像脚本 |
| `docs/` | 需求、架构、操作说明、实施方案和优化记录 |
| `.reclab/` | 本地运行数据、项目账本、Wiki、会话、缓存和凭据；由 Git 忽略 |

## Python 包的职责

包根目录只保留 `__init__.py`、`__main__.py`、`start.py`、`research_cli.py`。后三个是薄命令入口，实现位于子包。原 `research/`、`_execution/` 已移除。

| 子包 | 主要文件 | 负责什么 |
|---|---|---|
| `cli/` | `diagnostics.py`、`research.py`、`projects.py`、`worker.py`、`web.py`、`evaluate.py` | 参数解析、命令分发、工作台与本地评测入口 |
| `application/` | `start.py`、`launch.py`、`binding.py`、`inspection.py`、`environment.py`、`history.py`、`runs.py` | 应用装配、启动检查、归档及 CLI/Web 共用的会话查询 |
| `workflow/` | `stages.py`、`coordinator.py`、`resources.py` | 四阶段顺序、阶段权限、停止条件、共享模型和联网预算 |
| `agents/` | `optimization.py`、`team_agent.py`、各策略和 `factory.py` | 原有策略与可恢复的 Planner–Executor–Reviewer 协作 |
| `core/` | `agent.py`、`config.py`、`llm.py`、`session.py`、`trace.py` | 通用 Agent 接口、模型协议、对话运行、工具调度和审计轨迹 |
| `tools/` | 能力组、`context.py`、`toolbox.py`、`registry.py` | Agent 工具定义、处理函数及注册组合 |
| `domain/` | `contracts.py`、`research.py`、`configuration.py`、`authorization.py` | 任务、计划、实验、结论契约；配置校验和范围授权 |
| `experiments/` | `adapters.py`、`backend.py`、`workspace.py`、`runner.py`、`remote_runner.py`、`baselines.py`、`diagnostics.py`、`outcomes.py` | 数据/评分适配器、工作区、快照、训练→预测→独立评分、基线验证与指标比较 |
| `execution/` | `worker.py`、`models.py`、`config.py`、`docker.py`、`locking.py`、`ssh.py`、`remote_agent.py`、`watchdog.py` | 执行类型、Worker、SSH 传输、远端监督、Docker 和锁 |
| `memory/` | `project.py`、`storage.py`、`retrieval.py` | 项目账本、本地 Git/Wiki、经验检索、产物与状态存储 |
| `data/` | `beauty.py`、`datasets.py` | Beauty 数据转换及数据协议 |
| `evaluation/` | `ranking.py`、`traces.py`、`benchmark.py`、`retrieval.py`、`diagnostics.py`、`costs.py` | 验证集评分、检索/冻结诊断评测与显式单价成本估算 |
| `integrations/` | `tavily.py`、`text_models.py` | 外部文献服务与可选 CPU 文本模型适配 |
| `reporting/` | `model.py`、`rendering.py`、`__init__.py` | 报告数据整理、Markdown/HTML 渲染及公开接口 |
| `infrastructure/` | `files.py` | 共用路径约束、文件校验和原子写入 |
| `examples/` | `demo.py` | 合成演示，不代表真实模型优化验收 |
| `web/` | `app.py`、`templates/index.html`、`static/app.js`、`static/app.css` | 本地工作台与只读研究视图；真实优化仍从 CLI 启动 |

`execution/__init__.py` 导出 Worker 类型与执行 API。已有作业身份、数据库、实验产物格式和训练/预测协议不因目录调整而迁移。

## 工具在哪里

每组工具把参数定义和处理函数放在同一文件，处理函数调用底层确定性实现。

| 文件 | 工具 |
|---|---|
| `tools/evidence.py` | `list_evidence`、`inspect_artifact`、`read_json_values`、`compare_numeric_values`、`diagnose_beam_gap`、`diagnose_frequency_gap` |
| `tools/workspace.py` | `read_source`、`apply_patch`、`validate_candidate` |
| `tools/planning.py` | `record_hypothesis` |
| `tools/experiments.py` | `run_experiment`、`get_experiment`、`read_log`、`list_experiments` |
| `tools/diagnostics.py` | `write_diagnostic`、`run_diagnostics` |
| `tools/baselines.py` | `resolve_baseline` |
| `tools/evaluation.py` | `analyze_experiment` |
| `tools/knowledge.py` | `search_knowledge`、`save_research_note` |
| `tools/literature.py` | `search_literature`、`fetch_literature` |
| `tools/remote.py` | `upload`、`download`、`run_remote`、`job_status`、`tail_log`、`gpu_status`、`stop_job` |
| `tools/connectivity.py` | 启动兼容性探针 `startup_check`，不作为实验能力开放 |
| `tools/role_results.py` | `role_plan`、`role_review` 的数据输出 schema，不派发业务操作 |

工具支持层：

- `contracts.py`、`base.py`：工具契约、结构化错误、函数适配器。
- `registry.py`：工具注册、组合及调用。
- `catalog.py`：能力目录，以实际工具箱的 `specs` 判断本次可调用能力。
- `group.py`：能力组注册及共享上下文访问。
- `context.py`：`StudyContext` 保存阶段、计划、预算计数、实验记录和恢复状态。
- `toolbox.py`：`OptimizationToolbox` 组合能力组，不再集中实现所有业务方法。

默认 `RecLabToolbox` 仍只开放六个证据工具。`OptimizationToolbox` 在配置可信 runner 后注册训练工具，挂载知识和文献服务后注册对应工具；协调器按阶段继续限制范围。SSH 工具由宿主配置，与 `execution/ssh.py` 的传输实现分开。

## 调整代码时应放在哪里

- 新增 Agent 能力：在 `tools/` 对应能力组添加 schema 和处理函数，再由 `toolbox.py` 注册；阶段权限调整放在 `workflow/`。
- 更改研究顺序或停止规则：修改 `workflow/`；启动装配修改 `application/`。
- 更改训练、预测、快照或指标比较：修改 `experiments/`；远程通信与进程管理修改 `execution/`。
- 新增推荐模型：放在 `backends/`，通过任务配置和可信适配器接入。
- 更改账本与经验检索：修改 `memory/`；新增外部服务客户端放入 `integrations/`。
- 更改评分规则：修改 `evaluation/`，候选模型代码不能替代独立评分器。

依赖由入口和装配指向流程、工具与底层能力。工具和实验实现不反向依赖 CLI、应用装配或阶段协调器。低层代码可以使用 `tools.contracts.ToolError` 作为统一错误契约。

## Python 导入与命令迁移

```python
from reclab_agent.tools.toolbox import OptimizationToolbox
from reclab_agent.tools.remote import RemoteTools
from reclab_agent.workflow.coordinator import StudyCoordinator
from reclab_agent.domain.research import ResearchPlan, ExperimentRecord
from reclab_agent.experiments.adapters import TaskAdapterRegistry
from reclab_agent.application.binding import create_task_agent
from reclab_agent.execution import DockerWorker
from reclab_agent.memory.project import Project
```

原内部 `reclab_agent.research.*` 和平铺业务模块导入需要更新为分层路径，不保留重复实现。原命令 `python -m reclab_agent`、`python -m reclab_agent.start`、`python -m reclab_agent.research_cli` 保留；规范入口分别是 `cli.diagnostics`、`application.start`、`cli.research`。安装后的 `reclab` 和 `reclab-research` 直接指向规范入口。

Worker 与监督脚本入口改为 `python -m reclab_agent.cli.worker`、`python -m reclab_agent.execution.watchdog`，源码和脚本中的模块调用已同步调整。
