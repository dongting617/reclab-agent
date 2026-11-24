# Agent 框架与模型优化接入

目标是让 Agent 在受控范围内提出修改、执行实验并依据独立评分决定下一步。框架负责可复用的交互机制；推荐模型、数据协议和隔离执行保留各自接口，避免为了换一种 Agent 策略而重写训练代码。

本次参考 Hello Agents [第七章：构建你的 Agent 框架](https://github.com/datawhalechina/hello-agents/blob/main/docs/chapter7/第七章%20构建你的Agent框架.md) 的分层方式与统一接口，独立实现适合 RecLab 的组件。没有引入 hello-agents 包，也没有将教学示例的自由文本动作解析替换进现有实验链路。现有原生 Function Calling、结构化结果和独立评分继续使用。

## 分层与职责

```text
Agent / AgentConfig / ModelBackend       core/ 的公共接口
                 |
Simple / ReAct / PlanExecute / Reflection
                 |
RunSession：消息、模型轮次、权限、工具调用、Trace
                 |
ToolRegistry：组合已有工具箱或注册新 Tool / 函数
          /                        \
EvidenceToolbox                  OptimizationToolbox（显式配置）
只读历史证据                  Workspace + 可信实验 runner
                                      |
                     SSH+tmux / Docker → 预测 → 独立评分
```

`core/` 不知道 SASRec、Semantic ID、Docker 参数或推荐指标；策略只决定何时规划、执行或评价。`Toolbox` 只要求 `specs` 和 `invoke()`，因此旧 `EvidenceToolbox` 无需改写即可使用。

代码职责、工具位置和迁移路径见[项目架构与代码导航](architecture.md)。通用接口从 `core`、策略从 `agents`、工具从 `tools`、流程从 `workflow` 导入；任务契约在 `domain`，装配入口在 `application`。

`RunSession` 集中管理同一次 run 的身份、消息历史、白名单、模型轮次和工具结果，所有阶段共用限制。模型请求失败可按配置重试；重试次数与成功模型回合数分别计量。API 请求重试耗尽仍保留原来的抛异常行为，不伪造正常终态。

## 可运行策略

| 类型 | 实际控制流 | 重要边界 |
|---|---|---|
| `ReActAgent` | 模型决定下一步 → 执行工具 → 观察结果 → 继续或回答 | 保留原生 Function Calling；未强制输出私有推理过程 |
| `SimpleAgent` | 一次模型回答 | 不向模型暴露工具，也不执行违规工具请求 |
| `PlanExecuteAgent` | JSON 规划 → 顺序执行子任务 → 综合答案 | 规划和综合阶段禁用工具；计划本身不批准研究实验 |
| `ReflectionAgent` | 生成答案 → JSON 自评 → 按反馈修订 | 最后修订稿可能尚未再次自评，由 `review_approved` 明确标记 |

计划执行统一使用 `PlanExecuteAgent`。工厂识别现有 `TaskSpec.mode`：`single` 对应 ReAct，`plan_execute` 对应计划执行，`reviewer` 对应 Reflection，因此已保存任务的模式字段无需迁移。自评只表示模型自身给出的反馈，不能代替用户审阅或独立评分。

各策略的 `run()` 都返回原来的结构化结果，包括 `run_id`、`task_id`、`status`、`answer`、`steps`、`tool_calls`、`evidence_paths` 和 `elapsed_ms`。计划与反思结果按需增加字段。`completed` 表示交互完成，不能直接解释为答案正确或模型效果提高。

```python
from reclab_agent.core import AgentConfig, OpenAIResponsesBackend
from reclab_agent.agents import create_agent
from reclab_agent.tools import EvidenceToolbox

agent = create_agent(
    "plan_execute",
    EvidenceToolbox("."),
    OpenAIResponsesBackend("YOUR_MODEL_ID"),
    config=AgentConfig(max_steps=20, model_retries=2),
    max_plan_steps=4,
    trace_path="traces/unique-plan-run.jsonl",
)
result = agent.run("比较已有证据中的模型差距，区分观察和待检验假设")
```

以上在线调用需要安装 OpenAI 可选依赖并配置模型凭据。测试中的脚本后端遵循同一个 `ModelBackend.respond()` 接口，不发送在线请求。普通 Agent 每次 `run()` 创建新历史；`get_history()` 返回深拷贝。研究协调器显式提供同会话的阶段检查点及项目级账本和知识检索，不直接拼接全部历史对话。

## 扩展工具和策略

现有函数可直接注册，无需继承或修改框架：

```python
from reclab_agent.tools import EvidenceToolbox, ToolRegistry, ToolSpec

def describe_profile():
    return {"profile": "development", "test_labels_available": False}

tools = ToolRegistry().include(EvidenceToolbox("."))
tools.register_function(
    ToolSpec("describe_profile", "Describe the configured development profile", {
        "type": "object", "properties": {}, "additionalProperties": False,
    }),
    describe_profile,
)
```

复杂工具可继承 `Tool` 并实现 `spec` 和 `run(arguments)`。注册表拒绝重名，组合旧工具箱时保留其结果信封；可预期参数错误转成结构化结果，意外实现错误继续抛出。注册某项能力不表示所有任务都能使用它，任务白名单仍在模型和执行层检查。

新策略继承 `core.Agent`，通过 `_session()` 创建单次运行，复用 `turn()`、`execute()`、`run_loop()` 和 `finish()`。每个子任务继续使用同一会话，保证总预算和 Trace 连贯。保持规划、评价等只读阶段的工具限制，并注册到 `agents.factory.AGENT_TYPES`。

## 模型优化的接入方式

`OptimizationToolbox` 通过 `StudyContext` 组合 `Workspace` 与研究契约，基本能力包括（完整工具集见研究使用说明）：

1. `read_source`：返回登记文件文本和哈希。
2. `apply_patch`：检查匹配的人工审阅记录、允许路径、补丁预算和旧哈希，再执行精确替换。
3. `validate_candidate`：检查语法，绝不在宿主运行候选程序。
4. 配置可信 runner 后提供 `run_experiment`：校验候选、种子与计划范围，冻结快照后交给 runner。
5. `get_experiment`：返回本实例保存的实验记录和独立评分引用。

推荐通过任务配置绑定内置通用工具，模型作者只提供标准训练、预测入口、数据与模型配置。Agent 自主选择候选、修改代码并调用 `run_experiment(candidate_id, seed)`；工具内部的 `ExperimentRunner` 固定执行训练→预测→独立评分，无需为每个模型另外编写 runner。

```python
from reclab_agent.application.binding import create_task_agent

agent = create_task_agent(
    "configs/beauty-sasrec-task.json",
    plan=reviewed_plan,
    review=user_review,
)
result = agent.run()
```

这里的计划、审阅和执行 Policy 来自可信控制端；构造函数不发起实验，也不从任务 JSON 自动生成批准记录。数据与评分器由登记名选择，任务模型不提供评分数值作为正式结论。Skill 可描述研究步骤、对照设计与报告要求；实际执行、失败处理和评分由工具程序实现，不依赖语言模型每次重新拼接固定步骤。

训练入口固定接收 `--data --output --config` 并写出模型产物；预测入口还接收 `--model --split valid` 并导出逐用户排名。现有 `valid.py` 可通过薄封装映射为 `src/predict.py`；若它仅打印汇总指标，需要增加排名导出，供固定评分器独立重算。

通用工具保留全部登记源码，只开放计划中的可编辑子集。快照、数据、配置、种子及评分协议均纳入实验身份。训练失败不提交预测，预测失败不评分；控制器超时或提交结果不确定时保存停止观察，未确认的停止会阻断同一输出目录中的后续实验。阶段限额采用确定的训练 80%/预测 20% 分配以稳定作业身份；轮询截止时间不替代外部硬超时 supervisor 与独立 watchdog。每个实验保存其实际 Worker Policy，供后续恢复与核对。

底层仍支持下面的回调接口，供可信宿主集成其他执行后端及控制流测试；常规模型接入使用上述内置绑定即可：

```python
runner(snapshot, *, candidate_id, seed, max_seconds) -> ExperimentRecord
```

`snapshot` 是 `Workspace.snapshot()` 的真实返回值。标准绑定按配置选择 `ExperimentRunner`（Docker）或 `RemoteExperimentRunner`（SSH+tmux），复用数据适配器和固定评分器完成训练→预测→评分；新模型无需实现 runner。完成记录要求训练/预测作业身份与独立评分引用；候选自报指标不能替代评分。runner 只使用验证集反馈；基线与候选默认使用同一个固定随机种子训练和评估，每个方案一次，不要求另行测试集评估。独立大模型 Agent 的 API Key 留在控制端；SSH 工具不依赖大模型 API。

接入代码示意，变量由宿主提供的任务、工作区、审阅和实验配置产生：

```python
from reclab_agent.tools import EvidenceToolbox
from reclab_agent.tools.optimization_toolbox import OptimizationToolbox
from reclab_agent.agents.optimization import OptimizationAgent

research = OptimizationToolbox(
    workspace,
    task=task,
    plan=plan,
    review=user_review,
    snapshots_root=".reclab/snapshots",
    run_experiment=trusted_runner,
)
agent = OptimizationAgent(
    research, llm_backend, kind="react", max_steps=32,
    target_model="SASRec 下一商品推荐模型",
    evidence=EvidenceToolbox(repository_root),
)
result = agent.run()
```

`OptimizationAgent` 运行前会把任务、研究计划和宿主登记的目标模型描述整理为结构化上下文，明确可编辑文件、允许候选与种子、基线和数据 Profile、主指标与约束指标、评价协议以及预算。模型据此选择 `read_source`、补丁和实验参数；上下文只提供开发所需资料，不包含私有测试标签，也不会代替匹配的用户审阅。`target_model` 是描述信息，实际训练模型由宿主提供的工作区、数据和可信 runner 决定。目标模型与驱动 Agent 的语言模型分别配置，更换目标模型不要求改写策略或共享会话。

同样的工具箱可交给计划执行或反思策略。Agent 可以根据语法错误修复候选、根据有效负结果调整下一次允许的修改，也可以报告证据不足并停止；是否保留改进必须根据匹配协议下的独立评分判断。

独立评分报告应放在证据工具允许读取的仓库范围内。`ExperimentRecord.score` 保留报告路径、哈希和位置，模型再通过 `inspect_artifact` / `read_json_values` 读取真实指标；只有评分引用而没有可读取的指标时，应报告证据缺口，不推断收益。`repository_root` 为报告所在的仓库根目录。

## 已执行的约束与仍需接入的能力

适配层检查任务/计划身份、审阅版本、文件/候选/种子范围、补丁与实验尝试次数、候选数量、保守预留的训练秒数。模型轮次上限取调用方、任务和计划三者最小值；runner 报错也扣实验尝试次数。同实例内，相同快照/候选/种子的已记录请求复用结果。

通用 `ExperimentRunner` 保留训练→预测→独立评分协议。标准启动入口现由 `StudyCoordinator` 按四阶段组织，`OptimizationToolbox` 是服务的薄工具适配器；背景阶段复用或建立基线，候选须先登记假设。研究状态、预算预留和阶段会话检查点保存在原会话，项目 SQLite 保存实验、事件与知识引用。人民币实际账单仍未核验；这不是跨机器的全局调度系统。隔离和硬超时继续依赖 Worker/supervisor。2026-11-23 的离线测试及真实模型工具 schema 验证已执行，真实 GPU 验收被当前 SSH 连接拒绝阻断。使用方式见[研究使用说明](research-usage.md)。

当前新增测试验证了多种策略、共享权限与轮次、真实源码补丁/快照和可信 runner fixture 的往返。fixture 不是真实推荐训练；尚不能声称本次重构已经提高 SASRec/Semantic ID 的效果。后续验收需要真实模型的有效改进、有效负结果、失败修复和同预算策略对照。

## 兼容性

- 旧 `AgentRuntime` 和 `reclab_agent.runtime` 导入不再保留；原工具循环由 `agents.ReActAgent` 实现，默认诊断策略仍为 `react`。
- `ModelTurn` 与 `OpenAIResponsesBackend` 从 `core` 导入；`TraceWriter` 和 `function_tools` 分别位于 `core.trace` 与 `core.session`。扩展策略依赖这些规范模块。
- 工具接口和实现统一放在 `tools/` 包，继续使用 `reclab_agent.tools` 作为公开导入路径。
- 原 6 个证据工具及任务集不变，旧 Trace 仍可离线 Replay。
- 未修改 `JobSpec`、Worker 身份哈希、数据库格式、receipt、训练/预测命令、数据与评分协议。
- 不新增必装依赖；已有包发现配置包含新子包。

```bash
python -B -m unittest discover -s tests -v
python -B -m reclab_agent
python -B -m reclab_agent --help
```

Windows 沙箱若使用很长的系统临时目录，快照测试可能超过路径长度限制。可仅为测试进程设置仓库内的短临时目录，无需更改实验哈希或快照规则：

```powershell
$testTempRoot = Join-Path $PWD ".reclab/tmp"
New-Item -ItemType Directory -Path $testTempRoot -Force | Out-Null
$env:TEMP = $testTempRoot
$env:TMP = $testTempRoot
python -B -m unittest discover -s tests -v
```

2026-11-19 早期结构验证：短临时目录下全量回归运行 246 项，结果成功，11 项按平台或显式验收条件跳过；原开发任务 15/15。清理旧构建缓存后重建 wheel，检查包内没有已删除的 `runtime.py`、`tools.py`、`_tools/`，隔离导入和 ReAct 冒烟测试通过；该次结构验证没有启动真实推荐训练。同日后续已完成 SSH+tmux 的实际 GPU 训练，见[实测报告](../reports/autodl-ssh-tmux-20261119.json)，仍未调用独立 Agent 的大模型 API。
