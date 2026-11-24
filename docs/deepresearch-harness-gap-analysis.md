# RecLab Agent 对照 DeepResearch Harness 的功能差距

更新日期：2026-11-23。

RecLab 已具备推荐模型研究的四阶段流程、工具权限控制、独立评分、实验账本、多角色协作、任务恢复、经验检索和本地 Web 工作台。对照 DeepResearch Harness，主要缺口集中在完成质量门禁、证据上下文工程、统一运行控制、长会话记忆和受控 Skill 学习。现阶段更值得优先补齐这些运行能力；GraphRAG 和通用深度研究产品形态可以按实际需求引入。

## 对照范围与判断口径

- 参考仓库：[SichengLong26/deepresearch_agent_harness](https://github.com/SichengLong26/deepresearch_agent_harness)，本次读取的提交为 [`4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0`](https://github.com/SichengLong26/deepresearch_agent_harness/tree/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0)。以下远端链接均固定到该提交。
- 当前仓库：`reclab-agent`，HEAD 为 `1d01779655d6573e7f6a218a92dbb0e9d3b04d3f`，**判断包含当前工作区已修改及未跟踪的源码**，尤其是 `team_agent.py`、`application/runs.py`、Web 静态资源和新增评测代码，不能只用 HEAD 或旧计划判断状态。
- “未实现”：在本次检查的源码、依赖、入口及测试中未找到对应能力；“部分实现”：有相关机制，但未形成参考仓库的完整运行闭环；“待验收”：已有实现，仍缺真实环境结果。
- 这是源码层面的能力对照。本次没有运行两套系统的模型、Tavily、Neo4j 或 GPU 任务；参考实现有代码和测试不代表其所有能力已在当前环境验证。文档中的优先级和验收条件是面向 RecLab 的建议。

## 已实现的基础能力

下列能力不应再次列入“从零建设”的任务。

| 能力 | 当前代码依据 | 仍需区分的边界 |
|---|---|---|
| 四阶段研究与实验闭环 | [StudyCoordinator](../reclab_agent/workflow/coordinator.py)、[业务工具](../reclab_agent/tools/) | 推荐实验流程已经存在，通用研究 DAG 另见差距 7 |
| 多角色协作 | [TeamAgent](../reclab_agent/agents/team_agent.py) | Planner、Executor、Reviewer 串行协作，共享上下文和预算，支持角色恢复与一次修订 |
| 实验及控制器恢复 | [StudyContext](../reclab_agent/tools/context.py)、[启动与恢复](../reclab_agent/application/launch.py)、[进程锁](../reclab_agent/execution/locking.py) | 已有待确认副作用、恢复时核验、预算保留和单控制器限制 |
| Worker 租约及停止机制 | [Worker](../reclab_agent/execution/worker.py)、[Watchdog](../reclab_agent/execution/watchdog.py)、[远端执行器](../reclab_agent/execution/remote_agent.py) | 不能把“整个项目没有租约或取消”当作缺口；缺的是统一 Run 层控制 |
| SQLite 与可追溯产物 | [RunStore 与 ArtifactStore](../reclab_agent/memory/storage.py)、[项目账本](../reclab_agent/memory/project.py) | 已有实验身份、哈希、Git 快照、事件和项目经验 |
| 多种检索及文献工具 | [KnowledgeIndex](../reclab_agent/memory/retrieval.py)、[文本模型](../reclab_agent/integrations/text_models.py)、[Tavily](../reclab_agent/integrations/tavily.py) | BM25、dense、hybrid、reranked 和联网接口均已存在；GraphRAG 不等同于普通向量检索 |
| Web 与事件重放 | [FastAPI 工作台](../reclab_agent/web/app.py)、[统一查询](../reclab_agent/application/runs.py) | 已有 SSE、Last-Event-ID、报告、Diff 和等待任务取消；默认没有装配真实执行器 |
| 评测与成本统计 | [evaluation](../reclab_agent/evaluation/)、[评测 CLI](../reclab_agent/cli/evaluate.py) | 已有冻结诊断、检索评测、Trace 和含缓存 token 的成本统计；不等于完整 Harness 指标 |

## 尚未完整实现的能力

### 1 完成前的统一质量门禁

**状态：部分实现。优先级：P0。**

参考仓库的 [ContractEvaluator](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/harness/contracts.py) 和 [确定性检查器](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/harness/verifiers/deterministic.py) 会在结束前检查来源匹配、最少证据、引用完整性、必需章节、主张支持、报告一致性和来源多样性；提供证据卡覆盖信息时再加入覆盖检查。因此源码中第八项是条件启用，不能把 README 的“8 项”理解为每种工作流都无条件执行八项。

RecLab 已有独立评分、比较资格检查和报告中的评分复核。但 [StudyCoordinator.run](../reclab_agent/workflow/coordinator.py) 最终根据是否有 `comparison` 设置 `completed`；[RunSession](../reclab_agent/core/session.py) 可以在模型返回非空文本后结束；TeamAgent 最后一次复核未通过时也可能返回 `completed`，另用 `review_approved` 表示复核结果。尚未发现统一、持久化、强制执行的完成合同。

**需要补齐：** 将“流程执行完毕”和“结论通过验证”分开，保存每项检查的期望值、观察值、原因及证据引用。针对 RecLab，优先检查独立评分完整性、比较条件一致性、保护指标、必要诊断、数值引用及结论一致性，无需照搬来源多样性规则。

**建议验收：** 构造缺评分、引用不存在、诊断缺失及 Reviewer 拒绝的案例，均不能显示为已验证完成，且能查询失败检查。

### 2 验证驱动的局部修复与动态重规划

**状态：部分实现。优先级：P0。**

参考的 [Runtime](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/harness/runtime.py)、[恢复决策](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/harness/recovery.py) 和 [工作流驱动器](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/harness/workflow.py) 区分错误重试、报告修补和证据不足后的 Replan。

RecLab 有模型重试、Reflection、TeamAgent 一次修订和多个候选实验；但四阶段协调器按配置的 `candidate_ids` 顺序执行，没有依据统一验证失败项生成新计划版本、补证任务和受限重规划的通用闭环。

**需要补齐：** 报告缺陷只修报告；缺诊断时先尝试从现有产物重算；确需新增实验时检查剩余预算和原授权范围；不可修复时保存明确终止原因。所有修复继续使用原预算。

**建议验收：** 缺引用不触发重新训练；缺诊断能够生成补证任务；预算不足终止；中断恢复不重复执行已完成修复。

### 3 统一证据账本及主张关联

**状态：部分实现。优先级：P0。**

参考的 [EvidenceLedger](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/harness/evidence.py) 为检索结果分配稳定证据身份，并关联 Run、任务、工具调用和来源。

RecLab 的实验记录、文献快照、JSON Pointer、评分哈希和 `evidence_paths` 已能追溯不少事实，但尚未形成覆盖所有证据类型的统一证据实体，也没有通用的“报告主张 → evidence_id → 工具调用 → 任务”关联与验证接口。[Trace 评测](../reclab_agent/evaluation/traces.py) 的路径集合或字符串出现检查，不能证明一句结论在语义上获得支持。

**需要补齐：** 在现有实验账本上增加统一证据索引及主张引用，保留类型、内容哈希、出处、所属 Run、工具调用、适用比较范围和原文位置。经验解释与真实测量使用不同类型，历史数值仍需回读原始评分。

**建议验收：** 从一条指标结论追溯到原始评分字段和对应实验；内容被修改后校验失败；未核验研究笔记不能直接支持已验证的收益结论。

### 4 全量证据卡与报告章节压缩

**状态：未实现参考管线。优先级：P1。**

参考的 [EvidenceCardPipeline](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/agents/multi_agent/reporter/evidence_cards.py) 实现逐条卡片、内容哈希复用、章节路由、分批摘要及覆盖记录，并与 [MapReduce 报告模块](https://github.com/SichengLong26/deepresearch_agent_harness/tree/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/agents/multi_agent/reporter/mapreduce) 配合。

RecLab 有检索分块、经验笔记和确定性 Markdown/HTML 报告，但未发现上述按章节预算组织全部研究证据的管线。当前经验卡片与这里用于本轮报告的 Evidence Card 用途不同。

**需要补齐：** 优先面向大量实验日志和文献，为每条证据保留回指；章节只接收相关卡片；超预算时递归汇总并保留完整证据 ID 集合；报告检查卡片覆盖与引用有效性。少量结构化实验的确定性报告可继续使用现有方式。

**建议验收：** 大量证据超过单次上下文容量时仍能生成报告，每个摘要可回到原始材料，覆盖集合能发现漏项。

### 5 阶段上下文预算及可信边界

**状态：部分实现。优先级：P1。**

参考的 [ContextBuilder](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/context/builder.py) 将输入组织为有预算的上下文块；[ContextCompactor](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/context/compactor.py) 管理长历史摘要。

RecLab 的 [RunSession.inputs](../reclab_agent/core/session.py) 持续追加模型输出和完整工具结果，TeamAgent 共享此历史。工具权限、研究阶段和笔记的未核验标记已存在，但缺少模型调用前统一执行的 token 打包、压缩和上下文来源分类。

**需要补齐：** 区分系统规则、原任务、已核验测量、外部文献、历史解释和角色建议；为阶段设置输入上限；对旧工具结果保留摘要及原文引用；避免外部材料中的指令改变原任务范围。

**建议验收：** 长工具输出不会无限增长后续输入；压缩后关键授权、失败事实、评分身份和未完成任务保持可用。

### 6 Run 层统一状态机及服务恢复

**状态：部分实现。优先级：P1。**

参考的 [状态机](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/harness/state_machine.py)、[Checkpoint](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/harness/checkpoints.py) 和 Runtime 将阶段、租约心跳、暂停、取消、恢复及检查点接入统一生命周期，并由恢复扫描处理遗留运行。

RecLab 已有控制器文件锁、Worker 租约与 Watchdog、原子 JSON 状态、角色检查点、CLI 恢复和实验去重。差距是这些机制分散于 `RunStore`、项目账本、`service-state.json`、`resources.json` 和角色状态；Web 查询通过 RunCatalog 汇合，但没有统一的 Run 级执行调度及重启恢复协议。研究状态有 `plan_hash` 校验，尚未发现与参考相当的通用检查点版本、迁移和完整性管理。

**需要补齐：** 统一权威状态和阶段迁移，定义安全暂停边界、恢复入口、检查点版本及过期控制器的处理。保留现有 Worker 的操作系统锁和副作用核验。

**建议验收：** API 进程中断后能识别未完成 Run；恢复预算和阶段一致；重复控制器不能同时执行；损坏或不兼容检查点明确拒绝恢复。

### 7 依赖图任务与隔离并行 Worker

**状态：未实现通用 DAG 调度。优先级：P2。**

参考的 [计划结构](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/agents/multi_agent/core/plan_spec.py) 和 [WorkerCoordinator](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/agents/multi_agent/executor/worker_coordinator.py) 支持任务依赖、串并行执行、隔离状态与确定性结果合并；源码支持 sequential/parallel，不能据此声称已支持完整自适应调度。

RecLab 的 [PlanExecuteAgent](../reclab_agent/agents/plan_execute_agent.py) 使用字符串步骤列表，TeamAgent 为串行角色交接；均没有通用任务节点、依赖边、就绪队列与隔离并行合并。

**需要补齐：** 先将文献检索、源码分析等无写副作用的任务做成依赖图；需要修改模型源码的任务继续串行或使用独立工作区。共享预算下原子预留并发资源。

**建议验收：** 无环校验、上游失败阻断下游、并发不越预算、相同输入得到稳定合并顺序。

### 8 多轮会话及追问消歧

**状态：未实现完整会话层。优先级：P2。**

参考的 [ChatService](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/backend/app/services/chat_service.py)、[QueryResolver](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/context/resolver.py)、[FTS Episodic Recall](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/memory/episodic.py) 和 [报告局部编辑](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/context/artifact_edit.py) 支持同一会话下多个 Run、历史按需召回和追问处理。

RecLab 的“研究会话”主要是一次研究任务的持久化目录；阶段对话可恢复，项目经验可检索，但当前 API 没有 Session/Message 对话模型和“继续分析上一轮第二个候选”这类解析链路。

**需要补齐：** 区分 Project、对话 Session 和研究 Run；维护消息身份、会话摘要和按需历史召回；支持澄清后继续任务；报告局部编辑应保留原始实验事实及版本关系。

**建议验收：** 连续追问能够定位正确实验；歧义时提出澄清；历史回答不能自动成为本轮新测量。

### 9 原子长期记忆及生命周期治理

**状态：部分实现。优先级：P2。**

参考的 [CuratedMemoryService](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/memory/curated.py) 和 [记忆策略](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/memory/policies.py) 管理原子记忆候选、来源、容量、确认、过期及快照。

RecLab 的 Project Wiki 保存实验、文献和经验，支持比较范围过滤、哈希和替代标记，`save_research_note` 也明确标注未核验解释。但尚无独立的偏好、事实、决策、程序经验记忆模型，以及候选确认、容量淘汰、过期和冻结记忆快照等完整治理。

**需要补齐：** 在项目笔记之外增加小容量原子记忆；记录适用范围和来源；将未经确认的解释与可自动注入的经验分开。正式测量继续由实验账本管理。

**建议验收：** 过期或被替代记忆不再自动注入；能追溯某次运行实际使用的记忆版本。

### 10 受控 Skill 学习与版本回滚

**状态：未实现。优先级：P2。**

参考的 [evolution 模块](https://github.com/SichengLong26/deepresearch_agent_harness/tree/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/evolution) 包含轨迹蒸馏、Lint、安全校验、离线评测、注册、加载、人工 Promote 和回滚；[PromotionPolicy](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/evolution/promotion.py) 还提供可配置的分阶段推广及运行反馈。

RecLab 可以存储经验解释，但未发现运行时 Skill 注册表、从已验证轨迹生成程序性候选、版本评测、人工启用、按任务加载、反馈停用及回滚机制。开发环境中的 Codex 技能与 RecLab 产品运行时 Skill 是不同系统。

**需要补齐：** 先定义小范围的“推荐实验研究步骤”Skill，绑定适用模型与数据协议；从已验证轨迹产生候选；通过冻结任务回放后才允许启用；Skill 不得扩展任务权限。

**建议验收：** 未通过评测或未获人工启用的候选不会进入提示词；版本变更可追踪；退化后能够切回上一版本。

### 11 Token 与验证阶段预算

**状态：部分实现。优先级：P1。**

参考的 [BudgetManager](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/harness/budgets.py) 管理 token、工具、重试、Replan 等预算；工作流为报告生成保留资源。

RecLab 已有共享模型请求、联网次数、训练次数、补丁次数和墙钟期限，并在请求前持久化预留；[costs.py](../reclab_agent/evaluation/costs.py) 已统计输入、输出及缓存 token。缺口是运行时 token 总量约束、通用工具调用额度、按阶段为报告及验证保留额度，以及与 DAG 并发衔接的预算机制。

**需要补齐：** 将 token 用量和下一次调用的输出上限接入预算决策；为最终报告和验证保留资源；缺 usage 时显式记录未知，不能作为零消耗。供应商未提供调用前准确 token 时，应明确预估与实际结算的边界。

**建议验收：** 研究阶段不会耗尽收尾额度；重试和复核同样扣额；恢复不重置额度；真实 token 用量超预估时能停止后续调用。

### 12 Web 真实执行与完整控制面

**状态：部分实现。优先级：P1。**

参考的 [RunService](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/backend/app/services/run_service.py) 和 [Run API](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/backend/app/api/v1/runs.py) 将实际运行调度、事件、证据、报告及暂停恢复等能力接入服务。

RecLab `create_app(root, run_task=None)` 支持注入同步或异步执行器，但默认 CLI Web 入口没有装配真实研究执行链。默认提交任务进入 `waiting_input`；API 只允许取消排队或等待任务，没有运行中研究的统一 pause/resume、证据账本查询、Memory/Skill 管理和系统评测接口。

**需要补齐：** 装配与 CLI 相同的可信研究服务，采用可恢复的后台执行调度；接通 Run 层暂停、恢复和取消确认。页面展示实际执行和停止状态，保持当前真实远端作业停止核验规则。

**建议验收：** Web 提交真实任务后可查询同一 Run；服务重启可以恢复；取消不会只修改页面状态；SSE 重连补齐实际执行事件。

### 13 面向真实轨迹的系统评测

**状态：部分实现。优先级：P1。**

参考的 [EvaluationService](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/evaluation/service.py) 和 [系统评测脚本](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/scripts/evaluate_runs.py) 从持久化轨迹统计验证完成、首轮通过、恢复、重复副作用、延迟分位数、资源和缓存指标，并接收独立标签。

RecLab 已有工具选择、引用路径、诊断决策、检索 Recall/MRR、冻结集与成本评测；尚缺统一汇总真实研究 Run 的 Verified Completion、首轮门禁通过、恢复成功、检查点完整性、重复副作用、Honest Failure，以及每个已验证 Run 的 token 和工具成本。当前缓存 token 已能统计，缺的是系统级命中率及与验证结果的联合分析。推荐模型本身的 nDCG 与文献检索 nDCG 是不同指标，不能互相替代。

**需要补齐：** 先统一轨迹身份和完成合同，再统计研究运行指标；新增人工或独立 Judge 的主张支持、报告质量标签，规则引用检查继续作为代理指标。

**建议验收：** 指标可从保存的轨迹重算；能区分模型失败、预算不足、恢复失败和证据不足；同时报告任务分母、未知 usage 和未完成运行。

### 14 私域 GraphRAG 及信息源策略

**状态：GraphRAG 未实现，来源治理部分实现。优先级：P3。**

参考的 [图谱模块](https://github.com/SichengLong26/deepresearch_agent_harness/tree/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/graph)、[Community](https://github.com/SichengLong26/deepresearch_agent_harness/tree/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/community) 和 [检索工具](https://github.com/SichengLong26/deepresearch_agent_harness/tree/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/search) 提供实体关系抽取、对齐合并、Neo4j 索引、社区摘要及局部和全局查询。[SourcePolicy](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/harness/policies.py) 在运行中约束 graphrag/web 工具集及联网参数。

RecLab 已有本地 Wiki 检索、联网文献和范围过滤，但没有 Neo4j 实体关系图及社区检索，也未发现参考式的 Run 来源模式冻结和统一来源匹配合同。

**需要补齐：** 若研究材料确实需要多跳关联，再增加图谱检索。来源策略建议适配推荐研究：允许显式声明实验测量、项目知识、外部文献的不同用途，规定联网查询可外发字段；不应机械采用整轮只能选一个来源，因为实验事实与文献解释本来需要配合。

**建议验收：** 图谱答案能回指原始文档；历史经验不能冒充当前实验结果；不允许外发的内部字段不能通过联网工具参数传出。

### 15 应用级迁移与备份恢复交付

**状态：部分实现。优先级：P2。**

参考有 [数据库迁移](https://github.com/SichengLong26/deepresearch_agent_harness/tree/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/src/deepresearch_agent/persistence/migrations)、[整套服务 Compose](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/docker-compose.yaml) 和 [备份恢复脚本](https://github.com/SichengLong26/deepresearch_agent_harness/blob/4c98d6e144fde5bc28c821e6ee2c3b980eecc9a0/scripts/backup_restore.py)。

RecLab 已有训练容器、本地启动器、SQLite 账本和 Worker 层版本处理，但尚未发现覆盖应用 RunStore、Project、角色检查点及 Artifact 的统一迁移与一致性备份恢复方案，也没有相当的整套 Web 服务部署编排。训练用 Docker 与应用整栈部署应分别评估。

**需要补齐：** 明确状态目录与模式版本，提供只读核验、升级和一致性备份；恢复后校验数据库、源码快照和产物引用的一致性。是否采用 Compose 取决于部署需求，不必为形式一致迁移技术栈。

**建议验收：** 从备份恢复后可读取历史报告、回溯证据和恢复未完成研究；缺少产物或版本不兼容时明确报错。

## 已有实现但仍待真实验收

以下状态依据当前 [README](../README.md)、[本地验收说明](non-gpu-work.md) 和 [实施方案](implementation-plan.md)，不归入“尚未编码”的清单。

| 项目 | 当前状态 | 还需要的验收 |
|---|---|---|
| Tavily 搜索和正文抓取 | 接口与预算、来源保存已有实现 | 配置 Key 后检查真实搜索、正文范围及引用 |
| 推荐研究四阶段与远端恢复 | 主流程、作业身份、恢复与停止核验已有实现 | 真实 SASRec 端到端、运行中断恢复和副作用去重 |
| GRU 推荐模型迁移 | 模型与配置已准备，已有 CPU 接口检查记录 | 真实 GPU 训练与独立评分 |
| 匿名 v2 诊断冻结集 | 场景和参考检查已有，工程试跑记录存在 | 正式模型对照；不能用参考规则通过率充当 Agent 成绩 |
| 多策略收益与最终交付 | 策略实现、历史实验和报告工具已有 | 真实编码任务同预算消融、最终质量验收和演示 |

## 建议实施顺序

1. **先保证完成结论可信：差距 1、3、2。** 复用现有评分器和实验账本，增加统一证据索引、完成合同及失败处理；先覆盖数值与比较条件，之后扩展文本主张。
2. **再保证长任务可控：差距 11、5、6、12、13。** 统一预算、上下文、状态和服务运行链，再使用真实轨迹统计恢复、质量与成本。
3. **材料规模增大后补充：差距 4、7、8、9、15。** 按证据量和产品需求引入章节压缩、DAG、对话、原子记忆及应用运维。
4. **最后引入演进与图谱：差距 10、14。** Skill 学习应依赖可靠的已验证轨迹和回放评测；GraphRAG 应由多跳检索需求驱动。

P0 表示建议最先补齐的质量缺口，P1 表示可靠长任务与服务化所需能力，P2 表示扩展能力，P3 表示按需能力。上述分级不意味着 RecLab 必须复制参考仓库的所有产品功能。
