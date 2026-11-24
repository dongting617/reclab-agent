# 持久化、事件关联与恢复验收

沿用 SQLite、JSON 检查点和 JSONL Trace，不增加服务或改变目录分层。研究协调器、Agent 和审计者共享同一个 TraceWriter；工作台通过 SSE 读取持久事件。研究报告的 `observability` 字段给出事件版本、研究会话 ID 和 Trace 相对路径。

## 事件如何关联

新写入的 `agent.jsonl` 使用 `schema_version=2`，保留原有事件名及字段，增加：

| 字段 | 含义 |
|---|---|
| `study_id` | 研究会话目录名；独立 Agent 未指定时使用自身运行 ID |
| `run_id` | 当前协调器或 Agent 运行 ID |
| `parent_run_id` | 研究者指向研究会话；审计者指向研究者运行 |
| `task_id` | 任务 ID |
| `candidate_id`、`experiment_id` | 候选与实际实验 ID；未知时为 null，不提前编造 |
| `stage`、`role` | 研究阶段与执行角色；`phase` 继续表示轮内细分步骤 |
| `seq` | Trace 内递增序号，重启后从已保存行数继续 |
| `timestamp` | UTC 时间戳；耗时使用单调时钟计算 |

协调器的 `stage_changed` 和宿主 `tool_call` 进入同一 Trace，Agent 工具返回实验 ID 后可关联后续事件。共享写入器通过线程锁串行追加；同一研究 Trace 仍要求单控制器拥有，不支持多个独立进程并发写入。

工作台 SSE 外层 `run_id` 继续是路由中的会话标识；`payload` 是统一事件，里面的 `run_id` 是实际 Agent 运行。SQLite 任务事件在读取时转换为相同事件格式。旧 Trace 仍可查看，缺失关联和时间字段保留 null，不推测历史事实，也不改写原文件。

## 一次模型请求记录什么

每次应用层 `backend.respond()` 调用记录：

1. `model_request_started`：唯一 `request_id`、轮次、尝试次数、配置模型名称及内容哈希。
2. `model_turn`：同一请求 ID、响应 ID、耗时、模型文字输出和服务返回的 token 用量。
3. `model_error`：同一请求 ID、耗时、异常类型、错误类别，以及请求是否在本地被拒绝。

重试使用新的请求 ID。只有开始事件而没有结果事件，表示该请求结果未知，不能当作成功或免费请求；预算检查点仍是额度的依据。SDK 内部重试可能包含在一次应用层调用内，不能把 Trace 条数当作供应商账单。启动探针等直接后端调用不经过 RunSession，此处记录范围是 Agent 会话内的请求。

内容身份使用 SHA-256：

- `task_prompt_hash`：启动任务文本。
- `instructions_hash`：本轮实际系统提示词。
- `input_hash`：发送时的实际对话历史。
- `tool_schema_hash`：本轮提供的工具定义。
- `prompt_hash`：上述系统指令、输入和工具定义的组合。

哈希用于比较内容版本，不是加密，也不能恢复原始请求。原有对话检查点和工具记录继续用于复查。`model` 是宿主配置的模型名称或别名，不保证供应商公开底层权重版本。

错误类别包括 `timeout`、`rate_limit`、`authentication`、`connection`、`invalid_json`、`provider_server`、`provider_request`、`local_rejection` 和 `unknown`。按异常类型、HTTP 状态和明确的本地拒绝标记分类，不根据错误消息中的关键词猜测。审计 JSON/结构错误继续保留独立协议错误记录。

## 已验证的中断边界

以下使用模拟模型和临时目录，不调用真实模型 API、训练或 GPU：

| 边界 | 注入方式 | 验收结果 |
|---|---|---|
| 模型超时后重试 | 第一轮抛出 TimeoutError | 请求 ID 分离、输入哈希一致、失败和成功均有耗时 |
| 审计请求已预留额度但尚无回复 | 子进程执行 `os._exit(23)`，父进程重新创建 Agent | 已用 3 次保留；完成后累计 5 次；研究工具只执行 1 次；未知请求可从 Trace 定位 |
| 工具可能已产生副作用但尚未保存结果 | 工具记录一次执行后抛出 KeyboardInterrupt | 恢复返回 `reconciliation_required`；不重放工具、不继续调用模型，需人工核验 |
| 事件记录后工作台对象重建 | 关闭写入器、重建 RunCatalog，再从旧游标读取 | 仅返回新增事件；序号连续，无重复旧事件 |
| 最后一个事件只写了一部分 | 构造不完整 JSONL 尾行 | 拒绝继续追加，保留现场，需先核验；读取端仍可展示此前完整事件 |

强制退出与恢复发生在不同进程，验证跨进程持久状态。它不是远端 GPU 故障验收，也不证明所有中断点都可自动恢复。

测试入口：

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_observability
```

本轮验收记录见 [离线验收报告](../reports/observability-offline-20261124.json)。

## 当前边界

未增加分布式追踪平台、日志脱敏与轮转、统一历史指标汇总或分层时间线界面。JSONL 追加不提供断电不丢失保证；不完整尾行不会自动删除。SQLite 与 JSON 文件仍不是跨存储事务。前端仍展示事件 JSON，全部中断点自动恢复仍待验证。
