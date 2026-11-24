# tools 目录说明

这里包含模型可调用的业务工具，以及注册、组合、状态和校验等支持代码。
文件中的普通函数不一定是 Agent 工具；只有被工具箱暴露在 `specs` 中，并由 `invoke()` 分派的能力，才可供模型调用。

## 业务工具与工具箱

| 文件 | 实现 | 职责 |
|---|---|---|
| `evidence.py` | `EvidenceToolbox` | 读取、比较已有证据，以及两项内置报告诊断 |
| `workspace.py` | `WorkspaceTools` | 读取源码、应用补丁、校验候选 |
| `planning.py` | `PlanningTools` | 登记可检验的研究假设 |
| `experiments.py` | `ExperimentTools` | 运行和查询实验、读取日志和历史实验 |
| `baselines.py` | `BaselineTools` | 解析和复用可比基线 |
| `diagnostics.py` | `DiagnosticTools` | 编写过程诊断、检查实验过程指标 |
| `evaluation.py` | `EvaluationTools` | 分析实验结果 |
| `knowledge.py` | `KnowledgeTools` | 检索经验、保存研究笔记 |
| `literature.py` | `LiteratureTools` | 搜索文献、提取正文 |
| `remote.py` | `RemoteToolbox` | SSH 上传下载、远端作业与 GPU 状态工具；传输实现位于 `execution/ssh.py` |
| `coding.py` | `CodingToolbox` | 编码评测专用工具箱，限制补丁与公开检查次数 |
| `audit.py` | `AuditToolbox` | 只读审计证据工具；同文件还包含审计提交格式和宿主校验函数 |

`evidence.py` 的六个工具按职责命名：

| 模型可调用名称 | 作用 |
|---|---|
| `list_evidence` | 列出证据文件 |
| `inspect_artifact` | 检查文件标识、大小和 JSON 顶层信息 |
| `read_json_values` | 按 JSON Pointer 读取字段 |
| `compare_numeric_values` | 比较两个已记录数值的绝对及相对变化 |
| `diagnose_beam_gap` | 分析内置 G2 报告中的 Beam Search 漏召回 |
| `diagnose_frequency_gap` | 分析内置 B1/G2 报告中的频率分桶差距 |

后两项针对内置报告，不是任意实验通用的诊断器。工具调用名称保持不变，以保留历史任务和评测协议；工具箱类名从泛化的 `RecLabToolbox` 改为 `EvidenceToolbox`。

## 支持代码：不是业务工具

| 文件 | 职责 |
|---|---|
| `__init__.py` | 包的公开导出 |
| `base.py` | `Toolbox` 接口、`Tool` 抽象类及 `FunctionTool` 函数适配器 |
| `contracts.py` | `ToolSpec` 工具描述和 `ToolError` 错误类型 |
| `registry.py` | `ToolRegistry` 注册、组合、分派工具；内部 `invoke_tool` 包装单个工具，公开 `invoke` 按名称分派 |
| `tool_group.py` | 能力组共享上下文和注册逻辑 |
| `study_context.py` | `StudyContext` 保存研究阶段、权限、预算、实验记录和恢复状态 |
| `optimization_toolbox.py` | `OptimizationToolbox` 组装业务能力组，不直接实现业务处理函数 |
| `catalog.py` | 能力清单，不是执行注册表 |
| `citation_validation.py` | 根据已读取证据检查引用，不向模型开放新文件读取 |
| `connectivity.py` | 启动时的 Function Calling 兼容性探针定义，不属于研究实验工具 |

## 命名与调用关系

- `*Toolbox`：同时提供 `specs` 和 `invoke()` 的工具箱。
- `*Tools`：共享 `StudyContext` 的业务能力组，由优化工具箱组装。
- `ToolSpec`：给模型的说明及参数格式；实际函数保留在宿主程序中。
- 辅助函数、上下文对象和目录清单不会因放在 `tools/` 下而自动获得模型调用入口。

```text
业务工具组 → ToolRegistry → OptimizationToolbox
                                  ↓
RunSession.turn()：把工具定义发给模型
                                  ↓
RunSession.execute()：校验调用并交给 toolbox.invoke()
                                  ↓
实际工具函数 → 结果追加到 inputs → 下一轮模型调用
```

证据工具箱直接维护名称到方法的映射；优化工具箱使用 `ToolRegistry`。两者遵循相同的 `Toolbox` 接口。
