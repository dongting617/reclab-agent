# 四阶段研究、实验账本与记忆使用说明

代码已按职责重组，当前目录与导入路径以[项目架构与代码导航](architecture.md)为准。

更新日期：2026-11-23。实现依据：[实施方案](implementation-plan.md)，过程记录见[仓库优化记录](optimization-log.md)。

## 研究流程与基线

标准入口使用四阶段协调器：**理解背景 → 提出假设 → 模型实验 → 结果分析和得出结论**。ReAct、PlanExecute、Reflection 是阶段内执行策略，不是业务阶段。

理解背景时先解析基线：显式指定的历史报告优先，其次使用项目登记的当前已接受方案，再查找匹配的初始基线。没有可复用记录时，在背景阶段调用实验工具建立基线，不要求先生成优化假设。背景阶段不能修改模型或运行候选。

历史复用核对数据及划分、评分协议、源码、配置、种子、运行策略和作业限制，并验证评分与 checkpoint 的文件哈希。只有报告文字、缺少 checkpoint、损坏证据或条件不匹配的记录不能直接复用。复用本身不占新的训练次数。

候选必须先通过 `record_hypothesis` 登记修改、预期收益链路、证据、风险和过程指标。实验后由确定性分析工具比较独立验证集评分，决定保留、否定或证据不足。下一轮从当前最佳源码出发，训练仍从头进行，不继承旧权重。

## 启动与项目资源

Windows 继续使用 `start.cmd`；`start.cmd -Plan` 只检查任务与范围。以下 Python 命令适用于已经配置好 Python 3.10+ 环境的控制端，真实模型调用需要 OpenAI 可选依赖和本地凭据。

```bash
python -B -m reclab_agent.cli.research project init --config configs/beauty-sasrec-task.json
python -B -m reclab_agent.cli.research plan --config configs/beauty-sasrec-task.json
python -B -m reclab_agent.cli.research run --config configs/beauty-sasrec-task.json --env-file .reclab/agent.env --approve-plan
```

WSL 使用 Windows OpenSSH 连接时，为 `run` 或 `resume` 增加 `--windows-ssh`。标准启动器已经处理运行环境选择。

任务会在首次运行时自动建立项目基础资源，也可以提前执行 `project init` 建立初始 Git 提交。默认项目目录是任务原始输出目录下的 `project/`；每次运行仍使用独立的 `sessions/<session_id>/`。

```text
project/
  project.json       初始建模目标、模型、数据入口与评估范围
  ledger.sqlite3     运行、事件、实验、当前已接受方案及知识索引元数据
  code/              独立本地 Git，仅存模型与诊断源码
  wiki/
    experiments/     按实验生成的 Markdown 记录
    experience/      条件、观察、解释和决策卡片
    literature/      文献内容、覆盖范围及来源
  artifacts/         文献缓存等项目产物

sessions/<session_id>/
  plan.json / review.json
  report.json / report.md / report.html
  resources.json / service-state.json
  conversations/     阶段会话检查点
  snapshots/        完整源码快照
  experiments/      checkpoint、日志、预测、诊断及独立评分
```

账本引用会话产物的实际路径，不要求重复复制大型数据和权重。请一并保留项目目录和被引用的会话目录。项目 Git 不修改原源码仓库，也不自动推送远端。

## 可选配置与预算

旧版任务配置仍可读取。可以在顶层增加以下 `research` 对象；其他必需配置保持原格式：

```json
{
  "research": {
    "project_directory": "../.reclab/projects/beauty-sasrec",
    "max_wall_seconds": 2400,
    "max_network_requests": 10,
    "tavily_key_env": "TAVILY_API_KEY"
  }
}
```

路径相对于任务配置文件。项目目录不能与模型源码或数据目录重叠。时间和请求限制必须为正整数。没有单独配置总时限时，使用总作业秒数加上模型请求超时预留。

`task.budget.max_model_turns` 在标准协调器中同时限制实际模型请求尝试，包括启动预检、重试、规划和反思。训练尝试在提交前预留次数及时间，失败同样计入。实验时间包含训练和预测，默认按 80%/20% 分配。模型、网络和实验操作均检查任务截止时间，调用还具有各自的超时。

资源预留和实际已启动 GPU 训练不同；SSH 连接失败也可能已经消耗一次实验尝试。`max_cost_cny` 不代表人民币费用已经硬封顶，报告不将 tokens 或作业耗时等同于已核验账单。

默认基线与候选使用同一个种子；当前任务配置为 2026。通用 API 的历史种子设置不被重写，不增加三个种子的重复训练或最终测试阶段。

需要显式复用历史基线时，在 `experiment` 中增加：

```json
{
  "baseline_report": "../path/to/session/report.json"
}
```

标准归档为可复用实验生成 `baseline-<experiment_id>.json`，包含证据清单，也可将它作为 `baseline_report`。指定的历史记录无效时会明确失败，不静默换一个基线。

## 恢复、查询与知识维护

```bash
python -B -m reclab_agent.cli.research resume --config configs/beauty-sasrec-task.json --session path/to/sessions/session-id --env-file .reclab/agent.env --approve-plan
python -B -m reclab_agent.cli.research project runs --config configs/beauty-sasrec-task.json
python -B -m reclab_agent.cli.research project experiments --config configs/beauty-sasrec-task.json
python -B -m reclab_agent.cli.research project experiments --config configs/beauty-sasrec-task.json --experiment-id exp-id
python -B -m reclab_agent.cli.research project search --config configs/beauty-sasrec-task.json --query "非有限 logits 的历史观察"
python -B -m reclab_agent.cli.research project reindex --config configs/beauty-sasrec-task.json
python -B -m reclab_agent.cli.research project import --config configs/beauty-sasrec-task.json --input path/to/report.json
python -B -m reclab_agent.cli.research project note --config configs/beauty-sasrec-task.json --note-id revised-observation --input notes.md --replaces old-observation
```

恢复保持原会话、截止时间、请求次数和实验预留，先核对已有记录或远程作业。任务身份、数据、运行策略或训练限制变化时拒绝恢复。已完成任务应新建运行；训练进程终止后的重新训练不被视为中间 epoch 续训。

历史导入按实验身份去重；证据缺失时保留未核验状态。新任务自动加载相关经验，知识检索按项目中的数据、协议、模型接口和种子范围过滤，默认返回 5 条带位置和哈希的结果。不同范围的指标不能因为文字相似就直接比较。

经验卡片由程序从实验事实和结构化假设生成，模型解释单独标注；模型通过 `save_research_note` 保存的笔记明确标记为未核验解释。Wiki 可以修订和重建索引，原始实验记录不被摘要覆盖。

## 文献调研

在控制端环境或本地 `.reclab/agent.env` 配置 `TAVILY_API_KEY`，实际变量名可以通过任务配置修改。Key 不写入任务 JSON、Wiki、Git 或报告。

`search_literature` 搜索论文及官方来源，`fetch_literature` 提取选定页面。检索片段标记为 `snippet_only`，提取正文标记为 `extracted_text`；提取文本并不自动证明整篇论文已完整获取。内容保存 URL、时间、哈希及位置，重复请求可复用缓存。

未配置 Key 时返回 `SEARCH_NOT_CONFIGURED`，仍可使用本地知识；网络失败消耗已预留请求，但不生成“调研成功”的结果。接口依据 [Tavily Search](https://docs.tavily.com/documentation/api-reference/endpoint/search) 与 [Tavily Extract](https://docs.tavily.com/documentation/api-reference/endpoint/extract)。

## 过程指标和诊断代码

`record_hypothesis` 的 `process_metrics` 为指标定义列表：

```json
[
  {
    "name": "coverage",
    "purpose": "required",
    "expected": "覆盖率至少为 0.5",
    "operator": ">=",
    "threshold": 0.5
  }
]
```

`purpose` 为 `diagnostic` 或 `required`。诊断用途可以将 `operator` 和 `threshold` 设置为 null；必需条件必须提供有限数值阈值及比较操作。

实验阶段通过 `write_diagnostic` 写入 `src/diagnostics/<name>.py`，提供以下接口：

```python
def evaluate(context):
    predictions = context["predictions"]
    return {"coverage": len({item for ranking in predictions.values() for item in ranking})
                         / context["data"]["num_items"]}
```

SASRec 的预测 Worker 提供 model、data、config、device、predictions 和 selected 上下文，输出带定义哈希的 `predict/process_metrics.json`。控制端只读 JSON，不导入候选诊断代码。默认训练器在参数更新前执行有限值及形状 Smoke Test，并输出 `train/smoke.json`。

`run_diagnostics` 读取过程证据，`analyze_experiment` 结合正式验证集指标作出决策。必需指标缺失、阈值不满足或基线与候选的诊断定义不可比时，决策为证据不足，不能被报告判为成功改进。新增诊断代码后，历史基线可能没有匹配证据，此时不会补造数据或自动追加训练。

其他推荐后端可以实现同样的诊断产物协议；已有未实现此协议的模型仍能训练和评分，但不能声称完成过程指标验证。

## 验证范围

离线回归覆盖工具权限、四阶段、基线复用、账本、Git、检索、恢复及诊断决策，真实模型接口另行验证工具 schema。2026-11-23 的真实 GPU 验收在背景阶段被当前 SSH 连接拒绝阻断，没有获得新训练评分；Tavily 缺少本地 Key，尚未联网实测。后续状态以[验收记录](../reports/implementation-20261123.json)及仓库优化记录为准。
