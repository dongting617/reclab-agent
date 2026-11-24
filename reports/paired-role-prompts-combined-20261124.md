# 单 Agent 与多 Agent：评测集与完整结果

本轮两种模式均覆盖同一开发集的 30 道题：单 Agent 通过 **28/30（93.3%）**，多 Agent 通过 **27/30（90.0%）**。所有未通过案例均为 API 连接错误或超时。

## 评测集

- 使用冻结的 [diagnostic-v2 开发集](../tasks/diagnostic-v2/diagnostic-dev.json)，共 6 类，每类 5 题。每题给出结构化实验记录，包含训练状态、基线与候选的实验范围、验证集指标及资源状态，要求 Agent 读取证据、选择结论并引用相关字段。
- 题目由合成实验场景构造，数据集名称经过匿名化；同类题目是相关变体，不是 30 个独立真实训练实验。
- 本轮只评估读取实验记录后的决策与引用能力，没有评估真实训练、代码修改、推荐收益或完整研究流程；未包含 frozen 集和编码任务。

| 题型（标识） | 数量 | 题目考察内容 | 标准决策 |
|---|---:|---|---|
| 有效提升（`positive_gain`） | 5 | 训练完成，主指标 NDCG@20 提升，保护指标 Recall@20 不下降 | retain（保留） |
| 主指标退化（`primary_regression`） | 5 | 候选 NDCG@20 低于基线，即使 Recall@20 提升也不能保留 | reject（否定） |
| 保护指标退化（`guardrail_regression`） | 5 | NDCG@20 提升，但 Recall@20 下降 | reject（否定） |
| 训练失败（`failed_train`） | 5 | 训练状态为 failed，即使存在看似提升的指标也不能确认有效 | inconclusive（证据不足） |
| 评分缺失（`missing_score`） | 5 | 训练状态为 completed，但独立指标 metrics 缺失 | inconclusive（证据不足） |
| 随机种子不一致（`seed_mismatch`） | 5 | 基线与候选随机种子不同，不能直接比较指标 | inconclusive（证据不足） |

标准答案分布：保留 5 题、否定 10 题、证据不足 15 题。本轮没有标准答案为 stop（停止）的题目，也没有数据集不匹配、评估协议不匹配等题型；不能由本轮结果推断这些能力。

## 被测 Agent 与评分标准

| 模式 | 被测行为 |
|---|---|
| 单 Agent（`react`） | 一个研究者读取证据，直接输出决策、理由和引用 |
| 多 Agent（`research_audit`） | 主研究者输出结论；独立审计者使用隔离上下文和只读证据权限审核，审核通过后才放行 |

两种模式均使用 `qwen3.7-flash`，题目、主任务 Prompt 与评分规则一致；多 Agent 额外包含研究者和审计者的角色提示。每题最多 8 次请求。单 Agent 保留原结果，多 Agent 只补齐此前未运行题目，已尝试的失败案例没有重跑。

**一题通过须同时满足：Agent 完成任务、最终 JSON 符合格式、决策与标准答案一致、全部引用有效且至少一条定位到该题要求的相关证据。** 评分由固定 Python 程序执行，不使用另一个大模型打分；多 Agent 还须完成自身的审计放行流程。API 错误与超时计入未通过，但不解释为决策错误。

## 按题型比较

| 题型 | 单 Agent 通过/题数 | 单 Agent 通过率 | 多 Agent 通过/题数 | 多 Agent 通过率 | 未通过原因 |
|---|---:|---:|---:|---:|---|
| 有效提升 | 5/5 | 100% | 4/5 | 80% | 多：positive_gain-3（APIConnectionError） |
| 主指标退化 | 4/5 | 80% | 5/5 | 100% | 单：primary_regression-1（APIConnectionError） |
| 保护指标退化 | 5/5 | 100% | 5/5 | 100% | 无 |
| 训练失败 | 5/5 | 100% | 5/5 | 100% | 无 |
| 评分缺失 | 5/5 | 100% | 3/5 | 60% | 多：missing_score-3（APIConnectionError）；多：missing_score-4（APIConnectionError） |
| 随机种子不一致 | 4/5 | 80% | 5/5 | 100% | 单：seed_mismatch-1（APITimeoutError） |
| **合计** | **28/30** | **93.3%** | **27/30** | **90.0%** | 单 Agent 2 例；多 Agent 3 例 |

## 逐题结果

“通过”表示满足全部通过条件；错误名称表示本题因服务异常未完成，不是该题标准决策。

| 案例 | 标准决策 | 单 Agent | 多 Agent |
|---|---|---|---|
| `positive_gain-1` | retain | 通过 | 通过 |
| `positive_gain-2` | retain | 通过 | 通过 |
| `positive_gain-3` | retain | 通过 | APIConnectionError |
| `positive_gain-4` | retain | 通过 | 通过 |
| `positive_gain-5` | retain | 通过 | 通过 |
| `primary_regression-1` | reject | APIConnectionError | 通过 |
| `primary_regression-2` | reject | 通过 | 通过 |
| `primary_regression-3` | reject | 通过 | 通过 |
| `primary_regression-4` | reject | 通过 | 通过 |
| `primary_regression-5` | reject | 通过 | 通过 |
| `guardrail_regression-1` | reject | 通过 | 通过 |
| `guardrail_regression-2` | reject | 通过 | 通过 |
| `guardrail_regression-3` | reject | 通过 | 通过 |
| `guardrail_regression-4` | reject | 通过 | 通过 |
| `guardrail_regression-5` | reject | 通过 | 通过 |
| `failed_train-1` | inconclusive | 通过 | 通过 |
| `failed_train-2` | inconclusive | 通过 | 通过 |
| `failed_train-3` | inconclusive | 通过 | 通过 |
| `failed_train-4` | inconclusive | 通过 | 通过 |
| `failed_train-5` | inconclusive | 通过 | 通过 |
| `missing_score-1` | inconclusive | 通过 | 通过 |
| `missing_score-2` | inconclusive | 通过 | 通过 |
| `missing_score-3` | inconclusive | 通过 | APIConnectionError |
| `missing_score-4` | inconclusive | 通过 | APIConnectionError |
| `missing_score-5` | inconclusive | 通过 | 通过 |
| `seed_mismatch-1` | inconclusive | APITimeoutError | 通过 |
| `seed_mismatch-2` | inconclusive | 通过 | 通过 |
| `seed_mismatch-3` | inconclusive | 通过 | 通过 |
| `seed_mismatch-4` | inconclusive | 通过 | 通过 |
| `seed_mismatch-5` | inconclusive | 通过 | 通过 |

## 审计表现与结论

| 观察项 | 结果 |
|---|---:|
| 审计前研究者决策正确且引用达标 | 28 例 |
| 审计批准并最终通过 | 27 例 |
| 正确且引用达标，但因后续服务异常未完成 | 1 例 |
| 因审计判断误拦正确答案 | 0 例 |
| 可用于检验审计拦截能力的有效格式错误决策 | 0 例 |
| 审计协议格式纠正 | 1 次，成功解决 |
| 最终答案格式不合法 | 0 例 |

单 Agent 累计 64 次 API 请求，多 Agent 累计 124 次，约为单 Agent 的 1.94 倍。当前多 Agent 没有表现出更高的端到端通过率；两种模式完成的答案均通过评分，但失败样本主要反映服务可用性。由于缺少研究者错误决策样本，本轮无法证明独立审计能提高决策质量。多 Agent 分窗口补齐，服务状态差异也会影响结果。

原始逐题证据：[单 Agent](paired-role-prompts-20261124-dev-react.json)、[多 Agent](paired-role-prompts-combined-20261124-dev-research_audit.json)；[结构化汇总](paired-role-prompts-combined-20261124-summary.json)、[数据与来源核验](paired-role-prompts-combined-20261124-verification.json)。
