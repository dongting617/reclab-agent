# 主研究者与独立审计：真实 API 回归评测

**本轮未完成完整对照。** 单 Agent 已尝试 30 例、通过 28 例；多 Agent 只尝试 3 例、通过 2 例，27 例因全局截止时间未发出请求。不能把多 Agent 的原始 2/30 当作能力分数，也不能据此宣布模式优劣。

顺序执行共享总截止时间导致后执行模式覆盖不足，这是本轮评测安排的局限。后续应只补测尚未运行的多 Agent 案例，并明确每批时间分配。

本轮状态：`partial`。未运行项与实际尝试数分开记录，部分覆盖不作为完整任务集成绩。

模型：`qwen3.7-flash`。每例最多 8 次请求，SDK 与 Agent 均不重试，输出上限 2048 token，单请求超时 40 秒。
本轮运行模式：`react`、`research_audit`。场景此前已分析，属于开发回归；场景变体相关，不当作独立样本。
两个模式使用相同任务、模型、主任务 Prompt 与评分规则。

| 集合 | 模式 | 实际尝试/计划 | 通过/计划 | 请求数 | 输入 Token | 输出 Token | 缺少 usage |
|---|---|---:|---:|---:|---:|---:|---:|
| dev | react | 30/30 | 28/30 | 64 | 93605 | 39585 | 2 |
| dev | research_audit | 3/30 | 2/30 | 10 | 19739 | 7757 | 1 |

请求总预留数：74。未核验实际账单，不把 Token 数等同于货币费用。

如实际覆盖不足，不将未运行项解释为模型能力失败；覆盖不齐的批次不用于宣布模式优劣。请求数从真实 Trace 重算，排除请求未发出的本地预算拒绝。

## 失败与审计

- dev / react：失败分类 `{"PROVIDER_ERROR": 1, "PROVIDER_TIMEOUT": 1}`。
- dev / research_audit：失败分类 `{"PROVIDER_ERROR": 1, "REQUEST_LIMIT": 27}`。
  审计前正确且引用达标 2 例，其中未放行 0 例；有效格式的错误决策 0 例，审计批准 0 例。
  正确回答未放行中，审计判断造成 0 例，其余属于引用/提交协议或服务失败。
  引用纠正 0 例，已解决 0 例；审计协议纠正 0 例，已解决 0 例。纠正仍消耗原预算，解决格式问题不等于端到端通过。
  决策正确但引用不达标且被审计批准 0 例。
  最终答案格式不合法 0 例，其中审计批准 0 例；终端严格解析仍会将其计为失败。

审计模式目前没有自动修订：它可以拦截结论，但不会改写研究者答案。判断收益需同时考虑错误放行、正确结论误拦和资源消耗。

原评分器遇到无效引用时可能同时将 decision_correct 置为 false。这里用冻结答案独立核对决策，单独归因为引用错误；不改变原始通过数、原始失败字段或报告。

原始批次 model_requests 是局部预留计数：截止后本地拒绝也可能先占用一次局部计数。汇总以 Trace 计数和全局请求账本核对，不把这些本地拒绝算作已调用 API。

## 当前结论与后续

本轮开发回归不足以单独证明应切换默认模式。需要区分进入审计前的研究者回答质量与审计门禁影响；正确回答未放行也要区分审计判断、提交协议与服务失败。
根据具体失败复核提交协议、引用和超时；后续使用独立报告和新额度验证。已观察过的失败属于开发用例，不把重跑成绩覆盖历史记录。

## 范围与复现

- 本轮未使用远程 GPU。CPU 编码未运行：Not included in this diagnostic evaluation。
- 不以合成决策结果宣称真实推荐收益、端到端研究能力或不同模型组合效果。
- 执行顺序固定，缓存与服务波动会影响耗时，不根据本轮宣称稳定速度优势。
- 原始失败与 Trace 保留，不补跑失败或覆盖报告提高分数。
- 原始汇总：`paired-role-prompts-20261124.json`；逐任务结果与审计见同名前缀的 JSON 和被忽略的 `*-traces/`。

```bash
python -m scripts.evaluate_research_audit --name YOUR_NEW_RUN_NAME --modes react research_audit --splits dev --max-requests 480 --minutes 50
python -m scripts.summarize_research_audit reports/YOUR_NEW_RUN_NAME.json
```

首条命令会调用真实 API；第二条只读取已有证据。新运行应使用新名称，不延长原预算或截止时间。

## 固定源码快照

本轮从源码副本运行，119 个代码、资源与任务文件均通过结束时哈希校验；未复制 API 凭据。源码目录：`/mnt/e/DongTing/简历/reclab-agent/.reclab/evaluation-snapshots/paired-role-prompts-20261124`。

如需精确复现，应使用该快照，并显式提供外部 API 配置及新的输出名称。下面命令会启动新一轮付费调用，不是补测剩余案例的命令：

```bash
cd "/mnt/e/DongTing/简历/reclab-agent/.reclab/evaluation-snapshots/paired-role-prompts-20261124"
python -m scripts.evaluate_research_audit --name YOUR_NEW_RUN_NAME --modes react research_audit --splits dev --max-requests 480 --minutes 50 --env-file "/mnt/e/DongTing/简历/reclab-agent/.reclab/agent.env" --output-dir "/mnt/e/DongTing/简历/reclab-agent/reports"
```
