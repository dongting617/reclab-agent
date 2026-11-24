"""Build a descriptive comparison from completed live evaluation evidence."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from reclab_agent.infrastructure.files import atomic_write
from reclab_agent.domain.contracts import canonical_json
from reclab_agent.core.llm import parse_json_response
from reclab_agent.evaluation.diagnostics import load_diagnostic_tasks, grade_answer, validate_answer
from reclab_agent.evaluation.costs import summarize_usage


def summarize(path):
    state = json.loads(path.read_text(encoding="utf-8"))
    rows, paired = [], []
    for batch in state["batches"]:
        report = json.loads(Path(batch["report"]).read_text(encoding="utf-8"))
        tasks = {r["id"]: r for r in load_diagnostic_tasks(
            Path("tasks/diagnostic-v2") / ("diagnostic-"+batch["split"]+".json"))[0]["tasks"]}
        answers, answer_texts, corrections = {}, {}, {}
        for item in report["results"]:
            raw = item.get("raw_answer", "")
            if item.get("trace"):
                checkpoint = Path(item["trace"] + ".audit.json")
                if checkpoint.exists():
                    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                    raw = raw or saved.get("answer", "")
                    corrections[item["id"]] = saved
            try:
                answer_texts[item["id"]] = raw
                answer = parse_json_response(raw)
                validate_answer(answer)
                answers[item["id"]] = {"decision_correct": answer["decision"] == tasks[item["id"]]["expected_decision"],
                                      "passed": grade_answer(answer, tasks[item["id"]])["passed"]}
            except (ValueError, TypeError, KeyError):
                pass
        durations = sorted(r["elapsed_ms"] for r in report["results"] if r.get("elapsed_ms") is not None)
        events, attempted_tasks = [], set()
        for trace in Path(batch["report"]).with_suffix("").with_name(Path(batch["report"]).stem+"-traces").glob("*.jsonl"):
            task_events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()]
            events.extend(task_events)
            if any(e.get("event") == "model_turn" or e.get("event") == "model_error" and not e.get("request_not_sent") for e in task_events):
                attempted_tasks.add(trace.stem)
        audit = batch["audit_results"]
        approved = lambda row: (row.get("audit") or {}).get("verdict") == "approved"
        decision_errors = [r for r in audit if r["id"] in answers and not answers[r["id"]]["decision_correct"]]
        correct_answers = [r for r in audit if r["research_answer_passed"]]
        failures = Counter()
        for item in report["results"]:
            if item["passed"]:continue
            code = item.get("failure", {}).get("code", item.get("error_type", "unknown"))
            if code == "INCORRECT_DECISION" and answers.get(item["id"], {}).get("decision_correct"):
                code = "CITATION_INVALID_OR_NOT_LOCATED"
            failures[code] += 1
        row = {"split": batch["split"], "mode": batch["mode"], "tasks": report["tasks"],
            "attempted_tasks": len(attempted_tasks), "not_attempted_tasks": report["tasks"]-len(attempted_tasks),
            "passed": report["passed"], "not_run": sum(r["status"] == "not_run" for r in report["results"]),
            "failure_codes": dict(failures), "original_failure_codes": batch["failures"], "resource_usage": summarize_usage(events),
            "batch_reservation_counter": batch["model_requests"],
            "median_elapsed_ms": durations[len(durations)//2] if durations else None,
            "total_elapsed_ms": sum(durations), "report": batch["report"]}
        if audit:
            row["audit"] = {"verdicts": dict(Counter((r.get("audit") or {}).get("verdict", "not_completed") for r in audit)),
                "research_answer_passed_before_audit": len(correct_answers),
                "correct_answers_not_released": sum(r["agent_status"] != "completed" for r in correct_answers),
                "correct_answers_blocked_by_audit_decision": sum((r.get("audit") or {}).get("verdict")
                    in {"rejected", "insufficient_evidence"} for r in correct_answers),
                "valid_research_decision_errors": len(decision_errors),
                "decision_errors_approved_by_auditor": sum(approved(r) for r in decision_errors),
                "citation_failures_approved": sum(approved(r) and r["id"] in answers
                    and answers[r["id"]]["decision_correct"] and not answers[r["id"]]["passed"] for r in audit),
                "citation_correction_cases": sum(s.get("citation_repairs", 0) > 0 for s in corrections.values()),
                "citation_correction_resolved": sum(s.get("citation_repairs", 0) > 0 and
                    (s.get("citation_check") or {}).get("status") == "passed" for s in corrections.values()),
                "audit_correction_cases": sum(s.get("audit_repairs", 0) > 0 for s in corrections.values()),
                "audit_correction_resolved": sum(s.get("audit_repairs", 0) > 0 and bool(s.get("audit")) for s in corrections.values()),
                "answers_missing": sum(not answer_texts.get(r["id"]) for r in audit),
                "invalid_answer_shapes": sum(bool(answer_texts.get(r["id"])) and not r["answer_valid"] for r in audit),
                "invalid_answer_shapes_approved": sum(not r["answer_valid"] and approved(r) for r in audit)}
        rows.append(row)
    for split in state["protocol"]["splits"]:
        records = {b["mode"]: json.loads(Path(b["report"]).read_text(encoding="utf-8")) for b in state["batches"] if b["split"] == split}
        if set(records) != {"react", "research_audit"}:continue
        covered = [r for r in rows if r["split"] == split]
        if any(r["attempted_tasks"] != r["tasks"] for r in covered):
            paired.append({"split": split, "comparison_valid": False, "reason": "coverage_incomplete"})
            continue
        first = {r["id"]: r for r in records["react"]["results"]}
        second = {r["id"]: r for r in records["research_audit"]["results"]}
        paired.append({"split": split, "comparison_valid": True, "both_passed": sum(first[i]["passed"] and second[i]["passed"] for i in first),
            "only_react_passed": [i for i in first if first[i]["passed"] and not second[i]["passed"]],
            "only_audit_mode_passed": [i for i in first if not first[i]["passed"] and second[i]["passed"]],
            "both_failed": [i for i in first if not first[i]["passed"] and not second[i]["passed"]]})
    output = {"status": state["status"], "protocol": state["protocol"], "reserved_requests": state["requests"],
              "batches": rows, "paired": paired, "billing_verified": False}
    verification_path = path.with_name(path.stem + "-verification.json")
    verification = json.loads(verification_path.read_text(encoding="utf-8")) if verification_path.exists() else None
    if verification:
        output["workspace_verification"] = verification
    target = path.with_name(path.stem + "-summary.json")
    atomic_write(target, canonical_json(output).encode("utf-8"))
    text = ["# 主研究者与独立审计：真实 API 回归评测", "",
        "本轮状态：`" + state["status"] + "`。未运行项与实际尝试数分开记录，部分覆盖不作为完整任务集成绩。", "",
        "模型：`"+state["protocol"]["model"]+"`。每例最多 8 次请求，SDK 与 Agent 均不重试，输出上限 2048 token，单请求超时 40 秒。",
        "本轮运行模式：`" + "`、`".join(state["protocol"]["modes"]) + "`。场景此前已分析，属于开发回归；场景变体相关，不当作独立样本。",
        "本轮未安排的模式没有新增调用；历史结果只能作开发参照，不能作为同步对照。" if len(state["protocol"]["modes"]) == 1 else
        "两个模式使用相同任务、模型、主任务 Prompt 与评分规则。", "",
        "| 集合 | 模式 | 实际尝试/计划 | 通过/计划 | 请求数 | 输入 Token | 输出 Token | 缺少 usage |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        u = r["resource_usage"]
        text.append(f"| {r['split']} | {r['mode']} | {r['attempted_tasks']}/{r['tasks']} | {r['passed']}/{r['tasks']} | {u['model_requests']} | {u['input_tokens']} | {u['output_tokens']} | {u['requests_without_usage']} |")
    text += ["", "请求总预留数："+str(state["requests"])+"。未核验实际账单，不把 Token 数等同于货币费用。", "",
             "如实际覆盖不足，不将未运行项解释为模型能力失败；覆盖不齐的批次不用于宣布模式优劣。请求数从真实 Trace 重算，排除请求未发出的本地预算拒绝。", "",
             "## 失败与审计", ""]
    for r in rows:
        text += [f"- {r['split']} / {r['mode']}：失败分类 `{json.dumps(r['failure_codes'],ensure_ascii=False)}`。"]
        if r.get("audit"):
            a = r["audit"]
            text += [f"  审计前正确且引用达标 {a['research_answer_passed_before_audit']} 例，其中未放行 {a['correct_answers_not_released']} 例；有效格式的错误决策 {a['valid_research_decision_errors']} 例，审计批准 {a['decision_errors_approved_by_auditor']} 例。",
                     f"  正确回答未放行中，审计判断造成 {a['correct_answers_blocked_by_audit_decision']} 例，其余属于引用/提交协议或服务失败。",
                     f"  引用纠正 {a['citation_correction_cases']} 例，已解决 {a['citation_correction_resolved']} 例；审计协议纠正 {a['audit_correction_cases']} 例，已解决 {a['audit_correction_resolved']} 例。纠正仍消耗原预算，解决格式问题不等于端到端通过。",
                     f"  决策正确但引用不达标且被审计批准 {a['citation_failures_approved']} 例。",
                     f"  最终答案格式不合法 {a['invalid_answer_shapes']} 例，其中审计批准 {a['invalid_answer_shapes_approved']} 例；终端严格解析仍会将其计为失败。"]
    text += ["", "审计模式目前没有自动修订：它可以拦截结论，但不会改写研究者答案。判断收益需同时考虑错误放行、正确结论误拦和资源消耗。", "",
             "原评分器遇到无效引用时可能同时将 decision_correct 置为 false。这里用冻结答案独立核对决策，单独归因为引用错误；不改变原始通过数、原始失败字段或报告。", "",
             "原始批次 model_requests 是局部预留计数：截止后本地拒绝也可能先占用一次局部计数。汇总以 Trace 计数和全局请求账本核对，不把这些本地拒绝算作已调用 API。", "",
             "## 当前结论与后续", "",
             "本轮开发回归不足以单独证明应切换默认模式。需要区分进入审计前的研究者回答质量与审计门禁影响；正确回答未放行也要区分审计判断、提交协议与服务失败。",
             "根据具体失败复核提交协议、引用和超时；后续使用独立报告和新额度验证。已观察过的失败属于开发用例，不把重跑成绩覆盖历史记录。", "",
             "## 范围与复现", "",
             "- 本轮未使用远程 GPU。CPU 编码未运行：" + state["protocol"]["coding"]["reason"] + "。",
             "- 不以合成决策结果宣称真实推荐收益、端到端研究能力或不同模型组合效果。",
             "- 执行顺序固定，缓存与服务波动会影响耗时，不根据本轮宣称稳定速度优势。",
             "- 原始失败与 Trace 保留，不补跑失败或覆盖报告提高分数。",
             "- 原始汇总：`"+path.name+"`；逐任务结果与审计见同名前缀的 JSON 和被忽略的 `*-traces/`。", "",
             "```bash", "python -m scripts.evaluate_research_audit --name YOUR_NEW_RUN_NAME --modes " +
             " ".join(state["protocol"]["modes"]) + " --splits " + " ".join(state["protocol"]["splits"]) +
             " --max-requests " + str(state["protocol"]["global_request_cap"]) + " --minutes " + str(state["protocol"]["minutes"]),
             "python -m scripts.summarize_research_audit reports/YOUR_NEW_RUN_NAME.json", "```", "",
             "首条命令会调用真实 API；第二条只读取已有证据。新运行应使用新名称，不延长原预算或截止时间。", ""]
    if verification and not verification["workspace_sources_match_started_hash"]:
        text += ["## 来源核对", "", "启动时来源哈希保留在原始协议中。汇总时以下工作区文件与启动哈希不同：`" +
                 "`、`".join(verification["changed_workspace_files"]) +
                 "`。未覆盖本地编辑；评测进程不热重载模块，当前工作区不能直接视作启动时的精确源码副本。", ""]
    atomic_write(path.with_suffix(".md"), "\n".join(text).encode("utf-8"))
    print(canonical_json({"summary": str(target), "batches": [{"split": r["split"], "mode": r["mode"], "passed": r["passed"]} for r in rows]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    summarize(parser.parse_args().report)
