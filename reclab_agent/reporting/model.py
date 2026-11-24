"""Deterministic reports for people, built from recorded experiment evidence."""
from __future__ import annotations
from collections import Counter
import difflib
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import quote
from reclab_agent.infrastructure.files import atomic_write, confined_path, read_regular_file
from reclab_agent.reporting.progress import prepare_entries, progress_view


_STATUS = {"completed": "已结束", "incomplete": "已结束，结果不完整", "failed": "执行失败",
           "cancelled": "已取消", "running": "进行中"}
_MODE = {"single": "ReAct", "react": "ReAct", "plan_execute": "Plan and Execute",
         "research_audit": "主研究者 + 独立审计",
         "reviewer": "Reflection", "reflection": "Reflection", "strategy_comparison": "模式对比实验"}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _text(value, limit=1000):
    return str(value).replace("\x00", "")[:limit] if value is not None else "未记录"


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _fmt(value):
    if not _number(value):
        return "—"
    return f"{value:.6f}".rstrip("0").rstrip(".") if value else "0"


def _signed(value):
    return f"{value:+.6f}" if _number(value) else "—"


def _relative(value, baseline):
    return 100 * (value - baseline) / abs(baseline) if baseline else None


def _read(root, relative, maximum=2_000_000):
    if root is None or not isinstance(relative, str):
        return None
    try:
        return read_regular_file(confined_path(root, relative), max_bytes=maximum)
    except (OSError, ValueError):
        return None


def _json(root, relative):
    raw = _read(root, relative)
    try:
        return json.loads(raw) if raw is not None else None
    except (ValueError, UnicodeError):
        return None


def _href(base, relative, destination):
    if base is None or destination is None:
        return None
    try:
        path = confined_path(base, relative)
        if not path.is_file():
            return None
        # Only paths resolved inside the registered artifact root enter links.
        return quote(Path(os.path.relpath(path, destination)).as_posix(), safe="/._-")
    except (OSError, ValueError):
        return None


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _failure(record, root):
    failure = _text(record.get("failure"), 400) if record.get("failure") else ""
    experiment = record.get("experiment_id", "")
    if isinstance(experiment, str) and _ID.fullmatch(experiment):
        log = _json(root, f"experiments/{experiment}/train-log.json")
        evidence = "\n".join(_strings(log)) if log else ""
        if "nonfinite prediction logits" in evidence:
            return "验证预测出现非有限 logits（NaN/Inf），没有取得有效评分；日志未证明梯度爆炸。"
        if "RecursionError" in evidence:
            return "模型执行触发递归深度错误（RecursionError），没有取得有效评分。"
        if "CUDA out of memory" in evidence:
            return "GPU 显存不足（CUDA out of memory），没有取得有效评分。"
    if record.get("status") == "timed_out":
        return "作业超时，没有取得有效评分。"
    return failure or ("未获得可用评分。" if record.get("status") != "completed" else "评分未记录或尚未核验。")


def _changes(root, before, after, paths):
    if root is None or not _HASH.fullmatch(str(before)) or not _HASH.fullmatch(str(after)):
        return []
    changes = []
    for name in paths:
        if not isinstance(name, str):
            continue
        left = _read(root, f"snapshots/{before}/{name}", 1_000_000)
        right = _read(root, f"snapshots/{after}/{name}", 1_000_000)
        if left is None or right is None:
            continue
        try:
            old, new = (value.decode("utf-8").replace("\r\n", "\n") for value in (left, right))
        except UnicodeError:
            continue
        if old != new:
            diff = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                                fromfile=name, tofile=name))
            changes.append({"path": name, "diff": diff[:12_000], "truncated": len(diff) > 12_000,
                            "evidence": "冻结源码快照"})
    return changes


def _entry(value, root, destination, metrics):
    row = value if isinstance(value, dict) else {}
    record = row.get("record", {})
    record = record if isinstance(record, dict) else {}
    result = {"id": _text(record.get("candidate_id"), 160),
              "experiment_id": _text(record.get("experiment_id"), 160),
              "record": record, "metrics": None, "verified": False, "eligible": row.get("eligible", True),
              "audit_blocked": row.get("audit_blocked", False),
              "eligibility_recorded": "eligible" in row, "process_metrics": row.get("process_metrics"),
              "audit": row.get("audit"), "evaluation_hash": None,
              "warnings": [], "changes": [], "links": []}
    supplied = row.get("metrics")
    # Record-only entries intentionally have no scores, including after an
    # integrity failure. Never turn an LLM answer or a later file into a score.
    if record.get("status") == "completed" and isinstance(supplied, dict):
        if all(_number(supplied.get(name)) for name in metrics):
            result["metrics"] = {name: supplied[name] for name in metrics}
        else:
            result["warnings"].append("评分包含非有限值或缺少必要指标，未用于比较。")
    score = record.get("score")
    if result["metrics"] is not None:
        raw = _read(root, score.get("source"), 64_000_000) if isinstance(score, dict) else None
        if raw is None:
            result["warnings"].append("评分证据文件不可用，分数仅按实验记录展示，尚未重新核验。")
        else:
            try:
                grade = json.loads(raw)
                valid = (hashlib.sha256(raw).hexdigest() == score.get("sha256")
                         and all(grade["metrics"].get(name) == result["metrics"][name] for name in metrics)
                         and (not record.get("protocol_hash") or grade.get("protocol_hash") == record["protocol_hash"])
                         and ("seed" not in grade or grade["seed"] == record.get("seed")))
                if not valid:
                    raise ValueError("score evidence differs")
                result["verified"] = True
                result["evaluation_hash"] = grade.get("evaluation_hash")
            except (ValueError, KeyError, TypeError, UnicodeError):
                result["metrics"] = None
                result["warnings"].append("评分证据校验失败（文件哈希、协议或指标不一致），未用于比较。")
    result["failure"] = _failure(record, root) if result["metrics"] is None else ""
    experiment = record.get("experiment_id", "")
    if isinstance(experiment, str) and _ID.fullmatch(experiment):
        for label, name in (("训练记录", "train/training.json"), ("训练日志", "train-log.json"),
                            ("模型权重", "train/best.pt")):
            href = _href(root, f"experiments/{experiment}/{name}", destination)
            if href:
                result["links"].append({"label": label, "href": href})
        execution = _json(root, f"experiments/{experiment}/execution.json") or {}
        terminals = [item for item in execution.get("observations", [])
                     if isinstance(item, dict) and item.get("operation") == "terminal"]
        durations = [item["finished_at"] - item["started_at"] for item in terminals
                     if _number(item.get("finished_at")) and _number(item.get("started_at"))]
        result["server_seconds"] = sum(durations) if durations else execution.get("server_job_seconds")
    if isinstance(score, dict):
        href = _href(root, score.get("source"), destination)
        if href:
            result["links"].append({"label": "评分证据", "href": href})
    return result


def _compatible(entry, baseline):
    a, b = entry["record"], baseline["record"]
    return (type(a.get("seed")) is int and type(b.get("seed")) is int
            and bool(a.get("dataset_id")) and bool(b.get("dataset_id"))
            and bool(a.get("image") or a.get("runtime")) and bool(b.get("image") or b.get("runtime"))
            and bool(_HASH.fullmatch(str(a.get("protocol_hash"))))
            and bool(_HASH.fullmatch(str(b.get("protocol_hash"))))
            and all(a.get(key) == b.get(key) for key in ("seed", "protocol_hash", "dataset_id", "image", "runtime"))
            and entry.get("evaluation_hash") == baseline.get("evaluation_hash"))


def _group(name, entries, baseline, primary, guards, rounds, usage, calls, tokens):
    base_metrics = baseline.get("metrics")
    best = baseline
    for entry in entries:
        entry.update(deltas={}, relative={}, guardrail=None, decision="无有效评分", comparable=False,
                     comparison_eligible=False, milestone=False, previous_delta=None, plot_value=None,
                     best_value=best["metrics"][primary] if best["verified"] and best["metrics"] else None)
        if entry["metrics"] is None:
            continue
        if base_metrics is None:
            entry["decision"] = "基线无有效评分，无法比较"
            continue
        if not (entry["verified"] and baseline["verified"]):
            entry["decision"] = "评分证据尚未核验，不能确认提升"
            continue
        if not _compatible(entry, baseline):
            entry["decision"] = "种子、数据、协议或环境缺失或不一致，不能直接比较"
            entry["warnings"].append(entry["decision"])
            continue
        entry["comparable"] = True
        entry["plot_value"] = entry["metrics"][primary]
        entry["deltas"] = {key: entry["metrics"][key] - base_metrics[key] for key in entry["metrics"]}
        entry["relative"] = {key: _relative(entry["metrics"][key], base_metrics[key]) for key in entry["metrics"]}
        entry["guardrail"] = all(entry["metrics"][key] >= base_metrics[key] for key in guards)
        if entry.get("audit_blocked") or not entry.get("audit_pass"):
            entry["decision"] = "独立审计未通过或未完成，证据不足"
        elif not entry.get("process_pass"):
            entry["decision"] = "必需过程指标缺失、未达标或定义不可比，证据不足"
            entry["warnings"].append(entry["decision"])
        elif not entry["guardrail"]:
            entry["decision"] = "保护指标未达标"
        elif not entry.get("stop_confirmed"):
            entry["decision"] = "远端作业停止尚未确认，需先核验"
        else:
            entry["comparison_eligible"] = True
            if entry["metrics"][primary] > best["metrics"][primary]:
                entry["previous_delta"] = entry["metrics"][primary] - best["metrics"][primary]
                entry["milestone"] = True
                best = entry
                entry["decision"] = "优于基线且保护指标达标" if guards else "主指标提高，未配置保护指标"
            else:
                entry["decision"] = "未超过已记录最佳主指标"
        entry["best_value"] = best["metrics"][primary]
    failures = [entry for entry in entries if entry["metrics"] is None]
    scored = sum(entry["metrics"] is not None for entry in entries)
    skips = []
    for item in rounds:
        if not item.get("experiments"):
            status = item.get("agent_status", "unknown")
            errors = item.get("tool_error_codes", {})
            reason = {"invalid_plan": "规划输出未通过格式检查，未启动训练。",
                      "invalid_reflection": "本轮没有新增训练；后续反思输出未通过格式检查。",
                      "step_limit_exceeded": "模型调用额度用尽，本轮没有新增训练。"}.get(status,
                "本轮没有提交新的候选训练；可能受到预算或重复候选限制。")
            if errors.get("BUDGET_EXCEEDED") and errors.get("DUPLICATE_CANDIDATE"):
                reason = "补丁预算耗尽，重复候选请求被拦截，未启动新训练。"
            elif errors.get("DUPLICATE_CANDIDATE"):
                reason = "重复候选请求被拦截，未启动新训练。"
            skips.append({"id": _text(item.get("candidate_id", item.get("round"))), "reason": reason})
    seconds = [entry.get("server_seconds") for entry in entries if _number(entry.get("server_seconds"))]
    return {"name": _MODE.get(name, _text(name)), "key": name, "entries": entries, "best": best,
            "scored": scored, "failed": len(failures), "attempts": usage.get("experiment_attempts", len(entries)),
            "rounds": len(rounds), "skipped": skips, "calls": calls, "tokens": tokens,
            "patches": usage.get("patch_attempts"), "server_seconds": sum(seconds) if seconds else None}


def _normalize(report):
    """Read the archived one-off export as data, without adding an experiment mode."""
    if report.get("kind") != "one_off_agent_strategy_experiment":
        return report

    def entry(row):
        record = dict(row.get("record", {}))
        grade = row.get("independent_grade") or {}
        if grade.get("sha256") and _ID.fullmatch(str(record.get("experiment_id", ""))):
            record["score"] = {"source": f"experiments/{record['experiment_id']}/grade.json",
                               "sha256": grade["sha256"]}
        return {"record": record, "metrics": row.get("selection_metrics") or row.get("metrics")}

    provenance, protocol = report.get("provenance", {}), report.get("protocol", {})
    baseline = report.get("baseline", {})
    states = []
    for group in report.get("strategies", []):
        states.append({**group, "budget_usage": {"experiment_attempts": group.get("attempts", 0),
                                                 "patch_attempts": group.get("patch_attempts", 0)},
            "rounds": [{**item, "experiments": [entry(row) for row in item.get("experiments", [])]}
                       for item in group.get("rounds", [])]})
    context = {"task_question": "相同预算下的三种 Agent 模式模型优化实验",
               "target_model": (baseline.get("model") or {}).get("class"),
               "agent_mode": "strategy_comparison", "dataset_id": provenance.get("dataset_id"),
               "seed": provenance.get("seed"), "guardrail_metrics": protocol.get("guardrail_metrics", []),
               "editable_files": sorted({change["path"] for group in states for item in group["rounds"]
                                         for change in item.get("source_changes", []) if isinstance(change.get("path"), str)})}
    return {**report, "kind": "strategy_comparison", "status": report.get("status", {}).get("controller_status", "unknown"),
            "requires_reconciliation": report.get("status", {}).get("remote_stop_confirmed") is False,
            "baseline": entry(baseline), "strategies": states, "primary_metric": protocol.get("primary_metric", "ndcg@20"),
            "context": {**context, **(report.get("context") or {})}}


def build_report_view(report, *, output_directory=None, context=None):
    """Build a bounded, presentation-neutral view. Never execute model code."""
    if not isinstance(report, dict):
        raise ValueError("experiment report must be a JSON object")
    report = _normalize(report)
    if not isinstance(report.get("status", "unknown"), str):
        raise ValueError("unsupported experiment report status")
    if not isinstance(report.get("context", {}), dict) or (context is not None and not isinstance(context, dict)):
        raise ValueError("report context must be an object")
    for key in ("experiments", "strategies"):
        if key in report and (not isinstance(report[key], list) or any(not isinstance(row, dict) for row in report[key])):
            raise ValueError(f"report {key} must be a list of objects")
    explicit_context = context or {}
    context = {**(report.get("context") or {}), **explicit_context}
    destination = Path(output_directory).absolute() if output_directory is not None else None
    artifact_directory = context.get("artifact_directory", output_directory)
    root = Path(artifact_directory).absolute() if artifact_directory is not None else None
    primary = report.get("primary_metric") or context.get("primary_metric") or "ndcg@20"
    configured_guards = "guardrail_metrics" in context or "guardrail_metrics" in report
    guards = list(dict.fromkeys(context.get("guardrail_metrics", report.get("guardrail_metrics", []))))
    if not isinstance(primary, str) or any(not isinstance(name, str) for name in guards):
        raise ValueError("report metric names must be strings")
    if not configured_guards:
        source = report.get("baseline", {}).get("metrics") if isinstance(report.get("baseline"), dict) else None
        if not source:
            source = next((entry.get("metrics") for entry in report.get("experiments", [])
                           if isinstance(entry, dict) and entry.get("metrics")), {})
        guards = [name for name in source or {} if name != primary and name != "users"]
    metrics = [primary, *[name for name in guards if name != primary]]
    comparison = report.get("kind") == "strategy_comparison"
    paths = context.get("editable_files") or []
    groups = []
    baseline_row = report.get("baseline") if comparison else None
    if not comparison:
        baseline_id = report.get("baseline_id")
        baseline_row = next((entry for entry in report.get("experiments", [])
            if isinstance(entry, dict) and entry.get("record", {}).get("candidate_id") == baseline_id), None)
        if baseline_row is None and not baseline_id:
            baseline_row = next(iter(report.get("experiments", [])), None)
    baseline_root = root / "baseline" if root is not None and comparison else root
    baseline = _entry(baseline_row or {}, baseline_root, destination, metrics)
    baseline["id"] = "基线 · " + baseline["id"]
    baseline_hash = baseline["record"].get("source_hash")
    if _HASH.fullmatch(str(baseline_hash)):
        for name in paths:
            href = _href(baseline_root, f"snapshots/{baseline_hash}/{name}", destination)
            if href:
                baseline["links"].append({"label": "源码 · " + name, "href": href})
    if comparison:
        for state in report.get("strategies", []):
            name = state.get("strategy", "unknown")
            child = root / "strategies" / name if root is not None and _ID.fullmatch(str(name)) else None
            rounds = []
            for item in state.get("rounds", []):
                copied = dict(item)
                if not copied.get("tool_error_codes"):
                    raw = _read(child, f"traces/round-{item.get('round')}.jsonl")
                    try:
                        trace = [json.loads(line) for line in raw.splitlines() if line.strip()] if raw else []
                        copied["tool_error_codes"] = dict(Counter(
                            row["result"]["error"].get("code", "unknown") for row in trace
                            if row.get("event") == "tool_call" and isinstance(row.get("result"), dict)
                            and isinstance(row["result"].get("error"), dict)))
                    except (ValueError, UnicodeError, TypeError):
                        copied["tool_error_codes"] = {}
                rounds.append(copied)
            entries = []
            for item in rounds:
                for row in item.get("experiments", []):
                    entry = _entry(row, child, destination, metrics)
                    entry["changes"] = _changes(child, item.get("starting_source_hash"), entry["record"].get("source_hash"), paths)
                    if not entry["changes"]:
                        entry["changes"] = [{"path": _text(change.get("path")), "diff": _text(change.get("diff"), 12_000),
                                              "truncated": bool(change.get("truncated")), "evidence": "实验控制器记录的源码差异"}
                                             for change in item.get("source_changes", []) if isinstance(change, dict)]
                    entries.append(entry)
            prepare_entries(entries, report, name)
            groups.append(_group(name, entries, baseline, primary, guards, rounds,
                state.get("budget_usage", {}), state.get("model_calls"), state.get("model_usage", {})))
    else:
        rows = [row for row in report.get("experiments", []) if row is not baseline_row]
        entries = [_entry(row, root, destination, metrics) for row in rows]
        for entry in entries:
            entry["changes"] = _changes(root, baseline["record"].get("source_hash"), entry["record"].get("source_hash"), paths)
        mode = context.get("agent_mode", "single")
        agent = report.get("agent_result", {}) or {}
        usage = report.get("budget_usage", {})
        candidate_usage = {**usage, "experiment_attempts": max(len(entries), usage.get("experiment_attempts", len(entries)) - bool(baseline["record"]))}
        prepare_entries(entries, report, mode)
        groups.append(_group(mode, entries, baseline, primary, guards, [], candidate_usage,
                             report.get("model_calls", usage.get("model_calls", agent.get("steps"))),
                             report.get("model_usage", {})))
    for group in groups:
        for entry in group["entries"]:
            entry["display_id"] = group["name"] + " · " + entry["id"] if comparison else entry["id"]
            child = root / "strategies" / group["key"] if comparison and root is not None and _ID.fullmatch(str(group["key"])) else root
            source_hash = entry["record"].get("source_hash")
            if _HASH.fullmatch(str(source_hash)):
                for name in paths:
                    href = _href(child, f"snapshots/{source_hash}/{name}", destination)
                    if href:
                        entry["links"].append({"label": "源码 · " + name, "href": href})
    entries = [entry for group in groups for entry in group["entries"]]
    eligible = [group["best"] for group in groups if group["best"]["metrics"] is not None]
    best = max(eligible, key=lambda entry: entry["metrics"][primary], default=None)
    status = report.get("status", "unknown")
    improved = (best is not None and baseline["metrics"] is not None
                and best["metrics"][primary] > baseline["metrics"][primary])
    if baseline["metrics"] is None:
        summary = "基线没有可用评分，无法判断本轮优化效果。"
    elif not any(entry["metrics"] is not None for entry in entries):
        summary = "没有候选获得可用评分，本轮结论不充分。"
    elif improved:
        change = best["metrics"][primary] - baseline["metrics"][primary]
        relative = _relative(best["metrics"][primary], baseline["metrics"][primary])
        gain = f"（相对 {relative:+.2f}%）" if relative is not None else "（基线为零，不计算相对变化）"
        best_group = next((group["name"] for group in groups if group["best"] is best), "")
        best_label = best_group + " · " + best["id"] if comparison else best["id"]
        guard_text = "保护指标达标" if guards else "未配置保护指标"
        summary = (f"本轮记录的最佳候选为 {best_label}：{primary} 从 {_fmt(baseline['metrics'][primary])} "
                   f"提高到 {_fmt(best['metrics'][primary])}，绝对变化 {_signed(change)}{gain}，{guard_text}。")
        if not (best["verified"] and baseline["verified"]):
            summary += "评分证据尚未完整复核，结论待核验。"
    else:
        summary = "本轮没有找到评分已核验且满足全部接受条件的提升候选，保留基线。"
    if comparison and status != "completed":
        summary += " 各模式有效候选数量不完整，不宣布模式胜者。"
    if status in {"failed", "cancelled"}:
        summary = _STATUS[status] + "。" + summary + "已取得的分数不代表整个流程成功。"
    if report.get("requires_reconciliation") is True:
        summary += " 远端作业停止尚未确认，需要先核对作业状态。"
    limitations = ["由独立评分器对同一验证集的预测进行评估；验证集也用于 checkpoint 选择和候选搜索，结论仅适用于已记录的数据、配置和种子。"]
    if report.get("test_evaluated") is False:
        limitations.append("本轮评估范围为验证集。")
    elif report.get("test_evaluated") is not True:
        limitations.append("本轮仅依据已记录的验证集评分给出结论。")
    seeds = {entry["record"].get("seed") for entry in [baseline, *entries] if entry["record"].get("seed") is not None}
    if len(seeds) == 1:
        limitations.append("本轮使用一个固定随机种子进行训练和验证集评估；不据此宣称跨种子稳定提升或某种 Agent 模式普遍更优。")
    for entry in [baseline, *entries]:
        for warning in entry["warnings"]:
            if warning not in limitations:
                limitations.append(warning)
    limitations.append("未核验实际货币账单；只有记录中的调用量和作业耗时可用于本轮资源对照。")
    metadata = [("任务", context.get("task_question") or context.get("task_id") or report.get("session_id")),
                ("模型", context.get("target_model")), ("Agent 模式", _MODE.get(context.get("agent_mode"), context.get("agent_mode"))),
                ("数据集", context.get("dataset_id") or baseline["record"].get("dataset_id")),
                ("训练种子", context.get("seed") if context.get("seed") is not None else ", ".join(map(str, sorted(seeds))) or None)]
    json_href = _href(root, "report.json", destination)
    if explicit_context.get("report_json_path") is not None and destination is not None:
        json_path = Path(explicit_context["report_json_path"]).absolute()
        if json_path.is_file():
            json_href = quote(Path(os.path.relpath(json_path, destination)).as_posix(), safe="/._-")
    audit_notes = [_text(item.get("finding"), 2000) for item in report.get("model_implementation_audit_zh", [])
                   if isinstance(item, dict) and item.get("finding")]
    audit_notes += ["假设与风险：" + json.dumps(h, ensure_ascii=False)[:2000] for h in report.get("hypotheses", [])]
    audit_notes += ["独立审计 " + _text(candidate, 128) + "：" + json.dumps(value, ensure_ascii=False)[:4000]
                    for candidate, value in report.get("audits", {}).items()]
    audit_notes += ["过程指标：" + json.dumps(e["process_metrics"], ensure_ascii=False)[:2000]
                    for e in report.get("experiments", []) if e.get("process_metrics")]
    diagnostic = report.get("post_experiment_numerical_diagnostic") or {}
    audit_notes += ["工具失败：" + _text(x.get("tool")) + " / " + _text(x.get("code")) + " / " + _text(x.get("message"), 2000)
                    for x in report.get("tool_errors", [])]
    if report.get("archive_error"):
        limitations.append("项目归档未完整完成：" + _text(report["archive_error"]))
    audit_notes += [_text(note, 2000) for note in diagnostic.get("interpretation_zh", [])]
    return {"title": "模型优化实验报告", "status": _STATUS.get(status, _text(status)),
            "status_code": status, "summary": summary, "primary": primary, "guards": guards,
            "metrics": metrics, "metadata": [(key, _text(value)) for key, value in metadata if value is not None],
            "baseline": baseline, "groups": groups, "best": best, "improved": improved,
            "comparison": comparison, "entries": entries, "limitations": limitations,
            "progress": progress_view(groups, baseline, primary),
            "error": _text(report.get("error"), 600) if report.get("error") else None,
            "json_href": json_href, "audit_notes": audit_notes,
            "guard_note": "保护指标要求不低于基线。" if guards else "本次未配置保护指标，仅按主指标比较。",
            "wall_seconds": report.get("elapsed_seconds"),
            "resource_usage": report.get("resource_usage"),
            "total_attempts": report.get("budget_usage", {}).get("experiment_attempts",
                sum(group["attempts"] for group in groups) + bool(baseline["record"]))}
