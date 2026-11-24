"""Build the initial synthetic local evaluation corpus; refuses to overwrite a freeze."""
import argparse
import hashlib
import json
from pathlib import Path

from reclab_agent.domain.contracts import content_hash


def build(root):
    root = Path(root)
    target = root / "tasks"
    lock = target / "local-evaluation-freeze.json"
    if lock.exists():
        raise ValueError("evaluation is already frozen; create a new version instead of overwriting")
    documents = {
        "baseline.md": ("Historical baseline reuse", "历史基线只有在数据、协议、种子、源码和 checkpoint 可核验时才能复用。接受指针是显式记录，不根据报告文字选择最高分。\n\nA historical baseline requires matching data, protocol, seed and verified source/checkpoint evidence. Reuse saves training jobs but still consumes lookup and model requests."),
        "validation.md": ("Independent validation scoring", "验证集预测由固定评分器重算 NDCG 和 Recall；模型自述的分数不能替代评分。用户配对 bootstrap 只描述固定模型下的用户抽样不确定性。\n\nCheckpoint selection and candidate comparison use validation data. Paired user bootstrap is not repeated training and does not establish cross-seed stability."),
        "patch.md": ("Exact source patches", "应用补丁前读取源码哈希，旧文本必须唯一匹配；过期哈希会拒绝修改。候选只能修改登记的文件，评分器和标签保持只读。\n\nExact patches use expected_sha256 and a unique old-text match. An actual diff records the change; a model claiming to edit a file is not evidence."),
        "restore.md": ("Snapshot restoration", "源码快照按通用换行符归一化计算内容身份。恢复前校验完整内容及只读依赖，损坏快照不会先写入部分候选文件。\n\nSnapshot restore normalizes CRLF and CR to LF before hash verification. A dependency mismatch is rejected instead of silently repairing read-only code."),
        "recovery.md": ("Task recovery", "任务恢复先查询已有作业和产物，不重复提交或重置额度。模型角色在工具操作结果不明时报告需要核验，不盲目重放副作用。\n\nResume reconciles existing jobs and preserves spent budgets. A controller checkpoint is distinct from a model weight checkpoint; mid-epoch training continuation is not implemented."),
        "memory.md": ("Scoped experience memory", "SQLite 账本保存实测事实，Markdown Wiki 保存带来源的解释和经验。检索先限定项目、数据和评价协议，摘要不能替代原始评分。\n\nExperience search filters comparison scope before lexical or semantic retrieval. Human revisions can replace a note without deleting the original experiment facts."),
        "literature.md": ("Literature provenance", "Tavily 搜索摘要标记为 snippet_only，不宣称阅读全文。提取正文记录 URL、获取时间和内容哈希；外部文本不是执行指令。\n\nSearch snippets and extracted text have different coverage. Literature supplies hypotheses, not measured recommendation results. Missing API keys leave local knowledge available."),
        "diagnostics.md": ("Process metric definitions", "过程指标分为诊断信息和必需验收条件，基线与候选比较需要同一版本定义。必需过程指标缺失时标记证据不足，不能仅凭主指标提升保留方案。\n\nRequired diagnostics must be present and comparable before retaining a candidate. Nonfinite logits are an observation, not proof of gradient explosion."),
        "roles.md": ("Planner Executor Reviewer", "Planner 生成计划，Executor 调用批准范围内的工具，Reviewer 只复核证据。三个角色共用预算，最多修订一次；角色数量本身不能证明收益。\n\nPlanner and reviewer receive no executable tools. Role checkpoints preserve completed work, permissions and request reservations across restart."),
        "web.md": ("Workbench event recovery", "Web 与 CLI 查看同一份持久化任务和研究会话。任务幂等键绑定请求内容，SSE 用事件序号续读；刷新网页不会重新提交训练。\n\nLast-Event-ID resumes event delivery. A duplicate idempotency key returns the original task; a changed payload with the same key is a conflict."),
        "budget.md": ("Budget and estimated cost", "失败模型请求也计入次数。时间、次数和作业限制不是人民币账单；按显式单价计算的费用只能称为估算，未知 usage 不能当作免费。\n\nInput, cached input and output tokens have separately supplied rates. Missing usage and failed requests make cost estimates incomplete; provider billing is not verified."),
        "freeze.md": ("Evaluation separation", "工具参考调用通过率、诊断任务成功率和模型推荐效果使用不同分母。冻结任务答案不进入模型上下文或检索库，负结果和失败任务不能从分母删除。\n\nFreeze source, task definitions and corpus hashes before evaluation. Reference-policy checks validate the evaluator and never count as end-to-end Agent success."),
    }
    corpus = target / "retrieval-corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    docs = []
    for name, (title, text) in documents.items():
        raw = ("# " + title + "\n\n" + text + "\n").encode("utf-8")
        (corpus / name).write_bytes(raw)
        docs.append({"path": name, "sha256": hashlib.sha256(raw).hexdigest()})
    queries = [
        ("baseline", "什么时候能够复用历史基线？", "baseline.md", 3),
        ("validation", "Can user bootstrap prove stability across training seeds?", "validation.md", 5),
        ("patch", "如何防止使用过期源码覆盖候选文件？", "patch.md", 3),
        ("restore", "Why can a CRLF snapshot have the same source identity?", "restore.md", 5),
        ("recovery", "控制端重启后如何避免再次消耗训练额度？", "recovery.md", 3),
        ("memory", "Where should observed experiment facts and interpretations be stored?", "memory.md", 3),
        ("literature", "只有搜索摘要时能说读过论文全文吗？", "literature.md", 3),
        ("diagnostics", "NDCG 提升但必需过程指标缺失应该怎样处理？", "diagnostics.md", 3),
        ("roles", "Which agent roles are allowed to execute tools?", "roles.md", 5),
        ("web", "重连网页如何避免重复显示事件？", "web.md", 3),
        ("budget", "Are missing usage records equivalent to zero cost?", "budget.md", 5),
        ("freeze", "Can reference tool checks establish end-to-end agent performance?", "freeze.md", 5),
    ]
    retrieval = {"schema_version": 1, "corpus_root": "retrieval-corpus", "annotation": "curated technical smoke corpus; paragraph relevance",
                 "documents": docs, "queries": [{"id": qid, "query": query, "relevant": [{"source": source, "line_start": line, "line_end": line}]}
                                            for qid, query, source, line in queries] +
                [{"id": "unanswerable", "query": "unobtainium_session_9239", "relevant": []}]}
    write(target / "retrieval.json", retrieval)
    families = [
        ("dev", "positive_gain", "retain", "/metrics"),
        ("dev", "primary_regression", "reject", "/metrics"),
        ("dev", "guardrail_regression", "reject", "/metrics"),
        ("dev", "failed_train", "inconclusive", "/status"),
        ("dev", "missing_score", "inconclusive", "/metrics"),
        ("dev", "seed_mismatch", "inconclusive", "/scope"),
        ("frozen", "protocol_mismatch", "inconclusive", "/scope"),
        ("frozen", "data_mismatch", "inconclusive", "/scope"),
        ("frozen", "budget_exhausted", "stop", "/remaining_training_jobs"),
        ("frozen", "unconfirmed_stop", "inconclusive", "/stop_confirmed"),
        ("frozen", "missing_required_diagnostic", "inconclusive", "/required_process_pass"),
        ("frozen", "duplicate_candidate", "stop", "/duplicate_source"),
    ]
    cases = {"dev": [], "frozen": []}
    for split, family, decision, pointer in families:
        for variant in range(5):
            identity = {"dataset": family + "-" + str(variant), "protocol": "ranking-v1", "seed": 2026}
            base = round(.08 + variant * .003, 6)
            evidence = {"status": "completed", "scope": {"baseline": identity, "candidate": dict(identity)},
                        "metrics": {"baseline": {"ndcg@20": base, "recall@20": .15},
                                    "candidate": {"ndcg@20": round(base+.001*(variant+1), 6), "recall@20": .16}},
                        "remaining_training_jobs": 1, "stop_confirmed": True, "required_process_pass": True,
                        "duplicate_source": False}
            if family == "primary_regression":evidence["metrics"]["candidate"]["ndcg@20"] = base-.002
            elif family == "guardrail_regression":evidence["metrics"]["candidate"]["recall@20"] = .14
            elif family == "failed_train":evidence["status"] = "failed"
            elif family == "missing_score":evidence["metrics"] = None
            elif family == "seed_mismatch":evidence["scope"]["candidate"]["seed"] += variant+1
            elif family == "protocol_mismatch":evidence["scope"]["candidate"]["protocol"] = "different-ranking"
            elif family == "data_mismatch":evidence["scope"]["candidate"]["dataset"] = "other-data"
            elif family == "budget_exhausted":evidence["remaining_training_jobs"] = 0
            elif family == "unconfirmed_stop":evidence["stop_confirmed"] = False
            elif family == "missing_required_diagnostic":evidence["required_process_pass"] = False
            elif family == "duplicate_candidate":evidence["duplicate_source"] = True
            cases[split].append({"id": family+"-"+str(variant+1), "scene_group": family, "evidence": evidence,
                                 "expected_decision": decision, "required_pointer": pointer})
    for split, rows in cases.items():
        write(target / ("diagnostic-"+split+".json"), {"schema_version": 1, "split": split,
              "origin": "synthetic independent scenario groups; not real training results",
              "tasks": rows})
    frozen = {name: hashlib.sha256((target/name).read_bytes()).hexdigest()
              for name in ["retrieval.json", "diagnostic-dev.json", "diagnostic-frozen.json"]}
    write(lock, {"schema_version": 1, "files": frozen, "dev_scene_groups": sorted({x["scene_group"] for x in cases["dev"]}),
                 "frozen_scene_groups": sorted({x["scene_group"] for x in cases["frozen"]}),
                 "note": "Do not use frozen labels to tune prompts or place labels in the retrieval corpus."})
    print(json.dumps({"dev": len(cases["dev"]), "frozen": len(cases["frozen"]), "retrieval_queries": len(retrieval["queries"])}))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def build_anonymous(root):
    """Keep the pilot freeze intact and remove scene hints from active evidence."""
    target = Path(root) / "tasks"
    output = target / "diagnostic-v2"
    if (output / "local-evaluation-freeze.json").exists():
        raise ValueError("anonymous evaluation is already frozen")
    prior = json.loads((target / "local-evaluation-freeze.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split in ("dev", "frozen"):
        name = "diagnostic-" + split + ".json"
        raw = (target / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != prior["files"][name]:
            raise ValueError("pilot manifest was modified")
        manifest = json.loads(raw)
        manifest["origin"] = "synthetic scenario groups with anonymous dataset identities; not real training results"
        manifest["benchmark_version"] = 2
        for row in manifest["tasks"]:
            before, after = row["evidence"]["scope"]["baseline"], row["evidence"]["scope"]["candidate"]
            same = before["dataset"] == after["dataset"]
            before["dataset"] = "data-" + content_hash({"case": row["id"], "version": 2})[:16]
            after["dataset"] = before["dataset"] if same else "data-" + content_hash({"case": row["id"], "version": 2, "other": True})[:16]
        write(output / name, manifest)
        hashes[name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    write(output / "local-evaluation-freeze.json", {"schema_version": 1, "benchmark_version": 2, "files": hashes,
          "note": "Scene group names and labels are not passed to model inputs. Pilot scores do not apply to this freeze."})
    print(json.dumps({"version": 2, "dev": 30, "frozen": 30, "pilot_preserved": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--anonymous-v2", action="store_true")
    args = parser.parse_args()
    (build_anonymous if args.anonymous_v2 else build)(args.root)
