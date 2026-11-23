"""User-facing experiment reports use measured evidence, without API/GPU calls."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from reclab_agent.domain.contracts import canonical_json, content_hash
from reclab_agent.reporting import (
    build_report_view,
    format_user_summary,
    render_html,
    render_markdown,
    write_user_report,
)


class UserReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.protocol = "a" * 64
        self.context = {
            "task_id": "fixture-optimization",
            "task_question": "改进一个小型推荐模型并比较真实评分。",
            "target_model": "FixtureModel",
            "agent_mode": "react",
            "dataset_id": "fixture-data",
            "seed": 2026,
            "primary_metric": "ndcg@20",
            "guardrail_metrics": ["recall@20"],
            "editable_files": ["src/model.py"],
        }
        self.baseline = self.entry("baseline", .08, .15, source="WIDTH = 64\n")
        self.candidate = self.entry("candidate-safe", .10, .16, source="WIDTH = 128\n")

    def entry(self, candidate_id, ndcg=None, recall=None, *, status="completed",
              source="WIDTH = 64\n", output=None, failure=None, seed=2026):
        output = self.root if output is None else output
        files = {"src/model.py": source}
        source_hash = content_hash(files)
        snapshot = output / "snapshots" / source_hash / "src/model.py"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(source, encoding="utf-8")
        experiment_id = "exp-" + hashlib.sha256(candidate_id.encode()).hexdigest()[:32]
        record = {
            "experiment_id": experiment_id,
            "candidate_id": candidate_id,
            "status": status,
            "seed": seed,
            "source_hash": source_hash,
            "config_hash": "b" * 64,
            "dataset_id": "fixture-data",
            "protocol_hash": self.protocol,
            "plan_hash": "c" * 64,
            "program_validation": "passed",
            "image": "fixture/report@sha256:" + "d" * 64,
            "training_job_id": experiment_id + "-train",
            "prediction_job_id": experiment_id + "-predict" if status == "completed" else None,
            "failure": failure,
            "score": None,
        }
        result = {"record": record}
        if status == "completed" and ndcg is not None:
            metrics = {"ndcg@20": ndcg, "recall@20": recall}
            grade = {"protocol_hash": self.protocol, "seed": seed, "metrics": metrics}
            relative = f"experiments/{experiment_id}/grade.json"
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = canonical_json(grade).encode("utf-8")
            path.write_bytes(raw)
            record["score"] = {"source": relative, "sha256": hashlib.sha256(raw).hexdigest(),
                               "location": "/metrics/ndcg@20", "conditions": "Synthetic fixture only"}
            result["metrics"] = metrics
        return result

    def report(self, *entries, status="completed"):
        return {
            "status": status,
            "session_id": "fixture-session",
            "session_directory": str(self.root),
            "report_path": str(self.root / "report.json"),
            "baseline_id": "baseline",
            "primary_metric": "ndcg@20",
            "guardrail_metrics": ["recall@20"],
            "experiments": [self.baseline, *(entries or (self.candidate,))],
            "comparison": [],
            "budget_usage": {"experiment_attempts": 1 + len(entries or (self.candidate,)),
                             "patch_attempts": 1, "model_calls": 7},
            "test_evaluated": False,
            "cost_enforcement": "time/job limits only; billing not verified",
        }

    def markdown(self, report, *, output=None, context=None):
        return render_markdown(build_report_view(
            report, output_directory=output or self.root, context=context or self.context))

    def write(self, report, *, context=None):
        return write_user_report(self.root, report, context=context or self.context)

    def test_finished_run_writes_readable_reports_without_overwriting_machine_data(self):
        report = self.report()
        before = deepcopy(report)
        machine = self.root / "report.json"
        machine.write_text('{"original":true}\n', encoding="utf-8")
        metadata = self.write(report)
        self.assertEqual(Path(metadata["user_report_path"]), self.root / "report.html")
        self.assertEqual(Path(metadata["user_report_markdown_path"]), self.root / "report.md")
        self.assertTrue(metadata["user_summary"])
        self.assertEqual(report, before)
        self.assertEqual(machine.read_text(encoding="utf-8"), '{"original":true}\n')
        text = (self.root / "report.md").read_text(encoding="utf-8")
        self.assertIn("ndcg@20", text)
        self.assertIn("recall@20", text)
        self.assertIn("candidate-safe", text)
        self.assertRegex(text, r"25(?:\.0+)?%")
        self.assertRegex(text, r"0\.02(?:0*)")

    def test_best_candidate_must_also_meet_guardrail(self):
        unsafe = self.entry("candidate-unsafe", .12, .14, source="WIDTH = 256\n")
        report = self.report(self.candidate, unsafe)
        summary = self.write(report)["user_summary"]
        self.assertIn("candidate-safe", summary)
        self.assertNotIn("candidate-unsafe", summary)
        text = self.markdown(report)
        self.assertIn("candidate-unsafe", text)
        self.assertTrue(any(word in text for word in ("守护", "边界", "约束", "guardrail")))

    def test_seed_zero_is_a_real_seed_not_missing_comparison_metadata(self):
        baseline = self.entry("baseline-zero", .08, .15, seed=0)
        candidate = self.entry("candidate-other-seed", .10, .16, seed=42, source="WIDTH = 256\n")
        report = self.report(candidate)
        report["baseline_id"] = "baseline-zero"
        report["experiments"] = [baseline, candidate]
        metadata = self.write(report, context={**self.context, "seed": 0})
        text = Path(metadata["user_report_markdown_path"]).read_text(encoding="utf-8")
        self.assertNotIn("candidate-other-seed", metadata["user_summary"])
        self.assertIn("candidate-other-seed", text)
        self.assertIn("种子", text)
        self.assertTrue(any(word in text for word in ("不一致", "不可比较", "不能直接比较")))

    def test_explicit_empty_guardrails_do_not_infer_an_auxiliary_metric_constraint(self):
        candidate = self.entry("candidate-primary-only", .10, .14, source="WIDTH = 256\n")
        report = self.report(candidate)
        report["guardrail_metrics"] = []
        metadata = self.write(report, context={**self.context, "guardrail_metrics": []})
        self.assertIn("candidate-primary-only", metadata["user_summary"])
        self.assertRegex(metadata["user_summary"], r"25(?:\.0+)?%")

    def test_negative_measured_result_is_not_a_success_claim(self):
        worse = self.entry("candidate-worse", .07, .16, source="WIDTH = 32\n")
        report = self.report(worse)
        text = self.markdown(report)
        self.assertIn("candidate-worse", text)
        self.assertRegex(text, r"[-−]12\.5(?:0*)%")
        summary = self.write(report)["user_summary"]
        self.assertTrue(any(word in summary for word in ("未", "没有", "下降", "不足")))

    def test_failed_candidate_has_no_invented_zero_metric_or_llm_quality_claim(self):
        failed = self.entry("candidate-failed", status="failed",
                            failure="fixture numerical failure")
        report = self.report(failed, status="incomplete")
        report["agent_result"] = {"status": "completed", "answer": "NDCG=0.99999，提升99999%"}
        text = self.markdown(report)
        self.assertIn("candidate-failed", text)
        self.assertIn("fixture numerical failure", text)
        self.assertNotIn("0.99999", text)
        rows = [line for line in text.splitlines() if "candidate-failed" in line and "|" in line]
        self.assertTrue(rows, "failed attempts should remain in the user-visible results table")
        self.assertTrue(any(mark in rows[0] for mark in ("—", "--", "不可用", "无分数", "未评分")))

    def test_baseline_failure_still_produces_a_user_report(self):
        failed = self.entry("baseline", status="failed", failure="fixture baseline failure")
        report = self.report(status="failed")
        report["experiments"] = [failed]
        report["budget_usage"]["experiment_attempts"] = 1
        report["error"] = "LaunchError"
        metadata = self.write(report)
        text = Path(metadata["user_report_markdown_path"]).read_text(encoding="utf-8")
        self.assertIn("fixture baseline failure", text)
        self.assertTrue(any(word in metadata["user_summary"] for word in ("失败", "无法", "未")))
        self.assertNotIn("25.00%", text)

    def test_interrupted_run_retains_existing_baseline_measurement(self):
        report = self.report(status="cancelled")
        report["experiments"] = [self.baseline]
        report["budget_usage"]["experiment_attempts"] = 1
        text = self.markdown(report)
        self.assertTrue(any(word in text for word in ("取消", "中断")))
        self.assertIn("baseline", text)
        self.assertRegex(text, r"0\.08(?:0*)")

    def test_grade_hash_mismatch_cannot_be_presented_as_verified_improvement(self):
        report = self.report()
        grade = self.root / self.candidate["record"]["score"]["source"]
        grade.write_text('{"metrics":{"ndcg@20":0.99}}', encoding="utf-8")
        try:
            metadata = self.write(report)
        except ValueError as error:
            self.assertRegex(str(error).lower(), r"hash|sha|评分|校验|integrity|mismatch")
        else:
            text = Path(metadata["user_report_markdown_path"]).read_text(encoding="utf-8")
            self.assertTrue(any(word in text for word in ("校验", "证据", "失配", "不一致")))
            self.assertNotRegex(metadata["user_summary"], r"25(?:\.0+)?%")

    def test_source_changes_come_from_frozen_candidate_not_current_workspace(self):
        current = self.root / "workspaces/current/src/model.py"
        current.parent.mkdir(parents=True)
        current.write_text("WIDTH = 999\n", encoding="utf-8")
        report = self.report()
        report["agent_result"] = {"answer": "我已把 WIDTH 改为 999。"}
        text = self.markdown(report)
        self.assertIn("src/model.py", text)
        self.assertIn("WIDTH = 64", text)
        self.assertIn("WIDTH = 128", text)
        self.assertNotIn("WIDTH = 999", text)

    def test_missing_snapshot_does_not_invent_changes_or_erase_valid_score(self):
        source = self.root / "snapshots" / self.candidate["record"]["source_hash"] / "src/model.py"
        source.unlink()
        text = self.markdown(self.report())
        self.assertIn("candidate-safe", text)
        self.assertRegex(text, r"0\.1(?:0*)")
        self.assertTrue(any(word in text for word in ("未记录", "不可用", "缺失", "未找到", "无法")))
        self.assertNotIn("WIDTH = 128", text)

    def test_html_escapes_dynamic_text_and_has_no_network_assets(self):
        report = self.report()
        report["experiments"][1]["record"]["candidate_id"] = "candidate<script>alert(1)</script>"
        context = {**self.context, "task_question": '<img src=x onerror=alert(1)> & "quote"'}
        html = render_html(build_report_view(report, output_directory=self.root, context=context))
        self.assertNotIn("<script>", html.lower())
        self.assertNotIn("<img src=x", html.lower())
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&lt;img", html)
        self.assertNotRegex(html, r"(?:src|href)=[\"']https?://")
        self.assertIn("<meta", html.lower())
        self.assertIn("utf-8", html.lower())

    def test_untrusted_markdown_cannot_embed_an_external_image(self):
        attack = "![external-image](https://fixture.invalid/tracker.png)"
        context = {**self.context, "task_question": attack}
        text = self.markdown(self.report(), context=context)
        self.assertNotIn(attack, text)
        self.assertIn(r"\!\[external-image\]", text)

    def test_report_has_validation_and_cost_scope_with_recorded_resource_use(self):
        report = self.report()
        report["model_usage"] = {"input_tokens": 1234, "output_tokens": 432, "total_tokens": 1666}
        report["elapsed_seconds"] = 123.5
        text = self.markdown(report)
        self.assertTrue(any(word in text for word in ("1,666", "1666")))
        self.assertIn("验证集", text)
        self.assertIn("一个固定随机种子", text)
        self.assertNotIn("额外训练种子", text)
        self.assertNotIn("最终测试集验证", text)
        html = render_html(build_report_view(report, output_directory=self.root, context=self.context))
        self.assertIn("一个固定随机种子", html)
        self.assertNotIn("额外训练种子", html)
        self.assertTrue(any(word in text for word in ("费用", "账单", "计费")))
        self.assertTrue(any(word in text for word in ("未核验", "未验证", "未记录", "不代表")))

    def test_nonfinite_metric_never_becomes_a_positive_user_claim(self):
        report = self.report()
        report["experiments"][1]["metrics"]["ndcg@20"] = float("inf")
        try:
            metadata = self.write(report)
        except ValueError:
            return
        self.assertNotIn("inf%", metadata["user_summary"].lower())
        self.assertNotIn("nan%", metadata["user_summary"].lower())

    def test_incomplete_strategy_comparison_reports_observations_without_method_winner(self):
        baseline_root = self.root / "baseline"
        child = self.root / "strategies/react"
        baseline = self.entry("shared-baseline", .08, .15, output=baseline_root)
        candidate = self.entry("candidate-safe", .10, .16, output=child, source="WIDTH = 128\n")
        comparison = {
            "kind": "strategy_comparison", "status": "incomplete", "session_id": "fixture-comparison",
            "session_directory": str(self.root), "primary_metric": "ndcg@20",
            "guardrail_metrics": ["recall@20"], "baseline": baseline,
            "candidate_limit_per_strategy": 5, "winning_strategies": ["react"],
            "test_evaluated": False,
            "strategies": [{
                "strategy": "react", "status": "incomplete", "candidate_count": 1,
                "budget_usage": {"experiment_attempts": 1, "patch_attempts": 1}, "model_calls": 6,
                "rounds": [{"round": 1, "candidate_id": "candidate-safe", "decision": "accepted",
                            "experiments": [candidate], "source_changes": []}],
                "best": {**candidate, "score_root": str(child),
                         "source_directory": str(child / "snapshots" / candidate["record"]["source_hash"])},
            }],
        }
        metadata = self.write(comparison)
        text = Path(metadata["user_report_markdown_path"]).read_text(encoding="utf-8")
        self.assertIn("candidate-safe", text)
        self.assertRegex(text, r"0\.1(?:0*)")
        self.assertTrue(any(word in metadata["user_summary"] for word in
                            ("不宣布", "未宣布", "不判定", "不足以", "不完整", "未完成")))

    def test_console_summary_includes_user_report_location(self):
        metadata = self.write(self.report())
        text = format_user_summary({**self.report(), **metadata})
        self.assertIn(metadata["user_summary"], text)
        self.assertIn(metadata["user_report_path"], text)

    def test_archived_curated_comparison_retains_metrics_and_audit_findings(self):
        baseline = self.entry("shared-baseline", .08, .15, output=self.root / "baseline")
        candidate = self.entry("candidate-safe", .10, .16,
                               output=self.root / "strategies/react", source="WIDTH = 128\n")

        def archived(entry):
            record = deepcopy(entry["record"])
            score = record.pop("score")
            return {"record": record, "metrics": entry["metrics"],
                    "independent_grade": {"sha256": score["sha256"]}}

        curated = {
            "kind": "one_off_agent_strategy_experiment",
            "status": {"controller_status": "incomplete", "experiment_execution_finished": True},
            "session_id": "archived-fixture", "test_evaluated": False,
            "provenance": {"dataset_id": "fixture-data", "seed": 2026},
            "protocol": {"primary_metric": "ndcg@20", "guardrail_metrics": ["recall@20"]},
            "baseline": {**archived(baseline), "selection_metrics": baseline["metrics"],
                         "model": {"class": "CuratedFixtureModel"}},
            "strategies": [{"strategy": "react", "status": "incomplete", "attempts": 1,
                "patch_attempts": 1, "model_calls": 6,
                "rounds": [{"round": 1, "candidate_id": "candidate-safe", "decision": "accepted",
                            "experiments": [archived(candidate)], "source_changes": []}]}],
            "model_implementation_audit_zh": [{"finding": "新增分支没有进入实际评分路径。"}],
            "post_experiment_numerical_diagnostic": {"interpretation_zh": ["微型探针检查权重未变。"]},
        }
        metadata = write_user_report(self.root, curated)
        text = Path(metadata["user_report_markdown_path"]).read_text(encoding="utf-8")
        self.assertIn("CuratedFixtureModel", text)
        self.assertIn("candidate-safe", text)
        self.assertRegex(metadata["user_summary"], r"25(?:\.0+)?%")
        self.assertIn("新增分支没有进入实际评分路径", text)
        self.assertIn("微型探针检查权重未变", text)
        self.assertIn("不宣布模式胜者", metadata["user_summary"])

    def test_private_connection_and_provider_metadata_are_not_rendered(self):
        report = self.report()
        report["api_key"] = "fixture-secret-do-not-display"
        report["api_check"] = {"api_key": "fixture-provider-secret"}
        report["connection"] = {"host": "fixture-private-host.invalid",
                                "key_path": "C:/private/fixture-identity"}
        metadata = self.write(report)
        html = Path(metadata["user_report_path"]).read_text(encoding="utf-8")
        markdown = Path(metadata["user_report_markdown_path"]).read_text(encoding="utf-8")
        for value in ("fixture-secret-do-not-display", "fixture-provider-secret",
                      "fixture-private-host.invalid", "C:/private/fixture-identity"):
            self.assertNotIn(value, html + markdown + metadata["user_summary"])

    def test_unconfirmed_remote_stop_is_prominent_in_summary_and_both_reports(self):
        report = self.report()
        report["requires_reconciliation"] = True
        metadata = self.write(report)
        outputs = [metadata["user_summary"],
                   Path(metadata["user_report_path"]).read_text(encoding="utf-8"),
                   Path(metadata["user_report_markdown_path"]).read_text(encoding="utf-8")]
        for text in outputs:
            self.assertIn("停止尚未确认", text)
            for unsafe_claim in ("GPU 已空闲", "GPU已空闲", "可以关机", "可安全关闭"):
                self.assertNotIn(unsafe_claim, text)

    def test_curated_export_preserves_an_unconfirmed_remote_stop_warning(self):
        baseline = self.entry("shared-baseline", .08, .15, output=self.root / "baseline")
        record = deepcopy(baseline["record"])
        score = record.pop("score")
        report = {
            "kind": "one_off_agent_strategy_experiment", "session_id": "unconfirmed-fixture",
            "status": {"controller_status": "incomplete", "remote_stop_confirmed": False},
            "provenance": {"dataset_id": "fixture-data", "seed": 2026},
            "protocol": {"primary_metric": "ndcg@20", "guardrail_metrics": ["recall@20"]},
            "baseline": {"record": record, "metrics": baseline["metrics"],
                         "independent_grade": {"sha256": score["sha256"]}},
            "strategies": [], "test_evaluated": False,
        }
        metadata = write_user_report(self.root, report)
        self.assertIn("停止尚未确认", metadata["user_summary"])
        for path in (metadata["user_report_path"], metadata["user_report_markdown_path"]):
            self.assertIn("停止尚未确认", Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
