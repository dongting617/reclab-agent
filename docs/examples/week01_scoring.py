"""Week 1 learning example: trusted tiny rankings, no GPU/API/candidate execution.

From the repository root, with the project installed in its environment:
    python docs/examples/week01_scoring.py
All candidate rankings below are hand-written teaching data, not trained models.
"""
from collections import Counter
import copy
import json
import math

from reclab_agent.grading import RankingProtocol, compare_reports, grade_rankings


def main():
    histories = {"u1": [1, 4, 2], "u2": [3, 2, 5]}
    targets = {"u1": 5, "u2": 4}
    catalog = (1, 2, 3, 4, 5)
    protocol = RankingProtocol("week01-tutorial", "valid", catalog, ks=(1, 2))

    # Learn popularity only from training histories; ties break by item ID.
    counts = Counter(item for items in histories.values() for item in items)
    ordered = sorted(catalog, key=lambda item: (-counts[item], item))
    predictions = {user: [item for item in ordered if item not in set(history)]
                   for user, history in histories.items()}
    baseline = grade_rankings(predictions, targets, histories, protocol)
    assert predictions == {"u1": [3, 5], "u2": [1, 4]}
    assert baseline["metrics"]["recall@1"] == 0
    assert math.isclose(baseline["metrics"]["ndcg@2"], 1 / math.log2(3))

    # A constructed candidate illustrates rank changes; no optimization claim.
    candidate_predictions = {"u1": [5, 3], "u2": [1, 4]}
    candidate = grade_rankings(candidate_predictions, targets, histories, protocol)
    expected_ndcg = (1 + 1 / math.log2(3)) / 2
    assert candidate["metrics"]["recall@1"] == 0.5
    assert candidate["metrics"]["recall@2"] == 1
    assert math.isclose(candidate["metrics"]["ndcg@2"], expected_ndcg)
    comparison = compare_reports(baseline, candidate, bootstrap_samples=1000, seed=42)

    rejected = {}
    missing_user = {"u1": [5, 3]}
    repeated = {"u1": [5, 5], "u2": [1, 4]}
    for name, prediction in [("missing_user", missing_user), ("duplicate_item", repeated)]:
        try:
            grade_rankings(prediction, targets, histories, protocol)
        except ValueError as error:
            rejected[name] = str(error)
        else:
            raise AssertionError(f"{name} must be rejected")

    # Editing a claimed aggregate cannot override independently computed rows.
    corrupt_report = copy.deepcopy(candidate)
    corrupt_report["metrics"]["ndcg@2"] = 0.99
    try:
        compare_reports(baseline, corrupt_report)
    except ValueError as error:
        rejected["fake_aggregate"] = str(error)
    else:
        raise AssertionError("corrupt aggregate must be rejected")

    print(json.dumps({
        "scope": "constructed_teaching_example_not_a_trained_model",
        "training_counts": dict(sorted(counts.items())),
        "baseline_predictions": predictions,
        "baseline_metrics": baseline["metrics"],
        "candidate_metrics": candidate["metrics"],
        "paired_comparison": comparison["metrics"],
        "rejected_inputs": rejected,
        "checks": "all teaching assertions passed",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
