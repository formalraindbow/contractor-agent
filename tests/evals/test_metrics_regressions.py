from evals.metrics import compute_metrics
from evals.runner import RunRecord


def test_api_failures_remain_in_success_denominator():
    base = dict(inn="1", model="x", repeat=0, question="q")
    records = [
        RunRecord(
            question_id="ok",
            type="answer",
            checks_passed=True,
            agent_duration_s=2,
            duration_s=50,
            **base,
        ),
        RunRecord(question_id="failed", type="answer", error="provider unavailable", **base),
        RunRecord(
            question_id="refuse",
            type="refuse",
            checks_passed=False,
            check_notes={"refused": True},
            **base,
        ),
    ]
    result = compute_metrics(records)
    assert result.grounded_share == 0.5
    assert result.refusal_share == 0
    assert result.mean_duration_s == 2


def test_failed_repeat_is_not_hidden_and_missing_judge_is_unknown():
    base = dict(inn="1", model="x", question="q", question_id="same", type="answer")
    rows = [
        RunRecord(repeat=0, checks_passed=True, **base),
        RunRecord(repeat=1, error="timeout", **base),
    ]
    result = compute_metrics(rows)
    assert result.stability == 0
    assert result.by_type["answer"]["n"] == 2
    assert result.by_type["answer"]["checks_pass"] == 0.5
    assert result.by_type["answer"]["judge_mean"] is None
