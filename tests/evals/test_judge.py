import pytest
from evals.judge import JudgeVerdict
from pydantic import ValidationError


@pytest.mark.parametrize("score", [0, 1, 4])
def test_judge_cannot_penalise_an_answer_while_reporting_no_violations(score):
    with pytest.raises(ValidationError, match="contradicts"):
        JudgeVerdict(score=score, comment="Все факты подтверждены, снижений нет.")


def test_judge_cannot_pass_an_answer_with_a_fatal_invention():
    with pytest.raises(ValidationError, match="contradicts"):
        JudgeVerdict(score=5, failures=["invented_fact"], comment="Перепутаны компании.")


@pytest.mark.parametrize(
    "score, failures, deductions",
    [
        (5, [], []),
        (0, ["invented_fact"], []),
        (3, [], ["not_answering_question"]),
        (2, [], ["courts_without_role_period_field", "no_report_date"]),
    ],
)
def test_consistent_rubric_results_are_preserved(score, failures, deductions):
    verdict = JudgeVerdict(
        score=score, failures=failures, deductions=deductions, comment="Оценка по рубрике."
    )
    assert verdict.score == score
