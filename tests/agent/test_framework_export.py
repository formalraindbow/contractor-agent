from pathlib import Path

from evals.export_framework import expected_calls
from evals.gold import load_gold


def test_reference_keeps_comparison_context_and_explicit_single_target():
    direct = load_gold(Path("evals/suites/v8_direct_answers.yaml"))
    q = next(q for q in direct.questions() if q.id == "direct-v8-10")
    calls, inns = expected_calls(q.question, q.inn, ["5032257375", "1684017097"], names={}, spec=q)
    assert inns == ["5032257375", "1684017097"]
    assert calls[0] == ("compare_companies", {"inns": inns})
    q = next(q for q in load_gold().questions() if q.id == "edge-avrora-two-inns-one-target")
    _, inns = expected_calls(q.question, q.inn, [], names={}, spec=q)
    assert inns == ["9731131090"]


def test_reference_unknown_company_does_not_borrow_gold_card():
    q = next(q for q in load_gold().questions() if q.id == "edge-techprof-inn-not-found")
    calls, inns = expected_calls(q.question, q.inn, [], names={}, spec=q)
    assert inns == ["1234567890"]
    assert calls == [("get_report_summary", {"inn": "1234567890"})]
