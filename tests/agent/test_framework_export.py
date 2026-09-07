from pathlib import Path

import yaml
from evals.export_framework import expected_calls, export
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


def test_guard_reference_keeps_real_card_reads_before_the_offtopic_turn(tmp_path):
    export(tmp_path)
    case = yaml.safe_load((tmp_path / "regression/guard-maksmarket-insult.yml").read_text())
    messages = case["simulation"]["reference_outputs"]
    last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
    calls = [c for m in messages[:last_user] for c in m.get("tool_calls", [])]
    assert {c["function"]["name"] for c in calls} == {"get_report_summary", "get_risk_signals"}
    assert not any(m.get("tool_calls") for m in messages[last_user:])
