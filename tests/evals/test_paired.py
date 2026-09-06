import json

import pytest
from evals.gold import load_gold
from evals.paired import select_suite, summarize

from evals import paired


def test_suites_are_explicit_valid_and_not_holdouts():
    gold = load_gold()
    smoke = select_suite(gold, "smoke48")
    probe = select_suite(gold, "model_probe24")
    assert len(smoke) == 48
    assert len(probe) == 24
    assert {q.id for q in probe} <= {q.id for q in smoke}
    assert {q.type for q in smoke} == {
        "card",
        "answer",
        "refuse",
        "infer",
        "comparison",
        "dialog",
        "guard",
    }
    assert {f"{c.inn}-card" for c in gold.cards if c.terminal} <= {q.id for q in probe}


def test_unknown_suite_case_must_not_be_silently_skipped(tmp_path, monkeypatch):
    path = tmp_path / "suite.json"
    path.write_text(json.dumps({"broken": ["typo-in-critical-case"]}))
    monkeypatch.setattr(paired, "SUITES", path)
    with pytest.raises(ValueError, match="Unknown question IDs"):
        select_suite(load_gold(), "broken")


def row(model, repeat, *, passed=True, error=None):
    return {
        "record": {
            "model": model,
            "question_id": "case",
            "repeat": repeat,
            "type": "answer",
            "error": error,
            "checks_passed": passed,
            "duration_s": 10,
        },
        "calls": [],
        "turns": [],
        "failed_callbacks": 0,
    }


def test_errors_count_against_success_and_repeated_success():
    result = summarize(
        [row("a", 0), row("a", 1, error="timeout"), row("b", 0), row("b", 1)], ["a", "b"], 4
    )
    assert result["models"]["a"]["scenarios"] == 2
    assert result["models"]["a"]["code_passes"] == 1
    assert result["models"]["a"]["groups_all_repeats_pass"] == 0
    assert result["models"]["a"]["semantic_correctness"] is None
    assert result["paired_code_checks"]["only_second_pass"] == 1


def test_consistently_wrong_is_not_counted_as_repeated_success():
    result = summarize([row("a", 0, passed=False), row("a", 1, passed=False)], ["a", "b"], 4)
    assert result["models"]["a"]["groups_all_repeats_pass"] == 0
    assert result["completed_scenarios"] == 2
    assert result["planned_scenarios"] == 4
    assert result["paired_code_checks"]["matched_pairs"] == 0


def test_unfinished_repeats_do_not_count_as_a_stable_success():
    result = summarize([row("a", 0), row("a", 1)], ["a", "b"], 6, repeats=3)
    assert result["models"]["a"]["repeat_groups"] == 0
    assert result["models"]["a"]["groups_all_repeats_pass"] == 0
    assert result["models"]["a"]["planned_case_groups"] == 1
