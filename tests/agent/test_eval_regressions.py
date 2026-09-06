"""Regressions observed in the frozen 20B/120B probe, not model-specific wording tests."""

import pytest

from contractor_agent.agent.scoped_answers import scoped_answer
from contractor_agent.mcp_server.tools import Tools


@pytest.mark.parametrize("inn", ["5032257375", "2100006761", "2308177290"])
def test_deferral_answer_includes_registry_status_before_finances(snapshot, inn):
    answer = scoped_answer(Tools(snapshot), [inn], "Можно отгружать им товар с отсрочкой?")
    # Independent oracle: the raw report, not the risk engine or answer builder.
    reason = snapshot.get(inn).status.reason_name
    text = answer.text_md
    assert reason in text
    assert text.index(reason) < text.index("### Финансы")
    assert text.count(reason) == 1
    assert any(c.source_path == "report.status.reasonName" for c in answer.citations)


@pytest.mark.parametrize("inn", ["5032257375", "6165169320"])
def test_critical_obligations_are_not_duplicated_after_finances(snapshot, inn):
    answer = scoped_answer(Tools(snapshot), [inn], "Можно дать отсрочку?")
    bullets = [line for line in answer.lines if line.startswith("- ")]
    assert len(bullets) == len(set(bullets))
    assert any("исполнительн" in b or "арбитражн" in b for b in bullets)


def test_who_runs_company_and_who_to_sign_with_preserves_name_and_status(snapshot):
    answer = scoped_answer(
        Tools(snapshot),
        ["5032257375"],
        "Кто сейчас руководит ООО «МАКСМАРКЕТ» (ИНН 5032257375) — с кем мне подписывать договор?",
    )
    text = answer.text_md
    report = snapshot.get("5032257375")
    assert report.founders_info.auth_person.name.casefold() in text.casefold()
    assert report.status.reason_name in text
    assert "нельзя подтвердить" in text
    assert "арбитраж" not in text.casefold()
    assert "финанс" not in text.casefold()


def test_simple_identity_question_does_not_trigger_risk_summary(snapshot):
    answer = scoped_answer(Tools(snapshot), ["5032257375"], "Кто сейчас руководит МАКСМАРКЕТ?")
    assert "Москвина Ирина Витальевна" in answer.text_md
    assert "### Факты" not in answer.text_md
    assert "подписание" not in answer.text_md
