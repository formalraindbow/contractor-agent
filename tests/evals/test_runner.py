# ruff: noqa: E501
from evals.gold import Gold, GoldCard, GoldQuestion
from evals.metrics import compute_metrics
from evals.report import render
from evals.runner import EvalRunner
from langchain_core.messages import AIMessage

from contractor_agent.agent.runtime import AgentRuntime
from contractor_agent.agent.schema import Draft
from contractor_agent.data.loader import Snapshot
from contractor_agent.settings import Settings
from tests.agent.fakes import scripted_llm, tool_call

GOLD = Gold(
    cards=[
        GoldCard(
            inn="5032257375",
            company="ООО «МАКСМАРКЕТ»",
            report_date="2026-07-31",
            expected_verdict="not_recommended",
            terminal=True,
            must_name=["банкрот"],
            questions=[
                GoldQuestion(
                    id="maxmarket-staff",
                    inn="5032257375",
                    type="refuse",
                    question="Сколько сотрудников?",
                    must_mention=["численност"],
                )
            ],
        )
    ]
)


async def test_runner_caches_and_metrics(snapshot: Snapshot, tmp_path) -> None:
    llm = scripted_llm(
        [
            tool_call("get_risk_signals", "c1", inn="5032257375"),
            AIMessage(content="ok"),
            Draft(
                kind="card",
                lines=[
                    "Компания признана банкротом [report.status.reasonName].",
                    "Работать только на условиях: предоплата и подтверждающие документы. Отчёт от 31.07.2026.",
                ],
                citations=[],
            ),
            tool_call("get_report_summary", "c2", inn="5032257375"),
            AIMessage(content="ok"),
            Draft(
                kind="refusal",
                lines=["В отчёте нет сведений о численности — оценить нельзя."],
                citations=[],
            ),
            # инструмент вернул сводку, поэтому отказ уходит на круг исправления
            AIMessage(content="ok"),
            Draft(
                kind="refusal",
                lines=["В отчёте нет сведений о численности — оценить нельзя."],
                citations=[],
            ),
        ]
    )
    settings = Settings(runs_dir=tmp_path / "runs", openrouter_api_key="x")
    async with AgentRuntime(settings, source=snapshot, llm=llm) as runtime:
        runner = EvalRunner(GOLD, runtime, cache_dir=tmp_path / "evals", model_name="scripted/test")
        records = await runner.run(GOLD.questions())
        assert [r.type for r in records] == ["card", "refuse"]
        assert all(r.checks_passed for r in records), [r.check_failures for r in records]
        assert (
            records[0].trace[0]["name"] == "get_risk_signals"
            and "[get_risk_signals]" in records[0].tool_outputs
        )
        cached = await runner.run(
            GOLD.questions()
        )  # сценарий модели исчерпан — значит, читается кэш
        assert [r.question_id for r in cached] == [r.question_id for r in records]
    metrics = compute_metrics(records)
    assert metrics.grounded_share == 1.0 and metrics.refusal_share == 1.0
    assert metrics.invented_share == 0.0 and metrics.missed_critical_share == 0.0
    text = render([metrics], {"scripted/test": records})
    assert "Подтверждаемых ответов" in text and "100 %" in text


def test_follow_up_history_names_company_and_date():
    from evals.gold import load_gold
    from evals.runner import follow_up_history

    gold = load_gold()
    follow_ups = [q for q in gold.questions() if q.follow_up]
    assert len(follow_ups) == 5  # «у них…» без названия компании — только с засевом сессии
    history = follow_up_history(gold.card("5032257375"))
    assert [m.type for m in history] == ["human", "ai"]
    assert "МАКСМАРКЕТ" in history[0].content and "5032257375" in history[0].content
    assert "31.07.2026" in history[1].content


def test_initial_state_keeps_history_before_question():
    from langchain_core.messages import AIMessage, HumanMessage

    from contractor_agent.agent.graph import initial_state

    state = initial_state(
        "а суды?", [HumanMessage(content="проверь X"), AIMessage(content="смотрю")]
    )
    assert [m.content for m in state["messages"]] == ["проверь X", "смотрю", "а суды?"]
