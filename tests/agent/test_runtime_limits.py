import asyncio

import pytest
from langchain_core.messages import AIMessage

from contractor_agent.agent.runtime import AgentRuntime, RunBusyError
from contractor_agent.settings import Settings
from tests.agent.fakes import scripted_llm


async def test_same_conversation_serializes_without_blocking_other_conversations(
    snapshot, tmp_path
):
    runtime = AgentRuntime(
        Settings(runs_dir=tmp_path, max_concurrent_runs=2, run_queue_timeout=0.04),
        source=snapshot,
        llm=scripted_llm([]),
    )
    async with runtime.run_slot("one"):
        async with runtime.run_slot("two"):
            with pytest.raises(RunBusyError):
                async with runtime.run_slot("three"):
                    pytest.fail("Exceeded the process concurrency limit")
        with pytest.raises(RunBusyError):
            async with runtime.run_slot("one"):
                pytest.fail("Two writers entered one conversation")
        async with runtime.run_slot("three"):
            pass  # timing out in the queue did not consume a slot
    async with runtime.run_slot("one"):
        pass  # timing out on the conversation did not leave its lock acquired


async def test_cancelled_and_timed_out_runs_release_their_slots(snapshot, tmp_path):
    runtime = AgentRuntime(
        Settings(runs_dir=tmp_path, max_concurrent_runs=1, run_timeout=0.04),
        source=snapshot,
        llm=scripted_llm([]),
    )
    entered = asyncio.Event()

    async def wait():
        async with runtime.run_slot("chat"):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(wait())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(TimeoutError):
        await wait()
    async with runtime.run_slot("chat"):
        pass
    assert all(not lock.locked() for lock in runtime._thread_locks.values())


async def test_thirty_conversations_do_not_mix_companies(snapshot, tmp_path):
    inns = [r.inn for r in list(snapshot)[:30]]
    model = scripted_llm([AIMessage(content="Факты готовы") for _ in inns])
    settings = Settings(session_store="memory", runs_dir=tmp_path, max_concurrent_runs=4)
    async with AgentRuntime(settings, source=snapshot, llm=model) as runtime:
        answers = await asyncio.gather(
            *[runtime.ask(f"Проверь {inn}", f"parallel-{inn}") for inn in inns]
        )
    assert len(answers) == 30
    for inn, answer in zip(inns, answers, strict=True):
        assert answer.card.inn == inn
        assert set(answer.report_dates) == {inn}
        assert not answer.invalid_citations
