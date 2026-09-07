"""Credentials for agent and judge must follow their independent providers."""

from contractor_agent.settings import Settings


def test_polza_credentials_are_separate_from_openrouter():
    settings = Settings(
        _env_file=None,
        llm_base_url="https://polza.ai/api/v1",
        judge_base_url="https://openrouter.ai/api/v1",
        polza_api_key="test-polza",
        openrouter_api_key="test-openrouter",
        llm_api_key_override=None,
    )
    assert settings.llm_api_key == "test-polza"
    assert settings.judge_api_key == "test-openrouter"
    settings.judge_base_url = "https://polza.ai/api/v1"
    assert settings.judge_api_key == "test-polza"


def test_missing_polza_key_does_not_send_openrouter_key():
    settings = Settings(
        _env_file=None,
        polza_api_key=None,
        openrouter_api_key="test-openrouter",
    )
    assert settings.api_key_for("https://polza.ai/api/v1") is None


def test_agent_provider_order_does_not_leak_to_other_family_judge():
    from contractor_agent.agent.llm import make_judge_llm, make_llm
    from contractor_agent.settings import Settings

    settings = Settings(
        _env_file=None,
        llm_provider_order="groq,akashml",
        llm_fallback_models="",
        llm_model="openai/gpt-oss-20b",
        judge_model="z-ai/glm-5.3-flash",
    )
    agent = make_llm(settings).models[0]
    judge = make_judge_llm(settings).models[0]
    assert agent.extra_body == {"provider": {"order": ["groq", "akashml"], "allow_fallbacks": True}}
    assert not judge.extra_body
