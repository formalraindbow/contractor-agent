import json
from pathlib import Path

import pytest
from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters

from contractor_agent.data.loader import Snapshot
from contractor_agent.mcp_server.server import build_server

ROOT = Path(__file__).resolve().parent.parent.parent
TOOL_NAMES = {
    "search_company",
    "get_report_summary",
    "get_risk_signals",
    "get_financials",
    "get_enforcement_summary",
    "get_arbitration_summary",
    "get_section",
    "compare_companies",
}


@pytest.mark.anyio
async def test_in_memory_client(snapshot: Snapshot) -> None:
    async with Client(build_server(snapshot)) as client:
        tools = await client.list_tools()
        assert {t.name for t in tools.tools} == TOOL_NAMES
        assert all(t.description for t in tools.tools)
        schema = next(t for t in tools.tools if t.name == "get_section").input_schema
        assert set(schema["properties"]) == {"inn", "name"}

        found = await client.call_tool("search_company", {"query": "монлид"})
        assert not found.is_error
        assert found.structured_content["available"] is True
        assert found.structured_content["data"]["items"][0]["inn"] == "5029069967"

        absent = await client.call_tool("get_section", {"inn": "772377037026", "name": "licenses"})
        assert absent.structured_content["available"] is False
        assert absent.structured_content["reason"] == "section_absent"

        templates = await client.list_resource_templates()
        assert [t.uri_template for t in templates.resource_templates] == ["report://{inn}"]
        resource = await client.read_resource("report://5029069967")
        report = json.loads(resource.contents[0].text)
        assert report["baseInfo"]["inn"] == "5029069967"


@pytest.mark.anyio
async def test_stdio_subprocess_end_to_end() -> None:
    params = StdioServerParameters(
        command="uv", args=["run", "kontragent-mcp", "--source", "snapshot"], cwd=str(ROOT)
    )
    async with Client(params) as client:
        tools = await client.list_tools()
        assert {t.name for t in tools.tools} == TOOL_NAMES
        res = await client.call_tool("get_risk_signals", {"inn": "5032257375"})
        assert res.structured_content["data"]["verdict"] == "not_recommended"
        assert res.structured_content["report_date"] == "2026-07-31"
