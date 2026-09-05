import json

from mcp.client.client import Client

from contractor_agent.agent.tools import load_tools
from contractor_agent.data.loader import Snapshot
from contractor_agent.mcp_server.server import build_server


async def test_mcp_tools_become_langchain_tools(snapshot: Snapshot) -> None:
    async with Client(build_server(snapshot)) as client:
        tools = await load_tools(client)
        by_name = {t.name: t for t in tools}
        assert len(tools) == 8 and "get_section" in by_name
        assert "inn" in by_name["get_section"].args_schema["properties"]
        raw = await by_name["get_report_summary"].ainvoke({"inn": "5029069967"})
        payload = json.loads(raw)
        assert payload["available"] and payload["data"]["short_name"] == 'ООО "ЛЕ МОНЛИД"'
        absent = json.loads(
            await by_name["get_section"].ainvoke({"inn": "772377037026", "name": "licenses"})
        )
        assert absent["available"] is False and absent["reason"] == "section_absent"
