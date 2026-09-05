import json

import pytest

from contractor_agent.cli import main


def test_signals_text(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["signals", "5032257375"]) == 0
    out = capsys.readouterr().out
    assert "СУЩЕСТВЕННЫЕ РИСКИ" in out and "терминальный факт" in out
    assert "← report.status.reasonName" in out
    assert "ЗСК: зелёный" in out


def test_signals_json_and_unknown_inn(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["signals", "1684017097", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "ok" and payload["signals"] == []
    assert main(["signals", "0000000000"]) == 1
    assert "нет компании" in capsys.readouterr().err


def test_zsk_yellow_is_shown_grey(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["signals", "9705152496"]) == 0  # СПОРТ — LOW/YELLOW
    assert "ЗСК: серый" in capsys.readouterr().out
