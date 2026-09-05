"""Измерения одного ответа, включая structured output, повторы и резервные модели."""

from __future__ import annotations

import hashlib
import subprocess
import time
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler


class RunMetrics(BaseCallbackHandler):
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.calls: list[dict[str, Any]] = []
        self.pending: dict[UUID, dict[str, Any]] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        name = (serialized.get("kwargs") or {}).get("model_name") or serialized.get("name")
        self.pending[run_id] = {"requested_model": name, "started": time.monotonic()}

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        call = self.pending.pop(run_id, {"started": time.monotonic()})
        call["duration_s"] = round(time.monotonic() - call.pop("started"), 3)
        metadata, usage = {}, {}
        if response.generations and response.generations[0]:
            message = getattr(response.generations[0][0], "message", None)
            metadata = getattr(message, "response_metadata", None) or {}
            usage = getattr(message, "usage_metadata", None) or {}
        call.update(
            model=metadata.get("model_name") or metadata.get("model"),
            usage=usage,
            finish_reason=metadata.get("finish_reason"),
        )
        self.calls.append(call)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        call = self.pending.pop(run_id, {"started": time.monotonic()})
        call["duration_s"] = round(time.monotonic() - call.pop("started"), 3)
        call.update(error=type(error).__name__, status_code=getattr(error, "status_code", None))
        self.calls.append(call)

    def result(self) -> dict[str, Any]:
        usage = {
            k: sum(int((c.get("usage") or {}).get(k) or 0) for c in self.calls)
            for k in ("input_tokens", "output_tokens", "total_tokens")
        }
        models = list(dict.fromkeys(c["model"] for c in self.calls if c.get("model")))
        return {
            "usage_complete": not any(c.get("error") and not c.get("usage") for c in self.calls),
            "duration_s": round(time.monotonic() - self.started, 3),
            "actual_models": models,
            "llm_calls": len(self.calls),
            "usage": usage,
            "calls": self.calls,
        }


def build_fingerprint() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]

    def digest(paths):
        h = hashlib.sha256()
        for p in sorted(paths):
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
        return h.hexdigest()

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    return {
        "commit": commit,
        "code_sha256": digest((root / "src").rglob("*.py")),
        "prompt_sha256": digest([Path(__file__).with_name("prompt.py")]),
        "gold_sha256": digest((root / "evals").glob("*.yaml")),
    }
