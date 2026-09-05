# ruff: noqa: E501
import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path

import httpx

CASES = [
    ("max", "Финансы МАКСМАРКЕТ, ИНН 5032257375"),
    (
        "max",
        "Есть ли действующие исполнительные производства и на какую сумму? Речь о компании МАКСМАРКЕТ, ИНН 5032257375. Мне нужно сейчас оплатить счёт этой компании.",
    ),
    ("max", "С кем они судились в 2025 году и за что? Речь о компании МАКСМАРКЕТ, ИНН 5032257375."),
    ("max", "Финансы. Речь о компании МАКСМАРКЕТ, ИНН 5032257375."),
    (
        "max",
        "Кто сейчас руководитель и сколько лет компании? Речь о компании МАКСМАРКЕТ, ИНН 5032257375.",
    ),
    ("max", "Сколько у них сотрудников? Речь о компании МАКСМАРКЕТ, ИНН 5032257375."),
    (
        "max",
        "Почему ЗСК зелёный, если счета заблокированы? Речь о компании МАКСМАРКЕТ, ИНН 5032257375.",
    ),
    ("max", "Объясни рекомендацию. Речь о компании МАКСМАРКЕТ, ИНН 5032257375."),
    ("max", "Какие документы запросить перед оплатой? Речь о компании МАКСМАРКЕТ, ИНН 5032257375."),
    ("tech", "Можно давать им отсрочку на 60 дней? Речь о компании ТЕХПРОФ, ИНН 1684017097."),
    ("tech", "Есть ли у них суды и долги у приставов? Речь о компании ТЕХПРОФ, ИНН 1684017097."),
    (
        "gdk",
        "Почему метка банка зелёная, если счета заблокированы? Можно им платить? Речь о компании ГДК, ИНН 6165169320.",
    ),
    ("lemon", "Сколько у них судебных дел? Речь о компании ЛЕ МОНЛИД, ИНН 5029069967."),
    (
        "cmp",
        "Сравни ГДК 6165169320, ТЕХПРОФ 1684017097 и БИЛД-ЮГ 2311304742: с кем лучше работать?",
    ),
]


async def main():
    out = Path("runs/review-develop/acceptance-approved.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    run = uuid.uuid4().hex[:8]
    async with httpx.AsyncClient(timeout=240) as c:
        health = (await c.get("http://127.0.0.1:8084/v1/health")).json()
        digest = hashlib.sha256()
        for path in sorted(Path("src").rglob("*")):
            if path.suffix in {".py", ".html"}:
                digest.update(path.read_bytes())
        for tag, q in CASES:
            start = time.monotonic()
            record = {
                "question": q,
                "thread": f"acceptance-{run}-{tag}",
                "model": health["model"],
                "source_sha256": digest.hexdigest(),
            }
            try:
                r = await c.post(
                    "http://127.0.0.1:8084/v1/runs/stream",
                    json={"thread_id": record["thread"], "input": {"question": q}},
                )
                r.raise_for_status()
                events = [
                    json.loads(line[5:].strip())
                    for line in r.text.splitlines()
                    if line.startswith("data:")
                ]
                last = next((e for e in reversed(events) if e["type"] in ["end", "error"]), None)
                record.update(seconds=round(time.monotonic() - start, 2), event=last)
                print(tag, record["seconds"], last["type"] if last else "missing", flush=True)
                if last and last["type"] == "end":
                    print(last["data"]["output"]["text_md"], flush=True)
                else:
                    print(last, flush=True)
            except Exception as e:
                record["error"] = repr(e)
                print(record["error"], flush=True)
            with out.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


asyncio.run(main())
