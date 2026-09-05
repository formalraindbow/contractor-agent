#!/bin/zsh
# Поднять стенд целиком: агент + публичная ссылка. Запуск: ./runs/serve.sh
cd /Users/alantolparov/IdeaProjects/contractor-agent

if ! curl -s --max-time 3 -u alfa:kontragent2026 http://127.0.0.1:8080/v1/health > /dev/null; then
  echo "поднимаю агента…"
  WEB_PASSWORD=kontragent2026 \
  LLM_BASE_URL=https://ai.api.cloud.yandex.net/v1 \
  LLM_MODEL="gpt://b1gir8dkimq5j60i6ajf/gpt-oss-20b/latest" \
  LLM_FALLBACK_MODELS= LLM_TIMEOUT=180 \
    nohup uv run kontragent-api > runs/api.log 2>&1 &
  sleep 8
fi
curl -s --max-time 5 -u alfa:kontragent2026 http://127.0.0.1:8080/v1/health > /dev/null \
  && echo "агент работает: http://127.0.0.1:8080" || echo "агент НЕ поднялся, смотри runs/api.log"

if ! pgrep -f "runs/tunnel.sh" > /dev/null; then
  echo "поднимаю публичную ссылку…"
  nohup runs/tunnel.sh > /dev/null 2>&1 &
  sleep 20
fi
echo "ссылка для коллег: https://kontragent-alfa.loca.lt/?k=kontragent2026"
