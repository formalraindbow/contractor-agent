#!/usr/bin/env bash
# Поднять оба варианта агента на своём компьютере и сравнить их в браузере.
#
#   ./scripts/try-both.sh          поднять обе версии
#   ./scripts/try-both.sh stop     остановить
#
# Что поднимается:
#   http://127.0.0.1:8080  — наша версия, ветка develop
#   http://127.0.0.1:8083  — версия Codex, ветка codex/review-v4
#
# Обе работают на одной модели и на одних данных, поэтому сравнивать честно.
# Что нужно заранее: git, uv (https://docs.astral.sh/uv/) и файл .env с ключом модели —
# скрипт подскажет, если чего-то нет.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODEX_DIR="${ROOT}/../contractor-agent-review-v4"
CODEX_BRANCH="codex/review-v4"
PORT_OURS=8080
PORT_CODEX=8083
LOG_DIR="${ROOT}/runs"

# модель по умолчанию — та же для обеих версий, чтобы сравнивать интерфейс и логику, а не модель
: "${LLM_BASE_URL:=https://ai.api.cloud.yandex.net/v1}"
: "${LLM_MODEL:=gpt://b1gir8dkimq5j60i6ajf/gpt-oss-20b/latest}"

say() { printf '%s\n' "$*"; }
fail() { printf '\n%s\n' "$*" >&2; exit 1; }

stop_all() {
  say "останавливаю…"
  for port in "${PORT_OURS}" "${PORT_CODEX}"; do
    pids="$(lsof -ti :"${port}" 2>/dev/null)"
    [ -n "${pids}" ] && kill ${pids} 2>/dev/null
  done
  sleep 2
  for port in "${PORT_OURS}" "${PORT_CODEX}"; do
    pids="$(lsof -ti :"${port}" 2>/dev/null)"
    [ -n "${pids}" ] && kill -9 ${pids} 2>/dev/null
  done
  say "оба варианта остановлены"
}

[ "${1:-}" = "stop" ] && { stop_all; exit 0; }

command -v git >/dev/null || fail "нет git — поставьте его и повторите"
command -v uv  >/dev/null || fail "нет uv. Поставьте одной командой:
  curl -LsSf https://astral.sh/uv/install.sh | sh
и откройте терминал заново."

[ -f "${ROOT}/.env" ] || fail "нет файла .env с ключом модели.
Скопируйте образец и впишите ключ:
  cp .env.example .env
Для запуска на Яндексе достаточно строки YANDEX_API_KEY=<ключ>.
Ключ спросите у Алана — в общий чат он не выкладывается."

grep -qE "^(YANDEX_API_KEY|OPENROUTER_API_KEY|GROQ_API_KEY)=.+" "${ROOT}/.env" \
  || fail "в .env не заполнен ни один ключ модели (YANDEX_API_KEY, OPENROUTER_API_KEY или GROQ_API_KEY)"

mkdir -p "${LOG_DIR}"

# вторая версия живёт отдельной рабочей копией той же истории git
if [ ! -d "${CODEX_DIR}" ]; then
  say "готовлю вторую версию (ветка ${CODEX_BRANCH})…"
  git -C "${ROOT}" fetch -q origin "${CODEX_BRANCH}" || fail "не удалось получить ветку ${CODEX_BRANCH} с сервера"
  git -C "${ROOT}" worktree add -q "${CODEX_DIR}" "origin/${CODEX_BRANCH}" \
    || fail "не удалось создать рабочую копию в ${CODEX_DIR}"
fi
cp -f "${ROOT}/.env" "${CODEX_DIR}/.env"

start_one() {  # каталог, порт, название, лог
  local dir="$1" port="$2" name="$3" log="$4"
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:${port}/v1/health")"
  if [ "${code}" = "200" ] || [ "${code}" = "401" ]; then
    say "${name} уже работает на порту ${port}"
    return 0
  fi
  say "ставлю зависимости для «${name}» (первый раз это пара минут)…"
  (cd "${dir}" && uv sync --quiet) || fail "не удалось поставить зависимости в ${dir}"
  say "поднимаю «${name}» на порту ${port}…"
  # WEB_PASSWORD не трогаем: локально его обычно нет, а если он задан в .env
  # (машина, с которой стенд открыт наружу), пароль остаётся на месте
  (cd "${dir}" && API_PORT="${port}" \
     LLM_BASE_URL="${LLM_BASE_URL}" LLM_MODEL="${LLM_MODEL}" LLM_FALLBACK_MODELS= LLM_TIMEOUT=180 \
     nohup uv run kontragent-api --port "${port}" > "${log}" 2>&1 &)
  for _ in $(seq 1 60); do
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:${port}/v1/health")"
    [ "${code}" = "200" ] || [ "${code}" = "401" ] && return 0  # 401 = поднялся и закрыт паролем
    sleep 2
  done
  fail "«${name}» не поднялся за две минуты, смотрите ${log}"
}

start_one "${ROOT}"      "${PORT_OURS}"  "наша версия (develop)"  "${LOG_DIR}/api.log"
start_one "${CODEX_DIR}" "${PORT_CODEX}" "версия Codex"           "${LOG_DIR}/api-codex.log"

key=""
grep -qE "^WEB_PASSWORD=.+" "${ROOT}/.env" && key="/?k=$(grep -E '^WEB_PASSWORD=' "${ROOT}/.env" | head -1 | cut -d= -f2-)"

say ""
say "готово, открывайте в браузере:"
say "  наша версия (develop):     http://127.0.0.1:${PORT_OURS}${key}"
say "  версия Codex (review-v4):  http://127.0.0.1:${PORT_CODEX}${key}"
say ""
say "Попробуйте на обеих одно и то же: ИНН 5032257375 (МАКСМАРКЕТ — зелёные оценки банка,"
say "а внутри банкротство), потом спросите «что с судами» и «можно давать отсрочку»."
say "Остановить: ./scripts/try-both.sh stop"

command -v open >/dev/null && { open "http://127.0.0.1:${PORT_OURS}${key}"; open "http://127.0.0.1:${PORT_CODEX}${key}"; }
exit 0
