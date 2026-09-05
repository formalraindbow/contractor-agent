#!/bin/zsh
# публичная ссылка с постоянным адресом: localtunnel рвётся, поэтому поднимаем в цикле
cd /Users/alantolparov/IdeaProjects/contractor-agent
while true; do
  npx --yes localtunnel --port 8080 --subdomain kontragent-alfa >> runs/lt.log 2>&1
  echo "=== туннель упал $(date +%H:%M:%S), поднимаю заново" >> runs/lt.log
  sleep 3
done
