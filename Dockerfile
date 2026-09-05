FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY evals ./evals
RUN uv sync --frozen --no-dev
COPY data ./data
ENV PATH="/app/.venv/bin:$PATH" API_HOST=0.0.0.0 MCP_HOST=0.0.0.0
EXPOSE 8080 8765
CMD ["kontragent-api"]
