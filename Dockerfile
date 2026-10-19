# 单进程镜像：JSON 会话存储目前只保证单进程内的并发正确性，不能增加 workers。
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY static ./static

# 应用进程不以 root 身份运行；宿主机挂载 data/ 前需把它交给同一 UID，详见 docs/17。
RUN groupadd --gid 10001 agent \
    && useradd --uid 10001 --gid agent --create-home agent \
    && mkdir -p /app/data \
    && chown -R agent:agent /app

USER agent

EXPOSE 8000

CMD ["uvicorn", "src.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
