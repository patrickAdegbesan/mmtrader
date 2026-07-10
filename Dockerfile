# Project Cognition — paper trading runtime.
# CPU-only torch keeps the image ~1.5GB instead of ~7GB with CUDA wheels.
FROM python:3.11-slim

RUN useradd --create-home --shell /bin/bash cognition
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && grep -v "^torch" requirements.txt > /tmp/reqs.txt \
    && pip install --no-cache-dir -r /tmp/reqs.txt

COPY config/ config/
COPY src/ src/
COPY scripts/ scripts/

# Mutable state lives in volumes (see docker-compose.yml):
#   /app/data (journal + candles), /app/models, /app/logs, /app/reports
RUN mkdir -p data models logs reports && chown -R cognition:cognition /app
USER cognition

ENV PYTHONUNBUFFERED=1
# Testnet-only by default; the code refuses live mode without the
# explicit ALLOW_LIVE_TRADING override regardless of what is set here.
ENV BYBIT_ENV=testnet

CMD ["python", "scripts/paper_trade.py", "--live", "--symbols", "BTC/USDT,ETH/USDT"]
