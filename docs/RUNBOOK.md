# Project Cognition — Deployment & Operations Runbook

Everything needed to run the paper-trading system 24/7 on a VPS,
keep it healthy, and recover when something breaks.

**Phase rule: TESTNET ONLY.** The code enforces this twice (config load
and executor construction). Do not set `ALLOW_LIVE_TRADING` — that
override exists for a future phase the product owner must explicitly
approve.

---

## 1. Choosing and preparing a VPS

- **Size**: 2 vCPU / 4 GB RAM / 40 GB disk is comfortable ($5–12/month:
  Hetzner CX22, DigitalOcean basic droplet, Vultr, AWS Lightsail).
  No GPU needed — the networks are small and run CPU-only.
- **Region**: pick one where Bybit is accessible and close to Bybit's
  matching engine (Singapore/Tokyo are good). Check your provider region
  against Bybit's restricted-jurisdiction list.
- **OS**: Ubuntu 24.04 LTS assumed below.
- Basic hardening: create a non-root user, SSH keys only, enable ufw
  (allow SSH only — nothing here needs inbound ports), unattended
  upgrades.

```bash
adduser cognition && usermod -aG sudo,docker cognition
ufw allow OpenSSH && ufw enable
```

## 2. Credentials

1. **Bybit testnet key**: https://testnet.bybit.com → API Management.
   Permissions: trade ONLY (no withdrawal — it must not even be an
   option on the key). Whitelist the VPS's public IP
   (`curl ifconfig.me` on the VPS).
2. **Telegram**: create a bot via @BotFather (token), message the bot
   once, then get your chat id from
   `https://api.telegram.org/bot<TOKEN>/getUpdates`.
3. On the VPS: `cp .env.example .env` and fill in
   `BYBIT_API_KEY`, `BYBIT_API_SECRET`, `BYBIT_ENV=testnet`,
   `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`. `chmod 600 .env`.

## 3. Deploy (Docker — recommended)

```bash
git clone https://github.com/patrickAdegbesan/mmtrader.git /opt/mmtrader
cd /opt/mmtrader
cp .env.example .env && nano .env        # fill credentials
docker compose build
# One-time: pull history + train initial models BEFORE trading
docker compose run --rm paper python scripts/download_data.py
docker compose run --rm paper python scripts/train_ensemble.py \
    --data data/processed/BTC_USDT_1m.parquet --episodes 200
docker compose up -d                     # starts trader + nightly retrainer
```

The `paper` service trades continuously and restarts on crash/reboot;
the `retrainer` service retrains once per day with the promotion gate.

### Deploy without Docker (systemd)

```bash
git clone https://github.com/patrickAdegbesan/mmtrader.git /opt/mmtrader
cd /opt/mmtrader && python3.11 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# download data + train as above, then:
sudo cp deploy/cognition-paper.service /etc/systemd/system/
sudo cp deploy/cognition-retrain.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cognition-paper cognition-retrain.timer
```

## 4. Monitoring

| What | Where |
|---|---|
| Live status dashboard | `reports/dashboard.html` (fetch via `scp`, or serve read-only with `python -m http.server` bound to localhost + SSH tunnel) |
| Structured logs | `logs/*.log` (JSON) or `docker compose logs -f paper` / `journalctl -u cognition-paper -f` |
| Trade journal | `data/journal.db` — `sqlite3 data/journal.db 'SELECT COUNT(*), SUM(pnl) FROM trades;'` |
| Alerts on your phone | Telegram: circuit breaker (critical), feed stalls (critical), risk rejections summary, retrain rollbacks |

**Daily 2-minute check**: Telegram quiet? dashboard equity sane? last
log line recent? That's it — the guardians do the rest.

## 5. Failure playbook

| Symptom | Diagnosis | Action |
|---|---|---|
| Telegram: `circuit_breaker` | −5% daily loss; trading halted until next UTC day | Expected safety behavior. Review `data/journal.db` trades for the day. It self-resumes at UTC midnight; investigate before then if losses look anomalous (e.g. one symbol bleeding). |
| Telegram: `feed_stale` | No candles for >3 bar-lengths | Check Bybit status page; check VPS network. RestPollingFeed retries by itself; the WS feed reconnects with backoff. If stuck: restart the service. |
| Container/service restart loop | `docker compose logs paper` / `journalctl` | Usually bad `.env` (missing keys) or corrupted data file. Fix cause; state (journal, models) survives restarts via volumes. |
| Telegram: `retrain_rollback` | New model underperformed on eval — promotion refused | No action required (that's the gate working). If it repeats for days, the market may have shifted: consider a longer retrain (`--episodes`) on refreshed data. |
| Equity curve flat, zero trades for days | Vote threshold too high for regime, or agents halted | Check dashboard `halted` flag and meta weights; check risk-rejection events in journal (`SELECT * FROM events WHERE kind='risk_rejection' ORDER BY ts DESC LIMIT 20`). |
| Disk filling up | Journal/log growth | Journal grows slowly (KB/trade). Rotate `logs/` (logrotate or `docker compose restart`); never delete `models/` versions without a backup. |

## 6. Model rollback (manual)

Every retrain creates an immutable version under `models/<agent>/vNNNN/`
and only moves `LATEST` if the gate passes. To manually roll back:

```bash
python - <<'EOF'
import sys; sys.path.insert(0, "src")
from cognition.learning.registry import ModelRegistry
reg = ModelRegistry("models", "momentum")
print("versions:", reg.list_versions(), "active:", reg.latest_version())
reg.activate("v0003")            # <- pick the version to restore
EOF
# then restart the trader to load it
docker compose restart paper     # or: sudo systemctl restart cognition-paper
```

## 7. Backups

State worth backing up: `data/journal.db`, `models/`, `.env` (secrets —
store separately/encrypted). Nightly cron example:

```bash
30 3 * * * tar czf /backup/cognition-$(date +\%F).tgz -C /opt/mmtrader data/journal.db models && find /backup -mtime +14 -delete
```

Restore = untar over a fresh clone + `docker compose up -d`.

## 8. Updating the code

```bash
cd /opt/mmtrader
git pull
docker compose build && docker compose up -d       # Docker
# or: . .venv/bin/activate && pip install -r requirements.txt && sudo systemctl restart cognition-paper
```

Model versions and the journal are untouched by code updates.

## 9. Graduation criteria (from the architecture doc — do not skip)

Before ANY consideration of live capital:
1. 4–8 weeks of testnet paper trading across varied volatility;
2. paper results within a reasonable band of backtest results;
3. backtest gate passed on real data: positive expectancy after costs in
   ALL regimes, win rate 55–65%, Sharpe > 1.5, max DD < 15%;
4. explicit product-owner sign-off. Live mode additionally requires the
   `ALLOW_LIVE_TRADING=I_UNDERSTAND_THE_RISK` environment override, a
   live API key (again: trade-only, IP-whitelisted), and a small
   starting allocation.
