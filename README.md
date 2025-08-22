# NSLC Value Finder (one-port)

A tiny **one-port** web app that runs your NSLC scraper on the server and serves the frontend from the same FastAPI app. Results show in a sortable table with CSV export.

> Heads‑up: scraping retail sites can violate Terms of Use; use responsibly and cache results. Default cache = 24h.

## Quick start (local)
1) Ensure Python 3.11+ installed.
2) Put `nslc_spirits.py` next to `server.py` in this folder.
3) Create & activate a virtualenv.
4) Install deps:
   ```bash
   pip install -r requirements.txt
   python -m playwright install
   ```
5) Run the app (serves API **and** site on one port):
   ```bash
   python server.py
   # or
   uvicorn server:app --host 0.0.0.0 --port 8000 --reload
   ```
6) Open **http://localhost:8000/** in your browser.

## Deploying
- **Small VPS (recommended):** copy the folder, set up a venv, run the two install commands above (including `python -m playwright install`), then run with `uvicorn` behind a reverse proxy (Caddy/Nginx). Example `systemd` service:
  ```ini
  [Unit]
  Description=NSLC Value Finder
  After=network.target

  [Service]
  WorkingDirectory=/srv/nslc
  ExecStart=/srv/nslc/venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
  Restart=always
  Environment=CACHE_TTL_SECONDS=86400

  [Install]
  WantedBy=multi-user.target
  ```
- **Ports / domain:** Point your reverse proxy to `localhost:8000`. The app serves `/` and `/static/*` plus the JSON API under `/api/*`.

## How it uses your code
The backend shells out to your `nslc_spirits.py` (CLI) so you don't have to refactor. It creates a temporary CSV, loads it with pandas, and responds with JSON sorted by `score` descending.

Columns expected: `name, price_cad, abv_percent, volume_ml, score, category, url`
