# server.py
# FastAPI backend with SSE progress, using your scraper's own console output.

import os
import sys
import json
import time
import asyncio
import threading
import tempfile
import subprocess
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import FileResponse, StreamingResponse

# ---------- Paths ----------
BASE_DIR = Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
INDEX_HTML = TEMPLATES_DIR / "index.html"
NSLC_SCRIPT = BASE_DIR / "nslc_spirits.py"

# ---------- Config ----------
SUPPORTED = ["Spirits", "Wine", "Beer", "Cider", "Coolers"]
CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  # 24h
SAFE_MAX_CONCURRENCY = int(os.getenv("SAFE_MAX_CONCURRENCY", "12"))
SCRAPE_TIMEOUT_SECONDS = int(os.getenv("SCRAPE_TIMEOUT_SECONDS", "900"))  # 15m

# Tolerant progress line: [progress] 37/395 (9.4%)
PROG_RE = re.compile(r"^\[progress\]\s*(\d+)\s*/\s*(\d+)\s*\(\s*([\d.]+)\s*%\s*\)\s*$", re.I)
# Tolerant total line: [info] Scraping 395 products with concurrency=8...
INFO_TOTAL_RE = re.compile(r"^\[info\]\s*Scraping\s+(\d+)\s+products", re.I)

# Ensure Windows event loop policy (just in case we ever spawn asyncio subprocesses)
if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())  # type: ignore[attr-defined]
    except Exception:
        pass

# ---------- FastAPI ----------
app = FastAPI(title="NSLC Value Finder")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(INDEX_HTML))


@app.get("/api/health")
async def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}


# ---------- Models ----------
class ScrapeRequest(BaseModel):
    categories: List[str] = ["Spirits"]
    concurrency: int = 5


# ---------- Cache helpers ----------
def _cache_key(cats: List[str]) -> Path:
    key = "-".join(sorted(cats))
    return CACHE_DIR / f"nslc_{key}.json"


def _read_cache(path: Path) -> Optional[List[Dict[str, Any]]]:
    if not path.exists():
        return None
    try:
        age = time.time() - path.stat().st_mtime
        if age > CACHE_TTL_SECONDS:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(path: Path, items: List[Dict[str, Any]]) -> None:
    try:
        path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


# ---------- CSV -> items ----------
def _items_from_csv(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        raise RuntimeError("Expected CSV not created by scraper")
    df = pd.read_csv(csv_path)
    cols = ["name", "price_cad", "abv_percent", "volume_ml", "score", "category", "url"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols].sort_values(["score"], ascending=[False]).reset_index(drop=True)
    return df.to_dict(orient="records")


# ---------- Subprocess runners ----------
def _scrape_via_subprocess(cats: List[str], conc: int) -> List[Dict[str, Any]]:
    """Blocking run: execute script, read CSV, return items."""
    if not NSLC_SCRIPT.exists():
        raise HTTPException(500, detail="nslc_spirits.py not found next to server.py")

    with tempfile.TemporaryDirectory() as td:
        out_csv = Path(td) / "nslc_value.csv"
        cmd = [
            sys.executable, str(NSLC_SCRIPT),
            "--categories", *cats,
            "--out", str(out_csv),
            "--concurrency", str(max(1, min(conc, SAFE_MAX_CONCURRENCY))),
            "--headless",
        ]
        proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
        if proc.returncode != 0:
            raise HTTPException(
                500,
                detail=f"Subprocess failed (code {proc.returncode}).\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
            )
        return _items_from_csv(out_csv)


def _scrape_via_subprocess_stream(
    cats: List[str],
    conc: int,
    on_line: callable,
) -> List[Dict[str, Any]]:
    """
    Streaming run: execute script, forward stdout lines via on_line(),
    then read CSV and return items.
    """
    if not NSLC_SCRIPT.exists():
        raise RuntimeError("nslc_spirits.py not found next to server.py")

    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        out_csv = Path(td) / "nslc_value.csv"
        cmd = [
            sys.executable, str(NSLC_SCRIPT),
            "--categories", *cats,
            "--out", str(out_csv),
            "--concurrency", str(max(1, min(conc, SAFE_MAX_CONCURRENCY))),
            "--headless",
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=0,  # unbuffered so we can read char-by-char
        )
        assert proc.stdout is not None

        buf = ""
        while True:
            ch = proc.stdout.read(1)  # read single chars so '\r' progress updates are seen
            if ch == "" and proc.poll() is not None:
                break
            if not ch:
                # keep UI alive and enforce timeout
                if time.time() - t0 > SCRAPE_TIMEOUT_SECONDS:
                    proc.kill()
                    raise RuntimeError("Scrape timed out")
                time.sleep(0.05)
                continue

            if ch in ("\r", "\n"):
                line = buf.strip()
                if line:
                    try:
                        on_line(line)
                    except Exception:
                        pass
                buf = ""
            else:
                buf += ch

            if time.time() - t0 > SCRAPE_TIMEOUT_SECONDS:
                proc.kill()
                raise RuntimeError("Scrape timed out")

        if buf.strip():
            try:
                on_line(buf.strip())
            except Exception:
                pass

        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"Scraper exit code {ret}")

        return _items_from_csv(out_csv)


# ---------- SSE helpers ----------
def _sse(event: str, payload: Dict[str, Any]) -> str:
    return f"event: {event}\n" + f"data: {json.dumps(payload)}\n\n"


# ---------- Routes ----------
@app.post("/api/scrape")
async def scrape(req: ScrapeRequest):
    cats = [c for c in req.categories if c in SUPPORTED]
    if not cats:
        raise HTTPException(400, detail=f"No valid categories. Allowed: {SUPPORTED}")

    cache_path = _cache_key(cats)
    cached = _read_cache(cache_path)
    if cached is not None:
        return {"cached": True, "count": len(cached), "items": cached}

    try:
        items = _scrape_via_subprocess(cats, req.concurrency)
        _write_cache(cache_path, items)
        return {"cached": False, "count": len(items), "items": items}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Server error: {e}")


@app.get("/api/scrape/stream")
async def scrape_stream(categories: str = "Spirits", concurrency: int = 5):
    """
    Streams progress via SSE by parsing your scraper's stdout.
    Emits:
      - progress: {"pct": ..., "done": d, "total": t, "message": "Items d/t"}
      - log: {"line": "..."} for any non-progress lines
      - done: {"count": N} when cache is filled
    """
    cats = [c for c in (categories.split(",") if categories else []) if c in SUPPORTED] or ["Spirits"]
    conc = max(1, min(concurrency, SAFE_MAX_CONCURRENCY))
    cache_path = _cache_key(cats)

    async def event_gen():
        yield _sse("status", {"message": "Starting…"})
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[str] = asyncio.Queue()

        def on_line(line: str) -> None:
            loop.call_soon_threadsafe(q.put_nowait, line)

        done_fut: asyncio.Future = loop.create_future()

        def worker():
            try:
                items = _scrape_via_subprocess_stream(cats, conc, on_line)
                loop.call_soon_threadsafe(done_fut.set_result, items)
            except Exception as e:
                loop.call_soon_threadsafe(done_fut.set_exception, e)

        threading.Thread(target=worker, daemon=True).start()

        # Emit initial heartbeat
        yield _sse("progress", {"pct": 1, "message": "Launching scraper…"})

        # Pump lines and convert to events
        while not done_fut.done():
            emitted = False
            try:
                # drain quickly to keep UI snappy
                for _ in range(50):
                    line = q.get_nowait()
                    emitted = True

                    m = PROG_RE.match(line)
                    if m:
                        done_i = int(m.group(1))
                        total_i = int(m.group(2))
                        pct_val = float(m.group(3))
                        yield _sse("progress", {
                            "pct": pct_val,
                            "done": done_i,
                            "total": total_i,
                            "message": f"Items {done_i}/{total_i}",
                        })
                    else:
                        # Also try to capture the total from an info line
                        mi = INFO_TOTAL_RE.match(line)
                        if mi:
                            total_i = int(mi.group(1))
                            yield _sse("progress", {
                                "pct": 1,
                                "done": 0,
                                "total": total_i,
                                "message": f"Items 0/{total_i}",
                            })
                        # Forward the raw log so the UI can show it as status text
                        yield _sse("log", {"line": line})
            except asyncio.QueueEmpty:
                pass

            # If nothing new, send a small heartbeat to keep the bar alive
            if not emitted:
                yield _sse("progress", {"pct": 2, "message": "Working…"})
                await asyncio.sleep(0.5)

        # Finished: either success or error
        exc = done_fut.exception()
        if exc:
            yield _sse("error", {"message": f"{exc.__class__.__name__}: {exc}"})
            return

        items: List[Dict[str, Any]] = done_fut.result()  # type: ignore[assignment]
        _write_cache(cache_path, items)
        yield _sse("progress", {"pct": 100, "message": "Done — loading results…"})
        yield _sse("done", {"count": len(items)})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------- Dev entry ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)