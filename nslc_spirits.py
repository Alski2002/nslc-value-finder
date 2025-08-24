# nslc_spirits.py
import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set

import pandas as pd
from playwright.async_api import async_playwright, Page, BrowserContext, TimeoutError as PlaywrightTimeoutError

# ---------------------------------
# Config
# ---------------------------------
# Valid NSLC categories. Keys are what you pass on the CLI.

CATEGORY_SLUG = {
    "Spirits": "Spirits",
    "Wine": "Wine",
    "Beer": "Beer",
    "Cider": "Cider",
    "Coolers": "Coolers",
}

def category_url(cat: str) -> str:
    # e.g. "Wine" -> "https://www.mynslc.com/products/Wine"
    return f"https://www.mynslc.com/products/{CATEGORY_SLUG[cat]}"

PRODUCT_HREF_RE = re.compile(r"/(?:en/)?products/.*?\.aspx$", re.IGNORECASE)
DEBUG_SHOTS = False # set True to write screenshots to ./debug_shots

# ---------- Small helpers ----------

async def _screenshot(page: Page, name: str) -> None:
    if not DEBUG_SHOTS:
        return
    import os
    os.makedirs("debug_shots", exist_ok=True)
    await page.screenshot(path=f"debug_shots/{name}.png", full_page=True)

def _abs(href: Optional[str]) -> Optional[str]:
    if not href:
        return None
    if href.startswith("/"):
        return "https://www.mynslc.com" + href
    if href.startswith("http"):
        return href
    return None

async def inner_text(page: Page, selector: str) -> Optional[str]:
    try:
        loc = page.locator(selector)
        if await loc.count():
            return (await loc.first.inner_text()).strip()
    except:
        pass
    return None

async def body_text(page: Page) -> str:
    try:
        return await page.locator("body").inner_text()
    except:
        return ""

# ---------- Parsers ----------

def parse_price(text: str) -> Optional[float]:
    if not text:
        return None
    text = text.replace(",", "")
    m = re.search(r"\$\s*([0-9]+(?:\.[0-9]{1,2})?)", text)
    if m:
        return float(m.group(1))
    m2 = re.search(r"\b([0-9]+(?:\.[0-9]{1,2}))\b", text)
    return float(m2.group(1)) if m2 else None

def parse_abv(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    return float(m.group(1)) if m else None

def parse_volume_ml(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(mL|ml|L|l)\b", text)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).lower()
    return val if unit == "ml" else val * 1000.0

# ---------- Product page scraping ----------

async def expand_product_details(page: Page) -> None:
    try:
        toggle = page.get_by_text("Product Details", exact=False)
        if await toggle.count():
            await toggle.first.click()
            await page.wait_for_timeout(300)
    except:
        pass

async def extract_price(page: Page) -> Optional[float]:
    # Preferred container
    txt = await inner_text(page, ".Product_Info-price")
    if txt:
        p = parse_price(txt)
        if p and p > 0:
            return p

    # Alternates
    for sel in ("span.price", ".product-price", "[data-test='product-price']", ".price"):
        txt = await inner_text(page, sel)
        if txt:
            p = parse_price(txt)
            if p and p > 0:
                return p

    # itemprop="price"
    try:
        loc = page.locator("[itemprop='price']")
        if await loc.count():
            content = await loc.first.get_attribute("content")
            if content:
                try:
                    p = float(content.replace(",", ""))
                    if p > 0:
                        return p
                except:
                    pass
            txt = await loc.first.inner_text()
            if txt:
                p = parse_price(txt)
                if p and p > 0:
                    return p
    except:
        pass

    # JSON-LD offers.price
    try:
        scripts = page.locator("script[type='application/ld+json']")
        n = await scripts.count()
        for i in range(n):
            raw = await scripts.nth(i).inner_text()
            try:
                data = json.loads(raw)
            except:
                continue
            objs = data if isinstance(data, list) else [data]
            for obj in objs:
                if not isinstance(obj, dict):
                    continue
                offers = obj.get("offers")
                if not offers:
                    continue
                offers_list = offers if isinstance(offers, list) else [offers]
                for off in offers_list:
                    if not isinstance(off, dict):
                        continue
                    price_val = off.get("price")
                    if price_val is None:
                        continue
                    try:
                        p = float(str(price_val).replace(",", ""))
                        if p > 0:
                            return p
                    except:
                        continue
    except:
        pass

    # Fallback: first $ in body
    p = parse_price(await body_text(page))
    return p if (p and p > 0) else None

async def extract_abv_and_volume(page: Page) -> Tuple[Optional[float], Optional[float]]:
    txt = await body_text(page)
    abv = parse_abv(txt)
    vol_ml = parse_volume_ml(txt)

    if abv is None or vol_ml is None:
        try:
            heading = page.get_by_text("Product Details", exact=False).first
            container = heading.locator("xpath=ancestor::*[1] | xpath=ancestor::*[2]")
            if await container.count():
                scoped = await container.first.inner_text()
                abv = abv or parse_abv(scoped)
                vol_ml = vol_ml or parse_volume_ml(scoped)
        except:
            pass

    if vol_ml is None:
        name_text = await inner_text(page, "h1.product-name") or await inner_text(page, "h1")
        if name_text:
            vol_ml = parse_volume_ml(name_text)

    return abv, vol_ml

async def scrape_product(context: BrowserContext, url: str, retries: int = 1) -> Optional[Dict[str, Any]]:
    page = await context.new_page()
    try:
        for attempt in range(retries + 1):
            try:
                try:
                    await page.goto(url, timeout=180000, wait_until="domcontentloaded")
                except PlaywrightTimeoutError:
                    # warm up on homepage (some CDNs do a first-visit challenge)
                    try:
                        await page.goto("https://www.mynslc.com/", timeout=90000, wait_until="domcontentloaded")
                        await page.wait_for_timeout(6000)
                    except:
                        pass
                    await page.goto(url, timeout=180000, wait_until="domcontentloaded")
            except PlaywrightTimeoutError:
                if attempt < retries:
                    continue
                print(f"[warn] Timeout loading {url}")
                return None
            except Exception as e:
                if attempt < retries:
                    continue
                print(f"[warn] Failed to load {url}: {e}")
                return None

            # Cookie/consent (best effort)
            for name in ("Accept", "I Accept", "Allow all cookies", "Got it", "I Agree"):
                try:
                    btn = page.get_by_role("button", name=name)
                    if await btn.count():
                        await btn.first.click(timeout=1500)
                except:
                    pass

            name = (
                await inner_text(page, "h1.product-name")
                or await inner_text(page, "h1[itemprop='name']")
                or await inner_text(page, "h1")
                or (await page.title())
                or "Unknown"
            ).strip()

            await expand_product_details(page)
            try:
                await page.wait_for_selector("text=ABV", timeout=2000)
            except:
                pass

            price = await extract_price(page)
            abv, vol_ml = await extract_abv_and_volume(page)

            if (price is None or price <= 0 or abv is None or vol_ml is None) and attempt < retries:
                await page.wait_for_timeout(350)
                continue

            if price is None or price <= 0 or abv is None or vol_ml is None:
                missing: List[str] = []
                if price is None or price <= 0: missing.append("price")
                if abv is None: missing.append("ABV")
                if vol_ml is None: missing.append("volume")
                print(f"[warn] Skipping (missing {', '.join(missing)}): {url}")
                return None

            score = round((vol_ml * (abv / 100.0)) / price, 2)
            return {
                "name": name,
                "url": url,
                "price_cad": round(price, 2),
                "abv_percent": round(abv, 2),
                "volume_ml": round(vol_ml, 0),
                "score": score,
            }
        return None
    finally:
        await page.close()

# ---------- Collect ALL product links (PAGER-ONLY, count-based waits) ----------

async def collect_all_product_links(context: BrowserContext, start_url: str) -> List[str]:
    """
    Robust URL-based paginator:
      - Navigates with ?Start=<offset>
      - Parses "X–Y of N" in multiple formats (items/results/products; comma N allowed)
      - Advances using parsed end (Y) when available; otherwise uses observed page size
      - Stops when a page adds no new links (works even if the count string is missing)
    """
    page: Page = await context.new_page()
    try:
        print("[info] Opening category (URL-pagination mode)…")
        await page.goto(start_url, timeout=60000, wait_until="domcontentloaded")

        if "/en/products/" in page.url:
            await page.goto(page.url.replace("/en/products/", "/products/"),
                            timeout=60000, wait_until="domcontentloaded")

        # Helpers
        anchor_sel = ("a[href^='/products/'][href$='.aspx'], "
                      "a[href^='/en/products/'][href$='.aspx']")

        async def collect_links_into(seen: Set[str]) -> int:
            added = 0
            anchors = await page.locator(anchor_sel).all()
            for a in anchors:
                try:
                    href = await a.get_attribute("href")
                    if not href:
                        continue
                    if not PRODUCT_HREF_RE.search(href):
                        continue
                    if href.startswith("/"):
                        href = "https://www.mynslc.com" + href
                    if href not in seen:
                        seen.add(href)
                        added += 1
                except:
                    continue
            return added

        async def count_anchors() -> int:
            return await page.locator(anchor_sel).count()

        async def parse_counts() -> tuple[Optional[int], Optional[int], Optional[int]]:
            """
            Returns (first, last, total) if found, else (None, None, None).
            Accepts: '1-12 of 846 items', '1 – 24 of 182 results', '13-24 of 1,234 products', etc.
            """
            txt = (await body_text(page)) or ""
            m = re.search(r"(\d+)\s*[-–]\s*(\d+)\s+of\s+([\d,]+)\s+(?:items?|results?|products?)",
                          txt, re.I)
            if not m:
                # looser: allow missing trailing word
                m = re.search(r"(\d+)\s*[-–]\s*(\d+)\s+of\s+([\d,]+)", txt, re.I)
            if not m:
                return None, None, None
            first, last, total = int(m.group(1)), int(m.group(2)), int(m.group(3).replace(",", ""))
            return first, last, total

        # Initial page_size discovery
        # Nudge to mount grid (some lazy UIs)
        for _ in range(2):
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(200)

        first, last, total = await parse_counts()
        if first and last:
            page_size = last - first + 1
        else:
            # fallback to visible card count
            c = await count_anchors()
            page_size = c if c > 0 else 12

        if total is not None:
            print(f"[info] Detected page_size={page_size}, total_items={total}")
        else:
            print(f"[info] Detected page_size={page_size}, total_items=? (will infer)")

        seen: Set[str] = set()
        start = 0
        page_idx = 0
        max_pages_safety = 300  # hard stop to avoid infinite loops

        while page_idx < max_pages_safety:
            target = f"{start_url}?Start={start}"
            await page.goto(target, timeout=60000, wait_until="domcontentloaded")

            # Wait for product tiles to (re)render by watching count change
            before = await count_anchors()
            for _ in range(60):  # ~12s
                await page.wait_for_timeout(200)
                after = await count_anchors()
                if after != before and after > 0:
                    break

            prev_total = len(seen)
            await collect_links_into(seen)

            # Parse the current visible range, update totals if available
            first, last, total_now = await parse_counts()
            if total_now:
                total = total_now  # remember the correct total once we see it

            page_idx += 1
            label = f"{first}-{last}" if (first and last) else f"Start={start}"
            print(f"[info] Page {page_idx} ({label}) -> total links: {len(seen)}")

            # Stop if nothing new appeared on this page
            if len(seen) == prev_total:
                print("[info] No new links on this page; assuming end of category.")
                break

            # Compute next start
            if last:
                next_start = last  # e.g., '1-12' -> 12 ; '13-24' -> 24
            else:
                next_start = start + page_size

            # If we know total, stop once we reached/passed it
            if total and next_start >= total:
                break

            # Also break if page size looks smaller (= last page)
            if last and first and (last - first + 1) < page_size:
                if total is None:
                    # set a plausible total so the log reflects reality
                    total = last
                break

            # Advance
            start = next_start

        links = sorted(seen)
        print(f"[info] Total unique product links collected: {len(links)}")
        return links

    finally:
        try:
            await page.close()
        except:
            pass

# ---------- Runner ----------

async def scrape_one_category(context: BrowserContext, cat: str, headless: bool, concurrency: int) -> pd.DataFrame:
    start = category_url(cat)
    print(f"[info] Collecting links for category: {cat} -> {start}")
    urls = await collect_all_product_links(context, start)
    if not urls:
        print(f"[warn] No links found for {cat}")
        return pd.DataFrame()

    # reuse your worker pool exactly as-is, but return a DataFrame instead of writing CSV here
    results: List[Dict[str, Any]] = []
    sem = asyncio.Semaphore(max(1, concurrency))

    async def worker(u: str):
        async with sem:
            data = await scrape_product(context, u, retries=1)
            if data:
                data["category"] = cat  # tag the row with the category
                results.append(data)

    await asyncio.gather(*(worker(u) for u in urls))
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)

async def run(all_spirits: bool,
              input_file: Optional[Path],
              out_csv: Path,
              headless: bool,
              concurrency: int) -> None:
    """Single run that either scrapes the whole current category (via CATEGORY_URL)
    or scrapes a provided list of URLs from input_file.
    """
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        # launch browser with flags that work on small VPSes
        browser = await p.chromium.launch(
            headless=headless,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--no-zygote", "--disable-gpu", "--disable-blink-features=AutomationControlled"]
        )
        context = await browser.new_context(
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/123.0.0.0 Safari/537.36"),
            viewport={"width": 1400, "height": 1000},
            locale="en-CA",
            timezone_id="America/Halifax",
            # If your Playwright version supports it:
            # service_workers="block",
        )
        # Slightly more “human-like” headers and stealth hints
        await context.set_extra_http_headers({
            "Accept-Language": "en-CA,en;q=0.9",
            "DNT": "1",
            "Upgrade-Insecure-Requests": "1",
        })
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4,5] });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-CA','en'] });
        """)

        # -------- load URLs --------
        if all_spirits:
            # Uses the module-level CATEGORY_URL that you set in main() per category
            print("[info] Collecting product links for current category...")
            urls = await collect_all_product_links(context, CATEGORY_URL)
        else:
            if not input_file:
                raise SystemExit("You must pass --all or provide --in <urls.txt>")
            urls = [ln.strip() for ln in input_file.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and not ln.strip().startswith("#")]
            print(f"[info] Loaded {len(urls)} URLs from {input_file}")

        if not urls:
            print("[error] No product URLs found.")
            await context.close()
            await browser.close()
            return

        # -------- scrape with concurrency --------
        total = len(urls)
        sem = asyncio.Semaphore(max(1, concurrency))
        results: List[Dict[str, Any]] = []
        done = 0
        lock = asyncio.Lock()

        # OPTIONAL checkpoints
        ENABLE_CHECKPOINT = True
        CHECKPOINT_EVERY = 24

        async def worker(u: str) -> None:
            nonlocal done
            async with sem:
                data = await scrape_product(context, u, retries=1)
                if data:
                    results.append(data)
                async with lock:
                    done += 1
                    if ENABLE_CHECKPOINT and (done % CHECKPOINT_EVERY == 0):
                        tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
                        pd.DataFrame(results).to_csv(tmp, index=False)

        async def progress_printer() -> None:
            last = -1
            while True:
                async with lock:
                    d = done
                if d >= total:
                    print(f"\r[progress] {total}/{total} (100.0%)", end="", flush=True)
                    break
                pct = (d / total) * 100.0
                if d != last:
                    print(f"\r[progress] {d}/{total} ({pct:.1f}%)", end="", flush=True)
                    last = d
                await asyncio.sleep(0.5)
            print()

        print(f"[info] Scraping {total} products with concurrency={concurrency}...")
        prog_task = asyncio.create_task(progress_printer())
        await asyncio.gather(*(worker(u) for u in urls))
        await prog_task

        await context.close()
        await browser.close()

    # -------- write output --------
    if not results:
        print("[error] No products could be scraped successfully.")
        return

    df = pd.DataFrame(results).sort_values(by="score", ascending=False).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    with pd.option_context("display.max_rows", 30, "display.max_colwidth", 150):
        print("\n=== NSLC – Value Score (mL of pure alcohol per $) ===")
        print(df[["name", "price_cad", "abv_percent", "volume_ml", "score", "url"]])

    print(f"\nSaved: {out_csv.resolve()}")

def main() -> None:
    import tempfile

    # Which categories we allow on the CLI
    CATEGORY_SLUG = {
        "Spirits": "Spirits",
        "Wine": "Wine",
        "Beer": "Beer",
        "Cider": "Cider",
        "Coolers": "Coolers",
    }

    parser = argparse.ArgumentParser(description="NSLC value scraper")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["Spirits"],
        choices=list(CATEGORY_SLUG.keys()),
        help="One or more categories to scrape (Spirits, Wine, Beer, Cider, Coolers).",
    )
    parser.add_argument(
        "--out",
        dest="out_csv",
        type=Path,
        default=Path("nslc_value.csv"),
        help="Output CSV path (or filename prefix when using --split-per-category).",
    )
    parser.add_argument(
        "--split-per-category",
        action="store_true",
        help="Write one CSV per category instead of a single combined CSV.",
    )
    parser.add_argument("--headed", action="store_true", help="Show the browser UI while scraping.")
    parser.add_argument("--headless", action="store_true", help="Force headless mode.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of product pages scraped in parallel (default: 5).",
    )
    args = parser.parse_args()

    if args.headed and args.headless:
        raise SystemExit("Choose either --headed or --headless, not both.")
    headless = not args.headed

    # Helper to build a category URL
    def category_url(cat: str) -> str:
        return f"https://www.mynslc.com/products/{CATEGORY_SLUG[cat]}"

    # We’ll run your existing `run(...)` once per category.
    # If not splitting, we write to temp files, then combine.
    temp_paths: list[tuple[str, Path]] = []  # (category, temp_path)

    for cat in args.categories:
        # 1) Point your global CATEGORY_URL at the current category
        global CATEGORY_URL
        CATEGORY_URL = category_url(cat)

        # 2) Decide output for this pass
        if args.split_per_category:
            out_path = args.out_csv.with_name(f"{args.out_csv.stem}_{cat}{args.out_csv.suffix}")
        else:
            # write to a temp file for later concatenation
            tmp = Path(tempfile.gettempdir()) / f"nslc_{cat}_{next(tempfile._get_candidate_names())}.csv"
            out_path = tmp
            temp_paths.append((cat, out_path))

        print(f"[info] === Scraping category: {cat} -> {CATEGORY_URL} ===")
        # Reuse your existing run(...) signature unchanged:
        # run(all_spirits: bool, input_file: Optional[Path], out_csv: Path, headless: bool, concurrency: int)
        asyncio.run(run(
            all_spirits=True,
            input_file=None,
            out_csv=out_path,
            headless=headless,
            concurrency=args.concurrency,
        ))

    # 3) If we wrote temp files (combined mode), load + tag + merge
    if not args.split_per_category:
        frames: list[pd.DataFrame] = []
        for cat, p in temp_paths:
            try:
                df = pd.read_csv(p)
                if "category" not in df.columns:
                    df["category"] = cat
                frames.append(df)
            except Exception as e:
                print(f"[warn] Could not read temp CSV for {cat} ({p}): {e}")

        if not frames:
            print("[error] No data to combine.")
            return

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(["category", "score"], ascending=[True, False])
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.out_csv, index=False)
        print(f"[info] Saved combined CSV: {args.out_csv.resolve()}")

if __name__ == "__main__":
    main()
