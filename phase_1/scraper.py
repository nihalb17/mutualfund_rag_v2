"""
scraper.py — Phase 1
Scrapes the three Groww mutual fund pages using Playwright (headless Chromium).
Groww is a JavaScript-rendered SPA, so plain requests/BeautifulSoup won't work.
Saves raw inner-text of each page to data/raw/<scheme_key>.txt for parsing.
"""

import asyncio
import os
from pathlib import Path

from playwright.async_api import async_playwright

# ── Scheme registry ───────────────────────────────────────────────────────────
SCHEME_URLS: dict[str, str] = {
    "axis_liquid_direct_fund": (
        "https://groww.in/mutual-funds/axis-liquid-direct-fund-growth"
    ),
    "axis_elss_tax_saver": (
        "https://groww.in/mutual-funds/axis-elss-tax-saver-direct-plan-growth"
    ),
    "axis_flexi_cap_fund": (
        "https://groww.in/mutual-funds/axis-flexi-cap-fund-direct-growth"
    ),
}

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


# ── Core scrape function ──────────────────────────────────────────────────────
async def scrape_page(scheme_key: str, url: str) -> str:
    """
    Renders a single Groww page with headless Chromium, waits for dynamic
    content, and returns the full visible inner-text of the page body.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        page = await context.new_page()

        print(f"  -> Navigating to: {url}")
        await page.goto(url, wait_until="domcontentloaded", timeout=90_000)

        # Give JS time to render dynamic data (NAV, AUM, ratios etc.)
        await page.wait_for_timeout(6_000)

        # Attempt to scroll to trigger lazy-loaded sections
        for _ in range(5):
            await page.keyboard.press("End")
            await page.wait_for_timeout(1_000)

        text = await page.inner_text("body")
        await browser.close()
        return text


# ── Orchestrator ──────────────────────────────────────────────────────────────
async def scrape_all() -> dict[str, str]:
    """Scrapes all schemes sequentially and saves raw text files."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}

    for scheme_key, url in SCHEME_URLS.items():
        print(f"\n[Scraper] Fetching: {scheme_key}")
        try:
            text = await scrape_page(scheme_key, url)
            out_path = RAW_DATA_DIR / f"{scheme_key}.txt"
            out_path.write_text(text, encoding="utf-8")
            results[scheme_key] = text
            print(f"  [OK] Saved {len(text):,} chars -> {out_path}")
        except Exception as exc:
            print(f"  [FAIL] Failed to scrape {scheme_key}: {exc}")
            raise

    return results


def run_scraper() -> dict[str, str]:
    """Synchronous entry point."""
    return asyncio.run(scrape_all())


if __name__ == "__main__":
    run_scraper()
