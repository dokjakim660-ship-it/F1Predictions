"""One-shot helper: open the local Streamlit app in headless Edge and save a
screenshot to docs/screenshot.png. Used to regenerate the README hero image.

Prereqs:
    1. `uv pip install selenium`   (one-off; not in pyproject.toml on purpose
                                    -- only needed when regenerating the shot)
    2. `just app`                  (Streamlit serving on localhost:8501)
    3. `uv run python scripts/screenshot_app.py`

Selenium picks up Edge via the built-in Selenium Manager -- no extra driver
download needed on Windows where Edge ships with the OS."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.edge.options import Options

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshot.png"
URL = "http://localhost:8501"
WAIT_SECONDS = 14  # Streamlit WebSocket reconnect + first render


def main() -> int:
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1280,1100")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--hide-scrollbars")

    driver = webdriver.Edge(options=opts)
    try:
        driver.get(URL)
        time.sleep(WAIT_SECONDS)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        driver.save_screenshot(str(OUT))
        size = OUT.stat().st_size
        print(f"[screenshot] wrote {size} bytes -> {OUT}")
    finally:
        driver.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
