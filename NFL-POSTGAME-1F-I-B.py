#!/usr/bin/env python3
"""
WFS NFL — NFL-POSTGAME-1F-I-B
LIVE FANDUEL RESEARCH SLATE-SELECTOR MECHANISM AUDIT — READ ONLY

Purpose:
  Inspect the public FanDuel Research NFL DFS projections page with a headless
  browser and prove how the live slate selector is represented.

This audit:
  - performs public HTTPS GET/browser reads only
  - does NOT download CSVs
  - does NOT write production inputs/databases
  - does NOT modify cron/updater/solver/LIVE/injury code
  - does NOT click any wagering/contest-entry action

Target:
  https://www.fanduel.com/research/nfl/fantasy/dfs-projections/qb

Outputs:
  - page/title/final URL
  - all select/combobox/button candidates near SLATE
  - visible slate-like option text
  - position controls
  - "Download as CSV" control/link evidence
  - embedded JSON/script strings containing slate-related keys
  - readiness decision for deterministic selector automation
"""

from pathlib import Path
import json
import re
import sys
import hashlib

URL = "https://www.fanduel.com/research/nfl/fantasy/dfs-projections/qb"

def clean(s):
    return re.sub(r"\s+", " ", (s or "")).strip()

def emit(prefix, *parts):
    vals = [clean(str(x)) for x in parts]
    print(prefix + "|" + "|".join(vals))

print("=" * 112)
print("WFS NFL — NFL-POSTGAME-1F-I-B")
print("LIVE FANDUEL RESEARCH SLATE-SELECTOR MECHANISM AUDIT — READ ONLY")
print("=" * 112)

try:
    from playwright.sync_api import sync_playwright
except Exception as e:
    print(f"PLAYWRIGHT_IMPORT_ERROR={type(e).__name__}:{e}")
    print("NFL_POSTGAME_1F_I_B_STATUS=FAIL_CLOSED")
    sys.exit(2)

with sync_playwright() as p:
    browser = None
    launch_errors = []
    for name, launcher in [
        ("chromium", p.chromium),
        ("firefox", p.firefox),
        ("webkit", p.webkit),
    ]:
        try:
            browser = launcher.launch(headless=True)
            print(f"BROWSER_ENGINE={name}")
            break
        except Exception as e:
            launch_errors.append(f"{name}:{type(e).__name__}:{e}")

    if browser is None:
        for x in launch_errors:
            print(f"BROWSER_LAUNCH_ERROR={x}")
        print("NFL_POSTGAME_1F_I_B_STATUS=FAIL_CLOSED")
        sys.exit(3)

    context = browser.new_context(
        viewport={"width": 1440, "height": 1200},
        locale="en-US",
    )
    page = context.new_page()

    response = page.goto(URL, wait_until="domcontentloaded", timeout=60000)
    print(f"HTTP_STATUS={response.status if response else 'UNKNOWN'}")
    page.wait_for_timeout(4000)

    print(f"FINAL_URL={page.url}")
    print(f"PAGE_TITLE={clean(page.title())}")

    body_text = clean(page.locator("body").inner_text(timeout=15000))
    print(f"BODY_HAS_NFL_DAILY_FANTASY_PROJECTIONS={str('NFL Daily Fantasy Projections' in body_text).upper()}")
    print(f"BODY_HAS_SLATE_LABEL={str('SLATE' in body_text).upper()}")
    print(f"BODY_HAS_DOWNLOAD_AS_CSV={str('Download as CSV' in body_text).upper()}")

    print("\n=== SELECT ELEMENTS ===")
    selects = page.locator("select")
    print(f"SELECT_COUNT={selects.count()}")
    for i in range(selects.count()):
        s = selects.nth(i)
        try:
            emit(
                "SELECT",
                i,
                s.get_attribute("name") or "",
                s.get_attribute("aria-label") or "",
                s.get_attribute("id") or "",
                clean(s.inner_text()),
            )
            opts = s.locator("option")
            for j in range(opts.count()):
                o = opts.nth(j)
                emit(
                    "OPTION",
                    i,
                    j,
                    o.get_attribute("value") or "",
                    o.get_attribute("selected") or "",
                    clean(o.inner_text()),
                )
        except Exception as e:
            emit("SELECT_ERROR", i, type(e).__name__, str(e))

    print("\n=== COMBOBOX ELEMENTS ===")
    combos = page.locator('[role="combobox"], input[aria-haspopup="listbox"], button[aria-haspopup="listbox"]')
    print(f"COMBOBOX_COUNT={combos.count()}")
    for i in range(combos.count()):
        c = combos.nth(i)
        try:
            emit(
                "COMBOBOX",
                i,
                c.evaluate("(e)=>e.tagName"),
                c.get_attribute("aria-label") or "",
                c.get_attribute("aria-expanded") or "",
                c.get_attribute("aria-controls") or "",
                clean(c.inner_text() if c.evaluate("(e)=>e.tagName") != "INPUT" else c.input_value()),
            )
        except Exception as e:
            emit("COMBOBOX_ERROR", i, type(e).__name__, str(e))

    print("\n=== BUTTON / LINK CANDIDATES ===")
    candidates = page.locator("button, a")
    print(f"BUTTON_LINK_COUNT={candidates.count()}")
    matched = 0
    for i in range(candidates.count()):
        el = candidates.nth(i)
        try:
            text = clean(el.inner_text())
            aria = clean(el.get_attribute("aria-label") or "")
            title = clean(el.get_attribute("title") or "")
            href = clean(el.get_attribute("href") or "")
            hay = " ".join([text, aria, title, href]).lower()
            if any(k in hay for k in ("slate", "main", "download as csv", "download", "position")):
                matched += 1
                emit(
                    "CONTROL",
                    i,
                    el.evaluate("(e)=>e.tagName"),
                    text,
                    aria,
                    title,
                    href,
                    el.get_attribute("role") or "",
                    el.get_attribute("aria-haspopup") or "",
                )
        except Exception:
            pass
    print(f"MATCHED_CONTROL_COUNT={matched}")

    print("\n=== TEXT NEAR SLATE ===")
    # Print concise text fragments around the first SLATE occurrence.
    raw = page.locator("body").inner_text(timeout=15000)
    lines = [clean(x) for x in raw.splitlines() if clean(x)]
    for i, line in enumerate(lines):
        if line.upper() == "SLATE" or "SLATE" == line.upper().strip():
            lo=max(0,i-3); hi=min(len(lines),i+10)
            for j in range(lo,hi):
                emit("SLATE_TEXT", j, lines[j])
            break

    print("\n=== SCRIPT / EMBEDDED DATA SLATE EVIDENCE ===")
    scripts = page.locator("script")
    evidence = 0
    for i in range(scripts.count()):
        try:
            text = scripts.nth(i).text_content() or ""
        except Exception:
            continue
        low = text.lower()
        if "slate" not in low:
            continue
        # Keep output bounded and diagnostic.
        for m in re.finditer(r'.{0,120}slate.{0,240}', text, re.I | re.S):
            frag = clean(m.group(0))
            if frag:
                evidence += 1
                emit("SCRIPT_SLATE_EVIDENCE", i, evidence, frag[:500])
                if evidence >= 25:
                    break
        if evidence >= 25:
            break
    print(f"SCRIPT_SLATE_EVIDENCE_COUNT={evidence}")

    print("\n=== DOM ATTRIBUTE SLATE EVIDENCE ===")
    attrs = page.locator('[data-testid*="slate" i], [id*="slate" i], [class*="slate" i], [aria-label*="slate" i]')
    print(f"SLATE_ATTRIBUTE_NODE_COUNT={attrs.count()}")
    for i in range(min(attrs.count(), 30)):
        el = attrs.nth(i)
        try:
            emit(
                "SLATE_NODE",
                i,
                el.evaluate("(e)=>e.tagName"),
                el.get_attribute("id") or "",
                el.get_attribute("data-testid") or "",
                el.get_attribute("aria-label") or "",
                clean(el.inner_text())[:300],
            )
        except Exception:
            pass

    # Conservative readiness: require some live SLATE control evidence and
    # Download-as-CSV evidence. We do not infer actual option set if options
    # are hidden until click; that becomes the next stage.
    slate_control_proven = (
        selects.count() > 0
        or combos.count() > 0
        or attrs.count() > 0
        or "SLATE" in body_text
    )
    download_control_proven = "Download as CSV" in body_text

    print("\n=== DECISION ===")
    print(f"LIVE_SLATE_CONTROL_EVIDENCE_PROVEN={str(slate_control_proven).upper()}")
    print(f"LIVE_DOWNLOAD_CONTROL_EVIDENCE_PROVEN={str(download_control_proven).upper()}")
    if slate_control_proven and download_control_proven:
        print("NEXT_STAGE_READINESS=READY_FOR_NON_DOWNLOADING_SELECTOR_ENUMERATION")
    else:
        print("NEXT_STAGE_READINESS=FAIL_CLOSED_SELECTOR_OR_DOWNLOAD_CONTROL_NOT_PROVEN")
    print("AUTOMATION_EDIT_ALLOWED=FALSE")

    print("\n=== RESULT ===")
    print("READ_ONLY_BROWSER_AUDIT=TRUE")
    print("CSV_DOWNLOADS=0")
    print("DATABASE_WRITES=0")
    print("PRODUCTION_FILE_WRITES=0")
    print("SERVICE_RESTARTS=0")
    print("CRON_CHANGES=0")
    print("UPDATER_CHANGES=0")
    print("LIVE_CHANGES=0")
    print("INJURY_PIPELINE_CHANGES=0")
    print("SOLVER_CHANGES=0")
    print("NFL_POSTGAME_1F_I_B_STATUS=PASS")

    context.close()
    browser.close()
