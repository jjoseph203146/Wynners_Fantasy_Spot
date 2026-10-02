from playwright.sync_api import sync_playwright

URL = "https://www.fanduel.com/research/nfl/fantasy/dfs-projections"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)

    page = browser.new_page(
        viewport={"width": 1600, "height": 1200}
    )

    print("OPENING:", URL, flush=True)

    page.goto(
        URL,
        wait_until="domcontentloaded",
        timeout=120000,
    )

    page.wait_for_timeout(8000)

    button = (
        page.locator("button")
        .filter(has_text="SLATE")
        .first
    )

    button.wait_for(
        state="visible",
        timeout=60000,
    )

    print()
    print("CURRENT SLATE BUTTON:")
    print(repr(button.inner_text()))

    button.click()
    page.wait_for_timeout(1500)

    print()
    print("=" * 80)
    print("VISIBLE TEXT ELEMENTS CONTAINING KNOWN SLATE WORDS")
    print("=" * 80)

    keywords = (
        "Main",
        "Only",
        "Sun",
        "Mon",
        "SNF",
        "MNF",
        "TNF",
    )

    seen = set()

    elements = page.locator("body *")

    for i in range(elements.count()):
        el = elements.nth(i)

        try:
            if not el.is_visible():
                continue

            text = el.inner_text().strip()

            if not text:
                continue

            if len(text) > 250:
                continue

            if not any(k.lower() in text.lower() for k in keywords):
                continue

            key = (el.evaluate("(e) => e.tagName"), text)

            if key in seen:
                continue

            seen.add(key)

            print()
            print("TAG:", key[0])
            print("TEXT:", repr(text))

            try:
                print(
                    "HTML:",
                    el.evaluate(
                        "(e) => e.outerHTML"
                    )[:1200]
                )
            except Exception:
                pass

        except Exception:
            continue

    print()
    print("=" * 80)
    print("EXACT MAIN MATCHES")
    print("=" * 80)

    mains = page.get_by_text(
        "Main",
        exact=True,
    )

    print("COUNT:", mains.count())

    for i in range(mains.count()):
        el = mains.nth(i)

        try:
            print()
            print("MAIN INDEX:", i)
            print("VISIBLE:", el.is_visible())
            print("TEXT:", repr(el.inner_text()))

            print(
                "OUTER HTML:",
                el.evaluate(
                    "(e) => e.outerHTML"
                )[:2000]
            )

            print(
                "PARENT HTML:",
                el.evaluate(
                    "(e) => e.parentElement.outerHTML"
                )[:4000]
            )

        except Exception as exc:
            print("ERROR:", repr(exc))

    browser.close()
