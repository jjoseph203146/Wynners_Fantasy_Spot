from pathlib import Path
import re


ROOT = Path("/home/mwynn/nfl_data_engine")

SEARCH_ROOTS = [
    ROOT,
]

SKIP_DIR_NAMES = {
    "venv",
    ".venv",
    "__pycache__",
    ".git",
    "node_modules",
    "backups",
    "data",
    "processed",
    "raw",
    ".pytest_cache",
}

TEXT_SUFFIXES = {
    ".py",
    ".sql",
    ".sh",
    ".txt",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
}

TERMS = [
    "gametime",
    "game_date",
    "timezone",
    "time_zone",
    "tzinfo",
    "ZoneInfo",
    "America/New_York",
    "US/Eastern",
    "Eastern",
    " ET",
    "schedule",
    "schedules",
    "nflverse",
    "games",
]

HIGH_VALUE_PATTERNS = [
    re.compile(r"\bgametime\b", re.I),
    re.compile(r"\bgame_date\b", re.I),
    re.compile(r"America/New_York", re.I),
    re.compile(r"US/Eastern", re.I),
    re.compile(r"\btimezone\b", re.I),
    re.compile(r"\bZoneInfo\b", re.I),
    re.compile(r"\bnflverse\b", re.I),
    re.compile(r"INSERT\s+INTO\s+games", re.I),
    re.compile(r"UPDATE\s+games", re.I),
    re.compile(r"CREATE\s+TABLE\s+games", re.I),
]


def should_skip(path):
    return any(
        part in SKIP_DIR_NAMES
        for part in path.parts
    )


def candidate_files():
    seen = set()

    for base in SEARCH_ROOTS:
        for path in base.rglob("*"):

            if not path.is_file():
                continue

            if should_skip(path):
                continue

            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue

            resolved = path.resolve()

            if resolved in seen:
                continue

            seen.add(resolved)
            yield path


def read_text(path):
    try:
        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return ""


print("=" * 100)
print("WFS FORECAST CENTER — GAMETIME LINEAGE AUDIT V1")
print("=" * 100)

files = list(candidate_files())

print()
print("=== SOURCE SEARCH SCOPE ===")
print(f"Candidate text files | {len(files)}")
print("PASS | data/, backups/, venv/, processed/ excluded")


# =====================================================================
# Identify files with timing / schedule evidence
# =====================================================================

hits = []

for path in files:

    text = read_text(path)

    if not text:
        continue

    lines = text.splitlines()

    matched_lines = []

    for lineno, line in enumerate(
        lines,
        start=1,
    ):

        if any(
            pattern.search(line)
            for pattern in HIGH_VALUE_PATTERNS
        ):
            matched_lines.append(
                (
                    lineno,
                    line.rstrip(),
                )
            )

    if matched_lines:
        hits.append(
            (
                path,
                matched_lines,
            )
        )


print()
print("=== FILES WITH HIGH-VALUE TIMING EVIDENCE ===")
print(f"Files matched | {len(hits)}")

for path, matched_lines in hits:

    try:
        relative = path.relative_to(ROOT)
    except ValueError:
        relative = path

    print()
    print("-" * 100)
    print(f"FILE | {relative}")
    print("-" * 100)

    for lineno, line in matched_lines[:80]:

        print(
            f"{lineno:>6} | {line[:220]}"
        )

    if len(matched_lines) > 80:
        print(
            f"... {len(matched_lines) - 80} additional matching lines omitted"
        )


# =====================================================================
# Rank files likely responsible for games ingestion
# =====================================================================

print()
print("=" * 100)
print("LIKELY GAME / SCHEDULE INGESTION FILES")
print("=" * 100)

ranked = []

for path in files:

    text = read_text(path)

    lower = text.lower()

    score = 0
    reasons = []

    checks = [
        ("gametime", 5),
        ("game_date", 4),
        ("insert into games", 8),
        ("update games", 7),
        ("nflverse", 6),
        ("schedule", 3),
        ("timezone", 4),
        ("america/new_york", 8),
        ("us/eastern", 8),
    ]

    for needle, weight in checks:

        if needle in lower:
            score += weight
            reasons.append(needle)

    if score > 0:
        ranked.append(
            (
                score,
                path,
                reasons,
            )
        )

ranked.sort(
    key=lambda x: (
        -x[0],
        str(x[1]),
    )
)

for score, path, reasons in ranked[:30]:

    try:
        relative = path.relative_to(ROOT)
    except ValueError:
        relative = path

    print(
        f"{score:>3} | {relative}"
    )

    print(
        "      evidence | "
        + ", ".join(reasons)
    )


# =====================================================================
# Print contextual excerpts around exact "gametime" writes / transforms
# =====================================================================

print()
print("=" * 100)
print("GAMETIME CONTEXT EXCERPTS")
print("=" * 100)

context_count = 0

for path in files:

    text = read_text(path)

    if "gametime" not in text.lower():
        continue

    lines = text.splitlines()

    matching = [
        i
        for i, line in enumerate(lines)
        if "gametime" in line.lower()
    ]

    if not matching:
        continue

    try:
        relative = path.relative_to(ROOT)
    except ValueError:
        relative = path

    for index in matching[:15]:

        context_count += 1

        start = max(
            0,
            index - 6,
        )

        end = min(
            len(lines),
            index + 7,
        )

        print()
        print("-" * 100)
        print(
            f"FILE | {relative} | around line {index + 1}"
        )
        print("-" * 100)

        for i in range(
            start,
            end,
        ):
            marker = (
                ">>>"
                if i == index
                else "   "
            )

            print(
                f"{marker} {i + 1:>6} | {lines[i][:220]}"
            )


print()
print("=" * 100)
print("TIMEZONE EXPLICIT-EVIDENCE SEARCH")
print("=" * 100)

timezone_needles = [
    "America/New_York",
    "US/Eastern",
    "Eastern Time",
    "timezone",
    "ZoneInfo",
    "tz_localize",
    "tz_convert",
    "pytz",
    "UTC",
]

tz_hits = 0

for path in files:

    text = read_text(path)

    lines = text.splitlines()

    matches = []

    for lineno, line in enumerate(
        lines,
        start=1,
    ):

        if any(
            needle.lower()
            in line.lower()
            for needle in timezone_needles
        ):
            matches.append(
                (
                    lineno,
                    line.rstrip(),
                )
            )

    if not matches:
        continue

    tz_hits += len(matches)

    try:
        relative = path.relative_to(ROOT)
    except ValueError:
        relative = path

    print()
    print(f"FILE | {relative}")

    for lineno, line in matches[:40]:
        print(
            f"{lineno:>6} | {line[:220]}"
        )


print()
print("=" * 100)
print("LINEAGE AUDIT SUMMARY")
print("=" * 100)

print(
    f"Source files searched       | {len(files)}"
)

print(
    f"High-value files            | {len(hits)}"
)

print(
    f"Gametime context excerpts   | {context_count}"
)

print(
    f"Timezone evidence lines     | {tz_hits}"
)

print()
print("PASS | READ-ONLY SOURCE AUDIT COMPLETE")
print()
print("NEXT GATE:")
print(
    "PROVE the authoritative meaning of games.gametime "
    "before enabling prospective kickoff filtering."
)
print()
print("PASS | NO DATABASE WRITES")
print("PASS | NO FORECAST WRITES")
print("PASS | LEDGER UNTOUCHED")
print("PASS | OPTIMIZER / UI UNTOUCHED")
print("PASS | SITE RESTART NOT REQUIRED")
print("=" * 100)
