from pathlib import Path
import hashlib
import re
import sqlite3

ROOT = Path("/home/mwynn/nfl_data_engine")

FILES = {
    "schedule": ROOT / "schedule.py",
    "live_core_builder": ROOT / "scripts" / "build_forecast_live_core_v1.py",
    "live_core_inference": ROOT / "scripts" / "run_forecast_live_core_v1.py",
    "snapshot_writer": ROOT / "scripts" / "capture_forecast_snapshot_v1.py",
}

ARTIFACTS = {
    "live_core_csv": ROOT / "processed" / "forecast_live_core_v1.csv",
    "live_core_audit": ROOT / "processed" / "forecast_live_core_v1_audit.json",
    "prediction_csv": ROOT / "processed" / "forecast_live_core_v1_predictions.csv",
    "prediction_audit": ROOT / "processed" / "forecast_live_core_v1_predictions_audit.json",
}

NFL_DB = ROOT / "data" / "nfl.db"
LEDGER_DB = ROOT / "data" / "forecast_ledger.db"

EXPECTED_WRITER_SHA = (
    "81f7612bfc55c50b6f3f46599ff4f5c0"
    "f27f1b3badfae2cb0057e0217433d552"
)

EXPECTED_PREDICTION_BASELINE_SHA = (
    "8a58309ca2b71ab7d019a5697580eb7c"
    "3114716f94e8b4c5c5864ea7b40bc94c"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fail(msg: str):
    raise SystemExit(f"FAIL | {msg}")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def ro_connect(path: Path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def print_matches(label, text, patterns):
    print()
    print(f"--- {label} ---")

    lines = text.splitlines()
    hits = []

    for lineno, line in enumerate(lines, start=1):
        lowered = line.lower()

        if any(re.search(pattern, lowered) for pattern in patterns):
            hits.append((lineno, line.rstrip()))

    if not hits:
        print("NO MATCHES")
        return

    for lineno, line in hits[:200]:
        print(f"{lineno:5d}: {line}")

    if len(hits) > 200:
        print(f"... truncated {len(hits) - 200} additional matches")


print("=" * 100)
print("WFS FORECAST CENTER — LIVE FORECAST REFRESH V1 CONTRACT AUDIT")
print("MODE | READ ONLY")
print("=" * 100)


# =====================================================================
# Required source files
# =====================================================================

print()
print("=== REQUIRED SOURCE FILES ===")

for label, path in FILES.items():
    if not path.is_file():
        fail(f"missing {label}: {path}")

    print(f"PASS | {label:<20} | {path}")
    print(f"       SHA256             | {sha256(path)}")


writer_sha = sha256(FILES["snapshot_writer"])

if writer_sha != EXPECTED_WRITER_SHA:
    fail(
        "active snapshot writer no longer matches frozen V1 baseline"
    )

print()
print("PASS | active snapshot writer matches frozen V1 hash")


# =====================================================================
# Existing artifacts
# =====================================================================

print()
print("=== CURRENT LIVE ARTIFACTS ===")

for label, path in ARTIFACTS.items():
    if path.is_file():
        print(f"{label:<22} | EXISTS")
        print(f"  path   | {path}")
        print(f"  sha256 | {sha256(path)}")
        print(f"  bytes  | {path.stat().st_size}")
    else:
        print(f"{label:<22} | MISSING")
        print(f"  path   | {path}")


if ARTIFACTS["prediction_csv"].is_file():
    pred_sha = sha256(ARTIFACTS["prediction_csv"])

    print()
    print(
        "Prediction matches frozen prospective baseline | "
        f"{pred_sha == EXPECTED_PREDICTION_BASELINE_SHA}"
    )


# =====================================================================
# Database read-only state
# =====================================================================

print()
print("=== DATABASE READ-ONLY STATE ===")

for label, path in [
    ("nfl.db", NFL_DB),
    ("forecast_ledger.db", LEDGER_DB),
]:
    if not path.is_file():
        fail(f"missing database: {path}")

    conn = ro_connect(path)

    try:
        integrity = conn.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        fk = conn.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()

    finally:
        conn.close()

    print(f"{label:<20} | integrity={integrity} | fk_rows={len(fk)}")

    if integrity != "ok":
        fail(f"{label} integrity")

    if fk:
        fail(f"{label} foreign-key integrity")


# =====================================================================
# Inspect schedule refresh source
# =====================================================================

schedule_text = read_text(
    FILES["schedule"]
)

print()
print("=== SCHEDULE REFRESH CONTRACT ===")

print_matches(
    "schedule.py relevant lines",
    schedule_text,
    [
        r"def download_schedule",
        r"def update_schedule_database",
        r"load_schedules",
        r"to_pandas",
        r"sqlite3",
        r"connect",
        r"insert into",
        r"update ",
        r"on conflict",
        r"spread_line",
        r"total_line",
        r"moneyline",
        r"gametime",
        r"if __name__",
        r"argparse",
        r"sys\.argv",
        r"main\(",
    ],
)


# =====================================================================
# Inspect Live CORE builder
# =====================================================================

builder_text = read_text(
    FILES["live_core_builder"]
)

print()
print("=== LIVE CORE BUILDER CONTRACT ===")

print_matches(
    "build_forecast_live_core_v1.py relevant lines",
    builder_text,
    [
        r"if __name__",
        r"main\(",
        r"argparse",
        r"read_sql",
        r"sqlite",
        r"mode=ro",
        r"nfl\.db",
        r"forecast_live_core_v1",
        r"to_csv",
        r"json",
        r"market",
        r"completed",
        r"2026",
        r"sha256",
        r"write",
    ],
)


# =====================================================================
# Inspect inference runner
# =====================================================================

inference_text = read_text(
    FILES["live_core_inference"]
)

print()
print("=== LIVE CORE INFERENCE CONTRACT ===")

print_matches(
    "run_forecast_live_core_v1.py relevant lines",
    inference_text,
    [
        r"if __name__",
        r"main\(",
        r"argparse",
        r"forecast_live_core_v1",
        r"predictions",
        r"to_csv",
        r"json",
        r"sha256",
        r"historical",
        r"reconstruct",
        r"2025",
        r"76",
        r"market",
        r"ready",
        r"pending",
    ],
)


# =====================================================================
# Inspect capture writer assumptions
# =====================================================================

writer_text = read_text(
    FILES["snapshot_writer"]
)

print()
print("=== SNAPSHOT WRITER INPUT CONTRACT ===")

print_matches(
    "capture_forecast_snapshot_v1.py relevant lines",
    writer_text,
    [
        r"predictions =",
        r"predictions_audit",
        r"duplicate",
        r"source_prediction_sha256",
        r"source_audit_sha256",
        r"america/new_york",
        r"pre_kickoff",
        r"begin immediate",
        r"backup",
        r"forecast_ledger",
        r"systemexit",
    ],
)


# =====================================================================
# Main-entrypoint summary
# =====================================================================

print()
print("=== EXECUTION ENTRYPOINT SUMMARY ===")

for label, path in FILES.items():

    text = read_text(path)

    has_main_guard = (
        'if __name__ == "__main__"' in text
        or "if __name__ == '__main__'" in text
    )

    main_defs = re.findall(
        r"^\s*def\s+(main|run|execute)\s*\(",
        text,
        flags=re.MULTILINE,
    )

    argparse_used = "argparse" in text

    print(f"{label:<20}")
    print(f"  __main__ guard | {has_main_guard}")
    print(f"  main/run funcs | {main_defs}")
    print(f"  argparse       | {argparse_used}")


# =====================================================================
# Candidate project-level refresh files
# =====================================================================

print()
print("=== CANDIDATE REFRESH / UPDATE ENTRYPOINT FILES ===")

patterns = [
    "*schedule*.py",
    "*refresh*.py",
    "*update*.py",
    "*market*.py",
    "*forecast*.py",
]

seen = set()

for pattern in patterns:

    for path in sorted(ROOT.glob(pattern)) + sorted((ROOT / "scripts").glob(pattern)):

        resolved = path.resolve()

        if resolved in seen:
            continue

        seen.add(resolved)

        if (
            "/backups/" in str(resolved)
            or "/venv/" in str(resolved)
        ):
            continue

        print(resolved)


# =====================================================================
# Final gate
# =====================================================================

print()
print("=" * 100)
print("LIVE FORECAST REFRESH V1 — CONTRACT AUDIT GATE")
print("=" * 100)
print("PASS | required forecast source files present")
print("PASS | frozen snapshot writer hash preserved")
print("PASS | current live artifacts inventoried")
print("PASS | nfl.db opened READ ONLY")
print("PASS | forecast_ledger.db opened READ ONLY")
print("PASS | schedule refresh implementation inspected")
print("PASS | Live CORE builder implementation inspected")
print("PASS | Live CORE inference implementation inspected")
print("PASS | snapshot writer input contract inspected")
print("PASS | candidate refresh entrypoints inventoried")
print()
print("HOLD | NO REFRESH EXECUTED")
print("HOLD | NO FORECAST ARTIFACT REBUILT")
print("HOLD | NO LEDGER APPEND")
print("PASS | NO DATABASE WRITES")
print("PASS | OPTIMIZER / UI UNTOUCHED")
print("PASS | SITE RESTART NOT REQUIRED")
print()
print("NEXT GATE | LOCK EXACT MANUAL REFRESH COMMAND SEQUENCE")
print("=" * 100)
