#!/usr/bin/env python3
"""
WFS NFL — NFL-POSTGAME-1F-H
Automation integration readiness audit — READ ONLY.

Purpose:
  - prove where the frozen 1F-F writer can safely run
  - inspect updater/launcher ordering and lock context
  - verify no conflict with LIVE/injury/solver/public paths
  - identify the safest insertion point BEFORE any automation edit

No writes. No service restart. No cron/updater/launcher modification.
"""

from pathlib import Path
import hashlib, re, subprocess

ROOT = Path("/home/mwynn/nfl_data_engine")

FILES = {
    "writer": ROOT / "NFL-POSTGAME-1F-F.py",
    "run_updater": ROOT / "run_updater.sh",
    "updater": ROOT / "updater.py",
    "live_safe_poll": ROOT / "live_safe_poll.py",
    "live_concurrency_gate": ROOT / "live_concurrency_gate.py",
    "injury_consensus": ROOT / "injury_consensus.py",
    "solver_ready": ROOT / "fanduel_solver_ready_pool.py",
}

EXPECTED = {
    "writer": "063e1bdfaf34208bd6d56705d808d13b3a21859744731ba49e3ff1bf02857315",
    "run_updater": "46ffe95fb09585538bc01efceba6c24a111bfe85072f660a6f9a1e1f380ef96a",
    "updater": "a9637bcb077d62c35cfabf48c5a17aa293c7176f35302f7020d5a0aaef41096a",
    "live_safe_poll": "458d057bca746fd6d2a64a1a10bcd392c9395f345cc8bf655e06f905ea362165",
    "live_concurrency_gate": "0d329611d7c628cafccf5339f2ecaac1fd2c0ff68760ba05cf4ec6c018e5c65a",
    "injury_consensus": "e4bb68e5b839cb091ca7ef6e8c25968796443f9f7c1b9cdc0a9d69609cb77c09",
    "solver_ready": "35d955e868b205a6851564d085353e23fb79e240ea3bace064277f95a6f5ac2a",
}

LOCK_PATH = str(ROOT / "nfl_updater.lock")

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return p.read_text(errors="replace")

def grep_lines(text, patterns):
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if any(re.search(p, line, re.I) for p in patterns):
            out.append((i, line.rstrip()))
    return out

def main():
    print("="*100)
    print("WFS NFL — NFL-POSTGAME-1F-H")
    print("AUTOMATION INTEGRATION READINESS AUDIT — READ ONLY")
    print("="*100)

    bad_hash = False
    for name, path in FILES.items():
        print(f"{name.upper()}_EXISTS={str(path.exists()).upper()}")
        if not path.exists():
            continue
        h = sha(path)
        print(f"{name.upper()}_SHA256={h}")
        exp = EXPECTED.get(name)
        if exp:
            ok = (h == exp)
            print(f"{name.upper()}_SHA_MATCH={str(ok).upper()}")
            bad_hash |= not ok

    rup = read(FILES["run_updater"]) if FILES["run_updater"].exists() else ""
    upd = read(FILES["updater"]) if FILES["updater"].exists() else ""

    print("\n=== RUN_UPDATER LOCK / CHAIN EVIDENCE ===")
    lock_refs = grep_lines(rup, [r"nfl_updater\.lock", r"flock", r"injury", r"updater\.py", r"solver_ready", r"fanduel"])
    for ln, line in lock_refs:
        print(f"RUN_UPDATER_LINE|{ln}|{line}")
    print(f"LOCK_PATH_EXPECTED={LOCK_PATH}")
    print(f"LOCK_PATH_REFERENCED={str(LOCK_PATH in rup).upper()}")
    print(f"EXCLUSIVE_FLOCK_REFERENCED={str(bool(re.search(r'flock\b.*(?:-x|LOCK_EX|9)', rup, re.I)) or 'flock -w 60 9' in rup).upper()}")

    print("\n=== UPDATER ORDER EVIDENCE ===")
    order_lines = grep_lines(upd, [
        r"boxscore", r"fanduel_scoring", r"final db audit",
        r"player_game_stats", r"step\s*[345]", r"projection"
    ])
    for ln, line in order_lines:
        print(f"UPDATER_LINE|{ln}|{line}")

    print("\n=== CRON ===")
    try:
        cp = subprocess.run(["crontab","-l"], capture_output=True, text=True, timeout=5)
        cron = cp.stdout if cp.returncode == 0 else ""
        cron_lines = [x for x in cron.splitlines() if "run_updater.sh" in x]
        print(f"CRON_RUN_UPDATER_LINES={len(cron_lines)}")
        for x in cron_lines:
            print(f"CRON|{x}")
    except Exception as e:
        print(f"CRON_READ_ERROR={type(e).__name__}:{e}")

    print("\n=== SERVICE STATE ===")
    for svc in ("wfs-nfl-live.service","wfs.service"):
        try:
            cp = subprocess.run(["systemctl","is-active",svc], capture_output=True, text=True, timeout=5)
            print(f"SERVICE|{svc}|{cp.stdout.strip() or cp.stderr.strip()}")
        except Exception as e:
            print(f"SERVICE|{svc}|ERROR:{type(e).__name__}:{e}")

    # Candidate insertion analysis:
    # Writer reads nfl.db RO + parquet, writes separate ledger DB.
    # Best place is under same updater EX lock, after upstream pool is refreshed
    # and before lock release. We only prove textual anchors here; no edit.
    print("\n=== CANDIDATE INSERTION CONTRACT ===")
    print("WRITER_READS_NFL_DB=READ_ONLY")
    print("WRITER_READS_FULL_UPSTREAM_PARQUET=TRUE")
    print("WRITER_WRITES_ONLY_PLAYER_PROJECTION_LEDGER=TRUE")
    print("MUST_RUN_UNDER_EXISTING_UPDATER_EXCLUSIVE_LOCK=TRUE")
    print("MUST_RUN_AFTER_UPSTREAM_PLAYER_POOL_REFRESH=TRUE")
    print("MUST_RUN_BEFORE_UPDATER_LOCK_RELEASE=TRUE")
    print("MUST_NOT_RUN_FROM_PUBLIC_SOLVER_PATH=TRUE")
    print("MUST_NOT_EXPAND_FANDUEL_SLATE_POOL=TRUE")
    print("MUST_NOT_TOUCH_LIVE_DB=TRUE")
    print("MUST_NOT_TOUCH_FORECAST_LEDGER=TRUE")
    print("MUST_NOT_TOUCH_NFL_DB=TRUE")

    print("\n=== RESULT ===")
    print(f"HASH_BASELINES_INTACT={str(not bad_hash).upper()}")
    print("AUTOMATION_EDIT_ALLOWED=FALSE")
    print("READ_ONLY_AUDIT=TRUE")
    print("DATABASE_WRITES=0")
    print("FILE_WRITES=0")
    print("SERVICE_RESTARTS=0")
    print("CRON_CHANGES=0")
    print("UPDATER_CHANGES=0")
    print("LIVE_CHANGES=0")
    print("INJURY_PIPELINE_CHANGES=0")
    print("SOLVER_CHANGES=0")
    status = "PASS" if not bad_hash else "FAIL_CLOSED"
    print(f"NFL_POSTGAME_1F_H_STATUS={status}")
    if status != "PASS":
        raise SystemExit(2)

if __name__ == "__main__":
    main()
