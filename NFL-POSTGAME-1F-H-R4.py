#!/usr/bin/env python3
from pathlib import Path
import ast
import hashlib
import re
import subprocess
from datetime import datetime, timezone

ROOT = Path("/home/mwynn/nfl_data_engine")
PRODUCER = ROOT / "fanduel_player_pool.py"
CONFIG = ROOT / "config.py"

EXPECTED_PRODUCER_SHA = "b9e29f9278d7160eaec1285236aafdf091a24447aadbc7aa5c3c123f00c9f4b0"

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def txt(p):
    return p.read_text(errors="replace")

def mtime_utc(p):
    return datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()

def literal_assignments(path):
    vals = {}
    tree = ast.parse(txt(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            try:
                vals[name] = ast.literal_eval(node.value)
            except Exception:
                pass
    return vals

def grep_paths(root, needles):
    hits = []
    for pattern in ("*.py", "*.sh"):
        for p in sorted(root.glob(pattern)):
            try:
                t = txt(p)
            except Exception:
                continue
            for i, line in enumerate(t.splitlines(), 1):
                if any(n in line for n in needles):
                    hits.append((p.name, i, line.rstrip()))
    return hits

def cron_text():
    try:
        cp = subprocess.run(["crontab", "-l"], capture_output=True, text=True, timeout=5)
        return cp.stdout if cp.returncode == 0 else ""
    except Exception:
        return ""

def main():
    print("=" * 108)
    print("WFS NFL — NFL-POSTGAME-1F-H-R4")
    print("FULL-POOL PRODUCER INPUT LINEAGE / FRESHNESS AUDIT — READ ONLY")
    print("=" * 108)

    if not PRODUCER.exists():
        raise RuntimeError(f"missing producer: {PRODUCER}")
    if not CONFIG.exists():
        raise RuntimeError(f"missing config: {CONFIG}")

    ph = sha(PRODUCER)
    print(f"PRODUCER_SHA256={ph}")
    print(f"PRODUCER_SHA_MATCH={str(ph == EXPECTED_PRODUCER_SHA).upper()}")
    if ph != EXPECTED_PRODUCER_SHA:
        raise RuntimeError("producer hash mismatch")

    cfg_text = txt(CONFIG)
    print("\n=== CONFIG PATH EVIDENCE ===")
    for i, line in enumerate(cfg_text.splitlines(), 1):
        if any(k in line for k in ("CSV_DIR", "PARQUET_DIR", "DATABASE_PATH", "DATA_DIR")):
            print(f"CONFIG_LINE|{i}|{line.rstrip()}")

    # Resolve common Path expressions safely by importing config in a subprocess.
    probe = subprocess.run(
        [
            str(ROOT / "venv/bin/python"),
            "-c",
            (
                "import config; "
                "keys=['CSV_DIR','PARQUET_DIR','DATABASE_PATH','DATA_DIR']; "
                "print('\\n'.join(f'{k}={getattr(config,k)}' for k in keys if hasattr(config,k)))"
            ),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )
    if probe.returncode != 0:
        raise RuntimeError("could not resolve config paths: " + probe.stderr.strip())

    resolved = {}
    for line in probe.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            resolved[k.strip()] = Path(v.strip())

    for k, p in resolved.items():
        print(f"RESOLVED_CONFIG|{k}|{p}")

    csv_dir = resolved.get("CSV_DIR")
    parquet_dir = resolved.get("PARQUET_DIR")
    db_path = resolved.get("DATABASE_PATH")
    if not csv_dir or not parquet_dir or not db_path:
        raise RuntimeError("required config paths unresolved")

    print("\n=== PRODUCER DECLARED INPUTS ===")
    pos_files = ["QB.csv", "RB.csv", "WR.csv", "TE.csv", "D-ST.csv"]
    input_paths = [csv_dir / x for x in pos_files]
    projection = parquet_dir / "nfl_production_projection.parquet"
    target = parquet_dir / "nfl_fanduel_player_pool.parquet"

    all_inputs = input_paths + [projection, db_path]
    for p in all_inputs:
        print(
            f"INPUT|{p}|EXISTS={str(p.exists()).upper()}"
            + (f"|MTIME_UTC={mtime_utc(p)}|SIZE={p.stat().st_size}|SHA256={sha(p)}" if p.exists() and p.is_file() else "")
        )

    print("\n=== TARGET OUTPUT STATE ===")
    print(
        f"TARGET|{target}|EXISTS={str(target.exists()).upper()}"
        + (f"|MTIME_UTC={mtime_utc(target)}|SIZE={target.stat().st_size}|SHA256={sha(target)}" if target.exists() else "")
    )

    missing = [str(p) for p in all_inputs if not p.exists()]
    print(f"MISSING_DECLARED_INPUTS={len(missing)}")
    for x in missing:
        print(f"MISSING_INPUT|{x}")

    print("\n=== INPUT PRODUCER REFERENCES ===")
    needles = pos_files + ["nfl_production_projection.parquet"]
    hits = grep_paths(ROOT, needles)
    for name, ln, line in hits:
        print(f"INPUT_REF|{name}|{ln}|{line}")
    print(f"INPUT_REFERENCE_LINES={len(hits)}")

    print("\n=== CRON INPUT-REFRESH REFERENCES ===")
    cron = cron_text()
    cron_hits = 0
    for line in cron.splitlines():
        if any(n in line for n in needles):
            cron_hits += 1
            print(f"CRON_INPUT_REF|{line}")
    print(f"CRON_INPUT_REFRESH_REFERENCES={cron_hits}")

    print("\n=== SYSTEMD INPUT-REFRESH REFERENCES ===")
    systemd_hits = 0
    for base in (Path("/etc/systemd/system"), Path("/lib/systemd/system")):
        if not base.exists():
            continue
        for pat in ("*.service", "*.timer"):
            for p in base.glob(pat):
                try:
                    t = txt(p)
                except Exception:
                    continue
                if any(n in t for n in needles):
                    systemd_hits += 1
                    for i, line in enumerate(t.splitlines(), 1):
                        if any(n in line for n in needles):
                            print(f"SYSTEMD_INPUT_REF|{p}|{i}|{line.rstrip()}")
    print(f"SYSTEMD_INPUT_REFRESH_REFERENCES={systemd_hits}")

    print("\n=== FRESHNESS RELATIONSHIP ===")
    existing_file_inputs = [p for p in input_paths + [projection] if p.exists()]
    if target.exists() and existing_file_inputs:
        newest = max(existing_file_inputs, key=lambda p: p.stat().st_mtime)
        oldest = min(existing_file_inputs, key=lambda p: p.stat().st_mtime)
        newer_than_target = [p for p in existing_file_inputs if p.stat().st_mtime > target.stat().st_mtime]
        print(f"NEWEST_INPUT={newest}")
        print(f"NEWEST_INPUT_MTIME_UTC={mtime_utc(newest)}")
        print(f"OLDEST_INPUT={oldest}")
        print(f"OLDEST_INPUT_MTIME_UTC={mtime_utc(oldest)}")
        print(f"INPUTS_NEWER_THAN_TARGET={len(newer_than_target)}")
        for p in sorted(newer_than_target):
            print(f"NEWER_INPUT|{p}|MTIME_UTC={mtime_utc(p)}")
    else:
        print("INPUTS_NEWER_THAN_TARGET=UNKNOWN")

    print("\n=== DECISION ===")
    auto_input_refresh = (cron_hits > 0 or systemd_hits > 0)
    print(f"AUTOMATIC_DECLARED_INPUT_REFRESH_PATH_PROVEN={str(auto_input_refresh).upper()}")
    if missing:
        print("PRODUCER_EXECUTION_READINESS=FAIL_CLOSED_MISSING_INPUTS")
    else:
        print("PRODUCER_EXECUTION_READINESS=INPUTS_PRESENT")
    print("SAFE_ARCHITECTURE_ORDER=refresh authoritative producer inputs -> run fanduel_player_pool.py -> verify target -> run NFL-POSTGAME-1F-F.py")
    print("AUTOMATION_EDIT_ALLOWED=FALSE")

    print("\n=== RESULT ===")
    print("READ_ONLY_AUDIT=TRUE")
    print("DATABASE_WRITES=0")
    print("FILE_WRITES=0")
    print("SERVICE_RESTARTS=0")
    print("CRON_CHANGES=0")
    print("UPDATER_CHANGES=0")
    print("LIVE_CHANGES=0")
    print("INJURY_PIPELINE_CHANGES=0")
    print("SOLVER_CHANGES=0")
    print("NFL_POSTGAME_1F_H_R4_STATUS=PASS")

if __name__ == "__main__":
    main()
