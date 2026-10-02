#!/usr/bin/env python3
"""
WFS NFL — NFL-POSTGAME-1F-H-R5
Exact FanDuel source-path + production-projection refresh audit — READ ONLY.

Purpose:
  1) Prove the exact directories used by fanduel_player_pool.py for QB/RB/WR/TE/D-ST.
  2) Locate current authoritative positional files without fuzzy substitution.
  3) Prove whether production_projection.py is automatically refreshed.
  4) Determine whether a safe refresh chain can be built without guessing.

No writes. No service restart. No cron/updater/launcher changes.
"""

from pathlib import Path
import ast, hashlib, re, subprocess
from datetime import datetime, timezone

ROOT = Path("/home/mwynn/nfl_data_engine")
PRODUCER = ROOT / "fanduel_player_pool.py"
PROJECTION = ROOT / "production_projection.py"
TARGET_PROJ = ROOT / "data/parquet/nfl_production_projection.parquet"

EXPECTED_PRODUCER_SHA = "b9e29f9278d7160eaec1285236aafdf091a24447aadbc7aa5c3c123f00c9f4b0"

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def txt(p):
    return p.read_text(errors="replace")

def mtime(p):
    return datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()

def grep(path, pats):
    out=[]
    t=txt(path)
    for i,line in enumerate(t.splitlines(),1):
        if any(re.search(p,line,re.I) for p in pats):
            out.append((i,line.rstrip()))
    return out

def crontab():
    try:
        cp=subprocess.run(["crontab","-l"],capture_output=True,text=True,timeout=5)
        return cp.stdout if cp.returncode==0 else ""
    except Exception:
        return ""

def main():
    print("="*110)
    print("WFS NFL — NFL-POSTGAME-1F-H-R5")
    print("EXACT FANDUEL SOURCE-PATH + PROJECTION REFRESH AUDIT — READ ONLY")
    print("="*110)

    if not PRODUCER.exists():
        raise RuntimeError("missing fanduel_player_pool.py")
    ph=sha(PRODUCER)
    print(f"PRODUCER_SHA256={ph}")
    print(f"PRODUCER_SHA_MATCH={str(ph==EXPECTED_PRODUCER_SHA).upper()}")
    if ph != EXPECTED_PRODUCER_SHA:
        raise RuntimeError("producer hash mismatch")

    print("\n=== PRODUCER PATH-CONSTRUCTION EVIDENCE ===")
    for ln,line in grep(PRODUCER,[
        r"POSITION_FILES", r"FANDUEL", r"CSV_DIR", r"Path\(",
        r"read_csv", r"load.*file", r"for .*POSITION", r"/ filename",
        r"PROJECTION_PARQUET"
    ]):
        print(f"PRODUCER_LINE|{ln}|{line}")

    print("\n=== EXACT POSITIONAL FILE LOCATIONS ===")
    wanted=["QB.csv","RB.csv","WR.csv","TE.csv","D-ST.csv"]
    found={}
    for name in wanted:
        paths=sorted(ROOT.glob(f"**/{name}"))
        found[name]=paths
        print(f"FILE|{name}|MATCHES={len(paths)}")
        for p in paths:
            print(f"FILE_MATCH|{name}|{p}|MTIME_UTC={mtime(p)}|SIZE={p.stat().st_size}|SHA256={sha(p)}")

    ambiguous=sum(1 for v in found.values() if len(v)>1)
    missing=sum(1 for v in found.values() if len(v)==0)
    unique=sum(1 for v in found.values() if len(v)==1)
    print(f"POSITION_FILES_UNIQUE={unique}")
    print(f"POSITION_FILES_MISSING={missing}")
    print(f"POSITION_FILES_AMBIGUOUS={ambiguous}")

    print("\n=== PRODUCTION PROJECTION STATE ===")
    print(f"PRODUCTION_SCRIPT_EXISTS={str(PROJECTION.exists()).upper()}")
    if PROJECTION.exists():
        print(f"PRODUCTION_SCRIPT_SHA256={sha(PROJECTION)}")
        for ln,line in grep(PROJECTION,[
            r"nfl_production_projection\.parquet", r"to_parquet",
            r"read_parquet", r"sqlite3", r"DATABASE", r"main\("
        ]):
            print(f"PROJECTION_LINE|{ln}|{line}")
    print(f"PROJECTION_TARGET_EXISTS={str(TARGET_PROJ.exists()).upper()}")
    if TARGET_PROJ.exists():
        print(f"PROJECTION_TARGET_MTIME_UTC={mtime(TARGET_PROJ)}")
        print(f"PROJECTION_TARGET_SHA256={sha(TARGET_PROJ)}")

    print("\n=== AUTOMATIC PROJECTION REFRESH TRACE ===")
    cron=crontab()
    cron_refs=0
    for line in cron.splitlines():
        if "production_projection" in line:
            cron_refs+=1
            print(f"CRON_PROJECTION_REF|{line}")
    print(f"CRON_PROJECTION_REFERENCES={cron_refs}")

    systemd_refs=0
    for base in (Path("/etc/systemd/system"),Path("/lib/systemd/system")):
        if not base.exists():
            continue
        for pat in ("*.service","*.timer"):
            for p in base.glob(pat):
                try:
                    t=txt(p)
                except Exception:
                    continue
                if "production_projection" in t:
                    systemd_refs+=1
                    for i,line in enumerate(t.splitlines(),1):
                        if "production_projection" in line:
                            print(f"SYSTEMD_PROJECTION_REF|{p}|{i}|{line.rstrip()}")
    print(f"SYSTEMD_PROJECTION_REFERENCES={systemd_refs}")

    local_refs=[]
    for pat in ("*.py","*.sh"):
        for p in sorted(ROOT.glob(pat)):
            if p == PROJECTION:
                continue
            try:
                t=txt(p)
            except Exception:
                continue
            if "production_projection.py" in t or re.search(r"\bproduction_projection\b",t):
                for i,line in enumerate(t.splitlines(),1):
                    if "production_projection.py" in line or re.search(r"\bproduction_projection\b",line):
                        local_refs.append((p.name,i,line.rstrip()))
                        print(f"LOCAL_PROJECTION_REF|{p.name}|{i}|{line.rstrip()}")
    print(f"LOCAL_PROJECTION_REFERENCE_LINES={len(local_refs)}")

    auto_proj = cron_refs>0 or systemd_refs>0
    print(f"AUTOMATIC_PRODUCTION_PROJECTION_REFRESH_PROVEN={str(auto_proj).upper()}")

    print("\n=== MTIME ORDER ===")
    if TARGET_PROJ.exists():
        for name,paths in found.items():
            for p in paths:
                rel = "NEWER" if p.stat().st_mtime > TARGET_PROJ.stat().st_mtime else "OLDER_OR_EQUAL"
                print(f"POSITION_VS_PROJECTION|{name}|{rel}")
    pool=ROOT/"data/parquet/nfl_fanduel_player_pool.parquet"
    if pool.exists() and TARGET_PROJ.exists():
        print(f"PROJECTION_VS_POOL={'NEWER' if TARGET_PROJ.stat().st_mtime > pool.stat().st_mtime else 'OLDER_OR_EQUAL'}")

    print("\n=== DECISION ===")
    if missing or ambiguous:
        print("SOURCE_PATH_READINESS=FAIL_CLOSED_POSITION_FILE_SET_NOT_UNIQUE")
    else:
        print("SOURCE_PATH_READINESS=POSITION_FILE_SET_UNIQUE")

    if auto_proj:
        print("PROJECTION_REFRESH_READINESS=AUTOMATIC_PATH_PROVEN")
    else:
        print("PROJECTION_REFRESH_READINESS=FAIL_CLOSED_NO_AUTOMATIC_PATH_PROVEN")

    print("SAFE_CHAIN_REQUIREMENT=authoritative position inputs current -> production projection current -> fanduel_player_pool.py -> verify full pool -> NFL-POSTGAME-1F-F.py")
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
    print("NFL_POSTGAME_1F_H_R5_STATUS=PASS")

if __name__ == "__main__":
    main()
