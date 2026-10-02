#!/usr/bin/env python3
from pathlib import Path
import ast, hashlib, re, subprocess
from datetime import datetime, timezone

ROOT = Path("/home/mwynn/nfl_data_engine")
PRODUCER = ROOT / "fanduel_player_pool.py"
TARGET = ROOT / "data/parquet/nfl_fanduel_player_pool.parquet"

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def txt(p):
    return p.read_text(errors="replace")

def mtime_utc(p):
    return datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()

def grep_file(path, pats):
    out=[]
    try:
        t=txt(path)
    except Exception:
        return out
    for i,l in enumerate(t.splitlines(),1):
        if any(re.search(p,l,re.I) for p in pats):
            out.append((i,l.rstrip()))
    return out

def python_imports(path):
    out=set()
    try:
        tree=ast.parse(txt(path))
    except Exception:
        return out
    for n in ast.walk(tree):
        if isinstance(n,ast.Import):
            for x in n.names:
                out.add(x.name.split(".")[0])
        elif isinstance(n,ast.ImportFrom) and n.module:
            out.add(n.module.split(".")[0])
    return out

def main():
    print("="*108)
    print("WFS NFL — NFL-POSTGAME-1F-H-R3")
    print("EXACT PRODUCER INVOCATION / REFRESH-MECHANISM AUDIT — READ ONLY")
    print("="*108)

    if not PRODUCER.exists():
        raise RuntimeError(f"missing producer: {PRODUCER}")
    if not TARGET.exists():
        raise RuntimeError(f"missing target parquet: {TARGET}")

    print(f"PRODUCER_SHA256={sha(PRODUCER)}")
    print(f"TARGET_SHA256={sha(TARGET)}")
    print(f"TARGET_MTIME_UTC={mtime_utc(TARGET)}")
    print(f"TARGET_SIZE_BYTES={TARGET.stat().st_size}")

    print("\n=== LOCAL REFERENCES TO PRODUCER ===")
    refs=[]
    patterns=[r"\bfanduel_player_pool\.py\b", r"\bfanduel_player_pool\b"]
    for pat in ("*.py","*.sh","*.service","*.timer"):
        for p in sorted(ROOT.glob(pat)):
            if p == PRODUCER:
                continue
            for ln,line in grep_file(p,patterns):
                refs.append((p.name,ln,line))
                print(f"LOCAL_REF|{p.name}|{ln}|{line}")
    print(f"LOCAL_REFERENCE_COUNT={len(refs)}")

    print("\n=== CRON REFERENCES ===")
    cron_auto=False
    try:
        cp=subprocess.run(["crontab","-l"],capture_output=True,text=True,timeout=5)
        cron=cp.stdout if cp.returncode==0 else ""
        found=0
        for line in cron.splitlines():
            if "fanduel_player_pool" in line:
                found+=1
                print(f"CRON_REF|{line}")
        cron_auto = found > 0
        print(f"CRON_PRODUCER_REFERENCES={found}")
    except Exception as e:
        print(f"CRON_READ_ERROR={type(e).__name__}:{e}")
        print("CRON_PRODUCER_REFERENCES=UNKNOWN")

    print("\n=== SYSTEMD REFERENCES ===")
    systemd_hits=0
    for base in (Path("/etc/systemd/system"), Path("/lib/systemd/system")):
        if not base.exists():
            continue
        for pat in ("*.service","*.timer"):
            for p in base.glob(pat):
                try:
                    t=txt(p)
                except Exception:
                    continue
                if "fanduel_player_pool" in t:
                    systemd_hits+=1
                    for ln,line in grep_file(p,[r"fanduel_player_pool"]):
                        print(f"SYSTEMD_REF|{p}|{ln}|{line}")
    print(f"SYSTEMD_PRODUCER_REFERENCES={systemd_hits}")

    print("\n=== PRODUCER INPUT / OUTPUT EVIDENCE ===")
    producer_text=txt(PRODUCER)
    interesting=grep_file(PRODUCER,[
        r"read_csv", r"read_excel", r"read_parquet", r"sqlite3", r"connect\(",
        r"glob\(", r"INPUT", r"OUTPUT", r"PARQUET", r"CSV",
        r"to_parquet", r"to_csv", r"main\("
    ])
    for ln,line in interesting:
        print(f"PRODUCER_LINE|{ln}|{line}")

    print("\n=== FILE PATH CANDIDATES FROM PRODUCER ===")
    candidates=set()
    pattern = re.compile(r"[\"']([^\"']+\.(?:csv|xlsx|xls|parquet|db|json))[\"']", re.I)
    for m in pattern.finditer(producer_text):
        candidates.add(m.group(1))
    for s in sorted(candidates):
        p=Path(s)
        exists=False
        resolved=None
        if p.is_absolute():
            exists=p.exists()
            resolved=p
        else:
            for cand in (ROOT/p, ROOT/"data"/p):
                if cand.exists():
                    exists=True
                    resolved=cand
                    break
        extra=f"|PATH={resolved}" if exists else ""
        print(f"INPUT_LITERAL|{s}|EXISTS={str(exists).upper()}{extra}")

    print("\n=== PRODUCER IMPORTERS ===")
    importers=[]
    for p in sorted(ROOT.glob("*.py")):
        if p == PRODUCER:
            continue
        if "fanduel_player_pool" in python_imports(p):
            importers.append(p.name)
            print(f"PYTHON_IMPORTER={p.name}")
    print(f"PYTHON_IMPORTER_COUNT={len(importers)}")

    auto = cron_auto or systemd_hits > 0

    print("\n=== EXECUTION PATH DECISION ===")
    print(f"AUTOMATIC_PRODUCER_REFRESH_PATH_PROVEN={str(auto).upper()}")
    if auto:
        print("INSERTION_READINESS=TRACE_AUTOMATIC_PRODUCER_ORDER_BEFORE_PATCH")
    else:
        print("INSERTION_READINESS=FAIL_CLOSED_NO_AUTOMATIC_FULL_POOL_REFRESH_PATH")
        print("ARCHITECTURE_GAP=hourly postgame capture cannot rely on a parquet last refreshed manually")
        print("NEXT_REQUIRED_ACTION=design a safe producer-refresh stage under the existing updater EX lock, then capture immediately after it")

    print("\n=== RESULT ===")
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
    print("NFL_POSTGAME_1F_H_R3_STATUS=PASS")

if __name__ == "__main__":
    main()
