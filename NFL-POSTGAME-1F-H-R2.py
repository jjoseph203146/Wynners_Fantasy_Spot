#!/usr/bin/env python3
"""
WFS NFL — NFL-POSTGAME-1F-H-R2
Exact upstream parquet producer / hourly-chain audit — READ ONLY.

Rationale:
H-R1 proved the updater lock correctly, but its producer heuristic was too broad:
a file that merely references nfl_fanduel_player_pool.parquet and writes some parquet
could be misclassified as the producer. R2 proves the exact writer of the target
artifact and whether that writer is actually executed by the hourly chain.

No writes. No service/cron/updater/launcher changes.
"""

from pathlib import Path
import ast, hashlib, re, subprocess, sys

ROOT = Path("/home/mwynn/nfl_data_engine")
TARGET_NAME = "nfl_fanduel_player_pool.parquet"

RUN_UPDATER = ROOT / "run_updater.sh"
UPDATER = ROOT / "updater.py"
WRITER = ROOT / "NFL-POSTGAME-1F-F.py"

EXPECTED = {
    RUN_UPDATER: "46ffe95fb09585538bc01efceba6c24a111bfe85072f660a6f9a1e1f380ef96a",
    UPDATER: "a9637bcb077d62c35cfabf48c5a17aa293c7176f35302f7020d5a0aaef41096a",
    WRITER: "063e1bdfaf34208bd6d56705d808d13b3a21859744731ba49e3ff1bf02857315",
}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def txt(p):
    return p.read_text(errors="replace")

def lines_matching(t, pats):
    out=[]
    for i,l in enumerate(t.splitlines(),1):
        if any(re.search(p,l,re.I) for p in pats):
            out.append((i,l.rstrip()))
    return out

def python_imports(path):
    names=set()
    try:
        tree=ast.parse(txt(path))
    except Exception:
        return names
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for x in n.names:
                names.add(x.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom) and n.module:
            names.add(n.module.split(".")[0])
    return names

def exact_target_write_evidence(path):
    """
    Conservative textual/AST-assisted evidence:
    require TARGET_NAME in file AND a write sink whose argument is either:
      - a symbol assigned from an expression containing TARGET_NAME
      - a literal expression containing TARGET_NAME
    Supported sinks: DataFrame.to_parquet, pyarrow parquet.write_table,
    shutil/copy not considered producer writes.
    """
    t=txt(path)
    if TARGET_NAME not in t:
        return []

    assigned=set()
    # collect simple assignments where RHS source text contains target filename
    try:
        tree=ast.parse(t)
        src_lines=t.splitlines()
        for n in ast.walk(tree):
            if isinstance(n,(ast.Assign,ast.AnnAssign)):
                value = n.value if hasattr(n,"value") else None
                if value is None:
                    continue
                seg=ast.get_source_segment(t,value) or ""
                if TARGET_NAME in seg:
                    targets=[]
                    if isinstance(n, ast.Assign):
                        targets=n.targets
                    else:
                        targets=[n.target]
                    for tar in targets:
                        if isinstance(tar, ast.Name):
                            assigned.add(tar.id)
    except Exception:
        pass

    evidence=[]
    for i,l in enumerate(t.splitlines(),1):
        if re.search(r"\.to_parquet\s*\(", l):
            if TARGET_NAME in l or any(re.search(rf"\b{re.escape(v)}\b",l) for v in assigned):
                evidence.append((i,l.rstrip()))
        if re.search(r"\bwrite_table\s*\(", l):
            if TARGET_NAME in l or any(re.search(rf"\b{re.escape(v)}\b",l) for v in assigned):
                evidence.append((i,l.rstrip()))
        if re.search(r"\bParquetWriter\s*\(", l):
            if TARGET_NAME in l or any(re.search(rf"\b{re.escape(v)}\b",l) for v in assigned):
                evidence.append((i,l.rstrip()))

    # Multi-line sink fallback: inspect compact windows around write calls.
    ls=t.splitlines()
    for idx,l in enumerate(ls):
        if any(x in l for x in (".to_parquet(", "write_table(", "ParquetWriter(")):
            window="\n".join(ls[max(0,idx-2):min(len(ls),idx+5)])
            if TARGET_NAME in window or any(re.search(rf"\b{re.escape(v)}\b",window) for v in assigned):
                item=(idx+1,l.rstrip())
                if item not in evidence:
                    evidence.append(item)

    return sorted(evidence), assigned

def shell_invoked_scripts(t):
    # pull obvious *.py script names from launcher
    return set(re.findall(r'([A-Za-z0-9_.-]+\.py)', t))

def main():
    print("="*108)
    print("WFS NFL — NFL-POSTGAME-1F-H-R2")
    print("EXACT UPSTREAM PARQUET PRODUCER / HOURLY-CHAIN AUDIT — READ ONLY")
    print("="*108)

    for p,h in EXPECTED.items():
        if not p.exists():
            raise RuntimeError(f"missing baseline file: {p}")
        got=sha(p)
        print(f"HASH|{p.name}|{got}|MATCH={str(got==h).upper()}")
        if got != h:
            raise RuntimeError(f"baseline hash mismatch: {p.name}")

    # Find exact producers.
    exact_producers=[]
    print("\n=== EXACT TARGET WRITE EVIDENCE ===")
    for p in sorted(ROOT.glob("*.py")):
        try:
            result=exact_target_write_evidence(p)
        except Exception:
            continue
        if not result:
            continue
        ev, assigned = result
        if ev:
            exact_producers.append(p)
            print(f"EXACT_PRODUCER_FILE={p.name}")
            print(f"TARGET_BOUND_SYMBOLS={','.join(sorted(assigned)) or 'NONE'}")
            for ln,line in ev:
                print(f"TARGET_WRITE_LINE|{p.name}|{ln}|{line}")

    producer_names={p.name for p in exact_producers}
    producer_stems={p.stem for p in exact_producers}
    print(f"EXACT_PRODUCER_COUNT={len(exact_producers)}")
    print(f"EXACT_PRODUCERS={','.join(sorted(producer_names)) or 'NONE'}")

    if not exact_producers:
        print("FAIL_REASON=No exact writer of target parquet could be proven.")
        status="FAIL_CLOSED"
    else:
        status="PASS"

    rup=txt(RUN_UPDATER)
    upd=txt(UPDATER)

    print("\n=== DIRECT HOURLY EXECUTION EVIDENCE ===")
    invoked=shell_invoked_scripts(rup)
    for x in sorted(invoked):
        print(f"RUN_UPDATER_SCRIPT_REF={x}")
    direct=sorted(producer_names & invoked)
    print(f"EXACT_PRODUCERS_DIRECTLY_REFERENCED_BY_RUN_UPDATER={','.join(direct) or 'NONE'}")

    print("\n=== UPDATER IMPORT EVIDENCE ===")
    imports=python_imports(UPDATER)
    prod_imports=sorted(producer_stems & imports)
    print(f"EXACT_PRODUCER_MODULES_IMPORTED_BY_UPDATER={','.join(prod_imports) or 'NONE'}")
    for stem in prod_imports:
        for ln,line in lines_matching(upd,[rf"\b{re.escape(stem)}\b"]):
            print(f"UPDATER_PRODUCER_REF|{ln}|{line}")

    # Also inspect every Python script directly invoked by run_updater for imports
    # of the exact producer module.
    print("\n=== INVOKED-STAGE IMPORT EVIDENCE ===")
    stage_importers=[]
    for s in sorted(invoked):
        p=ROOT/s
        if not p.exists() or p.suffix!=".py":
            continue
        imps=python_imports(p)
        matches=sorted(producer_stems & imps)
        if matches:
            stage_importers.append((s,matches))
            print(f"INVOKED_STAGE_IMPORTS_PRODUCER|{s}|{','.join(matches)}")
            tt=txt(p)
            for stem in matches:
                for ln,line in lines_matching(tt,[rf"\b{re.escape(stem)}\b"]):
                    print(f"INVOKED_STAGE_PRODUCER_REF|{s}|{ln}|{line}")

    # Search invoked scripts for subprocess/runpy execution of exact producer filename.
    subprocess_refs=[]
    for s in sorted(invoked):
        p=ROOT/s
        if not p.exists():
            continue
        tt=txt(p)
        for prod in exact_producers:
            if prod.name in tt:
                for ln,line in lines_matching(tt,[re.escape(prod.name)]):
                    subprocess_refs.append((s,prod.name,ln,line))
                    print(f"INVOKED_STAGE_EXACT_PRODUCER_FILENAME_REF|{s}|{prod.name}|{ln}|{line}")

    chain_proven=bool(direct or prod_imports or stage_importers or subprocess_refs)
    print(f"EXACT_PRODUCER_EXECUTION_IN_HOURLY_CHAIN_PROVEN={str(chain_proven).upper()}")

    print("\n=== SOURCE FILE MTIME DIAGNOSTIC ===")
    target=ROOT/"data/parquet"/TARGET_NAME
    if target.exists():
        st=target.stat()
        from datetime import datetime, timezone
        print(f"TARGET_EXISTS=TRUE")
        print(f"TARGET_MTIME_UTC={datetime.fromtimestamp(st.st_mtime,timezone.utc).isoformat()}")
        print(f"TARGET_SIZE_BYTES={st.st_size}")
        print(f"TARGET_SHA256={sha(target)}")
    else:
        print("TARGET_EXISTS=FALSE")

    print("\n=== DECISION ===")
    if chain_proven:
        print("INSERTION_READINESS=READY_FOR_CONTROLLED_LAUNCHER_PATCH")
        print("REQUIRED_ORDER=exact_target_producer_success -> NFL-POSTGAME-1F-F.py -> remaining protected chain -> lock release")
    else:
        print("INSERTION_READINESS=FAIL_CLOSED_SOURCE_REFRESH_NOT_PROVEN")
        print("NEXT_REQUIRED_ACTION=trace exact producer invocation/refresh mechanism before launcher modification")

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
    print(f"NFL_POSTGAME_1F_H_R2_STATUS={status}")
    if status != "PASS":
        raise SystemExit(2)

if __name__ == "__main__":
    main()
