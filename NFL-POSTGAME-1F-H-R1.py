#!/usr/bin/env python3
"""
WFS NFL — NFL-POSTGAME-1F-H-R1
Exact automation insertion-point audit — READ ONLY.

Purpose:
  1) Prove the updater lock path semantically, not by absolute-string grep.
  2) Trace where nfl_fanduel_player_pool.parquet is produced/refreshed.
  3) Prove whether that refresh occurs inside the hourly run_updater.sh chain.
  4) Identify the exact safest insertion anchor for NFL-POSTGAME-1F-F.py.
  5) Fail closed if the source parquet is not refreshed by the chain.

No writes. No cron/service/updater/launcher changes.
"""

from pathlib import Path
import hashlib, re, subprocess, ast

ROOT = Path("/home/mwynn/nfl_data_engine")
TARGET_PARQUET = "nfl_fanduel_player_pool.parquet"
WRITER = ROOT / "NFL-POSTGAME-1F-F.py"
RUN_UPDATER = ROOT / "run_updater.sh"
UPDATER = ROOT / "updater.py"

EXPECTED_WRITER = "063e1bdfaf34208bd6d56705d808d13b3a21859744731ba49e3ff1bf02857315"
EXPECTED_RUN_UPDATER = "46ffe95fb09585538bc01efceba6c24a111bfe85072f660a6f9a1e1f380ef96a"
EXPECTED_UPDATER = "a9637bcb077d62c35cfabf48c5a17aa293c7176f35302f7020d5a0aaef41096a"

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def text(p):
    return p.read_text(errors="replace")

def line_matches(txt, patterns):
    out = []
    for i, line in enumerate(txt.splitlines(), 1):
        if any(re.search(p, line, re.I) for p in patterns):
            out.append((i, line.rstrip()))
    return out

def shell_assignments(txt):
    vals = {}
    for line in txt.splitlines():
        m = re.match(r'^\s*([A-Za-z_][A-Za-z0-9_]*)=(?:"([^"]*)"|\'([^\']*)\'|([^#\s]+))\s*(?:#.*)?$', line)
        if not m:
            continue
        vals[m.group(1)] = next(x for x in m.groups()[1:] if x is not None)
    return vals

def expand_shell(s, vals):
    # limited deterministic expansion sufficient for PROJECT_DIR/LOCK_FILE style assignments
    prev = None
    cur = s
    for _ in range(10):
        if cur == prev:
            break
        prev = cur
        for k, v in vals.items():
            cur = cur.replace("${"+k+"}", v).replace("$"+k, v)
    return cur

def main():
    print("="*104)
    print("WFS NFL — NFL-POSTGAME-1F-H-R1")
    print("EXACT AUTOMATION INSERTION-POINT AUDIT — READ ONLY")
    print("="*104)

    for p in (WRITER, RUN_UPDATER, UPDATER):
        if not p.exists():
            raise RuntimeError(f"missing required file: {p}")

    wh, rh, uh = sha(WRITER), sha(RUN_UPDATER), sha(UPDATER)
    print(f"WRITER_SHA256={wh}")
    print(f"RUN_UPDATER_SHA256={rh}")
    print(f"UPDATER_SHA256={uh}")
    print(f"WRITER_SHA_MATCH={str(wh == EXPECTED_WRITER).upper()}")
    print(f"RUN_UPDATER_SHA_MATCH={str(rh == EXPECTED_RUN_UPDATER).upper()}")
    print(f"UPDATER_SHA_MATCH={str(uh == EXPECTED_UPDATER).upper()}")
    if wh != EXPECTED_WRITER or rh != EXPECTED_RUN_UPDATER or uh != EXPECTED_UPDATER:
        raise RuntimeError("frozen baseline hash mismatch")

    rup = text(RUN_UPDATER)
    upd = text(UPDATER)

    print("\n=== LOCK SEMANTICS ===")
    assigns = shell_assignments(rup)
    for k in ("PROJECT_DIR","LOCK_FILE"):
        print(f"SHELL_ASSIGNMENT|{k}|{assigns.get(k,'<MISSING>')}")
    lock_raw = assigns.get("LOCK_FILE")
    project_raw = assigns.get("PROJECT_DIR")
    vals = dict(assigns)
    # normalize common dirname-derived PROJECT_DIR if parser couldn't resolve command substitution
    if project_raw and "$(" in project_raw:
        vals["PROJECT_DIR"] = str(ROOT)
    resolved_lock = expand_shell(lock_raw, vals) if lock_raw else None
    print(f"RESOLVED_LOCK_FILE={resolved_lock}")
    print(f"EXPECTED_LOCK_FILE={ROOT / 'nfl_updater.lock'}")
    lock_ok = resolved_lock == str(ROOT / "nfl_updater.lock")
    print(f"LOCK_PATH_SEMANTIC_MATCH={str(lock_ok).upper()}")

    flock_lines = line_matches(rup, [r"\bflock\b", r"\bexec\s+9>", r"LOCK_FILE"])
    for ln, line in flock_lines:
        print(f"LOCK_LINE|{ln}|{line}")

    fd9_open = bool(re.search(r'exec\s+9>\s*["\']?\$?\{?LOCK_FILE\}?["\']?', rup))
    flock9 = bool(re.search(r'\bflock\b[^\n]*\b9\b', rup))
    print(f"FD9_OPENS_LOCK_FILE={str(fd9_open).upper()}")
    print(f"FLOCK_USES_FD9={str(flock9).upper()}")

    print("\n=== TARGET PARQUET PRODUCER TRACE ===")
    # Search project Python/shell files for the exact target filename and generic producer calls.
    hits = []
    for pat in ("*.py","*.sh"):
        for p in sorted(ROOT.glob(pat)):
            try:
                t = text(p)
            except Exception:
                continue
            if TARGET_PARQUET in t:
                lm = line_matches(t, [re.escape(TARGET_PARQUET)])
                for ln, line in lm:
                    hits.append((p.name, ln, line))
    print(f"FILES_REFERENCING_TARGET_PARQUET={len(set(x[0] for x in hits))}")
    for name, ln, line in hits:
        print(f"PARQUET_REF|{name}|{ln}|{line}")

    # Detect likely writes versus reads by nearby keywords.
    producer_candidates = []
    reader_candidates = []
    write_words = re.compile(r"to_parquet|write_table|ParquetWriter|OUTPUT|output|save|export", re.I)
    read_words = re.compile(r"read_parquet|pd\.read_parquet", re.I)
    for name, ln, line in hits:
        if write_words.search(line):
            producer_candidates.append((name,ln,line))
        if read_words.search(line):
            reader_candidates.append((name,ln,line))

    # More robust scan: file containing target filename and any parquet write call anywhere.
    for name in sorted(set(x[0] for x in hits)):
        p = ROOT / name
        t = text(p)
        if re.search(r"\.to_parquet\s*\(|write_table\s*\(|ParquetWriter\s*\(", t):
            if not any(x[0] == name for x in producer_candidates):
                producer_candidates.append((name,0,"<file contains target filename and parquet write call>"))

    print(f"PRODUCER_CANDIDATE_FILES={','.join(sorted(set(x[0] for x in producer_candidates))) or 'NONE'}")
    print(f"READER_CANDIDATE_FILES={','.join(sorted(set(x[0] for x in reader_candidates))) or 'NONE'}")

    print("\n=== HOURLY CHAIN TRACE ===")
    # Print run_updater invocations and updater imports/calls that could lead to producer.
    run_calls = line_matches(rup, [
        r'"\$PYTHON"', r"python", r"run_stage", r"\$UPDATER",
        r"player_pool", r"parquet", r"projection"
    ])
    for ln, line in run_calls:
        print(f"RUN_UPDATER_CHAIN|{ln}|{line}")

    updater_refs = line_matches(upd, [
        r"player_pool", r"parquet", r"projection", r"export", r"build",
        r"refresh", r"fanduel"
    ])
    for ln, line in updater_refs:
        print(f"UPDATER_CHAIN|{ln}|{line}")

    producer_files = sorted(set(x[0] for x in producer_candidates))
    chain_text = rup + "\n" + upd
    producer_named_in_chain = [f for f in producer_files if f in chain_text]
    print(f"PRODUCER_FILES_NAMED_IN_HOURLY_CHAIN={','.join(producer_named_in_chain) or 'NONE'}")

    # A producer may be imported as a module rather than filename; check stems.
    producer_stems = [Path(f).stem for f in producer_files]
    producer_stem_refs = [s for s in producer_stems if re.search(rf"\b{re.escape(s)}\b", chain_text)]
    print(f"PRODUCER_MODULES_REFERENCED_IN_HOURLY_CHAIN={','.join(producer_stem_refs) or 'NONE'}")

    source_refresh_proven = bool(producer_named_in_chain or producer_stem_refs)
    print(f"UPSTREAM_PARQUET_REFRESH_IN_HOURLY_CHAIN_PROVEN={str(source_refresh_proven).upper()}")

    print("\n=== INSERTION DECISION ===")
    if source_refresh_proven and lock_ok and fd9_open and flock9:
        decision = "READY_FOR_CONTROLLED_LAUNCHER_PATCH"
        print("RECOMMENDED_LOCATION=inside run_updater.sh existing EX lock, after successful upstream pool refresh, before injury-chain completion/lock release")
    else:
        decision = "FAIL_CLOSED_MORE_TRACE_REQUIRED"
        if not source_refresh_proven:
            print("BLOCKER=target upstream parquet refresh is not proven to occur inside hourly updater chain")
        if not lock_ok:
            print("BLOCKER=lock path semantic resolution did not match expected nfl_updater.lock")
        if not fd9_open or not flock9:
            print("BLOCKER=FD9/exclusive flock contract not fully proven by text audit")

    print(f"INSERTION_READINESS={decision}")

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
    status = "PASS" if (lock_ok and fd9_open and flock9) else "FAIL_CLOSED"
    # PASS here means lock audit completed safely; readiness is independently explicit.
    print(f"NFL_POSTGAME_1F_H_R1_STATUS={status}")
    if status != "PASS":
        raise SystemExit(2)

if __name__ == "__main__":
    main()
