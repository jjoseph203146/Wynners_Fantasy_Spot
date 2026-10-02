"""Offline regression/compile/publication validation with process-wide I/O guards.

Run with python -B -m shadow_player_analysis_v1_2.validate.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
PROTECTED = ["wfs_ai_analyst.py", "fanduel_injury_ingest.py", "matchup_intelligence_v1.py",
             "scripts/starter_verification_v1.py", "injury_consensus.py", "injury_consensus_v2_compat.py",
             "app.py", "run_updater.sh"]


def main():
    sys.dont_write_bytecode = True
    protected = [PROJECT / p for p in PROTECTED] + list((PROJECT / "shadow_player_analysis_v1").glob("*.py"))
    before = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    counters = dict(network_attempts=0, database_connection_attempts=0, forbidden_mutation_attempts=0)
    allowed = [Path("/tmp").resolve(), ROOT / "artifacts", ROOT / "validation.json"]

    def check_write(value):
        if isinstance(value, int):
            return
        path = Path(os.fsdecode(value)).resolve()
        if not any(path == base or base in path.parents for base in allowed):
            counters["forbidden_mutation_attempts"] += 1
            raise RuntimeError("FORBIDDEN_WRITE: " + str(path))

    def guard(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            counters["network_attempts"] += 1
            raise RuntimeError("LIVE_NETWORK_FORBIDDEN")
        if event == "sqlite3.connect":
            counters["database_connection_attempts"] += 1
            raise RuntimeError("DATABASE_ACCESS_FORBIDDEN")
        if event == "open":
            _, mode, flags = args
            if (mode and any(c in mode for c in "wax+")) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
                check_write(args[0])
        if event in {"os.rename", "os.replace"}:
            check_write(args[0]); check_write(args[1])
        if event in {"os.remove", "os.rmdir", "os.mkdir"}:
            # shutil's directory-FD relative cleanup is inside approved temp roots.
            if len(args) < 2 or not isinstance(args[-1], int) or args[-1] < 0:
                check_write(args[0])
        if event in {"subprocess.Popen", "os.system"}:
            raise RuntimeError("SUBPROCESS_FORBIDDEN")

    sys.addaudithook(guard)
    modules = ["shadow_player_analysis_v1.test_foundation", "shadow_player_analysis_v1.test_espn_nfl_rss_v1",
               "shadow_player_analysis_v1_2.test_v1_2"]
    compiled = 0
    for folder in (PROJECT / "shadow_player_analysis_v1", ROOT):
        for path in sorted(folder.glob("*.py")):
            compile(path.read_bytes(), str(path), "exec")
            compiled += 1
    report = dict(compile_validation="PASS", compiled_files=compiled)
    for label, module in zip(("foundation_tests", "espn_v1_1_tests", "v1_2_tests"), modules):
        suite = unittest.defaultTestLoader.loadTestsFromName(module)
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        report[label] = dict(run=result.testsRun, failures=len(result.failures), errors=len(result.errors))
        if not result.wasSuccessful():
            raise SystemExit(1)
    from .fixtures import sample
    from .publication import publish, verify
    from .core import build
    bundles = []
    for ambiguous in (False, True):
        inputs = sample(True)
        if ambiguous:
            for event in inputs["records"]:
                event.update(raw_publication_timestamp="Sun, 07 Sep 2025 10:00:00 EST",
                             published_at_utc=None, temporal_status="AMBIGUOUS_SOURCE_TIMESTAMP",
                             temporal_confidence="UNRESOLVED")
        path = publish(inputs)
        manifest = verify(path)
        result = build(**inputs)
        if ambiguous:
            assert all(not r["pregame_eligible"] for r in result["classified_evidence"])
        assert all(not r["claim_eligible"] and not r["consensus_eligible"] for r in result["classified_evidence"])
        bundles.append(dict(path=str(path.relative_to(PROJECT)), run_id=manifest["run_id"],
                            ambiguous_timestamps=ambiguous, events=len(result["source_events"]),
                            entity_links=len(result["entity_links"])))
    after = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    assert before == after, "PROTECTED_CODE_CHANGED"
    assert not any(counters.values()), counters
    report.update(counters, protected_code_unchanged=True, protected_code_sha256=after, bundles=bundles)
    (ROOT / "validation.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
