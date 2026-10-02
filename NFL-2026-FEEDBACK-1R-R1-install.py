#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import os
import shutil
import subprocess


ROOT = Path("/home/mwynn/nfl_data_engine")
TARGET = ROOT / "run_updater.sh"

EXPECTED_UPDATER_SHA = (
    "46ffe95fb09585538bc01efceba6c24a111bfe85072f660a6f9a1e1f380ef96a"
)

EXPECTED_FEEDBACK_SHA = (
    "02320fca4a6db2592329c77152f2c1abb6212d4eb46e5e26de391ed5d2ca0049"
)

EXPECTED_PROJECTION_SHA = (
    "2ab7b98663e16d0a5747e244173b7b407c0bfb9975868000f7cba6627738439d"
)


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def replace_once(text, old, new, label):
    count = text.count(old)

    print(f"{label}_MATCH_COUNT={count}")

    if count != 1:
        raise RuntimeError(
            f"FAIL_CLOSED: {label} expected "
            f"exactly one match, found {count}"
        )

    return text.replace(
        old,
        new,
        1,
    )


def main():
    print(
        "=== NFL-2026-FEEDBACK-1R-R1 INSTALL ==="
    )

    feedback = ROOT / "nfl_2026_feedback_refresh.py"
    projection = ROOT / "production_projection.py"

    current_updater_sha = sha256(TARGET)
    feedback_sha = sha256(feedback)
    projection_sha = sha256(projection)

    print(
        f"CURRENT_UPDATER_SHA256="
        f"{current_updater_sha}"
    )

    print(
        f"FEEDBACK_SHA256="
        f"{feedback_sha}"
    )

    print(
        f"PROJECTION_SHA256="
        f"{projection_sha}"
    )

    if current_updater_sha != EXPECTED_UPDATER_SHA:
        raise RuntimeError(
            "FAIL_CLOSED: run_updater.sh "
            "baseline mismatch"
        )

    if feedback_sha != EXPECTED_FEEDBACK_SHA:
        raise RuntimeError(
            "FAIL_CLOSED: feedback orchestrator "
            "baseline mismatch"
        )

    if projection_sha != EXPECTED_PROJECTION_SHA:
        raise RuntimeError(
            "FAIL_CLOSED: production projection "
            "baseline mismatch"
        )

    original = TARGET.read_text()

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    backup = ROOT / (
        "run_updater.sh."
        f"pre-feedback-1r-r1-{stamp}.bak"
    )

    shutil.copy2(
        TARGET,
        backup,
    )

    print(
        f"BACKUP={backup}"
    )

    print(
        f"BACKUP_SHA256={sha256(backup)}"
    )

    text = original

    # -----------------------------------------------------
    # Add feedback script path.
    # -----------------------------------------------------

    old = '''SOLVER_READY="$PROJECT_DIR/fanduel_solver_ready_pool.py"
LOCK_FILE="$PROJECT_DIR/nfl_updater.lock"
'''

    new = '''SOLVER_READY="$PROJECT_DIR/fanduel_solver_ready_pool.py"
FEEDBACK_REFRESH="$PROJECT_DIR/nfl_2026_feedback_refresh.py"
LOCK_FILE="$PROJECT_DIR/nfl_updater.lock"
'''

    text = replace_once(
        text,
        old,
        new,
        "VARIABLE_INSERT",
    )

    # -----------------------------------------------------
    # Append feedback after frozen injury chain.
    # This remains inside the already-held updater lock.
    # -----------------------------------------------------

    old = '''    run_stage "Solver-ready injury rebuild" "$SOLVER_READY" 23
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    return 0
'''

    new = '''    run_stage "Solver-ready injury rebuild" "$SOLVER_READY" 23
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    if ! wait_for_live_quiet "2026 feedback refresh"; then
        return 34
    fi

    run_stage "2026 feedback refresh" "$FEEDBACK_REFRESH" 24
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    return 0
'''

    text = replace_once(
        text,
        old,
        new,
        "PIPELINE_INSERT",
    )

    # -----------------------------------------------------
    # Static safety gates.
    # -----------------------------------------------------

    if text.count(
        'FEEDBACK_REFRESH='
    ) != 1:
        raise RuntimeError(
            "FAIL_CLOSED: feedback variable "
            "count incorrect"
        )

    if text.count(
        'run_stage "2026 feedback refresh"'
    ) != 1:
        raise RuntimeError(
            "FAIL_CLOSED: feedback stage "
            "count incorrect"
        )

    if text.count(
        'exec 9>"$LOCK_FILE"'
    ) != 1:
        raise RuntimeError(
            "FAIL_CLOSED: updater lock contract "
            "changed unexpectedly"
        )

    if text.count(
        'flock -w 60 9'
    ) != 1:
        raise RuntimeError(
            "FAIL_CLOSED: updater flock contract "
            "changed unexpectedly"
        )

    tmp = ROOT / "run_updater.sh.feedback-1r-r1.tmp"

    tmp.write_text(text)

    os.chmod(
        tmp,
        TARGET.stat().st_mode,
    )

    syntax = subprocess.run(
        [
            "bash",
            "-n",
            str(tmp),
        ]
    )

    if syntax.returncode != 0:
        tmp.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "FAIL_CLOSED: candidate shell "
            "syntax failed"
        )

    candidate_sha = sha256(tmp)

    print(
        f"CANDIDATE_SHA256="
        f"{candidate_sha}"
    )

    os.replace(
        tmp,
        TARGET,
    )

    installed_sha = sha256(TARGET)

    print(
        f"INSTALLED_SHA256="
        f"{installed_sha}"
    )

    if installed_sha != candidate_sha:
        raise RuntimeError(
            "FAIL_CLOSED: installed SHA mismatch"
        )

    print(
        "NFL_2026_FEEDBACK_1R_R1_INSTALL_STATUS=PASS"
    )


if __name__ == "__main__":
    main()
