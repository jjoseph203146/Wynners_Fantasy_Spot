#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import shutil
import subprocess
import os

ROOT = Path("/home/mwynn/nfl_data_engine")
TARGET = ROOT / "run_updater.sh"

EXPECTED_UPDATER = (
    "54884d4636be3bedc41d0bc8eeb280c017d28aa272e85762cec19b216cfb4f0b"
)

EXPECTED_BUILD = (
    "6ab248295dcab64108397c7a074d5210e3f252827f1e23adf5a069607455df02"
)

EXPECTED_V3 = (
    "bf9a7e8b4645e3def7b9c67d2af7badceb67fffbddf0cd0da755e6fb07c94853"
)

EXPECTED_CAPTURE = (
    "fdb18e85e204fa29d2bff9d6ab9776eb45cdf860f5018e2f783a20ca9488fb77"
)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


files = {
    TARGET: EXPECTED_UPDATER,
    ROOT / "scripts/build_forecast_live_core_v1.py": EXPECTED_BUILD,
    ROOT / "scripts/run_forecast_live_core_v3.py": EXPECTED_V3,
    ROOT / "scripts/capture_forecast_snapshot_v2.py": EXPECTED_CAPTURE,
}

print("=== NFL-2026-FEEDBACK-1V INSTALL ===")

for path, expected in files.items():
    actual = sha(path)
    print(f"{path.name}_SHA256={actual}")

    if actual != expected:
        raise RuntimeError(
            f"FAIL_CLOSED: hash mismatch: {path}"
        )

print("HASH_GATE=PASS")

original = TARGET.read_text()

stamp = datetime.now(
    timezone.utc
).strftime("%Y%m%dT%H%M%SZ")

backup = ROOT / (
    f"run_updater.sh.pre-feedback-1v-{stamp}.bak"
)

shutil.copy2(TARGET, backup)

print(f"BACKUP={backup}")
print(f"BACKUP_SHA256={sha(backup)}")

old_vars = '''FEEDBACK_REFRESH="$PROJECT_DIR/nfl_2026_feedback_refresh.py"
LOCK_FILE="$PROJECT_DIR/nfl_updater.lock"
'''

new_vars = '''FEEDBACK_REFRESH="$PROJECT_DIR/nfl_2026_feedback_refresh.py"
FORECAST_BUILD="$PROJECT_DIR/scripts/build_forecast_live_core_v1.py"
FORECAST_V3="$PROJECT_DIR/scripts/run_forecast_live_core_v3.py"
FORECAST_CAPTURE="$PROJECT_DIR/scripts/capture_forecast_snapshot_v2.py"
LOCK_FILE="$PROJECT_DIR/nfl_updater.lock"
'''

if original.count(old_vars) != 1:
    raise RuntimeError(
        "FAIL_CLOSED: variable insertion anchor mismatch"
    )

text = original.replace(
    old_vars,
    new_vars,
    1,
)

old_pipeline = '''    run_stage "2026 feedback refresh" "$FEEDBACK_REFRESH" 24
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    return 0
'''

new_pipeline = '''    run_stage "2026 feedback refresh" "$FEEDBACK_REFRESH" 24
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    run_stage "Forecast LIVE CORE build" "$FORECAST_BUILD" 25
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    run_stage "Forecast LIVE CORE V3" "$FORECAST_V3" 26
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    run_stage "Forecast snapshot capture" "$FORECAST_CAPTURE" 27
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    return 0
'''

if text.count(old_pipeline) != 1:
    raise RuntimeError(
        "FAIL_CLOSED: pipeline insertion anchor mismatch"
    )

text = text.replace(
    old_pipeline,
    new_pipeline,
    1,
)

# Static contract checks.
checks = {
    'run_stage "2026 feedback refresh"': 1,
    'run_stage "Forecast LIVE CORE build"': 1,
    'run_stage "Forecast LIVE CORE V3"': 1,
    'run_stage "Forecast snapshot capture"': 1,
    'exec 9>"$LOCK_FILE"': 1,
    'flock -w 60 9': 1,
}

for needle, expected_count in checks.items():
    actual = text.count(needle)

    print(
        f"CHECK={needle!r} COUNT={actual}"
    )

    if actual != expected_count:
        raise RuntimeError(
            f"FAIL_CLOSED: contract count "
            f"{needle!r}={actual}"
        )

tmp = ROOT / "run_updater.sh.feedback-1v.tmp"

tmp.write_text(text)
os.chmod(tmp, TARGET.stat().st_mode)

result = subprocess.run(
    ["bash", "-n", str(tmp)]
)

if result.returncode != 0:
    tmp.unlink(missing_ok=True)
    raise RuntimeError(
        "FAIL_CLOSED: shell syntax"
    )

candidate = sha(tmp)

print(f"CANDIDATE_SHA256={candidate}")

os.replace(tmp, TARGET)

installed = sha(TARGET)

print(f"INSTALLED_SHA256={installed}")

if installed != candidate:
    raise RuntimeError(
        "FAIL_CLOSED: installed hash mismatch"
    )

print("NFL_2026_FEEDBACK_1V_INSTALL_STATUS=PASS")
