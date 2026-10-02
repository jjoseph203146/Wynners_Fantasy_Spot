#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/mwynn/nfl_data_engine")
TARGET = ROOT / "nfl_2026_feedback_refresh.py"
PROD = ROOT / "production_projection.py"
BACKUP_ROOT = ROOT / "backups"

OLD_SHA = "2ab7b98663e16d0a5747e244173b7b407c0bfb9975868000f7cba6627738439d"
APPROVED_SHA = "3a70783057ae57b544a2f367909e8a234a67e1895fe5e29863d5eed0ca34ae23"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not TARGET.exists():
        raise RuntimeError(f"Missing target: {TARGET}")
    if not PROD.exists():
        raise RuntimeError(f"Missing production projection: {PROD}")

    actual = sha256(PROD)
    if actual != APPROVED_SHA:
        raise RuntimeError(
            "ABORT: production_projection.py does not match the validated "
            f"approved SHA.\nExpected: {APPROVED_SHA}\nActual:   {actual}"
        )

    text = TARGET.read_text(encoding="utf-8")

    if APPROVED_SHA in text and OLD_SHA not in text:
        raise RuntimeError(
            "Approved production SHA is already installed; no changes made."
        )

    if text.count(OLD_SHA) != 1:
        raise RuntimeError(
            f"Expected old frozen SHA exactly once; found {text.count(OLD_SHA)}. "
            "No changes were written."
        )

    patched = text.replace(OLD_SHA, APPROVED_SHA, 1)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / f"feedback_projection_sha_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup = backup_dir / TARGET.name
    shutil.copy2(TARGET, backup)

    TARGET.write_text(patched, encoding="utf-8")

    try:
        compile(patched, str(TARGET), "exec")
    except Exception:
        shutil.copy2(backup, TARGET)
        raise

    verify = TARGET.read_text(encoding="utf-8")
    if verify.count(APPROVED_SHA) != 1 or OLD_SHA in verify:
        shutil.copy2(backup, TARGET)
        raise RuntimeError("Post-write verification failed; original restored.")

    print("=" * 96)
    print("ROLLING-FEEDBACK PRODUCTION BASELINE UPDATE: PASS")
    print("=" * 96)
    print(f"Target: {TARGET}")
    print(f"Backup: {backup}")
    print(f"Verified production_projection.py SHA256: {actual}")
    print()
    print("Changed:")
    print(f"  EXPECTED_PRODUCTION_SHA: {OLD_SHA}")
    print(f"                       -> {APPROVED_SHA}")
    print()
    print("Preserved:")
    print("  fail-closed SHA enforcement")
    print("  DB integrity checks")
    print("  feedback stage order")
    print("  all projection/model logic")
    print("  all solver logic")
    print()
    print("Python compile: PASS")


if __name__ == "__main__":
    main()
