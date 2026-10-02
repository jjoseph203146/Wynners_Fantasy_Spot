#!/usr/bin/env python3

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
SOURCE = APP_DIR / "data" / "wfs_accounts.db"
BACKUP_DIR = APP_DIR / "backups"

# Keep two weeks of automatic daily backups.
RETENTION_DAYS = 14


def main() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    if not SOURCE.exists():
        raise SystemExit(f"Source database not found: {SOURCE}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = BACKUP_DIR / f"wfs_accounts_auto_{stamp}.db"

    # SQLite online backup API. Safe for a live WAL database.
    source_uri = f"file:{SOURCE}?mode=ro"

    with sqlite3.connect(source_uri, uri=True) as src:
        with sqlite3.connect(destination) as dst:
            src.backup(dst)

    # Validate the newly created backup before considering it successful.
    with sqlite3.connect(
        f"file:{destination}?mode=ro",
        uri=True,
    ) as check:
        result = check.execute("PRAGMA integrity_check").fetchone()

    if not result or result[0] != "ok":
        destination.unlink(missing_ok=True)
        raise SystemExit(
            f"Backup integrity check FAILED: {result}"
        )

    destination.chmod(0o600)

    # Retention applies only to backups made by this script.
    cutoff = datetime.now().timestamp() - (RETENTION_DAYS * 86400)

    removed = 0

    for old in BACKUP_DIR.glob("wfs_accounts_auto_*.db"):
        if old == destination:
            continue

        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
                removed += 1
        except FileNotFoundError:
            pass

    print(f"Backup created: {destination}")
    print(f"Integrity check: ok")
    print(f"Permissions: 600")
    print(f"Expired backups removed: {removed}")


if __name__ == "__main__":
    main()
