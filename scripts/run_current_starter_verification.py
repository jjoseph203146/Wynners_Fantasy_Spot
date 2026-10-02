#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wfs_schedule_context import resolve_schedule_week_context


def main() -> int:
    context = resolve_schedule_week_context()

    season = int(context.season)
    week = int(context.planning_week)

    if week < 1:
        raise RuntimeError(
            f"Invalid planning week from schedule authority: {week}"
        )

    # Week 1 has no prior regular-season usage week.
    # Fail closed rather than inventing a prior-week value.
    if week == 1:
        raise RuntimeError(
            "Starter verification requires a prior regular-season week; "
            "planning_week=1"
        )

    prior_week = week - 1

    print("CURRENT_STARTER_VERIFICATION")
    print(f"SCHEDULE_SEASON={season}")
    print(f"SCHEDULE_PLANNING_WEEK={week}")
    print(f"PRIOR_USAGE_WEEK={prior_week}")

    command = [
        sys.executable,
        "-B",
        str(ROOT / "scripts" / "starter_verification_v1.py"),
        "--season",
        str(season),
        "--week",
        str(week),
        "--prior-week",
        str(prior_week),
        "--game-type",
        "REG",
    ]

    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
    )

    print(f"STARTER_VERIFICATION_RETURN_CODE={result.returncode}")

    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
