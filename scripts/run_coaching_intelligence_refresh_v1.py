#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

VERSION = "WFS_COACHING_INTELLIGENCE_REFRESH_V1"


STAGES = [
    (
        "LIVE GAME INTELLIGENCE",
        ROOT
        / "scripts"
        / "run_live_game_intelligence_all_v1.py",
    ),

    (
        "LIVE COACHING INTERPRETATION",
        ROOT
        / "scripts"
        / "run_live_coaching_interpretation_all_v1.py",
    ),

    (
        "POSTGAME ASSIMILATION",
        ROOT
        / "scripts"
        / "run_postgame_coaching_assimilation_v1.py",
    ),

    (
        "COACH IDENTITY ATTACHMENT",
        ROOT
        / "scripts"
        / "build_postgame_coach_attached_evidence_v1.py",
    ),

    (
        "COACH REGIME PROFILE",
        ROOT
        / "scripts"
        / "build_coach_regime_evidence_profile_v1.py",
    ),
]


def run_stage(
    name: str,
    script: Path,
) -> None:
    print()
    print("=" * 72)
    print(name)
    print("=" * 72)

    if not script.exists():
        raise RuntimeError(
            f"missing stage script: {script}"
        )

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
        ],
        cwd=str(ROOT),
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"{name} failed "
            f"with rc={proc.returncode}"
        )

    print(
        f"PASS: {name}"
    )


def main() -> int:
    print("=" * 72)
    print(VERSION)
    print("=" * 72)

    print(
        "POLICY: analysis-only; "
        "no production forecast, solver, "
        "or persistent coach-prior mutation"
    )

    try:
        for name, script in STAGES:
            run_stage(
                name,
                script,
            )

    except Exception as exc:
        print()
        print("=" * 72)
        print("FAIL")
        print("=" * 72)
        print(str(exc))
        return 1

    print()
    print("=" * 72)
    print(
        "COACHING INTELLIGENCE REFRESH PASS"
    )
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
