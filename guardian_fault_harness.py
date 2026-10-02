#!/usr/bin/env python3
from __future__ import annotations

from guardian_classification import classify_failures


TESTS = [
    {
        "name": "publication_hash_mismatch",
        "expected": "PUBLICATION",
        "results": [
            {
                "check": "FANDUEL_PUBLICATION_LIFECYCLE",
                "status": "FAIL",
                "detail": "hash_chain=False",
            }
        ],
    },
    {
        "name": "wrong_active_week",
        "expected": "SCHEDULE",
        "results": [
            {
                "check": "FANDUEL_SCHEDULE_ALIGNMENT",
                "status": "FAIL",
                "detail": "manifest_week=3 active_week=2",
            }
        ],
    },
    {
        "name": "duplicate_player_identity",
        "expected": "IDENTITY",
        "results": [
            {
                "check": "OFFENSE_IDENTITY_CONTRACT",
                "status": "FAIL",
                "detail": "duplicates=1",
            }
        ],
    },
    {
        "name": "broken_component_handoff",
        "expected": "HANDOFF",
        "results": [
            {
                "check": "COMPONENT_IDENTITY_HANDOFF",
                "status": "FAIL",
                "detail": "offense_diff=1 kicker_diff=0 dst_diff=0",
            }
        ],
    },
    {
        "name": "availability_failure",
        "expected": "AVAILABILITY",
        "results": [
            {
                "check": "STARTER_VERIFICATION_SHADOW",
                "status": "FAIL",
                "detail": "identity_gate=FAIL",
            }
        ],
    },
    {
        "name": "forecast_failure",
        "expected": "FORECAST",
        "results": [
            {
                "check": "OFFENSIVE_FORECAST_CONTRACT",
                "status": "FAIL",
                "detail": "projection_contract=FAIL",
            }
        ],
    },
    {
        "name": "database_integrity_failure",
        "expected": "INFRASTRUCTURE",
        "results": [
            {
                "check": "NFL_DATA_AUTHORITY",
                "status": "FAIL",
                "detail": "integrity=failed",
            }
        ],
    },
]


def main() -> int:
    print("=" * 72)
    print("WFS GUARDIAN V2.6 — ISOLATED FAULT-INJECTION HARNESS")
    print("=" * 72)
    print("PRODUCTION_DATA_USED=FALSE")
    print("PRODUCTION_MUTATION=FALSE")
    print("LKG_MUTATION=FALSE")
    print("AUDIT_LEDGER_MUTATION=FALSE")
    print("SERVICE_ACTION=FALSE")
    print("-" * 72)

    failures = 0

    for test in TESTS:
        classified = classify_failures(test["results"])

        categories = classified["categories"]

        actual = (
            categories[0]
            if len(categories) == 1
            else ",".join(categories)
        )

        passed = (
            classified["failure_count"] == 1
            and categories == [test["expected"]]
        )

        if not passed:
            failures += 1

        print(
            f"{test['name']:<32} "
            f"{'PASS' if passed else 'FAIL':<5} "
            f"expected={test['expected']} "
            f"actual={actual}"
        )

    print("-" * 72)

    if failures:
        print(f"FAULT_TEST_FAILURES={failures}")
        print("V26_CLASSIFICATION_HARNESS=FAIL")
        return 1

    print("FAULT_TEST_FAILURES=0")
    print("V26_CLASSIFICATION_HARNESS=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
