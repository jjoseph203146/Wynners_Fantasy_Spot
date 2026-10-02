#!/usr/bin/env python3
from __future__ import annotations


CLASSIFICATION_CONTRACT = "WFS_GUARDIAN_FAILURE_CLASSIFICATION_V1"

VALID_CATEGORIES = {
    "DATA_AUTHORITY",
    "IDENTITY",
    "SCHEDULE",
    "AVAILABILITY",
    "FORECAST",
    "HANDOFF",
    "PUBLICATION",
    "INFRASTRUCTURE",
}


def classify_check(check: str, detail: str = "") -> str:
    """
    Deterministic classification from Guardian contract/check names.

    No AI inference.
    No fuzzy classification.
    No production action.
    """

    name = str(check or "").upper()
    detail_u = str(detail or "").upper()

    # Infrastructure / database integrity
    if (
        name in {
            "NFL_DATA_AUTHORITY",
            "FORECAST_LEDGER",
            "PLAYER_PROJECTION_LEDGER",
            "ROLE_PERFORMANCE_DB",
            "NFL_LIVE_DB",
        }
        or "INTEGRITY" in name
        or "DB_" in name
        or "DATABASE" in name
        or "SQLITE" in detail_u
    ):
        return "INFRASTRUCTURE"

    # Schedule / lifecycle targeting
    if (
        "SCHEDULE" in name
        or name.endswith("_SCHEDULE_ALIGNMENT")
        or name == "SCHEDULE_ACTIVE_WEEK"
    ):
        return "SCHEDULE"

    # Explicit cross-stage handoffs/reconciliation.
    #
    # This must precede generic identity classification because checks
    # such as COMPONENT_IDENTITY_HANDOFF contain the word IDENTITY but
    # represent a failure between stages, not an intrinsic entity-ID
    # contract failure.
    if (
        "HANDOFF" in name
        or name == "COMPONENT_TO_UNIFIED"
        or name == "UNIFIED_TO_FD_INVENTORY"
        or name == "CROSS_STAGE_RECONCILIATION"
        or name.endswith("_STAGE_INVENTORY")
    ):
        return "HANDOFF"

    # Identity contracts
    if (
        "IDENTITY" in name
        or "ENTITY_ID" in name
        or "ONE_PER_ACTIVE_TEAM" in name
        or "DST_ID_CONVENTION" in name
    ):
        return "IDENTITY"

    # Availability-specific contracts.
    if (
        "AVAILABILITY" in name
        or "INJURY" in name
        or "STARTER_VERIFICATION" in name
    ):
        return "AVAILABILITY"

    # Publication/lifecycle/hash contracts
    if (
        "PUBLICATION" in name
        or "PUBLISH" in name
        or name == "GAV2_TO_PUBLISH_SHA"
        or "MANIFEST" in name
        or "HASH" in name
    ):
        return "PUBLICATION"

    # Forecast/model contracts
    if (
        "FORECAST" in name
        or "PROJECTION" in name
        or "MODEL" in name
    ):
        return "FORECAST"

    # Conservative deterministic fallback:
    # unknown Guardian failures are operationally unclassified,
    # therefore treat as infrastructure until explicitly mapped.
    return "INFRASTRUCTURE"


def classify_failures(results: list[dict]) -> dict:
    failures = []

    for row in results:
        if row.get("status") != "FAIL":
            continue

        check = str(row.get("check", "UNKNOWN"))
        detail = str(row.get("detail", ""))

        failures.append(
            {
                "check": check,
                "category": classify_check(check, detail),
                "detail": detail,
            }
        )

    categories = sorted(
        {
            row["category"]
            for row in failures
        }
    )

    return {
        "contract": CLASSIFICATION_CONTRACT,
        "failure_count": len(failures),
        "categories": categories,
        "failures": failures,
    }
