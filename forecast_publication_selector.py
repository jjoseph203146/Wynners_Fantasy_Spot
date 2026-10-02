from __future__ import annotations

from pathlib import Path
import hashlib
import json


ROOT = Path(__file__).resolve().parent

CORE = (
    ROOT
    / "processed"
    / "forecast_live_core_v1_predictions.csv"
)
COMPANION = (
    ROOT
    / "processed"
    / "forecast_live_injury_adjusted_v1_predictions.csv"
)
COMPANION_AUDIT = (
    ROOT
    / "processed"
    / "forecast_live_injury_adjusted_v1_predictions_audit.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def select_forecast_path() -> Path:
    """
    Return the injury-adjusted companion only when its audit
    is PASS and binds to the exact current CORE artifact.

    Any missing, stale, corrupt, or malformed companion falls
    back to the immutable CORE publication.
    """
    if not CORE.is_file():
        return CORE

    try:
        if (
            not COMPANION.is_file()
            or not COMPANION_AUDIT.is_file()
        ):
            return CORE

        audit = json.loads(
            COMPANION_AUDIT.read_text()
        )

        if audit.get("status") != "PASS":
            return CORE

        core_sha = _sha256(CORE)
        companion_sha = _sha256(COMPANION)

        bound_core_sha = (
            audit.get("inputs", {}).get(str(CORE))
        )
        audited_output_sha = (
            audit.get("output", {}).get("sha256")
        )

        if bound_core_sha != core_sha:
            return CORE

        if audited_output_sha != companion_sha:
            return CORE

        rows = int(audit["rows"])
        ready_injury = int(
            audit["ready_injury_adjusted_games"]
        )
        ready_core = int(
            audit["ready_core_only_games"]
        )
        pending = int(
            audit["market_pending_games"]
        )

        if rows <= 0:
            return CORE

        if (
            ready_injury < 0
            or ready_core < 0
            or pending < 0
        ):
            return CORE

        if ready_injury + ready_core + pending != rows:
            return CORE

        if ready_injury + ready_core <= 0:
            return CORE

        bounds = audit["bounds"]
        tolerance = float(
            bounds["numeric_tolerance"]
        )
        max_injury_delta_bound = float(
            bounds["max_injury_delta"]
        )

        zero_input_max_change = float(
            audit["zero_input_max_change"]
        )
        market_pending_max_change = float(
            audit["market_pending_max_change"]
        )
        max_abs_injury_delta = float(
            audit["max_abs_injury_delta"]
        )
        margin_reconstruction_max = float(
            audit["margin_reconstruction_max"]
        )
        total_reconstruction_max = float(
            audit["total_reconstruction_max"]
        )

        numeric_values = (
            tolerance,
            max_injury_delta_bound,
            zero_input_max_change,
            market_pending_max_change,
            max_abs_injury_delta,
            margin_reconstruction_max,
            total_reconstruction_max,
        )

        if not all(
            value >= 0.0
            and value < float("inf")
            for value in numeric_values
        ):
            return CORE

        if zero_input_max_change > tolerance:
            return CORE

        if market_pending_max_change > tolerance:
            return CORE

        if max_abs_injury_delta > max_injury_delta_bound:
            return CORE

        if margin_reconstruction_max > tolerance:
            return CORE

        if total_reconstruction_max > tolerance:
            return CORE

        return COMPANION

    except Exception:
        return CORE
