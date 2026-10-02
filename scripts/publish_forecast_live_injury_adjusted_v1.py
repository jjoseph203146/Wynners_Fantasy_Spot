#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import tempfile

import numpy as np
import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")
PROCESSED = ROOT / "processed"

CORE = (
    PROCESSED
    / "forecast_live_core_v1_predictions.csv"
)
CORE_AUDIT = (
    PROCESSED
    / "forecast_live_core_v1_predictions_audit.json"
)
SHADOW = (
    PROCESSED
    / "forecast_live_injury_shadow_v1.csv"
)
SHADOW_AUDIT = (
    PROCESSED
    / "forecast_live_injury_shadow_v1_audit.json"
)

OUTPUT = (
    PROCESSED
    / "forecast_live_injury_adjusted_v1_predictions.csv"
)
OUTPUT_AUDIT = (
    PROCESSED
    / "forecast_live_injury_adjusted_v1_predictions_audit.json"
)

TOLERANCE = 1e-10
MAX_DELTA = 3.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        frame.to_csv(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> None:
    print("=" * 88)
    print("WFS LIVE INJURY-ADJUSTED FORECAST PUBLISHER V1")
    print("=" * 88)
    print("CORE_MODIFIED=FALSE")
    print("DATABASE_MODIFIED=FALSE")
    print("LEDGER_MODIFIED=FALSE")
    print("UI_MODIFIED=FALSE")

    required = [
        CORE,
        CORE_AUDIT,
        SHADOW,
        SHADOW_AUDIT,
    ]
    for item in required:
        if not item.is_file():
            fail(f"Missing required input: {item}")

    core_sha = sha256(CORE)
    shadow_sha = sha256(SHADOW)

    core_audit = json.loads(
        CORE_AUDIT.read_text()
    )
    shadow_audit = json.loads(
        SHADOW_AUDIT.read_text()
    )

    if core_audit.get("status") != "PASS":
        fail("CORE audit is not PASS")

    if shadow_audit.get("status") != "PASS":
        fail("Shadow audit is not PASS")

    audited_core_sha = (
        core_audit.get("output", {}).get("sha256")
    )
    if audited_core_sha != core_sha:
        fail("CORE audit SHA mismatch")

    audited_shadow_sha = (
        shadow_audit.get("output", {}).get("sha256")
    )
    if audited_shadow_sha != shadow_sha:
        fail("Shadow audit SHA mismatch")

    shadow_inputs = shadow_audit.get("inputs", {})
    bound_core_sha = shadow_inputs.get(str(CORE))

    if bound_core_sha != core_sha:
        fail(
            "Shadow is stale or bound to a different CORE"
        )

    core = pd.read_csv(
        CORE,
        dtype={"game_id": str},
    )
    shadow = pd.read_csv(
        SHADOW,
        dtype={"game_id": str},
    )

    if core["game_id"].duplicated().any():
        fail("Duplicate CORE game_id")

    if shadow["game_id"].duplicated().any():
        fail("Duplicate shadow game_id")

    ready_ids = set(
        core.loc[
            core["forecast_status"]
            .astype(str)
            .eq("READY_CORE_ONLY"),
            "game_id",
        ]
    )
    shadow_ids = set(shadow["game_id"])

    if shadow_ids != ready_ids:
        fail(
            "Shadow game set does not equal current "
            "READY_CORE_ONLY game set"
        )

    required_shadow = {
        "game_id",
        "has_live_injury_input",
        "injury_margin_delta",
        "injury_total_delta",
        "shadow_pred_margin_residual",
        "shadow_pred_total_residual",
        "shadow_pred_home_margin",
        "shadow_pred_total_points",
        "shadow_pred_home_points",
        "shadow_pred_away_points",
        "shadow_pred_winner",
        "core_pred_home_margin",
        "core_pred_total_points",
        "core_pred_home_points",
        "core_pred_away_points",
    }

    missing = sorted(
        required_shadow - set(shadow.columns)
    )
    if missing:
        fail(
            "Shadow columns missing: " + repr(missing)
        )

    joined = core.merge(
        shadow[
            sorted(required_shadow)
        ],
        on="game_id",
        how="left",
        validate="one_to_one",
    )

    ready_mask = (
        joined["forecast_status"]
        .astype(str)
        .eq("READY_CORE_ONLY")
    )
    active_mask = (
        ready_mask
        & pd.to_numeric(
            joined["has_live_injury_input"],
            errors="coerce",
        ).fillna(0).eq(1)
    )
    zero_mask = ready_mask & ~active_mask
    pending_mask = ~ready_mask

    active_count = int(active_mask.sum())
    zero_count = int(zero_mask.sum())
    pending_count = int(pending_mask.sum())
    ready_count = int(ready_mask.sum())
    joined_count = len(joined)

    if active_count + zero_count != ready_count:
        fail(
            "Injury publication ready partition mismatch: "
            f"active={active_count}, zero={zero_count}, "
            f"ready={ready_count}"
        )

    if ready_count != len(ready_ids):
        fail(
            "Injury publication READY_CORE_ONLY count does not "
            "equal authoritative CORE ready game-ID population"
        )

    if active_count + zero_count + pending_count != joined_count:
        fail(
            "Injury publication total partition mismatch: "
            f"active={active_count}, zero={zero_count}, "
            f"pending={pending_count}, rows={joined_count}"
        )

    if ready_count <= 0:
        fail(
            "Injury publication has no READY_CORE_ONLY games"
        )

    delta_columns = [
        "injury_margin_delta",
        "injury_total_delta",
    ]
    delta_max = (
        joined.loc[active_mask, delta_columns]
        .abs()
        .to_numpy(dtype=float)
        .max(initial=0.0)
    )

    if not np.isfinite(delta_max):
        fail("Non-finite injury delta")

    if delta_max > MAX_DELTA:
        fail(
            f"Injury delta exceeds bound: {delta_max}"
        )

    core_values = {
        "pred_margin_residual":
            "shadow_pred_margin_residual",
        "pred_total_residual":
            "shadow_pred_total_residual",
        "pred_home_margin":
            "shadow_pred_home_margin",
        "pred_total_points":
            "shadow_pred_total_points",
        "pred_home_points":
            "shadow_pred_home_points",
        "pred_away_points":
            "shadow_pred_away_points",
        "pred_winner":
            "shadow_pred_winner",
    }

    for target in core_values:
        joined[
            "core_" + target
        ] = joined[target]

    for target, source in core_values.items():
        joined.loc[
            active_mask,
            target,
        ] = joined.loc[
            active_mask,
            source,
        ].to_numpy()

    # Residuals must reconstruct from the exact market
    # values carried by the companion publication.
    joined.loc[
        active_mask,
        "pred_margin_residual",
    ] = (
        pd.to_numeric(
            joined.loc[
                active_mask,
                "pred_home_margin",
            ],
            errors="raise",
        )
        -
        pd.to_numeric(
            joined.loc[
                active_mask,
                "market_home_spread_raw",
            ],
            errors="raise",
        )
    )

    joined.loc[
        active_mask,
        "pred_total_residual",
    ] = (
        pd.to_numeric(
            joined.loc[
                active_mask,
                "pred_total_points",
            ],
            errors="raise",
        )
        -
        pd.to_numeric(
            joined.loc[
                active_mask,
                "market_total",
            ],
            errors="raise",
        )
    )

    reconstructed_margin = (
        pd.to_numeric(
            joined.loc[
                ready_mask,
                "market_home_spread_raw",
            ],
            errors="raise",
        )
        +
        pd.to_numeric(
            joined.loc[
                ready_mask,
                "pred_margin_residual",
            ],
            errors="raise",
        )
    )

    reconstructed_total = (
        pd.to_numeric(
            joined.loc[
                ready_mask,
                "market_total",
            ],
            errors="raise",
        )
        +
        pd.to_numeric(
            joined.loc[
                ready_mask,
                "pred_total_residual",
            ],
            errors="raise",
        )
    )

    margin_reconstruction_max = float(
        (
            reconstructed_margin
            -
            pd.to_numeric(
                joined.loc[
                    ready_mask,
                    "pred_home_margin",
                ],
                errors="raise",
            )
        )
        .abs()
        .max()
    )

    total_reconstruction_max = float(
        (
            reconstructed_total
            -
            pd.to_numeric(
                joined.loc[
                    ready_mask,
                    "pred_total_points",
                ],
                errors="raise",
            )
        )
        .abs()
        .max()
    )

    if margin_reconstruction_max > TOLERANCE:
        fail(
            "Margin reconstruction failure: "
            f"{margin_reconstruction_max}"
        )

    if total_reconstruction_max > TOLERANCE:
        fail(
            "Total reconstruction failure: "
            f"{total_reconstruction_max}"
        )

    joined["injury_adjustment_applied"] = 0
    joined.loc[
        active_mask,
        "injury_adjustment_applied",
    ] = 1

    joined["core_forecast_status"] = (
        joined["forecast_status"]
    )

    joined.loc[
        active_mask,
        "model",
    ] = "WFS_CORE_PLUS_INJURY_COUNTERFACTUAL"

    joined.loc[
        active_mask,
        "forecast_variant",
    ] = "LIVE_INJURY_COUNTERFACTUAL_V1"

    joined.loc[
        active_mask,
        "impact_status",
    ] = "LIVE_IMPACT_V1_COUNTERFACTUAL"

    joined.loc[
        active_mask,
        "replacement_status",
    ] = "LIVE_REPLACEMENT_V1_COUNTERFACTUAL"

    joined.loc[
        active_mask,
        "forecast_status",
    ] = "READY_INJURY_ADJUSTED"

    unchanged_columns = [
        "pred_margin_residual",
        "pred_total_residual",
        "pred_home_margin",
        "pred_total_points",
        "pred_home_points",
        "pred_away_points",
    ]

    zero_difference = (
        joined.loc[
            zero_mask,
            unchanged_columns,
        ].to_numpy(dtype=float)
        -
        joined.loc[
            zero_mask,
            [
                "core_" + c
                for c in unchanged_columns
            ],
        ].to_numpy(dtype=float)
    )

    zero_max = (
        np.abs(zero_difference)
        .max(initial=0.0)
    )

    if zero_max > TOLERANCE:
        fail(
            "Zero-input games changed: "
            f"{zero_max}"
        )

    pending_difference = (
        joined.loc[
            pending_mask,
            unchanged_columns,
        ].fillna(0.0).to_numpy(dtype=float)
        -
        joined.loc[
            pending_mask,
            [
                "core_" + c
                for c in unchanged_columns
            ],
        ].fillna(0.0).to_numpy(dtype=float)
    )

    pending_max = (
        np.abs(pending_difference)
        .max(initial=0.0)
    )

    if pending_max > TOLERANCE:
        fail(
            "Market-pending games changed: "
            f"{pending_max}"
        )

    drop_shadow = [
        c
        for c in joined.columns
        if c.startswith("shadow_pred_")
    ]
    joined = joined.drop(columns=drop_shadow)

    joined = joined.sort_values(
        ["season", "week", "game_id"]
    ).reset_index(drop=True)

    atomic_csv(joined, OUTPUT)
    output_sha = sha256(OUTPUT)

    audit = {
        "version":
            "WFS_LIVE_INJURY_ADJUSTED_PUBLISHER_V1",
        "generated_at_utc":
            datetime.now(timezone.utc).isoformat(),
        "status":
            "PASS",
        "production_role":
            "COMPANION_PUBLICATION_NOT_YET_CONSUMED",
        "core_modified":
            False,
        "database_modified":
            False,
        "ledger_modified":
            False,
        "ui_modified":
            False,
        "rows":
            len(joined),
        "ready_core_only_games":
            int(zero_mask.sum()),
        "ready_injury_adjusted_games":
            int(active_mask.sum()),
        "market_pending_games":
            int(pending_mask.sum()),
        "zero_input_max_change":
            float(zero_max),
        "market_pending_max_change":
            float(pending_max),
        "max_abs_injury_delta":
            float(delta_max),
        "margin_reconstruction_max":
            margin_reconstruction_max,
        "total_reconstruction_max":
            total_reconstruction_max,
        "bounds": {
            "numeric_tolerance": TOLERANCE,
            "max_injury_delta": MAX_DELTA,
        },
        "inputs": {
            str(CORE): core_sha,
            str(CORE_AUDIT): sha256(CORE_AUDIT),
            str(SHADOW): shadow_sha,
            str(SHADOW_AUDIT): sha256(SHADOW_AUDIT),
        },
        "output": {
            "path": str(OUTPUT),
            "sha256": output_sha,
        },
    }

    atomic_json(audit, OUTPUT_AUDIT)

    hou = joined[
        joined["game_id"].eq(
            "2026_02_CIN_HOU"
        )
    ]

    print("ROWS=" + str(len(joined)))
    print(
        "READY_INJURY_ADJUSTED="
        + str(int(active_mask.sum()))
    )
    print(
        "READY_CORE_ONLY="
        + str(int(zero_mask.sum()))
    )
    print(
        "MARKET_PENDING="
        + str(int(pending_mask.sum()))
    )
    print(
        "ZERO_INPUT_MAX_CHANGE="
        + f"{zero_max:.18e}"
    )
    print(
        "PENDING_MAX_CHANGE="
        + f"{pending_max:.18e}"
    )
    print(
        "MAX_ABS_INJURY_DELTA="
        + f"{delta_max:.6f}"
    )
    print(
        "MARGIN_RECONSTRUCTION_MAX="
        + f"{margin_reconstruction_max:.18e}"
    )
    print(
        "TOTAL_RECONSTRUCTION_MAX="
        + f"{total_reconstruction_max:.18e}"
    )
    if not hou.empty:
        print(
            "HOU_CIN_PUBLICATION="
            + repr(hou.iloc[0].to_dict())
        )
    print("OUTPUT_SHA256=" + output_sha)
    print("INJURY_COMPANION_PUBLICATION=PASS")
    print("PUBLIC_CONSUMERS_MODIFIED=FALSE")


if __name__ == "__main__":
    main()
