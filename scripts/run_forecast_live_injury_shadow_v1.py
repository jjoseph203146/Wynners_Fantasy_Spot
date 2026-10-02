#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")

EXP004_SCRIPT = (
    ROOT
    / "backups"
    / "forecast_v1_exp004_frozen"
    / "run_forecast_v1_experiment_004.py"
)

TRAINING = (
    ROOT
    / "processed"
    / "forecast_v1_team_game_training_impact_replacement_v1.csv"
)

LIVE_CORE = (
    ROOT
    / "backups"
    / "forecast_live_core_v1_frozen"
    / "forecast_live_core_v1.csv"
)

LIVE_IMPACT = (
    ROOT
    / "processed"
    / "forecast_live_team_impact_v1.csv"
)

LIVE_REPLACEMENT = (
    ROOT
    / "processed"
    / "forecast_live_team_replacement_quality_v1.csv"
)

PUBLIC_FORECAST = (
    ROOT
    / "processed"
    / "forecast_live_core_v1_predictions.csv"
)

OUTPUT = (
    ROOT
    / "processed"
    / "forecast_live_injury_shadow_v1.csv"
)

AUDIT = (
    ROOT
    / "processed"
    / "forecast_live_injury_shadow_v1_audit.json"
)

ALPHA = 3000.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        dir=path.parent,
        delete=False,
    ) as handle:
        temp = Path(handle.name)
        frame.to_csv(handle, index=False)

    os.replace(temp, path)


def atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        dir=path.parent,
        delete=False,
    ) as handle:
        temp = Path(handle.name)
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
        )

    os.replace(temp, path)


def load_exp004():
    spec = importlib.util.spec_from_file_location(
        "wfs_exp004_frozen",
        EXP004_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Unable to load frozen Experiment 004"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    print("=" * 88)
    print("WFS LIVE INJURY FORECAST — SHADOW V1")
    print("=" * 88)
    print("PUBLIC_FORECAST_MODIFIED=FALSE")
    print("DATABASE_MODIFIED=FALSE")
    print("OPTIMIZER_MODIFIED=FALSE")

    required = [
        EXP004_SCRIPT,
        TRAINING,
        LIVE_CORE,
        LIVE_IMPACT,
        LIVE_REPLACEMENT,
        PUBLIC_FORECAST,
    ]

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    if missing:
        raise RuntimeError(
            "Missing required artifacts: "
            + repr(missing)
        )

    exp = load_exp004()

    training = pd.read_csv(
        TRAINING,
        low_memory=False,
    )

    if len(training) != 1710:
        raise RuntimeError(
            "Frozen training row count changed"
        )

    historical = exp.build_home_game_frame(
        training
    )

    (
        core_features,
        impact_features,
        replacement_features,
        core_impact_features,
        core_impact_replacement_features,
    ) = exp.build_feature_lists(
        historical
    )

    if len(core_features) != 76:
        raise RuntimeError(
            f"Expected 76 CORE features; "
            f"found {len(core_features)}"
        )

    if len(impact_features) != 60:
        raise RuntimeError(
            f"Expected 60 impact features; "
            f"found {len(impact_features)}"
        )

    if len(replacement_features) != 33:
        raise RuntimeError(
            f"Expected 33 replacement features; "
            f"found {len(replacement_features)}"
        )

    impact_margin_model = exp.fit_final(
        historical,
        core_impact_features,
        "margin_residual",
        ALPHA,
    )

    impact_total_model = exp.fit_final(
        historical,
        core_impact_features,
        "total_residual",
        ALPHA,
    )

    replacement_margin_model = exp.fit_final(
        historical,
        core_impact_replacement_features,
        "margin_residual",
        ALPHA,
    )

    live = pd.read_csv(
        LIVE_CORE,
        low_memory=False,
    )

    impact = pd.read_csv(
        LIVE_IMPACT,
        low_memory=False,
    )

    replacement = pd.read_csv(
        LIVE_REPLACEMENT,
        low_memory=False,
    )

    public = pd.read_csv(
        PUBLIC_FORECAST,
        low_memory=False,
    )

    population_keys = ["game_id", "team"]

    for label, frame_population in (
        ("live", live),
        ("impact", impact),
        ("replacement", replacement),
    ):
        missing_keys = [
            key
            for key in population_keys
            if key not in frame_population.columns
        ]
        if missing_keys:
            raise RuntimeError(
                f"{label} population missing identity keys: "
                + repr(missing_keys)
            )

        if frame_population.duplicated(
            population_keys
        ).any():
            raise RuntimeError(
                f"{label} population has duplicate "
                "(game_id, team) keys"
            )

    if live.empty:
        raise RuntimeError(
            "Current live population is empty"
        )

    live_keys = set(
        map(
            tuple,
            live[population_keys]
            .astype(str)
            .to_numpy(),
        )
    )
    impact_keys = set(
        map(
            tuple,
            impact[population_keys]
            .astype(str)
            .to_numpy(),
        )
    )
    replacement_keys = set(
        map(
            tuple,
            replacement[population_keys]
            .astype(str)
            .to_numpy(),
        )
    )

    if impact_keys != live_keys:
        raise RuntimeError(
            "Live impact population does not exactly match "
            "live CORE (game_id, team) population"
        )

    if replacement_keys != live_keys:
        raise RuntimeError(
            "Live replacement population does not exactly match "
            "live CORE (game_id, team) population"
        )

    team_impact_features = [
        feature
        for feature in impact_features
        if feature.startswith("team_impact_")
    ]

    opp_impact_features = [
        feature
        for feature in impact_features
        if feature.startswith("opp_impact_")
    ]

    team_payload = impact[
        ["game_id", "team"]
    ].copy()

    for feature in team_impact_features:
        raw = feature.removeprefix(
            "team_impact_"
        )

        if raw not in impact.columns:
            raise RuntimeError(
                f"Live impact missing raw feature: {raw}"
            )

        team_payload[feature] = impact[raw]

    opponent_payload = impact[
        ["game_id", "team"]
    ].copy()

    opponent_payload = opponent_payload.rename(
        columns={"team": "opponent_team"}
    )

    for feature in opp_impact_features:
        raw = feature.removeprefix(
            "opp_impact_"
        )

        if raw not in impact.columns:
            raise RuntimeError(
                f"Live opponent impact missing: {raw}"
            )

        opponent_payload[feature] = impact[raw]

    replacement_missing = sorted(
        set(replacement_features)
        - set(replacement.columns)
    )

    if replacement_missing:
        raise RuntimeError(
            "Live replacement features missing: "
            + repr(replacement_missing)
        )

    replacement_payload = replacement[
        [
            "game_id",
            "team",
            *replacement_features,
        ]
    ].copy()

    frame = live.merge(
        team_payload,
        on=["game_id", "team"],
        how="left",
        validate="one_to_one",
    )

    frame = frame.merge(
        opponent_payload,
        on=["game_id", "opponent_team"],
        how="left",
        validate="one_to_one",
    )

    frame = frame.merge(
        replacement_payload,
        on=["game_id", "team"],
        how="left",
        validate="one_to_one",
    )

    required_features = (
        core_impact_replacement_features
    )

    missing_live = sorted(
        set(required_features)
        - set(frame.columns)
    )

    if missing_live:
        raise RuntimeError(
            "Live modeling frame missing features: "
            + repr(missing_live)
        )

    home = frame[
        pd.to_numeric(
            frame["is_home"],
            errors="coerce",
        ).eq(1)
    ].copy()

    live_game_ids = set(
        live["game_id"].astype(str).tolist()
    )
    home_game_ids = set(
        home["game_id"].astype(str).tolist()
    )

    if home["game_id"].astype(str).duplicated().any():
        raise RuntimeError(
            "Current live population does not contain exactly "
            "one home row per game"
        )

    if home_game_ids != live_game_ids:
        raise RuntimeError(
            "Home game IDs do not exactly equal current "
            "live game-ID population"
        )

    if len(home) != len(live_game_ids):
        raise RuntimeError(
            "Current live population home-row count does not "
            "equal current live game count"
        )

    ready = home[
        pd.to_numeric(
            home["forecast_market_ready"],
            errors="coerce",
        ).fillna(0).astype(int).eq(1)
    ].copy()

    public_ready_ids = set(
        public.loc[
            public["forecast_status"]
            .astype(str)
            .eq("READY_CORE_ONLY"),
            "game_id",
        ]
        .astype(str)
        .tolist()
    )

    ready = ready[
        ready["game_id"]
        .astype(str)
        .isin(public_ready_ids)
    ].copy()

    print(
        "PUBLIC_READY_GAME_IDS="
        + str(len(public_ready_ids))
    )
    print(
        "SHADOW_COMPARISON_ROWS="
        + str(len(ready))
    )

    if ready.empty:
        raise RuntimeError(
            "No published market-ready games"
        )

    # Counterfactual injury isolation:
    # use the same fitted models for both predictions.
    # The only difference is live versus neutral injury inputs.
    neutral = ready.copy()

    for feature in impact_features:
        neutral[feature] = 0.0

    for feature in replacement_features:
        if feature.startswith("replacement_delta_"):
            neutral[feature] = np.nan
        else:
            neutral[feature] = 0.0

    live_margin_residual = (
        replacement_margin_model.predict(
            ready[
                core_impact_replacement_features
            ]
        )
    )
    neutral_margin_residual = (
        replacement_margin_model.predict(
            neutral[
                core_impact_replacement_features
            ]
        )
    )
    live_total_residual = (
        impact_total_model.predict(
            ready[
                core_impact_features
            ]
        )
    )
    neutral_total_residual = (
        impact_total_model.predict(
            neutral[
                core_impact_features
            ]
        )
    )

    injury_margin_delta = (
        live_margin_residual
        - neutral_margin_residual
    )
    injury_total_delta = (
        live_total_residual
        - neutral_total_residual
    )

    impact_input = (
        ready[impact_features]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0.0)
        .abs()
        .sum(axis=1)
        .gt(0)
        .to_numpy()
    )
    replacement_input = (
        pd.to_numeric(
            ready["replacement_injury_count"],
            errors="coerce",
        )
        .fillna(0.0)
        .gt(0)
        .to_numpy()
    )
    has_live_injury_input = (
        impact_input
        | replacement_input
    )

    zero_input = ~has_live_injury_input
    tolerance = 1e-10

    if (
        np.abs(
            injury_margin_delta[zero_input]
        ).max(initial=0.0)
        > tolerance
    ):
        raise RuntimeError(
            "Zero-input margin counterfactual changed"
        )

    if (
        np.abs(
            injury_total_delta[zero_input]
        ).max(initial=0.0)
        > tolerance
    ):
        raise RuntimeError(
            "Zero-input total counterfactual changed"
        )

    out = ready[
        [
            "game_id",
            "season",
            "week",
            "game_date",
            "team",
            "opponent_team",
            "market_home_spread_raw",
            "market_total",
        ]
    ].copy()

    out = out.rename(
        columns={
            "team": "home_team",
            "opponent_team": "away_team",
        }
    )

    baseline_columns = [
        "game_id",
        "pred_home_margin",
        "pred_total_points",
        "pred_home_points",
        "pred_away_points",
        "pred_winner",
        "forecast_status",
    ]

    out = out.merge(
        public[baseline_columns],
        on="game_id",
        how="left",
        validate="one_to_one",
    )

    out = out.rename(
        columns={
            "pred_home_margin":
                "core_pred_home_margin",
            "pred_total_points":
                "core_pred_total_points",
            "pred_home_points":
                "core_pred_home_points",
            "pred_away_points":
                "core_pred_away_points",
            "pred_winner":
                "core_pred_winner",
        }
    )

    # Bind all displayed market fields to the exact public
    # CORE artifact. The live modeling frame may contain a
    # newer market observation than the published CORE row.
    public_market = (
        public[
            [
                "game_id",
                "market_home_spread_raw",
                "market_total",
            ]
        ]
        .drop_duplicates("game_id")
        .set_index("game_id")
    )

    out["market_home_spread_raw"] = (
        out["game_id"].map(
            public_market[
                "market_home_spread_raw"
            ]
        )
    )
    out["market_total"] = (
        out["game_id"].map(
            public_market["market_total"]
        )
    )

    if out[
        [
            "market_home_spread_raw",
            "market_total",
        ]
    ].isna().any().any():
        raise RuntimeError(
            "Public CORE market binding incomplete"
        )

    out["has_live_injury_input"] = (
        has_live_injury_input.astype(int)
    )
    out["injury_margin_delta"] = (
        injury_margin_delta
    )
    out["injury_total_delta"] = (
        injury_total_delta
    )

    out["shadow_pred_home_margin"] = (
        out["core_pred_home_margin"]
        + out["injury_margin_delta"]
    )
    out["shadow_pred_total_points"] = (
        out["core_pred_total_points"]
        + out["injury_total_delta"]
    )

    out["shadow_pred_margin_residual"] = (
        out["shadow_pred_home_margin"]
        - pd.to_numeric(
            out["market_home_spread_raw"],
            errors="raise",
        )
    )
    out["shadow_pred_total_residual"] = (
        out["shadow_pred_total_points"]
        - pd.to_numeric(
            out["market_total"],
            errors="raise",
        )
    )

    out["shadow_pred_home_points"] = (
        out["shadow_pred_total_points"]
        + out["shadow_pred_home_margin"]
    ) / 2.0
    out["shadow_pred_away_points"] = (
        out["shadow_pred_total_points"]
        - out["shadow_pred_home_margin"]
    ) / 2.0

    out["shadow_pred_winner"] = np.where(
        out["shadow_pred_home_margin"] > 0,
        out["home_team"],
        np.where(
            out["shadow_pred_home_margin"] < 0,
            out["away_team"],
            "TIE",
        ),
    )

    out["home_points_change"] = (
        out["shadow_pred_home_points"]
        - out["core_pred_home_points"]
    )
    out["away_points_change"] = (
        out["shadow_pred_away_points"]
        - out["core_pred_away_points"]
    )
    out["margin_change"] = (
        out["injury_margin_delta"]
    )
    out["total_change"] = (
        out["injury_total_delta"]
    )
    out["winner_changed"] = (
        out["shadow_pred_winner"]
        != out["core_pred_winner"]
    ).astype(int)

    zero_output = out[
        out["has_live_injury_input"].eq(0)
    ]
    zero_change_max = (
        zero_output[
            [
                "home_points_change",
                "away_points_change",
                "margin_change",
                "total_change",
            ]
        ]
        .abs()
        .to_numpy(dtype=float)
        .max(initial=0.0)
    )

    if zero_change_max > tolerance:
        raise RuntimeError(
            "Zero-input public baseline changed: "
            f"{zero_change_max}"
        )

    print(
        "GAMES_WITH_LIVE_INJURY_INPUT="
        + str(int(out["has_live_injury_input"].sum()))
    )
    print(
        "ZERO_INPUT_GAMES="
        + str(int(
            out["has_live_injury_input"].eq(0).sum()
        ))
    )
    print(
        "ZERO_INPUT_MAX_CHANGE="
        + f"{zero_change_max:.12f}"
    )

    out["model"] = (
        "WFS_INJURY_COUNTERFACTUAL_SHADOW"
    )

    out["forecast_variant"] = (
        "LIVE_INJURY_COUNTERFACTUAL_SHADOW_V1"
    )

    out["impact_status"] = (
        "LIVE_IMPACT_V1_COUNTERFACTUAL"
    )

    out["replacement_status"] = (
        "LIVE_REPLACEMENT_V1_COUNTERFACTUAL"
    )

    out = out.sort_values(
        ["season", "week", "game_id"]
    ).reset_index(drop=True)

    numeric = [
        "shadow_pred_margin_residual",
        "shadow_pred_total_residual",
        "shadow_pred_home_margin",
        "shadow_pred_total_points",
        "shadow_pred_home_points",
        "shadow_pred_away_points",
        "home_points_change",
        "away_points_change",
        "margin_change",
        "total_change",
    ]

    if not np.isfinite(
        out[numeric].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Non-finite shadow predictions"
        )

    atomic_csv(out, OUTPUT)

    audit = {
        "version":
            "WFS_LIVE_INJURY_FORECAST_SHADOW_V1",
        "generated_at_utc":
            datetime.now(timezone.utc).isoformat(),
        "status":
            "PASS",
        "public_forecast_modified":
            False,
        "database_modified":
            False,
        "optimizer_modified":
            False,
        "training_seasons": [
            2023,
            2024,
        ],
        "live_season":
            2026,
        "alpha":
            ALPHA,
        "architecture": {
            "margin":
                "CORE_PLUS_IMPACT_PLUS_REPLACEMENT",
            "total":
                "CORE_PLUS_IMPACT",
        },
        "feature_counts": {
            "core":
                len(core_features),
            "impact":
                len(impact_features),
            "replacement":
                len(replacement_features),
        },
        "ready_games":
            len(out),
        "winner_changes":
            int(out["winner_changed"].sum()),
        "max_abs_home_points_change":
            float(
                out["home_points_change"]
                .abs()
                .max()
            ),
        "max_abs_away_points_change":
            float(
                out["away_points_change"]
                .abs()
                .max()
            ),
        "inputs": {
            str(path):
                sha256(path)
            for path in required
        },
        "output": {
            "path":
                str(OUTPUT),
            "sha256":
                sha256(OUTPUT),
        },
    }

    atomic_json(audit, AUDIT)

    print("SHADOW_ROWS=" + str(len(out)))
    print(
        "WINNER_CHANGES="
        + str(int(out["winner_changed"].sum()))
    )

    hou = out[
        out["game_id"].astype(str).eq(
            "2026_02_CIN_HOU"
        )
    ]

    if not hou.empty:
        print(
            "HOU_CIN_SHADOW="
            + repr(hou.iloc[0].to_dict())
        )

    changed = out[
        (
            out["home_points_change"].abs() >= 0.25
        )
        |
        (
            out["away_points_change"].abs() >= 0.25
        )
        |
        out["winner_changed"].eq(1)
    ]

    print(
        "MATERIAL_CHANGE_ROWS="
        + str(len(changed))
    )

    display = [
        "game_id",
        "core_pred_away_points",
        "core_pred_home_points",
        "core_pred_winner",
        "shadow_pred_away_points",
        "shadow_pred_home_points",
        "shadow_pred_winner",
        "away_points_change",
        "home_points_change",
        "winner_changed",
    ]

    print(
        changed[display]
        .sort_values(
            [
                "winner_changed",
                "game_id",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .to_string(index=False)
    )

    print("LIVE_INJURY_SHADOW_STATUS=PASS")


if __name__ == "__main__":
    main()
