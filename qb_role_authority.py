from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd


CURRENT_CONTRACT = "WFS_QB_ROLE_CURRENT_V1"
KICKOFF_CONTRACT = "WFS_QB_ROLE_KICKOFF_V1"

CURRENT_NAME = "current_qb_role_authority.parquet"
CURRENT_MANIFEST_NAME = "current_qb_role_authority_manifest.json"

KICKOFF_NAME = "qb_role_kickoff.parquet"
KICKOFF_MANIFEST_NAME = "qb_role_kickoff_manifest.json"

ROLE_COLUMNS = [
    "game_id",
    "team",
    "player_id",
    "reconciliation_role",
    "primary_qb_id",
]

ALLOWED_QB_ROLES = {
    "PRIMARY_QB",
    "CONTINGENCY_QB",
    "UNAVAILABLE",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json_write(obj: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())

        os.replace(tmp_name, path)

    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_parquet_write(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(fd)

    try:
        df.to_parquet(tmp_name, index=False)
        os.replace(tmp_name, path)

    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def normalize_authority(df: pd.DataFrame) -> pd.DataFrame:
    missing = [
        col for col in ROLE_COLUMNS
        if col not in df.columns
    ]
    if missing:
        raise RuntimeError(
            f"QB authority missing columns: {missing}"
        )

    out = df[ROLE_COLUMNS].copy()

    for col in ROLE_COLUMNS:
        out[col] = (
            out[col]
            .astype("string")
            .fillna("")
            .str.strip()
        )

    out["team"] = out["team"].str.upper()
    out["reconciliation_role"] = (
        out["reconciliation_role"].str.upper()
    )

    if (
        out[ROLE_COLUMNS]
        .eq("")
        .any(axis=None)
    ):
        raise RuntimeError(
            "QB authority contains blank required values"
        )

    bad_roles = sorted(
        set(out["reconciliation_role"])
        - ALLOWED_QB_ROLES
    )
    if bad_roles:
        raise RuntimeError(
            f"QB authority contains invalid roles: {bad_roles}"
        )

    if out.duplicated(
        subset=["game_id", "team", "player_id"],
        keep=False,
    ).any():
        raise RuntimeError(
            "QB authority contains duplicate player identity"
        )

    return out


def validate_authority(
    df: pd.DataFrame,
    *,
    forecast_df: pd.DataFrame | None = None,
    game_id: str | None = None,
) -> pd.DataFrame:
    out = normalize_authority(df)

    if game_id is not None:
        game_id = str(game_id).strip()
        out = out[
            out["game_id"].eq(game_id)
        ].copy()

        if out.empty:
            raise RuntimeError(
                f"{game_id}: QB authority has no rows"
            )

    for (gid, team), group in out.groupby(
        ["game_id", "team"],
        sort=True,
    ):
        primary = group[
            group["reconciliation_role"].eq(
                "PRIMARY_QB"
            )
        ]

        if len(primary) != 1:
            raise RuntimeError(
                f"{gid}/{team}: expected exactly one "
                f"PRIMARY_QB, found {len(primary)}"
            )

        primary_id = str(
            primary.iloc[0]["player_id"]
        )

        declared_ids = set(
            group["primary_qb_id"].astype(str)
        )

        if declared_ids != {primary_id}:
            raise RuntimeError(
                f"{gid}/{team}: primary_qb_id does not "
                "match authoritative PRIMARY_QB"
            )

    if forecast_df is not None:
        required = {
            "game_id",
            "team",
            "player_id",
            "position",
            "entity_type",
        }
        missing = sorted(
            required - set(forecast_df.columns)
        )
        if missing:
            raise RuntimeError(
                "Forecast missing identity columns: "
                f"{missing}"
            )

        forecast = forecast_df.copy()
        for col in [
            "game_id",
            "team",
            "player_id",
            "position",
            "entity_type",
        ]:
            forecast[col] = (
                forecast[col]
                .astype("string")
                .fillna("")
                .str.strip()
            )

        forecast["team"] = (
            forecast["team"].str.upper()
        )
        forecast["position"] = (
            forecast["position"].str.upper()
        )

        forecast_qb = forecast[
            forecast["position"].eq("QB")
            & forecast["entity_type"].eq(
                "OFFENSE_PLAYER"
            )
        ]

        expected_teams = set(
            zip(
                forecast_qb["game_id"],
                forecast_qb["team"],
            )
        )
        authority_teams = set(
            zip(
                out["game_id"],
                out["team"],
            )
        )

        if authority_teams != expected_teams:
            missing_teams = sorted(
                expected_teams - authority_teams
            )
            extra_teams = sorted(
                authority_teams - expected_teams
            )
            raise RuntimeError(
                "QB authority team coverage mismatch: "
                f"missing={missing_teams}, "
                f"extra={extra_teams}"
            )

        primary_rows = out[
            out["reconciliation_role"].eq(
                "PRIMARY_QB"
            )
        ]

        for row in primary_rows.itertuples(
            index=False
        ):
            matches = forecast_qb[
                forecast_qb["game_id"].eq(
                    row.game_id
                )
                & forecast_qb["team"].eq(
                    row.team
                )
                & forecast_qb["player_id"].eq(
                    row.player_id
                )
            ]

            if len(matches) != 1:
                raise RuntimeError(
                    f"{row.game_id}/{row.team}: "
                    "PRIMARY_QB is not exactly present "
                    "in forecast rows: "
                    f"{row.player_id}"
                )

    return out.reset_index(drop=True)


def build_refreshable_authority(
    validated_v3: pd.DataFrame,
    forecast_df: pd.DataFrame,
    refreshable_game_ids: set[str],
) -> pd.DataFrame:
    required = {
        "game_id",
        "team",
        "player_id",
        "position",
        "reconciliation_role",
        "primary_qb_id",
    }

    missing = sorted(
        required - set(validated_v3.columns)
    )
    if missing:
        raise RuntimeError(
            f"Validated V3 missing columns: {missing}"
        )

    game_ids = {
        str(value).strip()
        for value in refreshable_game_ids
        if str(value).strip()
    }

    if not game_ids:
        return pd.DataFrame(
            columns=ROLE_COLUMNS
        )

    v3 = validated_v3.copy()

    for col in required:
        v3[col] = (
            v3[col]
            .astype("string")
            .fillna("")
            .str.strip()
        )

    v3["team"] = v3["team"].str.upper()
    v3["position"] = v3["position"].str.upper()
    v3["reconciliation_role"] = (
        v3["reconciliation_role"].str.upper()
    )

    authority = v3[
        v3["game_id"].isin(game_ids)
        & v3["position"].eq("QB")
        & v3["reconciliation_role"].isin(
            ALLOWED_QB_ROLES
        )
    ][ROLE_COLUMNS].copy()

    forecast = forecast_df[
        forecast_df["game_id"]
        .astype("string")
        .fillna("")
        .str.strip()
        .isin(game_ids)
    ].copy()

    if forecast.empty:
        raise RuntimeError(
            "Refreshable QB authority has no "
            "matching forecast rows"
        )

    actual_games = set(
        authority["game_id"].astype(str)
    )

    if actual_games != game_ids:
        raise RuntimeError(
            "Refreshable QB authority game "
            "coverage mismatch: "
            f"missing={sorted(game_ids - actual_games)}, "
            f"extra={sorted(actual_games - game_ids)}"
        )

    return validate_authority(
        authority,
        forecast_df=forecast,
    )


def authority_history_dir(
    data_root: Path,
) -> Path:
    return (
        Path(data_root)
        / "qb_role_authority_history"
    )


def authority_history_paths(
    data_root: Path,
    forecast_sha256: str,
) -> tuple[Path, Path]:
    sha = str(forecast_sha256).strip().lower()

    if (
        len(sha) != 64
        or any(
            ch not in "0123456789abcdef"
            for ch in sha
        )
    ):
        raise RuntimeError(
            "Invalid forecast SHA256 for QB authority"
        )

    root = authority_history_dir(data_root) / sha

    return (
        root / "qb_role_authority.parquet",
        root / "manifest.json",
    )


def write_bound_authority(
    authority: pd.DataFrame,
    *,
    forecast_df: pd.DataFrame,
    forecast_sha256: str,
    data_root: Path,
) -> tuple[Path, Path]:
    normalized = validate_authority(
        authority,
        forecast_df=forecast_df,
    )

    parquet_path, manifest_path = (
        authority_history_paths(
            data_root,
            forecast_sha256,
        )
    )

    parquet_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if parquet_path.exists() or manifest_path.exists():
        loaded = load_bound_authority(
            data_root=data_root,
            forecast_sha256=forecast_sha256,
            forecast_df=forecast_df,
        )

        if not loaded.equals(normalized):
            raise RuntimeError(
                "Existing QB authority history "
                "conflicts with requested authority"
            )

        return parquet_path, manifest_path

    atomic_parquet_write(
        normalized,
        parquet_path,
    )

    authority_sha = sha256_file(
        parquet_path
    )

    manifest = {
        "contract": CURRENT_CONTRACT,
        "forecast_sha256": (
            str(forecast_sha256)
            .strip()
            .lower()
        ),
        "authority_sha256": authority_sha,
        "rows": int(len(normalized)),
        "games": int(
            normalized["game_id"].nunique()
        ),
        "immutable": True,
    }

    atomic_json_write(
        manifest,
        manifest_path,
    )

    return parquet_path, manifest_path


def load_bound_authority(
    *,
    data_root: Path,
    forecast_sha256: str,
    forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    parquet_path, manifest_path = (
        authority_history_paths(
            data_root,
            forecast_sha256,
        )
    )

    if (
        not parquet_path.exists()
        or not manifest_path.exists()
    ):
        raise RuntimeError(
            "Bound QB authority is not available "
            f"for forecast {forecast_sha256}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as fh:
        manifest = json.load(fh)

    if manifest.get("contract") != CURRENT_CONTRACT:
        raise RuntimeError(
            "Bound QB authority contract mismatch"
        )

    expected_forecast_sha = (
        str(forecast_sha256)
        .strip()
        .lower()
    )

    if (
        str(
            manifest.get("forecast_sha256")
            or ""
        ).strip().lower()
        != expected_forecast_sha
    ):
        raise RuntimeError(
            "Bound QB authority forecast SHA mismatch"
        )

    actual_authority_sha = sha256_file(
        parquet_path
    )

    if (
        str(
            manifest.get("authority_sha256")
            or ""
        ).strip().lower()
        != actual_authority_sha
    ):
        raise RuntimeError(
            "Bound QB authority artifact SHA mismatch"
        )

    authority = pd.read_parquet(
        parquet_path
    )

    normalized = validate_authority(
        authority,
        forecast_df=forecast_df,
    )

    if int(manifest.get("rows", -1)) != len(
        normalized
    ):
        raise RuntimeError(
            "Bound QB authority row count mismatch"
        )

    if int(manifest.get("games", -1)) != int(
        normalized["game_id"].nunique()
    ):
        raise RuntimeError(
            "Bound QB authority game count mismatch"
        )

    if manifest.get("immutable") is not True:
        raise RuntimeError(
            "Bound QB authority is not immutable"
        )

    return normalized


def kickoff_authority_paths(
    snapshot_root: Path,
    game_id: str,
) -> tuple[Path, Path]:
    root = (
        Path(snapshot_root)
        / str(game_id).strip()
    )
    return (
        root / KICKOFF_NAME,
        root / KICKOFF_MANIFEST_NAME,
    )


def create_kickoff_authority(
    *,
    game_id: str,
    forecast_snapshot_df: pd.DataFrame,
    source_forecast_sha256: str,
    data_root: Path,
    snapshot_root: Path,
) -> tuple[Path, Path]:
    game_id = str(game_id).strip()
    if not game_id:
        raise RuntimeError(
            "Kickoff QB authority requires game_id"
        )

    parquet_path, manifest_path = (
        authority_history_paths(
            data_root,
            source_forecast_sha256,
        )
    )

    if (
        not parquet_path.exists()
        or not manifest_path.exists()
    ):
        raise RuntimeError(
            "Bound QB authority is not available "
            f"for forecast {source_forecast_sha256}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as fh:
        bound_manifest = json.load(fh)

    if (
        bound_manifest.get("contract")
        != CURRENT_CONTRACT
    ):
        raise RuntimeError(
            "Bound QB authority contract mismatch"
        )

    expected_sha = (
        str(source_forecast_sha256)
        .strip()
        .lower()
    )

    if str(
        bound_manifest.get("forecast_sha256")
        or ""
    ).strip().lower() != expected_sha:
        raise RuntimeError(
            "Bound QB authority forecast SHA mismatch"
        )

    if str(
        bound_manifest.get("authority_sha256")
        or ""
    ).strip().lower() != sha256_file(
        parquet_path
    ):
        raise RuntimeError(
            "Bound QB authority artifact SHA mismatch"
        )

    bound = normalize_authority(
        pd.read_parquet(parquet_path)
    )

    if int(
        bound_manifest.get("rows", -1)
    ) != len(bound):
        raise RuntimeError(
            "Bound QB authority row count mismatch"
        )

    if int(
        bound_manifest.get("games", -1)
    ) != int(bound["game_id"].nunique()):
        raise RuntimeError(
            "Bound QB authority game count mismatch"
        )

    if bound_manifest.get("immutable") is not True:
        raise RuntimeError(
            "Bound QB authority is not immutable"
        )

    game_authority = bound[
        bound["game_id"].eq(game_id)
    ].copy()

    game_authority = validate_authority(
        game_authority,
        forecast_df=forecast_snapshot_df,
        game_id=game_id,
    )

    parquet_path, manifest_path = (
        kickoff_authority_paths(
            snapshot_root,
            game_id,
        )
    )

    if parquet_path.exists() or manifest_path.exists():
        return validate_kickoff_authority(
            game_id=game_id,
            forecast_snapshot_df=forecast_snapshot_df,
            source_forecast_sha256=source_forecast_sha256,
            snapshot_root=snapshot_root,
            return_paths=True,
        )

    parquet_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    atomic_parquet_write(
        game_authority,
        parquet_path,
    )

    authority_sha = sha256_file(
        parquet_path
    )

    manifest = {
        "contract": KICKOFF_CONTRACT,
        "game_id": game_id,
        "source_forecast_sha256": (
            str(source_forecast_sha256)
            .strip()
            .lower()
        ),
        "authority_sha256": authority_sha,
        "rows": int(len(game_authority)),
        "teams": int(
            game_authority["team"].nunique()
        ),
        "immutable": True,
    }

    atomic_json_write(
        manifest,
        manifest_path,
    )

    return parquet_path, manifest_path


def validate_kickoff_authority(
    *,
    game_id: str,
    forecast_snapshot_df: pd.DataFrame,
    source_forecast_sha256: str,
    snapshot_root: Path,
    return_paths: bool = False,
):
    game_id = str(game_id).strip()

    parquet_path, manifest_path = (
        kickoff_authority_paths(
            snapshot_root,
            game_id,
        )
    )

    if (
        not parquet_path.exists()
        or not manifest_path.exists()
    ):
        raise RuntimeError(
            f"{game_id}: kickoff QB authority missing"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as fh:
        manifest = json.load(fh)

    if manifest.get("contract") != KICKOFF_CONTRACT:
        raise RuntimeError(
            f"{game_id}: kickoff QB contract mismatch"
        )

    if str(
        manifest.get("game_id") or ""
    ).strip() != game_id:
        raise RuntimeError(
            f"{game_id}: kickoff QB game mismatch"
        )

    expected_source_sha = (
        str(source_forecast_sha256)
        .strip()
        .lower()
    )

    if str(
        manifest.get("source_forecast_sha256")
        or ""
    ).strip().lower() != expected_source_sha:
        raise RuntimeError(
            f"{game_id}: kickoff QB forecast SHA mismatch"
        )

    actual_sha = sha256_file(
        parquet_path
    )

    if str(
        manifest.get("authority_sha256")
        or ""
    ).strip().lower() != actual_sha:
        raise RuntimeError(
            f"{game_id}: kickoff QB artifact SHA mismatch"
        )

    authority = pd.read_parquet(
        parquet_path
    )

    authority = validate_authority(
        authority,
        forecast_df=forecast_snapshot_df,
        game_id=game_id,
    )

    if int(manifest.get("rows", -1)) != len(
        authority
    ):
        raise RuntimeError(
            f"{game_id}: kickoff QB row count mismatch"
        )

    if int(manifest.get("teams", -1)) != int(
        authority["team"].nunique()
    ):
        raise RuntimeError(
            f"{game_id}: kickoff QB team count mismatch"
        )

    if manifest.get("immutable") is not True:
        raise RuntimeError(
            f"{game_id}: kickoff QB authority not immutable"
        )

    if return_paths:
        return parquet_path, manifest_path

    return authority


def current_authority_paths(
    data_root: Path,
) -> tuple[Path, Path]:
    data_root = Path(data_root)
    return (
        data_root / CURRENT_NAME,
        data_root / CURRENT_MANIFEST_NAME,
    )


def write_current_authority(
    authority: pd.DataFrame,
    forecast_sha256: str,
    data_root: Path,
    forecast_df: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    """
    Publish the mutable QB-role authority for the current canonical
    forecast generation.

    The sidecar represents REFRESHABLE/PREGAME games only.

    An empty authority is valid and explicit when no games remain
    refreshable. It must never mean "reuse the previous sidecar".
    """
    authority = normalize_authority(authority)

    forecast_sha256 = str(
        forecast_sha256
    ).strip().lower()

    if not forecast_sha256:
        raise RuntimeError(
            "Current QB authority requires forecast SHA"
        )

    if authority.empty:
        authority = pd.DataFrame(
            columns=ROLE_COLUMNS
        )
    else:
        validate_authority(
            authority,
            forecast_df=forecast_df,
        )

    parquet_path, manifest_path = (
        current_authority_paths(data_root)
    )

    atomic_parquet_write(
        authority,
        parquet_path,
    )

    authority_sha = sha256_file(
        parquet_path
    )

    manifest = {
        "contract": CURRENT_CONTRACT,
        "forecast_sha256": forecast_sha256,
        "authority_sha256": authority_sha,
        "rows": int(len(authority)),
        "games": int(
            authority["game_id"].nunique()
        ) if not authority.empty else 0,
        "empty": bool(authority.empty),
        "refreshable_only": True,
    }

    atomic_json_write(
        manifest,
        manifest_path,
    )

    return parquet_path, manifest_path


def load_current_authority(
    data_root: Path,
    forecast_sha256: str,
    forecast_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Load the mutable current sidecar only when it is cryptographically
    bound to the requested canonical forecast generation.
    """
    parquet_path, manifest_path = (
        current_authority_paths(data_root)
    )

    if (
        not parquet_path.exists()
        or not manifest_path.exists()
    ):
        raise RuntimeError(
            "Current QB authority sidecar is missing"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as fh:
        manifest = json.load(fh)

    if manifest.get("contract") != CURRENT_CONTRACT:
        raise RuntimeError(
            "Current QB authority contract mismatch"
        )

    expected_forecast_sha = str(
        forecast_sha256
    ).strip().lower()

    actual_forecast_sha = str(
        manifest.get("forecast_sha256")
        or ""
    ).strip().lower()

    if actual_forecast_sha != expected_forecast_sha:
        raise RuntimeError(
            "Current QB authority forecast SHA mismatch"
        )

    actual_authority_sha = sha256_file(
        parquet_path
    )

    expected_authority_sha = str(
        manifest.get("authority_sha256")
        or ""
    ).strip().lower()

    if actual_authority_sha != expected_authority_sha:
        raise RuntimeError(
            "Current QB authority artifact SHA mismatch"
        )

    authority = normalize_authority(
        pd.read_parquet(parquet_path)
    )

    if int(
        manifest.get("rows", -1)
    ) != len(authority):
        raise RuntimeError(
            "Current QB authority row count mismatch"
        )

    games = (
        int(authority["game_id"].nunique())
        if not authority.empty
        else 0
    )

    if int(
        manifest.get("games", -1)
    ) != games:
        raise RuntimeError(
            "Current QB authority game count mismatch"
        )

    manifest_empty = manifest.get("empty")

    if manifest_empty is not bool(authority.empty):
        raise RuntimeError(
            "Current QB authority empty-state mismatch"
        )

    if manifest.get("refreshable_only") is not True:
        raise RuntimeError(
            "Current QB authority scope mismatch"
        )

    if authority.empty:
        if forecast_df is not None:
            # Empty authority is intentionally valid only for the
            # explicit no-refreshable-games lifecycle state. Coverage
            # is therefore not inferred from a full canonical forecast.
            pass
    else:
        validate_authority(
            authority,
            forecast_df=forecast_df,
        )

    return authority
