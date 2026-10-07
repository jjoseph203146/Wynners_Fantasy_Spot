"""Read-only Stage 4 Player Outlook relevance selection.

This module may select which already-validated immutable Player Outlook
evidence is public. It does not mutate forecasts, eligibility, solver
inputs, databases, starter artifacts, or the immutable capture.
"""

from dataclasses import replace
from datetime import timezone
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd

from player_outlook_data import GameContext, ValidatedOutlook


STARTER_CONTRACT = "WFS_STARTER_VERIFICATION_CURRENT_V1"
STARTER_SOURCE_CONTRACT = "WFS_STARTER_VERIFICATION_V1_1"
V3_CONTRACT = "WFS_OFFENSIVE_TEAM_RECONCILIATION_SHADOW_V3"
SECONDARY_SHARE = 0.20
POSITIONS = frozenset({"QB", "RB", "WR", "TE"})


class InvalidRelevance(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise InvalidRelevance(reason)


def _text(value):
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _player_id(value):
    return _text(value) and re.fullmatch(r"\d{2}-\d{7}", value) is not None


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _utc(value):
    stamp = pd.Timestamp(value)
    _require(not pd.isna(stamp) and stamp.tzinfo is not None, "INVALID_TIME")
    return stamp.tz_convert("UTC").to_pydatetime()


def _starter_ids(*, game, starter_path, starter_manifest_path, now_utc):
    path = Path(starter_path)
    manifest_path = Path(starter_manifest_path)
    _require(path.is_file() and manifest_path.is_file(), "STARTER_MISSING")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("contract") == STARTER_CONTRACT, "STARTER_CONTRACT")
    _require(
        manifest.get("source_contract") == STARTER_SOURCE_CONTRACT,
        "STARTER_SOURCE_CONTRACT",
    )
    _require(manifest.get("identity_gate") == "PASS", "STARTER_IDENTITY")
    _require(manifest.get("analysis_only") is True, "STARTER_SAFETY")
    _require(manifest.get("production_influence") is False, "STARTER_SAFETY")
    _require(manifest.get("solver_influence") is False, "STARTER_SAFETY")
    _require(manifest.get("forecast_mutation") is False, "STARTER_SAFETY")
    _require(manifest.get("database_mutation") is False, "STARTER_SAFETY")
    _require(int(manifest.get("season")) == game.season, "STARTER_PERIOD")
    _require(int(manifest.get("week")) == game.week, "STARTER_PERIOD")
    _require(str(manifest.get("game_type") or "").upper() == "REG", "STARTER_TYPE")

    generated = _utc(manifest.get("generated_at_utc"))
    now = _utc(now_utc)
    _require(generated <= now, "STARTER_FUTURE")
    _require((now - generated).total_seconds() <= 7200, "STARTER_STALE")

    frame = pd.read_parquet(path)
    required = {
        "team",
        "position",
        "verification_status",
        "rotowire_gsis_id",
        "depth_gsis_id",
    }
    _require(frame.columns.is_unique and required.issubset(frame.columns), "STARTER_SCHEMA")

    frame = frame[
        frame["team"].astype(str).isin({game.away_team, game.home_team})
        & frame["position"].astype(str).isin(POSITIONS)
    ].copy()

    starters = {team: {pos: set() for pos in POSITIONS}
                for team in (game.away_team, game.home_team)}

    for row in frame.to_dict(orient="records"):
        status = str(row.get("verification_status") or "").upper()
        rw = row.get("rotowire_gsis_id")
        depth = row.get("depth_gsis_id")
        if status == "AGREE":
            _require(_player_id(rw) and _player_id(depth) and rw == depth, "STARTER_ID")
        elif row["position"] == "QB" and status == "DEPTH_STARTER_BLOCKED":
            # The verified replacement must still match active V3 PRIMARY_QB below.
            _require(
                _player_id(rw) and _player_id(depth) and rw != depth
                and row.get("depth_availability_present") is True
                and row.get("rw_availability_present") is True
                and str(row.get("depth_injury_gate") or "").upper() == "BLOCK"
                and str(row.get("rw_injury_gate") or "").upper() == "ALLOW",
                "QB_REPLACEMENT_EVIDENCE",
            )
        else:
            continue
        starters[row["team"]][row["position"]].add(rw)

    for team in starters:
        # QB authority remains strict because downstream relevance requires one
        # exact verified PRIMARY_QB.
        _require(len(starters[team]["QB"]) == 1, "QB_STARTER")

        # Skill-position starter completeness is intentionally nonfatal.
        #
        # Injury, trade, committee, or game-time-decision conditions can leave
        # RB/WR/TE without a currently verified starter. That must not suppress
        # otherwise safe Player Outlook evidence for the entire game.
        #
        # Each rollover rereads current starter verification and V3 authority,
        # so an empty lane here automatically self-heals when authoritative
        # evidence becomes available on a later run.
        #
        # Starter artifact integrity, identity, chronology, and safety checks
        # above remain fail-closed.
        for position in ("RB", "WR", "TE"):
            _require(
                isinstance(starters[team][position], set),
                f"{position}_STARTER_STATE",
            )

    return starters


def _v3_rows(*, game, v3_path, v3_audit_path):
    path = Path(v3_path)
    audit_path = Path(v3_audit_path)
    _require(path.is_file() and audit_path.is_file(), "V3_MISSING")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    _require(audit.get("version") == V3_CONTRACT, "V3_CONTRACT")
    _require(audit.get("status") == "PASS", "V3_STATUS")
    _require(audit.get("production_modified") is False, "V3_SAFETY")

    output = (audit.get("outputs") or {}).get("player_shadow") or {}
    _require(output.get("sha256") == _sha256(path), "V3_HASH")

    frame = pd.read_csv(path)
    required = {
        "game_id",
        "team",
        "opponent_team",
        "player_id",
        "position",
        "reconciliation_role",
        "reconciled_carries",
        "reconciled_targets",
        "active_flag",
        "injury_flag",
        "primary_qb_id",
    }
    _require(frame.columns.is_unique and required.issubset(frame.columns), "V3_SCHEMA")
    _require(not frame.duplicated(["game_id", "player_id"]).any(), "V3_DUPLICATE")

    frame = frame[frame["game_id"].astype(str) == game.game_id].copy()
    _require(len(frame) > 0, "V3_GAME")
    _require(set(frame["team"].astype(str)) == {game.away_team, game.home_team}, "V3_TEAMS")
    _require(
        set(frame["opponent_team"].astype(str)).issubset({game.away_team, game.home_team}),
        "V3_OPPONENT",
    )
    return frame


def apply_player_outlook_relevance(
    *,
    validated_outlook,
    game_context,
    starter_path,
    starter_manifest_path,
    v3_path,
    v3_audit_path,
    now_utc,
):
    """Return only relevant players from a fully validated immutable capture.

    Fail closed on stale/ambiguous role authority. Initial V1 is pregame-only.
    """
    try:
        _require(
            isinstance(validated_outlook, ValidatedOutlook)
            and validated_outlook.available,
            "OUTLOOK_UNAVAILABLE",
        )
        _require(isinstance(game_context, GameContext), "GAME_CONTEXT")

        now = _utc(now_utc)
        kickoff = _utc(game_context.kickoff_utc)
        _require(not game_context.completed and now < kickoff, "PREGAME_ONLY")

        # The immutable capture has already been validated in full before
        # relevance is applied. Never use relevance to hide invalid capture rows.
        evidence = {}
        for player in validated_outlook.players:
            _require(_player_id(player.player_id), "CAPTURE_PLAYER_ID")
            _require(player.player_id not in evidence, "CAPTURE_DUPLICATE")
            evidence[player.player_id] = player

        starters = _starter_ids(
            game=game_context,
            starter_path=starter_path,
            starter_manifest_path=starter_manifest_path,
            now_utc=now,
        )
        v3 = _v3_rows(
            game=game_context,
            v3_path=v3_path,
            v3_audit_path=v3_audit_path,
        )

        selected = set()

        for team in (game_context.away_team, game_context.home_team):
            team_rows = v3[v3["team"].astype(str) == team].copy()

            # Availability first: V3 UNAVAILABLE / inactive rows never qualify.
            available = team_rows[
                (pd.to_numeric(team_rows["active_flag"], errors="coerce") == 1)
                & (team_rows["reconciliation_role"].astype(str) != "UNAVAILABLE")
            ].copy()

            # QB: exactly the verified current starter and V3 PRIMARY_QB.
            qb_id = next(iter(starters[team]["QB"]))
            qb_rows = available[
                (available["position"].astype(str) == "QB")
                & (available["player_id"].astype(str) == qb_id)
                & (available["reconciliation_role"].astype(str) == "PRIMARY_QB")
            ]
            _require(len(qb_rows) == 1, "QB_AUTHORITY")
            primary = str(qb_rows.iloc[0]["primary_qb_id"])
            _require(primary == qb_id, "QB_PRIMARY_MISMATCH")
            selected.add(qb_id)

            # WR: preserve every independently verified starting lane.
            wr_available = set(
                available.loc[
                    available["position"].astype(str) == "WR", "player_id"
                ].astype(str)
            )
            wr_starters = starters[team]["WR"]

            # A verified starter that conflicts with current V3 availability is
            # still an authority contradiction and remains fail-closed.
            #
            # An EMPTY starter set is different: it means starter resolution is
            # still pending. Do not fail the game; publish other safe evidence
            # and let the next rollover self-heal this lane.
            if wr_starters:
                _require(
                    wr_starters.issubset(wr_available),
                    "WR_STARTER_UNAVAILABLE",
                )
                selected.update(wr_starters)

            # RB / TE: verified starter plus material secondary current allocation.
            for position in ("RB", "TE"):
                pos_rows = available[
                    available["position"].astype(str) == position
                ].copy()
                available_ids = set(pos_rows["player_id"].astype(str))
                position_starters = starters[team][position]

                # Missing starter authority is a recoverable pending state.
                # Preserve a hard failure only when a starter IS asserted but
                # contradicts current V3 availability.
                if position_starters:
                    _require(
                        position_starters.issubset(available_ids),
                        f"{position}_STARTER_UNAVAILABLE",
                    )
                    selected.update(position_starters)

                allocations = {}
                for row in pos_rows.to_dict(orient="records"):
                    carries = _number(row.get("reconciled_carries"))
                    targets = _number(row.get("reconciled_targets"))
                    _require(carries is not None and targets is not None, "V3_ALLOCATION")
                    allocation = targets if position == "TE" else carries + targets
                    allocations[str(row["player_id"])] = allocation

                total = sum(allocations.values())
                if total > 0:
                    for player_id, allocation in allocations.items():
                        if allocation / total >= SECONDARY_SHARE:
                            selected.add(player_id)

        # Never manufacture evidence for a selected current player.
        # Missing capture evidence is simply suppressed.
        kept = tuple(
            player for player in validated_outlook.players
            if player.player_id in selected
        )
        _require(bool(kept), "NO_RELEVANT_CAPTURE_PLAYERS")

        return replace(validated_outlook, players=kept)

    except InvalidRelevance as exc:
        return ValidatedOutlook(False, reason=str(exc))
    except Exception:
        return ValidatedOutlook(False, reason="INVALID_RELEVANCE")
