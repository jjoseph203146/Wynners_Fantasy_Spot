"""Read-only adapter to the same pure WFS exact resolver used by foundation V1.

Snapshots are supplied by the caller; no database, index builder or ingestion calls.
"""
import pandas as pd
from fanduel_injury_ingest import (
    DIRECT_TEAMS, TEAM_ALIASES, normalize_name, normalize_position,
    normalize_team, resolve_identity,
)
from shadow_player_analysis_v1.core import digest, require, timestamp


class Authority:
    def __init__(self, snapshot, schedule, cutoff):
        require(snapshot.get("authority") == "WFS_IDENTITY_SNAPSHOT", "IDENTITY_AUTHORITY")
        require(schedule.get("authority") == "WFS_SCHEDULE_SNAPSHOT", "SCHEDULE_AUTHORITY")
        for value in (snapshot, schedule):
            require(timestamp(value["available_at_utc"]) <= timestamp(cutoff), "AUTHORITY_FROM_FUTURE")
        self.snapshot_hash = digest(snapshot)
        self.schedule_hash = digest(schedule)
        self.games = schedule["games"]
        rows = []
        for original in snapshot["players"]:
            require(bool(original.get("gsis_id")), "BLANK_SNAPSHOT_GSIS")
            row = dict(original)
            row.update(normalized_name=normalize_name(row["player_name"]),
                       team=normalize_team(row["team"]), position=normalize_position(row["position"]))
            rows.append(row)
        self.frame = pd.DataFrame(rows, columns=["gsis_id", "player_name", "normalized_name", "team", "position"])
        for _, group in self.frame.groupby("gsis_id"):
            require(len(group[["team", "position"]].drop_duplicates()) == 1, "CONFLICTING_IDENTITY_SNAPSHOT")
        self.names = sorted({r["player_name"] for r in rows if len(r["normalized_name"].split()) >= 2})

    def player(self, mention):
        name = normalize_name(mention.get("name"))
        candidates = self.frame[self.frame.normalized_name.eq(name)]
        if mention.get("gsis_id"):
            candidates = self.frame[self.frame.gsis_id.eq(mention["gsis_id"])]
            ids = sorted(set(candidates.gsis_id))
            # Supplied IDs must exist in WFS, and supplied name/context must agree.
            if name:
                candidates = candidates[candidates.normalized_name.eq(name)]
            for field, normalizer in (("team", normalize_team), ("position", normalize_position)):
                if mention.get(field):
                    candidates = candidates[candidates[field].eq(normalizer(mention[field]))]
            matched = len(ids) == 1 and not candidates.empty
            gsis, reason = (ids[0], "EXACT_WFS_GSIS") if matched else (None, "GSIS_CONTEXT_OR_AUTHORITY_MISMATCH")
        else:
            ids = sorted(set(candidates.gsis_id))
            gsis, _, reason = resolve_identity(pd.Series(dict(mention,
                normalized_name=name)), self.frame)
        status = "RESOLVED" if gsis else "AMBIGUOUS" if reason == "AMBIGUOUS_EXACT_IDENTITY" else "UNRESOLVED"
        positions = sorted(set(self.frame[self.frame.gsis_id.eq(gsis)].position)) if gsis else []
        return dict(identity_status=status, gsis_id=gsis or None, candidate_gsis_ids=ids,
                    identity_reason=reason, positions=positions, identity_method="FANDUEL_EXACT_RESOLVER_V1",
                    identity_snapshot_sha256=self.snapshot_hash)

    def team(self, value):
        team = normalize_team(value)
        valid = team in set(DIRECT_TEAMS.values())
        return dict(identity_status="RESOLVED" if valid else "UNRESOLVED", team_id=team if valid else None)

    def game(self, value):
        rows = [r for r in self.games if value and r.get("game_id") == value]
        valid = len(rows) == 1
        return dict(identity_status="RESOLVED" if valid else "UNRESOLVED",
                    game_id=value if valid else None, schedule_sha256=self.schedule_hash,
                    game=rows[0] if valid else None)
