"""ESPN structured current injury authority; exact external IDs, never names.

Retrieval/validation failure aborts consensus publication, retaining prior evidence.
Records apply only in the target team's game week (game date minus six days
through game date). Old Active/news records are not current designations.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tempfile
import os
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
URL = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries'
SOURCE = 'ESPN_STRUCTURED_INJURIES'
HARD = {'OUT', 'DOUBTFUL', 'INACTIVE'}
STATUSES = HARD | {'QUESTIONABLE', 'ACTIVE'}
LINEAGE = ['espn_athlete_id', 'espn_injury_record_id', 'espn_status',
           'espn_record_date', 'espn_feed_timestamp', 'espn_team',
           'espn_position', 'espn_mapped_gsis_id', 'espn_identity_result',
           'espn_source', 'espn_authority_decision']


def text(value):
    return '' if pd.isna(value) else str(value).strip()


def team(value):
    value = text(value).upper()
    return {'WSH': 'WAS', 'JAC': 'JAX', 'LA': 'LAR'}.get(value, value)


def position(value):
    value = text(value).upper()
    groups = {"CB": "DB", "S": "DB", "SS": "DB", "FS": "DB",
              "DT": "DL", "NT": "DL", "DE": "DL", "OT": "OL",
              "G": "OL", "C": "OL", "OG": "OL", "T": "OL",
              "PK": "K", "FB": "RB", "ILB": "LB", "OLB": "LB"}
    return groups.get(value, value)


def athlete_id(athlete):
    ids = set()
    direct = text(athlete.get('id', ''))
    if direct:
        if not direct.isdigit():
            return '', 'INVALID_ATHLETE_ID'
        ids.add(direct)
    for value in [athlete.get('uid', ''), *[x.get('href', '') for x in athlete.get('links', [])]]:
        ids.update(re.findall(r'(?:/id/|~a:)(\d+)(?=/|&|~|$)', value))
    if len(ids) != 1:
        return '', 'AMBIGUOUS_ATHLETE_ID' if ids else 'MISSING_ATHLETE_ID'
    return next(iter(ids)), 'EXACT_ESPN_ID'


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        temp = f.name
    os.replace(temp, path)


def parse_feed(payload, conn, season, week, now=None):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    feed = pd.to_datetime(payload.get('timestamp'), utc=True, errors='coerce')
    if (payload.get('status') != 'success' or pd.isna(feed)
            or not pd.Timedelta(minutes=-5) <= now - feed <= pd.Timedelta(hours=6)
            or payload.get('season', {}).get('year') != season
            or payload.get('season', {}).get('type') != 2):
        raise RuntimeError('ESPN feed failed current season/freshness contract')
    games = pd.read_sql_query("SELECT game_date,home_team,away_team FROM games WHERE season=? AND week=? AND game_type='REG'", conn, params=(season, week))
    if games.empty:
        raise RuntimeError('ESPN applicability: target schedule unavailable')
    dates = pd.to_datetime(games.game_date, utc=True)
    if not dates.min() - pd.Timedelta(days=6) <= now <= dates.max() + pd.Timedelta(days=2):
        raise RuntimeError('ESPN applicability: target week is not current')
    team_dates = {}
    for row in games.itertuples():
        for t in [row.home_team, row.away_team]:
            if team(t) in team_dates:
                raise RuntimeError('ESPN applicability: duplicate team game')
            team_dates[team(t)] = pd.Timestamp(row.game_date, tz='UTC')
    roster = pd.read_sql_query('SELECT gsis_id,espn_id,team,position FROM weekly_rosters WHERE season=? AND week=?', conn, params=(season, week)).fillna('')
    crosswalk = pd.read_sql_query("SELECT gsis_id,espn_id FROM player_identity WHERE source_roster=1", conn).fillna('')
    crosswalk = pd.concat([crosswalk, roster[['gsis_id', 'espn_id']]]).drop_duplicates()
    bridge = {}
    for r in crosswalk.itertuples():
        gid, eid = text(r.gsis_id), text(r.espn_id)
        if re.fullmatch(r'00-\d{7}', gid) and eid:
            bridge.setdefault(eid, set()).add(gid)
    identities = {}
    for r in roster.itertuples():
        identities.setdefault(text(r.gsis_id), set()).add((team(r.team), position(r.position)))
    rows = []
    groups = payload.get('injuries')
    if not isinstance(groups, list) or not groups:
        raise RuntimeError('ESPN injury feed empty/malformed')
    for group in groups:
        for injury in group['injuries']:
            a = injury['athlete']
            eid, result = athlete_id(a)
            t = team(a.get('team', {}).get('abbreviation', ''))
            pos = text(a.get('position', {}).get('abbreviation', '')).upper()
            candidates = bridge.get(eid, set())
            gid = next(iter(candidates)) if len(candidates) == 1 else ''
            if result == 'EXACT_ESPN_ID':
                result = 'AMBIGUOUS_CROSSWALK' if len(candidates) > 1 else 'UNRESOLVED_CROSSWALK' if not gid else 'MAPPED'
            if gid and (len(identities.get(gid, set())) != 1):
                result = 'AMBIGUOUS_CURRENT_ROSTER' if identities.get(gid) else 'UNRESOLVED_CURRENT_ROSTER'
            elif gid and identities[gid] != {(t, position(pos))}:
                result = 'INCOMPATIBLE_TEAM_POSITION'
            if str(a.get('team', {}).get('id', '')) != str(group.get('id', '')):
                result = 'INCOMPATIBLE_FEED_TEAM'
            date = pd.to_datetime(injury.get('date'), utc=True, errors='coerce')
            game = team_dates.get(t)
            status = text(injury.get('status', '')).upper()
            applicability_days = 7 if status in HARD else 6
            applicable = (game is not None and not pd.isna(date)
                          and game - pd.Timedelta(days=applicability_days) <= date < game + pd.Timedelta(days=1)
                          and date <= feed + pd.Timedelta(minutes=5))
            rows.append(dict(espn_athlete_id=eid, espn_injury_record_id=text(injury.get('id', '')),
                             espn_status=status, espn_record_date=text(injury.get('date', '')),
                             espn_feed_timestamp=payload['timestamp'], espn_team=t, espn_position=pos,
                             espn_mapped_gsis_id=gid, espn_identity_result=result, espn_source=SOURCE,
                             espn_authority_decision='PENDING', applicable=bool(applicable),
                             recognized=status in STATUSES, player_name=a.get('displayName', '')))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError('ESPN injury feed has zero records')
    # Multiple applicable records for a GSIS are quarantined, never last-row wins.
    relevant = frame.applicable & frame.recognized
    duplicate = relevant & frame.espn_identity_result.eq('MAPPED') & frame.loc[relevant, 'espn_mapped_gsis_id'].duplicated(keep=False).reindex(frame.index, fill_value=False)
    frame.loc[duplicate, 'espn_identity_result'] = 'AMBIGUOUS_DUPLICATE_GSIS'
    mapped = frame.espn_identity_result.eq('MAPPED')
    ambiguous = frame.espn_identity_result.str.startswith('AMBIGUOUS')
    stats = dict(total_rows=len(frame), total_relevant_rows=int(relevant.sum()),
                 mapped_rows=int((relevant & mapped).sum()), unresolved_rows=int((relevant & ~mapped & ~ambiguous).sum()),
                 ambiguous_rows=int((relevant & ambiguous).sum()), duplicate_gsis_rows=int(duplicate.sum()),
                 inapplicable_rows=int((~frame.applicable).sum()), unrecognized_rows=int((~frame.recognized).sum()),
                 season=season, week=week, feed_timestamp=payload['timestamp'], applicability='PASS')
    return frame, stats


def ingest(conn, season, week):
    directory = ROOT / 'data' / 'espn_injuries'
    try:
        max_attempts = 3
        response = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.get(URL, timeout=(10, 40))
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                if attempt == max_attempts:
                    raise
                delay = 5 if attempt == 1 else 10
                print(
                    f'ESPN transport attempt {attempt}/{max_attempts} failed: '
                    f'{type(exc).__name__}; retrying in {delay}s'
                )
                time.sleep(delay)

        response.raise_for_status()
        payload = response.json()
        frame, stats = parse_feed(payload, conn, season, week)
        atomic_json(directory / 'last_valid.json', {'payload': payload, 'statistics': stats, 'rows': frame.to_dict('records')})
        atomic_json(directory / 'status.json', dict(status='PASS', **stats))
        print('ESPN_INGEST=' + json.dumps(stats, sort_keys=True))
        return frame
    except Exception as exc:
        atomic_json(directory / 'status.json', dict(status='FAIL_CLOSED', error=str(exc), attempted_at=datetime.now(timezone.utc).isoformat()))
        raise RuntimeError('ESPN ingestion failed; prior consensus and last valid evidence retained') from exc


def merge_structured(structured, evidence, season, week):
    out = structured.copy()
    for idx, e in evidence.iterrows():
        if not (e.applicable and e.recognized and e.espn_identity_result == 'MAPPED'):
            evidence.at[idx, 'espn_authority_decision'] = 'QUARANTINED_OR_INAPPLICABLE'
            continue
        mask = out.gsis_id.eq(e.espn_mapped_gsis_id)
        if mask.any():
            r = out.loc[mask].iloc[0]
            if team(r.team) != e.espn_team or position(r.position) != position(e.espn_position):
                evidence.at[idx, 'espn_authority_decision'] = 'INCOMPATIBLE_STRUCTURED_IDENTITY'
                continue
            # No non-hard ESPN designation may erase valid hard evidence.
            if r.report_status in HARD and e.espn_status not in HARD:
                evidence.at[idx, 'espn_authority_decision'] = 'PRESERVE_NFLVERSE_HARD_BLOCK'
                continue
            out.loc[mask, 'report_status'] = e.espn_status
        else:
            out = pd.concat([out, pd.DataFrame([dict(season=season, week=week, game_type='REG',
                team=e.espn_team, gsis_id=e.espn_mapped_gsis_id, player_name=e.player_name,
                position=e.espn_position, primary_injury='', secondary_injury='',
                report_status=e.espn_status, practice_status='')])], ignore_index=True)
        evidence.at[idx, 'espn_authority_decision'] = 'PRIMARY'
    return out


def attach_lineage(consensus, evidence):
    out = consensus.copy()
    for col in LINEAGE:
        out[col] = ''
    # Only unique, applicable mapped evidence is attachable to production rows.
    selected = evidence[evidence.applicable & evidence.recognized & evidence.espn_identity_result.eq('MAPPED')]
    for e in selected.to_dict('records'):
        mask = out.gsis_id.eq(e['espn_mapped_gsis_id'])
        for col in LINEAGE:
            out.loc[mask, col] = e[col]
        if e['espn_authority_decision'] == 'PRIMARY':
            retained = mask & out.injury_gate.eq('BLOCK') & ~out.consensus_status.eq(e['espn_status'])
            out.loc[retained, 'espn_authority_decision'] = 'PRESERVE_EXISTING_HARD_BLOCK'
            applied = mask & ~retained
            out.loc[applied, 'structured_source'] = SOURCE
            out.loc[applied, 'authoritative_source'] = SOURCE
    return out
