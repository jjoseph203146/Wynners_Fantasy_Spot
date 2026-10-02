"""Evidence-only relationship shadow. No production imports or forecast computation.

Endpoint evidence is copied verbatim with source_/related_ prefixes. Team records
have NULL endpoint identities and sorted member IDs; member evidence is available
in each member's GAME_ENVIRONMENT record. Bring-backs are directed relationships.
All Week 3 relationships retain LIMITED_EVIDENCE because only two current-season
games exist; individual channel statuses remain separate and unaggregated.
"""
from pathlib import Path
import hashlib
import json
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/research/ceiling_intelligence_week3_pathway_authority_v2.parquet'
OUTPUT = ROOT / 'data/research/ceiling_intelligence_week3_relationship_shadow_v1.parquet'
AUDIT = OUTPUT.with_name(OUTPUT.stem + '_audit.txt')
CANONICAL = '24e2ecc54466528b5a838b02cf8690054ebfd89240445e00c93b1ef831bcfaf8'
ENV = ['pass_environment_status', 'rush_environment_status',
       'scoring_environment_status', 'play_volume_status']
CHANNELS = ['position_matchup_status', *ENV, 'high_value_context_status']
BLOBS = ['PASS_ENVIRONMENT_EVIDENCE', 'RUSH_ENVIRONMENT_EVIDENCE',
         'SCORING_ENVIRONMENT_EVIDENCE', 'PLAY_VOLUME_EVIDENCE']
TYPES = {'QB_PASS_CATCHER', 'QB_MULTI_PASS_CATCHER_CONTEXT', 'RB_TEAM_SCORING',
         'OPPOSING_BRING_BACK', 'GAME_ENVIRONMENT'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(a, b):
    return bool((pd.isna(a) and pd.isna(b)) or a == b)


def eligible(p):
    return (p['actionable_pathway_evidence'] is True
            and p['player_bound_pathway_status'] == 'SUPPORTED'
            and p['injury_gate'] == 'ALLOW' and p['active_flag'] == 1
            and p['current_reconciliation_role'] != 'UNAVAILABLE'
            and p['authority_status'] in {'AUTHORITATIVE_CURRENT_QB', 'VERIFIED_CURRENT_ROLE'}
            and (p['position'] != 'QB' or (
                p['authority_status'] == 'AUTHORITATIVE_CURRENT_QB'
                and p['qb_authority_match'] is True)))


def build(source):
    active = source.loc[source.actionable_pathway_evidence.eq(True)].sort_values(
        ['game_id', 'team', 'player_id'])
    players = active.to_dict('records')
    require(all(eligible(p) for p in players), 'INELIGIBLE_ACTIONABLE_SOURCE')
    rows = []

    def add(kind, p, related=None, members=None):
        team_record = members is not None
        row = {k: p[k] for k in ['season', 'week', 'game_id', 'team', 'opponent_team']}
        row.update(relationship_type=kind,
                   relationship_scope='TEAM_CONTEXT' if team_record else (
                       'PLAYER_PAIR' if related else 'PLAYER_CONTEXT'),
                   member_player_ids=json.dumps(sorted(m['player_id'] for m in members)) if team_record else None,
                   source_artifact_sha256=CANONICAL,
                   sample_evidence_status='LIMITED_EVIDENCE',
                   relationship_status='LIMITED_EVIDENCE',
                   analysis_only=True, production_influence=False, solver_influence=False)
        for prefix, endpoint in [('source', None if team_record else p), ('related', related)]:
            for c in source.columns:
                row[f'{prefix}_{c}'] = endpoint[c] if endpoint else None
        for c in CHANNELS + BLOBS:
            row[c] = p[c] if not team_record or c in ENV + BLOBS else None
        descriptions = {
            'QB_PASS_CATCHER': 'Authoritative current QB and actionable WR/TE teammate share a passing-game context.',
            'QB_MULTI_PASS_CATCHER_CONTEXT': 'Team has an authoritative actionable QB and at least two actionable WR/TE teammates; member evidence joins to GAME_ENVIRONMENT by game_id and source_player_id.',
            'RB_TEAM_SCORING': 'Actionable RB linked to own-team rushing, scoring and play-volume context; high-value evidence retains its source availability.',
            'OPPOSING_BRING_BACK': 'Actionable offensive players on opposing teams in the same game; positive statistical correlation has not been validated.',
            'GAME_ENVIRONMENT': 'Actionable player linked to own-team game context; pass, rush, scoring and play-volume channels remain separate.',
        }
        row['relationship_explanation'] = descriptions[kind] + (
            ' Only two completed 2026 team games; historical rolling windows may include prior seasons.'
            ' Historical rates are evidence, not current forecasts. Descriptive context only;'
            ' no validated ceiling, DFS recommendation, projection adjustment or solver preference.'
            ' RB G2G and TE I10 evidence does not validate ceiling outcomes.')
        identity = [p['season'], p['week'], p['game_id'], kind, p['team'],
                    None if team_record else p['player_id'], related['player_id'] if related else None]
        row['relationship_key'] = json.dumps(identity, separators=(',', ':'))
        rows.append(row)

    for p in players:
        add('GAME_ENVIRONMENT', p)
        if p['position'] == 'RB':
            add('RB_TEAM_SCORING', p)
        teammates = [q for q in players if q['game_id'] == p['game_id'] and q['team'] == p['team']
                     and q['position'] in {'WR', 'TE'}]
        if p['position'] == 'QB':
            for q in teammates:
                add('QB_PASS_CATCHER', p, q)
            if len(teammates) >= 2:
                add('QB_MULTI_PASS_CATCHER_CONTEXT', p, members=[p, *teammates])
        for q in players:
            if q['game_id'] == p['game_id'] and q['team'] == p['opponent_team']:
                add('OPPOSING_BRING_BACK', p, q)
    return pd.DataFrame(rows).sort_values('relationship_key').reset_index(drop=True)


def validate(out, source):
    lookup = source.set_index('player_id').to_dict('index')
    counts = dict.fromkeys(['DUPLICATE_RELATIONSHIP_KEYS', 'BLOCKED_PLAYER_RELATIONSHIPS',
                           'CONTINGENCY_QB_RELATIONSHIPS', 'UNVERIFIED_ROLE_RELATIONSHIPS',
                           'CROSS_GAME_RELATIONSHIPS', 'INVALID_SAME_TEAM_RELATIONSHIPS',
                           'INVALID_OPPONENT_RELATIONSHIPS'], 0)
    counts['DUPLICATE_RELATIONSHIP_KEYS'] = int(out.relationship_key.duplicated().sum())
    for r in out.to_dict('records'):
        endpoints = []
        for prefix in ['source', 'related']:
            pid = r[f'{prefix}_player_id']
            if pd.isna(pid):
                require(all(pd.isna(r[f'{prefix}_{c}']) for c in source), 'PARTIAL_NULL_ENDPOINT')
                continue
            require(pid in lookup, 'UNRESOLVED_PLAYER_ID')
            p = dict(lookup[pid], player_id=pid)
            endpoints.append(p)
            for c in source:
                require(same(r[f'{prefix}_{c}'], p[c]), f'ENDPOINT_EVIDENCE_MUTATION:{pid}:{c}')
        if r['relationship_type'] == 'QB_MULTI_PASS_CATCHER_CONTEXT':
            require(not endpoints, 'TEAM_RECORD_FAKE_IDENTITY')
            ids = json.loads(r['member_player_ids'])
            require(ids == sorted(set(ids)), 'INVALID_MEMBER_IDS')
            require(all(pid in lookup for pid in ids), 'UNRESOLVED_MEMBER')
            endpoints = [dict(lookup[pid], player_id=pid) for pid in ids]
            qbs = [p for p in endpoints if p['position'] == 'QB']
            catchers = [p for p in endpoints if p['position'] in {'WR', 'TE'}]
            require(len(qbs) == 1 and len(catchers) >= 2 and len(endpoints) == len(catchers)+1,
                    'INVALID_MULTI_CATCHER_MEMBERS')
            expected = source.loc[source.actionable_pathway_evidence & source.game_id.eq(r['game_id'])
                                  & source.team.eq(r['team']) & source.position.isin(['QB', 'WR', 'TE']), 'player_id']
            require(ids == sorted(expected.tolist()), 'INCOMPLETE_TEAM_MEMBERS')
            require(all(p['team'] == r['team'] for p in endpoints), 'INVALID_TEAM_MEMBERS')
        else:
            require(len(endpoints) == (2 if r['relationship_type'] in {'QB_PASS_CATCHER', 'OPPOSING_BRING_BACK'} else 1), 'INVALID_ENDPOINT_COUNT')
        for p in endpoints:
            counts['BLOCKED_PLAYER_RELATIONSHIPS'] += int(p['injury_gate'] == 'BLOCK' or p['active_flag'] != 1 or p['current_reconciliation_role'] == 'UNAVAILABLE')
            counts['CONTINGENCY_QB_RELATIONSHIPS'] += int(p['position'] == 'QB' and p['authority_status'] == 'CONTINGENCY_ONLY')
            counts['UNVERIFIED_ROLE_RELATIONSHIPS'] += int(p['authority_status'] == 'ROLE_NOT_VERIFIED')
            counts['CROSS_GAME_RELATIONSHIPS'] += int(p['game_id'] != r['game_id'])
            require(eligible(p), 'NON_ACTIONABLE_RELATIONSHIP')
            require(p['season'] == r['season'] and p['week'] == r['week'], 'INVALID_SEASON_WEEK')
        p = endpoints[0]
        require(p['team'] == r['team'] and p['opponent_team'] == r['opponent_team'], 'INVALID_TEAM_CONTEXT')
        if r['relationship_type'] == 'QB_PASS_CATCHER':
            q = endpoints[1]
            counts['INVALID_SAME_TEAM_RELATIONSHIPS'] += int(p['team'] != q['team'])
            require(p['position'] == 'QB' and q['position'] in {'WR', 'TE'}, 'INVALID_PASS_PAIR')
        if r['relationship_type'] == 'OPPOSING_BRING_BACK':
            q = endpoints[1]
            counts['INVALID_OPPONENT_RELATIONSHIPS'] += int(p['team'] == q['team'] or p['opponent_team'] != q['team'] or q['opponent_team'] != p['team'])
        if r['relationship_type'] == 'RB_TEAM_SCORING':
            require(p['position'] == 'RB', 'INVALID_RB_CONTEXT')
        for c in ENV + BLOBS:
            require(all(same(r[c], e[c]) for e in endpoints if e['team'] == r['team']), 'TEAM_ENVIRONMENT_MISMATCH')
        require(r['relationship_status'] == r['sample_evidence_status'] == 'LIMITED_EVIDENCE', 'LOST_SAMPLE_WARNING')
    require(not any(counts.values()), f'FAIL_CLOSED:{counts}')
    require(set(out.relationship_type) == TYPES, 'MISSING_RELATIONSHIP_TYPE')
    return counts


def main():
    require(sha(SOURCE) == CANONICAL, 'SOURCE_SHA256_MISMATCH')
    source = pd.read_parquet(SOURCE)
    require(len(source) == 582 and source.player_id.notna().all() and source.player_id.is_unique, 'SOURCE_ID_CONTRACT')
    active = source.loc[source.actionable_pathway_evidence.eq(True)]
    require(len(active) == 69 and active.game_id.nunique() == 13, 'ACTIONABLE_SOURCE_CONTRACT')
    require(source.season.eq(2026).all() and source.week.eq(3).all(), 'SOURCE_WEEK_CONTRACT')
    for blob in BLOBS:
        require(all(json.loads(v)['current_season_games'] == 2 for v in active[blob]), 'SAMPLE_DEPTH_CONTRACT')
    out = build(source)
    validate(out, source)
    # Validate actual serialized values, including all central_expected_* columns.
    with tempfile.TemporaryDirectory(prefix='relationship-shadow-') as temp:
        staged = Path(temp) / OUTPUT.name
        out.to_parquet(staged, index=False)
        restored = pd.read_parquet(staged)
        checks = validate(restored, source)
        pd.testing.assert_frame_equal(out, restored)
        require(sha(SOURCE) == CANONICAL, 'SOURCE_CHANGED_DURING_BUILD')
        report = {
            'SOURCE_SHA256': CANONICAL, 'SOURCE_ROWS': len(source),
            'ACTIONABLE_SOURCE_ROWS': len(active), 'UNIQUE_GAMES': restored.game_id.nunique(),
            'SOURCE_UNIQUE_GAMES': source.game_id.nunique(), 'RELATIONSHIP_ROWS': len(restored),
            'ROWS_BY_RELATIONSHIP_TYPE': json.dumps(restored.relationship_type.value_counts().sort_index().to_dict(), sort_keys=True),
            **checks,
        }
        for key in ['NUMERIC_CEILING_CREATED', 'CORRELATION_SCORE_CREATED', 'CENTRAL_FORECAST_MUTATION',
                    'PRODUCTION_INFLUENCE', 'SOLVER_INFLUENCE', 'CLASSIC_SOLVER_CHANGED',
                    'SINGLE_GAME_SOLVER_CHANGED', 'CRON_CHANGED']:
            report[key] = 'NO'
        report.update(
            QB_AUTHORITY_POLICY='Resolved V2 authority_status and qb_authority_match govern; legacy reconciliation/primary_qb_id fields retained verbatim, never used to re-resolve authority',
            EVIDENCE_ROUNDTRIP='EXACT_ALL_ENDPOINT_FIELDS_INCLUDING_CENTRAL_EXPECTED',
            STATUS_POLICY='LIMITED_EVIDENCE for all rows: only two completed 2026 games; channels copied independently, never scored',
            TEAM_RECORD_POLICY='NULL source/related identities; sorted member_player_ids join to GAME_ENVIRONMENT endpoint evidence',
            GAME_ENVIRONMENT_POLICY='One player-context record per actionable player; unprefixed channels refer to source team',
            BRING_BACK_POLICY='Directed pairs; reverse direction is distinct; no positive statistical correlation claim',
            MISSING_EVIDENCE_POLICY='Nulls and availability flags preserved; no imputation',
            FORECAST_AUTHORITY='Stat Outlook remains sole central forecast authority; full precision retained internally',
            PRESENTATION_POLICY='No forecast presentation created; future counting-stat display must use whole numbers without integer TD certainty from probability',
            FANDUEL_CLASSIC='0.5 PPR; no 100-yard rushing/receiving or 300-yard passing bonus; no scoring calculation performed',
            SUPPORTED_SEMANTICS='Evidence-supported descriptive context only; no validated numerical ceiling, recommendation, boost, or solver preference',
            OUTPUT_SHA256=sha(staged),
            FINAL_STATUS='PASS_ANALYSIS_ONLY_RELATIONSHIP_SHADOW',
        )
        OUTPUT.write_bytes(staged.read_bytes())
        AUDIT.write_text('\n'.join(f'{k}={v}' for k, v in report.items()) + '\n')
    print(AUDIT.read_text(), end='')
    for path in [OUTPUT, AUDIT]:
        print(f'SHA256 {sha(path)} {path.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
