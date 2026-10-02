"""Frozen, position-specific MI central components, after E14 and before V3.

This controlled promotion is bound to the validated current-shadow evidence.
Target rollover requires a CURRENT_SHADOW_VALIDATED manifest with exact artifact binding.
No roster, availability, starter, scoring, or ceiling authority is introduced.
"""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'data/research/models/matchup_intelligence_v1'
SHADOW = ROOT / 'data/research/matchup_intelligence_current_shadow_v1.parquet'
SHADOW_MANIFEST = ROOT / 'data/research/matchup_intelligence_current_shadow_v1_manifest.json'
SHADOW_CONTRACT = 'WFS_MATCHUP_INTELLIGENCE_CURRENT_SHADOW_V1'
MANIFEST_SHA256 = '0bf23ed204ce834d0fb76b5a0a588aee2cdf1cc50a615ede47481e86df1467eb'
KEYS = ['game_id', 'player_id', 'team', 'opponent_team', 'position']
CHANNELS = {'RB': ['rb_g2g_share', 'rb_g2g_share_available'],
            'WR': ['opportunities_allowed_avg_3'],
            'TE': ['te_i10_share', 'te_i10_share_available']}
COMPONENTS = {'RB': ['carries', 'rushing_yards'],
              'WR': ['targets', 'receptions', 'receiving_yards'],
              'TE': ['receptions', 'receiving_yards']}


def require(value, reason):
    if not value:
        raise RuntimeError('MI_FAIL_CLOSED:' + reason)


def exact_evidence(matrix, shadow):
    """One-way coverage; broad evidence rows never create forecast rows."""
    for label, frame in [('matrix', matrix), ('shadow', shadow)]:
        require(set(KEYS).issubset(frame), label + '_IDENTITY_COLUMNS')
        require(not frame[KEYS].isna().any().any(), label + '_NULL_IDENTITY')
        require(frame[KEYS].astype(str).apply(lambda c: c.str.strip().ne('').all()).all(), label + '_BLANK_IDENTITY')
        require(not frame.duplicated(KEYS[:2]).any(), label + '_DUPLICATE_IDENTITY')
    extras = list(dict.fromkeys(sum(CHANNELS.values(), []) +
                              ['prior_player_games', 'team_history_games', 'dvp_history_games']))
    require(set(extras).issubset(shadow), 'EVIDENCE_COLUMNS')
    work = matrix.merge(shadow[KEYS + extras], on=KEYS, how='left',
                        validate='one_to_one', indicator='_mi_identity', sort=False)
    require(len(work) == len(matrix), 'ROW_COUNT_CHANGED')
    required = work.position.isin(COMPONENTS)
    require(work.loc[required, '_mi_identity'].eq('both').all(), 'UNRESOLVED_APPLICABLE_IDENTITY')
    return work


def apply_components(matrix, out, e14_features):
    manifest_path = MODEL_DIR / 'manifest.json'
    require(hashlib.sha256(manifest_path.read_bytes()).hexdigest() == MANIFEST_SHA256, 'MANIFEST_HASH')
    require(SHADOW_MANIFEST.is_file(), 'SHADOW_MANIFEST_MISSING')
    shadow_manifest = json.loads(SHADOW_MANIFEST.read_text())
    require(shadow_manifest.get('contract') == SHADOW_CONTRACT, 'SHADOW_MANIFEST_CONTRACT')
    require(shadow_manifest.get('status') == 'CURRENT_SHADOW_VALIDATED', 'SHADOW_MANIFEST_STATUS')
    require(
        shadow_manifest.get('shadow_sha256')
        == hashlib.sha256(SHADOW.read_bytes()).hexdigest(),
        'SHADOW_HASH',
    )
    matrix_path = ROOT / 'data/parquet/nfl_current_offensive_model_matrix.parquet'
    require(
        shadow_manifest.get('matrix_sha256')
        == hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
        'SHADOW_MATRIX_HASH',
    )
    manifest = json.loads(manifest_path.read_text())
    shadow = pd.read_parquet(SHADOW)

    # Bind inference to the currently validated shadow target.
    # The model's historical training boundary remains frozen separately.
    target_season = shadow_manifest.get('season')
    target_week = shadow_manifest.get('week')
    require(
        isinstance(target_season, int)
        and isinstance(target_week, int),
        'SHADOW_TARGET_BINDING',
    )
    require(
        shadow_manifest.get('matrix_rows') == len(matrix)
        and shadow_manifest.get('rows') == len(shadow),
        'SHADOW_TARGET_ROW_COUNT',
    )
    for label, frame in [('matrix', matrix), ('shadow', shadow), ('output', out)]:
        require(
            frame.season.eq(target_season).all()
            and frame.week.eq(target_week).all(),
            label + '_TARGET_BINDING',
        )
    require(manifest['history_through'] == '2026_WEEK_2', 'TRAINING_BOUNDARY')
    require(manifest['e14_features'] == e14_features, 'E14_FEATURE_ORDER')
    require(out[KEYS].reset_index(drop=True).equals(matrix[KEYS].reset_index(drop=True)), 'OUTPUT_IDENTITY_ORDER')
    work = exact_evidence(matrix, shadow)
    expected = {(p, t) for p, ts in COMPONENTS.items() for t in ts}
    require(len(manifest['models']) == len(expected) and
            {(e['position'], e['target']) for e in manifest['models']} == expected, 'MODEL_COMPONENT_SET')
    before = out.copy(deep=True)
    audit = {'boundary': 'POST_E14_COMPONENT_FORECAST_PRE_V3_RECONCILIATION',
             'season': target_season, 'week': target_week, 'rows': len(out), 'components': {},
             'applicable_rows': int(work.position.isin(COMPONENTS).sum()),
             'manifest_sha256': MANIFEST_SHA256, 'shadow_sha256': shadow_manifest['shadow_sha256']}
    applied_rows = set()
    for entry in manifest['models']:
        pos, target = entry['position'], entry['target']
        features = e14_features + CHANNELS[pos]
        require(entry['features'] == features and entry['feature_count'] == len(features), 'MODEL_FEATURE_ORDER')
        require(entry['latest_training_season'] == 2026 and entry['latest_training_week_2026'] == 2, 'MODEL_TEMPORAL_BOUNDARY')
        path = MODEL_DIR / entry['artifact']
        require(entry['artifact'] == pos.lower() + '__' + target + '.joblib', 'MODEL_PATH')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], 'MODEL_HASH')
        mask = work.position.eq(pos) & work.dvp_history_games.ge(3)
        if pos in ['RB', 'TE']:
            mask &= (work.prior_player_games.ge(3) & work.team_history_games.ge(3)
                     & work.history_games_player.ge(3) & work.history_games_team.ge(3))
        q = work.loc[mask].copy()
        if pos in ['RB', 'TE']:
            share, flag = CHANNELS[pos]
            require(q[flag].notna().all() and q[flag].isin([True, False]).all(), 'AVAILABILITY_FLAG')
            available = q[flag].astype(bool)
            observed = pd.to_numeric(q.loc[available, share], errors='coerce')
            require(np.isfinite(observed).all() and observed.between(0, 1).all(), 'AVAILABLE_SHARE')
            # Missing evidence is encoded with a distinct availability feature.
            q.loc[~available, share] = 0.0
            q[flag] = available.astype(float)
        x = q[features].apply(pd.to_numeric, errors='coerce')
        require(np.isfinite(x.to_numpy(dtype=float)).all(), 'APPLICABLE_MODEL_INPUT')
        if len(q):
            model = joblib.load(path)
            require(model.n_features_in_ == len(features), 'MODEL_WIDTH')
            prediction = np.asarray(model.predict(x.to_numpy(dtype=float)), dtype=float)
            require(prediction.shape == (len(q),) and np.isfinite(prediction).all(), 'PREDICTION')
            out.iloc[q.index, out.columns.get_loc('expected_' + target)] = np.maximum(prediction, 0.0)
            applied_rows.update(q.index.tolist())
        audit['components'][pos + ':' + target] = int(mask.sum())
    for col in out.columns:
        different = ~(out[col].eq(before[col]) | (out[col].isna() & before[col].isna()))
        allowed = out.position.map(lambda p: col in ['expected_' + c for c in COMPONENTS.get(p, [])])
        require(not (different & ~allowed).any(), 'UNEXPECTED_COMPONENT_CHANGE:' + col)
    require(out.isna().equals(before.isna()), 'STRUCTURAL_NULL_CHANGED')
    audit['applied_rows'] = len(applied_rows)
    audit['fallback_e14_rows'] = audit['applicable_rows'] - len(applied_rows)
    out.attrs['matchup_intelligence_v1'] = audit
    return audit
