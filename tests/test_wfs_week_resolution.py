"""Focused regression tests for League Hub weekly tracker week handling."""

import ast
from pathlib import Path


APP_SOURCE = Path(__file__).resolve().parents[1] / "app.py"


def _load_week_helper():
    tree = ast.parse(APP_SOURCE.read_text())
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_wfs_decision_current_week"
    )
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(APP_SOURCE), "exec"), namespace)
    return namespace[node.name]


def test_dictionary_team_data_uses_authoritative_espn_period():
    assert _load_week_helper()({"currentMatchupPeriod": 3}) == 3


def test_dictionary_week_fields_are_supported_without_attribute_access():
    helper = _load_week_helper()
    assert helper({"current_week": 4}) == 4
    assert helper({"week": 5}) == 5
    assert helper({"currentScoringPeriod": 6}) == 6


def test_explicit_authoritative_week_wins_over_stale_team_value():
    assert _load_week_helper()({"week": 1}, authoritative_week=3) == 3


def test_unresolved_week_is_explicit_and_never_week_one():
    assert _load_week_helper()({}) is None
    assert _load_week_helper()(None) is None


def test_object_style_week_input_remains_supported():
    class Team:
        current_week = 7

    assert _load_week_helper()(Team()) == 7


def test_tracker_source_passes_authoritative_week_and_uses_snapshot_week_for_grading():
    source = APP_SOURCE.read_text()
    assert "authoritative_week=current_week" in source
    assert "snapshot_week = snapshot_rows[0].get(\"week\") if snapshot_rows else week" in source
    assert "season, league_id, snapshot_week" in source


def test_week_is_part_of_snapshot_uniqueness_and_query_scope():
    source = APP_SOURCE.read_text()
    assert "ON CONFLICT(user_sub, provider, season, league_id, week, decision_key)" in source
    assert "AND league_id = ? AND week = ?" in source


def test_week_one_rows_are_not_relabelled():
    source = APP_SOURCE.read_text()
    assert "INSERT INTO wfs_weekly_decisions" in source
    assert "int(week)" in source
    assert "UPDATE wfs_weekly_decisions" in source
    assert "SET sit_actual = ?, start_actual = ?, actual_gain = ?" in source


def test_duplicate_capture_remains_idempotent():
    source = APP_SOURCE.read_text()
    assert "DO NOTHING" in source


def test_grading_uses_the_requested_stored_week_scope():
    source = APP_SOURCE.read_text()
    assert "_wfs_espn_actual_points_for_week(my_team, week)" in source
    assert "str(league_id), int(week)" in source


def test_missing_actuals_remain_pending():
    source = APP_SOURCE.read_text()
    assert "if sit_key not in actuals or start_key not in actuals:" in source
    assert "continue" in source


def test_explicit_zero_actual_is_preserved():
    source = APP_SOURCE.read_text()
    assert "value = stat.get(\"appliedTotal\")" in source
    assert "points = float(value)" in source


def test_existing_grading_direction_is_unchanged():
    source = APP_SOURCE.read_text()
    assert 'status = "HIT" if actual_gain > 0 else ("MISS" if actual_gain < 0 else "PUSH")' in source


def test_unresolved_display_does_not_claim_week_one():
    source = APP_SOURCE.read_text()
    assert 'week_label = f"Week {week}" if week is not None else "Current week unavailable"' in source
