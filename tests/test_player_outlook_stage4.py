"""Stage 4 contract tests; all mutated fixtures live in pytest's temporary tree."""

import ast
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

import player_outlook_data as data
from components import player_outlook as public


ROOT = Path(__file__).resolve().parents[1]
CAPTURE_DIR = ROOT / "data/research/prospective/player_form_matchup_v1"
CAPTURE = CAPTURE_DIR / "2026_week_04_pregame.parquet"
MANIFEST = CAPTURE_DIR / "2026_week_04_pregame_manifest.json"
NOW = datetime(2026, 10, 5, 17, tzinfo=timezone.utc)
FROZEN = {
    "__init__.py": "5919156778a924594edc5525571cd2246876a87302346fa853d421498c337404",
    "admin.py": "d54abcdff86ecd425c689dc2d54cdf89d6fae21d9e4670235ab1d274d60b70f1",
    "build_lineups.py": "b0f2337537d6814904acd3056eb1647eb67ac866dca85bb9c02e9db8bda9466a",
    "data_center.py": "ec8d47c563318613bf6be47d5cf18170f8122d6b01c7d6f2d8e18b1399d114c0",
    "fantasy_league_hub.py": "9732fbe06cb48bc29e11b21b23b5f76d93333a7caa6578fc1fb59766cd48407f",
    "forecast_center.py": "d2bcbfe376b7844701760c5277f0869ccad30fb41db87d301552caea82d47044",
    "home.py": "15dcdc3acbd1e4c1f3c849b9eb20e28255e24d0e4c2ea3b5164e7fd850726df7",
    "wfs_analyst.py": "94a728d4ac2604cd564173fd6ac762a4aefc90bdab81d6eb0815bd4afad36eef",
}


@pytest.fixture
def source():
    return pd.read_parquet(CAPTURE)


@pytest.fixture
def game(source):
    return data.GameContext(
        2026, 4, "2026_04_ATL_NO", "ATL", "NO",
        datetime(2026, 10, 6, 0, 15, tzinfo=timezone.utc), False,
        tuple(data.PlayerIdentity(row.player_id, row.team, row.position) for row in source.itertuples()),
    )


@pytest.fixture
def altered(tmp_path):
    def make(frame=None, change_manifest=None):
        manifest = json.loads(MANIFEST.read_text())
        parquet = tmp_path / "capture.parquet"
        if frame is None:
            parquet.write_bytes(CAPTURE.read_bytes())
        else:
            frame.to_parquet(parquet, index=False)
        manifest["output"]["sha256"] = hashlib.sha256(parquet.read_bytes()).hexdigest()
        if change_manifest:
            change_manifest(manifest)
        target = tmp_path / "manifest.json"
        target.write_text(json.dumps(manifest))
        return parquet, target
    return make


def load(game, paths=(CAPTURE, MANIFEST), now=NOW):
    return data.load_player_outlook(game_context=game, artifact_path=paths[0], manifest_path=paths[1], now_utc=now)


def card_for(outlook, name):
    return next(card for card in public.build_public_player_cards(outlook).cards if card.name == name)


def test_correct_capture_and_no_dataframe_mutation(game, source, monkeypatch):
    before = source.copy(deep=True)
    monkeypatch.setattr(data.pd, "read_parquet", lambda _: source)
    result = load(game)
    assert result.available, result.reason
    assert len(result.players) == 37
    assert {p.position: sum(x.position == p.position for x in result.players) for p in result.players} == {"QB": 7, "RB": 8, "WR": 13, "TE": 9}
    pd.testing.assert_frame_equal(source, before)


@pytest.mark.parametrize("changes", [
    {"game_id": "2026_04_BUF_NE"}, {"season": 2025}, {"week": 3},
    {"away_team": "BUF"}, {"home_team": "ATL"},
    {"kickoff_utc": datetime(2026, 10, 6, 1, tzinfo=timezone.utc)},
    {"players": ()},
])
def test_wrong_context_fails_closed(game, changes):
    result = load(replace(game, **changes))
    assert not result.available
    assert public.build_public_player_cards(result) == public.PublicOutlook(False)


def test_valid_other_game_context_does_not_relabel_capture(game):
    other = replace(game, season=2025, week=3, game_id="2025_03_ATL_NO")
    assert not load(other).available


@pytest.mark.parametrize("missing", [0, 1])
def test_missing_file(game, tmp_path, missing):
    paths = [CAPTURE, MANIFEST]
    paths[missing] = tmp_path / "absent"
    assert load(game, paths).reason == "MISSING_CAPTURE"


@pytest.mark.parametrize("mutation", [
    lambda m: m["output"].update(sha256="0" * 64),
    lambda m: m.update(contract="wrong"),
    lambda m: m.update(status="SHADOW_COMPOSED"),
    lambda m: m.update(analysis_only=False),
    lambda m: m["safety"].update(production_influence=True),
    lambda m: m["safety"].update(production_influence=0),
    lambda m: m.update(week=4.0),
    lambda m: m.update(captured_at_utc="2026-10-06T00:15:00Z"),
    lambda m: m.update(captured_at_utc="2026-10-05T18:00:00Z"),
    lambda m: m.update(captured_at_utc="2026-10-05T07:50:52"),
    lambda m: m["games_detail"][0].update(home_team="BUF"),
    lambda m: m["games_detail"][0].update(kickoff_utc="2026-10-06T01:00:00Z"),
    lambda m: m["chronology"].update(all_games_future_at_capture=False),
    lambda m: m["source"].update(stage1_sha256="f" * 64),
    lambda m: m.update(rows=38),
])
def test_invalid_manifest(game, altered, mutation):
    assert not load(game, altered(change_manifest=mutation)).available


def test_malformed_files(game, tmp_path):
    bad = tmp_path / "bad"
    bad.write_text("{not json")
    assert not load(game, (CAPTURE, bad)).available
    manifest = json.loads(MANIFEST.read_text())
    manifest["output"]["sha256"] = hashlib.sha256(bad.read_bytes()).hexdigest()
    target = tmp_path / "manifest"
    target.write_text(json.dumps(manifest))
    assert not load(game, (bad, target)).available


def test_missing_required_schema(game, source, altered):
    assert load(game, altered(source.drop(columns="player_id"))).reason == "CAPTURE_SCHEMA"


def test_duplicate_player(game, source, altered):
    source.iloc[1] = source.iloc[0]
    assert load(game, altered(source)).reason == "DUPLICATE_PLAYER"


@pytest.mark.parametrize("field,value", [
    ("player_id", None), ("player_name", None), ("team", None),
    ("opponent_team", None), ("position", None), ("player_id", "00-9999999"),
    ("player_id", "unknown"), ("player_name", ""), ("position", "K"),
    ("team", "BUF"), ("opponent_team", "BUF"), ("week", 3),
    ("game_id", "2026_03_ATL_NO"), ("chronology_status", "UNKNOWN"),
    ("capture_mode", "MUTABLE"), ("built_at_utc", "2026-10-05T09:00:00Z"),
    ("target_kickoff_utc", "2026-10-06T01:00:00Z"),
    ("historical_available_at_cutoff_proven", True),
])
def test_invalid_row_identity_or_chronology(game, source, altered, field, value):
    source.loc[0, field] = value
    assert not load(game, altered(source)).available


def test_ambiguous_authoritative_roster(game):
    bad = replace(game.players[0], position="QB" if game.players[0].position != "QB" else "WR")
    assert not load(replace(game, players=game.players + (bad,))).available


@pytest.mark.parametrize("value,status", [(None, "AVAILABLE"), (float("inf"), "AVAILABLE"), (0.0, "NO_PRIOR_HISTORY")])
def test_optional_metric_is_omitted(game, source, altered, value, status):
    mask = source.player_name.eq("Alvin Kamara")
    source.loc[mask, "carries_avg_3"] = value
    source.loc[mask, "carries_avg_3_status"] = status
    result = load(game, altered(source))
    assert result.available
    card = card_for(result, "Alvin Kamara")
    assert "carries/game" not in card.usage
    assert "Recent carries" not in card.explanation


def test_missing_optional_columns_not_entire_game(game, source, altered):
    result = load(game, altered(source.drop(columns=["fd_avg_3", "targets_avg_3_status"])))
    assert result.available
    assert all(card.scoring == "Limited recent history." for card in public.build_public_player_cards(result).cards)


def test_bad_history_cannot_generate_neutral_claim(game, source, altered):
    mask = source.player_name.eq("Alvin Kamara")
    source.loc[mask, "player_window_game_ids"] = '["2026_04_ATL_NO"]'
    card = card_for(load(game, altered(source)), "Alvin Kamara")
    assert card.scoring == "Limited recent history."
    assert not card.usage
    assert "matches" not in card.explanation


def test_insufficient_windows_and_mismatched_direction(game, source, altered):
    mask = source.player_name.eq("Alvin Kamara")
    source.loc[mask, "production_direction_status"] = "INSUFFICIENT_DISTINCT_3_VS_5_WINDOWS"
    source.loc[mask, "carry_direction"] = "UP"
    card = card_for(load(game, altered(source)), "Alvin Kamara")
    assert "scoring is" not in card.explanation
    assert "Recent carries" not in card.explanation


def test_four_observations_do_not_claim_five_game_comparison(game, source, altered):
    mask = source.player_name.eq("Alvin Kamara")
    idx = source.index[mask][0]
    ids = json.loads(source.at[idx, "player_window_game_ids"])
    source.at[idx, "player_window_game_ids"] = json.dumps(ids[1:])
    source.at[idx, "player_history_games"] = 4
    source.at[idx, "production_window_observations_5"] = 4
    result = load(game, altered(source))
    assert result.available
    evidence = next(p for p in result.players if p.name == "Alvin Kamara")
    assert evidence.recent_games == 3 and evidence.baseline_games == 4
    assert not evidence.comparisons


def test_unavailable_opponent_field_omitted(game, source, altered):
    mask = source.player_name.eq("Alvin Kamara")
    source.loc[mask, "rushing_yards_allowed_avg_3_status"] = "DVP_CHRONOLOGY_OR_VALUE_UNPROVEN"
    card = card_for(load(game, altered(source)), "Alvin Kamara")
    assert "48 rushing yards" not in card.opponent_context
    assert "opposing RBs" in card.opponent_context


def test_qb_generic_opportunity_is_never_role(game, source, altered):
    mask = source.position.eq("QB")
    source.loc[mask, "opportunities_avg_3"] = 1000.0
    source.loc[mask, "opportunity_direction"] = "UP"
    source.loc[mask, "rising_role_flag"] = 1.0
    result = load(game, altered(source))
    assert result.available
    for card in public.build_public_player_cards(result).cards:
        if card.position == "QB":
            assert card.usage == ""
            assert "1000" not in str(card)
            assert not any(word in str(card).lower() for word in ["opportunities", "expanding", "contracting", "passing attempts", "sacks", "pressure"])


def test_rb_wording(game):
    card = card_for(load(game), "Alvin Kamara")
    assert "7 carries/game" in card.usage and "3 targets/game" in card.usage
    assert "ATL allowed opposing RBs 48 rushing yards" in card.opponent_context
    assert "Recent scoring is lower" in card.explanation
    assert "Recent carries are lower" in card.explanation
    assert "goal" not in str(card).lower()


def test_wr_wording(game):
    card = card_for(load(game), "Olamide Zaccheaus")
    assert "2 targets/game" in card.usage
    assert "NO allowed opposing WRs 17 targets" in card.opponent_context
    assert "Recent targets match his five-game average." in card.explanation


def test_te_wording(game):
    card = card_for(load(game), "Austin Hooper")
    assert "3 targets/game" in card.usage
    assert "NO allowed opposing TEs 8.3 receptions" in card.opponent_context
    assert "inside" not in str(card).lower()


def test_public_allowlist_and_language(game):
    payload = json.dumps(asdict(public.build_public_player_cards(load(game))))
    forbidden = ["elite", "breakout", "rebound", "smash", "guaranteed", "predicted ceiling", "production floor", "sha256", "mi_", "source_", "chronology", "manifest", "contract", "player_id", "00-", "/home/", "AVAILABLE", "WFS_PLAYER", "g2g", "i10"]
    assert all(word.lower() not in payload.lower() for word in forbidden if word != "AVAILABLE")
    assert "AVAILABLE" not in payload


@pytest.mark.parametrize("now,completed", [
    (NOW, False),
    (datetime(2026, 10, 6, 1, tzinfo=timezone.utc), False),
    (datetime(2026, 10, 7, tzinfo=timezone.utc), True),
])
def test_lifecycle_does_not_change_pregame_evidence(game, now, completed):
    baseline = public.build_public_player_cards(load(game))
    actual = public.build_public_player_cards(load(replace(game, completed=completed), now=now))
    assert actual == baseline
    assert "Pregame Outlook" in actual.game_label


def test_future_now_and_naive_now_rejected(game):
    assert not load(game, now=datetime(2026, 10, 5, 7, tzinfo=timezone.utc)).available
    assert not load(game, now=NOW.replace(tzinfo=None)).available


def test_internal_failure_not_public():
    assert public.build_public_player_cards(data.ValidatedOutlook(False, reason="/secret/path HASH_ERROR")) == public.PublicOutlook(False)


@pytest.fixture
def schedule_db(tmp_path, game):
    path = tmp_path / "schedule.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE games (season INTEGER, week INTEGER, game_id TEXT, game_type TEXT, away_team TEXT, home_team TEXT, game_date TEXT, gametime TEXT, completed INTEGER)")
        conn.execute("INSERT INTO games VALUES (2026,4,'2026_04_ATL_NO','REG','ATL','NO','2026-10-05','20:15',0)")
        conn.execute("CREATE TABLE weekly_rosters (season INTEGER, week INTEGER, game_type TEXT, gsis_id TEXT, team TEXT, position TEXT)")
        conn.executemany("INSERT INTO weekly_rosters VALUES (2026,4,'REG',?,?,?)", [(p.player_id, p.team, p.position) for p in game.players])
    return path


def test_schedule_read_exact_and_no_sidecars(schedule_db, game):
    before = schedule_db.read_bytes()
    files = set(schedule_db.parent.iterdir())
    actual = data.read_game_context(database_path=schedule_db, game_id=game.game_id)
    assert actual.kickoff_utc == game.kickoff_utc
    assert set(actual.players) == set(game.players)
    assert schedule_db.read_bytes() == before
    assert set(schedule_db.parent.iterdir()) == files
    with pytest.raises(data.InvalidOutlook):
        data.read_game_context(database_path=schedule_db, game_id="2026_04_BUF_NE")


@pytest.mark.parametrize("suffix", ["-wal", "-journal"])
def test_pending_journal_fails_closed(schedule_db, game, suffix):
    Path(str(schedule_db) + suffix).write_bytes(b"pending")
    with pytest.raises(data.InvalidOutlook, match="SCHEDULE_JOURNAL"):
        data.read_game_context(database_path=schedule_db, game_id=game.game_id)


def test_duplicate_schedule_rejected(schedule_db, game):
    with sqlite3.connect(schedule_db) as conn:
        conn.execute("INSERT INTO games SELECT * FROM games")
    with pytest.raises(data.InvalidOutlook, match="SCHEDULE_IDENTITY"):
        data.read_game_context(database_path=schedule_db, game_id=game.game_id)


def test_missing_schedule_does_not_create_file(tmp_path, game):
    path = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError):
        data.read_game_context(database_path=path, game_id=game.game_id)
    assert not path.exists()


def test_actual_schedule_authority_read_only():
    path = ROOT / "data/nfl.db"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = data.read_game_context(database_path=path, game_id="2026_04_ATL_NO")
    assert load(result).available
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def adapter_namespace():
    tree = ast.parse((ROOT / "app.py").read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_dc_render_game_with_player_outlook")
    namespace = {"_dc_render_live_game_ai": Mock(), "st": SimpleNamespace(caption=Mock()), "DATABASE_PATH": ROOT / "data/nfl.db", "APP_DIR": ROOT, "datetime": SimpleNamespace(now=lambda _: NOW), "ZoneInfo": lambda _: timezone.utc}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "adapter_test", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("failure", ["none", "schedule", "load", "wording", "render"])
def test_adapter_calls_existing_helper_once_and_isolates_failure(game, monkeypatch, failure):
    namespace = adapter_namespace()
    events = []
    namespace["_dc_render_live_game_ai"].side_effect = lambda *args, **kwargs: events.append("existing")
    def context(**kwargs):
        events.append("outlook")
        if failure == "schedule":
            raise ValueError("private backend details")
        return game
    monkeypatch.setattr(data, "read_game_context", context)
    if failure == "load":
        monkeypatch.setattr(data, "load_player_outlook", Mock(side_effect=ValueError("private")))
    if failure == "wording":
        monkeypatch.setattr(public, "build_public_player_cards", Mock(side_effect=ValueError("private")))
    render = Mock(side_effect=ValueError("private") if failure == "render" else None)
    monkeypatch.setattr(public, "render_player_outlook", render)
    namespace["_dc_render_game_with_player_outlook"](game.game_id, workspace_mode="Admin")
    namespace["_dc_render_live_game_ai"].assert_called_once_with(game.game_id, workspace_mode="Admin")
    assert events == ["existing", "outlook"]
    if failure != "none":
        namespace["st"].caption.assert_called_once_with(public.UNAVAILABLE)
    else:
        render.assert_called_once()


def test_adapter_only_wired_inside_data_center():
    tree = ast.parse((ROOT / "app.py").read_text())
    branch = next(n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "page == '🏈 NFL Data Center'")
    uses = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "_dc_render_game_with_player_outlook"]
    assert len(uses) == 1 and uses[0] in list(ast.walk(branch))
    calls = [n for n in ast.walk(branch) if isinstance(n, ast.Call) and ast.unparse(n.func) == "render_data_center"]
    assert len(calls) == 1
    assert next(k.value.id for k in calls[0].keywords if k.arg == "dc_render_live_game_ai") == "_dc_render_game_with_player_outlook"


def test_frozen_and_root_pages():
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "wfs_pages").glob("*.py")} == FROZEN
    assert not (ROOT / "pages").exists() and not (ROOT / "pages").is_symlink()


def test_modules_have_no_write_or_production_dependencies():
    for path in [ROOT / "player_outlook_data.py", ROOT / "components/player_outlook.py"]:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                assert not any(any(word in name for word in ("subprocess", "research", "solver", "optimizer", "projection", "portfolio", "eligibility", "fanduel", "config")) for name in names)
                if path.name == "player_outlook_data.py":
                    assert "streamlit" not in names
            if isinstance(node, ast.Call):
                call = ast.unparse(node.func)
                assert not any(call.endswith(word) for word in (".write_text", ".write_bytes", ".to_parquet", ".to_csv", ".to_sql", ".mkdir", ".unlink", ".rename", ".replace", ".system", ".Popen"))
                if call.endswith(".execute"):
                    assert isinstance(node.args[0], ast.Constant) and node.args[0].value.startswith("SELECT ")
        assert not any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) for n in tree.body)


def test_no_production_imports_new_modules():
    allowed = {ROOT / "app.py", ROOT / "player_outlook_data.py", ROOT / "components/player_outlook.py", Path(__file__).resolve()}
    for directory in [ROOT, ROOT / "wfs_pages"]:
        for path in directory.glob("*.py"):
            if path in allowed:
                continue
            source = path.read_text()
            assert "player_outlook_data" not in source and "components.player_outlook" not in source, path


def test_renderer_only_uses_public_data(game, monkeypatch):
    class FakeUI:
        def __init__(self):
            self.lines = []
            self.keys = []
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def caption(self, value): self.lines.append(value)
        def text(self, value): self.lines.append(value)
        def expander(self, *args, **kwargs): return self
        def container(self, **kwargs): return self
        def selectbox(self, label, options, **kwargs):
            self.keys.append(kwargs["key"])
            return options[0]
    ui = FakeUI()
    monkeypatch.setitem(sys.modules, "streamlit", ui)
    prepared = public.build_public_player_cards(load(game))
    monkeypatch.setattr(Path, "read_bytes", Mock(side_effect=AssertionError("renderer file read")))
    monkeypatch.setattr(sqlite3, "connect", Mock(side_effect=AssertionError("renderer database read")))
    public.render_player_outlook(outlook=prepared, key_prefix="game1")
    assert ui.keys == ["game1_group"]
    assert any("Pregame Outlook" in line for line in ui.lines)
    ui.lines.clear()
    public.render_player_outlook(outlook=public.PublicOutlook(False), key_prefix="game2")
    assert ui.lines == [public.UNAVAILABLE]
