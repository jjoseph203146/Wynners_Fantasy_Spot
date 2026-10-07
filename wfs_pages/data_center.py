"""NFL Data Center presentation page.

Behavior-preserving extraction from app.py.

Shared application helpers remain owned by app.py and are injected into
render_data_center(). This module does not alter projections, solver
authority, eligibility, Stage24, NFL Live, or data authority.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_data_center(
    workspace_mode="Public",
    *,
    nfl_team_names,
    dc_col,
    dc_game_status,
    dc_matchup_card,
    dc_matchup_has_team,
    dc_my_fantasy_players_by_team,
    dc_render_live_game_ai,
    dc_render_my_players_panel,
    dc_render_stat_table,
    dc_resolve_team_token,
    dc_score_text,
    dc_team_code,
    dc_team_logo_html,
    wfs_current_user,
    wfs_set_favorite_nfl_team,
    wfs_upsert_user,
    wfs_user_preferences,
    load_data_center_fn,
):
    sched, box, stbl, btbl = load_data_center_fn()

    dc_user = wfs_current_user()
    favorite_team = ""
    dc_my_players_by_team = {}

    if dc_user:
        try:
            wfs_upsert_user(dc_user)
            prefs = wfs_user_preferences(dc_user["sub"])
            favorite_team = dc_team_code(prefs.get("favorite_nfl_team", ""))

            # Presentation-only ESPN roster → canonical NFL team counts.
            # Exact ESPN player-ID matching through weekly_rosters.
            # Independent of box-score availability; fail closed on ambiguity.
            try:
                dc_my_players_by_team = dc_my_fantasy_players_by_team(
                    dc_user["sub"]
                )
            except Exception:
                dc_my_players_by_team = {}

        except Exception:
            favorite_team = ""
            dc_my_players_by_team = {}

    st.markdown(
        """
        <style>
        .wfs-dc-hero {
            background:linear-gradient(110deg,#0b2f68 0%,#0b5ec7 72%,#0874df 100%);
            border-radius:16px;padding:1.05rem 1.25rem;margin-bottom:.85rem;
            color:white;box-shadow:0 8px 24px rgba(15,55,105,.12);
        }
        .wfs-dc-hero h2 {margin:0;font-size:1.35rem;font-weight:900;}
        .wfs-dc-hero p {margin:.28rem 0 0;color:#d9eaff;font-size:.86rem;}
        .wfs-game-card {
            background:#fff;border:1px solid #dbe5f1;border-radius:14px;
            padding:.85rem .95rem;margin-bottom:.72rem;
            box-shadow:0 3px 10px rgba(25,55,90,.05);
            min-height:176px;
        }
        .wfs-game-card.favorite {
            border:2px solid #f2b632;
            background:linear-gradient(180deg,#fffdf5 0%,#ffffff 100%);
            box-shadow:0 5px 16px rgba(242,182,50,.14);
        }
        .wfs-game-top-right {
            display:flex;align-items:center;gap:.35rem;flex-wrap:wrap;justify-content:flex-end;
        }
        .wfs-my-team-pill {
            display:inline-flex;align-items:center;border-radius:999px;
            padding:.18rem .48rem;background:#fff4cc;color:#7a5200;
            font-size:.6rem;font-weight:900;letter-spacing:.05em;
        }
        .wfs-my-players-pill {
            display:inline-flex;
            align-items:center;
            border-radius:999px;
            padding:.18rem .48rem;
            background:#eaf3ff;
            color:#075db7;
            border:1px solid #cfe2fb;
            font-size:.6rem;
            font-weight:900;
            letter-spacing:.04em;
            white-space:nowrap;
        }
        .wfs-team-row.favorite-team {
            background:#fff9e8;border-radius:8px;padding-left:.35rem;padding-right:.35rem;
        }
        .wfs-team-star {
            margin-left:.35rem;font-size:.72rem;
        }
        .wfs-game-top {display:flex;justify-content:space-between;align-items:center;
            color:#64748b;font-size:.68rem;font-weight:850;letter-spacing:.08em;}
        .wfs-game-status {border-radius:999px;padding:.18rem .48rem;font-size:.62rem;}
        .wfs-game-status.upcoming {background:#eaf3ff;color:#075db7;}
        .wfs-game-status.final {background:#e8f8ef;color:#08783e;}
        .wfs-game-date {color:#7b8ca3;font-size:.72rem;margin:.28rem 0 .55rem;}
        .wfs-team-row {display:flex;justify-content:space-between;align-items:center;
            padding:.25rem 0;border-top:1px solid #f1f5f9;}
        .wfs-team-badge {display:inline-flex;align-items:center;justify-content:center;
            width:38px;height:28px;border-radius:7px;background:#eef5ff;color:#0b57a5;
            font-size:.7rem;font-weight:900;margin-right:.55rem;}
        .wfs-team-label {font-size:.9rem;font-weight:850;color:#172b45;}
        .wfs-score {font-size:1.05rem;font-weight:900;color:#0b4f9c;}
        .wfs-team-right {
            display:flex;
            flex-direction:column;
            align-items:flex-end;
            justify-content:center;
            gap:.05rem;
        }
        .wfs-pass-expectation {
            color:#6f8299;
            font-size:.62rem;
            font-weight:800;
            white-space:nowrap;
        }
        .wfs-game-id {margin-top:.4rem;color:#9aa9bb;font-size:.61rem;}
        .wfs-box-hero {background:#fff;border:1px solid #dbe5f1;border-radius:15px;
            padding:1rem 1.1rem;margin:.5rem 0 .9rem;box-shadow:0 3px 10px rgba(25,55,90,.05);}
        .wfs-team-identity {
            display:flex;
            align-items:center;
            gap:.48rem;
            min-width:0;
        }
        .wfs-team-logo {
            width:34px;
            height:34px;
            object-fit:contain;
            flex:0 0 34px;
        }
        .wfs-box-matchup {
            display:flex;
            align-items:center;
            gap:.7rem;
            flex-wrap:wrap;
            font-size:1.25rem;
            font-weight:900;
            color:#17365d;
        }
        .wfs-box-team {
            display:inline-flex;
            align-items:center;
            gap:.42rem;
            white-space:nowrap;
        }
        .wfs-box-team-logo {
            width:38px;
            height:38px;
            object-fit:contain;
            flex:0 0 38px;
        }
        .wfs-box-vs {
            color:#8293a7;
            font-weight:800;
        }
        .wfs-box-hero-row {
            display:flex;
            justify-content:space-between;
            align-items:center;
            gap:1rem;
        }
        .wfs-box-status {
            font-weight:800;
            letter-spacing:.08em;
        }
        .wfs-box-winner {
            background:#e8f8ef;
            color:#08783e;
            border-radius:999px;
            padding:.35rem .65rem;
            font-size:.68rem;
            font-weight:900;
            white-space:nowrap;
        }
        .wfs-box-sub {color:#6f8299;font-size:.76rem;margin-top:.2rem;}
        </style>
        <div class="wfs-dc-hero">
          <h2>🏈 NFL Data Center</h2>
          <p>Schedules, results, box scores and fantasy performance from around the NFL.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if workspace_mode == "Admin":
        a, b, c = st.columns(3)
        a.metric("Schedule Games", f"{len(sched):,}")
        b.metric("Player-Game Rows", f"{len(box):,}")
        c.metric("Data Status", "READY" if len(sched) else "CHECK DB")

    team_options = [""] + list(nfl_team_names.keys())

    def _favorite_team_label(code):
        if not code:
            return "No favorite team"
        return f"{code} — {nfl_team_names.get(code, code)}"

    if dc_user:
        current_index = (
            team_options.index(favorite_team)
            if favorite_team in team_options else 0
        )

        selected_favorite_team = st.selectbox(
            "⭐ My NFL Team",
            team_options,
            index=current_index,
            format_func=_favorite_team_label,
            key="dc_favorite_nfl_team",
            help="Your team will be highlighted throughout the NFL Data Center.",
        )

        selected_favorite_team = dc_team_code(selected_favorite_team)

        if selected_favorite_team != favorite_team:
            wfs_set_favorite_nfl_team(
                dc_user["sub"],
                selected_favorite_team,
            )
            favorite_team = selected_favorite_team
    else:
        st.caption("Sign in to save and highlight your favorite NFL team.")

    if workspace_mode == "Admin":
        st.markdown("### 🔒 Admin Postgame Feedback")
        from wfs_postgame_feedback import render_postgame_feedback_admin
        render_postgame_feedback_admin()
        st.divider()

    schedule_tab, box_tab, leaders_tab = st.tabs(
        ["📅 Schedule", "📊 Box Scores", "🏆 Weekly Leaders"]
    )

    with schedule_tab:
        if sched.empty:
            st.info("No NFL schedule is available yet.")
        else:
            season = dc_col(sched, ["season", "season_year", "year"])
            week = dc_col(sched, ["week", "week_num", "week_number"])
            away = dc_col(sched, ["away_team", "away"])
            home = dc_col(sched, ["home_team", "home"])
            date = dc_col(sched, ["gameday", "game_date", "date", "start_time", "kickoff"])
            typ = dc_col(sched, ["game_type", "season_type", "type", "season_stage"])
            ascore = dc_col(sched, ["away_score", "away_points", "away_pts"])
            hscore = dc_col(sched, ["home_score", "home_points", "home_pts"])
            gid = dc_col(sched, ["game_id", "gameid"])
            work = sched.copy()

            f1, f2, f3, f4 = st.columns([1, 1, 1.4, 1])
            if season:
                seasons = sorted(
                    pd.to_numeric(work[season], errors="coerce").dropna().astype(int).unique(),
                    reverse=True,
                )
                chosen = f1.selectbox("Season", seasons, key="dc_sched_season")
                work = work[pd.to_numeric(work[season], errors="coerce").eq(chosen)]

            if week:
                # Numeric week ordering: 1, 2, 3 ... 18.
                week_values = work[week].dropna().astype(str).unique().tolist()

                def _week_sort_key(value):
                    try:
                        return (0, int(float(str(value))))
                    except Exception:
                        return (1, str(value))

                weeks = sorted(week_values, key=_week_sort_key)
                week_options = list(weeks) + ["All Weeks"]

                # WFS_DATA_CENTER_SCHEDULE_WEEK_V1
                # Initialize from the authoritative NFL planning week.
                # Manual Data Center week selection remains sticky afterward.
                from wfs_schedule_context import (
                    ScheduleContextError,
                    resolve_schedule_week_context,
                )

                try:
                    dc_schedule_context = resolve_schedule_week_context(
                        season=int(chosen) if season else None,
                    )
                    dc_default_week = str(dc_schedule_context.planning_week)
                except ScheduleContextError as exc:
                    if "No current or future REG week for " not in str(exc):
                        raise
                    dc_default_week = weeks[-1] if weeks else "All Weeks"

                if dc_default_week not in weeks:
                    dc_default_week = weeks[-1] if weeks else "All Weeks"

                if "dc_selected_week" not in st.session_state:
                    st.session_state["dc_selected_week"] = dc_default_week

                if st.session_state["dc_selected_week"] not in week_options:
                    st.session_state["dc_selected_week"] = dc_default_week

                chosen_week = f2.selectbox(
                    "Week",
                    week_options,
                    index=week_options.index(st.session_state["dc_selected_week"]),
                    key="dc_sched_week",
                )
                st.session_state["dc_selected_week"] = chosen_week

                if chosen_week != "All Weeks":
                    work = work[work[week].astype(str).eq(chosen_week)]

            # QA F4 / WFS_DATA_CENTER_MULTI_TEAM_FILTER_V1
            # QA F15: team filter tokens also accept full team names and
            # nicknames (e.g. "Falcons", "Atlanta Falcons"), resolved via
            # the existing nfl_team_names canonical mapping. Exact-code
            # behavior is unchanged.
            team_filter = f3.text_input(
                "Team filter", placeholder="BUF, PHI, KC...", key="dc_team"
            )
            team_codes = {
                dc_resolve_team_token(token)
                for token in str(team_filter).split(",")
                if dc_resolve_team_token(token)
            }
            if team_codes and away and home:
                away_codes = work[away].map(dc_team_code)
                home_codes = work[home].map(dc_team_code)
                work = work[
                    away_codes.isin(team_codes)
                    | home_codes.isin(team_codes)
                ]

            view = f4.selectbox("View", ["Matchup Cards", "Data Table"], key="dc_sched_view")

            if date:
                try:
                    work = work.assign(
                        __sort_date=pd.to_datetime(work[date], errors="coerce")
                    ).sort_values(["__sort_date", gid] if gid else ["__sort_date"])
                except Exception:
                    pass

            if view == "Matchup Cards":
                card_cols = {
                    "away": away, "home": home, "date": date, "week": week,
                    "away_score": ascore, "home_score": hscore, "gid": gid,
                }

                if favorite_team and away and home and len(work):
                    work = work.copy()
                    work["__favorite_game"] = work.apply(
                        lambda r: 1 if dc_matchup_has_team(
                            r, card_cols, favorite_team
                        ) else 0,
                        axis=1,
                    )

                    sort_cols = ["__favorite_game"]
                    ascending = [False]

                    if "__sort_date" in work.columns:
                        sort_cols.append("__sort_date")
                        ascending.append(True)

                    work = work.sort_values(
                        sort_cols,
                        ascending=ascending,
                        kind="stable",
                    ).drop(columns="__favorite_game")

                rows = list(work.iterrows())
                for i in range(0, len(rows), 3):
                    grid = st.columns(3)
                    for j, (_, row) in enumerate(rows[i:i + 3]):
                        with grid[j]:
                            card_away = dc_team_code(
                                row[card_cols["away"]]
                            )
                            card_home = dc_team_code(
                                row[card_cols["home"]]
                            )
                            card_game_id = str(
                                row[card_cols["gid"]]
                            )
                            card_week = (
                                row[card_cols["week"]]
                                if card_cols.get("week")
                                else chosen_week
                            )
                            card_my_player_count = (
                                int(
                                    dc_my_players_by_team.get(
                                        card_away,
                                        0,
                                    )
                                )
                                +
                                int(
                                    dc_my_players_by_team.get(
                                        card_home,
                                        0,
                                    )
                                )
                            )

                            st.markdown(
                                dc_matchup_card(
                                    row,
                                    card_cols,
                                    favorite_team=favorite_team,
                                    my_player_count=0,
                                ),
                                unsafe_allow_html=True,
                            )

                            # WFS_LIVE_GAME_AI_UI_V1
                            dc_render_live_game_ai(
                                card_game_id,
                                workspace_mode=workspace_mode,
                            )

                            if (
                                card_my_player_count > 0
                                and dc_user
                            ):
                                player_word = (
                                    "PLAYER"
                                    if card_my_player_count == 1
                                    else "PLAYERS"
                                )

                                with st.expander(
                                    (
                                        f"⭐ {card_my_player_count} "
                                        f"MY {player_word}"
                                    ),
                                    expanded=False,
                                ):
                                    dc_render_my_players_panel(
                                        dc_user["sub"],
                                        card_game_id,
                                        card_away,
                                        card_home,
                                        card_week,
                                    )
            else:
                display = pd.DataFrame(index=work.index)
                for label, col in [
                    ("Date", date), ("Type", typ), ("Week", week), ("Away", away),
                    ("Away Score", ascore), ("Home", home), ("Home Score", hscore),
                    ("Game ID", gid),
                ]:
                    if col:
                        display[label] = work[col]
                st.dataframe(
                    (display if not display.empty else work).reset_index(drop=True),
                    width="stretch", hide_index=True,
                )

            st.caption(f"{len(work):,} game(s) shown.")

    with box_tab:
        if sched.empty or box.empty:
            st.info("Box scores will appear when game results and player statistics are available.")
        else:
            sgid = dc_col(sched, ["game_id", "gameid"])
            bgid = dc_col(box, ["game_id", "gameid"])
            season = dc_col(sched, ["season", "season_year", "year"])
            week = dc_col(sched, ["week", "week_num", "week_number"])
            away = dc_col(sched, ["away_team", "away"])
            home = dc_col(sched, ["home_team", "home"])
            ascore = dc_col(sched, ["away_score", "away_points", "away_pts"])
            hscore = dc_col(sched, ["home_score", "home_points", "home_pts"])

            if not sgid or not bgid:
                st.info("Box scores are temporarily unavailable for the selected games.")
            else:
                games = sched.copy()
                f1, f2 = st.columns(2)

                if season:
                    seasons = sorted(
                        pd.to_numeric(games[season], errors="coerce").dropna().astype(int).unique(),
                        reverse=True,
                    )
                    chosen = f1.selectbox("Season", seasons, key="dc_box_season")
                    games = games[pd.to_numeric(games[season], errors="coerce").eq(chosen)]

                if week:
                    week_values = games[week].dropna().astype(str).unique().tolist()

                    def _box_week_sort_key(value):
                        try:
                            return (0, int(float(str(value))))
                        except Exception:
                            return (1, str(value))

                    weeks = sorted(week_values, key=_box_week_sort_key)
                    week_options = list(weeks) + ["All Weeks"]

                    # WFS_DATA_CENTER_BOX_SCORES_LAST_PLAYED_WEEK_V1
                    # Box Scores show completed games, so default to the
                    # most recent week that has final scores rather than
                    # the Schedule tab's forward-looking planning week.
                    try:
                        if ascore and hscore:
                            played_mask = (
                                pd.to_numeric(games[ascore], errors="coerce").notna()
                                & pd.to_numeric(games[hscore], errors="coerce").notna()
                            )
                        else:
                            played_mask = pd.Series(False, index=games.index)

                        played_weeks = (
                            games.loc[played_mask, week]
                            .dropna()
                            .astype(str)
                            .tolist()
                        )
                        dc_default_week = (
                            sorted(played_weeks, key=_box_week_sort_key)[-1]
                            if played_weeks
                            else (weeks[-1] if weeks else "All Weeks")
                        )
                    except Exception:
                        dc_default_week = weeks[-1] if weeks else "All Weeks"

                    if dc_default_week not in weeks:
                        dc_default_week = weeks[-1] if weeks else "All Weeks"

                    if "dc_box_selected_week" not in st.session_state:
                        st.session_state["dc_box_selected_week"] = dc_default_week

                    if st.session_state["dc_box_selected_week"] not in week_options:
                        st.session_state["dc_box_selected_week"] = dc_default_week

                    chosen_week = f2.selectbox(
                        "Week",
                        week_options,
                        index=week_options.index(st.session_state["dc_box_selected_week"]),
                        key="dc_box_week",
                    )
                    st.session_state["dc_box_selected_week"] = chosen_week

                    if chosen_week != "All Weeks":
                        games = games[games[week].astype(str).eq(chosen_week)]

                labels = {}
                schedule_rows = {}
                for _, r in games.iterrows():
                    game_id = str(r[sgid])
                    av = str(r[away]) if away else "Away"
                    hv = str(r[home]) if home else "Home"
                    prefix = f"W{r[week]} • " if week else ""
                    score = ""
                    if ascore and hscore and pd.notna(r[ascore]) and pd.notna(r[hscore]):
                        score = f" • {av} {dc_score_text(r[ascore])} — {hv} {dc_score_text(r[hscore])}"
                    label = f"{prefix}{av} @ {hv}{score}"
                    labels[label] = game_id
                    schedule_rows[game_id] = r

                if labels:
                    label = st.selectbox("Select Game", list(labels), key="dc_game")
                    game_id = labels[label]
                    stats = box[box[bgid].astype(str).eq(game_id)].copy()
                    game_row = schedule_rows[game_id]

                    av = str(game_row[away]) if away else "Away"
                    hv = str(game_row[home]) if home else "Home"
                    avs = dc_score_text(game_row[ascore]) if ascore else "—"
                    hvs = dc_score_text(game_row[hscore]) if hscore else "—"
                    status = dc_game_status(
                        game_row[ascore] if ascore else None,
                        game_row[hscore] if hscore else None,
                    )

                    winner_note = ""
                    try:
                        a_num = float(game_row[ascore]) if ascore and pd.notna(game_row[ascore]) else None
                        h_num = float(game_row[hscore]) if hscore and pd.notna(game_row[hscore]) else None
                        if a_num is not None and h_num is not None:
                            if a_num > h_num:
                                winner_note = f"{av} WIN"
                            elif h_num > a_num:
                                winner_note = f"{hv} WIN"
                            else:
                                winner_note = "TIE"
                    except Exception:
                        winner_note = ""

                    winner_html = (
                        f'<div class="wfs-box-winner">{winner_note}</div>'
                        if winner_note
                        else ""
                    )

                    away_box_logo = dc_team_logo_html(
                        av, "wfs-box-team-logo"
                    )
                    home_box_logo = dc_team_logo_html(
                        hv, "wfs-box-team-logo"
                    )

                    box_hero_html = (
                        f'<div class="wfs-box-hero">'
                        f'<div class="wfs-box-hero-row">'
                        f'<div>'
                        f'<div class="wfs-box-sub wfs-box-status">{status}</div>'
                        f'<div class="wfs-box-matchup">'
                        f'<span class="wfs-box-team">{away_box_logo}<span>{av} {avs}</span></span>'
                        f'<span class="wfs-box-vs">—</span>'
                        f'<span class="wfs-box-team">{home_box_logo}<span>{hv} {hvs}</span></span>'
                        f'</div>'
                        f'</div>'
                        f'{winner_html}'
                        f'</div>'
                        f'</div>'
                    )

                    st.markdown(
                        box_hero_html,
                        unsafe_allow_html=True,
                    )

                    if stats.empty:
                        st.info("Player statistics are not available for this game yet.")
                    else:
                        # QA D12: prefer the canonical full display name over the
                        # abbreviated source "player_name" (e.g. "Bi.Robinson"),
                        # matching the priority already used by the Weekly Leaders
                        # table below. Presentation only -- no identity change.
                        p = dc_col(stats, ["player_display_name", "player_name", "player", "display_name", "fantasy_player_name"])
                        tm = dc_col(stats, ["team", "recent_team", "posteam"])
                        pos = dc_col(stats, ["position", "pos"])

                        passing = ["passing_attempts", "completions", "passing_yards",
                                   "passing_tds", "interceptions", "sacks"]
                        rushing = ["carries", "rushing_yards", "rushing_tds"]
                        receiving = ["targets", "receptions", "receiving_yards", "receiving_tds"]
                        fantasy = ["fumbles_lost", "fantasy_points", "fantasy_points_ppr",
                                   "fd_points", "fanduel_points"]

                        identity = [p, tm, pos]

                        passing = [
                            "passing_attempts", "attempts", "completions",
                            "passing_yards", "passing_tds", "interceptions", "sacks",
                        ]
                        rushing = [
                            "carries", "rushing_yards", "rushing_tds",
                        ]
                        receiving = [
                            "targets", "receptions", "receiving_yards", "receiving_tds",
                        ]
                        fantasy = [
                            "fumbles_lost", "fd_points", "fanduel_points",
                            "fantasy_points", "fantasy_points_ppr",
                        ]

                        t1, t2, t3, t4, t5 = st.tabs(
                            ["⭐ Leaders", "Passing", "Rushing", "Receiving", "Fantasy"]
                        )

                        with t1:
                            lead_a, lead_b, lead_c = st.columns(3)

                            def leader_card(container, title, candidates):
                                sort_col = None
                                for candidate in candidates:
                                    found = dc_col(stats, [candidate])
                                    if found:
                                        sort_col = found
                                        break
                                if not sort_col or not p:
                                    container.metric(title, "—")
                                    return
                                values = pd.to_numeric(stats[sort_col], errors="coerce")
                                if values.notna().any():
                                    idx = values.idxmax()
                                    player_name = str(stats.loc[idx, p])
                                    value = values.loc[idx]
                                    shown = int(value) if float(value).is_integer() else round(float(value), 2)
                                    # QA D11: this is a static leader value, not a
                                    # change/delta, so suppress the arrow and color
                                    # Streamlit otherwise infers from a numeric delta.
                                    container.metric(
                                        title,
                                        player_name,
                                        str(shown),
                                        delta_color="off",
                                        delta_arrow="off",
                                    )
                                else:
                                    container.metric(title, "—")

                            leader_card(lead_a, "Passing Leader", ["passing_yards"])
                            leader_card(lead_b, "Rushing Leader", ["rushing_yards"])
                            leader_card(lead_c, "Receiving Leader", ["receiving_yards"])

                            st.markdown("#### Fantasy Leaders")
                            dc_render_stat_table(
                                stats, identity, fantasy,
                                ["fd_points", "fanduel_points", "fantasy_points", "fantasy_points_ppr"],
                            )

                        with t2:
                            dc_render_stat_table(
                                stats, identity, passing,
                                ["passing_yards", "passing_tds"],
                            )
                        with t3:
                            dc_render_stat_table(
                                stats, identity, rushing,
                                ["rushing_yards", "carries"],
                            )
                        with t4:
                            dc_render_stat_table(
                                stats, identity, receiving,
                                ["receiving_yards", "targets"],
                            )
                        with t5:
                            dc_render_stat_table(
                                stats, identity, fantasy,
                                ["fd_points", "fanduel_points", "fantasy_points", "fantasy_points_ppr"],
                            )

                        st.caption(
                            f"{len(stats):,} player performances shown."
                        )
                else:
                    st.info("No games match the selected filters.")

    # WFS_WEEKLY_NFL_LEADERS_V1
    # League-wide completed weekly player leaders from the same authoritative
    # player_game_stats dataframe already loaded by the NFL Data Center.
    with leaders_tab:
        if box.empty:
            st.info("Weekly leaders will appear when completed player statistics are available.")
        else:
            leaders = box.copy()

            lseason = dc_col(leaders, ["season", "season_year", "year"])
            lweek = dc_col(leaders, ["week", "week_num", "week_number"])
            ltype = dc_col(leaders, ["season_type", "game_type", "type"])
            lname = dc_col(
                leaders,
                ["player_display_name", "player_name", "player", "display_name",
                 "fantasy_player_name"],
            )
            lteam = dc_col(leaders, ["team", "recent_team", "posteam"])

            if not lseason or not lweek or not lname or not lteam:
                st.info("Weekly leader data is temporarily unavailable.")
            else:
                if ltype:
                    season_type = leaders[ltype].fillna("").astype(str).str.upper()
                    regular_mask = season_type.isin(["REG", "REGULAR", "REGULAR SEASON"])
                    if regular_mask.any():
                        leaders = leaders[regular_mask].copy()

                season_numeric = pd.to_numeric(leaders[lseason], errors="coerce")
                available_seasons = sorted(
                    season_numeric.dropna().astype(int).unique().tolist(),
                    reverse=True,
                )

                if not available_seasons:
                    st.info("No completed NFL weeks are available yet.")
                else:
                    control_a, control_b = st.columns(2)

                    with control_a:
                        selected_leader_season = st.selectbox(
                            "Season",
                            available_seasons,
                            index=0,
                            key="dc_weekly_leaders_season",
                        )

                    season_rows = leaders[
                        season_numeric.eq(int(selected_leader_season))
                    ].copy()

                    week_numeric = pd.to_numeric(
                        season_rows[lweek], errors="coerce"
                    )
                    available_weeks = sorted(
                        week_numeric.dropna().astype(int).unique().tolist(),
                        reverse=True,
                    )

                    with control_b:
                        selected_leader_week = st.selectbox(
                            "Week",
                            available_weeks,
                            index=0,
                            key="dc_weekly_leaders_week",
                        )

                    week_rows = season_rows[
                        week_numeric.eq(int(selected_leader_week))
                    ].copy()

                    st.caption(
                        f"Top individual performances • "
                        f"{int(selected_leader_season)} Week "
                        f"{int(selected_leader_week)}"
                    )

                    def _leader_num(frame, names):
                        col = dc_col(frame, names)
                        if not col:
                            return pd.Series(0.0, index=frame.index, dtype=float)
                        return pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

                    def _leader_logo(team):
                        return dc_team_logo_html(
                            dc_team_code(team),
                            "wfs-leader-team-logo",
                        )

                    def _leader_player_html(row):
                        team = dc_team_code(row.get(lteam, ""))
                        name = str(row.get(lname, "") or "").strip()
                        logo = _leader_logo(team)
                        return (
                            '<div class="wfs-leader-player">'
                            f'{logo}'
                            '<div>'
                            f'<div class="wfs-leader-name">{name}</div>'
                            f'<div class="wfs-leader-team">{team}</div>'
                            '</div>'
                            '</div>'
                        )

                    def _fmt_int(value):
                        try:
                            return f"{int(round(float(value))):,}"
                        except Exception:
                            return "—"

                    def _fmt_one(value):
                        try:
                            return f"{float(value):.1f}"
                        except Exception:
                            return "—"

                    def _fmt_pct(value):
                        try:
                            return f"{float(value):.1f}%"
                        except Exception:
                            return "—"

                    def _render_weekly_leader_table(
                        frame,
                        columns,
                        sort_key,
                        limit=10,
                    ):
                        if frame.empty:
                            st.info("No qualifying performances are available.")
                            return

                        shown = (
                            frame.sort_values(
                                sort_key,
                                ascending=False,
                                kind="stable",
                            )
                            .head(limit)
                            .copy()
                        )

                        header = "".join(
                            f'<div class="wfs-leader-th">{label}</div>'
                            for label, _, _ in columns
                        )

                        rows_html = []
                        for rank, (_, row) in enumerate(shown.iterrows(), start=1):
                            cells = []
                            for _, field, formatter in columns:
                                if field == "__player__":
                                    value = _leader_player_html(row)
                                else:
                                    value = formatter(row.get(field, 0))
                                cells.append(
                                    f'<div class="wfs-leader-td">{value}</div>'
                                )

                            rows_html.append(
                                '<div class="wfs-leader-row">'
                                f'<div class="wfs-leader-rank">{rank}</div>'
                                + "".join(cells)
                                + '</div>'
                            )

                        st.markdown(
                            '<div class="wfs-leader-table">'
                            '<div class="wfs-leader-header">'
                            '<div class="wfs-leader-rank">#</div>'
                            f'{header}'
                            '</div>'
                            + "".join(rows_html)
                            + '</div>',
                            unsafe_allow_html=True,
                        )

                    passing = week_rows.copy()
                    passing["_attempts"] = _leader_num(
                        passing, ["attempts", "passing_attempts"]
                    )
                    passing["_completions"] = _leader_num(
                        passing, ["completions", "passing_completions"]
                    )
                    passing["_pass_yards"] = _leader_num(
                        passing, ["passing_yards"]
                    )
                    passing["_pass_tds"] = _leader_num(
                        passing, ["passing_tds"]
                    )
                    passing["_ints"] = _leader_num(
                        passing, ["interceptions", "passing_interceptions"]
                    )
                    passing["_fd"] = _leader_num(
                        passing, ["fanduel_points", "fd_points"]
                    )
                    passing = passing[passing["_attempts"] > 0].copy()
                    passing["_comp_pct"] = (
                        passing["_completions"]
                        .div(passing["_attempts"])
                        .mul(100.0)
                    )

                    rushing = week_rows.copy()
                    rushing["_carries"] = _leader_num(
                        rushing, ["carries", "rushing_attempts"]
                    )
                    rushing["_rush_yards"] = _leader_num(
                        rushing, ["rushing_yards"]
                    )
                    rushing["_rush_tds"] = _leader_num(
                        rushing, ["rushing_tds"]
                    )
                    rushing["_fd"] = _leader_num(
                        rushing, ["fanduel_points", "fd_points"]
                    )
                    rushing = rushing[rushing["_carries"] > 0].copy()
                    rushing["_ypc"] = rushing["_rush_yards"].div(
                        rushing["_carries"]
                    )

                    receiving = week_rows.copy()
                    receiving["_targets"] = _leader_num(
                        receiving, ["targets"]
                    )
                    receiving["_receptions"] = _leader_num(
                        receiving, ["receptions"]
                    )
                    receiving["_rec_yards"] = _leader_num(
                        receiving, ["receiving_yards"]
                    )
                    receiving["_rec_tds"] = _leader_num(
                        receiving, ["receiving_tds"]
                    )
                    receiving["_fd"] = _leader_num(
                        receiving, ["fanduel_points", "fd_points"]
                    )
                    receiving = receiving[
                        (receiving["_targets"] > 0)
                        | (receiving["_receptions"] > 0)
                    ].copy()

                    st.markdown(
                        """
                        <style>
                        .wfs-leader-table {
                            width:100%;
                            overflow-x:auto;
                            border:1px solid #dbe5f1;
                            border-radius:14px;
                            background:#fff;
                            margin:.45rem 0 1rem;
                        }
                        .wfs-leader-header,
                        .wfs-leader-row {
                            display:grid;
                            grid-template-columns:38px minmax(190px,2fr)
                                repeat(6,minmax(76px,1fr));
                            align-items:center;
                            min-width:780px;
                        }
                        .wfs-leader-header {
                            background:#f5f8fc;
                            color:#64748b;
                            font-size:.67rem;
                            font-weight:900;
                            letter-spacing:.035em;
                            border-bottom:1px solid #dbe5f1;
                        }
                        .wfs-leader-row {
                            border-bottom:1px solid #edf2f7;
                            color:#17365d;
                        }
                        .wfs-leader-row:last-child {border-bottom:none;}
                        .wfs-leader-rank,
                        .wfs-leader-th,
                        .wfs-leader-td {
                            padding:.58rem .48rem;
                        }
                        .wfs-leader-rank {
                            text-align:center;
                            font-weight:900;
                            color:#7b8ca3;
                        }
                        .wfs-leader-player {
                            display:flex;
                            align-items:center;
                            gap:.55rem;
                            min-width:0;
                        }
                        .wfs-leader-team-logo {
                            width:26px;
                            height:26px;
                            object-fit:contain;
                            flex:0 0 26px;
                        }
                        .wfs-leader-name {
                            font-size:.82rem;
                            font-weight:900;
                            color:#17365d;
                            white-space:nowrap;
                        }
                        .wfs-leader-team {
                            color:#7b8ca3;
                            font-size:.64rem;
                            font-weight:800;
                        }
                        .wfs-leader-td {
                            font-size:.76rem;
                            font-weight:800;
                        }
                        @media (max-width: 700px) {
                            .wfs-leader-header,
                            .wfs-leader-row {
                                min-width:740px;
                            }
                            .wfs-leader-team-logo {
                                width:24px;
                                height:24px;
                                flex-basis:24px;
                            }
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )

                    pass_tab, rush_tab, receive_tab = st.tabs(
                        ["🏈 Passing", "🏃 Rushing", "🎯 Receiving"]
                    )

                    with pass_tab:
                        _render_weekly_leader_table(
                            passing,
                            [
                                ("Player", "__player__", str),
                                ("Pass Yds", "_pass_yards", _fmt_int),
                                ("Pass TD", "_pass_tds", _fmt_int),
                                ("INT", "_ints", _fmt_int),
                                ("Comp %", "_comp_pct", _fmt_pct),
                                ("FD Pts", "_fd", _fmt_one),
                            ],
                            "_pass_yards",
                        )

                    with rush_tab:
                        _render_weekly_leader_table(
                            rushing,
                            [
                                ("Player", "__player__", str),
                                ("Att", "_carries", _fmt_int),
                                ("Rush Yds", "_rush_yards", _fmt_int),
                                ("Rush TD", "_rush_tds", _fmt_int),
                                ("YPC", "_ypc", _fmt_one),
                                ("FD Pts", "_fd", _fmt_one),
                            ],
                            "_rush_yards",
                        )

                    with receive_tab:
                        _render_weekly_leader_table(
                            receiving,
                            [
                                ("Player", "__player__", str),
                                ("Targets", "_targets", _fmt_int),
                                ("Rec", "_receptions", _fmt_int),
                                ("Rec Yds", "_rec_yards", _fmt_int),
                                ("Rec TD", "_rec_tds", _fmt_int),
                                ("FD Pts", "_fd", _fmt_one),
                            ],
                            "_rec_yards",
                        )
