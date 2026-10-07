"""WFS Forecast Center page.

Behavior-preserving presentation extraction from app.py.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from forecast_publication_selector import select_forecast_path


def render_forecast_center(
    *,
    home_navigate,
    render_public_disclaimer,
    expected_pass_rate_text,
    kickoff_lookup,
    forecast_logo_uri,
):
    if st.button(
        "← Back to Home",
        key="wfs_back_home_forecast",
        width="stretch",
    ):
        home_navigate("🏠 Home")

    st.title("🔮 WFS Forecast Center")
    st.caption(
        "Pregame NFL score projections, matchup outlooks and WFS game picks."
    )

    forecast_path = select_forecast_path()

    if not forecast_path.is_file():
        st.warning("Pregame forecasts are not available yet for the current NFL slate.")
    else:
        forecast_df = pd.read_csv(forecast_path)

        required = {
            "game_id",
            "season",
            "week",
            "game_date",
            "away_team",
            "home_team",
            "pred_away_points",
            "pred_home_points",
            "pred_winner",
            "pred_home_margin",
            "pred_total_points",
            "forecast_status",
        }

        missing = sorted(required - set(forecast_df.columns))

        if missing:
            st.error(
                "NFL forecasts are temporarily unavailable."
            )
        else:
            forecast_df = forecast_df[
                forecast_df["season"].eq(2026)
            ].copy()

            weeks = sorted(
                int(x)
                for x in forecast_df["week"].dropna().unique()
            )

            if not weeks:
                st.info("No 2026 forecasts are currently available.")
            else:
                try:
                    from wfs_schedule_context import resolve_schedule_week_context

                    forecast_context = resolve_schedule_week_context(
                        season=2026,
                    )
                    forecast_default_week = int(
                        forecast_context.planning_week
                    )
                except Exception:
                    forecast_default_week = None

                if forecast_default_week not in weeks:
                    st.warning(
                        "Current-week NFL forecasts are temporarily unavailable."
                    )
                    selected_week = None
                else:
                    if "wfs_forecast_week" not in st.session_state:
                        st.session_state["wfs_forecast_week"] = (
                            forecast_default_week
                        )

                    if st.session_state["wfs_forecast_week"] not in weeks:
                        st.session_state["wfs_forecast_week"] = (
                            forecast_default_week
                        )

                    selected_week = st.selectbox(
                        "NFL Week",
                        weeks,
                        index=weeks.index(
                            st.session_state["wfs_forecast_week"]
                        ),
                        key="wfs_forecast_week",
                    )

                if selected_week is None:
                    week_df = forecast_df.iloc[0:0].copy()
                else:
                    week_df = forecast_df[
                        forecast_df["week"].eq(selected_week)
                    ].copy()

                week_df = week_df.sort_values(
                    ["game_date", "game_id"]
                )

                kickoff_lookup = (
                    kickoff_lookup(2026, selected_week)
                    if selected_week is not None
                    else {}
                )

                ready_count = int(
                    week_df["forecast_status"]
                    .isin({
                        "READY_CORE_ONLY",
                        "READY_INJURY_ADJUSTED",
                    })
                    .sum()
                )

                status_html = (
                    '<div class="qt-card" '
                    'style="padding:.65rem .8rem;'
                    'margin:.3rem 0 .8rem 0;'
                    'text-align:center;">'
                    '<div style="font-size:.76rem;'
                    'color:#94a3b8;font-weight:800;">'
                    f'2026 • WEEK {selected_week} • '
                    f'{len(week_df)} GAMES • {ready_count} READY'
                    '</div></div>'
                )

                st.markdown(
                    status_html,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"### Week {selected_week} Forecasts"
                )

                # WFS_PREDICTION_RECORD_UI_V1
                # Read-only public accuracy summary.
                try:
                    import json as _wfs_accuracy_json

                    _wfs_accuracy_path = (
                        Path("/home/mwynn/nfl_data_engine")
                        / "processed"
                        / "wfs_prediction_accuracy_v1.json"
                    )

                    if _wfs_accuracy_path.is_file():
                        _wfs_accuracy = _wfs_accuracy_json.loads(
                            _wfs_accuracy_path.read_text(
                                encoding="utf-8"
                            )
                        )

                        if (
                            _wfs_accuracy.get("version")
                            == "WFS_PREDICTION_ACCURACY_V1"
                        ):
                            _season = _wfs_accuracy.get("season") or {}

                            _week = next(
                                (
                                    row
                                    for row in (
                                        _wfs_accuracy.get("weeks") or []
                                    )
                                    if int(row.get("week", -1))
                                    == int(selected_week)
                                ),
                                None,
                            )

                            _season_wins = int(
                                _season.get("wins", 0)
                            )
                            _season_losses = int(
                                _season.get("losses", 0)
                            )
                            _season_pct = float(
                                _season.get("accuracy_pct", 0.0)
                            )

                            if _week is not None:
                                _week_wins = int(
                                    _week.get("wins", 0)
                                )
                                _week_losses = int(
                                    _week.get("losses", 0)
                                )
                                _week_pct = float(
                                    _week.get("accuracy_pct", 0.0)
                                )

                                _week_text = (
                                    f"Week {selected_week}: "
                                    f"<b>{_week_wins}-{_week_losses}</b> "
                                    f"• {_week_pct:.1f}%"
                                )
                            else:
                                _week_text = (
                                    f"Week {selected_week}: "
                                    "<b>Not graded yet</b>"
                                )

                            _accuracy_html = (
                                '<div class="qt-card" '
                                'style="padding:.72rem .85rem;'
                                'margin:.25rem 0 .85rem 0;'
                                'text-align:center;">'
                                '<div style="font-size:.66rem;'
                                'font-weight:850;'
                                'letter-spacing:.08em;'
                                'color:#94a3b8;">'
                                'WFS PREDICTION RECORD'
                                '</div>'
                                '<div style="margin-top:.32rem;'
                                'font-size:.92rem;'
                                'font-weight:750;'
                                'color:#e2e8f0;">'
                                f'{_week_text}'
                                '</div>'
                                '<div style="margin-top:.16rem;'
                                'font-size:.78rem;'
                                'color:#94a3b8;">'
                                f'Season: '
                                f'<b>{_season_wins}-{_season_losses}</b> '
                                f'• {_season_pct:.1f}%'
                                '</div>'
                                '</div>'
                            )

                            st.markdown(
                                _accuracy_html,
                                unsafe_allow_html=True,
                            )

                except Exception:
                    pass

                for _, game in week_df.iterrows():
                    away = str(game["away_team"])
                    home = str(game["home_team"])
                    status = str(game["forecast_status"])
                    game_date = kickoff_lookup.get(
                        str(game["game_id"]),
                        str(game["game_date"]),
                    )

                    if status in {
                          "READY_CORE_ONLY",
                          "READY_INJURY_ADJUSTED",
                      }:
                        away_score_raw = float(game["pred_away_points"])
                        home_score_raw = float(game["pred_home_points"])

                        # Read-only Expected Pass Rate lookup.
                        # Preserve exact game + team matching.
                        forecast_game_id = ""
                        for game_id_field in (
                            "game_id",
                            "gid",
                            "event_id",
                        ):
                            if game_id_field in game.index:
                                candidate_game_id = game.get(
                                    game_id_field
                                )
                                if (
                                    candidate_game_id is not None
                                    and pd.notna(candidate_game_id)
                                ):
                                    forecast_game_id = str(
                                        candidate_game_id
                                    ).strip()
                                    if forecast_game_id:
                                        break

                        away_expected_pass_rate = (
                            expected_pass_rate_text(
                                forecast_game_id,
                                away,
                            )
                        )
                        home_expected_pass_rate = (
                            expected_pass_rate_text(
                                forecast_game_id,
                                home,
                            )
                        )

                        # Public single-game prediction presentation:
                        # deterministic nonnegative half-up whole numbers.
                        away_score = int(max(0.0, away_score_raw) + 0.5)
                        home_score = int(max(0.0, home_score_raw) + 0.5)

                        # Public total and margin are derived from the
                        # displayed whole-number scores so the card is
                        # internally consistent.
                        total = away_score + home_score
                        margin_value = abs(home_score - away_score)

                        # Presentation only: a rounded tie has no winning team.
                        displayed_winner = (
                            home if home_score > away_score else away
                        )
                        # A checkmark reads as "confirmed" and this is a
                        # pregame pick, not a result, so use a pick marker
                        # instead of a checkmark.
                        lean_text = (
                            "Even" if margin_value == 0
                            else f"\U0001F3C8 {displayed_winner}"
                        )
                        margin_text = (
                            "Even" if margin_value == 0 else
                            f"{displayed_winner} by {margin_value}"
                        )

                        card_html = (
                            '<div class="qt-card" '
                            'style="padding:.8rem;'
                            'margin:.55rem 0;">'

                            '<div style="text-align:center;'
                            'color:#8299b3;'
                            'font-size:.68rem;'
                            'font-weight:750;'
                            'margin-bottom:.4rem;">'
                            f'{game_date}'
                            '</div>'

                            '<div style="display:flex;'
                            'align-items:center;'
                            'justify-content:space-between;'
                            'gap:.25rem;">'

                            '<div style="width:29%;'
                            'text-align:center;">'
                            f'<img src="{forecast_logo_uri(away)}" '
                            'style="width:48px;height:48px;'
                            'object-fit:contain;'
                            'display:block;margin:0 auto .25rem auto;" '
                            f'alt="{away}">'
                            '<div style="font-size:1.15rem;'
                            'font-weight:900;'
                            'color:#e2e8f0;">'
                            f'{away}'
                            '</div>'
                            '<div style="font-size:1.8rem;'
                            'font-weight:900;'
                            'color:#8bdcf5;'
                            'line-height:1.1;">'
                            f'{away_score}'
                            '</div>'
                            '</div>'

                            '<div style="width:42%;'
                            'text-align:center;">'
                            '<div style="font-size:.62rem;'
                            'color:#8299b3;'
                            'font-weight:800;'
                            'letter-spacing:.08em;">'
                            'MODEL LEAN'
                            '</div>'
                            '<div style="font-size:1.2rem;'
                            'font-weight:900;'
                            'color:#ffffff;">'
                            f'{lean_text}'
                            '</div>'
                            '<div style="font-size:.67rem;'
                            'color:#94a3b8;">'
                            f'{away} @ {home}'
                            '</div>'
                            '</div>'

                            '<div style="width:29%;'
                            'text-align:center;">'
                            f'<img src="{forecast_logo_uri(home)}" '
                            'style="width:48px;height:48px;'
                            'object-fit:contain;'
                            'display:block;margin:0 auto .25rem auto;" '
                            f'alt="{home}">'
                            '<div style="font-size:1.15rem;'
                            'font-weight:900;'
                            'color:#e2e8f0;">'
                            f'{home}'
                            '</div>'
                            '<div style="font-size:1.8rem;'
                            'font-weight:900;'
                            'color:#8bdcf5;'
                            'line-height:1.1;">'
                            f'{home_score}'
                            '</div>'
                            '</div>'

                            '</div>'

                            '<div style="border-top:1px solid #263548;'
                            'margin-top:.6rem;'
                            'padding-top:.5rem;'
                            'text-align:center;">'
                            '<div style="font-size:.64rem;'
                            'color:#8299b3;'
                            'font-weight:850;'
                            'letter-spacing:.06em;">'
                            'EXPECTED PASS RATE'
                            '</div>'
                            '<div style="font-size:.78rem;'
                            'color:#dbeafe;'
                            'font-weight:850;'
                            'margin-top:.16rem;">'
                            f'{away} {away_expected_pass_rate}'
                            '&nbsp;&nbsp;•&nbsp;&nbsp;'
                            f'{home} {home_expected_pass_rate}'
                            '</div>'
                            '<div style="font-size:.74rem;'
                            'color:#cbd5e1;'
                            'margin-top:.42rem;">'
                            f'<b>Total</b> {total}'
                            '&nbsp;&nbsp;•&nbsp;&nbsp;'
                            f'{margin_text}'
                            '</div>'
                            '</div>'

                            '</div>'
                        )

                        st.markdown(
                            card_html,
                            unsafe_allow_html=True,
                        )

                    else:
                        pending_html = (
                            '<div class="qt-card" '
                            'style="padding:.8rem;'
                            'margin:.55rem 0;">'
                            '<div style="font-weight:850;'
                            'color:#e2e8f0;">'
                            f'{away} @ {home}'
                            '</div>'
                            '<div style="color:#8299b3;'
                            'font-size:.74rem;">'
                            f'{game_date}'
                            '</div>'
                            '<div style="color:#f5bd45;'
                            'margin-top:.35rem;'
                            'font-size:.78rem;">'
                            'Forecast pending market inputs'
                            '</div>'
                            '</div>'
                        )

                        st.markdown(
                            pending_html,
                            unsafe_allow_html=True,
                        )

                st.caption(
                    "WFS pregame projections based on currently available "
                    "team, matchup and player information."
                )

    render_public_disclaimer()

    st.markdown(
        '<div class="qt-footer">'
        'Wynners Fantasy Spot • Forecast Center'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()


