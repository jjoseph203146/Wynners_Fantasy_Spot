"""WFS Command Center / Home presentation page.

Behavior-preserving extraction from app.py.

Account, connection, navigation, and shared NFL helpers remain owned by
app.py and are injected into render_wfs_home(). This module does not alter
solver, projection, eligibility, Stage24, NFL Live, or data authority.
"""

from __future__ import annotations

import streamlit as st


def render_wfs_home(
    *,
    nfl_team_names,
    dc_team_code,
    wfs_accounts_init,
    wfs_current_user,
    wfs_espn_connections,
    wfs_home_navigate,
    wfs_sleeper_connections,
    wfs_upsert_user,
    wfs_user_preferences,
):
    """WFS public command-center landing page. Presentation only."""

    st.markdown(
        """
        <style>
        .wfs-home-hero {
            background: linear-gradient(115deg,#10233d 0%,#173b68 58%,#0b73f6 100%);
            border-radius: 22px;
            padding: 1.35rem 1.35rem 1.25rem;
            margin: .15rem 0 1rem;
            color: white;
            box-shadow: 0 10px 28px rgba(15,55,105,.16);
        }

        .wfs-home-eyebrow {
            font-size: .68rem;
            font-weight: 900;
            letter-spacing: .11em;
            color: #9fd0ff;
        }

        .wfs-home-title {
            font-size: 1.75rem;
            line-height: 1.08;
            font-weight: 950;
            margin-top: .25rem;
        }

        .wfs-home-subtitle {
            font-size: .9rem;
            line-height: 1.5;
            color: #d8eaff;
            margin-top: .45rem;
            max-width: 760px;
        }

        .wfs-myweek {
            background: #f7faff;
            border: 1px solid #d8e6f5;
            border-radius: 18px;
            padding: 1rem 1.05rem;
            margin: .35rem 0 1rem;
            box-shadow: 0 4px 14px rgba(15,35,60,.045);
        }

        .wfs-myweek-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: .6rem;
            margin-bottom: .75rem;
        }

        .wfs-myweek-title {
            font-size: .72rem;
            font-weight: 950;
            letter-spacing: .1em;
            color: #173b68;
        }

        .wfs-myweek-season {
            background: #e8f2ff;
            color: #1769c2;
            border-radius: 999px;
            padding: .25rem .55rem;
            font-size: .65rem;
            font-weight: 900;
        }

        .wfs-myweek-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0,1fr));
            gap: .65rem;
        }

        .wfs-myweek-item {
            background: #ffffff;
            border: 1px solid #e0e9f3;
            border-radius: 14px;
            padding: .78rem .82rem;
            min-width: 0;
        }

        .wfs-myweek-label {
            color: #71849b;
            font-size: .64rem;
            font-weight: 900;
            letter-spacing: .075em;
            text-transform: uppercase;
        }

        .wfs-myweek-value {
            color: #172033;
            font-size: .98rem;
            font-weight: 950;
            margin-top: .2rem;
            overflow-wrap: anywhere;
        }

        .wfs-myweek-detail {
            color: #71849b;
            font-size: .7rem;
            line-height: 1.35;
            margin-top: .16rem;
        }

        .wfs-myweek-empty {
            color: #71849b;
            font-size: .78rem;
            line-height: 1.45;
            margin-top: .7rem;
        }

        .wfs-home-card {
            background: #ffffff;
            border: 1px solid #dbe5f0;
            border-radius: 18px;
            padding: 1rem 1.05rem;
            min-height: 190px;
            box-shadow: 0 5px 16px rgba(15,35,60,.06);
            margin-bottom: .7rem;
        }

        .wfs-home-card-kicker {
            font-size: .67rem;
            font-weight: 900;
            letter-spacing: .09em;
            color: #64748b;
            text-transform: uppercase;
        }

        .wfs-home-card-title {
            font-size: 1.08rem;
            font-weight: 950;
            color: #172033;
            margin-top: .3rem;
        }

        .wfs-home-card-copy {
            font-size: .82rem;
            line-height: 1.48;
            color: #52657d;
            margin-top: .42rem;
        }

        .wfs-home-card-status {
            display: inline-block;
            margin-top: .7rem;
            padding: .28rem .55rem;
            border-radius: 999px;
            background: #eef6ff;
            color: #1769c2;
            font-size: .67rem;
            font-weight: 900;
            letter-spacing: .04em;
        }

        .wfs-home-flow {
            background: #f7faff;
            border: 1px solid #dbe7f4;
            border-radius: 18px;
            padding: 1rem 1.05rem;
            margin: .45rem 0 .9rem;
        }

        .wfs-home-flowline {
            color: #173b68;
            font-size: .92rem;
            font-weight: 950;
            letter-spacing: .035em;
            line-height: 1.6;
        }

        .wfs-home-flowcopy {
            color: #60738b;
            font-size: .79rem;
            line-height: 1.5;
            margin-top: .3rem;
        }

        .wfs-home-capability {
            background: #ffffff;
            border-left: 4px solid #0b73f6;
            border-radius: 10px;
            padding: .72rem .8rem;
            margin: .45rem 0;
            color: #42556d;
            font-size: .8rem;
            line-height: 1.45;
        }

        .wfs-home-capability b {
            color: #172033;
        }

        /* ========================================================
           WFS_HOME_THEME_FIX_V1
           Theme-aware Home surfaces only.
           Hero styling remains intentionally branded.
           ======================================================== */

        .wfs-myweek,
        .wfs-home-flow {
            background: var(--st-secondary-background-color) !important;
            border-color:
                color-mix(
                    in srgb,
                    var(--st-text-color) 18%,
                    transparent
                ) !important;
        }

        .wfs-myweek-item,
        .wfs-home-card,
        .wfs-home-capability {
            background: var(--st-background-color) !important;
            border-color:
                color-mix(
                    in srgb,
                    var(--st-text-color) 18%,
                    transparent
                ) !important;
        }

        .wfs-myweek-title,
        .wfs-home-card-title,
        .wfs-home-flowline,
        .wfs-home-capability b,
        .wfs-myweek-value {
            color: var(--st-text-color) !important;
        }

        .wfs-myweek-label,
        .wfs-myweek-detail,
        .wfs-myweek-empty,
        .wfs-home-card-kicker,
        .wfs-home-card-copy,
        .wfs-home-flowcopy,
        .wfs-home-capability {
            color:
                color-mix(
                    in srgb,
                    var(--st-text-color) 72%,
                    transparent
                ) !important;
        }

        .wfs-myweek-season,
        .wfs-home-card-status {
            background:
                color-mix(
                    in srgb,
                    var(--st-primary-color) 14%,
                    var(--st-secondary-background-color)
                ) !important;
            color: var(--st-primary-color) !important;
        }

        @media (max-width: 700px) {
            .wfs-home-hero {
                padding: 1.1rem 1rem;
                border-radius: 18px;
            }

            .wfs-home-title {
                font-size: 1.48rem;
            }

            .wfs-home-subtitle {
                font-size: .84rem;
            }

            .wfs-home-card {
                min-height: 0;
            }

            .wfs-myweek-grid {
                grid-template-columns: 1fr;
            }

            .wfs-myweek {
                padding: .9rem;
            }
        }
        </style>

        <div class="wfs-home-hero">
          <div class="wfs-home-eyebrow">WYNNERS FANTASY SPOT</div>
          <div class="wfs-home-title">Your NFL Command Center</div>
          <div class="wfs-home-subtitle">
            Prepare, build, monitor and react throughout the NFL week.
            WFS combines NFL game forecasts, player projections, matchup and
            availability intelligence, FanDuel tournament lineup construction,
            late swap, live game tracking and ESPN & Sleeper fantasy tools in
            one focused workspace.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # MY WEEK
    #
    # Read-only summary from existing WFS account persistence.
    # No ESPN network request is made from Home.
    # =====================================================

    home_user = wfs_current_user()

    favorite_code = ""
    favorite_name = ""
    espn_team_name = ""
    espn_league_name = ""
    espn_season = 2026
    espn_connection_count = 0

    if home_user:
        try:
            wfs_accounts_init()
            wfs_upsert_user(home_user)

            prefs = wfs_user_preferences(home_user["sub"])
            favorite_code = dc_team_code(
                prefs.get("favorite_nfl_team", "")
            )

            if favorite_code:
                favorite_name = nfl_team_names.get(
                    favorite_code,
                    favorite_code,
                )
        except Exception:
            favorite_code = ""
            favorite_name = ""

        try:
            espn_season = int(
                st.session_state.get("wfs_espn_season", 2026)
            )

            # WFS_HOME_MULTI_PROVIDER_MY_WEEK_V1
            # My Week should reflect whichever provider (ESPN or
            # Sleeper) the account actually has connected, not ESPN
            # only. _wfs_*_connections() each have a deterministic
            # database ordering.
            home_connections = (
                wfs_espn_connections(home_user["sub"], espn_season)
                + wfs_sleeper_connections(home_user["sub"], espn_season)
            )

            espn_connection_count = len(home_connections)

            followed_connections = [
                c for c in home_connections
                if c.get("team_id") is not None
                and str(c.get("team_name") or "").strip()
            ]

            if followed_connections:
                home_espn = followed_connections[0]
                espn_team_name = str(
                    home_espn.get("team_name") or ""
                ).strip()
                espn_league_name = str(
                    home_espn.get("league_name") or ""
                ).strip()

        except Exception:
            espn_team_name = ""
            espn_league_name = ""
            espn_connection_count = 0

    favorite_value = (
        f"{favorite_code} — {favorite_name}"
        if favorite_code and favorite_name
        else "Not selected"
    )

    favorite_detail = (
        "Highlighted throughout NFL Data Center"
        if favorite_code
        else "Choose your team in NFL Data Center"
    )

    fantasy_value = (
        espn_team_name
        if espn_team_name
        else "Not selected"
    )

    if espn_team_name and espn_league_name:
        fantasy_detail = espn_league_name
    elif espn_connection_count:
        fantasy_detail = (
            "Choose a followed team in Fantasy Hub"
        )
    else:
        fantasy_detail = (
            "Add an ESPN or Sleeper league in Fantasy Hub"
        )

    my_week_html = (
        f'<div class="wfs-myweek">'
        f'<div class="wfs-myweek-head">'
        f'<div class="wfs-myweek-title">MY WEEK</div>'
        f'<div class="wfs-myweek-season">NFL {espn_season}</div>'
        f'</div>'
        f'<div class="wfs-myweek-grid">'
        f'<div class="wfs-myweek-item">'
        f'<div class="wfs-myweek-label">⭐ My NFL Team</div>'
        f'<div class="wfs-myweek-value">{favorite_value}</div>'
        f'<div class="wfs-myweek-detail">{favorite_detail}</div>'
        f'</div>'
        f'<div class="wfs-myweek-item">'
        f'<div class="wfs-myweek-label">🏆 Fantasy League</div>'
        f'<div class="wfs-myweek-value">{fantasy_value}</div>'
        f'<div class="wfs-myweek-detail">{fantasy_detail}</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )

    st.markdown(
        my_week_html,
        unsafe_allow_html=True,
    )

    st.markdown("### What do you want to do?")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="wfs-home-card">
              <div class="wfs-home-card-kicker">🏗 BUILD</div>
              <div class="wfs-home-card-title">Lineup Generator</div>
              <div class="wfs-home-card-copy">
                Build FanDuel NFL tournament lineups using WFS projections,
                matchup information and tournament strategy.
              </div>
              <div class="wfs-home-card-status">DFS • TOURNAMENT FOCUSED</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Open Lineup Generator →",
            key="wfs_home_open_build",
            width="stretch",
        ):
            wfs_home_navigate("🏗 Build Lineups")

    with c2:
        st.markdown(
            """
            <div class="wfs-home-card">
              <div class="wfs-home-card-kicker">⚡ REACT</div>
              <div class="wfs-home-card-title">Late Swap</div>
              <div class="wfs-home-card-copy">
                Protect players whose games have started and optimize the
                remaining FanDuel roster decisions as the slate develops.
              </div>
              <div class="wfs-home-card-status">LOCKED PLAYERS • LIVE SLATE DECISIONS</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Open Late Swap →",
            key="wfs_home_open_late_swap",
            width="stretch",
        ):
            wfs_home_navigate(
                "🏗 Build Lineups",
                open_late_swap=True,
            )

    c3, c4 = st.columns(2)

    with c3:
        st.markdown(
            """
            <div class="wfs-home-card">
              <div class="wfs-home-card-kicker">🏈 FOLLOW</div>
              <div class="wfs-home-card-title">NFL Data Center</div>
              <div class="wfs-home-card-copy">
                Explore schedules, game forecasts, player projections,
                results and box scores with favorite-team personalization
                and fantasy-player context.
              </div>
              <div class="wfs-home-card-status">SCHEDULE • SCORES • BOX SCORES</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Open NFL Data Center →",
            key="wfs_home_open_data_center",
            width="stretch",
        ):
            wfs_home_navigate("🏈 NFL Data Center")

    with c4:
        st.markdown(
            """
            <div class="wfs-home-card">
              <div class="wfs-home-card-kicker">🏆 COMPETE</div>
              <div class="wfs-home-card-title">Fantasy League Hub</div>
              <div class="wfs-home-card-copy">
                Bring your ESPN or Sleeper leagues into WFS to follow
                your team, weekly matchup, roster and league standings.
              </div>
              <div class="wfs-home-card-status">ESPN • SLEEPER • MATCHUPS</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Open Fantasy League Hub →",
            key="wfs_home_open_fantasy",
            width="stretch",
        ):
            wfs_home_navigate("🏆 Fantasy League Hub")

    st.markdown("### Your WFS Workflow")

    st.markdown(
        """
        <div class="wfs-home-flow">
          <div class="wfs-home-flowline">
            PREPARE &nbsp;→&nbsp; BUILD &nbsp;→&nbsp; MONITOR &nbsp;→&nbsp; REACT
          </div>
          <div class="wfs-home-flowcopy">
            Study the slate and NFL week. Build tournament lineups.
            Monitor games and your fantasy players. Then use Late Swap
            when the remaining slate gives you another decision to make.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # =====================================================
    # WFS_PUBLIC_FEEDBACK_V1
    # Public slate availability + observation-only feedback.
    # No projection, solver, identity, injury, or slate mutation.
    # =====================================================

    st.markdown("### FanDuel Slate Availability")

    st.info(
        "Slates appear when FanDuel releases them. Main, 1 PM Only, "
        "4 PM Only, primetime, and other supported slates will "
        "automatically appear when available. "
        "For SNF–MNF, late-swap options are shown as the Sunday slate "
        "develops; players are protected once their games have started."
    )

    try:
        from wfs_user_feedback import render_public_feedback

        feedback_user_id = None

        if home_user:
            feedback_user_id = str(
                home_user.get("sub") or ""
            ).strip() or None

        # WFS_FEEDBACK_SCHEDULE_CONTEXT_V1
        # Planning context comes from authoritative NFL schedule state,
        # never file timestamps, ESPN state, or a hardcoded week.
        from wfs_schedule_context import resolve_schedule_week_context

        feedback_schedule_context = resolve_schedule_week_context()

        render_public_feedback(
            user_id=feedback_user_id,
            user_name=(
                str(home_user.get("name") or "").strip()
                if home_user
                else None
            ),
            user_email=(
                str(home_user.get("email") or "").strip()
                if home_user
                else None
            ),
            season=feedback_schedule_context.season,
            week=feedback_schedule_context.planning_week,
        )

    except Exception:
        st.info(
            "Feedback is temporarily unavailable. "
            "Please check back shortly."
        )
