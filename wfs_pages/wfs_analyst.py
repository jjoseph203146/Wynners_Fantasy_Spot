"""WFS Analyst page presentation and controller."""


def render_wfs_analyst(
    *,
    st,
    pd,
    sqlite3,
    DATABASE_PATH,
    _wfs_home_navigate,
    _wfs_render_matchup_outlook,
    render_wfs_public_disclaimer,
):
    if st.button(
        "← Back to Home",
        key="wfs_back_home_analyst",
        width="stretch",
    ):
        _wfs_home_navigate("🏠 Home")

    from wfs_ai_analyst import (
        FORECAST_CSV as WFS_ANALYST_FORECAST_CSV,
        build_analyst_response,
    )

    st.markdown(
        """
        <style>
        @media (max-width: 768px) {
            .wfs-analyst-mobile-spacer {
                height: 2.75rem;
            }
        }
        @media (min-width: 769px) {
            .wfs-analyst-mobile-spacer {
                height: 0;
            }
        }
        </style>
        <div class="wfs-analyst-mobile-spacer"></div>
        """,
        unsafe_allow_html=True,
    )

    st.title("🤖 WFS Analyst")
    st.caption(
        "Game outlooks, fantasy takeaways and postgame "
        "performance analysis powered by WFS football data."
    )

    try:
        with sqlite3.connect(
            f"file:{DATABASE_PATH}?mode=ro",
            uri=True,
        ) as analyst_conn:
            analyst_games = pd.read_sql_query(
                """
                SELECT
                    game_id,
                    season,
                    week,
                    game_date,
                    away_team,
                    home_team,
                    completed
                FROM games
                WHERE season = (
                    SELECT MAX(season)
                    FROM games
                )
                ORDER BY game_date, game_id
                """,
                analyst_conn,
            )
    except Exception:
        analyst_games = pd.DataFrame()

    if analyst_games.empty:
        st.info(
            "Game analysis will appear when NFL matchup "
            "information is available."
        )
    else:
        analyst_games = analyst_games.copy()

        analyst_games["game_date_sort"] = pd.to_datetime(
            analyst_games["game_date"],
            errors="coerce",
        )

        analyst_games["completed"] = pd.to_numeric(
            analyst_games["completed"],
            errors="coerce",
        ).fillna(0).astype(int)

        # WFS_ANALYST_SCHEDULE_PRIORITY_V1
        # Preserve the full Analyst game inventory, but prioritize
        # the schedule-authoritative planning week. Completed games
        # remain available for postgame review and later future weeks
        # remain available for advance analysis.
        from wfs_schedule_context import (
            resolve_schedule_week_context,
        )

        analyst_schedule_context = (
            resolve_schedule_week_context()
        )
        analyst_planning_week = int(
            analyst_schedule_context.planning_week
        )

        analyst_games["analyst_priority"] = 2

        analyst_games.loc[
            analyst_games["completed"].eq(1),
            "analyst_priority",
        ] = 1

        analyst_games.loc[
            pd.to_numeric(
                analyst_games["week"],
                errors="coerce",
            ).eq(analyst_planning_week),
            "analyst_priority",
        ] = 0

        # Planning/future games sort chronologically.
        # Historical finals sort most-recent first by using a
        # separate numeric date key.
        analyst_games["analyst_date_key"] = (
            analyst_games["game_date_sort"]
            .map(
                lambda value: (
                    value.value
                    if pd.notna(value)
                    else 0
                )
            )
        )

        analyst_games.loc[
            analyst_games["analyst_priority"].eq(1),
            "analyst_date_key",
        ] *= -1

        analyst_games = analyst_games.sort_values(
            [
                "analyst_priority",
                "analyst_date_key",
                "game_id",
            ],
            ascending=[True, True, True],
        )

        analyst_games["status_label"] = analyst_games[
            "completed"
        ].map(
            lambda value: " • Final"
            if int(value) == 1
            else ""
        )

        analyst_games["analyst_label"] = (
            analyst_games["away_team"].astype(str)
            + " @ "
            + analyst_games["home_team"].astype(str)
            + " • Week "
            + analyst_games["week"].astype(str)
            + analyst_games["status_label"]
        )

        analyst_options = dict(
            zip(
                analyst_games["analyst_label"],
                analyst_games["game_id"],
            )
        )

        selected_analyst_label = st.selectbox(
            "Select matchup",
            list(analyst_options.keys()),
            key="wfs_analyst_game",
        )

        selected_analyst_game = analyst_options[
            selected_analyst_label
        ]

        try:
            analyst_response = build_analyst_response(
                selected_analyst_game
            )
        except Exception:
            analyst_response = None

        if not analyst_response:
            st.info(
                "Analysis is not available for this matchup yet."
            )
        else:
            analyst_mode = analyst_response.get(
                "mode",
                "",
            )

            if analyst_mode == "PREGAME":
                mode_label = "Pregame Outlook"
            elif analyst_mode == "LIVE":
                mode_label = "Live Game Analysis"
            elif analyst_mode == "POSTGAME":
                mode_label = "Postgame Review"
            else:
                mode_label = "Game Analysis"

            st.markdown(f"### {mode_label}")

            headline = analyst_response.get("headline")

            if headline:
                st.markdown(f"## {headline}")

            game_summary = analyst_response.get(
                "game_summary"
            )

            if game_summary:
                st.write(game_summary)

            stat_outlook = analyst_response.get("stat_outlook")

            if isinstance(stat_outlook, dict):
                outlook_available = bool(
                    stat_outlook.get("available")
                    or stat_outlook.get("teams")
                )

                st.markdown("### Stat Outlook")

                # QA F2 / WFS_STAT_OUTLOOK_PUBLIC_SCOPE_NOTE_V1
                #
                # Stat Outlook publishes selected, rounded player-level
                # projections. It is not an exhaustive decomposition of
                # the separate game-level projected team score.
                st.caption(
                    "Selected, rounded player projections. "
                    "These stat lines are not a complete scoring "
                    "breakdown of the projected team score."
                )

                if analyst_mode in {"LIVE", "POSTGAME"}:
                    st.caption("Original kickoff outlook.")

                if outlook_available:
                    outlook_teams = stat_outlook.get("teams") or []

                    for team_outlook in outlook_teams:
                        if not isinstance(team_outlook, dict):
                            continue

                        team_label = str(
                            team_outlook.get("team")
                            or team_outlook.get("team_name")
                            or ""
                        ).strip()

                        if team_label:
                            st.markdown(f"#### {team_label}")

                        offense_lines = (
                            team_outlook.get("offense")
                            or team_outlook.get("offense_lines")
                            or []
                        )

                        if isinstance(offense_lines, str):
                            offense_lines = [offense_lines]

                        for line in offense_lines:
                            line = str(line or "").strip()
                            if line:
                                st.markdown(f"- {line}")

                        kicker_line = (
                            team_outlook.get("kicker")
                            or team_outlook.get("kicker_line")
                        )
                        kicker_line = str(kicker_line or "").strip()

                        if kicker_line:
                            st.markdown(f"- {kicker_line}")

                        dst_line = (
                            team_outlook.get("dst")
                            or team_outlook.get("dst_line")
                            or team_outlook.get("defense")
                        )
                        dst_line = str(dst_line or "").strip()

                        if dst_line:
                            st.markdown(f"- {dst_line}")

                else:
                    unavailable_note = str(
                        stat_outlook.get("unavailable_note")
                        or stat_outlook.get("note")
                        or "Stat outlook is not available for this game."
                    ).strip()

                    if unavailable_note:
                        st.caption(unavailable_note)

                _wfs_render_matchup_outlook(
                    selected_analyst_game,
                    stat_outlook.get("selected_players") or [],
                )

            impact_injuries = (
                analyst_response.get("impact_injuries")
                or []
            )

            if (
                analyst_mode == "PREGAME"
                and impact_injuries
            ):
                public_pregame_injuries = [
                    injury
                    for injury in impact_injuries
                    if str(
                        injury.get(
                            "impact_classification"
                        )
                        or ""
                    )
                    != "INJURY_WATCH"
                ]

                if public_pregame_injuries:
                    st.markdown(
                        "### Pregame Injury Impact"
                    )

                    for injury in (
                        public_pregame_injuries
                    ):
                        player = str(
                            injury.get("player")
                            or "Player"
                        )

                        team = str(
                            injury.get("team")
                            or ""
                        ).strip()

                        position = str(
                            injury.get("position")
                            or ""
                        ).strip()

                        status = str(
                            injury.get("report_status")
                            or injury.get("status")
                            or ""
                        ).strip().upper()

                        primary_injury = str(
                            injury.get(
                                "primary_injury"
                            )
                            or ""
                        ).strip()

                        impact = str(
                            injury.get(
                                "impact_classification"
                            )
                            or ""
                        )

                        player_label = player

                        if position:
                            player_label += (
                                f" ({position})"
                            )

                        if team:
                            player_label += (
                                f" — {team}"
                            )

                        status_label = {
                            "OUT": "Out",
                            "DOUBTFUL": "Doubtful",
                            "QUESTIONABLE": (
                                "Questionable"
                            ),
                        }.get(
                            status,
                            status.title()
                            if status
                            else "Injury Report",
                        )

                        if primary_injury:
                            injury_text = (
                                f"{primary_injury} injury. "
                            )
                        else:
                            injury_text = ""

                        if (
                            impact
                            == "MAJOR_PREGAME_IMPACT"
                        ):
                            public_label = (
                                "Major Pregame Impact"
                            )
                            public_text = (
                                f"{injury_text}"
                                f"Listed {status_label.lower()}. "
                                "Because this involves the "
                                "quarterback, availability "
                                "could materially affect the "
                                "offensive outlook."
                            )

                        elif (
                            impact
                            == "MATERIAL_PREGAME_IMPACT"
                        ):
                            public_label = (
                                "Pregame Injury Impact"
                            )

                            if status == "OUT":
                                availability_text = (
                                    "The player is listed out."
                                )
                            elif status == "DOUBTFUL":
                                availability_text = (
                                    "The player is listed "
                                    "doubtful."
                                )
                            else:
                                availability_text = (
                                    f"The player is listed "
                                    f"{status_label.lower()}."
                                )

                            public_text = (
                                f"{injury_text}"
                                f"{availability_text} "
                                "Recent offensive involvement "
                                "makes the availability change "
                                "potentially meaningful."
                            )

                        elif (
                            impact
                            == "AVAILABILITY_UNCERTAINTY"
                        ):
                            public_label = (
                                "Pregame Injury Watch"
                            )
                            public_text = (
                                f"{injury_text}"
                                "Listed questionable. "
                                "Availability remains "
                                "uncertain, and the player's "
                                "recent role makes the status "
                                "important to monitor."
                            )

                        elif impact == "TEAM_CONTEXT":
                            public_label = (
                                "Offensive Line Impact"
                            )

                            if status == "OUT":
                                availability_text = (
                                    "The player is listed out."
                                )
                            elif status == "DOUBTFUL":
                                availability_text = (
                                    "The player is listed "
                                    "doubtful."
                                )
                            else:
                                availability_text = (
                                    f"The player is listed "
                                    f"{status_label.lower()}."
                                )

                            public_text = (
                                f"{injury_text}"
                                f"{availability_text} "
                                "The availability change "
                                "could affect the offense at "
                                "the team level."
                            )

                        elif (
                            impact
                            == "LINE_AVAILABILITY_UNCERTAINTY"
                        ):
                            public_label = (
                                "Offensive Line Watch"
                            )
                            public_text = (
                                f"{injury_text}"
                                "Listed questionable. "
                                "Availability remains "
                                "uncertain and could affect "
                                "the offense at the team "
                                "level."
                            )

                        else:
                            continue

                        st.markdown(
                            f"**{player_label} — "
                            f"{status_label}**"
                        )

            if (
                analyst_mode in {"LIVE", "POSTGAME"}
                and impact_injuries
            ):
                st.markdown("### Injury Impact")

                for injury in impact_injuries:
                    player = str(
                        injury.get("player")
                        or "Player"
                    )

                    team = str(
                        injury.get("team")
                        or ""
                    ).strip()

                    position = str(
                        injury.get("position")
                        or ""
                    ).strip()

                    status = str(
                        injury.get("status")
                        or ""
                    )

                    impact = str(
                        injury.get("impact_classification")
                        or ""
                    )

                    player_label = player

                    if position:
                        player_label += f" ({position})"

                    if team:
                        player_label += f" — {team}"

                    if status == "RETURNED":
                        public_label = "Injury Update"
                        public_text = (
                            "Was reported injured earlier "
                            "and later returned to the game."
                        )

                    elif impact == "MAJOR_POTENTIAL_IMPACT":
                        public_label = "Impact Injury"
                        public_text = (
                            "Was reported injured and no "
                            "verified return update has been "
                            "observed. Because this involves "
                            "the quarterback, it could "
                            "materially affect the team's "
                            "offensive outlook."
                        )

                    elif impact == "POTENTIAL_IMPACT":
                        public_label = "Impact Injury"
                        public_text = (
                            "Was reported injured and no "
                            "verified return update has been "
                            "observed. His established role "
                            "makes this potentially meaningful "
                            "to the offense."
                        )

                    elif impact == "TEAM_CONTEXT":
                        public_label = (
                            "Offensive Line Context"
                        )
                        public_text = (
                            "Was reported injured and no "
                            "verified return update has been "
                            "observed. This could affect the "
                            "offense at a team level."
                        )

                    else:
                        public_label = "Injury Watch"
                        public_text = (
                            "Was reported injured and no "
                            "verified return update has been "
                            "observed."
                        )

                    st.markdown(
                        f"**{public_label} — "
                        f"{player_label}**"
                    )
                    st.write(public_text)

            observations = (
                analyst_response.get("observations")
                or []
            )

            injury_prefixes = (
                "Impact injury:",
                "Injury update:",
                "Offensive-line injury:",
                "Injury watch:",
                "Pregame injury impact:",
                "Pregame injury watch:",
                "Pregame offensive-line context:",
                "Pregame offensive-line watch:",
            )

            observations = [
                observation
                for observation in observations
                if not str(observation).startswith(
                    injury_prefixes
                )
            ]

            if observations:
                st.markdown("### What stands out")

                for observation in observations:
                    st.markdown(f"- {observation}")

            fantasy_leaders = (
                analyst_response.get("fantasy_leaders")
                or []
            )

            if fantasy_leaders:
                st.markdown("### Fantasy Leaders")

                for leader in fantasy_leaders:
                    player = leader.get(
                        "player",
                        "Player",
                    )
                    team = leader.get("team", "")
                    position = leader.get(
                        "position",
                        "",
                    )
                    points = leader.get(
                        "fanduel_points"
                    )

                    if points is None:
                        points_text = "—"
                    else:
                        try:
                            points_text = (
                                f"{float(points):.1f}"
                            )
                        except (
                            TypeError,
                            ValueError,
                        ):
                            points_text = str(points)

                    st.markdown(
                        f"**{player}** "
                        f"({team} {position}) — "
                        f"{points_text} FanDuel pts"
                    )

            if analyst_mode == "PREGAME":
                evidence_status = (
                    analyst_response.get(
                        "evidence_status"
                    )
                    or {}
                )

                if not evidence_status.get(
                    "win_probability_available",
                    False,
                ):
                    st.caption(
                        "Win probability is not shown until "
                        "WFS has a calibrated probability model."
                    )

    render_wfs_public_disclaimer()

    st.markdown(
        '<div class="qt-footer">'
        'Wynners Fantasy Spot • WFS Analyst'
        '</div>',
        unsafe_allow_html=True,
    )

    st.stop()
