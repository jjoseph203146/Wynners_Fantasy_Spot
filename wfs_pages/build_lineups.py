"""Internal Build Lineups family: Classic, Late Swap, and Single-Game.

App-owned cached loaders and shared helpers are injected unchanged. The factory
is called on each app run, so its closures never share user or workspace state
through persistent module globals. Repository paths come from app.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class BuildLineups:
    render_sidebar: Callable
    render_page: Callable
    available_slates: Callable
    slate_pool: Callable
    late_swap_slate_pool: Callable


def create_build_lineups(
    *,
    APP_DIR,
    CLASSIC_LINEUP_FRESHNESS_PATH,
    CSV_DIR,
    DATABASE_PATH,
    LateSwapSettings,
    PORTFOLIO_CACHE_DIR,
    PORTFOLIO_CACHE_INCREMENT,
    PORTFOLIO_CACHE_LOCK,
    PORTFOLIO_CACHE_MAX_SIZE,
    PORTFOLIO_CACHE_SIZE,
    Path,
    ROSTER_SLOTS,
    SHOWDOWN_POOL,
    SHOWDOWN_SOLVER,
    SOLVER_TABLE,
    UI_SOLVER,
    WFS_USER,
    ZoneInfo,
    __name__,
    _dc_col,
    _dc_team_code,
    _dc_team_logo_html,
    _wfs_get_seen_portfolio_signatures,
    _wfs_home_navigate,
    _wfs_render_public_data_freshness,
    _wfs_set_seen_portfolio_signatures,
    contextmanager,
    csv,
    current_classic_publication,
    datetime,
    fcntl,
    hashlib,
    io,
    json,
    load_data_center,
    load_showdown_projection_pool,
    logging,
    normalize_fanduel_name,
    normalize_name,
    normalize_slate,
    normalize_team,
    optimize_late_swap,
    pd,
    re,
    render_admin_diagnostics,
    render_wfs_public_disclaimer,
    safe_slug,
    showdown_pool_freshness_token,
    sqlite3,
    st,
    subprocess,
    sys,
    time,
    traceback,
    ui_player_key,
) -> BuildLineups:
    """Bind app dependencies for this run without changing cached loaders."""
    def _wfs_classic_failure_category(attempts, has_controls: bool) -> str:
        """Classify explicit diagnostics only; ambiguous failures stay UNKNOWN."""
        categories = []
        for attempt in attempts:
            if attempt.get("accepted"):
                continue
            detail = (attempt.get("stdout") or "") + "\n" + (attempt.get("stderr") or "")
            if any(marker in detail for marker in (
                "Stage24AttachmentError:", "CLASSIC_PUBLICATION_",
                "RuntimeError: FAIL_CLOSED:", "RuntimeError: Final injury gate ",
                "RuntimeError: Missing SQLite table:", "RuntimeError: Missing offensive parquet:",
                "RuntimeError: Offensive parquet missing fields:",
                "RuntimeError: Offensive GPP slate not found:",
                "RuntimeError: Duplicate eligible player/team/salary rows detected:",
                "RuntimeError: Duplicate exact offensive name+team+position keys detected.",
                "RuntimeError: Duplicate UI player keys detected:",
                "RuntimeError: Eligible pool contains invalid ",
                "RuntimeError: Incomplete exact offensive GPP attachment:",
                "RuntimeError: Validated GPP field contains missing/nonfinite values.",
                "RuntimeError: RB gpp_score_20 coverage is incomplete.",
                "RuntimeError: WR gpp_score_15/25 coverage is incomplete.",
                "RuntimeError: TE gpp_score_15/20 coverage is incomplete.",
                "RuntimeError: GPP objective signal creation is incomplete.",
                "RuntimeError: Could not resolve opponent for eligible QB rows:",
                "RuntimeError: No eligible candidates for roster slot ",
                "RuntimeError: Insufficient ", "RuntimeError: Slate not found:",
            )):
                categories.append("AUTHORITY")
            elif attempt.get("validation_failure") == "AUDIT_FAILED" or "OUTPUT_VALIDATION_ERROR:" in detail:
                categories.append("UNKNOWN")
            elif has_controls and (
                any(marker in detail for marker in (
                    "ValueError: A player cannot be both locked and excluded:",
                    "RuntimeError: Locked player(s) not found in eligible slate pool:",
                    "RuntimeError: Excluded player(s) not found in eligible slate pool:",
                ))
                or re.search(r"RuntimeError: Resume lineup \d+ (?:is missing one or more current locks|contains a currently excluded player)\.", detail)
            ):
                categories.append("PLAYER_CONTROL")
            elif re.search(
                r"Stopped at lineup \d+: constraints/exposure caps made the remaining portfolio infeasible\.", detail
            ) or "RuntimeError: No feasible lineups generated." in detail:
                categories.append("INFEASIBLE")
            else:
                categories.append("UNKNOWN")
        # Never let a strategy attempt hide an authority or unrecognized failure.
        for category in ("AUTHORITY", "UNKNOWN", "PLAYER_CONTROL", "INFEASIBLE"):
            if category in categories:
                return category
        return "UNKNOWN"


    def _wfs_classic_failure_message(category: str) -> str:
        return {
            "AUTHORITY": (
                "We couldn't generate lineups because the latest lineup data is still "
                "being prepared. Please try again shortly."
            ),
            "PLAYER_CONTROL": (
                "We couldn't use your locked or excluded player selections. "
                "Try changing a lock or exclusion and generate again."
            ),
            "INFEASIBLE": (
                "We couldn't complete the requested FanDuel lineups for this slate. "
                "Try another slate or try again shortly."
            ),
        }.get(category, "We couldn't generate lineups right now. Please try again shortly.")


    def _wfs_report_classic_exception(exc, workspace_mode, category=None, has_controls=False):
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        category = category or _wfs_classic_failure_category([{"stderr": detail}], has_controls)
        st.session_state["last_classic_generation_exception"] = detail
        st.session_state["last_classic_failure_category"] = category
        logging.getLogger(__name__).error(
            "Classic generation failure (%s)", category,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        if workspace_mode == "Admin":
            st.error("Classic lineup generation failed.")
            st.code(detail)
        else:
            st.error(_wfs_classic_failure_message(category))


    @contextmanager
    def _wfs_classic_generation_errors(workspace_mode, has_controls):
        """Present unhandled Public failures without exposing backend details."""
        try:
            yield
        except Exception as exc:
            if workspace_mode == "Admin":
                raise
            _wfs_report_classic_exception(exc, workspace_mode, has_controls=has_controls)
            st.stop()


    def require_current_classic_action(selected_slate: str, *, mode: str) -> None:
        """Re-read publication and schedule immediately before a Classic action."""
        try:
            with sqlite3.connect(
                f"file:{DATABASE_PATH}?mode=ro", uri=True
            ) as conn:
                publication = pd.read_sql_query(f'SELECT * FROM "{SOLVER_TABLE}"', conn)
            current = current_classic_publication(publication)
            selected = publication.loc[
                publication["slate_name_solver"].map(normalize_slate).eq(
                    normalize_slate(selected_slate)
                )
            ]
            if selected.empty or len(current.loc[selected.index.intersection(current.index)]) != len(selected):
                raise RuntimeError("Selected Classic slate is no longer current.")
        except Exception as exc:
            _wfs_report_classic_exception(exc, mode, category="AUTHORITY")
            st.stop()


    def available_slates(df: pd.DataFrame) -> list[str]:
        if "slate_name_solver" not in df.columns:
            return []
        names = (
            df["slate_name_solver"]
            .dropna()
            .astype(str)
            .str.strip()
        )
        return sorted([x for x in names.unique() if x], key=str.lower)


    def slate_pool(df: pd.DataFrame, slate_name: str) -> pd.DataFrame:
        target = normalize_slate(slate_name)
        mask = (
            df["slate_name_solver"].map(normalize_slate).eq(target)
            | df["slate_slug_solver"].map(normalize_slate).eq(target)
        )
        out = df.loc[mask].copy()
        eligible = (
            pd.to_numeric(out["solver_eligible"], errors="coerce")
            .fillna(0)
            .astype(int)
            .eq(1)
        )
        out = out.loc[eligible].copy()
        out["salary_solver"] = pd.to_numeric(out["salary_solver"], errors="coerce")
        out["projection_solver"] = pd.to_numeric(
            out["projection_solver"], errors="coerce"
        )
        out = out.dropna(subset=["salary_solver", "projection_solver"])
        out["salary_solver"] = out["salary_solver"].astype(int)
        out["_ui_key"] = out.apply(ui_player_key, axis=1)
        out["_label"] = out.apply(
            lambda r: (
                f'{r["player_solver"]} — {r["team_solver"]} '
                f'{r["solver_position"]} — ${int(r["salary_solver"]):,} '
                f'— {float(r["projection_solver"]):.2f}'
            ),
            axis=1,
        )
        return out.sort_values(
            ["solver_position", "projection_solver", "salary_solver", "player_solver"],
            ascending=[True, False, False, True],
            kind="mergesort",
        )


    def late_swap_slate_pool(df: pd.DataFrame, slate_name: str) -> pd.DataFrame:
        # Full selected-slate view for Late Swap only.
        # Retains solver-blocked rows so already-started roster occupants can be preserved.
        target = normalize_slate(slate_name)
        mask = (
            df["slate_name_solver"].map(normalize_slate).eq(target)
            | df["slate_slug_solver"].map(normalize_slate).eq(target)
        )

        out = df.loc[mask].copy()

        out["salary_solver"] = pd.to_numeric(
            out["salary_solver"],
            errors="coerce",
        )
        out["projection_solver"] = pd.to_numeric(
            out["projection_solver"],
            errors="coerce",
        )

        out = out.dropna(subset=["salary_solver"]).copy()
        out = out.loc[out["salary_solver"].gt(0)].copy()
        out["salary_solver"] = out["salary_solver"].astype(int)

        out["_ui_key"] = out.apply(
            ui_player_key,
            axis=1,
        )

        def _late_swap_label(row):
            projection = row.get("projection_solver")
            projection_text = (
                f"{float(projection):.2f}"
                if pd.notna(projection)
                else "locked/no current projection"
            )

            position_value = row.get("solver_position")
            position = (
                str(position_value).strip()
                if pd.notna(position_value)
                else ""
            ) or "—"

            return (
                f'{row["player_solver"]} — {row["team_solver"]} '
                f'{position} — ${int(row["salary_solver"]):,} '
                f'— {projection_text}'
            )

        out["_label"] = out.apply(
            _late_swap_label,
            axis=1,
        )

        return out.sort_values(
            [
                "solver_position",
                "projection_solver",
                "salary_solver",
                "player_solver",
            ],
            ascending=[
                True,
                False,
                False,
                True,
            ],
            kind="mergesort",
            na_position="last",
        )


    LATE_SWAP_SLOT_POSITIONS = {
        "QB": {"QB"},
        "RB1": {"RB"},
        "RB2": {"RB"},
        "WR1": {"WR"},
        "WR2": {"WR"},
        "WR3": {"WR"},
        "TE": {"TE"},
        "FLEX": {"RB", "WR", "TE"},
        "DST": {"DST"},
    }


    def _late_swap_schedule_columns(schedule_df: pd.DataFrame) -> dict:
        """Discover schedule fields without inventing kickoff data."""
        if schedule_df is None or schedule_df.empty:
            return {}
        return {
            "home": _dc_col(schedule_df, ["home_team", "home"]),
            "away": _dc_col(schedule_df, ["away_team", "away"]),
            "season": _dc_col(schedule_df, ["season", "year"]),
            "date": _dc_col(
                schedule_df,
                ["start_time", "kickoff", "game_datetime", "gameday", "game_date", "date"],
            ),
            "time": _dc_col(schedule_df, ["gametime", "game_time", "time"]),
            "home_score": _dc_col(schedule_df, ["home_score", "homescore", "home_points"]),
            "away_score": _dc_col(schedule_df, ["away_score", "awayscore", "away_points"]),
        }


    def _late_swap_parse_game_teams(value) -> set[str]:
        """Extract the two team abbreviations from the solver game label when possible."""
        raw = str(value or "").upper()
        tokens = re.findall(r"\b[A-Z]{2,3}\b", raw)
        return {normalize_team(x) for x in tokens if normalize_team(x)}


    def _late_swap_kickoff_et(row: pd.Series, cols: dict):
        """
    Return an aware Eastern kickoff timestamp only when schedule data supports it.
    A date with no clock time is intentionally UNKNOWN (fail closed).
    """
        date_col = cols.get("date")
        time_col = cols.get("time")
        if not date_col or pd.isna(row.get(date_col)):
            return None

        date_raw = str(row.get(date_col)).strip()
        time_raw = ""
        if time_col and not pd.isna(row.get(time_col)):
            time_raw = str(row.get(time_col)).strip()

        has_clock_in_date = bool(re.search(r"\d{1,2}:\d{2}", date_raw)) or "T" in date_raw
        if time_raw:
            candidate = f"{date_raw} {time_raw}"
        elif has_clock_in_date:
            candidate = date_raw
        else:
            return None

        ts = pd.to_datetime(candidate, errors="coerce")
        if pd.isna(ts):
            return None

        try:
            py_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            if py_dt.tzinfo is None:
                return py_dt.replace(tzinfo=ZoneInfo("America/New_York"))
            return py_dt.astimezone(ZoneInfo("America/New_York"))
        except Exception:
            return None


    def _late_swap_player_status(player_row: pd.Series, schedule_df: pd.DataFrame, now_et: datetime) -> dict:
        """Read-only lock audit. UNKNOWN is returned whenever evidence is insufficient."""
        if schedule_df is None or schedule_df.empty:
            return {"status": "UNKNOWN", "kickoff": "—", "reason": "Schedule unavailable"}

        cols = _late_swap_schedule_columns(schedule_df)
        if not cols.get("home") or not cols.get("away"):
            return {"status": "UNKNOWN", "kickoff": "—", "reason": "Schedule teams unavailable"}

        team = normalize_team(player_row.get("team_solver"))
        game_teams = _late_swap_parse_game_teams(player_row.get("game_solver"))

        candidates = schedule_df.copy()
        home_norm = candidates[cols["home"]].map(normalize_team)
        away_norm = candidates[cols["away"]].map(normalize_team)
        mask = home_norm.eq(team) | away_norm.eq(team)

        # When the solver game label exposes both teams, require that exact matchup.
        if len(game_teams) >= 2:
            mask &= home_norm.isin(game_teams) & away_norm.isin(game_teams)
        candidates = candidates.loc[mask].copy()

        season_col = cols.get("season")
        if season_col and len(candidates):
            season_num = pd.to_numeric(candidates[season_col], errors="coerce")
            current = candidates.loc[season_num.eq(now_et.year)]
            if len(current):
                candidates = current

        if candidates.empty:
            return {"status": "UNKNOWN", "kickoff": "—", "reason": "Exact game not found"}

        evaluated = []
        for _, sched_row in candidates.iterrows():
            kickoff = _late_swap_kickoff_et(sched_row, cols)
            home_score = sched_row.get(cols.get("home_score")) if cols.get("home_score") else None
            away_score = sched_row.get(cols.get("away_score")) if cols.get("away_score") else None
            final_by_score = (
                home_score is not None and away_score is not None
                and pd.notna(home_score) and pd.notna(away_score)
            )
            evaluated.append((sched_row, kickoff, final_by_score))

        # Prefer a dated game closest to now; this avoids selecting an old same-team matchup.
        dated = [x for x in evaluated if x[1] is not None]
        if dated:
            sched_row, kickoff, final_by_score = min(
                dated,
                key=lambda x: abs((x[1] - now_et).total_seconds()),
            )
        elif len(evaluated) == 1:
            sched_row, kickoff, final_by_score = evaluated[0]
        else:
            return {"status": "UNKNOWN", "kickoff": "—", "reason": "Kickoff time unavailable"}

        if final_by_score:
            status = "LOCKED"
            reason = "Game has scoring data"
        elif kickoff is None:
            status = "UNKNOWN"
            reason = "Kickoff time unavailable"
        elif kickoff <= now_et:
            status = "LOCKED"
            reason = "Game has started"
        else:
            status = "SWAPPABLE"
            reason = "Game has not started"

        kickoff_text = kickoff.strftime("%a %m/%d %I:%M %p ET") if kickoff else "—"
        return {"status": status, "kickoff": kickoff_text, "reason": reason}


    def _late_swap_options_for_slot(pool: pd.DataFrame, slot: str) -> list[str]:
        allowed = LATE_SWAP_SLOT_POSITIONS[slot]
        rows = pool.loc[pool["solver_position"].astype(str).str.upper().isin(allowed)]
        return ["— Select player —"] + rows["_label"].astype(str).tolist()


    def render_late_swap_stage1(pool: pd.DataFrame, selected_slate: str) -> None:
        """
    Reconstruct, audit, and optionally optimize one existing FanDuel lineup.

    Stage 1:
        - exact current-lineup reconstruction
        - FanDuel legality audit
        - LOCKED / SWAPPABLE / UNKNOWN game-state audit

    Stage 2:
        - deterministic late-swap optimization
        - exact-slot preservation for LOCKED and UNKNOWN slots
        - only SWAPPABLE players may be newly introduced
        - production projection/GPP frontier semantics

    The frozen production solver and frozen Stage 2 solver are not modified.
    """
        st.markdown("#### Late Swap")

        st.caption(
            "Recreate the lineup exactly as it is currently entered on FanDuel. "
            "WFS audits every roster slot before any late-swap optimization occurs."
        )

        # -----------------------------------------------------------------
        # Optional exact FanDuel contest boundary
        # -----------------------------------------------------------------

        st.markdown("##### Contest Boundary")

        slate_slug = normalize_slate(selected_slate)

        upload_key = f"late_swap_template_{slate_slug}"
        bytes_key = f"late_swap_template_bytes_{slate_slug}"
        name_key = f"late_swap_template_name_{slate_slug}"

        late_swap_template = st.file_uploader(
            "FanDuel entries template for Late Swap (recommended)",
            type=["csv"],
            key=upload_key,
            help=(
                "When supplied, Stage 2 may select only players that match the "
                "exact FanDuel contest player pool. Exact matching only; no fuzzy "
                "player resolution."
            ),
        )

        # Persist the exact uploaded bytes across Streamlit reruns.
        # Late Swap must not silently lose its contest boundary when the user
        # changes lineup selections or clicks Optimize.
        if late_swap_template is not None:
            uploaded_bytes = late_swap_template.getvalue()

            if uploaded_bytes:
                st.session_state[bytes_key] = uploaded_bytes
                st.session_state[name_key] = str(late_swap_template.name)

        persisted_template_bytes = st.session_state.get(bytes_key)
        persisted_template_name = st.session_state.get(name_key)

        late_swap_pool = pool.copy()
        late_swap_contest_verified = False

        if persisted_template_bytes:
            try:
                contest_pool, blocked_pool, contest_meta = (
                    intersect_with_fanduel_contest(
                        pool,
                        persisted_template_bytes,
                    )
                )

                if contest_pool.empty:
                    st.error(
                        "Late Swap blocked: the uploaded FanDuel template has "
                        "zero exact overlap with this solver slate."
                    )
                    return

                late_swap_pool = contest_pool.copy()
                late_swap_contest_verified = True

                st.success(
                    f'Contest verified — '
                    f'{contest_meta["solver_players_in_contest"]} solver-ready '
                    f'players matched exactly; '
                    f'{contest_meta["solver_players_blocked"]} non-contest '
                    f'players blocked.'
                )

                if persisted_template_name:
                    st.caption(
                        f"Contest template locked for this session: "
                        f"{persisted_template_name}"
                    )

                if st.button(
                    "Clear Late Swap contest template",
                    key=f"late_swap_clear_template_{slate_slug}",
                ):
                    st.session_state.pop(bytes_key, None)
                    st.session_state.pop(name_key, None)
                    st.rerun()

            except Exception as exc:
                # Fail closed. Never fall back to the unrestricted slate after a
                # template has been supplied but cannot be validated.
                st.error(
                    "Late Swap contest validation could not be completed. "
                    "Please verify the FanDuel contest template and try again."
                )
                return

        else:
            st.warning(
                "No Late Swap FanDuel template is currently locked. "
                "Stage 2 can run against the selected solver slate, but exact "
                "FanDuel contest membership will not be verified."
            )

        if late_swap_pool.empty:
            st.error(
                "Late Swap blocked: no eligible players are available."
            )
            return

        # -----------------------------------------------------------------
        # Stage 1 — reconstruct current lineup
        # -----------------------------------------------------------------

        st.markdown("##### 1. Current FanDuel Lineup")

        st.caption(
            "Enter all nine players exactly as they currently appear in your "
            "FanDuel entry."
        )

        label_to_row = {
            str(row["_label"]): row
            for _, row in late_swap_pool.iterrows()
        }

        # -------------------------------------------------------------
        # Auto-load current FanDuel lineup from the verified template.
        #
        # FanDuel IDs are first resolved through FanDuel's own embedded
        # player table, then exact-matched to the already contest-filtered
        # WFS pool. No fuzzy matching is permitted.
        # -------------------------------------------------------------

        auto_slot_labels = {}
        auto_load_error = None

        if late_swap_contest_verified and persisted_template_bytes:
            try:
                parsed_template = parse_fanduel_template(
                    persisted_template_bytes
                )

                entry_lineups = parsed_template.get(
                    "entry_lineups",
                    [],
                )

                fd_players_by_id = parsed_template.get(
                    "player_rows_by_id",
                    {},
                )

                if len(entry_lineups) == 1:
                    fd_entry = entry_lineups[0]
                    fd_slots = fd_entry.get("slots", {})

                    for slot in ROSTER_SLOTS:
                        fd_id = str(
                            fd_slots.get(slot, "")
                        ).strip()

                        if not fd_id:
                            raise ValueError(
                                f"FanDuel entry is missing slot {slot}."
                            )

                        fd_player = fd_players_by_id.get(fd_id)

                        if fd_player is None:
                            raise ValueError(
                                f"FanDuel player ID {fd_id!r} from slot "
                                f"{slot} is missing from the embedded "
                                "FanDuel player table."
                            )

                        fd_name_key = normalize_fanduel_name(
                            fd_player.get("nickname", "")
                        )

                        fd_team = normalize_team(
                            fd_player.get("team", "")
                        )

                        fd_position = str(
                            fd_player.get("position", "")
                        ).strip().upper()

                        fd_salary = fd_player.get(
                            "salary"
                        )

                        # FanDuel identity has already been canonicalized
                        # by normalize_fanduel_name(). No fuzzy matching.
                        fd_match_name_key = fd_name_key

                        matches = []

                        for _, candidate in late_swap_pool.iterrows():
                            candidate_name_key = normalize_name(
                                candidate["player_solver"]
                            )

                            candidate_team = normalize_team(
                                candidate["team_solver"]
                            )

                            candidate_position = str(
                                candidate["solver_position"]
                            ).strip().upper()

                            try:
                                candidate_salary = int(
                                    candidate["salary_solver"]
                                )
                            except Exception:
                                candidate_salary = None

                            if fd_position in {"D", "DEF", "DST"}:
                                identity_match = (
                                    candidate_position
                                    in {"DST", "DEF", "D"}
                                    and candidate_team == fd_team
                                )
                            else:
                                identity_match = (
                                    candidate_name_key
                                    == fd_match_name_key
                                    and candidate_team == fd_team
                                    and candidate_position == fd_position
                                )

                            # When FanDuel supplied salary, require it too.
                            # This adds another exact identity guard.
                            if (
                                identity_match
                                and fd_salary is not None
                                and candidate_salary is not None
                            ):
                                identity_match = (
                                    int(fd_salary)
                                    == candidate_salary
                                )

                            if identity_match:
                                matches.append(candidate)

                        if len(matches) != 1:
                            raise ValueError(
                                f"FanDuel slot {slot} / player "
                                f"{fd_player.get('nickname', fd_id)!r} "
                                f"resolved to {len(matches)} WFS players; "
                                "exactly one is required."
                            )

                        matched_row = matches[0]
                        auto_slot_labels[slot] = str(
                            matched_row["_label"]
                        )

                    if len(auto_slot_labels) != len(ROSTER_SLOTS):
                        raise ValueError(
                            "FanDuel current lineup did not resolve "
                            "all nine roster slots."
                        )

                elif len(entry_lineups) > 1:
                    st.info(
                        f"FanDuel template contains {len(entry_lineups)} "
                        "existing lineups. Automatic current-lineup loading "
                        "is disabled because the specific entry to late swap "
                        "is ambiguous."
                    )

            except Exception as exc:
                auto_load_error = (
                    "Automatic FanDuel lineup loading could not be completed. "
                    "Please verify the uploaded contest template."
                )

        if auto_load_error:
            st.error(
                "Automatic FanDuel lineup loading failed closed: "
                f"{auto_load_error}"
            )
            return

        # -------------------------------------------------------------
        # Initialize Streamlit widget state exactly once per template.
        #
        # The template fingerprint prevents reruns from continually
        # overwriting a user's intentional manual dropdown change.
        # -------------------------------------------------------------

        auto_state_key = (
            f"late_swap_autoload_signature_{slate_slug}"
        )

        template_signature = (
            hashlib.sha256(persisted_template_bytes).hexdigest()
            if persisted_template_bytes
            else ""
        )

        if auto_slot_labels:
            previous_signature = st.session_state.get(
                auto_state_key
            )

            if previous_signature != template_signature:
                for slot in ROSTER_SLOTS:
                    widget_key = (
                        f"late_swap_stage1_"
                        f"{normalize_slate(selected_slate)}_{slot}"
                    )

                    options = _late_swap_options_for_slot(
                        late_swap_pool,
                        slot,
                    )

                    default_label = auto_slot_labels[slot]

                    if default_label not in options:
                        st.error(
                            f"Automatic FanDuel lineup loading failed: "
                            f"{slot} player is not a legal option for "
                            "that roster slot."
                        )
                        return

                    st.session_state[widget_key] = default_label

                st.session_state[
                    auto_state_key
                ] = template_signature

                st.success(
                    "Current FanDuel lineup loaded automatically "
                    "from the uploaded template."
                )

        selections = {}

        for slot in ROSTER_SLOTS:
            selections[slot] = st.selectbox(
                slot,
                _late_swap_options_for_slot(
                    late_swap_pool,
                    slot,
                ),
                key=(
                    f"late_swap_stage1_"
                    f"{normalize_slate(selected_slate)}_{slot}"
                ),
            )

        chosen = {
            slot: label
            for slot, label in selections.items()
            if label != "— Select player —"
        }

        if not chosen:
            st.caption(
                "Select the players from the existing FanDuel lineup "
                "to begin the audit."
            )
            return

        selected_rows = [
            label_to_row[label]
            for label in chosen.values()
        ]

        selected_keys = [
            str(row["_ui_key"])
            for row in selected_rows
        ]

        duplicate_players = (
            len(selected_keys)
            != len(set(selected_keys))
        )

        total_salary = sum(
            int(row["salary_solver"])
            for row in selected_rows
        )

        team_counts = pd.Series(
            [
                normalize_team(row["team_solver"])
                for row in selected_rows
            ],
            dtype="object",
        ).value_counts().to_dict()

        max_team_count = (
            max(team_counts.values())
            if team_counts
            else 0
        )

        if duplicate_players:
            st.error(
                "The same player cannot occupy more than one "
                "FanDuel roster slot."
            )

        if max_team_count > 4:
            st.error(
                "FanDuel legality failed: more than 4 selected "
                "players are from one NFL team."
            )

        if total_salary > 60000:
            st.error(
                "FanDuel legality failed: selected salary exceeds "
                "the $60,000 cap."
            )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Slots entered",
            f"{len(chosen)}/9",
        )

        c2.metric(
            "Salary entered",
            f"${total_salary:,.0f}",
        )

        c3.metric(
            "Remaining",
            f"${max(0, 60000 - total_salary):,.0f}",
        )

        # -----------------------------------------------------------------
        # Game-state audit
        # -----------------------------------------------------------------

        try:
            schedule_df, _, _, _ = load_data_center()
        except Exception:
            schedule_df = pd.DataFrame()

        now_et = datetime.now(
            ZoneInfo("America/New_York")
        )

        audit_rows = []

        current_slot_keys = {}
        slot_status = {}

        for slot in ROSTER_SLOTS:
            label = selections[slot]

            if label == "— Select player —":
                audit_rows.append(
                    {
                        "Slot": slot,
                        "Player": "—",
                        "Team": "—",
                        "Salary": "—",
                        "Kickoff": "—",
                        "Status": "UNKNOWN",
                        "Reason": "Player not entered",
                    }
                )
                continue

            row = label_to_row[label]

            lock = _late_swap_player_status(
                row,
                schedule_df,
                now_et,
            )

            key = str(row["_ui_key"])

            current_slot_keys[slot] = key
            slot_status[slot] = lock["status"]

            audit_rows.append(
                {
                    "Slot": slot,
                    "Player": str(row["player_solver"]),
                    "Team": str(row["team_solver"]),
                    "Salary": (
                        f'${int(row["salary_solver"]):,}'
                    ),
                    "Kickoff": lock["kickoff"],
                    "Status": lock["status"],
                    "Reason": lock["reason"],
                }
            )

        audit_df = pd.DataFrame(audit_rows)

        st.markdown("##### 2. Player Lock Status")

        st.dataframe(
            audit_df,
            width="stretch",
            hide_index=True,
        )

        valid_current_lineup = (
            len(chosen) == 9
            and not duplicate_players
            and max_team_count <= 4
            and total_salary <= 60000
        )

        if not valid_current_lineup:
            st.error(
                "Complete a valid nine-player FanDuel lineup before "
                "continuing with Late Swap."
            )
            return

        unknown_count = int(
            audit_df["Status"].eq("UNKNOWN").sum()
        )

        locked_count = int(
            audit_df["Status"].eq("LOCKED").sum()
        )

        swappable_count = int(
            audit_df["Status"].eq("SWAPPABLE").sum()
        )

        if unknown_count:
            st.warning(
                f"{unknown_count} roster slot(s) could not be confirmed as swappable. "
                "Those players will remain protected and will not be changed."
            )
        else:
            st.success(
                f"Late Swap status confirmed — "
                f"{locked_count} locked slot(s), "
                f"{swappable_count} available to swap."
            )

        # -----------------------------------------------------------------
        # Build player-level state for the entire legal candidate pool.
        #
        # Missing/uncertain evidence intentionally becomes UNKNOWN.
        # -----------------------------------------------------------------

        player_status = {}

        for _, player_row in late_swap_pool.iterrows():
            player_key = str(player_row["_ui_key"])

            state = _late_swap_player_status(
                player_row,
                schedule_df,
                now_et,
            )

            player_status[player_key] = state["status"]

        # -----------------------------------------------------------------
        # Current-lineup display
        # -----------------------------------------------------------------

        original_rows = []

        for slot in ROSTER_SLOTS:
            row = label_to_row[selections[slot]]

            original_rows.append(
                {
                    "Slot": slot,
                    "Player": str(row["player_solver"]),
                    "Team": str(row["team_solver"]),
                    "Pos": str(row["solver_position"]),
                    "Salary": int(row["salary_solver"]),
                    "Projection": float(
                        row["projection_solver"]
                    ),
                    "Game": str(row.get("game_solver", "")),
                    "Status": slot_status[slot],
                    "Player Key": str(row["_ui_key"]),
                }
            )

        original_df = pd.DataFrame(original_rows)

        # -----------------------------------------------------------------
        # Stage 2 launch
        # -----------------------------------------------------------------

        st.markdown("##### 3. Optimize Remaining Lineup")

        if swappable_count == 0:
            st.info(
                "No roster slots are currently available to swap."
            )
            return

        st.caption(
            "Players whose games have started or cannot be safely confirmed remain protected. "
            "Only eligible players from games that have not started can be added."
        )

        optimize = st.button(
            "⚡ Optimize Late Swap",
            type="primary",
            width="stretch",
            key=(
                f"late_swap_optimize_"
                f"{normalize_slate(selected_slate)}"
            ),
        )

        if not optimize:
            return

        # Production-aligned settings used by the validated Stage 2 tests.
        settings = LateSwapSettings(
            qb_stack=1,
            bring_back=0,
            projection_loss_pct=0.015,
            salary_floor=0,
            max_team=0,
        )

        try:
            with st.spinner(
                "Optimizing your Late Swap lineup…"
            ):
                result = optimize_late_swap(
                    pool=late_swap_pool,
                    current_slot_keys=current_slot_keys,
                    slot_status=slot_status,
                    player_status=player_status,
                    settings=settings,
                )

        except Exception as exc:
            st.error(
                "Late Swap could not be completed. Please review the lineup and try again."
            )
            return

        if not result.get("audit_ok", False):
            st.error(
                "The generated Late Swap lineup did not meet FanDuel lineup requirements."
            )
            return

        if not result.get("exact_lock_ok", False):
            st.error(
                "Late Swap could not safely preserve all locked players."
            )
            return

        result_df = pd.DataFrame(
            result["lineup"]
        ).copy()

        slot_order = {
            slot: n
            for n, slot in enumerate(ROSTER_SLOTS)
        }

        result_df["_slot_order"] = (
            result_df["slot"]
            .map(slot_order)
            .fillna(999)
        )

        result_df = (
            result_df
            .sort_values(
                "_slot_order",
                kind="mergesort",
            )
            .drop(columns=["_slot_order"])
            .reset_index(drop=True)
        )

        # -----------------------------------------------------------------
        # True roster-composition changes
        # -----------------------------------------------------------------

        original_key_to_player = dict(
            zip(
                original_df["Player Key"].astype(str),
                original_df["Player"].astype(str),
            )
        )

        result_key_to_player = dict(
            zip(
                result_df["player_key"].astype(str),
                result_df["player"].astype(str),
            )
        )

        original_keys = set(
            original_key_to_player
        )

        result_keys = set(
            result_key_to_player
        )

        removed_keys = sorted(
            original_keys - result_keys
        )

        added_keys = sorted(
            result_keys - original_keys
        )

        players_removed = [
            original_key_to_player[key]
            for key in removed_keys
        ]

        players_added = [
            result_key_to_player[key]
            for key in added_keys
        ]

        roster_composition_changed = bool(
            players_removed or players_added
        )

        # -----------------------------------------------------------------
        # Result headline
        # -----------------------------------------------------------------

        st.markdown("##### 4. Optimized Late Swap")

        if roster_composition_changed:
            st.success(
                "Stage 2 found a legal roster-composition change."
            )
        elif result.get("changed_slot_count", 0):
            st.info(
                "Stage 2 changed slot assignments but retained "
                "the same nine players."
            )
        else:
            st.info(
                "Current lineup remains the strongest legal "
                "late-swap result inside the projection frontier."
            )

        # -----------------------------------------------------------------
        # Metrics
        # -----------------------------------------------------------------

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Salary",
            f'${int(result["salary_total"]):,}',
        )

        m2.metric(
            "Remaining",
            f'${int(result["salary_remaining"]):,}',
        )

        m3.metric(
            "Projection",
            f'{float(result["projection_total"]):.3f}',
        )

        m4.metric(
            "Changed Slots",
            int(result["changed_slot_count"]),
        )

        m5, m6, m7, m8 = st.columns(4)

        m5.metric(
            "Projection Optimum",
            f'{float(result["projection_optimum"]):.3f}',
        )

        m6.metric(
            "Projection Floor",
            f'{float(result["projection_floor"]):.3f}',
        )

        m7.metric(
            "Max One Team",
            int(result["max_players_one_team"]),
        )

        m8.metric(
            "Team Limit",
            int(result["team_limit"]),
        )

        # -----------------------------------------------------------------
        # Before / after
        # -----------------------------------------------------------------

        left, right = st.columns(2)

        with left:
            st.markdown("###### Before")

            before_display = original_df[
                [
                    "Slot",
                    "Player",
                    "Team",
                    "Pos",
                    "Salary",
                    "Projection",
                    "Game",
                    "Status",
                ]
            ].copy()

            before_display["Salary"] = (
                before_display["Salary"]
                .map(lambda x: f"${int(x):,}")
            )

            before_display["Projection"] = (
                before_display["Projection"]
                .map(lambda x: f"{float(x):.3f}")
            )

            st.dataframe(
                before_display,
                width="stretch",
                hide_index=True,
            )

        with right:
            st.markdown("###### After")

            after_display = result_df.rename(
                columns={
                    "slot": "Slot",
                    "player": "Player",
                    "team": "Team",
                    "position": "Pos",
                    "salary": "Salary",
                    "projection": "Projection",
                    "game": "Game",
                    "late_swap_status": "Status",
                }
            )[
                [
                    "Slot",
                    "Player",
                    "Team",
                    "Pos",
                    "Salary",
                    "Projection",
                    "Game",
                    "Status",
                ]
            ].copy()

            after_display["Salary"] = (
                after_display["Salary"]
                .map(lambda x: f"${int(x):,}")
            )

            after_display["Projection"] = (
                after_display["Projection"]
                .map(lambda x: f"{float(x):.3f}")
            )

            st.dataframe(
                after_display,
                width="stretch",
                hide_index=True,
            )

        # -----------------------------------------------------------------
        # True OUT / IN changes
        # -----------------------------------------------------------------

        st.markdown("##### 5. Roster Changes")

        change_left, change_right = st.columns(2)

        with change_left:
            st.markdown("###### Players OUT")

            if players_removed:
                for player in players_removed:
                    st.write(f"• {player}")
            else:
                st.caption("None")

        with change_right:
            st.markdown("###### Players IN")

            if players_added:
                for player in players_added:
                    st.write(f"• {player}")
            else:
                st.caption("None")

        changed_slots = result.get(
            "changed_slots",
            [],
        )

        if changed_slots:
            st.caption(
                "Slot assignments changed: "
                + ", ".join(changed_slots)
            )
        else:
            st.caption(
                "No exact slot assignments changed."
            )

        # -----------------------------------------------------------------
        # FanDuel Late Swap export
        # -----------------------------------------------------------------

        st.markdown("##### 6. FanDuel Late Swap Export")

        if late_swap_contest_verified and persisted_template_bytes:
            try:
                # Convert Stage 2's slot-oriented result into the same
                # one-row lineup shape used by the shared FanDuel exporter.
                result_by_slot = {
                    str(row["slot"]): str(row["player"])
                    for _, row in result_df.iterrows()
                }

                missing_export_slots = [
                    slot
                    for slot in ROSTER_SLOTS
                    if slot not in result_by_slot
                    or not result_by_slot[slot].strip()
                ]

                if missing_export_slots:
                    raise ValueError(
                        "Late Swap export missing roster slot(s): "
                        + ", ".join(missing_export_slots)
                    )

                late_swap_export_df = pd.DataFrame(
                    [
                        {
                            "lineup": 1,
                            "QB": result_by_slot["QB"],
                            "RB1": result_by_slot["RB1"],
                            "RB2": result_by_slot["RB2"],
                            "WR1": result_by_slot["WR1"],
                            "WR2": result_by_slot["WR2"],
                            "WR3": result_by_slot["WR3"],
                            "TE": result_by_slot["TE"],
                            "FLEX": result_by_slot["FLEX"],
                            "DST": result_by_slot["DST"],
                        }
                    ]
                )

                late_swap_fd_bytes, late_swap_fd_meta = (
                    build_fanduel_upload(
                        persisted_template_bytes,
                        late_swap_export_df,
                        late_swap_pool,
                    )
                )

                validate_fanduel_upload(
                    late_swap_fd_bytes,
                    late_swap_fd_meta["entries_written"],
                )

                if late_swap_fd_meta["entries_written"] != 1:
                    raise ValueError(
                        "Late Swap export expected exactly one existing "
                        "FanDuel entry row to be updated."
                    )

                if late_swap_fd_meta["mapping_count"] != 9:
                    raise ValueError(
                        "Late Swap export did not map exactly nine "
                        "FanDuel roster cells."
                    )

                st.success(
                    "FanDuel Late Swap export PASS — "
                    "9/9 roster cells mapped to exact contest-specific "
                    "FanDuel player IDs."
                )

                st.download_button(
                    "Download FanDuel Late Swap CSV",
                    data=late_swap_fd_bytes,
                    file_name=(
                        f"FanDuel-NFL-{safe_slug(selected_slate)}-"
                        "late-swap-upload.csv"
                    ),
                    mime="text/csv",
                    type="primary",
                    key=(
                        f"late_swap_fd_download_"
                        f"{normalize_slate(selected_slate)}"
                    ),
                    on_click="ignore",
                )

                st.caption(
                    "The original FanDuel template structure is preserved. "
                    "Only the nine roster cells of the existing entry are changed."
                )

            except Exception as exc:
                st.error(
                    "FanDuel Late Swap export was blocked because the lineup "
                    "could not be validated against the selected contest."
                )
                st.caption(
                    "Export requires all nine players to match the selected FanDuel contest."
                )

        else:
            st.warning(
                "FanDuel Late Swap export requires a verified "
                "FanDuel contest template."
            )

        # -----------------------------------------------------------------
        # Hard audit summary
        # -----------------------------------------------------------------

        st.markdown("##### 7. Late Swap Summary")

        a1, a2, a3, a4 = st.columns(4)

        a1.metric(
            "FanDuel Lineup",
            "PASS" if result.get("audit_ok") else "FAIL",
        )

        a2.metric(
            "Locked Players",
            "PASS" if result.get("exact_lock_ok") else "FAIL",
        )

        a3.metric(
            "Locked Slots",
            len(result.get("locked_slots", [])),
        )

        a4.metric(
            "Available Slots",
            len(result.get("swappable_slots", [])),
        )

        locked_slots = result.get(
            "locked_slots",
            [],
        )

        swappable_slots = result.get(
            "swappable_slots",
            [],
        )

        st.caption(
            "Protected exact-slot cells: "
            + (
                ", ".join(locked_slots)
                if locked_slots
                else "None"
            )
        )

        st.caption(
            "Replaceable cells: "
            + (
                ", ".join(swappable_slots)
                if swappable_slots
                else "None"
            )
        )

        if late_swap_contest_verified:
            st.success(
                "Late Swap is ready for the selected FanDuel contest."
            )
        else:
            st.warning(
                "Late Swap lineup created, but a FanDuel contest template was not provided. "
                "Confirm player eligibility before uploading."
            )



    def intersect_with_fanduel_contest(
        slate_df: pd.DataFrame,
        template_bytes: bytes,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
    Intersect the solver-ready slate with the exact FanDuel contest player pool.

    This changes only player eligibility at the UI boundary. It does not alter
    projections, GPP ranks, objectives, exposure logic, or the frozen solver.

    Offense requires exact normalized player name + team + position.
    D/ST requires exact team and FanDuel position D.
    """
        parsed = parse_fanduel_template(template_bytes)
        fd_players = parsed["player_rows"]

        allowed = []
        blocked = []

        for idx, row in slate_df.iterrows():
            name_key = normalize_name(row["player_solver"])
            team = normalize_team(row["team_solver"])
            pos = str(row["solver_position"]).strip().upper()

            if pos in {"DST", "DEF", "D"}:
                matches = [
                    p for p in fd_players
                    if p["position"] == "D" and p["team"] == team
                ]
            else:
                matches = [
                    p for p in fd_players
                    if p["name_key"] == name_key
                    and p["team"] == team
                    and p["position"] == pos
                ]

            if len(matches) == 1:
                allowed.append(idx)
            else:
                blocked.append(idx)

        contest_pool = slate_df.loc[allowed].copy()
        outside_pool = slate_df.loc[blocked].copy()

        fd_teams = sorted({
            p["team"] for p in fd_players if p["team"]
        })
        solver_teams = sorted({
            normalize_team(x) for x in contest_pool["team_solver"].dropna()
        })

        return contest_pool, outside_pool, {
            "fanduel_players": len(fd_players),
            "solver_players_before": len(slate_df),
            "solver_players_in_contest": len(contest_pool),
            "solver_players_blocked": len(outside_pool),
            "fanduel_teams": fd_teams,
            "solver_teams_in_contest": solver_teams,
            "entry_count": len(parsed["entry_row_indexes"]),
        }


    def apply_showdown_injury_gate(df: pd.DataFrame) -> pd.DataFrame:
        """
    Apply authoritative injury/availability eligibility to the
    isolated Single-Game UI universe.

    Single-Game inherits Classic semantics:
      * BLOCK may only remove eligibility.
      * Missing consensus is healthy-by-absence.
      * Malformed authority fails closed.

    The underlying projection pool is not modified.
    """
        injury_path = (
            APP_DIR
            / "data"
            / "parquet"
            / "injury_consensus_current.parquet"
        )

        if not injury_path.exists():
            raise RuntimeError(
                "Single-Game availability authority is unavailable."
            )

        injury = pd.read_parquet(injury_path).copy()

        required = {
            "gsis_id",
            "injury_gate",
            "consensus_status",
            "injury_gate_reason",
            "injury_risk",
        }

        missing = sorted(required - set(injury.columns))

        if missing:
            raise RuntimeError(
                "Single-Game availability authority is missing "
                "required fields: "
                + ", ".join(missing)
            )

        if injury.empty:
            raise RuntimeError(
                "Single-Game availability authority returned no rows."
            )

        for col in required:
            injury[col] = (
                injury[col]
                .fillna("")
                .astype(str)
                .str.strip()
            )

        injury["injury_gate"] = (
            injury["injury_gate"]
            .str.upper()
        )

        if injury["gsis_id"].eq("").any():
            raise RuntimeError(
                "Single-Game availability authority contains "
                "blank player IDs."
            )

        duplicate_ids = injury["gsis_id"].duplicated(
            keep=False
        )

        if duplicate_ids.any():
            raise RuntimeError(
                "Single-Game availability authority contains "
                "duplicate player IDs."
            )

        invalid = sorted(
            set(injury["injury_gate"])
            - {"ALLOW", "BLOCK"}
        )

        if invalid:
            raise RuntimeError(
                "Single-Game availability authority contains "
                "invalid gate values."
            )

        block_ids = set(
            injury.loc[
                injury["injury_gate"].eq("BLOCK"),
                "gsis_id",
            ]
        )

        out = df.copy()

        if "player_id" not in out.columns:
            raise RuntimeError(
                "Single-Game player pool is missing player_id."
            )

        player_ids = (
            out["player_id"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        if player_ids.eq("").any():
            raise RuntimeError(
                "Single-Game player pool contains blank player IDs."
            )

        blocked_mask = player_ids.isin(block_ids)

        out = out.loc[
            ~blocked_mask
        ].copy()

        remaining_ids = set(
            out["player_id"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        escapes = remaining_ids & block_ids

        if escapes:
            raise RuntimeError(
                "Single-Game availability gate escape detected."
            )

        return out


    def available_showdown_slates(df: pd.DataFrame) -> list[str]:
        """
    Return complete READY Single-Game slates whose authoritative NFL game
    is still active.

    Lifecycle authority is nfl.db -> games.completed.

    The Single-Game artifact currently carries matchup identity rather than
    game_id. Resolution therefore fails closed unless the matchup maps
    unambiguously to exactly one relevant schedule row.
    """
        if (
            df is None
            or df.empty
            or "public_slate_name" not in df.columns
            or "game" not in df.columns
        ):
            return []

        df = apply_showdown_injury_gate(df)

        if df.empty:
            return []

        def _clean_team(value) -> str:
            team = str(value or "").strip().upper()
            aliases = {
                "LA": "LAR",
                "WAS": "WSH",
                "JAC": "JAX",
            }
            return aliases.get(team, team)

        def _parse_matchup(value):
            raw = str(value or "").strip().upper()
            if "@" not in raw:
                return None

            parts = raw.split("@")
            if len(parts) != 2:
                return None

            away = _clean_team(parts[0])
            home = _clean_team(parts[1])

            if not away or not home:
                return None

            return away, home

        # Read schedule authority once for the selector build.
        try:
            with sqlite3.connect(
                f"file:{DATABASE_PATH}?mode=ro",
                uri=True,
            ) as conn:
                conn.row_factory = sqlite3.Row
                schedule_rows = conn.execute(
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
                """
                ).fetchall()
        except Exception:
            # Schedule authority is mandatory for lineup-generation eligibility.
            return []

        ready = []

        for slate_name, group in df.groupby(
            "public_slate_name",
            sort=True,
        ):
            slate_name = str(slate_name).strip()
            if not slate_name:
                continue

            # ------------------------------------------------------
            # Existing projection / identity readiness gates.
            # ------------------------------------------------------
            status_ok = (
                group["projection_status"]
                .fillna("")
                .astype(str)
                .str.upper()
                .str.startswith("READY")
                .all()
            )

            numeric_ok = (
                group[
                    [
                        "salary",
                        "projection",
                        "mvp_salary",
                        "mvp_projection",
                    ]
                ]
                .notna()
                .all()
                .all()
            )

            identity_ok = (
                group["player_id"]
                .fillna("")
                .astype(str)
                .str.strip()
                .ne("")
                .all()
                and group["team"]
                .fillna("")
                .astype(str)
                .str.strip()
                .ne("")
                .all()
            )

            unique_ok = not group["player_id"].astype(str).duplicated().any()

            if not (
                status_ok
                and numeric_ok
                and identity_ok
                and unique_ok
            ):
                continue

            # ------------------------------------------------------
            # Exact matchup identity must be internally consistent.
            # ------------------------------------------------------
            game_values = sorted(
                {
                    str(v).strip()
                    for v in group["game"].tolist()
                    if str(v).strip()
                    and str(v).strip().lower()
                    not in {"nan", "none", "null"}
                }
            )

            if len(game_values) != 1:
                continue

            matchup = _parse_matchup(game_values[0])
            if matchup is None:
                continue

            away_team, home_team = matchup

            # ------------------------------------------------------
            # Schedule lifecycle authority.
            #
            # Prefer the latest scheduled occurrence of the exact
            # away/home orientation. If more than one row exists on
            # that latest date, identity is ambiguous -> fail closed.
            # ------------------------------------------------------
            matches = [
                row
                for row in schedule_rows
                if _clean_team(row["away_team"]) == away_team
                and _clean_team(row["home_team"]) == home_team
            ]

            if not matches:
                continue

            dated_matches = [
                row
                for row in matches
                if str(row["game_date"] or "").strip()
            ]

            if not dated_matches:
                continue

            latest_date = max(
                str(row["game_date"]).strip()
                for row in dated_matches
            )

            relevant = [
                row
                for row in dated_matches
                if str(row["game_date"]).strip() == latest_date
            ]

            if len(relevant) != 1:
                continue

            game_row = relevant[0]

            try:
                completed = bool(int(game_row["completed"] or 0))
            except Exception:
                # Malformed lifecycle authority fails closed.
                continue

            if completed:
                continue

            ready.append(slate_name)

        return sorted(ready, key=str.lower)

    def render_public_showdown_lineup_cards(
        display_df: pd.DataFrame,
        long_df: pd.DataFrame,
    ) -> None:
        """Render Single-Game lineups using the Classic public card presentation."""

        import html

        if display_df is None or display_df.empty:
            st.info("No lineups to display.")
            return

        if long_df is None or long_df.empty:
            raise RuntimeError(
                "Single-Game lineup card detail is unavailable."
            )

        required = {
            "lineup",
            "role",
            "player",
            "team",
            "salary",
            "projection",
            "lineup_salary",
            "lineup_projection",
        }
        missing = sorted(required - set(long_df.columns))
        if missing:
            raise RuntimeError(
                "Single-Game lineup card detail is missing required fields: "
                + ", ".join(missing)
            )

        cards = []

        for _, lineup_row in display_df.iterrows():
            lineup_id = int(lineup_row["Lineup"])

            detail = long_df[
                pd.to_numeric(long_df["lineup"], errors="coerce")
                == lineup_id
            ].copy()

            if len(detail) != 6:
                raise RuntimeError(
                    f"Single-Game lineup card expected 6 players for lineup "
                    f"{lineup_id}; found {len(detail)}."
                )

            detail["_role_order"] = (
                detail["role"]
                .astype(str)
                .str.upper()
                .map({"MVP": 0, "FLEX": 1})
                .fillna(99)
            )
            detail["_original_order"] = range(len(detail))
            detail = detail.sort_values(
                ["_role_order", "_original_order"],
                kind="mergesort",
            )

            salary = pd.to_numeric(
                pd.Series([lineup_row.get("Salary")]),
                errors="coerce",
            ).iloc[0]

            projection = pd.to_numeric(
                pd.Series([lineup_row.get("Projection")]),
                errors="coerce",
            ).iloc[0]

            salary_text = (
                f"${salary:,.0f}"
                if pd.notna(salary)
                else "—"
            )
            projection_text = (
                f"{projection:.2f} PTS"
                if pd.notna(projection)
                else "—"
            )

            rows = []

            for _, player_row in detail.iterrows():
                role = str(
                    player_row.get("role", "")
                ).strip().upper()

                display_slot = (
                    "MVP"
                    if role == "MVP"
                    else "FLEX"
                )

                player = html.escape(
                    str(player_row.get("player", "—"))
                )

                raw_team = str(
                    player_row.get("team", "") or ""
                ).strip()

                canonical_team = _dc_team_code(raw_team)
                team = html.escape(
                    canonical_team or raw_team or "—"
                )

                opponent_value = (
                    player_row.get("opponent", "—")
                    if "opponent" in detail.columns
                    else "—"
                )
                opponent = html.escape(
                    str(opponent_value or "—")
                )

                team_logo = _dc_team_logo_html(
                    canonical_team,
                    "wfs-lc-team-logo",
                )

                if role == "MVP":
                    player_salary = pd.to_numeric(
                        pd.Series([
                            player_row.get(
                                "mvp_salary",
                                player_row.get("salary"),
                            )
                        ]),
                        errors="coerce",
                    ).iloc[0]

                    player_projection = pd.to_numeric(
                        pd.Series([
                            player_row.get(
                                "mvp_projection",
                                player_row.get("projection"),
                            )
                        ]),
                        errors="coerce",
                    ).iloc[0]
                else:
                    player_salary = pd.to_numeric(
                        pd.Series([
                            player_row.get("salary")
                        ]),
                        errors="coerce",
                    ).iloc[0]

                    player_projection = pd.to_numeric(
                        pd.Series([
                            player_row.get("projection")
                        ]),
                        errors="coerce",
                    ).iloc[0]

                player_salary_text = (
                    f"${player_salary / 1000:.1f}K"
                    if pd.notna(player_salary)
                    else "—"
                )

                player_projection_text = (
                    f"{player_projection:.2f}"
                    if pd.notna(player_projection)
                    else "—"
                )

                rows.append(
                    f"""
                <div class="wfs-lc-row">
                    <div class="wfs-lc-pos">{html.escape(display_slot)}</div>
                    <div class="wfs-lc-player">
                        {team_logo}
                        <span class="wfs-lc-player-name">{player}</span>
                    </div>
                    <div class="wfs-lc-team">{team}</div>
                    <div class="wfs-lc-opp">{opponent}</div>
                    <div class="wfs-lc-sal">{player_salary_text}</div>
                    <div class="wfs-lc-fpts">{player_projection_text}</div>
                </div>
                """
                )

            cards.append(
                f"""
            <div class="wfs-lineup-card">
                <div class="wfs-lc-head">
                    <div class="wfs-lc-number">#{lineup_id}</div>
                    <div class="wfs-lc-proj">{projection_text}</div>
                    <div class="wfs-lc-salary">{salary_text}</div>
                </div>

                <div class="wfs-lc-columns">
                    <div>POS</div>
                    <div>PLAYER</div>
                    <div>TEAM</div>
                    <div>OPP</div>
                    <div>SAL</div>
                    <div>FPTS</div>
                </div>

                {''.join(rows)}
            </div>
            """
            )

        st.markdown(
            """
        <style>
        .wfs-lineup-grid {
            display:grid;
            grid-template-columns:repeat(auto-fit,minmax(430px,1fr));
            gap:14px;
            width:100%;
            margin:.35rem 0 1rem 0;
        }

        .wfs-lineup-card {
            min-width:0;
            overflow:hidden;
            border:1px solid rgba(148,163,184,.22);
            border-radius:12px;
            background:rgba(15,23,42,.72);
            box-shadow:0 8px 24px rgba(0,0,0,.14);
        }

        .wfs-lc-head {
            display:grid;
            grid-template-columns:1fr auto auto;
            gap:14px;
            align-items:center;
            padding:11px 13px;
            border-bottom:1px solid rgba(148,163,184,.20);
        }

        .wfs-lc-number {
            font-size:1rem;
            font-weight:900;
            color:#f8fafc;
        }

        .wfs-lc-proj {
            font-size:.88rem;
            font-weight:900;
            color:#e2e8f0;
            white-space:nowrap;
        }

        .wfs-lc-salary {
            font-size:.88rem;
            font-weight:900;
            color:#f8fafc;
            white-space:nowrap;
        }

        .wfs-lc-columns,
        .wfs-lc-row {
            display:grid;
            grid-template-columns:42px minmax(130px,1fr) 45px 45px 55px 52px;
            gap:5px;
            align-items:center;
        }

        .wfs-lc-columns {
            padding:7px 10px;
            font-size:.62rem;
            font-weight:900;
            letter-spacing:.04em;
            color:#94a3b8;
            border-bottom:1px solid rgba(148,163,184,.16);
        }

        .wfs-lc-row {
            padding:8px 10px;
            min-height:35px;
            border-bottom:1px solid rgba(148,163,184,.10);
            font-size:.76rem;
            color:#e2e8f0;
        }

        .wfs-lc-row:last-child {
            border-bottom:0;
        }

        .wfs-lc-pos {
            font-weight:900;
            color:#cbd5e1;
        }

        .wfs-lc-player {
            min-width:0;
            overflow:hidden;
            display:flex;
            align-items:center;
            gap:6px;
            white-space:nowrap;
            font-weight:800;
            color:#f8fafc;
        }

        .wfs-lc-player-name {
            min-width:0;
            overflow:hidden;
            text-overflow:ellipsis;
            white-space:nowrap;
        }

        .wfs-lc-team-logo {
            width:22px;
            height:22px;
            object-fit:contain;
            flex:0 0 22px;
        }

        .wfs-lc-team,
        .wfs-lc-opp {
            font-weight:700;
            color:#cbd5e1;
        }

        .wfs-lc-sal,
        .wfs-lc-fpts {
            text-align:right;
            font-variant-numeric:tabular-nums;
            white-space:nowrap;
        }

        .wfs-lc-fpts {
            font-weight:900;
            color:#f8fafc;
        }

        @media (max-width:640px) {
            .wfs-lineup-grid {
                grid-template-columns:minmax(0,1fr);
                gap:10px;
            }

            .wfs-lc-head {
                gap:8px;
                padding:10px 9px;
            }

            .wfs-lc-columns,
            .wfs-lc-row {
                grid-template-columns:
                    34px minmax(105px,1fr) 34px 34px 48px 43px;
                gap:3px;
                padding-left:7px;
                padding-right:7px;
            }

            .wfs-lc-columns {
                font-size:.54rem;
            }

            .wfs-lc-row {
                font-size:.68rem;
            }

            .wfs-lc-player {
                gap:4px;
            }

            .wfs-lc-team-logo {
                width:20px;
                height:20px;
                flex-basis:20px;
            }

            .wfs-lc-proj,
            .wfs-lc-salary {
                font-size:.78rem;
            }
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        card_html = (
            '<div class="wfs-lineup-grid">'
            + "".join(cards)
            + "</div>"
        )
        card_html = " ".join(card_html.split())

        st.markdown(
            card_html,
            unsafe_allow_html=True,
        )


    def _wfs_showdown_display(long_df: pd.DataFrame) -> pd.DataFrame:
        """Convert solver long-form output to one human-facing row per lineup."""
        if long_df is None or long_df.empty:
            return pd.DataFrame()
        required = {"lineup", "role", "player", "lineup_salary", "lineup_projection"}
        missing = sorted(required - set(long_df.columns))
        if missing:
            raise RuntimeError("Single-Game solver output is missing required fields: " + ", ".join(missing))
        rows = []
        for lineup_id, group in long_df.groupby("lineup", sort=True):
            mvp = group.loc[group["role"].astype(str).eq("MVP")]
            flex = group.loc[group["role"].astype(str).eq("FLEX")]
            if len(mvp) != 1 or len(flex) != 5 or len(group) != 6:
                raise RuntimeError(f"Single-Game lineup {lineup_id} failed roster-shape validation.")
            players = group["player_id"].astype(str).tolist() if "player_id" in group.columns else group["player"].astype(str).tolist()
            if len(set(players)) != 6:
                raise RuntimeError(f"Single-Game lineup {lineup_id} contains a duplicate player.")
            salary = int(round(float(group["lineup_salary"].iloc[0])))
            projection = float(group["lineup_projection"].iloc[0])
            if salary > 60000:
                raise RuntimeError(f"Single-Game lineup {lineup_id} exceeds the $60,000 salary cap.")
            row = {"Lineup": int(lineup_id), "MVP": str(mvp["player"].iloc[0]), "Salary": salary, "Projection": projection}
            for n, player in enumerate(flex["player"].astype(str).tolist(), start=1):
                row[f"AnyFLEX {n}"] = player
            rows.append(row)
        cols = ["Lineup", "MVP", "AnyFLEX 1", "AnyFLEX 2", "AnyFLEX 3", "AnyFLEX 4", "AnyFLEX 5", "Salary", "Projection"]
        return pd.DataFrame(rows).reindex(columns=cols).sort_values("Lineup", kind="mergesort").reset_index(drop=True)


    def render_showdown_builder(selected_slate: str, lineups: int, workspace_mode: str = "Public") -> None:
        """Render the isolated Single-Game builder without entering the Classic generator."""
        st.markdown('<div class="qt-section">1. Build New Lineups</div>', unsafe_allow_html=True)
        st.caption("Build FanDuel NFL Single-Game lineups from the selected matchup.")

        if workspace_mode == "Public":
            _wfs_render_public_data_freshness(
                SHOWDOWN_POOL,
                product_label="Single-Game player data",
                authority_mode="single_game",
            )

        try:
            showdown_pool = load_showdown_projection_pool(showdown_pool_freshness_token())
        except Exception as exc:
            if workspace_mode == "Admin":
                st.error(f"Single-Game is unavailable: {exc}")
            else:
                st.error("Single-Game contests are temporarily unavailable.")
            return
        slate_df = showdown_pool.loc[showdown_pool["public_slate_name"].astype(str).eq(str(selected_slate))].copy()
        if slate_df.empty:
            st.error("Single-Game is unavailable for the selected matchup.")
            return

        try:
            slate_df = apply_showdown_injury_gate(
                slate_df
            )
        except Exception as exc:
            if workspace_mode == "Admin":
                st.error(
                    "Single-Game availability validation failed: "
                    f"{exc}"
                )
            else:
                st.error(
                    "Single-Game contests are temporarily unavailable."
                )
            return

        if slate_df.empty:
            st.error(
                "No eligible players are available for the "
                "selected matchup."
            )
            return

        ready_mask = slate_df["projection_status"].fillna("").astype(str).str.upper().str.startswith("READY")
        if not ready_mask.all():
            st.error("Single-Game player data is not fully ready for this matchup.")
            return

        st.markdown('<div class="qt-section">2. Contest Intelligence</div>', unsafe_allow_html=True)
        team_count = int(slate_df["team"].astype(str).nunique())
        top_projection = float(pd.to_numeric(slate_df["projection"], errors="coerce").max())
        salary_min = int(pd.to_numeric(slate_df["salary"], errors="coerce").min())
        salary_max = int(pd.to_numeric(slate_df["salary"], errors="coerce").max())
        st.caption(f"{selected_slate} Single-Game")
        st.markdown(
            f"""
        <div class="qt-card" style="padding:.9rem 1rem;margin:.55rem 0 1rem 0;">
          <div style="display:flex;gap:.55rem;flex-wrap:wrap;align-items:center;font-size:.88rem;font-weight:800;color:#0f172a;">
            <span>{len(slate_df)} Players</span><span>•</span><span>{team_count} Teams</span><span>•</span>
            <span>Top projection: {top_projection:.2f} pts</span><span>•</span><span>${salary_min:,}–${salary_max:,}</span>
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if workspace_mode == "Public":
            with st.expander("🏈 FanDuel Single-Game Rules", expanded=False):
                st.markdown(
                    """
                **FanDuel NFL Single-Game lineup**

                - **6 players:** 1 MVP + 5 AnyFLEX
                - **Salary cap:** $60,000 maximum
                - **MVP:** 1.5× fantasy points and 1.5× salary
                - **AnyFLEX:** Any eligible player from the selected Single-Game player pool
                - **One underlying player:** Cannot occupy both MVP and AnyFLEX
                """
                )
        else:
            with st.expander("Admin Single-Game diagnostics", expanded=False):
                d1, d2, d3 = st.columns(3)
                d1.metric("Pool rows", len(slate_df))
                d2.metric("Solver SHA", _wfs_hash_file(SHOWDOWN_SOLVER)[:12])
                d3.metric("Pool SHA", _wfs_hash_file(SHOWDOWN_POOL)[:12])
                source_counts = slate_df["projection_source"].fillna("UNKNOWN").astype(str).value_counts().rename_axis("Projection Source").reset_index(name="Rows")
                st.dataframe(source_counts, width="stretch", hide_index=True)

        st.markdown(
            '<div class="qt-section">3. Player Controls</div>',
            unsafe_allow_html=True,
        )

        control_df = slate_df.copy()

        control_df["_control_label"] = control_df.apply(
            lambda r: (
                f'{str(r["player"])} — '
                f'{str(r["team"])} '
                f'{str(r["position"])} — '
                f'${int(round(float(r["salary"]))):,} — '
                f'{float(r["projection"]):.2f}'
            ),
            axis=1,
        )

        label_to_player_id = {
            str(r["_control_label"]): str(r["player_id"])
            for _, r in control_df.iterrows()
        }

        player_labels = sorted(
            label_to_player_id.keys(),
            key=str.lower,
        )

        st.caption(
            "Customize this Single-Game build. "
            "Locks must appear in every lineup, excluded players "
            "cannot appear, and MVP Override forces your chosen "
            "player into the MVP spot."
        )

        locked_labels = st.multiselect(
            "🔒 Lock Players",
            player_labels,
            default=[],
            key=(
                "wfs_showdown_locks_"
                + safe_slug(selected_slate)
            ),
            help=(
                "Every locked player must appear in every generated lineup."
            ),
        )

        excluded_labels = st.multiselect(
            "🚫 Exclude Players",
            player_labels,
            default=[],
            key=(
                "wfs_showdown_excludes_"
                + safe_slug(selected_slate)
            ),
            help=(
                "Excluded players cannot appear in any generated lineup."
            ),
        )

        mvp_options = [
            "Automatic — let WFS choose"
        ] + player_labels

        mvp_choice = st.selectbox(
            "⭐ MVP",
            mvp_options,
            index=0,
            key=(
                "wfs_showdown_mvp_"
                + safe_slug(selected_slate)
            ),
            help=(
                "Leave this on Automatic to use the normal optimizer. "
                "Choose a player to force that player into the MVP role."
            ),
        )

        lock_player_ids = [
            label_to_player_id[label]
            for label in locked_labels
        ]

        exclude_player_ids = [
            label_to_player_id[label]
            for label in excluded_labels
        ]

        mvp_player_id = ""

        if mvp_choice != "Automatic — let WFS choose":
            mvp_player_id = label_to_player_id[mvp_choice]

        control_errors = []

        lock_set = set(lock_player_ids)
        exclude_set = set(exclude_player_ids)

        overlap = lock_set & exclude_set

        if overlap:
            overlap_names = sorted(
                control_df.loc[
                    control_df["player_id"]
                    .astype(str)
                    .isin(overlap),
                    "player",
                ]
                .astype(str)
                .tolist(),
                key=str.lower,
            )

            control_errors.append(
                "A player cannot be both locked and excluded: "
                + ", ".join(overlap_names)
            )

        if len(lock_set) > 6:
            control_errors.append(
                "Single-Game lineups contain only 6 players, "
                "so no more than 6 players can be locked."
            )

        if len(control_df) - len(exclude_set) < 6:
            control_errors.append(
                "Too many players are excluded to build a legal "
                "6-player Single-Game lineup."
            )

        if mvp_player_id and mvp_player_id in exclude_set:
            control_errors.append(
                "The selected MVP cannot also be excluded."
            )

        if mvp_player_id:
            mvp_name_rows = control_df.loc[
                control_df["player_id"]
                .astype(str)
                .eq(str(mvp_player_id)),
                "player",
            ]

            if not mvp_name_rows.empty:
                st.caption(
                    "MVP Override: "
                    + str(mvp_name_rows.iloc[0])
                )

        if locked_labels:
            st.caption(
                f"Locked players: {len(locked_labels)}"
            )

        if excluded_labels:
            st.caption(
                f"Excluded players: {len(excluded_labels)}"
            )

        for message in control_errors:
            st.error(message)

        injury_authority_path = (
            APP_DIR
            / "data"
            / "parquet"
            / "injury_consensus_current.parquet"
        )

        showdown_state_payload = {
            "cache_schema": 1,
            "slate": str(selected_slate),
            "lineups": int(lineups),
            "locks": sorted(str(x) for x in lock_set),
            "excludes": sorted(str(x) for x in exclude_set),
            "mvp": str(mvp_player_id or "AUTO"),
            "pool_sha256": _wfs_hash_file(
                SHOWDOWN_POOL
            ),
            "injury_sha256": _wfs_hash_file(
                injury_authority_path
            ),
            "solver_sha256": _wfs_hash_file(
                SHOWDOWN_SOLVER
            ),
        }

        showdown_state_signature = hashlib.sha256(
            json.dumps(
                showdown_state_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        previous_signature = st.session_state.get(
            "showdown_state_signature"
        )

        if (
            previous_signature
            and previous_signature
            != showdown_state_signature
        ):
            st.session_state.pop(
                "showdown_lineups_long_df",
                None,
            )
            st.session_state.pop(
                "showdown_lineups_display_df",
                None,
            )
            st.session_state.pop(
                "showdown_last_slate",
                None,
            )
            st.session_state.pop(
                "showdown_last_lineup_count",
                None,
            )
            st.session_state.pop(
                "showdown_state_signature",
                None,
            )

        st.markdown(
            '<div class="qt-section">4. Launch Portfolio</div>',
            unsafe_allow_html=True,
        )
        generate = st.button(
            "⚡ Generate Single-Game Lineups",
            type="primary",
            width="stretch",
            key=f"wfs_showdown_generate_{safe_slug(selected_slate)}_{int(lineups)}",
        )
        if generate:
            if control_errors:
                st.error(
                    "Fix the Player Controls above before generating lineups."
                )
                return

            if not SHOWDOWN_SOLVER.exists():
                st.error("Single-Game solver is unavailable.")
                return
            cache_dir = APP_DIR / "data" / "cache" / "fanduel_showdown_ui"
            cache_dir.mkdir(parents=True, exist_ok=True)
            user_key = str((WFS_USER or {}).get("sub") or "anonymous")
            token_raw = f"{user_key}|{selected_slate}|{int(lineups)}|{datetime.utcnow().isoformat()}"
            token = hashlib.sha256(token_raw.encode("utf-8")).hexdigest()[:20]
            output_path = cache_dir / f"showdown_{safe_slug(selected_slate)}_{token}.csv"
            cmd = [
                sys.executable,
                str(SHOWDOWN_SOLVER),
                "--slate",
                str(selected_slate),
                "--lineups",
                str(int(lineups)),
                "--output",
                str(output_path),
            ]

            for player_id in sorted(lock_set):
                cmd.extend(
                    ["--lock-player-id", player_id]
                )

            for player_id in sorted(exclude_set):
                cmd.extend(
                    ["--exclude-player-id", player_id]
                )

            if mvp_player_id:
                cmd.extend(
                    ["--mvp-player-id", mvp_player_id]
                )

            try:
                with st.spinner("Generating Single-Game lineups…"):
                    proc = subprocess.run(cmd, cwd=APP_DIR, capture_output=True, text=True, timeout=120)
                if workspace_mode == "Admin":
                    st.session_state["last_showdown_solver_stdout"] = proc.stdout
                    st.session_state["last_showdown_solver_stderr"] = proc.stderr
                    st.session_state["last_showdown_solver_returncode"] = proc.returncode
                if proc.returncode != 0:
                    st.error("Single-Game lineup generation could not be completed.")
                    if workspace_mode == "Admin":
                        with st.expander("Single-Game solver diagnostics", expanded=True):
                            st.code((proc.stdout or "") + "\n" + (proc.stderr or ""))
                    return
                if not output_path.exists():
                    st.error("Single-Game lineup generation did not produce a validated output file.")
                    return
                long_df = pd.read_csv(output_path)
                display_df = _wfs_showdown_display(long_df)
                if len(display_df) != int(lineups):
                    st.error("Single-Game lineup count failed validation.")
                    return
                st.session_state["showdown_lineups_long_df"] = long_df
                st.session_state["showdown_lineups_display_df"] = display_df
                st.session_state["showdown_last_slate"] = str(selected_slate)
                st.session_state["showdown_last_lineup_count"] = int(lineups)
                st.session_state[
                    "showdown_state_signature"
                ] = showdown_state_signature
                st.success(f"Generated {len(display_df)} Single-Game lineup" + ("s." if len(display_df) != 1 else "."))
            except subprocess.TimeoutExpired:
                st.error("Single-Game lineup generation timed out and was stopped.")
            except Exception as exc:
                if workspace_mode == "Admin":
                    st.error(
                        f"Single-Game lineup generation failed closed: {exc}"
                    )
                else:
                    st.error(
                        "Single-Game lineup generation could not be completed. "
                        "Please try again."
                    )
            finally:
                try:
                    if output_path.exists():
                        output_path.unlink()
                except Exception:
                    pass

        display_df = st.session_state.get("showdown_lineups_display_df")
        long_df = st.session_state.get("showdown_lineups_long_df")
        last_slate = st.session_state.get("showdown_last_slate")
        last_lineup_count = st.session_state.get(
            "showdown_last_lineup_count"
        )
        stored_state_signature = st.session_state.get(
            "showdown_state_signature"
        )

        showdown_result_current = (
            isinstance(display_df, pd.DataFrame)
            and not display_df.empty
            and last_slate == str(selected_slate)
            and last_lineup_count == int(lineups)
            and stored_state_signature
            == showdown_state_signature
        )

        if showdown_result_current:
            st.markdown('<div class="qt-section" style="margin-top:1.2rem;">5. Lineup Command Center</div>', unsafe_allow_html=True)
            st.caption("Review your generated Single-Game lineups and download when ready.")
            high_projection = pd.to_numeric(display_df["Projection"], errors="coerce").max()
            max_salary = pd.to_numeric(display_df["Salary"], errors="coerce").max()
            st.markdown(
                f"""
            <div class="qt-card" style="padding:.85rem 1rem;margin:.5rem 0 1rem 0;">
              <div style="display:flex;gap:.65rem;flex-wrap:wrap;align-items:center;font-size:.86rem;font-weight:800;color:#e2e8f0;">
                <span>{len(display_df)} Lineup{"s" if len(display_df) != 1 else ""}</span><span>•</span><span>Roster PASS</span>
                <span>•</span><span>High {high_projection:.2f}</span><span>•</span><span>Max ${max_salary:,.0f}</span>
              </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            lineup_tab, exposure_tab = st.tabs(
                ["🏟 Lineups", "📊 Exposures"]
            )

            with lineup_tab:
                render_public_showdown_lineup_cards(
                    display_df,
                    long_df,
                )

            with exposure_tab:
                if isinstance(long_df, pd.DataFrame) and not long_df.empty:
                    exposure_source = long_df.copy()

                    exposure_source["player"] = (
                        exposure_source["player"]
                        .fillna("")
                        .astype(str)
                    )

                    exposure_source["role"] = (
                        exposure_source["role"]
                        .fillna("")
                        .astype(str)
                        .str.upper()
                    )

                    total_lineups = max(
                        1,
                        int(display_df["Lineup"].nunique()),
                    )

                    overall = (
                        exposure_source
                        .groupby("player", sort=False)["lineup"]
                        .nunique()
                        .rename("Lineups")
                    )

                    mvp = (
                        exposure_source[
                            exposure_source["role"].eq("MVP")
                        ]
                        .groupby("player", sort=False)["lineup"]
                        .nunique()
                        .rename("MVP")
                    )

                    exposure = (
                        pd.concat([overall, mvp], axis=1)
                        .fillna(0)
                        .reset_index()
                        .rename(columns={"player": "Player"})
                    )

                    exposure["Lineups"] = (
                        pd.to_numeric(
                            exposure["Lineups"],
                            errors="coerce",
                        )
                        .fillna(0)
                        .astype(int)
                    )

                    exposure["MVP"] = (
                        pd.to_numeric(
                            exposure["MVP"],
                            errors="coerce",
                        )
                        .fillna(0)
                        .astype(int)
                    )

                    exposure["Exposure"] = (
                        exposure["Lineups"] / total_lineups * 100
                    ).map(lambda x: f"{x:.0f}%")

                    exposure["MVP Exposure"] = (
                        exposure["MVP"] / total_lineups * 100
                    ).map(lambda x: f"{x:.0f}%")

                    exposure = exposure[
                        [
                            "Player",
                            "Lineups",
                            "Exposure",
                            "MVP",
                            "MVP Exposure",
                        ]
                    ].sort_values(
                        ["Lineups", "MVP", "Player"],
                        ascending=[False, False, True],
                        kind="mergesort",
                    )

                    st.dataframe(
                        exposure,
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.info("No exposure data available.")
            st.download_button(
                "Download Single-Game lineup CSV",
                data=display_df.to_csv(index=False).encode("utf-8"),
                file_name=f"nfl_single_game_lineups_{safe_slug(selected_slate)}.csv",
                mime="text/csv",
                key=f"wfs_showdown_download_{safe_slug(selected_slate)}",
            )
            st.caption("Name-based Single-Game CSV for review and auditing.")
            if workspace_mode == "Admin" and isinstance(long_df, pd.DataFrame):
                with st.expander("Admin Single-Game player detail"):
                    st.dataframe(long_df, width="stretch", hide_index=True)

    def output_paths(slate_name: str) -> dict[str, Path]:
        slug = safe_slug(slate_name)
        base = Path(CSV_DIR)
        return {
            "lineups": base / f"fanduel_gpp_ui_portfolio_{slug}.csv",
            "players": base / f"fanduel_gpp_ui_portfolio_players_{slug}.csv",
            "exposure": base / f"fanduel_gpp_ui_exposure_{slug}.csv",
            "audit": base / f"audit_fanduel_gpp_ui_portfolio_{slug}.csv",
        }


    def read_outputs(slate_name: str):
        paths = output_paths(slate_name)
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise RuntimeError(
                "Solver completed but expected output files are missing:\n"
                + "\n".join(missing)
            )
        return (
            pd.read_csv(paths["lineups"]),
            pd.read_csv(paths["players"]),
            pd.read_csv(paths["exposure"]),
            pd.read_csv(paths["audit"]),
        )



    def _wfs_hash_file(path: Path) -> str:
        """Small deterministic fingerprint for code/data-version invalidation."""
        path = Path(path)
        if not path.exists():
            return "MISSING"
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()


    def _wfs_pool_fingerprint(df: pd.DataFrame) -> str:
        """
    Fingerprint the exact solver-ready slate visible to the UI.

    This intentionally includes every solver-table column so any salary,
    projection, objective signal, eligibility, team, game, or identity change
    invalidates the cached production portfolio.
    """
        if df is None or df.empty:
            return hashlib.sha256(b"EMPTY_POOL").hexdigest()

        work = df.copy()
        cols = sorted(str(c) for c in work.columns)
        work = work.reindex(columns=cols)

        if "_ui_key" in work.columns:
            work = work.sort_values("_ui_key", kind="mergesort")
        elif "player_solver" in work.columns:
            sort_cols = [
                c for c in ["player_solver", "team_solver", "solver_position", "salary_solver"]
                if c in work.columns
            ]
            if sort_cols:
                work = work.sort_values(sort_cols, kind="mergesort")

        work = work.reset_index(drop=True)
        payload = work.to_csv(index=False, na_rep="<NA>", lineterminator="\n").encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


    @contextmanager
    def _classic_authority_transaction(selected_slate: str, displayed_pool: pd.DataFrame, *, mode: str):
        """Keep one updater generation from fresh inputs through result acceptance.

    Display-time data is not an action snapshot. Re-read without Streamlit's
    cache and reject stale controls before fingerprinting or launching a solver.
    The parent owns this lock while subprocess.run waits; inner bridge reader
    locks use separate descriptors and cannot release this outer reader lock.
    """
        try:
            lock = (APP_DIR / "nfl_updater.lock").open("rb")
        except OSError as exc:
            _wfs_report_classic_exception(exc, mode, category="AUTHORITY")
            st.stop()

        with lock:
            try:
                deadline = time.monotonic() + 1200.0
                while True:
                    try:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.25)
            except OSError as exc:
                _wfs_report_classic_exception(exc, mode, category="AUTHORITY")
                st.stop()

            try:
                try:
                    with sqlite3.connect(
                        f"{Path(DATABASE_PATH).resolve().as_uri()}?mode=ro", uri=True,
                    ) as conn:
                        publication = pd.read_sql_query(f'SELECT * FROM "{SOLVER_TABLE}"', conn)
                    fresh_pool = slate_pool(current_classic_publication(publication), selected_slate)
                    if fresh_pool.empty or _wfs_pool_fingerprint(fresh_pool) != _wfs_pool_fingerprint(displayed_pool):
                        raise RuntimeError("Classic selection changed since display")
                except Exception as exc:
                    _wfs_report_classic_exception(exc, mode, category="AUTHORITY")
                    st.stop()
                yield fresh_pool
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


    def _wfs_portfolio_cache_key(
        slate_name: str,
        selected_pool: pd.DataFrame,
        salary_floor: int,
        max_overlap: int,
        max_team: int,
        qb_stack: int,
        bring_back: bool,
        projection_loss_pct: float,
        max_player_exposure: float,
        max_qb_exposure: float,
        max_dst_exposure: float,
        lock_keys: list[str],
        exclude_keys: list[str],
    ) -> str:
        """
    Deterministic strategy fingerprint for a solver portfolio.

    Portfolio depth is intentionally NOT part of this fingerprint.  A single
    exact strategy family grows append-only from 5 -> 10 -> 15 -> 20.
    """
        from stage24_solver_attachment import classic_publication_fingerprint

        payload = {
            "cache_schema": 2,
            "classic_projection_authority": classic_publication_fingerprint(),
            "slate": str(slate_name),
            "solver_sha256": _wfs_hash_file(UI_SOLVER),
            "pool_sha256": _wfs_pool_fingerprint(selected_pool),
            "salary_floor": int(salary_floor),
            "max_overlap": int(max_overlap),
            "max_team": int(max_team),
            "qb_stack": int(qb_stack),
            "bring_back": bool(bring_back),
            "projection_loss_pct": float(projection_loss_pct),
            "max_player_exposure": float(max_player_exposure),
            "max_qb_exposure": float(max_qb_exposure),
            "max_dst_exposure": float(max_dst_exposure),
            "lock_keys": sorted(str(x) for x in lock_keys),
            "exclude_keys": sorted(str(x) for x in exclude_keys),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


    def _wfs_portfolio_cache_paths(cache_key: str) -> dict[str, Path]:
        base = PORTFOLIO_CACHE_DIR / cache_key
        return {
            "base": base,
            "lineups": base / "lineups.csv",
            "players": base / "players.csv",
            "exposure": base / "exposure.csv",
            "audit": base / "audit.csv",
            "meta": base / "meta.json",
        }


    def _wfs_read_cache_meta(cache_key: str) -> dict:
        paths = _wfs_portfolio_cache_paths(cache_key)
        if not paths["meta"].exists():
            return {}
        try:
            meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
            return meta if isinstance(meta, dict) else {}
        except Exception:
            return {}


    def _wfs_read_cached_portfolio(
        cache_key: str,
        expected_size: int | None = None,
    ):
        paths = _wfs_portfolio_cache_paths(cache_key)
        required = ["lineups", "players", "exposure", "audit", "meta"]
        if any(not paths[name].exists() for name in required):
            return None

        try:
            meta = _wfs_read_cache_meta(cache_key)
            actual_size = int(meta.get("portfolio_size", 0))
            if actual_size < 1:
                return None
            if expected_size is not None and actual_size != int(expected_size):
                return None

            lineups_df = pd.read_csv(paths["lineups"])
            players_df = pd.read_csv(paths["players"])
            exposure_df = pd.read_csv(paths["exposure"])
            audit_df = pd.read_csv(paths["audit"])

            if len(lineups_df) != actual_size:
                return None
            if len(players_df) != actual_size * len(ROSTER_SLOTS):
                return None
            # Fail closed: a cache is valid only when solver audit authority is
            # present, non-empty, and every recorded audit is PASS.
            if "audit" not in audit_df.columns or audit_df.empty:
                return None
            if not audit_df["audit"].astype(str).eq("PASS").all():
                return None

            # Canonical nine-player sets must be unique across the shared portfolio.
            signatures = lineups_df.apply(_wfs_lineup_signature, axis=1)
            if signatures.duplicated().any():
                return None

            return lineups_df, players_df, exposure_df, audit_df
        except Exception:
            return None


    def _wfs_write_cached_portfolio(
        cache_key: str,
        slate_name: str,
        lineups_df: pd.DataFrame,
        players_df: pd.DataFrame,
        exposure_df: pd.DataFrame,
        audit_df: pd.DataFrame,
        expected_size: int = PORTFOLIO_CACHE_SIZE,
        solver_profile: dict | None = None,
    ) -> None:
        expected_size = int(expected_size)
        if expected_size < 1:
            raise RuntimeError("Cache write blocked: invalid expected portfolio size.")

        if len(lineups_df) != expected_size:
            raise RuntimeError(
                f"Cache write blocked: expected {expected_size} "
                f"lineups, received {len(lineups_df)}."
            )
        if len(players_df) != expected_size * len(ROSTER_SLOTS):
            raise RuntimeError(
                "Cache write blocked: players.csv does not contain exactly nine "
                "rows per lineup."
            )

        signatures = lineups_df.apply(_wfs_lineup_signature, axis=1)
        if signatures.duplicated().any():
            raise RuntimeError(
                "Cache write blocked: duplicate canonical nine-player lineup detected."
            )

        # Fail closed on missing audit authority as well as explicit failures.
        if "audit" not in audit_df.columns or audit_df.empty:
            raise RuntimeError("Cache write blocked: solver audit authority is missing.")
        if not audit_df["audit"].astype(str).eq("PASS").all():
            raise RuntimeError("Cache write blocked: solver audit did not fully PASS.")

        paths = _wfs_portfolio_cache_paths(cache_key)
        paths["base"].mkdir(parents=True, exist_ok=True)

        # Meta is the commit marker. Readers reject mixed generations because the
        # lineups/player counts must agree with the committed portfolio_size.
        frames = {
            "lineups": lineups_df,
            "players": players_df,
            "exposure": exposure_df,
            "audit": audit_df,
        }
        tmp_paths = {}
        for name, frame in frames.items():
            tmp = paths["base"] / f".{name}.csv.tmp"
            frame.to_csv(tmp, index=False)
            tmp_paths[name] = tmp

        meta = {
            "cache_schema": 2,
            "cache_key": cache_key,
            "slate": str(slate_name),
            "portfolio_size": expected_size,
            "solver_sha256": _wfs_hash_file(UI_SOLVER),
            "solver_profile": solver_profile or {},
        }
        meta_tmp = paths["base"] / ".meta.json.tmp"
        meta_tmp.write_text(
            json.dumps(meta, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        for name, tmp in tmp_paths.items():
            tmp.replace(paths[name])
        meta_tmp.replace(paths["meta"])


    def _wfs_merge_portfolio_extension(
        existing,
        extension,
        target_size: int,
    ):
        old_lineups, old_players, _old_exposure, old_audit = existing
        new_lineups, new_players, new_exposure, new_audit = extension

        old_size = len(old_lineups)
        expected_new = int(target_size) - old_size
        if expected_new < 1 or len(new_lineups) != expected_new:
            raise RuntimeError(
                f"Portfolio extension expected {expected_new} new lineups, "
                f"received {len(new_lineups)}."
            )

        expected_numbers = list(range(old_size + 1, int(target_size) + 1))
        actual_numbers = (
            pd.to_numeric(new_lineups["lineup"], errors="raise").astype(int).tolist()
        )
        if actual_numbers != expected_numbers:
            raise RuntimeError(
                "Portfolio extension returned unexpected lineup numbering."
            )

        lineups_df = pd.concat([old_lineups, new_lineups], ignore_index=True)
        players_df = pd.concat([old_players, new_players], ignore_index=True)
        audit_df = pd.concat([old_audit, new_audit], ignore_index=True)

        if len(lineups_df) != int(target_size):
            raise RuntimeError("Portfolio extension merge produced wrong total depth.")

        signatures = lineups_df.apply(_wfs_lineup_signature, axis=1)
        if signatures.duplicated().any():
            raise RuntimeError(
                "Portfolio extension produced a duplicate canonical nine-player set."
            )

        return lineups_df, players_df, new_exposure, audit_df

    def _wfs_lineup_signature(row) -> str:
        """
    Exact player-set signature. Slot ordering does not make the same nine-player
    lineup "new".
    """
        players = []
        for slot in ROSTER_SLOTS:
            if slot in row.index:
                players.append(normalize_name(row.get(slot)))
        raw = "|".join(sorted(x for x in players if x))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


    def _wfs_take_next_unseen(
        portfolio_df: pd.DataFrame,
        requested: int,
        seen_signatures,
    ):
        """
    Serve the strongest unseen lineups in production order.

    Seen history is permanent for the authenticated user and exact cache key.
    Exhaustion fails closed. A previously served canonical nine-player set
    is never returned as a new lineup.
    """
        if portfolio_df is None or portfolio_df.empty:
            raise RuntimeError(
                "No unique lineup inventory is available."
            )

        requested = max(
            1,
            min(int(requested), len(portfolio_df)),
        )
        seen = {
            str(value).strip()
            for value in (seen_signatures or [])
            if str(value).strip()
        }

        rows = []
        selected_sigs = []

        for idx, row in portfolio_df.iterrows():
            sig = _wfs_lineup_signature(row)

            if sig in seen:
                continue

            rows.append(idx)
            selected_sigs.append(sig)

            if len(rows) >= requested:
                break

        if len(rows) != requested:
            raise RuntimeError(
                "No additional unique lineup is available for "
                "these slate settings and player controls."
            )

        seen.update(selected_sigs)

        out = (
            portfolio_df
            .loc[rows]
            .copy()
            .reset_index(drop=True)
        )

        if "lineup" in out.columns:
            out["portfolio_lineup"] = out["lineup"]
            out["lineup"] = range(
                1,
                len(out) + 1,
            )

        return out, sorted(seen), False


    def _wfs_subset_detail_frame(df: pd.DataFrame, portfolio_lineup_ids) -> pd.DataFrame:
        if df is None:
            return pd.DataFrame()
        if "lineup" not in df.columns:
            return df.copy()
        ids = {str(x) for x in portfolio_lineup_ids}
        mask = df["lineup"].astype(str).isin(ids)
        return df.loc[mask].copy().reset_index(drop=True)


    def build_command(
        slate_name: str,
        lineups: int,
        salary_floor: int,
        max_overlap: int,
        max_team: int,
        qb_stack: int,
        bring_back: bool,
        projection_loss_pct: float,
        max_player_exposure: float,
        max_qb_exposure: float,
        max_dst_exposure: float,
        lock_keys: list[str],
        exclude_keys: list[str],
        resume_players: str = "",
    ) -> list[str]:
        cmd = [
            sys.executable,
            str(UI_SOLVER),
            "--slate", slate_name,
            "--lineups", str(lineups),
            "--salary-floor", str(salary_floor),
            "--max-overlap", str(max_overlap),
            "--max-team", str(max_team),
            "--qb-stack", str(qb_stack),
            "--bring-back", "1" if bring_back else "0",
            "--projection-loss-pct", str(projection_loss_pct),
            "--max-player-exposure", str(max_player_exposure),
            "--max-qb-exposure", str(max_qb_exposure),
            "--max-dst-exposure", str(max_dst_exposure),
        ]

        if str(resume_players).strip():
            cmd.extend(["--resume-players", str(resume_players)])

        for key in lock_keys:
            cmd.extend(["--lock-key", key])
        for key in exclude_keys:
            cmd.extend(["--exclude-key", key])

        return cmd


    # QA F7 / WFS_CLASSIC_RESULT_FRESHNESS_V1
    #
    # Deterministic, order-independent fingerprint of the Classic generation
    # inputs actually passed to build_command. This is UI/session metadata
    # only: it never touches solver inputs, portfolio cache keys, or output
    # files. It exists solely so a previously displayed portfolio can be
    # compared against the controls currently on screen and flagged as
    # outdated when they no longer match.
    def _wfs_classic_freshness_signature(
        slate_name: str,
        lineups: int,
        salary_floor: int,
        max_overlap: int,
        max_team: int,
        qb_stack: int,
        bring_back: bool,
        projection_loss_pct: float,
        max_player_exposure: float,
        max_qb_exposure: float,
        max_dst_exposure: float,
        lock_keys: list[str],
        exclude_keys: list[str],
    ) -> str:
        payload = {
            "slate": str(slate_name),
            "lineups": int(lineups),
            "salary_floor": int(salary_floor),
            "max_overlap": int(max_overlap),
            "max_team": int(max_team),
            "qb_stack": int(qb_stack),
            "bring_back": bool(bring_back),
            "projection_loss_pct": float(projection_loss_pct),
            "max_player_exposure": float(max_player_exposure),
            "max_qb_exposure": float(max_qb_exposure),
            "max_dst_exposure": float(max_dst_exposure),
            "lock_keys": sorted({str(x) for x in lock_keys}),
            "exclude_keys": sorted({str(x) for x in exclude_keys}),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()



    def _wfs_public_profiles(
        max_overlap: int,
        qb_stack: int,
        bring_back: bool,
        max_player_exposure: float,
        max_qb_exposure: float,
        max_dst_exposure: float,
    ) -> list[dict]:
        preferred = {
            "name": "PREFERRED",
            "max_overlap": int(max_overlap),
            "qb_stack": int(qb_stack),
            "bring_back": bool(bring_back),
            "max_player_exposure": float(max_player_exposure),
            "max_qb_exposure": float(max_qb_exposure),
            "max_dst_exposure": float(max_dst_exposure),
        }
        candidates = [
            preferred,
            {
                **preferred,
                "name": "EXPOSURE_RELIEF",
                "max_player_exposure": 1.0,
                "max_qb_exposure": 1.0,
                "max_dst_exposure": 1.0,
            },
            {
                **preferred,
                "name": "OVERLAP_RELIEF",
                "max_overlap": 8,
                "max_player_exposure": 1.0,
                "max_qb_exposure": 1.0,
                "max_dst_exposure": 1.0,
            },
            {
                **preferred,
                "name": "QB_STACK_RELIEF",
                "max_overlap": 8,
                "qb_stack": 1,
                "max_player_exposure": 1.0,
                "max_qb_exposure": 1.0,
                "max_dst_exposure": 1.0,
            },
            {
                **preferred,
                "name": "FULL_SOFT_RELIEF",
                "max_overlap": 8,
                "qb_stack": 1,
                "bring_back": False,
                "max_player_exposure": 1.0,
                "max_qb_exposure": 1.0,
                "max_dst_exposure": 1.0,
            },
        ]
        out = []
        seen = set()
        for profile in candidates:
            signature = (
                int(profile["max_overlap"]),
                int(profile["qb_stack"]),
                bool(profile["bring_back"]),
                float(profile["max_player_exposure"]),
                float(profile["max_qb_exposure"]),
                float(profile["max_dst_exposure"]),
            )
            if signature not in seen:
                seen.add(signature)
                out.append(profile)
        return out


    def _wfs_run_public_solver_target(
        *,
        mode: str,
        selected_slate: str,
        target_size: int,
        salary_floor: int,
        max_team: int,
        projection_loss_pct: float,
        lock_keys: list[str],
        exclude_keys: list[str],
        profiles: list[dict],
        existing=None,
        resume_players: str = "",
        display_count: int = 1,
    ):
        existing_size = 0 if existing is None else len(existing[0])
        expected_new = int(target_size) - existing_size
        if expected_new < 1:
            raise RuntimeError("Public solver target must exceed existing cache depth.")

        successful_profile = None
        attempt_log = []
        accepted = None

        spinner_text = (
            "Generating your lineup…"
            if int(display_count) == 1
            else "Generating your lineups…"
        )

        with st.spinner(spinner_text):
            for profile in profiles:
                cmd = build_command(
                    slate_name=selected_slate,
                    lineups=int(target_size),
                    salary_floor=int(salary_floor),
                    max_overlap=int(profile["max_overlap"]),
                    max_team=int(max_team),
                    qb_stack=int(profile["qb_stack"]),
                    bring_back=bool(profile["bring_back"]),
                    projection_loss_pct=float(projection_loss_pct),
                    max_player_exposure=float(profile["max_player_exposure"]),
                    max_qb_exposure=float(profile["max_qb_exposure"]),
                    max_dst_exposure=float(profile["max_dst_exposure"]),
                    lock_keys=lock_keys,
                    exclude_keys=exclude_keys,
                    resume_players=resume_players,
                )

                require_current_classic_action(selected_slate, mode=mode)
                proc = subprocess.run(
                    cmd,
                    cwd=APP_DIR,
                    capture_output=True,
                    text=True,
                )
                record = {
                    "profile": str(profile["name"]),
                    "returncode": int(proc.returncode),
                    "stdout": proc.stdout or "",
                    "stderr": proc.stderr or "",
                    "generated": 0,
                    "accepted": False,
                }

                if proc.returncode == 0:
                    try:
                        candidate = read_outputs(selected_slate)
                        candidate_lineups = candidate[0]
                        candidate_audit = candidate[3]
                        record["generated"] = len(candidate_lineups)

                        full_count_ok = len(candidate_lineups) == expected_new
                        audit_ok = (
                            not candidate_audit.empty
                            and "audit" in candidate_audit.columns
                            and candidate_audit["audit"].astype(str).eq("PASS").all()
                        )
                        if not audit_ok:
                            record["validation_failure"] = "AUDIT_FAILED"
                        elif not full_count_ok:
                            record["validation_failure"] = "COUNT_MISMATCH"
                        if full_count_ok and audit_ok:
                            accepted = (
                                candidate
                                if existing is None
                                else _wfs_merge_portfolio_extension(
                                    existing, candidate, int(target_size)
                                )
                            )
                            successful_profile = dict(profile)
                            record["accepted"] = True
                    except Exception as exc:
                        record["stderr"] += (
                            "\nOUTPUT_VALIDATION_ERROR: "
                            f"{type(exc).__name__}: {exc}"
                        )

                attempt_log.append(record)
                if not record["accepted"]:
                    logging.getLogger(__name__).error(
                        "Classic generation attempt failed: %s", record,
                    )
                if accepted is not None:
                    break

        st.session_state["last_solver_attempts"] = attempt_log
        st.session_state["last_solver_stdout"] = attempt_log[-1]["stdout"] if attempt_log else ""
        st.session_state["last_solver_stderr"] = attempt_log[-1]["stderr"] if attempt_log else ""
        st.session_state["last_returncode"] = attempt_log[-1]["returncode"] if attempt_log else -1
        st.session_state["last_slate"] = selected_slate
        st.session_state["last_public_solver_profile"] = (
            successful_profile["name"] if successful_profile else ""
        )
        st.session_state["last_classic_failure_category"] = (
            "" if accepted is not None else _wfs_classic_failure_category(
                attempt_log, bool(lock_keys or exclude_keys),
            )
        )
        return accepted, successful_profile


    def render_public_lineup_cards(
        lineups_df: pd.DataFrame,
        players_df: pd.DataFrame,
    ) -> None:
        """Render compact responsive public lineup cards without changing solver data."""

        import html

        if lineups_df is None or lineups_df.empty:
            st.info("No lineups to display.")
            return

        slot_order = {
            "QB": 0,
            "RB1": 1,
            "RB2": 2,
            "WR1": 3,
            "WR2": 4,
            "WR3": 5,
            "TE": 6,
            "FLEX": 7,
            "DST": 8,
        }

        cards = []

        for _, lineup_row in lineups_df.iterrows():
            display_id = int(lineup_row.get("lineup", len(cards) + 1))

            portfolio_id = lineup_row.get("portfolio_lineup", display_id)
            detail = players_df[
                players_df["lineup"].astype(str) == str(portfolio_id)
            ].copy()

            if detail.empty:
                raise RuntimeError(
                    f"Public lineup card detail missing for portfolio lineup {portfolio_id}."
                )

            if len(detail) != 9:
                raise RuntimeError(
                    f"Public lineup card expected 9 players for portfolio lineup "
                    f"{portfolio_id}; found {len(detail)}."
                )

            detail["_slot_order"] = (
                detail["slot"].astype(str).map(slot_order).fillna(99)
            )
            detail = detail.sort_values("_slot_order")

            salary = pd.to_numeric(
                pd.Series([lineup_row.get("salary")]), errors="coerce"
            ).iloc[0]
            projection = pd.to_numeric(
                pd.Series([lineup_row.get("projection")]), errors="coerce"
            ).iloc[0]

            salary_text = (
                f"${salary:,.0f}" if pd.notna(salary) else "—"
            )
            projection_text = (
                f"{projection:.2f} PTS" if pd.notna(projection) else "—"
            )

            rows = []

            for _, player_row in detail.iterrows():
                raw_slot = str(player_row.get("slot", "")).upper()
                display_slot = raw_slot

                if raw_slot in {"RB1", "RB2"}:
                    display_slot = "RB"
                elif raw_slot in {"WR1", "WR2", "WR3"}:
                    display_slot = "WR"

                player = html.escape(str(player_row.get("player", "—")))
                raw_team = str(player_row.get("team", "") or "").strip()
                canonical_team = _dc_team_code(raw_team)
                team = html.escape(canonical_team or raw_team or "—")
                opponent = html.escape(str(player_row.get("opponent", "—")))

                # WFS_PUBLIC_LINEUP_TEAM_LOGO_V1
                # Presentation only: reuse the NFL Data Center's local team-logo
                # resolver. No solver, lineup identity, cache, or scoring changes.
                team_logo = _dc_team_logo_html(
                    canonical_team,
                    "wfs-lc-team-logo",
                )

                player_salary = pd.to_numeric(
                    pd.Series([player_row.get("salary")]), errors="coerce"
                ).iloc[0]
                player_projection = pd.to_numeric(
                    pd.Series([player_row.get("projection")]), errors="coerce"
                ).iloc[0]

                player_salary_text = (
                    f"${player_salary / 1000:.1f}K"
                    if pd.notna(player_salary)
                    else "—"
                )
                player_projection_text = (
                    f"{player_projection:.2f}"
                    if pd.notna(player_projection)
                    else "—"
                )

                rows.append(
                    f"""
                <div class="wfs-lc-row">
                    <div class="wfs-lc-pos">{html.escape(display_slot)}</div>
                    <div class="wfs-lc-player">
                        {team_logo}
                        <span class="wfs-lc-player-name">{player}</span>
                    </div>
                    <div class="wfs-lc-team">{team}</div>
                    <div class="wfs-lc-opp">{opponent}</div>
                    <div class="wfs-lc-sal">{player_salary_text}</div>
                    <div class="wfs-lc-fpts">{player_projection_text}</div>
                </div>
                """
                )

            cards.append(
                f"""
            <div class="wfs-lineup-card">
                <div class="wfs-lc-head">
                    <div class="wfs-lc-number">#{display_id}</div>
                    <div class="wfs-lc-proj">{projection_text}</div>
                    <div class="wfs-lc-salary">{salary_text}</div>
                </div>

                <div class="wfs-lc-columns">
                    <div>POS</div>
                    <div>PLAYER</div>
                    <div>TEAM</div>
                    <div>OPP</div>
                    <div>SAL</div>
                    <div>FPTS</div>
                </div>

                {''.join(rows)}
            </div>
            """
            )

        st.markdown(
            """
        <style>
        .wfs-lineup-grid {
            display:grid;
            grid-template-columns:repeat(auto-fit,minmax(430px,1fr));
            gap:14px;
            width:100%;
            margin:.35rem 0 1rem 0;
        }

        .wfs-lineup-card {
            min-width:0;
            overflow:hidden;
            border:1px solid rgba(148,163,184,.22);
            border-radius:12px;
            background:rgba(15,23,42,.72);
            box-shadow:0 8px 24px rgba(0,0,0,.14);
        }

        .wfs-lc-head {
            display:grid;
            grid-template-columns:1fr auto auto;
            gap:14px;
            align-items:center;
            padding:11px 13px;
            border-bottom:1px solid rgba(148,163,184,.20);
        }

        .wfs-lc-number {
            font-size:1rem;
            font-weight:900;
            color:#f8fafc;
        }

        .wfs-lc-proj {
            font-size:.88rem;
            font-weight:900;
            color:#e2e8f0;
            white-space:nowrap;
        }

        .wfs-lc-salary {
            font-size:.88rem;
            font-weight:900;
            color:#f8fafc;
            white-space:nowrap;
        }

        .wfs-lc-columns,
        .wfs-lc-row {
            display:grid;
            grid-template-columns:42px minmax(130px,1fr) 45px 45px 55px 52px;
            gap:5px;
            align-items:center;
        }

        .wfs-lc-columns {
            padding:7px 10px;
            font-size:.62rem;
            font-weight:900;
            letter-spacing:.04em;
            color:#94a3b8;
            border-bottom:1px solid rgba(148,163,184,.16);
        }

        .wfs-lc-row {
            padding:8px 10px;
            min-height:35px;
            border-bottom:1px solid rgba(148,163,184,.10);
            font-size:.76rem;
            color:#e2e8f0;
        }

        .wfs-lc-row:last-child {
            border-bottom:0;
        }

        .wfs-lc-pos {
            font-weight:900;
            color:#cbd5e1;
        }

        .wfs-lc-player {
            min-width:0;
            overflow:hidden;
            display:flex;
            align-items:center;
            gap:6px;
            white-space:nowrap;
            font-weight:800;
            color:#f8fafc;
        }

        .wfs-lc-player-name {
            min-width:0;
            overflow:hidden;
            text-overflow:ellipsis;
            white-space:nowrap;
        }

        .wfs-lc-team-logo {
            width:22px;
            height:22px;
            object-fit:contain;
            flex:0 0 22px;
        }

        .wfs-lc-team,
        .wfs-lc-opp {
            font-weight:700;
            color:#cbd5e1;
        }

        .wfs-lc-sal,
        .wfs-lc-fpts {
            text-align:right;
            font-variant-numeric:tabular-nums;
            white-space:nowrap;
        }

        .wfs-lc-fpts {
            font-weight:900;
            color:#f8fafc;
        }

        @media (max-width: 640px) {
            .wfs-lineup-grid {
                grid-template-columns:minmax(0,1fr);
                gap:10px;
            }

            .wfs-lc-head {
                gap:8px;
                padding:10px 9px;
            }

            .wfs-lc-columns,
            .wfs-lc-row {
                grid-template-columns:34px minmax(105px,1fr) 34px 34px 48px 43px;
                gap:3px;
                padding-left:7px;
                padding-right:7px;
            }

            .wfs-lc-columns {
                font-size:.54rem;
            }

            .wfs-lc-row {
                font-size:.68rem;
            }

            .wfs-lc-player {
                gap:4px;
            }

            .wfs-lc-team-logo {
                width:20px;
                height:20px;
                flex-basis:20px;
            }

            .wfs-lc-proj,
            .wfs-lc-salary {
                font-size:.78rem;
            }
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        card_html = '<div class="wfs-lineup-grid">' + "".join(cards) + "</div>"
        card_html = " ".join(card_html.split())

        st.markdown(
            card_html,
            unsafe_allow_html=True,
        )


    def lineup_display(lineups_df: pd.DataFrame) -> pd.DataFrame:
        cols = [
            "lineup",
            *ROSTER_SLOTS,
            "salary",
            "projection",
            "best_feasible_projection",
            "projection_loss_actual_pct",
            "gpp_rank_total",
            "qb",
            "stack_type",
            "bring_back_players",
        ]
        cols = [c for c in cols if c in lineups_df.columns]
        out = lineups_df[cols].copy()
        if "projection_loss_actual_pct" in out.columns:
            out["projection_loss_actual_pct"] = (
                pd.to_numeric(out["projection_loss_actual_pct"], errors="coerce") * 100
            ).round(2)
            out = out.rename(
                columns={"projection_loss_actual_pct": "projection_loss_pct"}
            )
        for c in ["projection", "best_feasible_projection"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").round(3)
        return out


    def generic_lineup_csv(lineups_df: pd.DataFrame) -> bytes:
        """
    Clean lineup export using player names by slot.

    This is intentionally NOT labeled FanDuel-upload-ready because the
    solver-ready schema may or may not expose FanDuel contest player IDs.
    """
        cols = [c for c in ["lineup", *ROSTER_SLOTS] if c in lineups_df.columns]
        return lineups_df[cols].to_csv(index=False).encode("utf-8")



    def parse_fanduel_template(template_bytes: bytes) -> dict:
        """
    Parse an official FanDuel NFL lineup-upload template.

    Supports both known FanDuel layouts:

    1. Legacy entries layout:
       entry_id, contest_id, contest_name, entry_fee,
       QB, RB, RB, WR, WR, WR, TE, FLEX, DEF

    2. Current lineup-upload layout:
       QB, RB, RB, WR, WR, WR, TE, FLEX, DEF

       The current lineup rows appear directly beneath those nine roster
       columns and the embedded FanDuel player table appears farther right /
       farther down in the same CSV.

    Player identity remains exact. No fuzzy matching is introduced.
    """
        decoded = template_bytes.decode("utf-8-sig")
        rows = list(csv.reader(io.StringIO(decoded)))

        if not rows:
            raise ValueError("FanDuel template is empty.")

        # ------------------------------------------------------------
        # Locate FanDuel embedded player table first.
        # ------------------------------------------------------------

        pool_header_row = None
        pool_start_col = None

        for r_idx, row in enumerate(rows):
            for c_idx, value in enumerate(row):
                if str(value).strip() == "Player ID + Player Name":
                    pool_header_row = r_idx
                    pool_start_col = c_idx
                    break

            if pool_header_row is not None:
                break

        if pool_header_row is None or pool_start_col is None:
            raise ValueError(
                "Could not locate FanDuel's embedded player table."
            )

        pool_headers = rows[pool_header_row][pool_start_col:]

        header_index = {
            str(name).strip(): pool_start_col + i
            for i, name in enumerate(pool_headers)
            if str(name).strip()
        }

        required = [
            "Id",
            "Position",
            "Nickname",
            "Team",
        ]

        missing = [
            name
            for name in required
            if name not in header_index
        ]

        if missing:
            raise ValueError(
                "FanDuel player table is missing: "
                + ", ".join(missing)
            )

        def read_cell(row, name):
            idx = header_index.get(name)

            if idx is None or idx >= len(row):
                return ""

            return str(row[idx]).strip()

        # ------------------------------------------------------------
        # Parse exact FanDuel player pool.
        # ------------------------------------------------------------

        player_rows = []

        for row in rows[pool_header_row + 1:]:

            fd_id = read_cell(row, "Id")
            raw_position = read_cell(
                row,
                "Position",
            ).upper()

            nickname = read_cell(
                row,
                "Nickname",
            )

            team = normalize_team(
                read_cell(
                    row,
                    "Team",
                )
            )

            if not fd_id or not raw_position:
                continue

            # FanDuel may express defense as D or DEF.
            # WFS internally uses DST, but intersect_with_fanduel_contest()
            # intentionally expects FanDuel defense as position D.
            if raw_position in {"D", "DEF", "DST"}:
                position = "D"
            else:
                position = raw_position

            salary_raw = read_cell(
                row,
                "Salary",
            )

            try:
                salary = (
                    int(float(salary_raw))
                    if salary_raw
                    else None
                )
            except Exception:
                salary = None

            roster_position = read_cell(
                row,
                "Roster Position",
            )

            player_rows.append(
                {
                    "fd_id": fd_id,
                    "nickname": nickname,
                    "name_key": normalize_fanduel_name(nickname),
                    "position": position,
                    "team": team,
                    "salary": salary,
                    "game": read_cell(row, "Game"),
                    "opponent": normalize_team(
                        read_cell(
                            row,
                            "Opponent",
                        )
                    ),
                    "injury_indicator": read_cell(
                        row,
                        "Injury Indicator",
                    ),
                    "injury_details": read_cell(
                        row,
                        "Injury Details",
                    ),
                    "roster_position": roster_position,
                }
            )

        if not player_rows:
            raise ValueError(
                "FanDuel player mapping table contains no players."
            )

        fd_players_by_id = {
            str(player["fd_id"]): player
            for player in player_rows
        }

        # ------------------------------------------------------------
        # Detect lineup-header format.
        # ------------------------------------------------------------

        roster_header = [
            "QB",
            "RB",
            "RB",
            "WR",
            "WR",
            "WR",
            "TE",
            "FLEX",
            "DEF",
        ]

        roster_slots = [
            "QB",
            "RB1",
            "RB2",
            "WR1",
            "WR2",
            "WR3",
            "TE",
            "FLEX",
            "DST",
        ]

        legacy_header = [
            "entry_id",
            "contest_id",
            "contest_name",
            "entry_fee",
            "QB",
            "RB",
            "RB",
            "WR",
            "WR",
            "WR",
            "TE",
            "FLEX",
            "DEF",
        ]

        entry_header_row = None
        entry_start_col = None
        layout = None

        # Legacy format.
        if (
            len(rows[0]) >= len(legacy_header)
            and rows[0][:len(legacy_header)] == legacy_header
        ):
            entry_header_row = 0
            entry_start_col = 4
            layout = "legacy_entries"

        # Current FanDuel format.
        else:
            for r_idx, row in enumerate(
                rows[:min(len(rows), pool_header_row + 1)]
            ):
                normalized = [
                    str(value).strip().upper()
                    for value in row
                ]

                for c_idx in range(
                    0,
                    max(0, len(normalized) - 8),
                ):
                    candidate = normalized[c_idx:c_idx + 9]

                    if candidate == roster_header:
                        entry_header_row = r_idx
                        entry_start_col = c_idx
                        layout = "lineup_upload"
                        break

                if entry_header_row is not None:
                    break

        if entry_header_row is None or entry_start_col is None:
            raise ValueError(
                "Unexpected FanDuel template header. "
                "Could not locate the NFL "
                "QB/RB/RB/WR/WR/WR/TE/FLEX/DEF lineup columns."
            )

        # ------------------------------------------------------------
        # Parse existing FanDuel lineup rows.
        # ------------------------------------------------------------

        def parse_player_token(value):
            """
        FanDuel accepts either:

            133104-88431:Tyler Shough

        or:

            133104-88431

        Return only the exact FanDuel ID.
        """
            raw = str(value or "").strip()

            if not raw:
                return ""

            if ":" in raw:
                return raw.split(":", 1)[0].strip()

            return raw

        entry_row_indexes = []
        entry_lineups = []

        for r_idx in range(
            entry_header_row + 1,
            pool_header_row,
        ):

            row = rows[r_idx]

            cells = []

            for offset in range(9):
                col = entry_start_col + offset

                value = (
                    str(row[col]).strip()
                    if col < len(row)
                    else ""
                )

                cells.append(value)

            # Ignore FanDuel instruction / spacer rows.
            if not any(cells):
                continue

            # A late-swap current lineup must be complete.
            # Partial rows are not silently interpreted as entries.
            if not all(cells):
                continue

            slot_ids = {}

            for slot, raw_value in zip(
                roster_slots,
                cells,
            ):
                fd_id = parse_player_token(
                    raw_value
                )

                if not fd_id:
                    raise ValueError(
                        f"FanDuel lineup row {r_idx + 1} contains "
                        f"an empty player in slot {slot}."
                    )

                if fd_id not in fd_players_by_id:
                    raise ValueError(
                        f"FanDuel lineup row {r_idx + 1} contains "
                        f"unknown player ID {fd_id!r} in slot {slot}."
                    )

                player = fd_players_by_id[fd_id]

                player_position = str(
                    player["position"]
                ).upper()

                if slot == "QB":
                    allowed = {"QB"}

                elif slot in {"RB1", "RB2"}:
                    allowed = {"RB"}

                elif slot in {"WR1", "WR2", "WR3"}:
                    allowed = {"WR"}

                elif slot == "TE":
                    allowed = {"TE"}

                elif slot == "FLEX":
                    allowed = {"RB", "WR", "TE"}

                elif slot == "DST":
                    allowed = {"D"}

                else:
                    raise ValueError(
                        f"Unknown FanDuel roster slot: {slot}"
                    )

                if player_position not in allowed:
                    raise ValueError(
                        f"FanDuel lineup row {r_idx + 1} has "
                        f"{player['nickname']!r} ({player_position}) "
                        f"in incompatible slot {slot}."
                    )

                slot_ids[slot] = fd_id

            if len(set(slot_ids.values())) != 9:
                raise ValueError(
                    f"FanDuel lineup row {r_idx + 1} contains "
                    "duplicate players."
                )

            entry_row_indexes.append(
                r_idx
            )

            entry_lineups.append(
                {
                    "row_index": r_idx,
                    "slots": slot_ids,
                }
            )

        return {
            "rows": rows,
            "layout": layout,
            "entry_header_row": entry_header_row,
            "entry_start_col": entry_start_col,
            "entry_row_indexes": entry_row_indexes,
            "entry_lineups": entry_lineups,
            "player_rows": player_rows,
            "player_rows_by_id": fd_players_by_id,
            "pool_header_row": pool_header_row,
            "pool_start_col": pool_start_col,
        }



    def resolve_solver_player(player_name: str, selected_pool: pd.DataFrame) -> dict:
        """Resolve a generated name to exactly one solver-ready row."""
        key = normalize_name(player_name)
        matches = selected_pool.loc[
            selected_pool["player_solver"].map(normalize_name).eq(key)
        ].copy()

        if len(matches) != 1:
            raise ValueError(
                f'"{player_name}" resolved to {len(matches)} solver-ready rows.'
            )

        row = matches.iloc[0]
        return {
            "player": str(row["player_solver"]),
            "team": normalize_team(row["team_solver"]),
            "position": str(row["solver_position"]).strip().upper(),
        }


    def resolve_fanduel_id(
        solver_player: dict,
        roster_slot: str,
        fd_players: list[dict],
    ) -> str:
        """
    Exact deterministic WFS -> FanDuel player-ID mapping.

    Offense:
        canonical name + normalized team + compatible position

    Defense:
        normalized team + FanDuel defense position

    No fuzzy matching.
    """
        team = normalize_team(
            solver_player["team"]
        )

        name_key = normalize_fanduel_name(
            solver_player["player"]
        )

        if roster_slot == "DEF":
            candidates = [
                p
                for p in fd_players
                if p["position"] == "D"
                and p["team"] == team
            ]

        else:
            allowed = {
                "QB": {"QB"},
                "RB": {"RB"},
                "WR": {"WR"},
                "TE": {"TE"},
                "FLEX": {"RB", "WR", "TE"},
            }[roster_slot]

            candidates = [
                p
                for p in fd_players
                if p["team"] == team
                and p["position"] in allowed
                and p["name_key"] == name_key
            ]

        if len(candidates) != 1:
            raise ValueError(
                f'{solver_player["player"]} ({team}) '
                f'in {roster_slot}: '
                f'{len(candidates)} exact FanDuel matches.'
            )

        return candidates[0]["fd_id"]




    def build_fanduel_upload(
        template_bytes: bytes,
        lineups_df: pd.DataFrame,
        selected_pool: pd.DataFrame,
    ) -> tuple[bytes, dict]:
        """
    Build a FanDuel-ready NFL upload using the uploaded template.

    Supports both:
        - legacy FanDuel entries templates
        - current FanDuel lineup-upload templates

    Only the nine roster cells of existing FanDuel lineup rows are
    changed. All other template content is preserved.

    Exact FanDuel player-ID mapping is mandatory.
    No fuzzy matching.
    """
        parsed = parse_fanduel_template(
            template_bytes
        )

        rows = [
            list(row)
            for row in parsed["rows"]
        ]

        entry_indexes = list(
            parsed["entry_row_indexes"]
        )

        fd_players = parsed["player_rows"]

        entry_start_col = int(
            parsed["entry_start_col"]
        )

        slot_map = [
            ("QB", "QB"),
            ("RB1", "RB"),
            ("RB2", "RB"),
            ("WR1", "WR"),
            ("WR2", "WR"),
            ("WR3", "WR"),
            ("TE", "TE"),
            ("FLEX", "FLEX"),
            ("DST", "DEF"),
        ]

        usable = min(
            len(entry_indexes),
            len(lineups_df),
        )

        if usable <= 0:
            raise ValueError(
                "No existing FanDuel lineup rows can be updated."
            )

        mapped_lineups = []
        errors = []

        for offset in range(usable):
            lineup = lineups_df.iloc[offset]
            mapped = []

            for lineup_col, fd_slot in slot_map:
                player_name = str(
                    lineup[lineup_col]
                ).strip()

                try:
                    solver_player = resolve_solver_player(
                        player_name,
                        selected_pool,
                    )

                    fd_id = resolve_fanduel_id(
                        solver_player,
                        fd_slot,
                        fd_players,
                    )

                    mapped.append(fd_id)

                except Exception as exc:
                    errors.append(
                        f'Lineup '
                        f'{int(lineup.get("lineup", offset + 1))} '
                        f'{fd_slot}: {player_name} — {exc}'
                    )

            mapped_lineups.append(mapped)

        if errors:
            raise ValueError(
                "FanDuel export blocked: exact player mapping "
                "did not reach 100% coverage.\n"
                + "\n".join(errors)
            )

        for offset, row_idx in enumerate(
            entry_indexes[:usable]
        ):
            required_len = (
                entry_start_col + 9
            )

            while len(rows[row_idx]) < required_len:
                rows[row_idx].append("")

            rows[row_idx][
                entry_start_col:
                entry_start_col + 9
            ] = mapped_lineups[offset]

        output = io.StringIO(
            newline=""
        )

        csv.writer(
            output,
            lineterminator="\n",
        ).writerows(rows)

        result = output.getvalue().encode(
            "utf-8"
        )

        return result, {
            "layout": parsed["layout"],
            "template_entries": len(entry_indexes),
            "generated_lineups": len(lineups_df),
            "entries_written": usable,
            "mapping_count": usable * 9,
            "entry_start_col": entry_start_col,
        }




    def validate_fanduel_upload(
        upload_bytes: bytes,
        entries_written: int,
    ) -> None:
        """
    Final structural hard gate before exposing a FanDuel upload.

    Validation is performed through the same parser used for import,
    so both legacy and current FanDuel NFL template layouts are
    supported.

    Every written lineup must contain:
        QB/RB/RB/WR/WR/WR/TE/FLEX/DEF

    with nine valid, unique FanDuel player IDs.
    """
        parsed = parse_fanduel_template(
            upload_bytes
        )

        entry_lineups = parsed[
            "entry_lineups"
        ]

        if len(entry_lineups) < entries_written:
            raise ValueError(
                "Generated FanDuel file contains fewer "
                "validated lineup rows than expected."
            )

        checked = 0

        for entry in entry_lineups[
            :entries_written
        ]:
            slots = entry["slots"]

            expected_slots = {
                "QB",
                "RB1",
                "RB2",
                "WR1",
                "WR2",
                "WR3",
                "TE",
                "FLEX",
                "DST",
            }

            if set(slots) != expected_slots:
                raise ValueError(
                    "Generated FanDuel lineup failed "
                    "slot-structure validation."
                )

            ids = [
                str(slots[slot]).strip()
                for slot in (
                    "QB",
                    "RB1",
                    "RB2",
                    "WR1",
                    "WR2",
                    "WR3",
                    "TE",
                    "FLEX",
                    "DST",
                )
            ]

            if any(not fd_id for fd_id in ids):
                raise ValueError(
                    "Generated FanDuel lineup contains "
                    "an empty player ID."
                )

            if len(set(ids)) != 9:
                raise ValueError(
                    "Generated FanDuel lineup contains "
                    "duplicate players."
                )

            checked += 1

        if checked != entries_written:
            raise ValueError(
                "Generated FanDuel file did not validate "
                "the expected number of lineup rows."
            )


    def render_sidebar(*, pool, mode):
        contest_format = st.radio(
            "Contest Format",
            ["Classic", "Single-Game"],
            horizontal=True,
            key="wfs_contest_format",
        )

        if contest_format == "Classic":
            try:
                pool = current_classic_publication(pool)
                slates = available_slates(pool)
            except Exception:
                slates = []
            if not slates:
                st.info("Current slate data is not available yet." if mode == "Public" else "Current Classic contests are unavailable or staged pending publication. Please check again later.")
                st.stop()
            default_main = next(
                (i for i, s in enumerate(slates) if normalize_slate(s) == "main"),
                0,
            )
            selected_slate = st.selectbox(
                "Slate",
                slates,
                index=default_main,
            )

            if mode == "Public":
                lineups = st.selectbox(
                    "Lineups",
                    [1, 5],
                    index=0,
                    help="Choose 1 lineup or a 5-lineup portfolio.",
                )
            else:
                lineups = st.number_input(
                    "Lineups",
                    min_value=1,
                    max_value=150,
                    value=20,
                    step=1,
                    help="Admin mode retains direct production-solver portfolio sizing.",
                )

            if mode == "Public":
                salary_floor = 0
                max_overlap = 7
                qb_stack = 2
                bring_back = True
                max_player_exposure = 0.60
                max_qb_exposure = 0.40
                max_dst_exposure = 0.40
                projection_loss_pct = 0.01
                max_team = 0
            else:
                salary_floor = st.number_input(
                    "Salary floor",
                    min_value=0,
                    max_value=60000,
                    value=0,
                    step=100,
                    help=(
                        "0 disables the floor. WFS may leave salary unused when the "
                        "production objective prefers that lineup."
                    ),
                )

                max_overlap = st.slider(
                    "Maximum lineup overlap",
                    min_value=0,
                    max_value=8,
                    value=7,
                    step=1,
                )

                qb_stack = st.selectbox(
                    "QB pass-catcher stack",
                    [1, 2],
                    index=1,
                )

                bring_back = st.toggle(
                    "Require opponent bring-back",
                    value=True,
                )

                with st.expander("Exposure limits"):
                    max_player_exposure = st.slider(
                        "RB / WR / TE max exposure", 5, 100, 60, 5,
                    ) / 100.0
                    max_qb_exposure = st.slider(
                        "QB max exposure", 5, 100, 40, 5,
                    ) / 100.0
                    max_dst_exposure = st.slider(
                        "D/ST max exposure", 5, 100, 40, 5,
                    ) / 100.0

                with st.expander("Advanced"):
                    projection_loss_pct = st.slider(
                        "Projection loss allowance",
                        0.0, 5.0, 1.0, 0.25,
                        help="Production default is 1%.",
                    ) / 100.0
                    max_team = st.number_input(
                        "Max players from one team",
                        min_value=0,
                        max_value=9,
                        value=0,
                        step=1,
                        help="0 disables this optional strategy constraint.",
                    )

                st.markdown("---")
                st.markdown(
                    f"""
                    <div class="qt-card">
                      <div style="font-size:.73rem;color:#64748b;text-transform:uppercase;
                                  letter-spacing:.08em;font-weight:800;">Portfolio profile</div>
                      <div style="margin-top:.45rem;color:#e2e8f0;font-size:.86rem;">
                        <b>{int(lineups)}</b> {"lineups" if mode == "Public" else "admin direct-solve lineups"} •
                        QB+{int(qb_stack)} stack • {"Bring-back ON" if bring_back else "Bring-back OFF"}
                      </div>
                      <div style="margin-top:.25rem;color:#94a3b8;font-size:.78rem;">
                        Projection frontier: {projection_loss_pct * 100:.2f}% •
                        Overlap ≤ {int(max_overlap)}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            try:
                showdown_sidebar_pool = load_showdown_projection_pool(showdown_pool_freshness_token())
                showdown_slates = available_showdown_slates(showdown_sidebar_pool)
            except Exception as exc:
                if mode == "Admin":
                    st.error(f"Single-Game is unavailable: {exc}")
                else:
                    st.error("Single-Game contests are temporarily unavailable.")
                st.stop()

            if not showdown_slates:
                st.error("No Single-Game slates are available right now.")
                st.stop()

            selected_slate = st.selectbox(
                "Single-Game",
                showdown_slates,
                key="wfs_showdown_slate",
            )

            if mode == "Public":
                lineups = st.selectbox(
                    "Lineups",
                    [1, 5, 10, 20],
                    index=1,
                    help="Choose how many Single-Game lineups to generate.",
                    key="wfs_showdown_lineups_public",
                )
            else:
                lineups = st.number_input(
                    "Lineups",
                    min_value=1,
                    max_value=20,
                    value=20,
                    step=1,
                    help="Showdown Solver V1 is validated through 20 lineups.",
                    key="wfs_showdown_lineups_admin",
                )

        return {
            "pool": pool,
            "contest_format": contest_format,
            "selected_slate": selected_slate,
            "lineups": lineups,
            **({
                "salary_floor": salary_floor,
                "max_overlap": max_overlap,
                "qb_stack": qb_stack,
                "bring_back": bring_back,
                "max_player_exposure": max_player_exposure,
                "max_qb_exposure": max_qb_exposure,
                "max_dst_exposure": max_dst_exposure,
                "projection_loss_pct": projection_loss_pct,
                "max_team": max_team,
            } if contest_format == "Classic" else {}),
        }


    def render_classic(
        *, pool, mode, selected_slate, lineups, selected_pool, late_swap_selected_pool,
        salary_floor, max_overlap, qb_stack, bring_back, max_player_exposure, max_qb_exposure, max_dst_exposure, projection_loss_pct, max_team,
    ):
        st.markdown('<div class="qt-section">1. Build New Lineups</div>', unsafe_allow_html=True)
        st.caption("Build directly from the selected FanDuel slate. No upload is required.")

        if mode == "Public":
            _wfs_render_public_data_freshness(
                CLASSIC_LINEUP_FRESHNESS_PATH,
                product_label="Player data",
            )

        if mode == "Public":
            open_late_swap = bool(
                st.session_state.pop("wfs_open_late_swap", False)
            )

            with st.expander("Late Swap", expanded=open_late_swap):
                render_late_swap_stage1(
                    late_swap_selected_pool,
                    selected_slate,
                )

        # The normal Lineup Generator uses the backend solver-ready slate directly.
        # FanDuel template import/export is intentionally isolated to Late Swap.
        active_pool = selected_pool.copy()

        label_to_key = dict(zip(active_pool["_label"], active_pool["_ui_key"]))
        player_labels = list(label_to_key.keys())

        st.markdown(
            '<div class="qt-section">2. Contest Intelligence</div>',
            unsafe_allow_html=True,
        )
        games_count = (
            active_pool["game_solver"].nunique()
            if "game_solver" in active_pool.columns and len(active_pool) else "—"
        )
        top_projection = (
            f'{active_pool["projection_solver"].max():.2f}'
            if len(active_pool) else "—"
        )
        salary_range = (
            (
                f'${active_pool["salary_solver"].min():,.0f}–'
                f'${active_pool["salary_solver"].max():,.0f}'
            )
            if len(active_pool) else "—"
        )

        if mode == "Public":
            st.caption(f"{selected_slate} slate")
            st.markdown(
                f"""
        <div class="qt-card" style="padding:.9rem 1rem;margin:.55rem 0 1rem 0;">
          <div style="display:flex;gap:.55rem;flex-wrap:wrap;align-items:center;
                      font-size:.88rem;font-weight:800;color:#0f172a;">
            <span>{len(active_pool)} Players</span><span>•</span>
            <span>{games_count} Games</span><span>•</span>
            <span>Top projection: {top_projection} pts</span><span>•</span>
            <span>{salary_range}</span>
          </div>
        </div>
        """,
                unsafe_allow_html=True,
            )
        else:
            st.caption(
                f"{selected_slate} • backend solver-ready slate"
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Eligible players", len(active_pool))
            m2.metric("Games", games_count)
            m3.metric("Top projection", top_projection)
            m4.metric("Salary range", salary_range)

            st.caption("Backend solver-ready slate is active.")

        # Selected-slate matchup visibility.
        if "game_solver" in active_pool.columns and len(active_pool):
            slate_games = sorted(
                {
                    str(game).strip()
                    for game in active_pool["game_solver"].dropna()
                    if str(game).strip()
                }
            )

            slate_teams = sorted(
                {
                    team.strip()
                    for game in slate_games
                    for team in game.split("@")
                    if team.strip()
                }
            )

            if slate_games:
                matchup_text = "  •  ".join(
                    game.replace("@", " @ ")
                    for game in slate_games
                )

                st.markdown(
                    f"""
            <div class="qt-card" style="padding:1rem 1rem;margin:.55rem 0 1rem 0;">
              <div style="font-size:.73rem;color:#94a3b8;text-transform:uppercase;
                          letter-spacing:.08em;font-weight:800;">
                {selected_slate} Slate Teams &amp; Matchups
              </div>
              <div style="margin-top:.45rem;color:#e2e8f0;font-size:.88rem;
                          font-weight:800;">
                {len(slate_games)} Games • {len(slate_teams)} Teams
              </div>
              <div style="margin-top:.5rem;color:#e2e8f0;font-size:.90rem;
                          line-height:1.65;overflow-wrap:anywhere;">
                {matchup_text}
              </div>
            </div>
            """,
                    unsafe_allow_html=True,
                )

        if mode == "Public":
            with st.expander(
                "🏈 FanDuel Lineup Rules",
                expanded=False,
            ):
                st.markdown(
                    """
            **FanDuel NFL Classic lineup**

            - **9 players:** QB, RB, RB, WR, WR, WR, TE, FLEX, D/ST
            - **Salary cap:** $60,000 maximum
            - **FLEX:** RB, WR, or TE
            - **Team limit:** Maximum 4 players from one NFL team
            - **Scoring:** FanDuel NFL Classic, including 0.5 points per reception
            - **No yardage bonuses:** No 300-yard passing or 100-yard rushing/receiving bonus

            **Player controls:** Locked players must appear in every generated
            lineup. Excluded players will never be selected. WFS may adapt
            portfolio strategy settings when necessary, but it will not ignore
            your locks/excludes or violate FanDuel roster rules.
            """
                )

        st.markdown(
            '<div class="qt-section">3. Player Controls</div>',
            unsafe_allow_html=True,
        )
        if mode == "Public":
            with st.expander("Optional Player Controls", expanded=True):
                st.caption("Quickly find players by name.")
                locked_labels = st.multiselect(
                    "Lock players",
                    player_labels,
                    placeholder="Search & lock players...",
                    help="Tap and type a player name. Search uses fuzzy matching.",
                    filter_mode="fuzzy",
                    select_all=False,
                    key="wfs_classic_locks_" + safe_slug(selected_slate),
                )
                excluded_options = [x for x in player_labels if x not in locked_labels]
                excluded_labels = st.multiselect(
                    "Exclude players",
                    excluded_options,
                    placeholder="Search & exclude players...",
                    help="Tap and type a player name. Search uses fuzzy matching.",
                    filter_mode="fuzzy",
                    select_all=False,
                    key="wfs_classic_excludes_" + safe_slug(selected_slate),
                )
        else:
            st.caption("Optional overrides. Leave blank to let the production engine build freely.")
            left, right = st.columns(2)
            with left:
                locked_labels = st.multiselect(
                    "Lock players", player_labels,
                    help="Locked players are forced into every generated lineup.",
                    select_all=False,
                    key="wfs_classic_admin_locks_" + safe_slug(selected_slate),
                )
            with right:
                excluded_options = [x for x in player_labels if x not in locked_labels]
                excluded_labels = st.multiselect(
                    "Exclude players", excluded_options,
                    help="Excluded players cannot appear in generated lineups.",
                    select_all=False,
                    key="wfs_classic_admin_excludes_" + safe_slug(selected_slate),
                )

        lock_keys = [label_to_key[x] for x in locked_labels]
        user_exclude_keys = [label_to_key[x] for x in excluded_labels]

        # Normal generator exclusions come only from explicit user controls.
        exclude_keys = sorted(set(user_exclude_keys))

        # QA F7 / WFS_CLASSIC_RESULT_FRESHNESS_V1
        #
        # Signature of the generation inputs as they stand on THIS render. Compared
        # against the signature stored at the time of the last successful generation
        # to detect whether a displayed portfolio is stale.
        current_generation_signature = _wfs_classic_freshness_signature(
            slate_name=selected_slate,
            lineups=int(lineups),
            salary_floor=int(salary_floor),
            max_overlap=int(max_overlap),
            max_team=int(max_team),
            qb_stack=int(qb_stack),
            bring_back=bool(bring_back),
            projection_loss_pct=float(projection_loss_pct),
            max_player_exposure=float(max_player_exposure),
            max_qb_exposure=float(max_qb_exposure),
            max_dst_exposure=float(max_dst_exposure),
            lock_keys=lock_keys,
            exclude_keys=exclude_keys,
        )

        # QA F1 / WFS_CLASSIC_LOCK_PREFLIGHT_V1
        #
        # Reject structurally impossible Classic lock combinations before launching
        # the production solver. The production solver remains final authority.
        def _wfs_classic_locks_fit_roster(locked_positions: list[str]) -> bool:
            slot_eligibility = {
                "QB": ("QB",),
                "RB": ("RB1", "RB2", "FLEX"),
                "WR": ("WR1", "WR2", "WR3", "FLEX"),
                "TE": ("TE", "FLEX"),
                "DST": ("DST",),
            }

            positions = [str(pos).strip().upper() for pos in locked_positions]

            if len(positions) > len(ROSTER_SLOTS):
                return False

            if any(pos not in slot_eligibility for pos in positions):
                return False

            positions.sort(
                key=lambda pos: (len(slot_eligibility[pos]), pos)
            )

            def assign(index: int, used_slots: set[str]) -> bool:
                if index >= len(positions):
                    return True

                for slot in slot_eligibility[positions[index]]:
                    if slot in used_slots:
                        continue

                    used_slots.add(slot)

                    if assign(index + 1, used_slots):
                        return True

                    used_slots.remove(slot)

                return False

            return assign(0, set())


        # QA F1 / WFS_CLASSIC_LOCK_PREFLIGHT_V1 (diagnostic)
        #
        # Given a set of locked positions that fails _wfs_classic_locks_fit_roster,
        # identify and describe the specific slot conflict so the user knows what
        # to unlock, instead of a generic "cannot all fit" message.
        def _wfs_classic_lock_conflict_reason(locked_positions: list[str]) -> str:
            dedicated_capacity = {"QB": 1, "DST": 1, "RB": 2, "WR": 3, "TE": 1}
            plural = {"QB": "QBs", "DST": "D/STs", "RB": "RBs", "WR": "WRs", "TE": "TEs"}
            positions = [str(pos).strip().upper() for pos in locked_positions]

            counts = {pos: positions.count(pos) for pos in dedicated_capacity}

            for pos in ("QB", "DST"):
                cap = dedicated_capacity[pos]
                if counts[pos] > cap:
                    label = "QB" if pos == "QB" else "D/ST"
                    return (
                        f"You locked {counts[pos]} {plural[pos]}. "
                        f"A lineup allows {cap} {label}."
                    )

            for pos in ("RB", "WR", "TE"):
                cap_with_flex = dedicated_capacity[pos] + 1
                if counts[pos] > cap_with_flex:
                    return (
                        f"You locked {counts[pos]} {plural[pos]}. "
                        f"A lineup allows {dedicated_capacity[pos]} plus 1 FLEX."
                    )

            flex_eligible_total = counts["RB"] + counts["WR"] + counts["TE"]
            flex_eligible_slots = (
                dedicated_capacity["RB"] + dedicated_capacity["WR"] + dedicated_capacity["TE"] + 1
            )
            if flex_eligible_total > flex_eligible_slots:
                return (
                    f"You locked {flex_eligible_total} RB/WR/TE players, but a "
                    f"lineup only has {flex_eligible_slots} RB/WR/TE slots (FLEX included)."
                )

            return (
                "Those locked players cannot all fit into one legal FanDuel "
                "Classic lineup. Adjust your player locks."
            )


        if len(lock_keys) > 9:
            st.error("A FanDuel lineup contains only nine roster slots.")
            st.stop()

        if lock_keys:
            locked_key_set = set(lock_keys)

            locked_rows = active_pool.loc[
                active_pool["_ui_key"].astype(str).isin(locked_key_set)
            ].copy()

            found_lock_keys = set(
                locked_rows["_ui_key"].astype(str).tolist()
            )

            missing_lock_keys = locked_key_set - found_lock_keys

            if missing_lock_keys:
                st.error(
                    "One or more locked players are no longer available in "
                    "the selected slate."
                )
                st.stop()

            locked_positions = (
                locked_rows["solver_position"]
                .astype(str)
                .str.strip()
                .str.upper()
                .tolist()
            )

            if not _wfs_classic_locks_fit_roster(locked_positions):
                st.error(_wfs_classic_lock_conflict_reason(locked_positions))
                st.stop()

            locked_salary = int(
                locked_rows["salary_solver"].sum()
            )

            if locked_salary > 60000:
                st.error(
                    f"Your locked players cost ${locked_salary:,}. "
                    "The cap is $60,000."
                )
                st.stop()

            effective_team_cap = (
                min(int(max_team), 4)
                if int(max_team) > 0
                else 4
            )

            locked_team_counts = (
                locked_rows["team_solver"]
                .astype(str)
                .value_counts()
            )

            if (
                not locked_team_counts.empty
                and int(locked_team_counts.max()) > effective_team_cap
            ):
                st.error(
                    "Too many locked players are from the same NFL team "
                    f"(maximum {effective_team_cap})."
                )
                st.stop()

        st.markdown(
            '<div class="qt-section">4. Launch Portfolio</div>',
            unsafe_allow_html=True,
        )
        if mode == "Admin":
            st.caption("Backend solver-ready slate loaded. Ready to build.")

        # QA F6 / WFS_CLASSIC_GENERATION_PROGRESS_V1
        #
        # Best-effort in-progress flag. Streamlit runs one script pass per session at
        # a time and generation is already serialized by PORTFOLIO_CACHE_LOCK, so
        # this cannot create overlapping solver runs; it exists to (a) give an
        # accurate "already running" message on any rerun that lands while the flag
        # is set, and (b) let the button render disabled on that rerun. The button
        # rendered on the same click that starts generation cannot itself be
        # disabled retroactively -- that is why the info message below is the
        # primary, reliable form of immediate feedback.
        generate = st.button(
            "⚡ Generate GPP Portfolio",
            type="primary",
            width="stretch",
            disabled=bool(st.session_state.get("wfs_classic_generation_in_progress")),
        )

        if generate and st.session_state.get("wfs_classic_generation_in_progress"):
            st.warning(
                "A lineup generation is already running for this session. "
                "Please wait for it to finish."
            )
        elif generate:
            st.session_state["wfs_classic_generation_in_progress"] = True
            try:
                st.info(
                    "⚡ Generating your GPP portfolio… this typically takes "
                    "25–35 seconds. Please keep this tab open."
                )
                with _wfs_classic_generation_errors(mode, bool(lock_keys or exclude_keys)), _classic_authority_transaction(selected_slate, selected_pool, mode=mode) as selected_pool:
                    require_current_classic_action(selected_slate, mode=mode)
                    if not UI_SOLVER.exists():
                        if mode == "Admin":
                            st.error(f"Missing UI solver: {UI_SOLVER}")
                        else:
                            st.error("Lineup generation is temporarily unavailable.")
                        st.stop()

                    if mode == "Admin":
                        cmd = build_command(
                            slate_name=selected_slate,
                            lineups=int(lineups),
                            salary_floor=int(salary_floor),
                            max_overlap=int(max_overlap),
                            max_team=int(max_team),
                            qb_stack=int(qb_stack),
                            bring_back=bool(bring_back),
                            projection_loss_pct=float(projection_loss_pct),
                            max_player_exposure=float(max_player_exposure),
                            max_qb_exposure=float(max_qb_exposure),
                            max_dst_exposure=float(max_dst_exposure),
                            lock_keys=lock_keys,
                            exclude_keys=exclude_keys,
                        )

                        PORTFOLIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        with PORTFOLIO_CACHE_LOCK.open("a+") as lock_handle:
                            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                            try:
                                with st.spinner("Generating deterministic GPP portfolio…"):
                                    require_current_classic_action(selected_slate, mode=mode)
                                    proc = subprocess.run(
                                        cmd,
                                        cwd=APP_DIR,
                                        capture_output=True,
                                        text=True,
                                    )

                                st.session_state["last_solver_stdout"] = proc.stdout
                                st.session_state["last_solver_stderr"] = proc.stderr
                                st.session_state["last_returncode"] = proc.returncode
                                st.session_state["last_slate"] = selected_slate

                                if proc.returncode != 0:
                                    st.error("Portfolio generation failed.")
                                    with st.expander("Solver error", expanded=True):
                                        st.code((proc.stdout or "") + "\n" + (proc.stderr or ""))
                                    st.stop()

                                (
                                    lineups_df,
                                    players_df,
                                    exposure_df,
                                    audit_df,
                                ) = read_outputs(selected_slate)

                                st.session_state["lineups_df"] = lineups_df
                                st.session_state["players_df"] = players_df
                                st.session_state["exposure_df"] = exposure_df
                                st.session_state["audit_df"] = audit_df
                                st.session_state["portfolio_audit_df"] = audit_df
                                st.session_state["portfolio_cache_status"] = "ADMIN LIVE"
                                st.session_state["portfolio_wrapped"] = False
                                # QA F7 / WFS_CLASSIC_RESULT_FRESHNESS_V1
                                st.session_state["classic_result_signature"] = (
                                    current_generation_signature
                                )

                                st.success(
                                    f"Generated {len(lineups_df)} deterministic admin lineups."
                                )
                            finally:
                                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                    else:
                        display_count = int(lineups)
                        initial_size = int(PORTFOLIO_CACHE_SIZE)

                        cache_key = _wfs_portfolio_cache_key(
                            slate_name=selected_slate,
                            selected_pool=selected_pool,
                            salary_floor=int(salary_floor),
                            max_overlap=int(max_overlap),
                            max_team=int(max_team),
                            qb_stack=int(qb_stack),
                            bring_back=bool(bring_back),
                            projection_loss_pct=float(projection_loss_pct),
                            max_player_exposure=float(max_player_exposure),
                            max_qb_exposure=float(max_qb_exposure),
                            max_dst_exposure=float(max_dst_exposure),
                            lock_keys=lock_keys,
                            exclude_keys=exclude_keys,
                        )

                        seen_state_key = (
                            f"wfs_seen_portfolio::{WFS_USER.get('sub', 'anonymous')}::{cache_key}"
                        )
                        if seen_state_key in st.session_state:
                            seen_before = set(st.session_state[seen_state_key])
                        else:
                            seen_before = set(
                                _wfs_get_seen_portfolio_signatures(
                                    WFS_USER.get("sub"),
                                    cache_key,
                                )
                            )
                            st.session_state[seen_state_key] = seen_before

                        cached = _wfs_read_cached_portfolio(cache_key)
                        cache_status = "HIT"

                        # Determine whether this request needs initial generation or an append-only
                        # +5 extension.  We extend only when the current user has exhausted the
                        # committed shared inventory needed for this request.
                        need_generation = cached is None
                        current_size = 0 if cached is None else len(cached[0])

                        if cached is not None:
                            current_signatures = {
                                _wfs_lineup_signature(row)
                                for _, row in cached[0].iterrows()
                            }
                            unseen_count = len(current_signatures - seen_before)
                            if (
                                unseen_count < display_count
                                and current_size < int(PORTFOLIO_CACHE_MAX_SIZE)
                            ):
                                need_generation = True

                        if need_generation:
                            PORTFOLIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                            with PORTFOLIO_CACHE_LOCK.open("a+") as lock_handle:
                                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                                try:
                                    # Re-read after acquiring the global solver-output lock.
                                    cached = _wfs_read_cached_portfolio(cache_key)
                                    current_size = 0 if cached is None else len(cached[0])

                                    if cached is not None:
                                        current_signatures = {
                                            _wfs_lineup_signature(row)
                                            for _, row in cached[0].iterrows()
                                        }
                                        unseen_count = len(current_signatures - seen_before)
                                    else:
                                        unseen_count = 0

                                    should_build = cached is None or (
                                        unseen_count < display_count
                                        and current_size < int(PORTFOLIO_CACHE_MAX_SIZE)
                                    )

                                    if should_build:
                                        if cached is None:
                                            target_size = initial_size
                                            profiles = _wfs_public_profiles(
                                                max_overlap=int(max_overlap),
                                                qb_stack=int(qb_stack),
                                                bring_back=bool(bring_back),
                                                max_player_exposure=float(max_player_exposure),
                                                max_qb_exposure=float(max_qb_exposure),
                                                max_dst_exposure=float(max_dst_exposure),
                                            )
                                            resume_players = ""
                                            existing = None
                                            cache_status = "BUILT"
                                        else:
                                            target_size = min(
                                                current_size + int(PORTFOLIO_CACHE_INCREMENT),
                                                int(PORTFOLIO_CACHE_MAX_SIZE),
                                            )
                                            meta = _wfs_read_cache_meta(cache_key)
                                            saved_profile = meta.get("solver_profile") or {}
                                            required_profile = {
                                                "name",
                                                "max_overlap",
                                                "qb_stack",
                                                "bring_back",
                                                "max_player_exposure",
                                                "max_qb_exposure",
                                                "max_dst_exposure",
                                            }
                                            if not required_profile.issubset(saved_profile):
                                                raise RuntimeError(
                                                    "This cached portfolio predates resumable profile "
                                                    "metadata and cannot be safely extended."
                                                )
                                            # Extension must use the exact profile that created the
                                            # existing family; adaptive fallback cannot change midway.
                                            profiles = [saved_profile]
                                            resume_players = str(
                                                _wfs_portfolio_cache_paths(cache_key)["players"]
                                            )
                                            existing = cached
                                            cache_status = f"EXTENDED_{current_size}_TO_{target_size}"

                                        generated, successful_profile = _wfs_run_public_solver_target(
                                            selected_slate=selected_slate,
                                            target_size=target_size,
                                            salary_floor=int(salary_floor),
                                            max_team=int(max_team),
                                            projection_loss_pct=float(projection_loss_pct),
                                            lock_keys=lock_keys,
                                            exclude_keys=exclude_keys,
                                            profiles=profiles,
                                            existing=existing,
                                            resume_players=resume_players,
                                            display_count=display_count,
                                            mode=mode,
                                        )

                                        if generated is None:
                                            st.error(
                                                _wfs_classic_failure_message(
                                                    st.session_state.get("last_classic_failure_category", "UNKNOWN")
                                                )
                                            )
                                            st.stop()

                                        _wfs_write_cached_portfolio(
                                            cache_key,
                                            selected_slate,
                                            *generated,
                                            expected_size=target_size,
                                            solver_profile=successful_profile,
                                        )
                                        cached = generated
                                    else:
                                        cache_status = "HIT"
                                finally:
                                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

                        try:
                            (
                                portfolio_lineups_df,
                                portfolio_players_df,
                                portfolio_exposure_df,
                                portfolio_audit_df,
                            ) = cached

                            shown_lineups_df, seen_after, wrapped = _wfs_take_next_unseen(
                                portfolio_lineups_df,
                                display_count,
                                seen_before,
                            )

                            st.session_state[seen_state_key] = seen_after
                            _wfs_set_seen_portfolio_signatures(
                                WFS_USER.get("sub"),
                                cache_key,
                                seen_after,
                            )

                            portfolio_ids = (
                                shown_lineups_df["portfolio_lineup"].tolist()
                                if "portfolio_lineup" in shown_lineups_df.columns
                                else []
                            )
                            shown_players_df = _wfs_subset_detail_frame(
                                portfolio_players_df,
                                portfolio_ids,
                            )
                            shown_audit_df = _wfs_subset_detail_frame(
                                portfolio_audit_df,
                                portfolio_ids,
                            )

                            st.session_state["lineups_df"] = shown_lineups_df
                            st.session_state["players_df"] = shown_players_df
                            st.session_state["exposure_df"] = portfolio_exposure_df
                            st.session_state["audit_df"] = shown_audit_df
                            st.session_state["portfolio_audit_df"] = portfolio_audit_df
                            st.session_state["portfolio_cache_key"] = cache_key
                            st.session_state["portfolio_cache_status"] = cache_status
                            st.session_state["portfolio_wrapped"] = wrapped
                            st.session_state["last_slate"] = selected_slate
                            # QA F7 / WFS_CLASSIC_RESULT_FRESHNESS_V1
                            st.session_state["classic_result_signature"] = (
                                current_generation_signature
                            )

                            st.success(
                                f"Generated {len(shown_lineups_df)} WFS lineup"
                                f'{"s" if len(shown_lineups_df) != 1 else ""}.'
                            )
                        except Exception as exc:
                            if mode == "Admin":
                                st.error(str(exc))
                            else:
                                _wfs_report_classic_exception(exc, mode, category="UNKNOWN")
            finally:
                st.session_state["wfs_classic_generation_in_progress"] = False

        if "lineups_df" in st.session_state:
            lineups_df = st.session_state["lineups_df"]
            players_df = st.session_state["players_df"]
            exposure_df = st.session_state["exposure_df"]
            audit_df = st.session_state["audit_df"]

            st.markdown(
                '<div class="qt-section" style="margin-top:1.2rem;">5. Lineup Command Center</div>',
                unsafe_allow_html=True,
            )

            # QA F7 / WFS_CLASSIC_RESULT_FRESHNESS_V1
            #
            # Compare the signature stored at the time of the last successful
            # generation against the signature of the controls on screen right now.
            # A missing stored signature (e.g. results carried over from before this
            # feature existed) is treated as not-stale rather than guessed at.
            _stored_generation_signature = st.session_state.get("classic_result_signature")
            _classic_result_is_stale = (
                _stored_generation_signature is not None
                and _stored_generation_signature != current_generation_signature
            )
            if _classic_result_is_stale:
                st.warning(
                    "⚠️ Outdated — your Build Lineups settings have changed since this "
                    "portfolio was generated. Click **Generate GPP Portfolio** above to "
                    "refresh these results before reviewing or downloading them."
                )

            cache_status = st.session_state.get("portfolio_cache_status", "LIVE")
            if mode == "Public":
                st.caption("Review your generated lineups and download when ready.")
            else:
                st.caption(
                    "Generation complete. Review quality, exposures, structure, and export. "
                    f"Portfolio source: {cache_status}."
                )

            audit_source = st.session_state.get("portfolio_audit_df", audit_df)
            audit_pass = (
                "audit" in audit_source.columns
                and len(audit_source) > 0
                and audit_source["audit"].astype(str).eq("PASS").all()
            )
            projection_high = pd.to_numeric(lineups_df["projection"], errors="coerce").max()
            max_salary = pd.to_numeric(lineups_df["salary"], errors="coerce").max()

            if mode == "Public":
                st.markdown(
                    f"""
            <div class="qt-card" style="padding:.85rem 1rem;margin:.5rem 0 1rem 0;">
              <div style="display:flex;gap:.65rem;flex-wrap:wrap;align-items:center;
                          font-size:.86rem;font-weight:800;color:#e2e8f0;">
                <span>{len(lineups_df)} Lineup{"s" if len(lineups_df) != 1 else ""}</span>
                <span>•</span><span>Top Projection {projection_high:.2f}</span>
                <span>•</span><span>Highest Salary ${max_salary:,.0f}</span>
              </div>
            </div>
            """,
                    unsafe_allow_html=True,
                )
            else:
                a1, a2, a3, a4, a5 = st.columns(5)
                a1.metric("Generated", len(lineups_df))
                a2.metric("Audit", "PASS" if audit_pass else "FAIL")
                a3.metric("Projection high", f"{projection_high:.2f}")
                a4.metric(
                    "Projection low",
                    f'{pd.to_numeric(lineups_df["projection"], errors="coerce").min():.2f}',
                )
                a5.metric("Max salary", f"${max_salary:,.0f}")

            if mode == "Public":
                tab1, tab2 = st.tabs(
                    ["🏟 Lineups", "📊 Exposures"]
                )
                tab3 = None
                tab4 = None
            else:
                tab1, tab2, tab3, tab4 = st.tabs(
                    ["🏟 Lineups", "📊 Exposures", "🧬 Portfolio Detail", "✅ Audit"]
                )

            with tab1:
                if mode == "Public":
                    render_public_lineup_cards(lineups_df, players_df)
                else:
                    display = lineup_display(lineups_df)
                    st.dataframe(
                        display,
                        width="stretch",
                        hide_index=True,
                    )
                st.download_button(
                    "Download lineup CSV",
                    data=generic_lineup_csv(lineups_df),
                    file_name=f"nfl_gpp_lineups_{safe_slug(selected_slate)}.csv",
                    mime="text/csv",
                )
                st.caption(
                    "Download your generated lineups for review or FanDuel entry preparation."
                )

            with tab2:
                if mode == "Public":
                    st.caption("Portfolio exposure summary.")
                else:
                    st.caption(
                        "Exposure is audited across the full validated 20-lineup production "
                        "portfolio, even when this click displays a smaller unseen subset."
                    )
                exposure_view = exposure_df.copy()
                if "exposure_pct" in exposure_view.columns:
                    exposure_view["exposure_pct"] = pd.to_numeric(
                        exposure_view["exposure_pct"], errors="coerce"
                    ).round(1)
                st.dataframe(
                    exposure_view,
                    width="stretch",
                    hide_index=True,
                )

            if mode == "Admin":
                with tab3:
                    st.dataframe(
                        players_df,
                        width="stretch",
                        hide_index=True,
                    )

                with tab4:
                    st.dataframe(
                        audit_df,
                        width="stretch",
                        hide_index=True,
                    )

        render_wfs_public_disclaimer()

        st.markdown(
            """
    <div class="qt-footer">
      Wynners Fantasy Spot • NFL GPP • data driven • tournament focused
    </div>
    """,
            unsafe_allow_html=True,
        )

        if mode == "Admin":
            st.divider()
            render_admin_diagnostics(pool, active_pool)

            if "last_solver_stdout" in st.session_state:
                with st.expander("Raw solver diagnostics"):
                    st.code(st.session_state.get("last_solver_stdout", ""))
                stderr = st.session_state.get("last_solver_stderr", "")
                if stderr:
                    with st.expander("Solver stderr"):
                        st.code(stderr)
            if st.session_state.get("last_classic_generation_exception"):
                with st.expander("Classic generation exception"):
                    st.code(st.session_state["last_classic_generation_exception"])


    def render_page(
        *, pool, mode, contest_format, selected_slate, lineups,
        selected_pool, late_swap_selected_pool,
        salary_floor=None, max_overlap=None, qb_stack=None, bring_back=None, max_player_exposure=None, max_qb_exposure=None, max_dst_exposure=None, projection_loss_pct=None, max_team=None,
    ):
        if st.button(
            "← Back to Home",
            key="wfs_back_home_build",
            width="stretch",
        ):
            _wfs_home_navigate("🏠 Home")

        if contest_format == "Single-Game":
            render_showdown_builder(
                selected_slate=selected_slate,
                lineups=int(lineups),
                workspace_mode=mode,
            )
            render_wfs_public_disclaimer()

            st.markdown(
                '<div class="qt-footer">Wynners Fantasy Spot • NFL Single-Game • tournament focused</div>',
                unsafe_allow_html=True,
            )
            st.stop()

        render_classic(
            pool=pool, mode=mode, selected_slate=selected_slate, lineups=lineups,
            selected_pool=selected_pool, late_swap_selected_pool=late_swap_selected_pool,
            salary_floor=salary_floor,
            max_overlap=max_overlap,
            qb_stack=qb_stack,
            bring_back=bring_back,
            max_player_exposure=max_player_exposure,
            max_qb_exposure=max_qb_exposure,
            max_dst_exposure=max_dst_exposure,
            projection_loss_pct=projection_loss_pct,
            max_team=max_team,
        )

    return BuildLineups(
        render_sidebar=render_sidebar,
        render_page=render_page,
        available_slates=available_slates,
        slate_pool=slate_pool,
        late_swap_slate_pool=late_swap_slate_pool,
    )
