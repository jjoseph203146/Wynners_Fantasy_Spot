import pandas as pd
import streamlit as st


def render_admin(*, accounts_connect, health_state_path) -> None:
    st.title("🔒 WFS Admin")
    st.caption("Private administration, account activity, and user feedback.")

    users_tab, feedback_tab, health_tab = st.tabs(
        ["👥 Users", "💬 User Feedback", "🩺 Health Monitor"]
    )

    with health_tab:
        import json as _hm_json
        from pathlib import Path as _HMPath

        st.markdown("### 🩺 NFL APP Health Monitor")
        st.caption(
            "Read-only operational status. "
            "This panel cannot execute repairs or modify monitor state."
        )

        _hm_path = health_state_path

        if not _hm_path.exists():
            st.warning("Health Monitor state has not been published yet.")
        else:
            try:
                _hm_state = _hm_json.loads(_hm_path.read_text())
            except Exception:
                _hm_state = None
                st.error("Health Monitor state could not be read.")

            if isinstance(_hm_state, dict):
                _hm_raw = _hm_state.get("incidents", {})

                if isinstance(_hm_raw, dict):
                    _hm_incidents = [
                        dict(v, incident_id=v.get("incident_id", k))
                        if isinstance(v, dict)
                        else {"incident_id": k}
                        for k, v in _hm_raw.items()
                    ]
                elif isinstance(_hm_raw, list):
                    _hm_incidents = [
                        x for x in _hm_raw if isinstance(x, dict)
                    ]
                else:
                    _hm_incidents = []

                _hm_active = []
                for _row in _hm_incidents:
                    _state = str(
                        _row.get("state")
                        or _row.get("incident_state")
                        or ""
                    ).upper()

                    _closed = (
                        _state in {"CLOSED", "RESOLVED", "RECOVERED"}
                        or bool(_row.get("closed_at_utc"))
                        or bool(_row.get("resolved_at_utc"))
                        or bool(_row.get("recovered_at_utc"))
                    )

                    if not _closed:
                        _hm_active.append(_row)

                def _hm_status(row):
                    return str(
                        row.get("severity")
                        or row.get("status")
                        or "UNKNOWN"
                    ).upper()

                _hm_order = {
                    "OWNER_ACTION_REQUIRED": 4,
                    "CRITICAL": 3,
                    "WARNING": 2,
                    "HEALTHY": 1,
                }

                if _hm_active:
                    _hm_overall = max(
                        (_hm_status(x) for x in _hm_active),
                        key=lambda x: _hm_order.get(x, 0),
                    )
                else:
                    _hm_overall = "HEALTHY"

                _hm_owner = sum(
                    _hm_status(x) == "OWNER_ACTION_REQUIRED"
                    for x in _hm_active
                )
                _hm_critical = sum(
                    _hm_status(x) == "CRITICAL"
                    for x in _hm_active
                )
                _hm_warning = sum(
                    _hm_status(x) == "WARNING"
                    for x in _hm_active
                )

                _h1, _h2, _h3, _h4 = st.columns(4)
                _hm_display_status = {
                    "OWNER_ACTION_REQUIRED": "ACTION REQUIRED",
                    "CRITICAL": "CRITICAL",
                    "WARNING": "WARNING",
                    "HEALTHY": "HEALTHY",
                }.get(_hm_overall, _hm_overall)

                _h1.metric("System Status", _hm_display_status)
                _h2.metric("Active Incidents", len(_hm_active))
                _h3.metric("Owner Action", _hm_owner)
                _h4.metric(
                    "Critical / Warning",
                    _hm_critical + _hm_warning,
                )

                _hm_reason_labels = {
                    "CURRENT_CONSUMER_FORECAST_VALID": "Forecast Evidence Aging",
                    "STARTER_EVIDENCE_VALID": "Starter Evidence Aging",
                }

                st.markdown("#### Active Incidents")

                if not _hm_active:
                    st.success("No active Health Monitor incidents.")
                else:
                    _hm_rows = []
                    for _row in sorted(
                        _hm_active,
                        key=lambda x: -_hm_order.get(
                            _hm_status(x), 0
                        ),
                    ):
                        _hm_rows.append(
                            {
                                "Status": _hm_status(_row),
                                "Check": (
                                    _row.get("check_id")
                                    or _row.get("check")
                                    or "—"
                                ),
                                "Scope": _row.get("scope") or "—",
                                "Reason": (
                                    _row.get("reason_code")
                                    or _row.get("reason")
                                    or "—"
                                ),
                                "First Seen": (
                                    _row.get("first_seen_utc") or "—"
                                ),
                                "Last Seen": (
                                    _row.get("last_seen_utc") or "—"
                                ),
                            }
                        )

                    for _item in _hm_rows:
                        _display_status = {
                            "OWNER_ACTION_REQUIRED": "ACTION REQUIRED",
                            "CRITICAL": "CRITICAL",
                            "WARNING": "WARNING",
                            "HEALTHY": "HEALTHY",
                        }.get(
                            _item["Status"],
                            _item["Status"],
                        )

                        _reason_raw = str(_item["Reason"])
                        _reason_display = _hm_reason_labels.get(
                            _reason_raw,
                            _reason_raw.replace("_", " ").title(),
                        )

                        _check_display = str(
                            _item["Check"]
                        ).replace("_", " ").title()

                        with st.container(border=True):
                            st.markdown(
                                f"**{_display_status} — "
                                f"{_item['Scope']}**"
                            )
                            st.markdown(f"**{_check_display}**")
                            st.write(_reason_display)

                            def _hm_time(value):
                                try:
                                    _ts = pd.to_datetime(
                                        value,
                                        utc=True,
                                    )
                                    return _ts.strftime(
                                        "%b %d • %I:%M %p UTC"
                                    ).replace(" 0", " ")
                                except Exception:
                                    return str(value)

                            st.caption(
                                "First seen: "
                                f"{_hm_time(_item['First Seen'])}"
                            )
                            st.caption(
                                "Last seen: "
                                f"{_hm_time(_item['Last Seen'])}"
                            )

                if isinstance(_hm_raw, dict):
                    _hm_recovery_source = [
                        v for v in _hm_raw.values()
                        if isinstance(v, dict)
                    ]
                elif isinstance(_hm_raw, list):
                    _hm_recovery_source = [
                        v for v in _hm_raw
                        if isinstance(v, dict)
                    ]
                else:
                    _hm_recovery_source = []

                _hm_recovered = [
                    row for row in _hm_recovery_source
                    if (
                        row.get("resolved_at_utc")
                        or row.get("recovered_at_utc")
                        or row.get("closed_at_utc")
                    )
                ]

                def _hm_resolution_time(row):
                    return (
                        row.get("resolved_at_utc")
                        or row.get("recovered_at_utc")
                        or row.get("closed_at_utc")
                        or ""
                    )

                _hm_recovered.sort(
                    key=lambda row: str(
                        _hm_resolution_time(row)
                    ),
                    reverse=True,
                )

                if _hm_recovered:
                    with st.expander(
                        f"✅ Recent Recoveries ({len(_hm_recovered)})",
                        expanded=False,
                    ):
                        for _rec in _hm_recovered[:10]:
                            _rec_scope = str(
                                _rec.get("scope") or "System"
                            )
                            _rec_check = str(
                                _rec.get("check_id") or "monitor"
                            ).replace("_", " ").title()
                            _rec_reason_raw = str(
                                _rec.get("reason_code") or "RECOVERED"
                            )
                            _rec_reason = _hm_reason_labels.get(
                                _rec_reason_raw,
                                _rec_reason_raw.replace("_", " ").title(),
                            )

                            try:
                                _rec_time = pd.to_datetime(
                                    _hm_resolution_time(_rec),
                                    utc=True,
                                ).strftime(
                                    "%b %d • %I:%M %p UTC"
                                ).replace(" 0", " ")
                            except Exception:
                                _rec_time = str(
                                    _hm_resolution_time(_rec)
                                )

                            with st.container(border=True):
                                st.markdown(
                                    f"**RECOVERED — {_rec_scope}**"
                                )
                                st.write(
                                    f"{_rec_check} — {_rec_reason}"
                                )
                                st.caption(
                                    f"Recovered: {_rec_time}"
                                )

                st.caption(
                    "Display only • repair execution disabled"
                )

    with users_tab:
        try:
            with accounts_connect() as conn:
                users_df = pd.read_sql_query(
                    """
                        SELECT
                            display_name,
                            email,
                            created_at,
                            updated_at
                        FROM wfs_users
                        ORDER BY datetime(created_at) DESC
                        """,
                    conn,
                )

            now_utc = pd.Timestamp.now(tz="UTC")

            if not users_df.empty:
                created = pd.to_datetime(
                    users_df["created_at"], errors="coerce", utc=True
                )
                updated = pd.to_datetime(
                    users_df["updated_at"], errors="coerce", utc=True
                )

                registered = len(users_df)
                active_today = int(
                    (updated >= now_utc - pd.Timedelta(days=1)).sum()
                )
                active_week = int(
                    (updated >= now_utc - pd.Timedelta(days=7)).sum()
                )
                new_week = int(
                    (created >= now_utc - pd.Timedelta(days=7)).sum()
                )
            else:
                registered = active_today = active_week = new_week = 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Registered Users", registered)
            c2.metric("Active 24 Hours", active_today)
            c3.metric("Active 7 Days", active_week)
            c4.metric("New Users 7 Days", new_week)

            st.markdown("### Users")

            if users_df.empty:
                st.info("No registered users yet.")
            else:
                users_df["Signed Up"] = pd.to_datetime(
                    users_df["created_at"], errors="coerce", utc=True
                )
                users_df["Last Seen"] = pd.to_datetime(
                    users_df["updated_at"], errors="coerce", utc=True
                )

                users_df["Account Age"] = (
                    now_utc - users_df["Signed Up"]
                ).dt.days.map(
                    lambda days: (
                        f"{int(days)} days"
                        if pd.notna(days)
                        else "—"
                    )
                )

                users_df = users_df.rename(
                    columns={
                        "display_name": "User",
                        "email": "Email",
                    }
                )[
                    [
                        "User",
                        "Email",
                        "Signed Up",
                        "Last Seen",
                        "Account Age",
                    ]
                ]

                st.dataframe(
                    users_df,
                    width="stretch",
                    hide_index=True,
                )

        except Exception as exc:
            st.error(
                f"Could not load Admin user activity: {exc}"
            )

    with feedback_tab:
        feedback_flash = st.session_state.pop(
            "wfs_admin_feedback_flash",
            None,
        )
        if feedback_flash:
            flash_kind, flash_message = feedback_flash
            if flash_kind == "success":
                st.success(flash_message)
            else:
                st.warning(flash_message)

        try:
            from wfs_user_feedback import (
                list_feedback,
                save_admin_response,
                close_feedback,
                email_feedback_response,
            )

            feedback_rows = list_feedback()

            with accounts_connect() as conn:
                identity_rows = conn.execute(
                    """
                        SELECT user_sub, display_name, email
                        FROM wfs_users
                        """
                ).fetchall()

            identity_map = {
                str(row[0]): {
                    "name": str(row[1] or "").strip(),
                    "email": str(row[2] or "").strip(),
                }
                for row in identity_rows
            }

            new_count = sum(
                1 for row in feedback_rows
                if row.get("status") == "NEW"
            )
            replied_count = sum(
                1 for row in feedback_rows
                if row.get("status") == "REPLIED"
            )
            closed_count = sum(
                1 for row in feedback_rows
                if row.get("status") == "CLOSED"
            )

            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Total Reports", len(feedback_rows))
            f2.metric("New", new_count)
            f3.metric("Replied", replied_count)
            f4.metric("Closed", closed_count)

            status_filter = st.selectbox(
                "Status",
                ["All", "NEW", "REPLIED", "CLOSED"],
                key="wfs_admin_feedback_status",
            )

            visible_rows = [
                row for row in feedback_rows
                if status_filter == "All"
                or row.get("status") == status_filter
            ]

            if not visible_rows:
                st.info("No feedback reports match this status.")
            else:
                for row in visible_rows:
                    feedback_id = str(row["feedback_id"])
                    short_id = feedback_id[:8].upper()
                    identity = identity_map.get(
                        str(row.get("user_id") or ""),
                        {},
                    )

                    user_name = (
                        identity.get("name")
                        or "WFS User"
                    )
                    user_email = (
                        identity.get("email")
                        or "Email unavailable"
                    )

                    status = str(
                        row.get("status") or "NEW"
                    ).upper()

                    category = str(
                        row.get("category") or "Other"
                    )

                    created = pd.to_datetime(
                        row.get("created_at_utc"),
                        errors="coerce",
                        utc=True,
                    )

                    created_label = (
                        created.strftime(
                            "%b %d, %Y %I:%M %p UTC"
                        )
                        if pd.notna(created)
                        else "Unknown time"
                    )

                    with st.expander(
                        f"{status} • {category} • "
                        f"{user_name} • {short_id}",
                        expanded=(status == "NEW"),
                    ):
                        st.caption(
                            f"{created_label} • {user_email}"
                        )

                        subject = str(
                            row.get("player_or_team") or ""
                        ).strip()

                        if subject:
                            st.write("Player / Team:", subject)

                        context_parts = []

                        if row.get("page"):
                            context_parts.append(
                                f"Page: {row['page']}"
                            )

                        if row.get("slate"):
                            context_parts.append(
                                f"Slate: {row['slate']}"
                            )

                        if row.get("season"):
                            context_parts.append(
                                f"Season: {row['season']}"
                            )

                        if row.get("week"):
                            context_parts.append(
                                f"Week: {row['week']}"
                            )

                        if context_parts:
                            st.caption(
                                " • ".join(context_parts)
                            )

                        st.markdown("**User message**")
                        st.write(row.get("message") or "")

                        existing_response = str(
                            row.get("admin_response") or ""
                        )

                        response = st.text_area(
                            "Your response",
                            value=existing_response,
                            max_chars=4000,
                            key=(
                                "wfs_admin_feedback_response_"
                                f"{feedback_id}"
                            ),
                        )

                        b1, b2 = st.columns(2)

                        if b1.button(
                            "Save Reply",
                            key=(
                                "wfs_admin_feedback_reply_"
                                f"{feedback_id}"
                            ),
                            width="stretch",
                        ):
                            try:
                                save_admin_response(
                                    feedback_id,
                                    response,
                                )
                            except Exception as exc:
                                st.error(
                                    f"Could not save reply: {exc}"
                                )
                            else:
                                # The response is already committed to
                                # the feedback DB before email delivery.
                                delivery_ok = False
                                delivery_detail = (
                                    "User email is unavailable."
                                )

                                if (
                                    user_email
                                    and user_email
                                    != "Email unavailable"
                                ):
                                    try:
                                        (
                                            delivery_ok,
                                            delivery_detail,
                                        ) = email_feedback_response(
                                            feedback_id=feedback_id,
                                            user_email=user_email,
                                            response=response,
                                        )
                                    except Exception as exc:
                                        delivery_detail = (
                                            "Email delivery failed: "
                                            f"{type(exc).__name__}"
                                        )

                                if delivery_ok:
                                    st.session_state[
                                        "wfs_admin_feedback_flash"
                                    ] = (
                                        "success",
                                        "Reply saved and emailed.",
                                    )
                                else:
                                    st.session_state[
                                        "wfs_admin_feedback_flash"
                                    ] = (
                                        "warning",
                                        "Reply saved, but email was "
                                        "not delivered. "
                                        f"{delivery_detail}",
                                    )

                                st.rerun()

                        if status != "CLOSED":
                            if b2.button(
                                "Close",
                                key=(
                                    "wfs_admin_feedback_close_"
                                    f"{feedback_id}"
                                ),
                                width="stretch",
                            ):
                                try:
                                    close_feedback(feedback_id)
                                except Exception as exc:
                                    st.error(
                                        f"Could not close report: {exc}"
                                    )
                                else:
                                    st.success("Report closed.")
                                    st.rerun()

                        if row.get("responded_at_utc"):
                            st.caption(
                                "Response saved: "
                                f"{row['responded_at_utc']}"
                            )

                        if row.get("closed_at_utc"):
                            st.caption(
                                "Closed: "
                                f"{row['closed_at_utc']}"
                            )

        except Exception as exc:
            st.error(
                f"Could not load user feedback: {exc}"
            )
