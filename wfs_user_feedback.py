"""
WFS Public User Feedback V1

Public observation/reporting layer only.

Security / production contract:
- User feedback is evidence, never authority.
- This module does not modify projections.
- This module does not modify injury authority.
- This module does not modify player identity.
- This module does not modify schedules or slates.
- This module does not modify solver eligibility.
- This module does not modify production artifacts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
import smtplib
import sqlite3
import uuid

import streamlit as st


ROOT = Path(__file__).resolve().parent
FEEDBACK_DB = ROOT / "data" / "wfs_user_feedback.db"

FEEDBACK_CATEGORIES = [
    "Player / Team Information",
    "Injury / Availability",
    "Missing Player",
    "Slate / Game Information",
    "Projection Concern",
    "Late Swap",
    "Lineup Generation",
    "Website / Display",
    "Other",
]


def _connect() -> sqlite3.Connection:
    FEEDBACK_DB.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str(FEEDBACK_DB),
        timeout=10,
    )

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    return conn


def init_feedback_store() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_feedback (
                feedback_id TEXT PRIMARY KEY,
                created_at_utc TEXT NOT NULL,
                user_id TEXT,
                category TEXT NOT NULL,
                message TEXT NOT NULL,
                page TEXT NOT NULL,
                contest_format TEXT,
                slate TEXT,
                season INTEGER,
                week INTEGER,
                player_or_team TEXT,
                status TEXT NOT NULL DEFAULT 'NEW',
                ai_classification TEXT,
                ai_summary TEXT,
                admin_notes TEXT
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_user_feedback_created_at
            ON user_feedback(created_at_utc)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_user_feedback_status
            ON user_feedback(status)
            """
        )

        # WFS_ADMIN_FEEDBACK_INBOX_V1
        # Additive migration for Admin replies. Existing public feedback
        # submissions remain unchanged.
        existing_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(user_feedback)")
        }

        for column_name, column_sql in (
            ("admin_response", "TEXT"),
            ("responded_at_utc", "TEXT"),
            ("closed_at_utc", "TEXT"),
        ):
            if column_name not in existing_columns:
                conn.execute(
                    f"ALTER TABLE user_feedback "
                    f"ADD COLUMN {column_name} {column_sql}"
                )




# =========================================================
# WFS_FEEDBACK_EMAIL_V1
# Notification transport only.
#
# Contract:
# - Feedback DB remains authoritative.
# - Callers persist feedback/replies BEFORE attempting email.
# - Missing/bad email configuration returns failure safely.
# - Credentials come only from Streamlit secrets.
# =========================================================

def send_feedback_email(
    *,
    to_email: str,
    subject: str,
    body: str,
    reply_to: str | None = None,
) -> tuple[bool, str]:
    """
    Send one feedback-related email.

    Returns:
        (True, "sent") on success.
        (False, reason) on configuration/delivery failure.

    This function never modifies feedback records.
    """
    to_email = str(to_email or "").strip()
    subject = str(subject or "").strip()
    body = str(body or "").strip()
    reply_to = str(reply_to or "").strip() or None

    if not to_email:
        return False, "Recipient email is unavailable."

    try:
        cfg = st.secrets.get("email", {})
    except Exception:
        return False, "Email configuration is unavailable."

    host = str(cfg.get("host", "") or "").strip()
    username = str(cfg.get("username", "") or "").strip()
    password = str(cfg.get("password", "") or "").strip()
    from_email = str(
        cfg.get("from_email", "") or username
    ).strip()
    from_name = str(
        cfg.get("from_name", "") or "WFS"
    ).strip()

    try:
        port = int(cfg.get("port", 587))
    except (TypeError, ValueError):
        return False, "Email port configuration is invalid."

    if not host or not username or not password or not from_email:
        return False, "Email is not configured."

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"{from_name} <{from_email}>"
    msg["To"] = to_email

    if reply_to:
        msg["Reply-To"] = reply_to

    msg.set_content(body)

    try:
        with smtplib.SMTP(host, port, timeout=15) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(username, password)
            smtp.send_message(msg)

    except Exception as exc:
        return False, (
            f"Email delivery failed: "
            f"{type(exc).__name__}"
        )

    return True, "sent"


def notify_admin_of_feedback(
    *,
    feedback_id: str,
    category: str,
    message: str,
    user_name: str | None = None,
    user_email: str | None = None,
    player_or_team: str | None = None,
    season: int | None = None,
    week: int | None = None,
) -> tuple[bool, str]:
    """Notify the configured WFS feedback inbox."""
    try:
        cfg = st.secrets.get("email", {})
    except Exception:
        return False, "Email configuration is unavailable."

    admin_email = str(
        cfg.get("feedback_to", "") or ""
    ).strip()

    if not admin_email:
        return False, "Feedback notification recipient is not configured."

    short_id = str(feedback_id or "")[:8].upper()

    lines = [
        "A new WFS user feedback report was submitted.",
        "",
        f"Reference: {short_id}",
        f"Category: {category}",
        f"User: {user_name or 'WFS User'}",
        f"User email: {user_email or 'Unavailable'}",
    ]

    if player_or_team:
        lines.append(f"Player / Team: {player_or_team}")

    if season is not None:
        lines.append(f"Season: {season}")

    if week is not None:
        lines.append(f"Week: {week}")

    lines.extend(
        [
            "",
            "Message:",
            str(message or ""),
            "",
            "Open WFS Admin → User Feedback to review.",
        ]
    )

    return send_feedback_email(
        to_email=admin_email,
        subject=f"WFS Feedback [{short_id}] - {category}",
        body="\n".join(lines),
        reply_to=user_email,
    )


def email_feedback_response(
    *,
    feedback_id: str,
    user_email: str,
    response: str,
) -> tuple[bool, str]:
    """Email a previously persisted Admin response to the user."""
    short_id = str(feedback_id or "")[:8].upper()

    body = "\n".join(
        [
            "WFS has responded to your feedback report.",
            "",
            f"Reference: {short_id}",
            "",
            "Response:",
            str(response or ""),
            "",
            "Thank you for helping us improve WFS.",
        ]
    )

    return send_feedback_email(
        to_email=user_email,
        subject=f"WFS Feedback Response [{short_id}]",
        body=body,
    )

def list_feedback(
    status: str | None = None,
) -> list[dict]:
    """Read feedback for the private Admin inbox."""
    init_feedback_store()

    sql = """
        SELECT
            feedback_id,
            created_at_utc,
            user_id,
            category,
            message,
            page,
            contest_format,
            slate,
            season,
            week,
            player_or_team,
            status,
            admin_response,
            responded_at_utc,
            closed_at_utc
        FROM user_feedback
    """
    params: tuple = ()

    if status:
        normalized = str(status).strip().upper()
        if normalized not in {"NEW", "REPLIED", "CLOSED"}:
            raise ValueError("Invalid feedback status.")
        sql += " WHERE status = ?"
        params = (normalized,)

    sql += " ORDER BY datetime(created_at_utc) DESC"

    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def save_admin_response(
    feedback_id: str,
    response: str,
) -> None:
    """Persist an Admin response before any external delivery occurs."""
    feedback_id = str(feedback_id or "").strip()
    response = str(response or "").strip()

    if not feedback_id:
        raise ValueError("Feedback ID is required.")

    if not response:
        raise ValueError("Response cannot be empty.")

    if len(response) > 4000:
        raise ValueError("Response is too long.")

    responded_at = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        cur = conn.execute(
            """
            UPDATE user_feedback
            SET
                admin_response = ?,
                responded_at_utc = ?,
                status = 'REPLIED',
                closed_at_utc = NULL
            WHERE feedback_id = ?
            """,
            (response, responded_at, feedback_id),
        )

        if cur.rowcount != 1:
            raise ValueError("Feedback record was not found.")


def close_feedback(feedback_id: str) -> None:
    """Close one feedback item without deleting its history."""
    feedback_id = str(feedback_id or "").strip()

    if not feedback_id:
        raise ValueError("Feedback ID is required.")

    closed_at = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        cur = conn.execute(
            """
            UPDATE user_feedback
            SET
                status = 'CLOSED',
                closed_at_utc = ?
            WHERE feedback_id = ?
            """,
            (closed_at, feedback_id),
        )

        if cur.rowcount != 1:
            raise ValueError("Feedback record was not found.")

def submit_feedback(
    *,
    category: str,
    message: str,
    user_id: str | None = None,
    page: str = "Home",
    contest_format: str | None = None,
    slate: str | None = None,
    season: int | None = None,
    week: int | None = None,
    player_or_team: str | None = None,
) -> str:
    category = str(category or "").strip()
    message = str(message or "").strip()

    if category not in FEEDBACK_CATEGORIES:
        raise ValueError("Invalid feedback category.")

    if not message:
        raise ValueError("Feedback message cannot be empty.")

    if len(message) > 4000:
        raise ValueError("Feedback message is too long.")

    player_or_team = str(player_or_team or "").strip() or None

    if player_or_team and len(player_or_team) > 160:
        raise ValueError("Player or team value is too long.")

    feedback_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO user_feedback (
                feedback_id,
                created_at_utc,
                user_id,
                category,
                message,
                page,
                contest_format,
                slate,
                season,
                week,
                player_or_team,
                status,
                ai_classification,
                ai_summary,
                admin_notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'NEW', NULL, NULL, NULL)
            """,
            (
                feedback_id,
                created_at,
                user_id,
                category,
                message,
                str(page or "Home"),
                contest_format,
                slate,
                season,
                week,
                player_or_team,
            ),
        )

    return feedback_id


def render_public_feedback(
    *,
    user_id: str | None = None,
    user_name: str | None = None,
    user_email: str | None = None,
    season: int | None = None,
    week: int | None = None,
) -> None:
    """
    Public feedback form.

    This is intentionally observation-only. Submissions enter NEW state.
    No production repair or AI mutation occurs here.
    """

    init_feedback_store()

    # =========================================================
    # WFS_FEEDBACK_MOBILE_UX_V1
    # Presentation only. No feedback storage/authority changes.
    # =========================================================

    st.markdown(
        """
        <style>
        /* Feedback textarea only */
        div[data-testid="stTextArea"] textarea {
            min-height: 118px !important;
            border: 2px solid
                color-mix(
                    in srgb,
                    var(--st-text-color) 48%,
                    transparent
                ) !important;
            border-radius: 12px !important;
            font-size: 16px !important;
            line-height: 1.45 !important;
        }

        div[data-testid="stTextArea"] textarea:focus {
            border-color: var(--st-primary-color) !important;
            box-shadow: 0 0 0 1px var(--st-primary-color) !important;
        }

        div[data-testid="stTextArea"] textarea::placeholder {
            color:
                color-mix(
                    in srgb,
                    var(--st-text-color) 68%,
                    transparent
                ) !important;
            opacity: 1 !important;
        }

        @media (max-width: 700px) {
            div[data-testid="stTextArea"] textarea {
                min-height: 96px !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Help Us Keep the Data Accurate")

    st.caption(
        "See something that doesn't look right? Report incorrect player "
        "information, injuries, missing players, slate information, lineup "
        "issues, or other problems. Your feedback helps us investigate "
        "potential issues and improve WFS."
    )

    with st.form(
        "wfs_public_feedback_form",
        clear_on_submit=True,
    ):
        category = st.selectbox(
            "What looks wrong?",
            FEEDBACK_CATEGORIES,
            key="wfs_public_feedback_category",
        )

        player_or_team = st.text_input(
            "Player or team (optional)",
            max_chars=160,
            placeholder="Example: Player name, BUF, DET @ BUF",
            key="wfs_public_feedback_subject",
        )

        message = st.text_area(
            "Message",
            max_chars=4000,
            height=110,
            placeholder=(
                "Tap here and tell us what you noticed."
            ),
            help=(
                "Required — describe what you expected to see "
                "and what appears incorrect."
            ),
            key="wfs_public_feedback_message",
        )

        submitted = st.form_submit_button(
            "Submit Report",
            width="stretch",
        )

    if submitted:
        try:
            feedback_id = submit_feedback(
                category=category,
                message=message,
                user_id=user_id,
                page="Home",
                season=season,
                week=week,
                player_or_team=player_or_team,
            )

        except ValueError as exc:
            st.warning(str(exc))

        except Exception:
            st.error(
                "Your feedback could not be submitted right now. "
                "Please try again."
            )

        else:
            # Database commit has already succeeded. Email notification
            # is secondary and must never invalidate the submission.
            try:
                notify_admin_of_feedback(
                    feedback_id=feedback_id,
                    category=category,
                    message=message,
                    user_name=user_name,
                    user_email=user_email,
                    player_or_team=player_or_team,
                    season=season,
                    week=week,
                )
            except Exception:
                pass

            st.success(
                "Report received. Thank you — we'll review the "
                "information you provided."
            )

            # Public-safe short reference only.
            st.caption(
                f"Report reference: {feedback_id[:8].upper()}"
            )
