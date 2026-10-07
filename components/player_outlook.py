"""Public allowlist, literal football descriptions, and presentation only."""

from dataclasses import dataclass
from zoneinfo import ZoneInfo

from player_outlook_data import ValidatedOutlook


UNAVAILABLE = "Player matchup outlook is not available for this game."


@dataclass(frozen=True)
class PublicPlayerCard:
    name: str
    position: str
    team: str
    opponent: str
    scoring: str
    usage: str
    opponent_context: str
    explanation: str


@dataclass(frozen=True)
class PublicOutlook:
    available: bool
    message: str = UNAVAILABLE
    game_label: str = ""
    capture_label: str = ""
    cards: tuple[PublicPlayerCard, ...] = ()


def _number(value):
    return f"{value:.1f}".removesuffix(".0")


def build_public_player_cards(validated_outlook) -> PublicOutlook:
    """Deliberately copy no identities, statuses, hashes, paths, or debug fields."""
    if not isinstance(validated_outlook, ValidatedOutlook) or not validated_outlook.available:
        return PublicOutlook(False)
    cards = []
    for player in validated_outlook.players:
        metrics, comparisons = dict(player.metrics), dict(player.comparisons)
        scoring = "Limited recent history."
        if "fd_avg_3" in metrics:
            scoring = (
                f"{_number(metrics['fd_avg_3'])} FD points/game over his last "
                f"{player.recent_games} recorded games."
            )
            if "fd_avg_5" in metrics and player.baseline_games > player.recent_games:
                scoring += f" {_number(metrics['fd_avg_5'])} over {player.baseline_games}."
        usage_parts = []
        if player.position != "QB":
            fields = [("targets_avg_3", "targets/game")]
            if player.position == "RB":
                fields.insert(0, ("carries_avg_3", "carries/game"))
            for field, label in fields:
                if field in metrics:
                    usage_parts.append(f"{_number(metrics[field])} {label}")
            if "snap_pct_avg_3" in metrics:
                usage_parts.append(f"{_number(metrics['snap_pct_avg_3'] * 100)}% of offensive snaps")
        usage = (
            " · ".join(usage_parts) + f" over his last {player.recent_games} recorded games."
            if usage_parts else ""
        )
        preferences = {
            "QB": [("fd_allowed_avg_3", "FD points")],
            "RB": [("rushing_yards_allowed_avg_3", "rushing yards"), ("carries_allowed_avg_3", "carries"), ("opportunities_allowed_avg_3", "carries plus targets"), ("fd_allowed_avg_3", "FD points")],
            "WR": [("targets_allowed_avg_3", "targets"), ("receiving_yards_allowed_avg_3", "receiving yards"), ("receptions_allowed_avg_3", "receptions"), ("fd_allowed_avg_3", "FD points")],
            "TE": [("receptions_allowed_avg_3", "receptions"), ("receiving_yards_allowed_avg_3", "receiving yards"), ("fd_allowed_avg_3", "FD points")],
        }[player.position]
        opponent_parts = [f"{_number(metrics[field])} {label}" for field, label in preferences if field in metrics][:2]
        opponent_context = (
            f"{player.opponent} allowed opposing {player.position}s "
            + " and ".join(opponent_parts)
            + " per game over its last three recorded games."
            if opponent_parts else "Limited opponent history."
        )
        explanation = []
        if "scoring" in comparisons:
            value = comparisons["scoring"]
            explanation.append(
                "Recent scoring matches his five-game average."
                if value == "unchanged" else f"Recent scoring is {value} than his five-game average."
            )
        if player.position != "QB":
            channel = "carries" if player.position == "RB" and "carries" in comparisons else "targets"
            if channel in comparisons:
                value = comparisons[channel]
                explanation.append(
                    f"Recent {channel} match his five-game average."
                    if value == "unchanged" else f"Recent {channel} are {value} than his five-game average."
                )
        cards.append(PublicPlayerCard(
            player.name, player.position, player.team, player.opponent,
            scoring, usage, opponent_context,
            " ".join(explanation) or "Limited history for a three-game versus five-game comparison.",
        ))
    stamp = validated_outlook.captured_at_utc.astimezone(ZoneInfo("America/New_York"))
    return PublicOutlook(
        True, message="",
        game_label=f"Pregame Outlook · {validated_outlook.away_team} at {validated_outlook.home_team} · {validated_outlook.season} Week {validated_outlook.week}",
        capture_label=f"Captured before kickoff · {stamp.strftime('%b %d, %I:%M %p %Z')}",
        cards=tuple(sorted(cards, key=lambda card: (card.team, card.position, card.name))),
    )


def render_player_outlook(*, outlook: PublicOutlook, key_prefix: str) -> None:
    """Render prepared public data only; no artifact or authority access."""
    import streamlit as st

    if not outlook.available:
        st.caption(UNAVAILABLE)
        return
    with st.expander("Player Matchup Outlook", expanded=False):
        st.caption(outlook.game_label)
        st.caption(outlook.capture_label)
        st.caption("Historical scoring and usage for currently relevant pregame players. Opponent totals describe the position group, not a player forecast.")
        groups = sorted({(card.team, card.position) for card in outlook.cards})
        selected = st.selectbox(
            "Team / position", groups,
            format_func=lambda group: f"{group[0]} · {group[1]}",
            key=f"{key_prefix}_group",
        )
        for card in outlook.cards:
            if (card.team, card.position) != selected:
                continue
            with st.container(border=True):
                st.text(f"{card.name} — {card.position} — {card.team}")
                st.caption("Recent scoring")
                st.text(card.scoring)
                if card.usage:
                    st.caption("Role / usage")
                    st.text(card.usage)
                st.caption("Opponent positional context")
                st.text(card.opponent_context)
                st.caption("Why it matters")
                st.text(card.explanation)
