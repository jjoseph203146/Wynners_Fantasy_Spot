"""Explicit, feed-only shadow adapter. Importing this module performs no I/O."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

from .core import SAFETY, digest, events, evaluate, require, timestamp

SOURCE = "ESPN_NFL_RSS_V1"
ENDPOINT = "https://www.espn.com/espn/rss/nfl/news"
ORIGIN = "ESPN"
USER_AGENT = "WFS-Player-Analysis-Shadow/1.0 (NFL RSS client)"
TIMEOUT_SECONDS = 15
MAX_BYTES = 2_000_000


class FeedError(ValueError):
    """No usable feed was acquired; callers must not substitute stale evidence."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise FeedError("REDIRECT_FORBIDDEN")


def fetch(prior_events=()):
    """One bounded request to ENDPOINT only. No redirects, retries or article requests."""
    request = urllib.request.Request(ENDPOINT, headers={"User-Agent": USER_AGENT,
                                                     "Accept": "application/rss+xml, application/xml, text/xml"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
            if response.status != 200 or response.geturl() != ENDPOINT:
                raise FeedError("UNEXPECTED_HTTP_RESPONSE")
            content_type = response.headers.get_content_type()
            if content_type not in {"application/rss+xml", "application/xml", "text/xml"}:
                raise FeedError("UNEXPECTED_CONTENT_TYPE")
            raw = response.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise FeedError("FEED_TOO_LARGE")
    except (OSError, urllib.error.URLError) as exc:
        raise FeedError("FEED_REQUEST_FAILED") from exc
    retrieved = datetime.now(timezone.utc).isoformat()
    return ingest(raw, retrieved, prior_events)


def classify(text):
    """Routing labels only; never signals, identities, or inferred opportunity."""
    if re.search(r"\b(recap|in (?:a |the )?(?:win|loss)|caught \d+|catches \d+|had \d+|finished with)\b", text, re.I):
        return "POST_GAME_RECAP"
    if re.search(r"\bcoach\b.*\b(?:says|said)\b.*\bwill\b.*\bstarting role\b", text, re.I):
        return "COACH_INTENT"
    return "INSUFFICIENT_UNRESOLVED"


def _publication(value):
    try:
        # Only explicit RFC-822 numeric offsets or unambiguous UTC designators.
        require(bool(re.search(r"(?:[+-]\d{4}|GMT|UTC)\s*$", value)), "PUBLICATION_TIMEZONE_REQUIRED")
        parsed = parsedate_to_datetime(value)
        require(parsed.tzinfo is not None, "PUBLICATION_TIMEZONE_REQUIRED")
        return parsed.astimezone(timezone.utc).isoformat()
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("INVALID_PUBLICATION_TIMESTAMP") from exc


def ingest(raw, retrieved_at_utc, prior_events=()):
    """Offline parsing. Pass previous accepted captures to detect cross-fetch revisions.

    RSS lacks reliable revision times: same-ID changes without a changed pubDate
    quarantine the entire conflicting logical event, never fabricate update times.
    """
    retrieved = timestamp(retrieved_at_utc).astimezone(timezone.utc).isoformat()
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_BYTES:
        raise FeedError("EMPTY_OR_OVERSIZED_FEED")
    # Reject declarations (including UTF-16 encodings) before XML entity expansion.
    if b"<!DOCTYPE" in raw.replace(b"\x00", b"").upper() or b"<!ENTITY" in raw.replace(b"\x00", b"").upper():
        raise FeedError("XML_DECLARATION_FORBIDDEN")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise FeedError("MALFORMED_XML") from exc
    if root.tag != "rss" or len(root.findall("channel")) != 1:
        raise FeedError("UNEXPECTED_FEED")
    channel = root.find("channel")
    items = channel.findall("item")
    if not items:
        raise FeedError("EMPTY_FEED")
    captures, quarantine = [], []
    timestamp_guids = {}
    for item in items:
        raw_date = item.findtext("pubDate", default="")
        guid = item.findtext("guid", default="").strip()
        if raw_date and guid:
            timestamp_guids.setdefault(raw_date, set()).add(guid)
    # Feed-local source quality only; never participates in event normalization.
    timestamp_diagnostics = [dict(raw_pubDate=value,
        repeated_source_timestamp_count=len(guids), source_timestamp_suspect=len(guids) > 1)
        for value, guids in sorted(timestamp_guids.items())]
    for item in items:
        fields = {key: item.findtext(key, default="") for key in ("title", "description", "link", "guid", "pubDate")}
        fields["categories"] = [node.text or "" for node in item.findall("category")]
        base = dict(source=SOURCE, origin_id=ORIGIN, headline=fields["title"], summary=fields["description"],
                    article_url=fields["link"], categories=fields["categories"], raw_feed_fields=fields,
                    retrieved_at_utc=retrieved, first_seen_at_utc=retrieved,
                    content_hash=digest(fields), approval_id="ESPN_RSS_SHADOW_CAPTURE_ONLY")
        base["evidence_text"] = fields["title"] + "\n" + fields["description"]
        base["source_phase"] = classify(base["evidence_text"])
        # Assign provenance even when publication parsing cannot succeed.
        base["source_event_id"] = ("guid:" + fields["guid"].strip() if fields["guid"].strip() else
            "derived:" + digest([SOURCE, fields["link"]] if fields["link"].strip() else
                                [SOURCE, fields["title"], fields["pubDate"]]))
        base.update(published_at_utc=None, temporal_status="AMBIGUOUS_SOURCE_TIMESTAMP",
                    temporal_confidence="UNRESOLVED")
        try:
            require(all(len(item.findall(k)) <= 1 for k in ("title", "description", "link", "guid", "pubDate")), "DUPLICATE_ITEM_FIELD")
            require(bool(fields["title"].strip()), "MISSING_HEADLINE")
            base["published_at_utc"] = _publication(fields["pubDate"])
            base.update(temporal_status="UNAMBIGUOUS_SOURCE_TIMESTAMP", temporal_confidence="RESOLVED")
            base["source_event_id"] = ("guid:" + fields["guid"].strip() if fields["guid"].strip() else
                "derived:" + digest([SOURCE, fields["link"]] if fields["link"].strip() else
                                    [SOURCE, fields["title"], base["published_at_utc"]]))
            captures.extend(events([base]))
        except (ValueError, TypeError) as exc:
            if str(exc) == "EVENT_TIME_ORDER":
                base.update(temporal_status="INVALID_SOURCE_TIME_ORDER", temporal_confidence="UNRESOLVED")
            quarantine.append(dict(base, reason_codes=[str(exc)], eligibility_status="QUARANTINED"))
    groups = {}
    for event in [*prior_events, *captures]:
        require(event.get("source") == SOURCE and event.get("origin_id") == ORIGIN, "PRIOR_SOURCE_BOUNDARY")
        clean = {k: v for k, v in event.items() if k not in {"supersedes_revision_id"}}
        groups.setdefault(clean["source_event_id"], []).append(clean)
    normalized = []
    for _, group in sorted(groups.items()):
        try:
            normalized.extend(events(group))
        except ValueError as exc:
            quarantine.extend(dict(row, eligibility_status="QUARANTINED", reason_codes=[str(exc)]) for row in group)
    # Every source event is retained, but none is automatically player evidence.
    quarantine.extend(dict(row, eligibility_status="QUARANTINED", gsis_id=None,
                           reason_codes=["IDENTITY_UNRESOLVED", "MANUAL_CLAIM_REVIEW_REQUIRED"])
                      for row in normalized)
    quarantine = sorted({digest(row): row for row in quarantine}.values(), key=digest)
    return dict(events=normalized, quarantine=quarantine, claims=[], consensus=[], safety=dict(SAFETY),
                retrieval=dict(source=SOURCE, endpoint=ENDPOINT, retrieved_at_utc=retrieved,
                               feed_sha256=hashlib.sha256(raw).hexdigest(), feed_title=channel.findtext("title", ""),
                               source_timestamp_diagnostics=timestamp_diagnostics))


def evaluate_candidate(claim, event, snapshot, game, cutoff):
    """Offline manual review helper; does not publish or promote claims.

    Identity and time checks belong to the unchanged foundation. A routing label
    cannot be overridden to turn recap/insufficient text into opportunity.
    """
    require(event.get("source") == SOURCE and event.get("origin_id") == ORIGIN, "SOURCE_BOUNDARY")
    require(event.get("temporal_confidence") != "UNRESOLVED" and
            bool(event.get("published_at_utc")), "UNRESOLVED_SOURCE_TIMESTAMP")
    row = evaluate(claim, event, snapshot, game, cutoff)
    phase = classify(event["evidence_text"])
    if phase in {"POST_GAME_RECAP", "INSUFFICIENT_UNRESOLVED"}:
        row["eligibility_status"] = "QUARANTINED"
        row["reason_codes"].append(phase)
    return row
