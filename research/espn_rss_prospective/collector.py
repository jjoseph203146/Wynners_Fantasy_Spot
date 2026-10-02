#!/usr/bin/env python3
"""Manual ESPN evidence capture. No production imports or timestamp normalization."""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
DATASET = HERE / "evidence"
RSS_URL = "https://www.espn.com/espn/rss/nfl/news"
API_BASE = "https://content.core.api.espn.com/v1/sports/news/"
ALLOWED_HOSTS = {"www.espn.com", "espn.com", "content.core.api.espn.com"}
MAX_BYTES = 4 * 1024 * 1024
SOCKET_TIMEOUT = 8
REQUEST_SECONDS = 12
RUN_SECONDS = 120
MAX_ITEMS = 100
WORKERS = 4
EXPLICIT_TIME = re.compile(r"^\d{4}-\d\d-\d\d[Tt]\d\d:\d\d:\d\d(?:\.\d+)?(?:[Zz]|[+-]\d\d:?\d\d)$")


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, data):
    """Exclusive creation; files are never opened for overwrite."""
    with path.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def write_json(path, value):
    write_new(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode())


def permitted_url(url):
    try:
        p = urllib.parse.urlsplit(url)
        return (p.scheme == "https" and p.hostname in ALLOWED_HOSTS
                and p.port in (None, 443) and not p.username and not p.password)
    except (ValueError, TypeError):
        return False


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Preserve the redirect response, never follow it to another endpoint.
        return None


def http_get(url, run_deadline):
    """One public GET, no retry/redirect/auth; bounded size, socket and elapsed time."""
    started = time.monotonic()
    record = {"url": url, "started_at_utc": utc_now(), "status": None,
              "headers": [], "complete": False, "error": None, "attempted": False}
    body = bytearray()
    response = None
    try:
        if not permitted_url(url):
            raise ValueError("URL is outside the public ESPN HTTPS allowlist")
        remaining = run_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("run request budget exhausted; request not attempted")
        deadline = min(started + REQUEST_SECONDS, run_deadline)
        opener = urllib.request.build_opener(NoRedirect(), urllib.request.ProxyHandler({}))
        request = urllib.request.Request(url, headers={
            "User-Agent": "ESPN-RSS-Timestamp-Research/1.0",
            "Accept-Encoding": "identity",
        })
        record["attempted"] = True
        try:
            response = opener.open(request, timeout=min(SOCKET_TIMEOUT, remaining))
        except urllib.error.HTTPError as error:
            response = error  # Preserve complete error/redirect response bodies too.
        record["status"] = response.code
        record["final_url"] = response.geturl()
        record["headers"] = list(response.headers.items())
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("response elapsed-time budget exhausted")
            chunk = response.read1(min(65536, MAX_BYTES + 1 - len(body)))
            if not chunk:
                record["complete"] = True
                break
            body.extend(chunk)
            if len(body) > MAX_BYTES:
                raise ValueError("response exceeds raw artifact size bound")
        if not 200 <= record["status"] < 300:
            record["error"] = "HTTP status " + str(record["status"])
    except Exception as error:
        record["error"] = type(error).__name__ + ": " + str(error)
    finally:
        if response is not None:
            response.close()
        record["ended_at_utc"] = utc_now()
        record["elapsed_seconds"] = round(time.monotonic() - started, 6)
    return bytes(body), record


def parse_rss(raw):
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("RSS DTD/entity declarations are not accepted")
    root = ET.fromstring(raw)
    if root.tag != "rss" or root.find("channel") is None:
        raise ValueError("expected RSS channel")
    nodes = root.findall("./channel/item")
    if not nodes or len(nodes) > MAX_ITEMS:
        raise ValueError("RSS item count outside 1..100")
    items = {}
    duplicate_count = 0
    for node in nodes:
        fields = {k: node.findtext(k) for k in ("guid", "link", "title", "description", "pubDate")}
        if not fields["guid"] or not fields["guid"].strip():
            raise ValueError("RSS item missing GUID")
        if not fields["pubDate"] or not fields["pubDate"].strip():
            raise ValueError("RSS item missing pubDate")
        # Deliberately no strip(), date parser, abbreviation lookup or UTC conversion.
        guid = fields["guid"]
        if guid in items:
            if items[guid] != fields:
                raise ValueError("conflicting duplicate GUID: " + guid)
            duplicate_count += 1
        else:
            items[guid] = fields
    return list(items.values()), duplicate_count, len(nodes)


def classification(item, previous):
    if item["guid"] not in previous:
        return "NEW_GUID"
    if item["pubDate"] != previous[item["guid"]]["pubDate"]:
        return "PUBDATE_CHANGED"
    return "UNCHANGED"


def resolve_id(item):
    guid_match = re.search(r"(?:^|[-/])(\d+)$", item["guid"])
    url = item.get("link") or ""
    parsed = urllib.parse.urlsplit(url)
    link_match = re.search(r"/id/(\d+)(?:/|$)", parsed.path)
    query_id = urllib.parse.parse_qs(parsed.query).get("storyId", [None])[0]
    link_id = link_match.group(1) if link_match else query_id
    guid_id = guid_match.group(1) if guid_match else None
    if link_id and not link_id.isdigit():
        raise ValueError("non-numeric URL content ID")
    if link_id and guid_id and link_id != guid_id:
        raise ValueError("GUID and article URL content IDs disagree")
    return link_id or guid_id


def explicit_timestamps(value, path="$", result=None):
    result = [] if result is None else result
    if isinstance(value, dict):
        for k, v in value.items():
            explicit_timestamps(v, path + "." + k, result)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            explicit_timestamps(v, path + "[" + str(i) + "]", result)
    elif isinstance(value, str) and EXPLICIT_TIME.fullmatch(value):
        result.append({"path": path, "raw_value": value})
    return result


def api_metadata(raw, content_id):
    data = json.loads(raw)
    headlines = [h for h in data.get("headlines", []) if str(h.get("id")) == content_id]
    if len(headlines) != 1:
        raise ValueError("API did not return one matching content ID")
    h = headlines[0]
    return {"content_id": content_id, "canonical_url": h.get("links", {}).get("web", {}).get("href"),
            "contentKey": h.get("contentKey"), "dataSourceIdentifier": h.get("dataSourceIdentifier"),
            "timestamps": {k: h[k] for k in ("originallyPosted", "published", "lastModified", "categorized") if k in h},
            "explicit_timestamps": explicit_timestamps(h),
            "note": "Nested paths may describe related content; no revision correspondence asserted."}


class StructuredHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.meta = []
        self.canonical = []
        self.structured = []
        self.script_type = None
        self.script = []
        self.embedded = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta":
            self.meta.append(attrs)
        if tag == "link" and attrs.get("rel") == "canonical":
            self.canonical.append(attrs.get("href"))
        if tag == "time":
            self.meta.append({"element": "time", **attrs})
        if tag == "script":
            self.script_type = attrs.get("type", "")
            self.script = []

    def handle_data(self, data):
        if self.script_type is not None:
            self.script.append(data)

    def handle_endtag(self, tag):
        if tag != "script" or self.script_type is None:
            return
        text = "".join(self.script)
        if self.script_type == "application/ld+json":
            try:
                self.structured.append(json.loads(text))
            except ValueError as error:
                self.errors.append("JSON-LD: " + str(error))
        for match in re.finditer(r'''["']([^"'\s]{1,100})["']\s*:\s*["']([^"']{1,80})["']''', text):
            if EXPLICIT_TIME.fullmatch(match.group(2)):
                self.embedded.append({"key": match.group(1), "raw_value": match.group(2)})
        self.script_type = None


def html_metadata(raw):
    parser = StructuredHTML()
    parser.feed(raw.decode("utf-8", errors="replace"))
    return {"canonical_urls": parser.canonical, "meta": parser.meta,
            "json_ld": parser.structured,
            "structured_timestamps": explicit_timestamps(parser.structured, "$.json_ld"),
            "meta_timestamps": explicit_timestamps(parser.meta, "$.meta"),
            "embedded_explicit_timestamps": parser.embedded, "parse_errors": parser.errors}


def latest_snapshot(dataset):
    candidates = []
    for path in (dataset / "runs").glob("*/run.json"):
        value = json.loads(path.read_text())
        if value.get("snapshot_valid"):
            candidates.append((value["finished_at_utc"], path, value))
    if not candidates:
        return {}, None
    _, path, value = max(candidates, key=lambda x: (x[0], str(x[1])))
    return value["snapshot"], {"path": str(path.relative_to(dataset)), "sha256": digest(path.read_bytes())}


def capture(dataset=DATASET, fetch=http_get):
    dataset = Path(dataset).resolve()
    if not dataset.is_relative_to(HERE) or dataset == HERE:
        raise ValueError("dataset must be a subdirectory of the isolated collector directory")
    dataset.mkdir(parents=True, exist_ok=True)
    runs = dataset / "runs"
    runs.mkdir(exist_ok=True)
    previous, previous_ref = latest_snapshot(dataset)
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "_" + uuid.uuid4().hex[:12]
    staging = dataset / (".pending_" + run_id)
    staging.mkdir()
    raw_dir = staging / "raw"
    raw_dir.mkdir()
    deadline = time.monotonic() + RUN_SECONDS
    run = {"schema": "ESPN_RSS_PROSPECTIVE_EVIDENCE_V1", "run_id": run_id,
           "started_at_utc": utc_now(), "previous_snapshot": previous_ref,
           "snapshot_valid": False, "snapshot": {}, "items": [], "errors": [],
           "timezone_decision": "NONE", "timezone_conversion_performed": False,
           "production_files_changed": "NONE", "database_writes": "NONE"}

    def request(kind, url, token):
        # Also contain unexpected transport exceptions supplied by a test/custom fetcher.
        started = utc_now()
        try:
            body, record = fetch(url, deadline)
        except Exception as error:
            body, record = b"", {"url": url, "started_at_utc": started, "ended_at_utc": utc_now(),
                                 "status": None, "complete": False, "error": repr(error), "headers": []}
        relative = "raw/" + token + "_" + kind + ".body"
        write_new(staging / relative, body)
        record = {**record, "kind": kind, "artifact_path": relative,
                  "sha256": digest(body), "bytes": len(body)}
        write_json(staging / (relative + ".request.json"), record)
        return body, record

    raw, rss_request = request("rss", RSS_URL, "feed")
    run["rss_request"] = rss_request
    try:
        if rss_request.get("error") or not rss_request.get("complete"):
            raise ValueError("RSS request failed or response incomplete")
        parsed, duplicates, occurrences = parse_rss(raw)
        run["duplicate_identical_occurrences"] = duplicates
        run["rss_item_occurrences"] = occurrences
        jobs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for item in parsed:
                observation = {"guid": item["guid"], "article_url": item["link"], "title": item["title"],
                               "description": item["description"], "raw_pubDate": item["pubDate"],
                               "classification": classification(item, previous), "content_id": None,
                               "previous_raw_pubDate": previous.get(item["guid"], {}).get("pubDate"),
                               "requests": {}, "metadata": {}, "errors": []}
                run["items"].append(observation)
                try:
                    observation["content_id"] = resolve_id(item)
                except ValueError as error:
                    observation["errors"].append(str(error))
                if observation["classification"] == "UNCHANGED":
                    continue
                token = digest(item["guid"].encode())
                if observation["content_id"]:
                    jobs.append((observation, "api", pool.submit(request, "api", API_BASE + observation["content_id"], token)))
                else:
                    observation["errors"].append("API skipped: no unambiguous content ID")
                if permitted_url(item["link"]):
                    jobs.append((observation, "html", pool.submit(request, "html", item["link"], token)))
                else:
                    observation["errors"].append("HTML skipped: no allowlisted article URL")
            for observation, kind, future in jobs:
                body, record = future.result()
                observation["requests"][kind] = record
                if record.get("error") or not record.get("complete"):
                    observation["errors"].append(kind + ": " + str(record.get("error", "incomplete response")))
                    continue
                try:
                    metadata = api_metadata(body, observation["content_id"]) if kind == "api" else html_metadata(body)
                    observation["metadata"][kind] = metadata
                    if metadata.get("parse_errors"):
                        observation["errors"].extend(metadata["parse_errors"])
                except Exception as error:
                    observation["errors"].append(kind + " metadata: " + str(error))
        run["snapshot"] = {i["guid"]: i for i in parsed}
        run["snapshot_valid"] = True
    except (ValueError, ET.ParseError) as error:
        run["errors"].append(str(error))
    run["finished_at_utc"] = utc_now()
    run["status"] = ("FAIL" if not run["snapshot_valid"] else
                     "PARTIAL" if any(i["errors"] for i in run["items"]) else "PASS")
    write_json(staging / "run.json", run)
    manifest = {str(p.relative_to(staging)): {"sha256": digest(p.read_bytes()), "bytes": p.stat().st_size}
                for p in sorted(staging.rglob("*")) if p.is_file()}
    write_json(staging / "manifest.json", manifest)
    # A run becomes visible to snapshot discovery only after every artifact is durable.
    destination = runs / run_id
    os.rename(staging, destination)
    directory_fd = os.open(runs, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return destination, run


def summary(run):
    items = run["items"]
    values = {"COLLECTOR_STATUS": run["status"], "RSS_ITEMS": len(items),
              "NEW_GUIDS": sum(i["classification"] == "NEW_GUID" for i in items),
              "PUBDATE_CHANGES": sum(i["classification"] == "PUBDATE_CHANGED" for i in items),
              "UNCHANGED_ITEMS": sum(i["classification"] == "UNCHANGED" for i in items)}
    for kind in ("api", "html"):
        values[kind.upper() + "_CAPTURES"] = sum(
            r.get("complete", False) and not r.get("error")
            for i in items for k, r in i["requests"].items() if k == kind)
    values.update(PRODUCTION_FILES_CHANGED="NONE", DATABASE_WRITES="NONE", TIMEZONE_DECISION="NONE")
    return "\n".join(str(k) + "=" + str(v) for k, v in values.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True, help="perform exactly one manual capture")
    parser.parse_args()
    path, run = capture()
    print(summary(run))
    print("EVIDENCE_RUN=" + str(path))
    return 0 if run["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
