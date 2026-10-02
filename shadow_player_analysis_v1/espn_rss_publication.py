"""ESPN capture bundles using the foundation's canonical manifest and verifier."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .core import CONTRACT, SAFETY, canonical, digest, require
from .espn_nfl_rss_v1 import SOURCE
from .publication import code_hashes, verify

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "espn_nfl_rss_v1_artifacts"


def bundle(capture):
    require(capture["safety"] == SAFETY and capture["retrieval"]["source"] == SOURCE, "CAPTURE_CONTRACT")
    require(capture["claims"] == [] and capture["consensus"] == [], "CAPTURE_ONLY")
    payloads = {name + ".json": canonical(dict(safety=SAFETY, rows=capture[name]))
                for name in ("events", "quarantine", "claims", "consensus")}
    payloads["input.json"] = canonical(capture)
    payloads["validation.json"] = canonical(dict(safety=SAFETY, status="SHADOW_ONLY",
        accepted=0, normalized_events=len(capture["events"]), quarantined=len(capture["quarantine"]),
        external_connectors=[SOURCE], article_crawling=False))
    manifest = dict(contract=CONTRACT, schema_version=1, safety=SAFETY, source=SOURCE,
                    input_mode="RSS_CAPTURE_ONLY", retrieval=capture["retrieval"],
                    code_sha256=code_hashes(), source_approvals=["ESPN_RSS_SHADOW_CAPTURE_ONLY"],
                    files={name: dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
                           for name, data in sorted(payloads.items())})
    manifest["run_id"] = digest(manifest)
    payloads["manifest.json"] = canonical(manifest)
    return payloads


def publish(capture):
    """Fixed isolated destination, atomic rename, no historical overwrite or pointer."""
    payloads = bundle(capture)
    require(not OUTPUT.is_symlink() and OUTPUT.resolve() == ROOT / "espn_nfl_rss_v1_artifacts", "OUTPUT_PATH")
    OUTPUT.mkdir(exist_ok=True)
    destination = OUTPUT / json.loads(payloads["manifest.json"])["run_id"]
    if destination.exists() or destination.is_symlink():
        verify(destination)
        require(all((destination / n).read_bytes() == b for n, b in payloads.items()), "IMMUTABLE_RUN_CONFLICT")
        return destination
    stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=OUTPUT))
    try:
        for name, data in payloads.items():
            with (stage / name).open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        verify(stage)
        os.rename(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return destination
