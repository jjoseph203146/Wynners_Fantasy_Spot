"""Canonical, content-addressed, immutable shadow publication; no current pointer."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from shadow_player_analysis_v1.core import SAFETY, canonical, digest, require
from .core import CONTRACT, build

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts"
FILES = {"source_events.json", "classified_evidence.json", "entity_links.json", "quarantine.json",
         "source_quality.json", "input.json"}


def bundle(inputs):
    result = build(**inputs)
    payloads = {name + ".json": canonical(dict(safety=SAFETY, rows=rows)) for name, rows in result.items()}
    payloads["input.json"] = canonical(inputs)
    dependencies = [*ROOT.glob("*.py"), ROOT.parent / "fanduel_injury_ingest.py",
                    ROOT.parent / "shadow_player_analysis_v1" / "core.py"]
    manifest = dict(contract=CONTRACT, safety=SAFETY, schema_version="1.2", input_mode="OFFLINE",
                    identity_sha256=digest(inputs["identity"]), schedule_sha256=digest(inputs["schedule"]),
                    cutoff_utc=inputs["cutoff"], source_approvals=inputs["approvals"],
                    code_sha256={str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(dependencies)},
                    files={name: dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
                           for name, data in sorted(payloads.items())})
    manifest["run_id"] = digest(manifest)
    payloads["manifest.json"] = canonical(manifest)
    return payloads


def verify(path):
    path = Path(path)
    require(not path.is_symlink(), "SYMLINK_BUNDLE")
    require(not (path / "manifest.json").is_symlink(), "SYMLINK_MANIFEST")
    manifest = json.loads((path / "manifest.json").read_bytes())
    require(manifest["contract"] == CONTRACT and manifest["safety"] == SAFETY, "MANIFEST_CONTRACT")
    unsigned = dict(manifest)
    require(digest({k: v for k, v in unsigned.items() if k != "run_id"}) == manifest["run_id"], "MANIFEST_HASH")
    require(set(manifest["files"]) == FILES, "ARTIFACT_SET")
    require({p.name for p in path.iterdir()} == FILES | {"manifest.json"}, "BUNDLE_FILE_SET")
    for name, meta in manifest["files"].items():
        file = path / name
        require(file.is_file() and not file.is_symlink(), "ARTIFACT_TYPE")
        data = file.read_bytes()
        require(len(data) == meta["bytes"] and hashlib.sha256(data).hexdigest() == meta["sha256"], "ARTIFACT_HASH")
    return manifest


def publish(inputs):
    payloads = bundle(inputs)
    manifest = json.loads(payloads["manifest.json"])
    require(not OUTPUT.is_symlink() and OUTPUT.resolve() == ROOT / "artifacts", "OUTPUT_PATH")
    OUTPUT.mkdir(exist_ok=True)
    destination = OUTPUT / manifest["run_id"]
    if destination.exists():
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
        try:
            os.rename(stage, destination)
        except OSError:
            if not destination.exists():
                raise
            verify(destination)
            require(all((destination / n).read_bytes() == b for n, b in payloads.items()), "IMMUTABLE_RUN_CONFLICT")
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return destination
