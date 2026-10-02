"""Immutable content-addressed publication restricted to the isolated shadow root."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .core import CONTRACT, SAFETY, build, canonical, digest, require

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts"


def code_hashes():
    paths = list(ROOT.glob("*.py")) + [ROOT.parent / "fanduel_injury_ingest.py"]
    return {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def bundle(fixture):
    result = build(fixture)
    payloads = {name + ".json": canonical({"safety": SAFETY, "rows": rows}) for name, rows in result.items()}
    payloads["input.json"] = canonical(fixture)
    payloads["validation.json"] = canonical({"safety": SAFETY, "status": "SHADOW_ONLY",
        "accepted": sum(r["eligibility_status"] == "ELIGIBLE" for r in result["claims"]),
        "quarantined": len(result["quarantine"]), "external_connectors": []})
    manifest = {"contract": CONTRACT, "schema_version": 1, "safety": SAFETY,
                "cutoff_utc": fixture["cutoff_utc"], "game": fixture["game"],
                "identity_sha256": digest(fixture["identity"]), "code_sha256": code_hashes(),
                "input_mode": "OFFLINE_FIXTURE", "source_approvals": ["SYNTHETIC_FIXTURE_ONLY"],
                "freshness_policy": "EXPLICIT_HORIZON_CAPPED_AT_KICKOFF_V1",
                "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                          for name, data in sorted(payloads.items())}}
    manifest["run_id"] = digest(manifest)
    payloads["manifest.json"] = canonical(manifest)
    return payloads


def verify(path):
    path = Path(path)
    require(not path.is_symlink(), "SYMLINK_BUNDLE")
    manifest = json.loads((path / "manifest.json").read_bytes())
    require(manifest["contract"] == CONTRACT and manifest["safety"] == SAFETY, "MANIFEST_CONTRACT")
    unsigned = dict(manifest)
    run_id = unsigned.pop("run_id")
    require(digest(unsigned) == run_id, "MANIFEST_HASH")
    expected = {"events.json", "claims.json", "consensus.json", "quarantine.json", "input.json", "validation.json"}
    require(set(manifest["files"]) == expected, "ARTIFACT_SET")
    require({p.name for p in path.iterdir()} == expected | {"manifest.json"}, "BUNDLE_FILE_SET")
    for name, metadata in manifest["files"].items():
        file = path / name
        require(not file.is_symlink() and file.is_file(), "ARTIFACT_TYPE")
        data = file.read_bytes()
        require(len(data) == metadata["bytes"] and hashlib.sha256(data).hexdigest() == metadata["sha256"], "ARTIFACT_HASH")
    return manifest


def publish(fixture):
    """No configurable destination, overwrite, current pointer, or production path."""
    payloads = bundle(fixture)
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
        os.rename(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return destination
