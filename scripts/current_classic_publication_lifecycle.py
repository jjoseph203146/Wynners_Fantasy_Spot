#!/usr/bin/env python3
"""Refresh existing evidence and publish a bound pair under the updater lock.

The caller must hold nfl_updater.lock exclusively for the entire handoff.
All evidence is created and checked by the unchanged production validators.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import preflight_offensive_reconciliation_v3 as preflight
from scripts import v3_adaptive_rebind as rebind


def refresh_evidence(root=ROOT):
    root = Path(root)
    proof_path = root / preflight.REBIND_PATH
    matrix = root / preflight.INPUTS["matrix.parquet"]
    adaptive = (root / preflight.INPUTS["adaptive.csv"]).read_bytes()
    audit = (root / preflight.INPUTS["adaptive_audit.json"]).read_bytes()
    if not proof_path.exists():
        # The original direct-binding path remains subject to normal preflight.
        rebind.require(json.loads(audit)["source_hashes"].get(str(matrix))
                       == rebind.digest(matrix.read_bytes()), "direct matrix binding")
        return "DIRECT_BINDING"
    previous = proof_path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(previous)) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    meta = json.loads(files["original/validation.json"])
    # Validate the existing immutable archive under the contract that
    # created it. V2 evidence must remain valid under V2 semantics before
    # it may be transitioned to the current V3 lifecycle contract.
    previous_proof = json.loads(files["proof.json"])
    if previous_proof.get("contract") == "WFS_V3_ADAPTIVE_INPUT_EQUIVALENCE_REBIND_V2":
        import importlib.util

        legacy_rebind_path = (
            root
            / "backups"
            / "adaptive_rebind_completed_game_20260926T104008Z"
            / "v3_adaptive_rebind.py"
        )
        rebind.require(legacy_rebind_path.exists(), "V2 validator unavailable")

        legacy_rebind_sha = rebind.digest(legacy_rebind_path.read_bytes())
        rebind.require(
            legacy_rebind_sha
            == "5a8664ed6aadaad7ca5c61a3334e766f4645fd09a619e16bf9f232f3d2ec85b9",
            "V2 validator hash",
        )

        spec = importlib.util.spec_from_file_location(
            "_v3_adaptive_rebind_v2_validator",
            legacy_rebind_path,
        )
        legacy_rebind = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy_rebind)

        # The immutable V2 module was loaded from a backup directory, so
        # __file__ would otherwise make its ROOT point at /backups.
        # Restore only its filesystem anchors to the real project root;
        # all V2 validation logic and hashes remain unchanged.
        legacy_rebind.ROOT = root
        legacy_rebind.BUILDER = (
            root
            / "scripts"
            / "build_offensive_reconciliation_current_adaptive_team_v1.py"
        )

        legacy_rebind.validate_rebind(
            previous,
            files["current_matrix.parquet"],
            adaptive,
            audit,
            meta["season"],
            meta["week"],
        )
    else:
        rebind.validate_rebind(
            previous,
            files["current_matrix.parquet"],
            adaptive,
            audit,
            meta["season"],
            meta["week"],
        )
    if files["current_matrix.parquet"] == matrix.read_bytes():
        return "CURRENT"

    # Reconstruct only the already validated immutable historical bundle.
    # create_rebind performs the full exact consumed-input equivalence proof.
    with tempfile.TemporaryDirectory(dir=proof_path.parent, prefix=".rebind-") as temp:
        historical = Path(temp) / meta["candidate_id"]
        historical.mkdir()
        for name in ["validation.json", *meta["inputs"]]:
            (historical / name).write_bytes(files["original/" + name])
        candidate = Path(temp) / "adaptive_rebind.zip"
        rebind.create_rebind(historical / "validation.json", matrix, candidate)
        rebind.validate_rebind(candidate.read_bytes(), matrix.read_bytes(), adaptive,
                               audit, meta["season"], meta["week"])
        rebind.require(proof_path.read_bytes() == previous, "concurrent evidence change")
        os.replace(candidate, proof_path)
    return "REFRESHED"


def publish_pair(bundle, root=ROOT):
    """Keep the prior pair if either existing publisher or pair check fails."""
    import pandas as pd

    root, bundle = Path(root), Path(bundle).resolve()
    preflight.load_validated_candidate(bundle / "validation.json")
    names = ["current_unified_stat_forecasts.parquet",
             "current_unified_stat_forecasts_manifest.json",
             "current_unified_fanduel_expectation.parquet",
             "current_unified_fanduel_expectation_manifest.json"]
    directory = root / "data/parquet"
    backup = Path(tempfile.mkdtemp(prefix="classic_pair_", dir=root / "backups"))
    hashes = {}
    for name in names:
        shutil.copy2(directory / name, backup / name)
        hashes[name] = rebind.digest((backup / name).read_bytes())
    (backup / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"PUBLICATION_PAIR_BACKUP={backup}", flush=True)
    try:
        subprocess.run([sys.executable, str(root / "scripts/wfs_stat_forecast_publish_v1.py"),
                        "--source", str(bundle / "unified.parquet"),
                        "--source-binding", str(bundle / "unified.binding.json")], check=True)
        subprocess.run(["bash", str(root / "scripts/run_current_stage24_refresh.sh"),
                        "--publish-only"], check=True)
        for name in (names[0], names[2]):
            manifest = json.loads((directory / name.replace(".parquet", "_manifest.json")).read_bytes())
            rebind.require(rebind.digest((directory / name).read_bytes()) == manifest["current_sha256"],
                           "published manifest hash")
        stat = pd.read_parquet(directory / names[0])
        stage = pd.read_parquet(directory / names[2])
        stat = stat.loc[stat.entity_type.eq("OFFENSE_PLAYER")]
        stage = stage.loc[stage.position.isin(["QB", "RB", "WR", "TE"])]
        from stage24_solver_attachment import canon_team
        def identities(frame, id_column):
            return set(zip(frame[id_column], frame.game_id, frame.team.map(canon_team)))
        rebind.require(identities(stat, "player_id") == identities(stage, "entity_id"),
                       "publication pair identity")
    except BaseException:
        for name in names:
            temporary = directory / ("." + name + ".rollback")
            shutil.copy2(backup / name, temporary)
            os.replace(temporary, directory / name)
        raise
    print("PUBLICATION_PAIR_STATUS=PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["refresh-evidence", "publish-pair"])
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    if args.action == "refresh-evidence":
        print("ADAPTIVE_REBIND_STATUS=" + refresh_evidence())
    else:
        if args.bundle is None:
            parser.error("publish-pair requires --bundle")
        publish_pair(args.bundle)


if __name__ == "__main__":
    main()
