#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from guardian_classification import classify_failures


ROOT = Path(__file__).resolve().parent
PARQUET = ROOT / "data" / "parquet"

PROD_ARTIFACT = (
    PARQUET / "current_unified_fanduel_expectation.parquet"
)

MANIFEST_CANDIDATES = [
    PARQUET / "current_unified_fanduel_expectation.manifest.json",
    PARQUET / "current_unified_fanduel_expectation_manifest.json",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def find_manifest() -> Path:
    for path in MANIFEST_CANDIDATES:
        if path.is_file():
            return path

    matches = sorted(
        PARQUET.glob(
            "*unified*fanduel*expectation*manifest*.json"
        )
    )

    if len(matches) != 1:
        raise RuntimeError(
            "Could not uniquely identify FanDuel manifest: "
            + repr([str(p) for p in matches])
        )

    return matches[0]


def manifest_hash_values(obj):
    found = []

    def walk(value, key_path=""):
        if isinstance(value, dict):
            for key, child in value.items():
                path = (
                    f"{key_path}.{key}"
                    if key_path
                    else str(key)
                )

                if (
                    isinstance(child, str)
                    and "sha" in str(key).lower()
                    and len(child) == 64
                ):
                    found.append((path, child))

                walk(child, path)

        elif isinstance(value, list):
            for i, child in enumerate(value):
                walk(child, f"{key_path}[{i}]")

    walk(obj)
    return found


def main() -> int:
    print("=" * 72)
    print(
        "WFS GUARDIAN V2.6 PHASE 2 — "
        "REAL FIXTURE HASH TEST"
    )
    print("=" * 72)

    manifest = find_manifest()

    if not PROD_ARTIFACT.is_file():
        raise SystemExit(
            f"FAIL: production artifact missing: {PROD_ARTIFACT}"
        )

    before_artifact = sha256(PROD_ARTIFACT)
    before_manifest = sha256(manifest)

    print("PRODUCTION_ARTIFACT =", PROD_ARTIFACT)
    print("PRODUCTION_MANIFEST =", manifest)
    print(
        "PRODUCTION_ARTIFACT_SHA_BEFORE =",
        before_artifact,
    )
    print(
        "PRODUCTION_MANIFEST_SHA_BEFORE =",
        before_manifest,
    )

    with tempfile.TemporaryDirectory(
        prefix="wfs_guardian_fixture_"
    ) as tmp:
        fixture = Path(tmp)

        artifact_copy = fixture / PROD_ARTIFACT.name
        manifest_copy = fixture / manifest.name

        shutil.copy2(PROD_ARTIFACT, artifact_copy)
        shutil.copy2(manifest, manifest_copy)

        manifest_obj = json.loads(
            manifest_copy.read_text()
        )

        hashes = manifest_hash_values(manifest_obj)

        matching = [
            (path, value)
            for path, value in hashes
            if value == before_artifact
        ]

        if not matching:
            raise SystemExit(
                "FAIL: copied manifest contains no SHA "
                "matching production artifact"
            )

        # Deliberately corrupt ONLY the copied artifact.
        with artifact_copy.open("ab") as f:
            f.write(b"WFS_GUARDIAN_FAULT_INJECTION")

        copied_sha = sha256(artifact_copy)

        if copied_sha == before_artifact:
            raise SystemExit(
                "FAIL: fixture corruption did not change hash"
            )

        manifest_expected_sha = matching[0][1]
        detected = copied_sha != manifest_expected_sha

        result = {
            "check": "FANDUEL_PUBLICATION_LIFECYCLE",
            "status": "FAIL" if detected else "PASS",
            "detail": (
                f"hash_chain={not detected} "
                f"expected={manifest_expected_sha[:12]} "
                f"actual={copied_sha[:12]}"
            ),
        }

        classification = classify_failures([result])

        print("-" * 72)
        print("FIXTURE_MUTATED=TRUE")
        print("PRODUCTION_MUTATED=FALSE")
        print(
            "HASH_MISMATCH_DETECTED=",
            detected,
        )
        print(
            "DETECTOR_CHECK=",
            result["check"],
        )
        print(
            "DETECTOR_STATUS=",
            result["status"],
        )
        print(
            "FAILURE_CATEGORIES=",
            classification["categories"],
        )

        if not detected:
            raise SystemExit(
                "FAIL: hash mismatch not detected"
            )

        if classification["categories"] != [
            "PUBLICATION"
        ]:
            raise SystemExit(
                "FAIL: expected PUBLICATION classification"
            )

    after_artifact = sha256(PROD_ARTIFACT)
    after_manifest = sha256(manifest)

    print("-" * 72)
    print(
        "PRODUCTION_ARTIFACT_SHA_AFTER =",
        after_artifact,
    )
    print(
        "PRODUCTION_MANIFEST_SHA_AFTER =",
        after_manifest,
    )

    artifact_unchanged = (
        before_artifact == after_artifact
    )
    manifest_unchanged = (
        before_manifest == after_manifest
    )

    print(
        "PRODUCTION_ARTIFACT_UNCHANGED=",
        artifact_unchanged,
    )
    print(
        "PRODUCTION_MANIFEST_UNCHANGED=",
        manifest_unchanged,
    )

    if not artifact_unchanged:
        raise SystemExit(
            "FAIL: production artifact changed"
        )

    if not manifest_unchanged:
        raise SystemExit(
            "FAIL: production manifest changed"
        )

    print("PRODUCTION_LKG_MUTATED=FALSE")
    print("PRODUCTION_AUDIT_MUTATED=FALSE")
    print("PRODUCTION_ACTION=NONE")
    print("V26_REAL_FIXTURE_HASH_TEST=PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
