#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import os
import py_compile
import shutil
import sys


ROOT = Path("/home/mwynn/nfl_data_engine")
TARGET = ROOT / "production_projection.py"

EXPECTED_SHA256 = (
    "5059834e29083ceb634d58f4f605524995ef03620d01f1f8e74a762080813fb8"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)

    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly 1 match, found {count}"
        )

    return text.replace(old, new, 1)


def main():
    print("=== NFL-2026-FEEDBACK-1P INSTALLER ===")

    if not TARGET.exists():
        raise RuntimeError(
            f"Missing target: {TARGET}"
        )

    current_sha = sha256(TARGET)

    print(f"CURRENT_SHA256={current_sha}")

    if current_sha != EXPECTED_SHA256:
        raise RuntimeError(
            "FAIL_CLOSED: production_projection.py "
            "does not match frozen baseline."
        )

    original = TARGET.read_text()

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    backup = TARGET.with_name(
        f"production_projection.py.pre-feedback-1p-{stamp}.bak"
    )

    shutil.copy2(
        TARGET,
        backup,
    )

    print(f"BACKUP={backup}")
    print(f"BACKUP_SHA256={sha256(backup)}")

    text = original

    # ---------------------------------------------------------
    # 1. CONFIGURATION
    # ---------------------------------------------------------

    old = '''TRAINING_SEASONS = [
    2023,
    2024,
    2025,
]
'''

    new = '''BASE_TRAINING_SEASONS = [
    2023,
    2024,
    2025,
]

ROLLING_TRAINING_START_SEASON = 2026
'''

    text = replace_once(
        text,
        old,
        new,
        "CONFIG",
    )

    # ---------------------------------------------------------
    # 2. TRAINING FUNCTION SIGNATURE
    # ---------------------------------------------------------

    old = '''def load_training_matrix(
    core,
):
'''

    new = '''def load_training_matrix(
    core,
    target_season,
    target_week,
):
'''

    text = replace_once(
        text,
        old,
        new,
        "TRAINING_SIGNATURE",
    )

    # ---------------------------------------------------------
    # 3. TEMPORAL TRAINING FILTER
    # ---------------------------------------------------------

    old = '''    matrix[
        "season"
    ] = pd.to_numeric(
        matrix[
            "season"
        ],
        errors="coerce",
    )

    matrix = matrix[
        matrix[
            "season"
        ].isin(
            TRAINING_SEASONS
        )
    ].copy()
'''

    new = '''    matrix[
        "season"
    ] = pd.to_numeric(
        matrix[
            "season"
        ],
        errors="coerce",
    )

    matrix[
        "week"
    ] = pd.to_numeric(
        matrix[
            "week"
        ],
        errors="coerce",
    )

    base_mask = matrix[
        "season"
    ].isin(
        BASE_TRAINING_SEASONS
    )

    rolling_mask = (
        (
            matrix[
                "season"
            ]
            >=
            ROLLING_TRAINING_START_SEASON
        )
        &
        (
            (
                matrix[
                    "season"
                ]
                <
                target_season
            )
            |
            (
                (
                    matrix[
                        "season"
                    ]
                    ==
                    target_season
                )
                &
                (
                    matrix[
                        "week"
                    ]
                    <
                    target_week
                )
            )
        )
    )

    matrix = matrix[
        base_mask
        |
        rolling_mask
    ].copy()
'''

    text = replace_once(
        text,
        old,
        new,
        "TEMPORAL_FILTER",
    )

    # ---------------------------------------------------------
    # 4. TRAINING REPORT
    # ---------------------------------------------------------

    old = '''    print(
        f"Training seasons: "
        f"{TRAINING_SEASONS}"
    )

    print(
        f"Training rows: "
        f"{len(matrix)}"
    )
'''

    new = '''    print(
        f"Base training seasons: "
        f"{BASE_TRAINING_SEASONS}"
    )

    print(
        f"Rolling training start season: "
        f"{ROLLING_TRAINING_START_SEASON}"
    )

    print(
        f"Target season/week: "
        f"{target_season}/{target_week}"
    )

    print(
        f"Training rows: "
        f"{len(matrix)}"
    )
'''

    text = replace_once(
        text,
        old,
        new,
        "TRAINING_REPORT",
    )

    # ---------------------------------------------------------
    # 5. TARGET SEASON/WEEK AUTHORITY
    # ---------------------------------------------------------

    old = '''    return current


# =========================================================
# RIDGE MODEL
# =========================================================
'''

    new = '''    return current


def resolve_target_season_week(
    current,
):

    required = [
        "season",
        "week",
    ]

    missing = [
        column
        for column in required
        if column not in current.columns
    ]

    if missing:

        raise RuntimeError(
            "Current inference pool missing "
            "target authority columns: "
            +
            ", ".join(
                missing
            )
        )

    pairs = (
        current[
            [
                "season",
                "week",
            ]
        ]
        .copy()
    )

    pairs[
        "season"
    ] = pd.to_numeric(
        pairs[
            "season"
        ],
        errors="coerce",
    )

    pairs[
        "week"
    ] = pd.to_numeric(
        pairs[
            "week"
        ],
        errors="coerce",
    )

    if (
        pairs[
            [
                "season",
                "week",
            ]
        ]
        .isna()
        .any()
        .any()
    ):

        raise RuntimeError(
            "Current inference pool contains "
            "NULL/non-numeric season or week."
        )

    pairs = (
        pairs[
            [
                "season",
                "week",
            ]
        ]
        .drop_duplicates()
    )

    if len(
        pairs
    ) != 1:

        raise RuntimeError(
            "Current inference pool must contain "
            "exactly one season/week pair; "
            f"found {len(pairs)}."
        )

    target_season = int(
        pairs.iloc[
            0
        ][
            "season"
        ]
    )

    target_week = int(
        pairs.iloc[
            0
        ][
            "week"
        ]
    )

    if target_week < 1:

        raise RuntimeError(
            "Target week must be >= 1."
        )

    return (
        target_season,
        target_week,
    )


# =========================================================
# RIDGE MODEL
# =========================================================
'''

    text = replace_once(
        text,
        old,
        new,
        "TARGET_AUTHORITY",
    )

    # ---------------------------------------------------------
    # 6. MAIN HEADER
    # ---------------------------------------------------------

    old = '''    print(
        "Training window: "
        "2023 + 2024 + 2025"
    )

    print(
        "2026 is forward inference only."
    )

    print(
        "No 2026 realized results are used "
        "for fitting or tuning."
    )
'''

    new = '''    print(
        "Training window: 2023-2025 base "
        "+ rolling completed prior weeks "
        "from 2026 onward."
    )

    print(
        "Current-season realized rows are "
        "eligible only when their season/week "
        "is strictly before the target "
        "season/week."
    )
'''

    text = replace_once(
        text,
        old,
        new,
        "MAIN_HEADER",
    )

    # ---------------------------------------------------------
    # 7. MAIN LOAD ORDER
    # ---------------------------------------------------------

    old = '''    core = load_core_features()

    historical = load_training_matrix(
        core
    )

    current = load_current_features(
        core
    )
'''

    new = '''    core = load_core_features()

    current = load_current_features(
        core
    )

    (
        target_season,
        target_week,
    ) = resolve_target_season_week(
        current
    )

    print()

    print(
        f"Resolved inference target: "
        f"season={target_season} "
        f"week={target_week}"
    )

    historical = load_training_matrix(
        core,
        target_season,
        target_week,
    )
'''

    text = replace_once(
        text,
        old,
        new,
        "MAIN_LOAD_ORDER",
    )

    # ---------------------------------------------------------
    # 8. FINAL CONTRACT TEXT
    # ---------------------------------------------------------

    old = '''    print(
        "Historical fit = "
        "2023-2025 completed games."
    )

    print(
        "Live inference = "
        "2026 current slate only."
    )
'''

    new = '''    print(
        "Historical fit = 2023-2025 base "
        "plus rolling prior-week results "
        "from 2026 onward."
    )

    print(
        f"Live inference target = "
        f"{target_season} week {target_week}."
    )
'''

    text = replace_once(
        text,
        old,
        new,
        "FINAL_CONTRACT_TEXT",
    )

    # ---------------------------------------------------------
    # STATIC SAFETY ASSERTIONS
    # ---------------------------------------------------------

    if "\nTRAINING_SEASONS = [\n" in text:
        raise RuntimeError(
            "FAIL_CLOSED: legacy TRAINING_SEASONS "
            "definition remains."
        )

    required_tokens = [
        "BASE_TRAINING_SEASONS",
        "ROLLING_TRAINING_START_SEASON",
        "resolve_target_season_week",
        "target_season",
        "target_week",
        "base_mask",
        "rolling_mask",
    ]

    for token in required_tokens:
        if token not in text:
            raise RuntimeError(
                f"FAIL_CLOSED: missing token {token}"
            )

    tmp = TARGET.with_name(
        "production_projection.py.feedback-1p.tmp"
    )

    tmp.write_text(
        text
    )

    try:
        py_compile.compile(
            str(tmp),
            doraise=True,
        )
    except Exception:
        tmp.unlink(
            missing_ok=True
        )
        raise

    new_sha = sha256(
        tmp
    )

    print(
        f"CANDIDATE_SHA256={new_sha}"
    )

    os.replace(
        tmp,
        TARGET,
    )

    installed_sha = sha256(
        TARGET
    )

    if installed_sha != new_sha:
        raise RuntimeError(
            "FAIL_CLOSED: installed SHA mismatch."
        )

    print(
        f"INSTALLED_SHA256={installed_sha}"
    )

    print(
        "NFL_2026_FEEDBACK_1P_INSTALL_STATUS=PASS"
    )


if __name__ == "__main__":
    main()
