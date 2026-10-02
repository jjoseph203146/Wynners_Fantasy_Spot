#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse
import hashlib
import html
import json
import re
import sys
import time

import requests


ROOT = Path("/home/mwynn/nfl_data_engine")
AUDIT = ROOT / "data/audits/game_environment_stage25"

CLASSIC = ROOT / "fanduel_nfl_ui_solver.py"
SHOWDOWN = ROOT / "fanduel_nfl_showdown_solver_v1.py"
HELPER = ROOT / "stage24_solver_attachment.py"
STAGE24 = ROOT / "data/parquet/current_unified_fanduel_expectation.parquet"

Z_M = AUDIT / (
    "stage25c_h_d_c5_z_m_remaining_current_authority_research_planning_"
    "20260913T010002Z.json"
)

Z_O = AUDIT / (
    "stage25c_h_d_c5_z_o_wave_1_classification_consolidation_"
    "20260913T010853Z.json"
)


EXPECTED = {
    CLASSIC:
        "a86abf91315e4c7f1b7f98273f545baf3ddb73f05f1b00129a57a162c60efc4d",

    SHOWDOWN:
        "57d1118dc5ca7c0204b977151c2265430762752294dc41dd40583d525ea2dd56",

    HELPER:
        "b2364abd2139341c6f29eadb9b5d9db88a9f4a0d744f616ccb9c3e019b5a0942",

    STAGE24:
        "4ca010fe4b6b93d6d981e735a5fe5caef7d2e682cf44a16e1369fbc8ff1b558f",

    Z_M:
        "6c68d6ef1e80f8e9ce2701db59348c5d237f4b4cf9f067e62e4163c07809a232",

    Z_O:
        "801d3702fe7dbb1cec0145d8c129719f38865c406d639b2704d8d10e676c1e3b",
}


# Previously validated current-host corrections.
# These are exact routing overrides only — never roof/surface authority.
EXPLICIT_CURRENT_SITE_OVERRIDES = {
    "Q583085": [
        "https://northweststadium.com/",
    ],

    "Q320454": [
        "https://estadiobanorte.com.mx/",
    ],

    "Q1100756": [
        "https://huntingtonbankfield.com/",
    ],
}


KEYWORDS = [
    "roof",
    "retractable roof",
    "fixed roof",
    "open-air",
    "open air",
    "outdoor",
    "indoors",
    "indoor",
    "dome",
    "canopy",
    "covered",

    "surface",
    "playing surface",
    "field surface",
    "natural grass",
    "grass",
    "bermuda",
    "ryegrass",
    "artificial",
    "synthetic",
    "turf",
    "fieldturf",
    "hybrid",
]


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_json(path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def fail(reason):
    print()
    print("=" * 120)
    print("STAGE25C_H_D_C5_Z_P_RESULT=FAIL_CLOSED")
    print(f"FAIL_REASON={reason}")
    print("WAVES_2_TO_5_EVIDENCE_BATCH_FROZEN=FALSE")
    print("NETWORK_FETCH_PERFORMED=TRUE")
    print("CLASSIFICATION_PERFORMED=FALSE")
    print("AUTHORITY_GRANTED=FALSE")
    print("FACT_PROMOTION_ALLOWED=FALSE")
    print("REGISTRY_PROMOTED=FALSE")
    print("DB_EDITED=FALSE")
    print("PRODUCTION_EDITED=FALSE")
    print("SOLVER_EDITED=FALSE")
    print("SERVICE_CHANGED=FALSE")
    print("SERVICE_RESTARTED=FALSE")
    print("STAGE25_PRODUCTION_INFLUENCE_ALLOWED=FALSE")
    print("=" * 120)
    sys.exit(1)


def normalize_host(host):
    host = (
        host
        or
        ""
    ).strip().lower()

    if host.startswith("www."):
        host = host[4:]

    return host


def clean_html(raw):
    text = re.sub(
        r"(?is)<script.*?>.*?</script>",
        " ",
        raw,
    )

    text = re.sub(
        r"(?is)<style.*?>.*?</style>",
        " ",
        text,
    )

    text = re.sub(
        r"(?is)<[^>]+>",
        " ",
        text,
    )

    text = html.unescape(text)

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def page_title(raw):
    m = re.search(
        r"(?is)<title[^>]*>(.*?)</title>",
        raw,
    )

    if not m:
        return None

    return clean_html(
        m.group(1)
    )[:300]


def keyword_snippets(text, radius=300):
    results = []
    seen = set()
    lowered = text.lower()

    for keyword in KEYWORDS:
        needle = keyword.lower()
        start = 0

        while True:
            pos = lowered.find(
                needle,
                start,
            )

            if pos < 0:
                break

            left = max(
                0,
                pos - radius,
            )

            right = min(
                len(text),
                pos + len(needle) + radius,
            )

            snippet = re.sub(
                r"\s+",
                " ",
                text[left:right],
            ).strip()

            signature = (
                needle,
                snippet,
            )

            if signature not in seen:
                seen.add(signature)

                results.append({
                    "keyword":
                        keyword,

                    "snippet":
                        snippet,
                })

            if len(results) >= 40:
                return results

            start = (
                pos
                +
                len(needle)
            )

    return results


print("=" * 120)
print("WFS STAGE25C-H-D-C5-Z-P")
print("REMAINING RESEARCH WAVES 2-5")
print("BATCH OFFICIAL-SOURCE EVIDENCE ACQUISITION")
print("ANALYSIS_ONLY=TRUE")
print("NETWORK_FETCH_PERFORMED=TRUE")
print("SOURCE_PLAN=FROZEN_Z_M")
print("SOURCE_STATE=FROZEN_Z_O")
print("WIKIDATA_P856_LOCATOR_ONLY=TRUE")
print("OFFICIAL_SITE_CONTENT_REQUIRED_FOR_EVIDENCE=TRUE")
print("CROSS_DOMAIN_REDIRECT_AUTO_TRUST=FALSE")
print("CLASSIFICATION_PERFORMED=FALSE")
print("AUTHORITY_GRANTED=FALSE")
print("FACT_PROMOTION_ALLOWED=FALSE")
print("REGISTRY_PROMOTED=FALSE")
print("STAGE25_PRODUCTION_INFLUENCE_ALLOWED=FALSE")
print("=" * 120)


# ============================================================
# 1. PROTECTED ARTIFACT GATE
# ============================================================

print()
print("[1] TRUE IMMUTABLE ARTIFACT GATE")
print("=" * 120)

for path, expected in EXPECTED.items():

    if not path.exists():
        fail(
            f"MISSING_PROTECTED_ARTIFACT:{path}"
        )

    actual = sha256(path)

    ok = (
        actual
        ==
        expected
    )

    print(
        f"{path}|"
        f"EXPECTED={expected}|"
        f"ACTUAL={actual}|"
        f"STATUS={'OK' if ok else 'FAIL'}"
    )

    if not ok:
        fail(
            f"PROTECTED_SHA_MISMATCH:{path}"
        )


# ============================================================
# 2. VALIDATE FROZEN STATE
# ============================================================

print()
print("=" * 120)
print("[2] VALIDATE FROZEN REMAINING-RESEARCH STATE")
print("=" * 120)

zm = load_json(Z_M)
zo = load_json(Z_O)


if (
    zm.get("contract")
    !=
    "WFS_STAGE25C_H_D_C5_Z_M_REMAINING_CURRENT_AUTHORITY_RESEARCH_PLANNING_V1"
):
    fail(
        "Z_M_CONTRACT_MISMATCH"
    )


if (
    zo.get("contract")
    !=
    "WFS_STAGE25C_H_D_C5_Z_O_WAVE_1_CLASSIFICATION_CONSOLIDATION_V1"
):
    fail(
        "Z_O_CONTRACT_MISMATCH"
    )


if zo.get(
    "wave_resolved_count"
) != 0:
    fail(
        "EXPECTED_WAVE1_ZERO_RESOLVED"
    )


zo_accounting = zo.get(
    "global_accounting",
    {},
)


if zo_accounting.get(
    "post_wave1_current_2026_authority_candidates"
) != 20:
    fail(
        "EXPECTED_POST_WAVE1_CURRENT_AUTHORITY_20"
    )


if zo_accounting.get(
    "post_wave1_normalized_research_debt"
) != 62:
    fail(
        "EXPECTED_POST_WAVE1_NORMALIZED_DEBT_62"
    )


if zo_accounting.get(
    "post_wave1_actionable_current_2026_research_tasks"
) != 60:
    fail(
        "EXPECTED_POST_WAVE1_ACTIONABLE_DEBT_60"
    )


waves = zm.get(
    "remaining_research_waves"
)


if not isinstance(
    waves,
    list,
):
    fail(
        "Z_M_REMAINING_WAVES_MISSING"
    )


waves_by_number = {
    row.get(
        "remaining_wave_number"
    ):
        row
    for row in waves
}


for wave_number in [
    2,
    3,
    4,
    5,
]:

    if wave_number not in waves_by_number:
        fail(
            f"MISSING_WAVE:{wave_number}"
        )

    tasks = waves_by_number[
        wave_number
    ].get(
        "tasks"
    )

    if not isinstance(
        tasks,
        list,
    ):
        fail(
            f"WAVE_TASKS_INVALID:{wave_number}"
        )

    if len(
        tasks
    ) != 12:
        fail(
            f"EXPECTED_12_TASKS_WAVE_{wave_number}:GOT={len(tasks)}"
        )


total_tasks = sum(
    len(
        waves_by_number[
            wave_number
        ][
            "tasks"
        ]
    )
    for wave_number in [
        2,
        3,
        4,
        5,
    ]
)


if total_tasks != 48:
    fail(
        f"EXPECTED_48_WAVES_2_TO_5_TASKS:GOT={total_tasks}"
    )


print(
    "WAVES_2_TO_5_TASK_COUNT=48"
)

print(
    "FROZEN_REMAINING_RESEARCH_STATE_VALID=True"
)


# ============================================================
# 3. SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent":
        "WFS-NFL-Stage25-Research/1.0 "
        "(analysis-only venue authority research)"
})


# ============================================================
# 4. PROCESS EACH WAVE
# ============================================================

wave_results = []
all_routing_gaps = []
all_transport_failures = []
all_artifacts = []


for wave_number in [
    2,
    3,
    4,
    5,
]:

    print()
    print("=" * 120)
    print(
        f"[WAVE {wave_number}] OFFICIAL-SOURCE ACQUISITION"
    )
    print("=" * 120)

    tasks = waves_by_number[
        wave_number
    ][
        "tasks"
    ]


    task_keys = [
        (
            row.get("qid"),
            row.get("field"),
        )
        for row in tasks
    ]


    if len(
        set(
            task_keys
        )
    ) != 12:
        fail(
            f"WAVE_{wave_number}_TASK_KEYS_NOT_UNIQUE"
        )


    for index, task in enumerate(
        tasks,
        start=1,
    ):

        print(
            "WAVE_TASK|"
            f"WAVE={wave_number}|"
            f"INDEX={index}|"
            f"QID={task.get('qid')}|"
            f"NAME={task.get('canonical_name')!r}|"
            f"FIELD={task.get('field')}|"
            f"NEED={task.get('research_need')}"
        )


    unique_qids = sorted({
        row[
            "qid"
        ]
        for row in tasks
    })


    # --------------------------------------------------------
    # P856 LOCATOR
    # --------------------------------------------------------

    locator_results = {}


    for qid in unique_qids:

        if not re.fullmatch(
            r"Q\d+",
            qid,
        ):
            fail(
                f"INVALID_QID:{qid}"
            )


        locator_url = (
            "https://www.wikidata.org/wiki/Special:EntityData/"
            f"{qid}.json"
        )


        try:

            response = session.get(
                locator_url,
                timeout=25,
            )


            if response.status_code != 200:
                raise RuntimeError(
                    f"HTTP_{response.status_code}"
                )


            payload = response.json()

            entity = (
                payload.get(
                    "entities",
                    {}
                )
                .get(
                    qid,
                    {}
                )
            )


            claims = entity.get(
                "claims",
                {}
            )


            urls = []


            for claim in claims.get(
                "P856",
                [],
            ):

                value = (
                    claim.get(
                        "mainsnak",
                        {}
                    )
                    .get(
                        "datavalue",
                        {}
                    )
                    .get(
                        "value"
                    )
                )


                if (
                    isinstance(
                        value,
                        str,
                    )
                    and
                    value.startswith(
                        (
                            "http://",
                            "https://",
                        )
                    )
                ):
                    urls.append(
                        value
                    )


            urls = sorted(
                set(
                    urls
                )
            )


            locator_results[
                qid
            ] = {
                "qid":
                    qid,

                "locator_success":
                    True,

                "locator_source":
                    "WIKIDATA_P856",

                "locator_url":
                    locator_url,

                "raw_p856_urls":
                    urls,

                "locator_is_fact_authority":
                    False,
            }


        except Exception as exc:

            locator_results[
                qid
            ] = {
                "qid":
                    qid,

                "locator_success":
                    False,

                "locator_source":
                    "WIKIDATA_P856",

                "locator_url":
                    locator_url,

                "raw_p856_urls":
                    [],

                "locator_error":
                    (
                        type(exc).__name__
                        +
                        ": "
                        +
                        str(exc)
                    ),

                "locator_is_fact_authority":
                    False,
            }


        print(
            "P856_LOCATOR|"
            f"WAVE={wave_number}|"
            f"QID={qid}|"
            f"SUCCESS={locator_results[qid]['locator_success']}|"
            f"URLS={locator_results[qid]['raw_p856_urls']}"
        )


        time.sleep(
            0.20
        )


    # --------------------------------------------------------
    # BUILD FETCH PLAN
    # --------------------------------------------------------

    fetch_plan = []


    for qid in unique_qids:

        if qid in EXPLICIT_CURRENT_SITE_OVERRIDES:

            selected_urls = (
                EXPLICIT_CURRENT_SITE_OVERRIDES[
                    qid
                ]
            )

            mode = (
                "PREVIOUSLY_VALIDATED_CURRENT_HOST_OVERRIDE"
            )

        else:

            selected_urls = (
                locator_results[
                    qid
                ][
                    "raw_p856_urls"
                ]
            )

            mode = "DIRECT_P856"


        selected_urls = sorted(
            set(
                selected_urls
            )
        )


        locator_results[
            qid
        ][
            "selected_official_urls"
        ] = selected_urls


        locator_results[
            qid
        ][
            "selection_mode"
        ] = mode


        print(
            "OFFICIAL_URL_SELECTION|"
            f"WAVE={wave_number}|"
            f"QID={qid}|"
            f"MODE={mode}|"
            f"SELECTED={selected_urls}"
        )


        for ordinal, url in enumerate(
            selected_urls,
            start=1,
        ):

            expected_host = (
                urlparse(
                    url
                ).hostname
                or
                ""
            ).lower()


            fetch_plan.append({
                "source_id":
                    f"W{wave_number}_{qid}_{ordinal}",

                "wave":
                    wave_number,

                "qid":
                    qid,

                "url":
                    url,

                "expected_host":
                    expected_host,

                "selection_mode":
                    mode,
            })


    print(
        f"WAVE_{wave_number}_FETCH_PLAN_COUNT={len(fetch_plan)}"
    )


    # --------------------------------------------------------
    # FETCH
    # --------------------------------------------------------

    fetch_results = {}


    for spec in fetch_plan:

        sid = spec[
            "source_id"
        ]


        try:

            response = session.get(
                spec[
                    "url"
                ],
                timeout=25,
                allow_redirects=True,
            )


            status = response.status_code
            final_url = response.url

            final_host = (
                urlparse(
                    final_url
                ).hostname
                or
                ""
            ).lower()


            expected_norm = normalize_host(
                spec[
                    "expected_host"
                ]
            )

            final_norm = normalize_host(
                final_host
            )


            same_host = (
                expected_norm
                ==
                final_norm
            )


            routing_gap = (
                bool(
                    final_host
                )
                and
                not same_host
            )


            raw = response.text

            text = clean_html(
                raw
            )


            evidence_eligible = (
                status == 200
                and
                same_host
            )


            if evidence_eligible:

                snippets = keyword_snippets(
                    text
                )

            else:

                snippets = []


            result = {
                "source_id":
                    sid,

                "wave":
                    wave_number,

                "qid":
                    spec[
                        "qid"
                    ],

                "selection_mode":
                    spec[
                        "selection_mode"
                    ],

                "requested_url":
                    spec[
                        "url"
                    ],

                "expected_host":
                    spec[
                        "expected_host"
                    ],

                "final_url":
                    final_url,

                "final_host":
                    final_host,

                "http_status":
                    status,

                "transport_success":
                    status == 200,

                "same_host_identity":
                    same_host,

                "cross_domain_routing_gap":
                    routing_gap,

                "evidence_eligible":
                    evidence_eligible,

                "page_title":
                    (
                        page_title(
                            raw
                        )
                        if
                        status == 200
                        else
                        None
                    ),

                "content_length_chars":
                    len(
                        text
                    ),

                "keyword_snippets":
                    snippets,

                "keyword_snippet_count":
                    len(
                        snippets
                    ),

                "retrieved_at_utc":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                "classification_performed":
                    False,

                "authority_granted":
                    False,

                "fact_promoted":
                    False,
            }


        except Exception as exc:

            result = {
                "source_id":
                    sid,

                "wave":
                    wave_number,

                "qid":
                    spec[
                        "qid"
                    ],

                "selection_mode":
                    spec[
                        "selection_mode"
                    ],

                "requested_url":
                    spec[
                        "url"
                    ],

                "expected_host":
                    spec[
                        "expected_host"
                    ],

                "final_url":
                    None,

                "final_host":
                    None,

                "http_status":
                    None,

                "transport_success":
                    False,

                "same_host_identity":
                    False,

                "cross_domain_routing_gap":
                    False,

                "evidence_eligible":
                    False,

                "page_title":
                    None,

                "content_length_chars":
                    0,

                "keyword_snippets":
                    [],

                "keyword_snippet_count":
                    0,

                "retrieved_at_utc":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                "error":
                    (
                        type(exc).__name__
                        +
                        ": "
                        +
                        str(exc)
                    ),

                "classification_performed":
                    False,

                "authority_granted":
                    False,

                "fact_promoted":
                    False,
            }


        fetch_results[
            sid
        ] = result


        print(
            "FETCH_RESULT|"
            f"WAVE={wave_number}|"
            f"SOURCE={sid}|"
            f"QID={result['qid']}|"
            f"HTTP={result['http_status']}|"
            f"SAME_HOST={result['same_host_identity']}|"
            f"ROUTING_GAP={result['cross_domain_routing_gap']}|"
            f"EVIDENCE_ELIGIBLE={result['evidence_eligible']}|"
            f"SNIPPETS={result['keyword_snippet_count']}|"
            f"FINAL_URL={result['final_url']!r}"
        )


        for snippet in result[
            "keyword_snippets"
        ][:8]:

            print(
                "  EVIDENCE_SNIPPET|"
                f"WAVE={wave_number}|"
                f"QID={result['qid']}|"
                f"KEYWORD={snippet['keyword']!r}|"
                f"TEXT={snippet['snippet']!r}"
            )


        time.sleep(
            0.30
        )


    successful = [
        row
        for row in fetch_results.values()
        if row[
            "transport_success"
        ]
        is True
    ]


    eligible = [
        row
        for row in fetch_results.values()
        if row[
            "evidence_eligible"
        ]
        is True
    ]


    transport_failures = [
        row
        for row in fetch_results.values()
        if row[
            "transport_success"
        ]
        is False
    ]


    routing_gaps = [
        row
        for row in fetch_results.values()
        if row[
            "cross_domain_routing_gap"
        ]
        is True
    ]


    all_transport_failures.extend(
        transport_failures
    )

    all_routing_gaps.extend(
        routing_gaps
    )


    # --------------------------------------------------------
    # TASK EVIDENCE
    # --------------------------------------------------------

    task_evidence = []


    for index, task in enumerate(
        tasks,
        start=1,
    ):

        qid = task[
            "qid"
        ]


        eligible_rows = [
            row
            for row in fetch_results.values()
            if (
                row[
                    "qid"
                ]
                ==
                qid
                and
                row[
                    "evidence_eligible"
                ]
                is True
            )
        ]


        qid_routing_gaps = [
            row
            for row in routing_gaps
            if row[
                "qid"
            ]
            ==
            qid
        ]


        snippet_count = sum(
            row[
                "keyword_snippet_count"
            ]
            for row in eligible_rows
        )


        envelope = {
            "task_index":
                index,

            "remaining_wave_number":
                wave_number,

            "remaining_wave_position":
                task.get(
                    "remaining_wave_position"
                ),

            "qid":
                qid,

            "canonical_name":
                task.get(
                    "canonical_name"
                ),

            "field":
                task.get(
                    "field"
                ),

            "original_batch_number":
                task.get(
                    "original_batch_number"
                ),

            "original_batch_position":
                task.get(
                    "original_batch_position"
                ),

            "research_tier":
                task.get(
                    "research_tier"
                ),

            "research_need":
                task.get(
                    "research_need"
                ),

            "official_url_selection_mode":
                locator_results[
                    qid
                ].get(
                    "selection_mode"
                ),

            "selected_official_urls":
                locator_results[
                    qid
                ].get(
                    "selected_official_urls",
                    [],
                ),

            "evidence_eligible_source_count":
                len(
                    eligible_rows
                ),

            "cross_domain_routing_gap_count":
                len(
                    qid_routing_gaps
                ),

            "keyword_evidence_snippet_count":
                snippet_count,

            "evidence_acquired":
                (
                    len(
                        eligible_rows
                    )
                    >
                    0
                    and
                    snippet_count
                    >
                    0
                ),

            "routing_review_required":
                len(
                    qid_routing_gaps
                )
                >
                0,

            "classification":
                "NOT_PERFORMED",

            "current_2026_class":
                "NOT_EVALUATED",

            "current_2026_authority_verified":
                False,

            "fact_promoted":
                False,
        }


        task_evidence.append(
            envelope
        )


        print(
            "TASK_EVIDENCE|"
            f"WAVE={wave_number}|"
            f"INDEX={index}|"
            f"QID={qid}|"
            f"NAME={task.get('canonical_name')!r}|"
            f"FIELD={task.get('field')}|"
            f"ELIGIBLE_SOURCES={len(eligible_rows)}|"
            f"ROUTING_GAPS={len(qid_routing_gaps)}|"
            f"SNIPPETS={snippet_count}|"
            f"EVIDENCE_ACQUIRED={envelope['evidence_acquired']}|"
            f"ROUTING_REVIEW_REQUIRED={envelope['routing_review_required']}"
        )


    with_evidence = [
        row
        for row in task_evidence
        if row[
            "evidence_acquired"
        ]
        is True
    ]


    without_evidence = [
        row
        for row in task_evidence
        if row[
            "evidence_acquired"
        ]
        is False
    ]


    routing_review_tasks = [
        row
        for row in task_evidence
        if row[
            "routing_review_required"
        ]
        is True
    ]


    # --------------------------------------------------------
    # WAVE SAFETY
    # --------------------------------------------------------

    checks = {
        "EXACTLY_12_TASKS":
            len(
                task_evidence
            )
            ==
            12,

        "TASK_KEYS_UNIQUE":
            len(
                set(
                    task_keys
                )
            )
            ==
            12,

        "NO_CLASSIFICATION":
            all(
                row[
                    "classification"
                ]
                ==
                "NOT_PERFORMED"
                for row in task_evidence
            ),

        "NO_AUTHORITY_GRANTED":
            all(
                row[
                    "current_2026_authority_verified"
                ]
                is False
                for row in task_evidence
            ),

        "NO_FACT_PROMOTION":
            all(
                row[
                    "fact_promoted"
                ]
                is False
                for row in task_evidence
            ),

        "ROUTING_GAPS_NOT_EVIDENCE_ELIGIBLE":
            all(
                row[
                    "evidence_eligible"
                ]
                is False
                for row in routing_gaps
            ),
    }


    if not all(
        checks.values()
    ):
        fail(
            f"WAVE_{wave_number}_SAFETY_FAILURE"
        )


    # --------------------------------------------------------
    # WRITE PER-WAVE FREEZE
    # --------------------------------------------------------

    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )


    artifact = AUDIT / (
        f"stage25c_h_d_c5_z_p_wave_{wave_number}_official_evidence_"
        f"{stamp}.json"
    )


    output = {
        "contract":
            "WFS_STAGE25C_H_D_C5_Z_P_REMAINING_WAVE_OFFICIAL_EVIDENCE_V1",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "analysis_only":
            True,

        "network_fetch_performed":
            True,

        "official_source_acquisition_only":
            True,

        "classification_performed":
            False,

        "authority_granted":
            False,

        "fact_promotion_allowed":
            False,

        "registry_promoted":
            False,

        "remaining_wave_number":
            wave_number,

        "task_count":
            12,

        "task_signature": [
            {
                "qid":
                    row[
                        "qid"
                    ],

                "field":
                    row[
                        "field"
                    ],
            }
            for row in task_evidence
        ],

        "locator_results":
            locator_results,

        "explicit_current_site_overrides":
            EXPLICIT_CURRENT_SITE_OVERRIDES,

        "official_site_fetch_results":
            fetch_results,

        "task_evidence":
            task_evidence,

        "summary": {
            "wave_tasks":
                12,

            "unique_qids":
                len(
                    unique_qids
                ),

            "official_site_requests":
                len(
                    fetch_results
                ),

            "http_200_successful_sources":
                len(
                    successful
                ),

            "evidence_eligible_same_host_sources":
                len(
                    eligible
                ),

            "transport_failures":
                len(
                    transport_failures
                ),

            "cross_domain_routing_gaps":
                len(
                    routing_gaps
                ),

            "tasks_with_acquired_official_evidence":
                len(
                    with_evidence
                ),

            "tasks_without_acquired_official_evidence":
                len(
                    without_evidence
                ),

            "tasks_requiring_explicit_routing_review":
                len(
                    routing_review_tasks
                ),
        },

        "routing_policy": {
            "automatic_cross_domain_redirect_trust":
                False,

            "cross_domain_redirect_content_evidence_eligible":
                False,

            "cross_domain_redirect_becomes_routing_gap":
                True,

            "explicit_followup_required_before_using_new_domain":
                True,
        },

        "authority_policy": {
            "wikidata_p856_is_locator_only":
                True,

            "keyword_presence_is_authority":
                False,

            "venue_name_is_fact":
                False,

            "generic_surface_word_is_playing_surface":
                False,

            "classification_performed":
                False,

            "authority_granted":
                False,

            "fact_promotion_allowed":
                False,
        },

        "safety_checks":
            checks,

        "upstream": {
            "remaining_research_plan": {
                "path":
                    str(
                        Z_M
                    ),

                "sha256":
                    sha256(
                        Z_M
                    ),
            },

            "wave_1_closeout": {
                "path":
                    str(
                        Z_O
                    ),

                "sha256":
                    sha256(
                        Z_O
                    ),
            },
        },

        "policy": {
            "database_edit":
                False,

            "production_edit":
                False,

            "solver_edit":
                False,

            "service_restart":
                False,

            "production_influence":
                False,
        },
    }


    artifact.write_text(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )


    artifact_sha = sha256(
        artifact
    )


    all_artifacts.append({
        "wave":
            wave_number,

        "path":
            str(
                artifact
            ),

        "sha256":
            artifact_sha,

        "task_count":
            12,

        "official_site_requests":
            len(
                fetch_results
            ),

        "http_200_successful_sources":
            len(
                successful
            ),

        "transport_failures":
            len(
                transport_failures
            ),

        "routing_gaps":
            len(
                routing_gaps
            ),

        "tasks_with_evidence":
            len(
                with_evidence
            ),

        "tasks_without_evidence":
            len(
                without_evidence
            ),
    })


    wave_results.append({
        "wave":
            wave_number,

        "task_count":
            12,

        "unique_qids":
            len(
                unique_qids
            ),

        "official_site_requests":
            len(
                fetch_results
            ),

        "http_200_successful_sources":
            len(
                successful
            ),

        "transport_failures":
            len(
                transport_failures
            ),

        "routing_gaps":
            len(
                routing_gaps
            ),

        "tasks_with_evidence":
            len(
                with_evidence
            ),

        "tasks_without_evidence":
            len(
                without_evidence
            ),

        "artifact":
            str(
                artifact
            ),

        "sha256":
            artifact_sha,
    })


    print()
    print(
        f"WAVE_{wave_number}_EVIDENCE_ARTIFACT={artifact}"
    )

    print(
        f"WAVE_{wave_number}_EVIDENCE_SHA={artifact_sha}"
    )

    print(
        f"WAVE_{wave_number}_ROUTING_GAPS={len(routing_gaps)}"
    )

    print(
        f"WAVE_{wave_number}_TASKS_WITH_EVIDENCE={len(with_evidence)}"
    )

    print(
        f"WAVE_{wave_number}_TASKS_WITHOUT_EVIDENCE={len(without_evidence)}"
    )


# ============================================================
# 5. MASTER CONSOLIDATED ACQUISITION MANIFEST
# ============================================================

print()
print("=" * 120)
print("[5] WRITE WAVES 2-5 MASTER ACQUISITION MANIFEST")
print("=" * 120)


if len(
    all_artifacts
) != 4:
    fail(
        "EXPECTED_4_WAVE_ARTIFACTS"
    )


if sum(
    row[
        "task_count"
    ]
    for row in all_artifacts
) != 48:
    fail(
        "EXPECTED_48_TOTAL_TASKS_IN_ARTIFACTS"
    )


total_requests = sum(
    row[
        "official_site_requests"
    ]
    for row in all_artifacts
)

total_http_200 = sum(
    row[
        "http_200_successful_sources"
    ]
    for row in all_artifacts
)

total_transport_failures = sum(
    row[
        "transport_failures"
    ]
    for row in all_artifacts
)

total_routing_gaps = sum(
    row[
        "routing_gaps"
    ]
    for row in all_artifacts
)

total_tasks_with_evidence = sum(
    row[
        "tasks_with_evidence"
    ]
    for row in all_artifacts
)

total_tasks_without_evidence = sum(
    row[
        "tasks_without_evidence"
    ]
    for row in all_artifacts
)


stamp = datetime.now(
    timezone.utc
).strftime(
    "%Y%m%dT%H%M%SZ"
)


manifest = AUDIT / (
    "stage25c_h_d_c5_z_p_waves_2_to_5_official_evidence_manifest_"
    f"{stamp}.json"
)


manifest_payload = {
    "contract":
        "WFS_STAGE25C_H_D_C5_Z_P_WAVES_2_TO_5_OFFICIAL_EVIDENCE_MANIFEST_V1",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "analysis_only":
        True,

    "network_fetch_performed":
        True,

    "classification_performed":
        False,

    "authority_granted":
        False,

    "fact_promotion_allowed":
        False,

    "registry_promoted":
        False,

    "wave_numbers":
        [
            2,
            3,
            4,
            5,
        ],

    "total_task_count":
        48,

    "wave_artifacts":
        all_artifacts,

    "wave_results":
        wave_results,

    "summary": {
        "total_tasks":
            48,

        "total_official_site_requests":
            total_requests,

        "total_http_200_successful_sources":
            total_http_200,

        "total_transport_failures":
            total_transport_failures,

        "total_cross_domain_routing_gaps":
            total_routing_gaps,

        "total_tasks_with_acquired_evidence":
            total_tasks_with_evidence,

        "total_tasks_without_acquired_evidence":
            total_tasks_without_evidence,
    },

    "routing_policy": {
        "automatic_cross_domain_redirect_trust":
            False,

        "cross_domain_redirect_content_evidence_eligible":
            False,

        "explicit_routing_review_required":
            total_routing_gaps
            >
            0,
    },

    "policy": {
        "database_edit":
            False,

        "production_edit":
            False,

        "solver_edit":
            False,

        "service_restart":
            False,

        "production_influence":
            False,
    },

    "upstream": {
        "remaining_research_plan": {
            "path":
                str(
                    Z_M
                ),

            "sha256":
                sha256(
                    Z_M
                ),
        },

        "wave_1_closeout": {
            "path":
                str(
                    Z_O
                ),

            "sha256":
                sha256(
                    Z_O
                ),
        },
    },
}


manifest.write_text(
    json.dumps(
        manifest_payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    +
    "\n",
    encoding="utf-8",
)


manifest_sha = sha256(
    manifest
)


print(
    f"WAVES_2_TO_5_MASTER_MANIFEST={manifest}"
)

print(
    f"WAVES_2_TO_5_MASTER_MANIFEST_SHA={manifest_sha}"
)


# ============================================================
# 6. FINAL PROTECTED ARTIFACT CHECK
# ============================================================

print()
print("=" * 120)
print("[6] FINAL TRUE IMMUTABLE CHECK")
print("=" * 120)


protected_clean = True


for path, expected in EXPECTED.items():

    same = (
        sha256(
            path
        )
        ==
        expected
    )

    print(
        f"{path.name}_UNCHANGED={same}"
    )

    protected_clean &= same


if not protected_clean:
    fail(
        "PROTECTED_ARTIFACT_CHANGED"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 120)
print("STAGE25C-H-D-C5-Z-P FINAL")
print("=" * 120)

print(
    "WAVES_PROCESSED=2,3,4,5"
)

print(
    "WAVES_2_TO_5_TOTAL_TASKS=48"
)

print(
    f"TOTAL_OFFICIAL_SITE_REQUESTS={total_requests}"
)

print(
    f"TOTAL_HTTP_200_SUCCESSFUL_SOURCES={total_http_200}"
)

print(
    f"TOTAL_TRANSPORT_FAILURES={total_transport_failures}"
)

print(
    f"TOTAL_CROSS_DOMAIN_ROUTING_GAPS={total_routing_gaps}"
)

print(
    f"TOTAL_TASKS_WITH_ACQUIRED_OFFICIAL_EVIDENCE={total_tasks_with_evidence}"
)

print(
    f"TOTAL_TASKS_WITHOUT_ACQUIRED_OFFICIAL_EVIDENCE={total_tasks_without_evidence}"
)

print(
    f"PROTECTED_ARTIFACTS_IMMUTABLE={protected_clean}"
)

print(
    f"WAVES_2_TO_5_MASTER_MANIFEST={manifest}"
)

print(
    f"WAVES_2_TO_5_MASTER_MANIFEST_SHA={manifest_sha}"
)

print("ANALYSIS_ONLY=TRUE")
print("NETWORK_FETCH_PERFORMED=TRUE")
print("OFFICIAL_SOURCE_ACQUISITION_ONLY=TRUE")
print("CLASSIFICATION_PERFORMED=FALSE")
print("AUTHORITY_GRANTED=FALSE")
print("FACT_PROMOTION_ALLOWED=FALSE")
print("REGISTRY_PROMOTED=FALSE")
print("DB_EDITED=FALSE")
print("PRODUCTION_EDITED=FALSE")
print("SOLVER_EDITED=FALSE")
print("SERVICE_CHANGED=FALSE")
print("SERVICE_RESTARTED=FALSE")
print("STAGE25_PRODUCTION_INFLUENCE_ALLOWED=FALSE")

print(
    "WAVES_2_TO_5_EVIDENCE_BATCH_FROZEN=TRUE"
)

print(
    "STAGE25C_H_D_C5_Z_P_RESULT="
    "PASS_WAVES_2_TO_5_OFFICIAL_EVIDENCE_ACQUISITION"
)


if total_routing_gaps > 0:

    print(
        "NEXT_GATE="
        "STAGE25C_H_D_C5_Z_Q_WAVES_2_TO_5_ROUTING_REVIEW"
    )

else:

    print(
        "NEXT_GATE="
        "STAGE25C_H_D_C5_Z_Q_WAVES_2_TO_5_COMBINED_CLASSIFICATION_CONSOLIDATION"
    )


print(
    "STOP_GATE="
    "RETURN_STAGE25C_H_D_C5_Z_P_OUTPUT_BEFORE_FINAL_REVIEW"
)

print("=" * 120)
