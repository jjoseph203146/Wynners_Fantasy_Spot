import nflreadpy as nfl


SEASONS_TO_TEST = [
    2025,
    2026,
]


SOURCES = [
    (
        "WEEKLY ROSTERS",
        nfl.load_rosters_weekly,
    ),
    (
        "SNAP COUNTS",
        nfl.load_snap_counts,
    ),
    (
        "INJURIES",
        nfl.load_injuries,
    ),
    (
        "DEPTH CHARTS",
        nfl.load_depth_charts,
    ),
]


def print_separator():
    print()
    print("=" * 80)


def inspect_source(
    name,
    loader,
    season,
):
    print_separator()

    print(
        f"{name} — {season}"
    )

    print("=" * 80)

    try:

        df = loader(
            seasons=[season]
        )

    except Exception as exc:

        print(
            "STATUS: NOT AVAILABLE"
        )

        print(
            f"ERROR: {exc}"
        )

        return

    print(
        "STATUS: AVAILABLE"
    )

    print(
        f"ROWS: {df.height}"
    )

    print(
        f"COLUMNS: {df.width}"
    )

    print()

    print(
        "COLUMN NAMES:"
    )

    for column in df.columns:
        print(
            f"  {column}"
        )

    print()

    print(
        "SCHEMA:"
    )

    print(
        df.schema
    )

    if df.height > 0:

        print()

        print(
            "FIRST 5 ROWS:"
        )

        print(
            df.head(5)
        )


def main():

    print()
    print("=" * 80)
    print("NFL CONTEXT SOURCE INSPECTOR")
    print("=" * 80)

    for season in SEASONS_TO_TEST:

        for name, loader in SOURCES:

            inspect_source(
                name=name,
                loader=loader,
                season=season,
            )

    print_separator()

    print(
        "SOURCE INSPECTION COMPLETE"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
