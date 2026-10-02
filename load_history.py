from schedule import (
    download_schedule,
    update_schedule_database,
    export_schedule,
    print_summary,
)


HISTORICAL_SEASONS = [
    2023,
    2024,
    2025,
]


def main():

    print()
    print("=" * 70)
    print("NFL HISTORICAL SCHEDULE LOADER")
    print("=" * 70)

    for season in HISTORICAL_SEASONS:

        print()
        print(f"Loading season {season}")

        df = download_schedule(
            season=season
        )

        update_schedule_database(
            df
        )

    export_schedule()

    print_summary()

    print()
    print("=" * 70)
    print("HISTORICAL SCHEDULE LOAD SUCCESSFUL")
    print("=" * 70)


if __name__ == "__main__":
    main()
