"""
NFL APP Health Monitor V2 controller CLI.

Default:
    dry-run observation + incident state only

Explicit repair execution:
    --execute-repairs

This CLI does not install or modify scheduling.
"""

from __future__ import annotations

import argparse
import json

from .controller import production_cycle


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--execute-repairs",
        action="store_true",
        help=(
            "Permit only explicitly allowlisted guarded repairs. "
            "Without this flag the controller is dry-run."
        ),
    )

    args = parser.parse_args()

    result = production_cycle(
        execute_repairs=args.execute_repairs,
    )

    printable = {
        key: value
        for key, value in result.items()
        if key != "state"
    }

    print(
        json.dumps(
            printable,
            sort_keys=True,
            allow_nan=False,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
