"""Explicit offline inputs only; this entry point has no fetch capability."""
import argparse
import json
from pathlib import Path

from .fixtures import sample
from .publication import publish, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-json", type=Path, help="Local normalized records and WFS snapshots")
    inputs.add_argument("--synthetic-fixtures", choices=("normal", "ambiguous"))
    args = parser.parse_args()
    if args.input_json:
        data = json.loads(args.input_json.read_text())
    else:
        data = sample(all_stories=True)
        if args.synthetic_fixtures == "ambiguous":
            for record in data["records"]:
                record.update(raw_publication_timestamp="Sun, 07 Sep 2025 10:00:00 EST",
                              published_at_utc=None, temporal_status="AMBIGUOUS_SOURCE_TIMESTAMP",
                              temporal_confidence="UNRESOLVED")
    path = publish(data)
    verify(path)
    print(path)


if __name__ == "__main__":
    main()
