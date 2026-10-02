"""Explicit CLI; writes only immutable, content-addressed shadow bundles."""
import argparse
import hashlib
import json
from pathlib import Path

from .core import CONTRACT, build

SHADOW_ROOT = Path(__file__).resolve().parents[1] / 'processed' / CONTRACT


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def bundle(data):
    result = build(data)
    files = {name + '.json': encoded(result[name]) for name in ('players', 'ledger', 'balances')}
    manifest = dict(result['manifest'], artifacts={name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()})
    files['manifest.json'] = encoded(manifest)
    return files


def publish(data):
    files = bundle(data)
    run_id = hashlib.sha256(files['manifest.json']).hexdigest()
    # No user-selectable output path and no canonical-path writer.
    root = SHADOW_ROOT
    require_safe = root.resolve() == root and not root.is_symlink()
    if not require_safe:
        raise ValueError('UNSAFE_SHADOW_ROOT')
    root.mkdir(parents=True, exist_ok=True)
    destination = root / run_id
    try:
        destination.mkdir()
    except FileExistsError:
        if destination.is_symlink() or any((destination / name).is_symlink() or
                (destination / name).read_bytes() != raw for name, raw in files.items()):
            raise ValueError('EXISTING_BUNDLE_MISMATCH')
        return destination
    for name, raw in files.items():
        with (destination / name).open('xb') as handle:
            handle.write(raw)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Explicit offline JSON snapshot')
    args = parser.parse_args()
    print(publish(json.loads(args.input.read_text())))


if __name__ == '__main__':
    main()
