"""Explicit immutable shadow writer; no arbitrary output destination."""
import argparse
import json
from pathlib import Path
from .core import build, digest, CONTRACT
ROOT = Path(__file__).resolve().parents[1] / 'processed' / CONTRACT

def encode(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()

def publish(data):
    result=build(data)
    c=data['context']
    key=digest({k:c[k] for k in ('season','week','game_id','team','scenario_version')})
    require_safe=ROOT.resolve()==ROOT and not ROOT.is_symlink()
    if not require_safe:
        raise ValueError('UNSAFE_SHADOW_ROOT')
    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/key
    files={name+'.json':encode(value) for name,value in result.items()}
    try:
        path.mkdir()
    except FileExistsError:
        if path.is_symlink() or any((path/name).is_symlink() or not (path/name).is_file() or (path/name).read_bytes()!=raw for name,raw in files.items()):
            raise ValueError('IMMUTABLE_SCENARIO_CONFLICT')
        return path
    for name,raw in files.items():
        with (path/name).open('xb') as out:
            out.write(raw)
    return path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args()
    print(publish(json.loads(args.input.read_text())))

if __name__=='__main__':
    main()
