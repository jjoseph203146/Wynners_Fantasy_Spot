"""Read-only input adapters and isolated immutable validation publication."""
import argparse
import csv
import json
from pathlib import Path
import hashlib
from .core import build,digest
PACKAGE=Path(__file__).resolve().parent
ROOT=PACKAGE.parent/'processed'/'simulator_recursive_state_replay_v1'

def contract(): return json.loads((PACKAGE/'feature_contract.json').read_text())

def frozen_reference(path,game_id,team,c):
    raw=Path(path).read_bytes()
    sha=hashlib.sha256(raw).hexdigest()
    if sha!=c['training_source']['sha256']: raise ValueError('SOURCE_VERSION_MISMATCH')
    rows=[r for r in csv.DictReader(raw.decode().splitlines()) if r['game_id']==game_id and r['team']==team]
    if len(rows)!=1: raise ValueError('IDENTITY_MISMATCH')
    r=rows[0]
    def num(s): return None if s in ('','nan','NaN') else float(s)
    return dict(game_id=game_id,team=team,opponent=r['opponent_team'],source_hash=sha,
                feature_contract_version=c['version'],features={n:num(r[n]) for n in c['margin']})

def publish(data,c):
    result=build(data,c)
    if ROOT.resolve()!=ROOT or ROOT.is_symlink(): raise ValueError('UNSAFE_OUTPUT')
    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/result['state_manifest']['state_version']
    files={k+'.json':(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode() for k,v in result.items()}
    try: path.mkdir()
    except FileExistsError:
        if path.is_symlink() or any((path/n).is_symlink() or not (path/n).is_file() or (path/n).read_bytes()!=b for n,b in files.items()):
            raise ValueError('IMMUTABLE_ARTIFACT_CONFLICT')
        return path
    for name,raw in files.items():
        with (path/name).open('xb') as f: f.write(raw)
    return path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--frozen-csv',type=Path)
    a=p.parse_args(); d=json.loads(a.input.read_text()); c=contract()
    if a.frozen_csv:
        d['reference']=frozen_reference(a.frozen_csv,d['context']['game_id'],d['context']['team'],c)
    print(publish(d,c))

if __name__=='__main__': main()


def source_readiness(db_path):
    """Inspect only the two already identified historical sources, read-only."""
    import sqlite3
    with sqlite3.connect(Path(db_path).resolve().as_uri()+'?mode=ro',uri=True) as con:
        columns={table:[r[1] for r in con.execute('PRAGMA table_info('+table+')')]
                 for table in ('team_game_stats','games')}
    explicit=all('observation_time' in fields for fields in columns.values()) and 'completed_at' in columns['games']
    return dict(status='READY_FOR_TIMESTAMP_REVIEW' if explicit else 'EXPECTED_UNAVAILABLE',
                reason='Original publication and completion timestamps must be provided; updated_at is not a substitute.',
                observed_time_columns={k:[n for n in v if any(s in n for s in ('date','time','completed','updated'))] for k,v in columns.items()},
                full_historical_replay_performed=False,monte_carlo_authorized=False)
