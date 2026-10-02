"""Usage: venv/bin/python -B -m health_monitor [--json]."""
import argparse
import json
from pathlib import Path
import signal
import sys
from datetime import datetime, timezone

from .core import run, EXIT_CODES
from .production import Production, install_write_guard


class MonitorTimeout(BaseException):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='Print machine-readable report')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    install_write_guard()
    def timeout(*_):
        raise MonitorTimeout('Monitor time budget exceeded')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(90)
    try:
        report = run(Production(Path(__file__).resolve().parents[1]), datetime.now(timezone.utc))
        code = EXIT_CODES[report['overall_status']]
    except (Exception, MonitorTimeout) as exc:
        report = dict(contract='NFL_APP_HEALTH_MONITOR_V1', generated_at_utc=datetime.now(timezone.utc).isoformat(),
                      overall_status='OWNER_ACTION_REQUIRED', summary_counts={'HEALTHY':0,'WARNING':0,'CRITICAL':0,'OWNER_ACTION_REQUIRED':1},
                      checks=[dict(check_id='monitor', status='OWNER_ACTION_REQUIRED', reason_code='MONITOR_EXECUTION_FAILURE',
                                   scope='', summary='monitor could not complete', evidence={'exception_type':type(exc).__name__})])
        code = 40
    finally:
        signal.alarm(0)
    if args.json:
        print(json.dumps(report, sort_keys=True, allow_nan=False))
    else:
        print(report['contract']+' '+report['overall_status'])
        for check in report['checks']:
            if check['status'] == 'HEALTHY':
                continue
            print(f"{check['status']} {check['check_id']} {check['scope']} {check['reason_code']}")
        print('summary_counts='+json.dumps(report['summary_counts'], sort_keys=True))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
