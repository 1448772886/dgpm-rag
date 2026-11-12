"""One dependent experiment: start the pinned new-prompt run after 800 old answers."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/V16_run')
OLD = ROOT / 'eval/experiments/neoqa_v16_q36plus_200x4_20261111'
NAME = 'neoqa_v16_q36plus_200x4_newprompt_20261111'
OUT = ROOT / 'eval/experiments' / NAME
RUNTIME = OUT / 'runtime'


def save(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def verify_runtime():
    manifest = json.loads((OUT / 'pinned_runtime.json').read_text())
    for name, expected in manifest['sha256'].items():
        assert hashlib.sha256((RUNTIME / name).read_bytes()).hexdigest() == expected, name


def run(preflight=False):
    verify_runtime()
    sys.path.insert(0, str(RUNTIME))
    from adapt import run_neoqa_v16_200 as runner
    runner.RUN = NAME
    runner.OUT = OUT
    runner.WORK = ROOT / 'eval/database' / NAME
    runner.KEYFILE = ROOT / 'adapt/.secrets/neoqa_qwen_20261111.json'
    sys.argv = [__file__] + (['--preflight-only'] if preflight else [])
    runner.main()


def wait_and_launch():
    # The lock lasts for this launcher and the subsequent entire experiment.
    lock = (OUT / 'dependent_job.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT / 'dependent_job_state.json').exists():
        previous = json.loads((OUT / 'dependent_job_state.json').read_text())
        if previous['status'] in ('running', 'finished'):
            raise RuntimeError('Dependent experiment already launched; refusing duplicate')
    save(OUT / 'dependent_job_state.json', dict(status='waiting', pid=os.getpid(), source=str(OLD)))
    while not (OLD / 'run_completion.json').exists():
        time.sleep(15)
    complete = json.loads((OLD / 'run_completion.json').read_text())
    if not complete.get('complete'):
        raise RuntimeError('Previous run finished incompletely; dependent run not launched')
    for phase in ('fresh', 'round1', 'round2', 'round3'):
        rows = [json.loads(p.read_text()) for p in (OLD / 'questions').glob(f'*/{phase}/result.json')]
        assert len(rows) == 200 and all(x['status'] == 'completed' for x in rows), phase
        assert len({x['question_id'] for x in rows}) == 200, phase
    verify_runtime()
    command = ['/root/miniconda3/bin/python', '-u', str(Path(__file__).resolve()), '--run']
    with (OUT / 'supervisor.log').open('ab', buffering=0) as stream:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=stream, start_new_session=True)
    (OUT / 'supervisor.pid').write_text(str(proc.pid))
    save(OUT / 'dependent_job_state.json', dict(status='running', pid=proc.pid,
         launcher_pid=os.getpid(), started_at=time.time(), source=str(OLD)))
    code = proc.wait()
    finished = json.loads((OUT / 'run_completion.json').read_text()) if (OUT / 'run_completion.json').exists() else {}
    save(OUT / 'dependent_job_state.json', dict(status='finished' if code == 0 and finished.get('complete') else 'failed',
         exitcode=code, pid=proc.pid, ended_at=time.time(), source=str(OLD)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.run or args.preflight_only:
        run(preflight=args.preflight_only)
    else:
        wait_and_launch()
