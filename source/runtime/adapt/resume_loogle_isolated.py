"""Resume saved Longdep results with five isolated article processes."""
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/V16_run')
RUN = 'v16_loogle_longdep_q36plus_149_20261109'
OUT = ROOT / 'eval' / 'experiments' / RUN
LOGS = ROOT / 'loogle_resume_worker_logs'
LOGS.mkdir(exist_ok=True)


def complete(ti):
    graph = Path('/root/autodl-tmp/shortdep_qwen36_graphs_20261109') / f'title_{ti}/round0/v15_build_complete.json'
    if not graph.is_file():
        return True
    rows = json.loads((ROOT / 'loogle_longdep_v16_20261109_data/choices' / f'title_{ti}.json').read_text())
    return all((OUT / f'title_{ti}' / phase / f'q{qi}.json').is_file()
               for phase in ('fresh', 'round1', 'round2', 'round3') for qi in range(len(rows)))


def run(ti):
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    code = 'import json,sys; from adapt.run_loogle_longdep_v16 import run_title; print(json.dumps(run_title(int(sys.argv[1])),ensure_ascii=False),flush=True)'
    logfile = LOGS / f'title_{ti}.log'
    with logfile.open('a', buffering=1) as log:
        log.write('\nRESUME ' + datetime.datetime.now().isoformat() + '\n')
        proc = subprocess.Popen([sys.executable, '-u', '-c', code, str(ti)], cwd=ROOT,
                                env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps({'title': ti, 'pid': proc.pid, 'status': 'started'}), flush=True)
        rc = proc.wait()
    result = {'title': ti, 'exit_code': rc, 'complete': complete(ti), 'log': str(logfile)}
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    todo = [ti for ti in range(20) if not complete(ti)]
    print(json.dumps({'started': datetime.datetime.now().isoformat(), 'workers': 5, 'pending_titles': todo}), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(run, ti) for ti in todo]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    print(json.dumps({'status': 'finished', 'results': results}), flush=True)
