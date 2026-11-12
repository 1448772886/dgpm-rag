import concurrent.futures,subprocess,os,json
from pathlib import Path
r=Path('/root/autodl-tmp/V16_run');logs=r/'round1_restore_logs';logs.mkdir(exist_ok=True)
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
def run(ti):
 p=logs/f'title_{ti}.log'
 with p.open('w') as f:rc=subprocess.call(['/root/miniconda3/bin/python','-u',str(r/'adapt/restore_round1_for_rerun.py'),str(ti)],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
 print('title',ti,'exit',rc,flush=True)
 return ti,rc
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
 results=list(ex.map(run,[i for i in range(1,20) if i!=13]))
(r/'round1_restore_batch_status.json').write_text(json.dumps(results))
