"""Rerun round2 and round3 only, from verified round1 graphs, with five isolated workers."""
import concurrent.futures,datetime,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path('/root/autodl-tmp/V16_run')
OLD_RUN='v16_loogle_longdep_q36plus_149_20261109'
RUN='v16_loogle_longdep_q36plus_r2rerun_20261109'
OUT=ROOT/'eval/experiments'/RUN
DB=ROOT/'eval/database'/RUN
CHECKPOINTS=ROOT/'eval/checkpoints'/RUN
LOGS=ROOT/'loogle_r2rerun_worker_logs'
TITLES=[i for i in range(20) if i!=13]

def emit(x):print(json.dumps(x,ensure_ascii=False),flush=True)

def snapshot(ti,phase):
    src=DB/str(ti); dst=CHECKPOINTS/f'title_{ti}'/phase
    if dst.exists():raise RuntimeError('Snapshot already exists: '+str(dst))
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(src,dst)
    (dst/'snapshot_complete.json').write_text(json.dumps({'title':ti,'phase':phase,'at':datetime.datetime.now().isoformat()}))

def run(ti):
    snapshot(ti,'round1')
    rows=json.loads((ROOT/f'loogle_longdep_v16_20261109_data/choices/title_{ti}.json').read_text())
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    code="import sys,json; from pathlib import Path; import adapt.run_loogle_longdep_v16 as m; m.RUN=sys.argv[1]; m.OUT=m.ROOT/'eval'/'experiments'/m.RUN; m.PHASES=(sys.argv[3],); print(json.dumps(m.run_title(int(sys.argv[2])),ensure_ascii=False),flush=True)"
    for phase in ('round2','round3'):
        log=LOGS/f'title_{ti}_{phase}.log'
        with log.open('ab',buffering=0) as f:
            p=subprocess.Popen([sys.executable,'-u','-c',code,RUN,str(ti),phase],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            emit({'title':ti,'phase':phase,'pid':p.pid,'status':'started'})
            rc=p.wait()
        done=len(list((OUT/f'title_{ti}'/phase).glob('q*.json')))
        emit({'title':ti,'phase':phase,'exit':rc,'done':done,'expected':len(rows)})
        if rc or done!=len(rows):return {'title':ti,'status':'incomplete','phase':phase,'exit':rc}
        snapshot(ti,phase)
    return {'title':ti,'status':'complete'}

if __name__=='__main__':
    for ti in TITLES:
        marker=DB/str(ti)/'round1_restore_verified.json'
        if not marker.is_file():raise RuntimeError('Missing verified round1 restore: '+str(ti))
    OUT.mkdir(parents=True,exist_ok=True);LOGS.mkdir(exist_ok=True)
    for ti in TITLES:
        source=ROOT/'eval/experiments'/OLD_RUN/f'title_{ti}'
        target=OUT/f'title_{ti}';target.mkdir(exist_ok=True)
        for phase in ('fresh','round1'):
            shutil.copytree(source/phase,target/phase)
            f=source/(phase+'_summary.json')
            if f.is_file():shutil.copy2(f,target/f.name)
    (OUT/'rerun_manifest.json').write_text(json.dumps({'source_run':OLD_RUN,'run':RUN,'rerun_phases':['round2','round3'],'workers':5,'model':'qwen3.6-plus','round1_restore':'source_round0_plus_recorded_round1_updates_verified_against_retained_historical_vectors','skipped_title':13},indent=2))
    emit({'status':'started','run':RUN,'workers':5,'titles':TITLES})
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures=[pool.submit(run,ti) for ti in TITLES]
        for future in concurrent.futures.as_completed(futures):
            try:result=future.result()
            except Exception as e:result={'status':'exception','error':repr(e)}
            results.append(result);emit(result)
    (OUT/'run_completion.json').write_text(json.dumps(results,indent=2))
    emit({'status':'finished','completed_titles':sum(x.get('status')=='complete' for x in results),'results':results})
