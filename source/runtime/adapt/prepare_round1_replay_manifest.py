import json,re,ast,collections
from pathlib import Path
ROOT=Path('/root/autodl-tmp/V16_run')
OLD=ROOT/'eval/experiments/v16_loogle_longdep_q36plus_149_20261109'
DB=ROOT/'eval/database/v16_loogle_longdep_q36plus_149_20261109'
results=[]
for d in OLD.glob('title_*'):
 for p in (d/'round1').glob('q*.json'):
  x=json.loads(p.read_text());x['_dir']=str(d);results.append(x)
queries=sorted({x['question'] for x in results},key=len,reverse=True)
events=collections.defaultdict(list)
for p in sorted(DB.glob('*/log_*.log'),key=lambda p:p.name):
 for ix,block in enumerate(re.split('PathFinder - INFO - Get Query: ',p.read_text(errors='replace'))[1:]):
  q=next((q for q in queries if block.startswith(q+'\n')),None)
  if q is None:continue
  es=re.findall(r'PathFinder - DEBUG - Enhance Edges:\s*\n([^\n]+)',block);ps=re.findall(r'PathFinder - DEBUG - Punish Edges:\s*\n([^\n]+)',block)
  if len(es)!=1 or len(ps)!=1:continue
  try:e=ast.literal_eval(es[0]);pun=ast.literal_eval(ps[0])
  except:continue
  writes=re.findall(r'Database - INFO - Enhance Path (.*)\.\s*$',block,re.M)
  events[q].append({'file':str(p),'ix':ix,'inner':e,'punish':pun,'writes':writes,'block':block})
missing=[];amb=[];manifest=[]
for x in sorted(results,key=lambda x:(x['title_index'],x['question_index'])):
 ti=x['title_index'];qi=x['question_index'];d=Path(x['_dir']);trace=json.loads((d/f'round1/traces/q{qi}/c2_trace.json').read_text());outer=(x.get('memory_update') or {}).get('enhanced_edges',[])
 expected=set()
 for field in ['paths_by_chunk','paths_by_graph']:
  for routes in trace.get(field,{}).values():
   for route in routes:
    for ed in route:expected.add((str(ed['type']),str(ed['id'])))
 candidates=[]
 for e in events[x['question']]:
  if set(map(tuple,e['inner']))!=expected:continue
  if e['writes'] != [a[1] for a in e['inner']]+[a[1] for a in outer]:continue
  candidates.append(e)
 if not candidates:
  missing.append((ti,qi,len(events[x['question']]),len(expected),len(outer)))
  continue
 candidates.sort(key=lambda e:(int(Path(e['file']).parent.name)!=ti,Path(e['file']).name,e['ix']))
 if len({str(e['punish']) for e in candidates})>1:amb.append((ti,qi,len(candidates)))
 e=candidates[0];manifest.append({'title':ti,'qi':qi,'query':x['question'],'inner_enhance':e['inner'],'inner_punish':e['punish'],'outer_enhance':outer,'outer_punish':(x.get('d1') or {}).get('d1_applied',[]),'source_log':e['file'],'source_block':e['ix']})
print('matched',len(manifest),'missing',missing,'ambiguous',amb)
(ROOT/'round1_replay_manifest_candidate.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
(ROOT/'round1_replay_match_audit.json').write_text(json.dumps({'matched':len(manifest),'missing':missing,'ambiguous':amb}))
