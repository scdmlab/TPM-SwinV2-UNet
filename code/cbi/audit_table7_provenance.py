"""Recompute historical and current CBI errors without fitting any model."""
from pathlib import Path
import json, math, statistics, hashlib, datetime
R=Path(__file__).resolve().parent
V=R.parent/'cbi_adaptation_v3_20260914'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=read(V/'manifest.json')
expected={(p['event'],p['id']):p['cbi'] for p in m['test']}
def metrics(rows):
 assert len(rows)==28 and {(r['event'],r['id']):r['cbi'] for r in rows}==expected
 by={}
 for event in sorted({r['event'] for r in rows}):
  errors=[r['prediction']-r['cbi'] for r in rows if r['event']==event]
  assert all(math.isfinite(x) for x in errors)
  by[event]={'n':len(errors),'rmse':math.sqrt(statistics.mean(x*x for x in errors)),'mae':statistics.mean(abs(x) for x in errors)}
 return {'by_fire':by,'macro':{k:statistics.mean(v[k] for v in by.values()) for k in ['rmse','mae']}}
out={'time':datetime.datetime.now().astimezone().isoformat(),'historical':{},'current':{},'comparisons':{}}
for variant in ['A04_T','A09_TPM']:
 old=V/'runs'/(variant+'_full_ft')
 res=read(old/'result.json');cfg=read(old/'config.json');init=read(old/'initialization.json')
 assert cfg['mode']=='full_ft' and cfg['seed']==1 and cfg['manifest_sha256']==sha(V/'manifest.json')
 assert cfg['train_sha256']==sha(V/'train.py') and cfg['data_sha256']==sha(V/'data.py')
 hist=[json.loads(x) for x in (old/'history.jsonl').read_text().splitlines()]
 assert len(hist)==40 and sum(x['steps'] for x in hist)==12200
 assert min(hist,key=lambda x:x['validation_macro_rmse'])['epoch']==res['best_epoch']
 mm=metrics(read(old/'test_predictions.json'))
 for e,v in mm['by_fire'].items():
  for k in ['rmse','mae']:assert abs(v[k]-res['test_metrics'][e][k])<1e-12
 out['historical'][variant]={'config':cfg,'initialization':init,'metrics':mm,'best_epoch':res['best_epoch'],'first_epoch':hist[0]}
 current_root=R if variant=='A04_T' else R/'remote_snapshot'
 for seed in [1,2,3]:
  run=current_root/'runs'/f'{variant}_full_ft_s{seed}'
  if not (run/'result.json').exists():continue
  rr=read(run/'result.json');cc=read(run/'config.json');ii=read(run/'initialization.json')
  assert cc['mode']=='full_ft' and cc['seed']==seed and cc['manifest_sha256']==cfg['manifest_sha256']
  assert ii['checkpoint_sha256']==init['checkpoint_sha256']
  mm=metrics(read(run/'test_predictions.json'))
  for k in ['rmse','mae']:assert abs(mm['macro'][k]-rr['test']['macro'][k])<1e-12
  hh=[json.loads(x) for x in (run/'history.jsonl').read_text().splitlines()]
  assert len(hh)==40 and sum(x['steps'] for x in hh)==12200
  assert min(hh,key=lambda x:x['validation']['macro']['rmse'])['epoch']==rr['best_epoch']
  out['current'][f'{variant}_s{seed}']={'metrics':mm,'best_epoch':rr['best_epoch'],'first_epoch':hh[0],'config':cc,'initialization':ii}
 new=out['current'][variant+'_s1']
 out['comparisons'][variant]={'same_manifest':True,'same_source_checkpoint':True,'same_recorded_torch_gpu':all(cfg[k]==new['config'][k] for k in ['device','torch']),'same_seed1_training_script_hash':cfg['train_sha256']==new['config']['train_sha256'],'old_best_epoch':res['best_epoch'],'new_best_epoch':new['best_epoch'],'seed1_rmse_delta':new['metrics']['macro']['rmse']-out['historical'][variant]['metrics']['macro']['rmse'],'old_first_loss':hist[0]['loss'],'new_first_loss':new['first_epoch']['loss']}
oldf=read(V/'frozen_cbi_matched/results.json')
out['old_frozen_metrics']=oldf['metrics']
out['old_frozen_settings']=oldf['sanity']['models']
out['new_frozen_settings']={v:read((R if v=='A04_T' else R/'remote_snapshot')/'frozen'/(v+'_result.json'))['feature_meta'] for v in ['A04_T','A09_TPM']}
out['summary']='Both tables recompute correctly from their own predictions. Historical single seed and current seed aggregate must not be conflated. TPM seed1 replay differs; causal source of trajectory difference remains unresolved.'
(R/'TABLE7_PROVENANCE_AUDIT.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'status':'prediction_and_provenance_checks_passed','comparisons':out['comparisons'],'historical':{k:v['metrics']['macro'] for k,v in out['historical'].items()},'current':{k:v['metrics']['macro'] for k,v in out['current'].items()}},indent=2))
