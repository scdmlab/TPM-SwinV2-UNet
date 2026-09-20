"""Audit collected predictions and summarize only completed experiments."""
from pathlib import Path
import json,datetime,csv,hashlib
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent.parent;remote=R/'remote_snapshot'
q=json.loads((R/'queue.json').read_text());m=json.loads((R.parent/'cbi_adaptation_v3_20260914/manifest.json').read_text())
expected={(p['event'],p['id']):p['cbi'] for p in m['test']}
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def metrics(rows):
 by={}
 for e in sorted({p['event'] for p in rows}):
  d=np.array([p['prediction']-p['cbi'] for p in rows if p['event']==e]);by[e]=dict(n=len(d),rmse=float(np.sqrt((d*d).mean())),mae=float(abs(d).mean()),bias=float(d.mean()))
 d=np.array([p['prediction']-p['cbi'] for p in rows])
 return dict(by_fire=by,macro={k:float(np.mean([v[k] for v in by.values()])) for k in ['rmse','mae','bias']},pooled=dict(n=len(d),rmse=float(np.sqrt((d*d).mean())),mae=float(abs(d).mean()),bias=float(d.mean())))
def verify(rows):
 assert len(rows)==28 and len({(r['event'],r['id']) for r in rows})==28
 assert {(r['event'],r['id']):r['cbi'] for r in rows}==expected
 assert all(np.isfinite(r['prediction']) and 0<=r['prediction']<=3 for r in rows)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
preds={};done={};status={};summaries={};flags=[]
for j in q['jobs']:
 root=R if j['host']=='local' else remote;run=root/'runs'/j['id'];rp=run/'result.json'
 if not rp.exists():
  progress=run/'progress.json';status[j['id']]=json.loads(progress.read_text()) if progress.exists() else dict(status='queued')
  controller=root/(j['host']+'_queue.json')
  if controller.exists():
   cs=json.loads(controller.read_text())
   if cs.get('current')==j['id']:status[j['id']]['status']=cs['status']
  continue
 res=json.loads(rp.read_text());rows=json.loads((run/'test_predictions.json').read_text());verify(rows);mm=metrics(rows)
 for k in ['rmse','mae','bias']:assert np.isclose(mm['macro'][k],res['test']['macro'][k],rtol=0,atol=1e-12)
 h=[json.loads(s) for s in (run/'history.jsonl').read_text().splitlines()]
 assert len(h)==40 and sum(s['steps'] for s in h)==12200 and res['completed_updates']==12200
 assert min(h,key=lambda s:s['validation']['macro']['rmse'])['epoch']==res['best_epoch']
 assert all(np.isfinite(s['loss']) for s in h)
 assert res['config']['manifest_sha256']==q['manifest_sha256']
 if j['host']=='local':assert sha(run/'best.pth')==res['best_sha256']
 else:assert json.loads((remote/'remote_checkpoint_audit.json').read_text())[j['id']]['best_sha256']==res['best_sha256']
 status[j['id']]=dict(status='complete',best_epoch=res['best_epoch']);done[j['id']]=res;preds[j['id']]=rows;summaries[j['id']]=mm
 ratio=mm['macro']['rmse']/res['validation']['macro']['rmse']
 if ratio<.5 or ratio>2:flags.append(dict(run=j['id'],type='validation_test_RMSE_ratio_outside_0.5_2_review',ratio=ratio))
for v in ['A04_T','A05_TP','A08_TM','A09_TPM','unet','deeplabv3plus','segformer']:
 root=R if v in ['A04_T','A05_TP'] else remote;p=root/'frozen'/(v+'_result.json')
 if p.exists():
  z=json.loads(p.read_text());verify(z['predictions']);preds[v+'_frozen']=z['predictions'];summaries[v+'_frozen']=metrics(z['predictions'])
classical=json.loads((R/'classical/results.json').read_text())
for k,rr in classical['predictions'].items():
 rows=[dict(r,id=r['id'].split('::',1)[1]) for r in rr];verify(rows);preds[k]=rows;summaries[k]=metrics(rows)
groups={}
for key in summaries:
 base=key.rsplit('_s',1)[0] if key.rsplit('_s',1)[-1].isdigit() else key
 groups.setdefault(base,[]).append(key)
stats={g:dict(n_runs=len(keys),macro={metric:dict(mean=float(np.mean([summaries[k]['macro'][metric] for k in keys])),sd=float(np.std([summaries[k]['macro'][metric] for k in keys],ddof=1)) if len(keys)>1 else None) for metric in ['rmse','mae','bias']}) for g,keys in groups.items()}
paired={}
for mode in ['full_ft','scratch','frozen']:
 pairs=[]
 for seed in ([None] if mode=='frozen' else [1,2,3]):
  suffix=mode if seed is None else mode+'_s'+str(seed);a='A04_T_'+suffix;b='A09_TPM_'+suffix
  if a not in preds or b not in preds:continue
  aa={(r['event'],r['id']):r for r in preds[a]};bb={(r['event'],r['id']):r for r in preds[b]}
  rr=[dict(event=e,id=i,cbi=aa[e,i]['cbi'],T=aa[e,i]['prediction'],TPM=bb[e,i]['prediction'],absolute_error_reduction=abs(aa[e,i]['prediction']-aa[e,i]['cbi'])-abs(bb[e,i]['prediction']-bb[e,i]['cbi'])) for e,i in expected]
  pairs.append(dict(seed=seed,macro_rmse_reduction=summaries[a]['macro']['rmse']-summaries[b]['macro']['rmse'],macro_mae_reduction=summaries[a]['macro']['mae']-summaries[b]['macro']['mae'],rows=rr))
 paired[mode]=pairs
contrasts=[]
for seed in [1,2,3]:
 ids=[v+'_full_ft_s'+str(seed) for v in ['A04_T','A05_TP','A08_TM','A09_TPM']]
 if not all(k in summaries for k in ids):continue
 a,b,c,d=[summaries[k]['macro']['rmse'] for k in ids]
 contrasts.append(dict(seed=seed,ppm_error_reduction_without_mmoe=a-b,ppm_error_reduction_with_mmoe=c-d,mmoe_error_reduction_without_ppm=a-c,mmoe_error_reduction_with_ppm=b-d,interaction_error_contrast=d-c-b+a))
ranges={}
for key,rows in preds.items():
 ranges[key]={}
 for label,lo,hi in [('CBI_lt1',0,1),('CBI_1to2',1,2),('CBI_ge2',2,3.001)]:
  rr=[r for r in rows if lo<=r['cbi']<hi]
  if rr:ranges[key][label]=metrics(rr)
gate_summary={}
for p in sorted((remote/'gates').glob('*.json')):
 z=json.loads(p.read_text());rr=[r for r in z['rows'] if (r['event'],r['id']) in expected];assert len(rr)==28
 weights=np.array([r['gate_weights'] for r in rr]);ent=np.array([r['mean_pixel_gate_entropy'] for r in rr]);assert weights.shape==(28,4,5) and np.isfinite(ent).all()
 gate_summary[p.stem]=dict(n=28,mean_gate_weights=weights.mean(0).tolist(),mean_pixel_entropy=ent.mean(0).tolist())
 # Mean activation/routing is descriptive; no semantic expert names or causal claim.
 if 'prediction' in rr[0]:
  error=np.array([abs(r['prediction']-r['cbi']) for r in rr]);entropy=ent.mean(1)
  gate_summary[p.stem]['pearson_entropy_abs_error']=float(np.corrcoef(error,entropy)[0,1]) if error.std()>0 and entropy.std()>0 else None
neural_total=len(q['jobs'])
local_total=sum(j['host']=='local' for j in q['jobs']);remote_total=neural_total-local_total
save(R/'paired/analysis.json',dict(status='complete' if len(done)==neural_total and len(gate_summary)==11 and len([k for k in preds if k.endswith('_frozen')])>=4 else 'partial',paired=paired,contrasts=contrasts,ranges=ranges,gate_summary=gate_summary,flags=flags,interpretation='descriptive paired differences; adaptation seed variability; no significance claim from three seeds or spatially correlated plots'))
save(R/'summary.json',dict(time=datetime.datetime.now().astimezone().isoformat(),neural_complete=len(done),neural_total=neural_total,status=status,metrics=summaries,seed_summary=stats,flags=flags))
lines=['# CBI supplementary experiment report','',f'Updated: {datetime.datetime.now().astimezone().isoformat()}',f'Neural runs completed and audited: {len(done)}/{neural_total}. Local{local_total} / remote{remote_total}.','', '75 training / 46 validation / 28 test locations; 610 auxiliary source windows. Nominal30m support remains unverified at individual plots. Retrospective development comparisons on previously examined test locations. No manuscript changes.','', '| Model / workflow | Completed runs | Macro RMSE mean (SD) | Macro MAE mean (SD) |','|---|---:|---:|---:|']
def val(z):return f"{z['mean']:.4f}"+(f" ({z['sd']:.4f})" if z['sd'] is not None else '')
for g,z in stats.items():
 name=g.replace('segformer','SegFormer decoder / ResNet-34').replace('deeplabv3plus','DeepLabv3+ / ResNet-34').replace('unet','U-Net / ResNet-34')
 lines.append(f"| {name} | {z['n_runs']} | {val(z['macro']['rmse'])} | {val(z['macro']['mae'])} |")
lines+=['','All completed results retained. Macro averages give each of the two test fires equal weight. Classical comparisons use validation-only selection; neural means and SD describe CBI adaptation seeds, with source initialization fixed at seed1. A one-run result has no SD estimate.','', 'Detailed per-fire/pooled errors: summary.json. Per-plot paired errors, fixed CBI-range summaries, 2x2 module contrasts and routing summaries: paired/analysis.json.','', 'Audit flags: '+json.dumps(flags), '', 'Remaining work: '+', '.join(k for k,v in status.items() if v['status']!='complete')]
(R/'RESULTS_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
# Scientific figures only when new neural/frozen paired results exist.
if any(paired.values()):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 for mode,ps in paired.items():
  if not ps:continue
  fig,axs=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
  for ax,model in zip(axs,['T','TPM']):
   obs=[r['cbi'] for r in ps[0]['rows']];pred=np.mean([[r[model] for r in p['rows']] for p in ps],axis=0)
   ax.scatter(obs,pred,s=30,alpha=.8);ax.plot([0,3],[0,3],color='grey',linestyle='--');ax.set(xlim=(0,3),ylim=(0,3),xlabel='Observed CBI',ylabel='Predicted CBI',title=f'{model}: {mode} ({len(ps)} runs)');ax.set_aspect('equal')
  fig.savefig(R/'paired'/(mode+'_scatter.png'),dpi=200);plt.close(fig)
with (P/'EXPERIMENT_LOG.csv').open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;records=list(reader)
for rec in records:
 prefix='CBI_QUEUE_20260915_'
 if not rec['experiment_id'].startswith(prefix):continue
 key=rec['experiment_id'][len(prefix):]
 if key in status:
  rec['status']=status[key]['status'];j=next(j for j in q['jobs'] if j['id']==key);rec['config']=f"{j['mode']};adaptation_seed{j['seed']};40epochs;305steps;host={j['host']}"
  if key in summaries:rec['change']='macro_RMSE='+str(summaries[key]['macro']['rmse'])+';macro_MAE='+str(summaries[key]['macro']['mae'])+'; report only'
 elif key=='classical':rec['status']='complete';rec['change']='12 selected model/seed evaluations; validation-only tuning; results in classical/results.json'
 elif key=='frozen':rec['status']='complete' if sum(k.endswith('_frozen') for k in preds)>=4 else 'queued'
 elif key=='external_frozen':rec['status']='complete' if all(v+'_frozen' in preds for v in ['unet','deeplabv3plus','segformer']) else 'queued'
 elif key=='gates':rec['status']='complete' if len(gate_summary)==11 else 'queued'
 elif key=='paired':rec['status']='complete' if len(done)==neural_total and len(paired['frozen'])==1 and len(gate_summary)==11 else 'pending_training'
tmp=P/'EXPERIMENT_LOG.csv.tmp'
with tmp.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
tmp.replace(P/'EXPERIMENT_LOG.csv')
print('REPORT_UPDATED',len(done),f'of{neural_total} neural runs audited; classical complete')
