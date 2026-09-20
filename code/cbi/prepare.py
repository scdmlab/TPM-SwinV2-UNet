from pathlib import Path
import json,hashlib,csv
R=Path(__file__).resolve().parent;P=R.parent.parent
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2),encoding='utf-8')
V=R.parent/'cbi_adaptation_v3_20260914';m=json.loads((V/'manifest.json').read_text());assert [len(m[k]) for k in ['train','val','test']]==[75,46,28]
jobs=json.loads((R.parent/'cbi_field_overnight_20260910/frozen_models.json').read_text())['jobs']
registry={}
for v in ['A04_T','A05_TP','A08_TM','A09_TPM']:
 j=next(j for j in jobs if j['variant']==v);assert Path(j['path']).exists() and sha(j['path'])==j['sha256'];registry[v]=dict(path=j['path'],sha256=j['sha256'],arch='internal')
for arch in ['unet','deeplabv3plus','segformer']:
 p=R.parent/'runs'/('D_'+arch+'_s1')/'best.pth';assert p.exists();registry[arch]=dict(path=str(p),sha256=sha(p),arch=arch,encoder='resnet34')
save(R/'weights.json',registry)
queue=[]
for seed in [1,2,3]:
 for v in ['A05_TP','A04_T','A08_TM','A09_TPM','unet','deeplabv3plus','segformer']:
  queue.append(dict(id=f'{v}_full_ft_s{seed}',variant=v,mode='full_ft',seed=seed,host='remote' if v in ['A08_TM','A09_TPM'] else 'local',status='queued'))
 for v in ['A04_T','A09_TPM']:queue.append(dict(id=f'{v}_scratch_s{seed}',variant=v,mode='scratch',seed=seed,host='remote' if v=='A09_TPM' else 'local',status='queued'))
save(R/'queue.json',dict(protocol_sha256=sha(R/'PROTOCOL.md'),manifest_sha256=sha(V/'manifest.json'),jobs=queue))
log=P/'EXPERIMENT_LOG.csv'
with log.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.reader(f))
assert all(len(r)==10 for r in rows if r);known={r[0] for r in rows[1:]}
new=[]
for j in queue:
 key='CBI_QUEUE_20260915_'+j['id']
 if key not in known:new.append([key,j['variant'],'CBI_V3_75_46_28',f"{j['mode']};adaptation_seed{j['seed']};40epochs;305steps;host={j['host']}",'development extension; same nominal support; external comparisons included','','',str(R/'runs'/j['id']),'queued','2026-09-15'])
for key,model in [('classical','ridge_RF_SVR_XGBoost'),('frozen','T_TP_TM_TPM_frozen_ridge'),('paired','paired_sample_error_analysis'),('gates','nominal_support_gate_analysis')]:
 eid='CBI_QUEUE_20260915_'+key
 if eid not in known:new.append([eid,model,'CBI_V3_75_46_28','fixed protocol; no test tuning','report only; no manuscript edits','','',str(R/key),'queued','2026-09-15'])
with log.open('a',encoding='utf-8',newline='') as f:csv.writer(f).writerows(new)
print(json.dumps({'neural_runs':len(queue),'log_rows_added':len(new),'weights':{k:v['sha256'] for k,v in registry.items()}},indent=2))
