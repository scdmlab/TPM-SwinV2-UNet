"""Run frozen features and gates after the machine's training queue exits cleanly."""
from pathlib import Path
import sys,json,time,os,subprocess
R=Path(__file__).resolve().parent
host=sys.argv[1];assert host in ['local','remote']
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));os.replace(q,p)
state=R/(host+'_post.json')
try:
 save(state,dict(status='waiting_for_training_queue',pid=os.getpid()))
 while True:
  if (R/'STOP').exists():raise RuntimeError('STOP file present')
  q=R/(host+'_queue.json')
  if q.exists():
   s=json.loads(q.read_text())['status']
   if s=='complete':break
   if s in ['failed','paused']:raise RuntimeError('training queue '+s)
  time.sleep(30)
 import train as T
 import numpy as np
 torch=T.torch;assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.8)
 m=T.D.load_manifest();variants=['A04_T','A05_TP'] if host=='local' else ['A08_TM','A09_TPM']
 points=[p for s in ['train','val','test'] for p in m[s]]
 def inspect(net,ps,features=False):
  collected={};handles=[];out=[];feat=[]
  if net.mmoe:
   def gates(module,args,value):
    g=value[1]['gate_probs'].float();w=net.w.float();assert g.shape[2]==5 and g.shape[-2:]==w.shape[-2:]
    assert torch.allclose(g.sum(2),torch.ones_like(g.sum(2)),atol=.03,rtol=0)
    collected['gate_weights']=torch.einsum('btehw,bhw->bte',g,w)[0].cpu().tolist()
    entropy=-(g*g.clamp_min(1e-12).log()).sum(2)
    collected['mean_pixel_gate_entropy']=torch.einsum('bthw,bhw->bt',entropy,w)[0].cpu().tolist()
   handles.append(net.net.head.register_forward_hook(gates))
  net.eval()
  with torch.inference_mode():
   for p in ps:
    a,w,y=T.D.field(p)
    with torch.amp.autocast('cuda',dtype=torch.bfloat16):
     _,pred=net([torch.from_numpy(x[None]).cuda() for x in a],torch.from_numpy(w[None]).cuda())
    z=net.features[0].float().cpu().numpy().copy();assert np.isfinite(z).all()
    if features:feat.append(z)
    out.append(dict(id=p['id'],event=p['event'],cbi=y,**({} if features else dict(prediction=float(pred[0]))),**collected))
  for h in handles:h.remove()
  return out,np.array(feat)
 for v in variants:
  p=R/'frozen'/(v+'_features.npz');meta=p.with_suffix('.json')
  if meta.exists():assert json.loads(meta.read_text())['sha256']==T.D.sha(p);continue
  save(state,dict(status='extracting_frozen',current=v,pid=os.getpid()))
  net=T.Model(v,'full_ft',1,float(np.mean([p['cbi'] for p in m['train']]))).cuda()
  small,z=inspect(net,points[:2],True);assert z.shape==(2,net.initialization['feature_dim'])
  rows,X=inspect(net,points,True);assert X.shape[0]==149
  p.parent.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(p,X=X,y=[p['cbi'] for p in points],ids=[p['event']+'::'+p['id'] for p in points])
  save(meta,dict(status='complete',sha256=T.D.sha(p),initialization=net.initialization,manifest_sha256=T.D.sha(T.V/'manifest.json'),gate_temperature=.5,autocast='bfloat16',sanity_first_two=True))
  if net.mmoe:save(R/'gates'/(v+'_source.json'),dict(status='complete',checkpoint_sha256=net.initialization['checkpoint_sha256'],rows=rows))
  del net;torch.cuda.empty_cache()
 for j in json.loads((R/'queue.json').read_text())['jobs']:
  if j['host']!=host or j['variant'] not in ['A08_TM','A09_TPM']:continue
  p=R/'gates'/(j['id']+'.json')
  if p.exists():continue
  save(state,dict(status='analyzing_gates',current=j['id'],pid=os.getpid()))
  run=R/'runs'/j['id'];result=json.loads((run/'result.json').read_text());assert T.D.sha(run/'best.pth')==result['best_sha256']
  net=T.Model(j['variant'],j['mode'],j['seed'],float(np.mean([p['cbi'] for p in m['train']]))).cuda()
  ck=torch.load(run/'best.pth',map_location='cpu',weights_only=False);net.load_state_dict(ck['model'],strict=True);del ck
  rows,_=inspect(net,m['test']);original=json.loads((run/'test_predictions.json').read_text())
  assert all(r['id']==o['id'] and r['event']==o['event'] and abs(r['prediction']-o['prediction'])<1e-6 for r,o in zip(rows,original))
  save(p,dict(status='complete',best_sha256=result['best_sha256'],rows=rows,interpretation='descriptive routing; task towers originate from source classes, not CBI strata'))
  del net;torch.cuda.empty_cache()
 cpu=sys.executable if host=='remote' else 'C:/Users/caiku/AppData/Local/Programs/Python/Python312/python.exe'
 subprocess.run([cpu,str(R/'frozen_ridge.py'),host],check=True)
 save(state,dict(status='complete',variants=variants))
 print('POST_GPU_COMPLETE',host,flush=True)
except Exception as e:save(state,dict(status='failed',error=repr(e)));raise
