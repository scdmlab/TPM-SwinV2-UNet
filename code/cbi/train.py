from pathlib import Path
import sys,os,json,time,math,gc,importlib.util,argparse,shutil
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent.parent;V=R.parent/'cbi_adaptation_v3_20260914'
sys.path.insert(0,str(V));import data as D
s=importlib.util.spec_from_file_location('queue_source_trainer',P/'revision_2026/scripts/tpsm_train.py');T=importlib.util.module_from_spec(s);s.loader.exec_module(T)
torch=T.torch;nn=torch.nn;F=nn.functional;torch.set_num_threads(4)
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,p)
def config(variant,mode,seed):
 return dict(variant=variant,mode=mode,seed=seed,epochs=40,steps_per_epoch=305,protocol_sha256=D.sha(R/'PROTOCOL.md'),manifest_sha256=D.sha(V/'manifest.json'),train_sha256=D.sha(__file__),device=torch.cuda.get_device_name(),torch=torch.__version__)
class Model(nn.Module):
 def __init__(self,variant,mode,seed,mean):
  super().__init__();self.variant=variant;self.mode=mode;T.set_seed(seed);torch.backends.cudnn.benchmark=False
  self.internal=variant.startswith('A');self.mmoe=variant in ['A08_TM','A09_TPM']
  if self.internal:
   c=T.variant_cfg(variant);O=T.load_original(str(P/'wildfire-TPSM-SwinV2-UNet'))
   self.net=T.SwinUNetV2Flex(O,streams=3,unet=True,ppm=c['ppm'],scse=False,mmoe=c['mmoe'],experts=5,topk=2,gate_hint=True,num_classes=4,chans=[9,9,3])
   dim=768 if self.mmoe else 384;layers=list(self.net.head.towers) if self.mmoe else [self.net.head]
  else:
   self.net=T.SMPBaseline(variant,streams=3,num_classes=4,encoder='resnet34',chans=[9,9,3]);layers=[self.net.net.segmentation_head[0]];dim=layers[0].in_channels
  self.initialization=dict(mode=mode,feature_dim=dim)
  if mode!='scratch':
   reg=json.loads((R/'weights.json').read_text())[variant];p=Path(reg['path']);assert p.exists() and D.sha(p)==reg['sha256'];ck=torch.load(p,map_location='cpu',weights_only=False);self.net.load_state_dict(ck['state_dict'],strict=True);del ck
   self.initialization.update(checkpoint_sha256=reg['sha256'],source_adaptation_seeds='source seed1 fixed; CBI seed varies')
  torch.manual_seed(10000+seed);self.reg=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,32),nn.GELU(),nn.Linear(32,1));nn.init.normal_(self.reg[-1].weight,std=.001);nn.init.constant_(self.reg[-1].bias,float(np.log(mean/(3-mean))))
  self.captured={};self.handles=[]
  for k,l in enumerate(layers):
   def hook(mod,args,k=k):
    a=args[0][:len(self.w)].float()
    if a.shape[-2:]!=self.w.shape[-2:]:a=F.interpolate(a,size=self.w.shape[-2:],mode='bilinear',align_corners=False)
    self.captured[k]=torch.einsum('bchw,bhw->bc',a,self.w.float())
   self.handles.append(l.register_forward_pre_hook(hook))
  self.net.set_gate_temperature(.5);T.set_seed(seed);torch.backends.cudnn.benchmark=False
 def forward(self,xs,w):
  self.w=w;self.captured.clear();logits=self.net(*xs);z=torch.cat([self.captured[k] for k in sorted(self.captured)],dim=1);self.features=z;return logits,3*torch.sigmoid(self.reg(z).squeeze(1))
def batches(m,epoch,seed,limit=305):
 rng=np.random.default_rng(10000+epoch+1000*(seed-1));order=rng.permutation(610)
 for start in range(0,limit*2,2):
  xs=[];ws=[];ys=[];labs=[]
  for _ in range(2):
   p=m['train'][int(rng.integers(len(m['train'])))];a,w,y=D.field(p,int(rng.integers(8)));xs.append(a);ws.append(w);ys.append(y)
  for j in order[start:start+2]:
   a,l=D.D.source_example(m['source'][int(j)]['stem'],int(rng.integers(8)));xs.append(a);labs.append(l)
  yield [torch.from_numpy(np.stack([x[c] for x in xs])).cuda() for c in range(3)],torch.from_numpy(np.stack(ws)).cuda(),torch.tensor(ys,device='cuda'),torch.from_numpy(np.stack(labs)).cuda()
def optimizer(net):return torch.optim.AdamW([dict(params=net.net.parameters(),lr=1e-4),dict(params=net.reg.parameters(),lr=3e-4)],weight_decay=.01)
def step(net,opt,b):
 xs,w,y,label=b;opt.zero_grad(set_to_none=True)
 with torch.amp.autocast('cuda',dtype=torch.bfloat16):
  logits,pred=net(xs,w);assert logits.shape==(4,4,256,256) and pred.shape==(2,);cbi=F.huber_loss(pred.float(),y,delta=.5);seg=F.cross_entropy(logits[2:].float(),label,ignore_index=255);loss=cbi+.1*seg
 assert torch.isfinite(loss);loss.backward();norm=nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();return [float(x.detach()) for x in [loss,cbi,seg,norm]]
def summarize(rows):
 by={}
 for event in sorted({p['event'] for p in rows}):
  rr=[p for p in rows if p['event']==event];e=np.array([p['prediction']-p['cbi'] for p in rr]);by[event]=dict(n=len(rr),rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(abs(e))),bias=float(np.mean(e)))
 return dict(by_fire=by,macro={k:float(np.mean([b[k] for b in by.values()])) for k in ['rmse','mae','bias']})
def evaluate(net,points):
 net.eval();rows=[]
 with torch.inference_mode():
  for p in points:
   a,w,y=D.field(p)
   with torch.amp.autocast('cuda',dtype=torch.bfloat16):_,pred=net([torch.from_numpy(x[None]).cuda() for x in a],torch.from_numpy(w[None]).cuda())
   v=float(pred[0]);assert np.isfinite(v);rows.append(dict(event=p['event'],id=p['id'],cbi=y,prediction=v))
 return rows,summarize(rows)
def sanity(m,variant,mode):
 net=Model(variant,mode,1,float(np.mean([p['cbi'] for p in m['train']]))).cuda();net.train();opt=optimizer(net);first=next(net.net.parameters());before=first.detach().clone();head=net.reg[1].weight.detach().clone();losses=[]
 for b in batches(m,0,1,limit=3):losses.append(step(net,opt,b))
 assert float((first-before).abs().max())>0 and float((net.reg[1].weight-head).abs().max())>0
 gate_grad=None
 if net.mmoe:
  grads=[p.grad for n,p in net.net.named_parameters() if 'head.gates' in n and p.grad is not None];gate_grad=float(sum(g.abs().sum() for g in grads));assert gate_grad>0
 rows,metrics=evaluate(net,m['val'][:2]);assert len(rows)==2
 save(R/'sanity'/f'{variant}_{mode}.json',dict(status='passed',config=config(variant,mode,1),initialization=net.initialization,steps=3,losses=losses,gate_gradient_l1=gate_grad,parameters=sum(p.numel() for p in net.parameters()),peak_cuda_bytes=torch.cuda.max_memory_allocated()))
 print('SANITY_PASSED',variant,mode,flush=True)
def train(m,v,mode,seed):
 cfg=config(v,mode,seed);sn=json.loads((R/'sanity'/f'{v}_{mode}.json').read_text());assert sn['status']=='passed' and all(sn['config'][k]==cfg[k] for k in ['train_sha256','manifest_sha256','protocol_sha256'])
 out=R/'runs'/f'{v}_{mode}_s{seed}';out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==cfg
 else:save(out/'config.json',cfg)
 net=Model(v,mode,seed,float(np.mean([p['cbi'] for p in m['train']]))).cuda();opt=optimizer(net);save(out/'initialization.json',net.initialization);start=0;best=float('inf');resume=out/'resume.pth'
 if resume.exists():
  c=torch.load(resume,map_location='cpu',weights_only=False);assert c['config']==cfg;net.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);start=c['epoch'];best=c['best'];torch.set_rng_state(c['rng']);torch.cuda.set_rng_state_all(c['cuda_rng']);del c
 print('TRAIN_START',v,mode,seed,'pid',os.getpid(),flush=True)
 for epoch in range(start,40):
  assert shutil.disk_usage(R).free>8*1024**3,'less than8GiB free';begin=time.time();net.train();factor=.01+.99*.5*(1+math.cos(math.pi*epoch/39));opt.param_groups[0]['lr']=1e-4*factor;opt.param_groups[1]['lr']=3e-4*factor
  vals=np.array([step(net,opt,b) for b in batches(m,epoch,seed)]);assert vals.shape==(305,4)
  rows,metrics=evaluate(net,m['val']);score=metrics['macro']['rmse'];selected=score<best
  if selected:
   best=score;torch.save(dict(config=cfg,epoch=epoch+1,model=net.state_dict(),validation=metrics),out/'best.pth');save(out/'best_validation_predictions.json',rows)
  record=dict(epoch=epoch+1,steps=305,loss=float(vals[:,0].mean()),cbi_huber=float(vals[:,1].mean()),source_ce=float(vals[:,2].mean()),max_grad_norm=float(vals[:,3].max()),validation=metrics,best_validation_rmse=best,seconds=time.time()-begin,selected=selected)
  with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  save(out/'progress.json',dict(status='training',pid=os.getpid(),**record));print('EPOCH',v,mode,seed,json.dumps(record),flush=True)
  tmp=out/'resume.tmp.pth';torch.save(dict(config=cfg,epoch=epoch+1,model=net.state_dict(),optimizer=opt.state_dict(),best=best,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),tmp);os.replace(tmp,resume)
 c=torch.load(out/'best.pth',map_location='cpu',weights_only=False);net.load_state_dict(c['model']);rows,metrics=evaluate(net,m['test']);save(out/'test_predictions.json',rows)
 history=[json.loads(s) for s in (out/'history.jsonl').read_text().splitlines()];assert len(history)==40 and sum(x['steps'] for x in history)==12200
 # Independent recomputation from serialized predictions is part of completion.
 assert summarize(json.loads((out/'test_predictions.json').read_text()))==metrics
 save(out/'result.json',dict(status='complete',config=cfg,best_epoch=c['epoch'],validation=c['validation'],test=metrics,best_sha256=D.sha(out/'best.pth'),completed_updates=12200));save(out/'progress.json',dict(status='complete',epoch=40,best_epoch=c['epoch']))
 if resume.exists():resume.unlink()
 print('TRAIN_COMPLETE',v,mode,seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--variant',required=True);p.add_argument('--mode',choices=['scratch','full_ft'],default='full_ft');p.add_argument('--seed',type=int,default=1);p.add_argument('--sanity',action='store_true');a=p.parse_args()
 assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>10*1024**3;torch.cuda.set_per_process_memory_fraction(.8)
 m=D.load_manifest();q=json.loads((R/'queue.json').read_text());assert q['protocol_sha256']==D.sha(R/'PROTOCOL.md') and q['manifest_sha256']==D.sha(V/'manifest.json')
 verified=json.loads((V/'data_verified.json').read_text());assert verified['manifest_sha256']==D.sha(V/'manifest.json')
 if a.sanity:sanity(m,a.variant,a.mode)
 else:train(m,a.variant,a.mode,a.seed)
