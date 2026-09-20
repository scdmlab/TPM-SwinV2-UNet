from pathlib import Path
import json,hashlib,itertools
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parent;V=R.parent/'cbi_adaptation_v3_20260914';threadpool_limits(2)
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf-8')
def metrics(y,p,events):
 by={}
 for e in sorted(set(events)):
  d=p[events==e]-y[events==e];by[e]=dict(n=len(d),rmse=float(np.sqrt(np.mean(d*d))),mae=float(np.mean(abs(d))),bias=float(np.mean(d)))
 return dict(by_fire=by,macro={k:float(np.mean([v[k] for v in by.values()])) for k in ['rmse','mae','bias']})
m=json.loads((V/'manifest.json').read_text());a=np.load(R/'spectral_features21.npz');ids=a['ids'];y=a['y'];X=a['X'];assert X.shape[1]==21
assert json.loads((R/'spectral_sanity.json').read_text())['status']=='passed'
lookup={v:i for i,v in enumerate(ids)};ix={s:np.array([lookup[p['event']+'::'+p['id']] for p in m[s]]) for s in ['train','val','test']};events=np.array([i.split('::')[0] for i in ids])
for s in ix:assert np.array_equal(y[ix[s]],np.array([p['cbi'] for p in m[s]]))
assert not any(set(ix[s])&set(ix[t]) for s,t in [('train','val'),('train','test'),('val','test')])
specs=[]
for n in ['mean','median']:specs.append((n,0,[{}]))
for n in ['dNBR','RdNBR','dNBR2']:specs.append((n,0,[dict(alpha=x) for x in [1.,10.,100.,1000.]]))
specs.append(('SVR',0,[dict(C=c,epsilon=e) for c,e in itertools.product([1.,10.,100.],[.05,.2])]))
for seed in [1,2,3]:
 specs.append(('RF',seed,[dict(max_depth=d,min_samples_leaf=l) for d,l in itertools.product([3,6],[5,10])]))
 specs.append(('XGBoost',seed,[dict(max_depth=d,learning_rate=lr) for d,lr in itertools.product([2,3],[.03,.1])]))
save(R/'classical/plan.json',dict(specs=specs,fit_n=75,validation_n=46,test_n=28,selection='validation fire average RMSE only',feature_source=str(R/'spectral_features21.npz'),feature_sha256=json.loads((R/'spectral_sanity.json').read_text())['features_sha256']))
def predict(name,seed,param,indices,labels=y):
 xx=X[:,{'dNBR':18,'RdNBR':19,'dNBR2':20}[name]:{'dNBR':18,'RdNBR':19,'dNBR2':20}[name]+1] if name in ['dNBR','RdNBR','dNBR2'] else X
 if name in ['mean','median']:return np.full(len(indices),float((np.mean if name=='mean' else np.median)(labels[ix['train']])))
 if name in ['dNBR','RdNBR','dNBR2']:model=make_pipeline(StandardScaler(),Ridge(**param))
 elif name=='SVR':model=make_pipeline(StandardScaler(),SVR(kernel='rbf',gamma='scale',**param))
 elif name=='RF':model=RandomForestRegressor(n_estimators=200,max_features=1.,n_jobs=1,random_state=seed,**param)
 else:model=XGBRegressor(n_estimators=200,objective='reg:squarederror',subsample=.8,colsample_bytree=1.,min_child_weight=3,n_jobs=2,random_state=seed,**param)
 model.fit(xx[ix['train']],labels[ix['train']]);p=np.clip(model.predict(xx[indices]),0,3);assert np.isfinite(p).all();return p
# Every model family must be insensitive to held-out label changes.
changed=y.copy();changed[ix['test']]=3-changed[ix['test']];changed[ix['val']]=0
for n,s,grid in specs:
 p=predict(n,s,grid[0],ix['test']);q=predict(n,s,grid[0],ix['test'],changed);assert np.array_equal(p,q),(n,s)
save(R/'classical/sanity.json',dict(status='passed',heldout_label_invariance=True,families=len(specs)))
selection={}
for n,s,grid in specs:
 scores=[]
 for param in grid:
  p=predict(n,s,param,ix['val']);scores.append(dict(param=param,validation=metrics(y[ix['val']],p,events[ix['val']])) )
 selected=min(scores,key=lambda q:q['validation']['macro']['rmse']);selection[f'{n}_s{s}']=dict(name=n,seed=s,selected=selected,candidates=scores)
save(R/'classical/selection.json',selection)
results={};predictions={}
for key,v in selection.items():
 pred=predict(v['name'],v['seed'],v['selected']['param'],ix['test']);results[key]=metrics(y[ix['test']],pred,events[ix['test']]);predictions[key]=[dict(id=str(ids[i]),event=str(events[i]),cbi=float(y[i]),prediction=float(pred[j])) for j,i in enumerate(ix['test'])]
save(R/'classical/results.json',dict(status='complete',metrics=results,predictions=predictions))
print(json.dumps({k:v['macro'] for k,v in results.items()},indent=2))
