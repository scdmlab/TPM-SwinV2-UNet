from pathlib import Path
import json, hashlib
import numpy as np
R=Path(__file__).resolve().parent
B=R.parent/'cbi_v2_support_sensitivity_20260912'
m=json.loads((B/'manifest.json').read_text())
a=np.load(B/'spectral_features.npz'); X=[]; cache={}
for p in m['points']:
 e=p['event']
 if e not in cache:
  q=np.load(R.parent/'cbi_field_overnight_20260910/imagery_v2'/e.replace(' ','_')/'arrays.npz')
  cache[e]={k:q[k] for k in ['pre','post','valid']}
 q=cache[e]; pre=[];post=[];w=[]
 for rr,cc,v in p['nominal_support_weights']:
  rr=int(rr)-128+p['row'];cc=int(cc)-128+p['col'];assert q['valid'][rr,cc]
  pre.append(q['pre'][:,rr,cc].astype(float));post.append(q['post'][:,rr,cc].astype(float));w.append(v)
 pre=np.array(pre);post=np.array(post);w=np.array(w)
 def ratio(z,i):
  den=z[:,i]+z[:,8];assert (den>0).all();return (z[:,i]-z[:,8])/den
 n=ratio(pre,6);dn=n-ratio(post,6);rd=dn/np.sqrt(np.maximum(abs(n),.001));d2=ratio(pre,7)-ratio(post,7)
 X.append(w@np.c_[pre/10000,post/10000,dn,rd,d2])
X=np.array(X);assert X.shape==(182,21) and np.isfinite(X).all()
assert np.allclose(X[:,:20],a['X'],rtol=0,atol=1e-12)
assert np.array_equal(a['ids'],np.array([p['event']+'::'+p['id'] for p in m['points']]))
np.savez_compressed(R/'spectral_features21.npz',X=X,y=a['y'],ids=a['ids'])
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
(R/'spectral_sanity.json').write_text(json.dumps(dict(status='passed',shape=list(X.shape),old20_max_difference=float(abs(X[:,:20]-a['X']).max()),features_sha256=sha(R/'spectral_features21.npz'),old_features_sha256=sha(B/'spectral_features.npz'),new_column='area weighted dNBR2; per-pixel (B11-B12)/(B11+B12) pre minus post'),indent=2))
print('SPECTRAL_SANITY_PASSED',X.shape)
