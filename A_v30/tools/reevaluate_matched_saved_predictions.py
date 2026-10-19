"""Read-only reevaluation of A29 matched saved states (Python standard library).
MIT; Copyright (c)2026 Salih Colakoglu and Mustafa Onat.
No fitting, optimizer updates, source-mode admission or network access.
Run: python tools/reevaluate_matched_saved_predictions.py
Tolerance2e-11; original MATLAB output authoritative, no bitwise identity claim.
"""
from pathlib import Path
import hashlib,json,math
ROOT=Path(__file__).resolve().parents[1]
TOL=2e-11
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sigmoid(x):
 if x>=0:return 1/(1+math.exp(-x))
 e=math.exp(x);return e/(1+e)
def forward(theta,z,D):
 assert len(theta)==12+5*D and len(z)==2
 h=[math.tanh(z[0]*theta[2*j]+z[1]*theta[2*j+1]+theta[8+j]) for j in range(4)]
 return [sum(h[k]*theta[12+4*j+k] for k in range(4))+theta[12+4*D+j] for j in range(D)]
def main():
 index=load(ROOT/'SAVED_MODEL_INDEX.json');checks=0;predictions=0;maxdiff=0.0;mismatches=[];folds=[]
 def close(a,b):
  nonlocal checks,maxdiff
  checks+=1
  if b is None:return a is None
  if not isinstance(a,(float,int)) or not math.isfinite(a):return False
  diff=abs(a-b);maxdiff=max(maxdiff,diff)
  return diff<=TOL*max(1,abs(b))
 for pin in index['matched']:
  path=ROOT/pin['file'];assert sha(path)==pin['sha256'],'Saved result hash mismatch'
  R=load(path);models={int(f['job_id']):f for f in R['unique_fits']};n=0
  assert R['failure_count']==0 and R['physical_validation']==0
  for p in R['predictions']:
   fit=models[int(p['job_id'])];D=int(fit['D']);theta=fit['final_state']['theta'];z=p['Z'];eligible=p['eligible'];assert eligible==sorted(set(eligible)) and len(eligible)>=2
   a=forward(theta,z,D)
   if D==4:
    mx=max(a[c-1] for c in eligible);e=[math.exp(a[c-1]-mx) for c in eligible];den=sum(e);scores=[v/den for v in e];outputs=[None]*4
    for c,v in zip(eligible,scores):outputs[c-1]=v
    dist=[None]*len(eligible)
   else:
    outputs=[sigmoid(v) for v in a];C=R['codebooks']['OH8' if p['arm'].startswith('OH8_') else 'WH8'];dist=[sum((outputs[j]-C[c-1][j])**2 for j in range(8))/8 for c in eligible];scores=[-v for v in dist]
   best=max(scores);ties=[eligible[i] for i,v in enumerate(scores) if v==best];ordered=sorted(scores,reverse=True);label=ties[0];gap=ordered[0]-ordered[1]
   ok=(label==p['label'] and (len(ties)>1)==p['tie'] and p['valid'] is True and p['failure'] is False);checks+=4
   for actual,saved in [(a,p['logits']),(scores,p['scores']),(outputs,p['outputs']),(dist,p['distances'])]:
    ok=ok and len(actual)==len(saved)
    for x,y in zip(actual,saved):
     if not close(x,y):ok=False
   if not close(gap,p['gap']):ok=False
   if not ok and len(mismatches)<20:mismatches.append(dict(fold=p['fold'],seed=p['seed'],arm=p['arm'],stream=p['stream'],stage=p['stage'],parentrow=p['parentrow'],saved_label=p['label'],reevaluated_label=label))
   n+=1;predictions+=1
  assert n==pin['predictions'];folds.append(dict(fold=pin['fold'],predictions=n))
 assert predictions==40950
 result=dict(schema='A29_S1_READONLY_INFERENCE_REEVALUATION_V1',status='PASS' if not mismatches else 'MISMATCH',folds=folds,predictions=predictions,numeric_and_decision_checks=checks,max_absolute_numeric_difference=maxdiff,tolerance=TOL,mismatches=mismatches,new_training=0,physical_validation=0,scope='Saved-state inference only; source training and exact original MATLAB bitwise execution are not performed.')
 print(json.dumps(result,indent=2));return 0 if not mismatches else 1
if __name__=='__main__':raise SystemExit(main())
