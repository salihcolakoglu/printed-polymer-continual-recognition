"""Structural projection of accepted saved states; no forward/training calls.
Canonical inputs remain unmodified. Expected outputs never enter the SLX bank.
"""
from pathlib import Path
import argparse,csv,hashlib,json

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,o):p.write_text(json.dumps(o,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')
def pin(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def colmajor(a):return [a[i][j] for j in range(len(a[0])) for i in range(len(a))]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--a29',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);x=ap.parse_args()
 assert not x.out.exists();x.out.mkdir(parents=True)
 csvp=x.a29/'02_Veriler/A_source_average_records_v1.csv'
 with csvp.open(encoding='utf-8-sig',newline='') as f:raw={r['row_id']:r for r in csv.DictReader(f)}
 assert len(raw)==104
 bp=x.a29/'07_Sonuclar/BASELINE_RUN_02/results/a29_baseline_result_v1.json';base=read(bp)
 jobs={j['job_id']:j for j in base['linear_jobs']};dets={d['state_id']:d for d in base['deterministic_states']}
 proof={'schema':'A30_STRUCTURAL_EXPORT_V1','inputs':[pin(csvp),pin(bp)],'folds':[],'training':False,'forward_execution':False}
 for fold in range(1,6):
  mp=x.a29/f'07_Sonuclar/CL_SOURCE_FOLD_0{fold}_RUN_01/a29cl_source_fold{fold}_v1.json';m=read(mp);proof['inputs'].append(pin(mp))
  theta=[];meta=[];mapping=[];query=[];expected=[];refrows=[[0.,0.,0.] for _ in range(252)]
  norm=next(n for n in base['normalizers'] if n['fold']==fold)
  assert next(n for n in m['normalizers'] if n['fold']==fold)==norm
  books=m['codebooks']['OH8']+m['codebooks']['WH8']
  fit={r['job_id']:r for r in m['unique_fits']};logical={}
  def add(p,kind,D,book=0,snap=0,nref=0,**identity):
   assert len(p)<=52;theta.append(p+[0.]*(52-len(p)));meta.append([kind,D,book,snap,nref]);mid=len(theta)
   mapping.append(dict(model_id=mid,fold=fold,**identity));return mid
  for l in m['logical_stages']:
   assert l['fold']==fold
   j=fit[l['job_id']];D=j['D'];book=0 if D==4 else (2 if 'WH' in l['arm'] else 1)
   assert D==4 or ('WH' in l['arm'] or 'OH' in l['arm'])
   mid=add(j['final_state']['theta'],2 if D==4 else 3,D,book,
    family='matched',logical_stage_id=l['id'],job_id=l['job_id'],seed=l['seed'],arm=l['arm'],stream=l['stream'],stage=l['stage'])
   logical[l['id']]=mid
  aliases={}
  for a in base['aliases']:
   if a['fold']!=fold:continue
   j=jobs[a['job_id']];state=j['end_state']
   mid=add(colmajor(state['W'])+state['b'],1,4,family='baseline',method=a['method'],seed=a['seed'],stream=a['stream'],stage=a['stage'],state_id=a['job_id'])
   aliases[(a['stream'],a['stage'],a['method'],a['seed'])]=mid
  deterministic={}
  for stage in range(1,4):
   d=dets[f'D_f{fold}_s{stage}'];nn=d['nn'];assert len(nn['X'])<=84
   for j,(xx,yy) in enumerate(zip(nn['X'],nn['y'])):refrows[(stage-1)*84+j]=xx+[yy]
   for stream in ['TIL','CIL']:
    for method in ['PROTO','NN1_FULL']:
     kind=4 if method=='PROTO' else 5
     p=colmajor(d['proto']['sumS'])+d['proto']['countn'] if kind==4 else []
     mid=add(p,kind,0,snap=stage,nref=len(nn['X']),family='baseline',method=method,seed=None,stream=stream,stage=stage,state_id=d['state_id'])
     deterministic[(stream,stage,method)]=mid
  assert len(theta)==522
  def append(pred,mid,family,index):
   r=raw[pred['parentrow']];mask=[int(c in pred['eligible']) for c in range(1,5)]
   assert len(pred['eligible'])==sum(mask)>=2
   query.append([float(r['E_MPa']),float(r['Hardness_ShoreD']),mid]+mask)
   expected.append(dict(family=family,source_prediction_index=index,model_id=mid,record=pred))
  for i,p in enumerate(m['predictions']):append(p,logical[p['logical_stage_id']],'matched',i)
  for i,p in enumerate(base['predictions']):
   if p['fold']!=fold:continue
   key=(p['stream'],p['stage'],p['method'])
   mid=aliases[key+(p['seed'],)] if p['method'].startswith('LIN_') else deterministic[key]
   assert mapping[mid-1]['state_id']==p['state_id'];append(p,mid,'baseline',i)
  assert len(query)==(10962 if fold==1 else 9135)
  qdir=x.out/f'fold_{fold}';qdir.mkdir()
  write(qdir/'bank.json',dict(schema='A30_FROZEN_INFERENCE_BANK_V1',fold=fold,theta=theta,meta=meta,mean=norm['mean'],sd=norm['sd'],refs=refrows,books=books))
  write(qdir/'model_map.json',mapping);write(qdir/'reference_predictions.json',expected)
  with (qdir/'inputs.csv').open('w',encoding='utf-8',newline='') as f:csv.writer(f).writerows(query)
  proof['folds'].append(dict(fold=fold,logical_models=len(theta),queries=len(query),files=[pin(p) for p in sorted(qdir.iterdir())]))
 write(x.out/'structural_export_receipt.json',proof)
 print(json.dumps(dict(folds=5,logical_models=2610,queries=47502,output=str(x.out)),ensure_ascii=False))
if __name__=='__main__':main()
