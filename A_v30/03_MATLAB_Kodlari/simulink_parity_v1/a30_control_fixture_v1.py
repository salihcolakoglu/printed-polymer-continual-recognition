"""Source-free hand-set fixtures, serialized only (no inference)."""
import csv,json
from pathlib import Path
def make(out):
 assert not out.exists();out.mkdir(parents=True)
 theta=[[0.]*52 for _ in range(522)];meta=[[0]*5 for _ in range(522)];refs=[[0.]*3 for _ in range(252)]
 mu=[10.,20.];sd=[2.,4.]
 oh=[[int(i==j) for j in range(8)] for i in range(4)]
 wh=[[0,1,0,1,0,1,0,1],[0,0,1,1,0,0,1,1],[0,1,1,0,0,1,1,0],[0,0,0,0,1,1,1,1]]
 theta[0][:12]=[.2,-.4,.6,.8,-1.,1.2,1.4,-1.6,.1,-.2,.3,-.4];meta[0]=[1,4,0,0,0]
 for mid,D,kind,book in [(2,4,2,0),(3,8,3,1),(4,8,3,2)]:
  # Non-symmetric matrices exercise MATLAB column-major theta orientation.
  p=[.11,-.23,.37,.41,-.53,.67,.71,-.83,.17,-.19,.29,-.31]
  p += [((j%7)-3)*.13 for j in range(4*D)]+[(j-2)*.09 for j in range(D)]
  theta[mid-1][:len(p)]=p;meta[mid-1]=[kind,D,book,0,0]
 theta[4][:12]=[0,4,-3,8,0,-4,6,2,2,2,3,4];meta[4]=[4,0,0,0,0]
 refs[:4]=[[0,0,1],[2,2,2],[-1,2,3],[3,-1,4]];meta[5]=[5,0,0,1,4]
 meta[6]=[2,4,0,0,0];meta[7]=[1,4,0,0,0];meta[8]=[3,8,1,0,0]
 meta[9]=[2,4,0,0,0];theta[9][0]=None
 meta[10]=[4,0,0,0,0];theta[10][:12]=[0]*8+[1,0,0,0]
 meta[11]=[5,0,0,1,1]
 # Stable sigmoid saturation without exponential overflow.
 meta[12]=[3,8,2,0,0];theta[12][44:52]=[1000,-1000,1000,-1000,700,-700,0,1]
 meta[13]=[5,0,0,2,5]
 refs[84:89]=[[None,0,1],[0,0,1],[1,1,2],[2,0,3],[0,2,4]]
 cases=[]
 def add(name,mid,mask=[1,1,1,1],raw=[12.,24.],valid=True):cases.append(dict(name=name,input=raw+[mid]+mask,valid=valid))
 for mid in range(1,7):add('asymmetric_full_'+str(mid),mid);add('pair_1_4_'+str(mid),mid,[1,0,0,1],[8.,26.])
 for mid in [7,8,9]:add('exact_tie_'+str(mid),mid,[0,1,0,1])
 add('proto_exact_tie',5,[1,1,0,0],[12.,16.]) # normalized[1,-1], equal centers0 and[2,-2]
 add('nn_exact_tie',6,[1,1,0,0])
 add('stable_sigmoid',13)
 add('three_candidate_ce',2,[1,1,0,1],[14.,18.])
 add('raw_nan',1,raw=[float('nan'),20.],valid=False)
 add('raw_inf',2,raw=[10.,float('inf')],valid=False)
 for mid in [0,523,1.5]:add('invalid_selector_'+str(mid),mid,valid=False)
 add('nonbinary_mask',1,[1,.5,0,1],valid=False)
 add('one_eligible',1,[1,0,0,0],valid=False)
 add('nonfinite_theta',10,valid=False)
 add('missing_proto_class',11,valid=False)
 add('missing_nn_class',12,valid=False)
 add('malformed_nn_reference_not_omitted',14,valid=False)
 bank=dict(schema='A30_SOURCE_FREE_FIXTURE_BANK_V1',fold=0,theta=theta,meta=meta,mean=mu,sd=sd,refs=refs,books=oh+wh)
 (out/'bank.json').write_text(json.dumps(bank,allow_nan=False),encoding='utf-8')
 with (out/'inputs.csv').open('w',newline='',encoding='utf-8') as f:csv.writer(f).writerows(c['input'] for c in cases)
 (out/'cases.json').write_text(json.dumps([dict(name=c['name'],valid=c['valid']) for c in cases],indent=2),encoding='utf-8')
 return len(cases)
if __name__=='__main__':
 import sys;print(make(Path(sys.argv[1])))
