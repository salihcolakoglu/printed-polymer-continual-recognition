"""Six static A29 figures from accepted saved data only; no learning/refits.
Public path-only adapter of the manuscript figure generator. No fitting.
Requires reportlab and pypdf. Optional pdftoppm on PATH renders PNGs.
Original numerical plotting logic is retained; this adapter was statically
checked, not executed during supplement assembly. Existing supplied figures
were generated and visually reviewed using the original recorded builder.
"""
from pathlib import Path
import csv,json,hashlib,math,statistics,subprocess,sys,shutil
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor,Color,black,white
from pypdf import PdfReader

A=Path(__file__).resolve().parents[1]
OUT=A/'09_Sekiller/reproduced_from_S1'
REPORT=A/'07_Sonuclar/CL_SOURCE_DERIVED_REPORT_01/A29_Matched_CL_Source_Rollup_v2.json'
DATA=A/'02_Veriler/A_source_average_records_v1.csv'
POPPLER=shutil.which('pdftoppm')  # Optional PNG rendering; PDFs/CSVs always generated.
SEEDS=[101,211,307,401,503]
CW=6.8*72
COLORS=['#2374AB','#D47718','#278263','#9A4B9A','#626262']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
assert sha(REPORT)=='8350ca9d46a3194f433ac74b5a2c6e7e5228595cc912ba2132aef4f1b4dd66f2'
assert sha(DATA)=='6d3902c59e696a53cd36e7bb9a42df1bac1fd9216737e34bfc90bb53b07c70a4'
R=read(REPORT);D=list(csv.DictReader(DATA.open(encoding='utf-8-sig')))
assert len(D)==104 and len({r['group_order'] for r in D})==26
assert not OUT.exists(),'Preserve prior figure versions.'
OUT.mkdir(parents=True)

def text(c,x,y,s,size=9,font='Times-Roman',align='left',color=black):
 c.setFillColor(color);c.setFont(font,size)
 {'left':c.drawString,'right':c.drawRightString,'center':c.drawCentredString}[align](x,y,str(s))
def line(c,x1,y1,x2,y2,col='#BDBDBD',width=.5):
 c.setStrokeColor(HexColor(col));c.setLineWidth(width);c.line(x1,y1,x2,y2)
def lighten(h,f=.55):
 c=HexColor(h);return Color(c.red+(1-c.red)*f,c.green+(1-c.green)*f,c.blue+(1-c.blue)*f)
def marker(c,x,y,size,color,shape=0,filled=True):
 c.setStrokeColor(color);c.setFillColor(color);c.setLineWidth(.5)
 if shape==0:c.circle(x,y,size,stroke=1,fill=int(filled))
 elif shape==1:c.rect(x-size,y-size,2*size,2*size,stroke=1,fill=int(filled))
 elif shape==2:
  p=c.beginPath();p.moveTo(x,y+size*1.2);p.lineTo(x-size,y-size);p.lineTo(x+size,y-size);p.close();c.drawPath(p,stroke=1,fill=int(filled))
 elif shape==3:
  p=c.beginPath();p.moveTo(x,y+size*1.3);p.lineTo(x-size,y);p.lineTo(x,y-size*1.3);p.lineTo(x+size,y);p.close();c.drawPath(p,stroke=1,fill=int(filled))
 else:
  c.line(x-size,y,x+size,y);c.line(x,y-size,x,y+size)
def axes(c,box,xlim,ylim,xticks,yticks,xlabel='',ylabel='',ticksize=7.5):
 x,y,w,h=box
 X=lambda v:x+w*(v-xlim[0])/(xlim[1]-xlim[0]);Y=lambda v:y+h*(v-ylim[0])/(ylim[1]-ylim[0])
 for v in yticks:
  line(c,x,Y(v),x+w,Y(v),'#E1E1E1',.4);text(c,x-5,Y(v)-2.5,f'{v:g}',ticksize,align='right')
 for v in xticks:
  line(c,X(v),y,X(v),y-3,'#333333',.5);text(c,X(v),y-12,f'{v:g}',ticksize,align='center')
 line(c,x,y,x+w,y,'#222222',.6);line(c,x,y,x,y+h,'#222222',.6)
 if xlabel:text(c,x+w/2,y-27,xlabel,8.5,align='center')
 if ylabel:
  c.saveState();c.translate(x-32,y+h/2);c.rotate(90);text(c,0,0,ylabel,8.5,align='center');c.restoreState()
 return X,Y
def create(name,height):
 p=OUT/(name+'.pdf');c=canvas.Canvas(str(p),pagesize=(CW,height*72),pageCompression=1)
 c.setTitle(name.replace('_',' '));c.setAuthor('Salih Colakoglu and Mustafa Onat');return c,p
def writecsv(name,rows):
 assert rows
 p=OUT/(name+'.csv')
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return p
def finish(c,p):c.showPage();c.save();assert len(PdfReader(p).pages)==1

CAPTIONS={}
name='A29_Figure_1_grouped_evaluation_workflow';c,p=create(name,4.7)
text(c,CW/2,322,'Grouped evaluation of published average records',12,'Times-Bold','center')
def box(x,y,w,h,ls):
 c.setFillColor(white);c.setStrokeColor(HexColor('#A0A0A0'));c.setLineWidth(.65);c.roundRect(x,y,w,h,4,stroke=1,fill=1)
 for i,s in enumerate(ls):text(c,x+w/2,y+h-15-12*i,s,9.5,'Times-Roman','center')
def arrow(x,y,to):
 line(c,x,y,x,to,'#555555',.7);line(c,x,to,x-3,to+5,'#555555',.7);line(c,x,to,x+3,to+5,'#555555',.7)
box(40,255,CW-80,48,['104 published averages in 26 complete condition groups','Four source contexts; tensile modulus and Shore D hardness'])
arrow(CW/2,255,239)
box(40,194,CW-80,44,['Five outer folds; groups remain intact (6 / 5 / 5 / 5 / 5 held-out groups)','Normalization uses first-stage TRAIN records only'])
line(c,CW/2,194,CW/2,182,'#555555',.7);line(c,125,182,CW-125,182,'#555555',.7)
arrow(125,182,170);arrow(CW-125,182,170)
box(25,84,213,86,['Task-incremental stream','PLA+ / ABS  ->  PLA+ / PETG','->  PLA+ / PLA-CF','Candidate pair supplied at each query'])
box(CW-238,84,213,86,['Class-incremental stream','PLA+ and ABS  ->  add PETG','->  add PLA-CF','All introduced classes; no query task ID'])
arrow(125,84,71);arrow(CW-125,84,71)
box(25,23,CW-50,48,['After every stage: fixed-population decisions and acquisition / retention','All five seeds; 13 primary arms plus two EWC sensitivity arms','Final OOF: 156 task views or 104 class views from the same 104 averages'])
finish(c,p)
writecsv(name+'_source',[{'node':i+1,'description':s} for i,s in enumerate(['104publishedaverages26groups4contexts2features','Wholegroupfivefoldassignment6/5/5/5/5;stage1TRAINnormalizer','TaskILpairs12->13->14;querypairknown','ClassIL12->3->4;querytaskunknown','15arms5seeds3stages;fixedOOFcounts'])])
CAPTIONS[name]='Grouped evaluation workflow. The data unit is a previously inspected published average, not an individually identified physical specimen. The four averages within each of 26 processing-condition groups stay together in a deterministic outer fold. Each model uses a fold-specific normalizer fitted on first-stage TRAIN records. Task-incremental queries receive the eligible pair; class-incremental queries receive no task identifier. The final 156 task views reuse the PLA+ anchor, whereas the final class view contains 104 unique averages. No new blind test or production chronology is claimed.'

name='A29_Figure_2_mechanical_average_records';c,p=create(name,4.8)
text(c,CW/2,328,'Two observed mechanical features',12,'Times-Bold','center')
Xv=[float(r['E_MPa']) for r in D];Yv=[float(r['Hardness_ShoreD']) for r in D]
xstep=500;xlo=math.floor(min(Xv)/xstep)*xstep;xhi=math.ceil(max(Xv)/xstep)*xstep
ylo=math.floor(min(Yv)/5)*5;yhi=math.ceil(max(Yv)/5)*5
X,Y=axes(c,(55,55,CW-80,245),(xlo,xhi),(ylo,yhi),list(range(int(xlo),int(xhi)+1,500)),list(range(int(ylo),int(yhi)+1,5)),'Tensile modulus (MPa)','Shore D hardness')
for r in D:marker(c,X(float(r['E_MPa'])),Y(float(r['Hardness_ShoreD'])),2.35,HexColor(COLORS[int(r['label'])-1]),int(r['label'])-1,False)
for i,l in enumerate(['PLA+','ABS','PETG','PLA-CF']):
 x=80+i*105;marker(c,x,20,2.5,HexColor(COLORS[i]),i,False);text(c,x+7,17,l,9)
finish(c,p)
writecsv(name+'_source',[{k:r[k] for k in ['row_id','group_order','label','E_MPa','Hardness_ShoreD']} for r in D])
CAPTIONS[name]='The 104 published source-average records in the two observed input features. Color and symbol identify the four source contexts. All 26 condition groups are shown; no markers indicate the earlier fixed allocation. Points are group averages retained by the source, and their plot does not establish specimen-level distributions or independent physical validation.'

name='A29_Figure_3_all_final_OOF_comparisons';c,p=create(name,7)
labels={'PROTO':'Prototype','NN1_FULL':'1-NN full history','LIN_SEQ':'Linear sequential','LIN_JOINT_GLOBAL':'Linear joint','MLP_NAIVE':'MLP naive','MLP_ER':'MLP replay','MLP_EWC_L10':'EWC lambda 10','MLP_LWF':'LwF','MLP_JOINT_GLOBAL':'MLP joint'}
for enc in ['OH8','WH8']:
 for neg in ['STATIC','NEG']:
  for rep in ['NOREPLAY','REPLAY']:
   arm=f'{enc}_{neg}_{rep}';labels[arm]=enc+(' static' if neg=='STATIC' else ' negotiated')+(' + replay' if rep=='REPLAY' else '')
labels.update(MLP_EWC_L1='EWC lambda 1 (sensitivity)',MLP_EWC_L100='EWC lambda 100 (sensitivity)')
arms=['PROTO','NN1_FULL','LIN_SEQ','LIN_JOINT_GLOBAL']+[x['arm'] for x in R['final_primary'] if x['stream']=='ClassIL']+['MLP_EWC_L1','MLP_EWC_L100']
assert len(arms)==len(set(arms))==19
lookup={(s['arm'],s['stream']):s for s in R['final_primary']+R['final_EWC_sensitivity']}
for b in R['baseline_reused']['final_summary']:
 lookup[b['method'],{'TIL':'TaskIL','CIL':'ClassIL'}[b['stream']]]=dict(arm=b['method'],seeds=b['seeds'],correct_vector=b['correct_vector'],N_per_seed=b['N_per_seed'],macro_accuracy_vector=[v/100 for v in b['accuracy_percent_vector']])
text(c,CW/2,487,'Final out-of-fold accuracy with all seeds retained',12,'Times-Bold','center')
text(c,232,463,'Class-incremental',10,'Times-Bold','center');text(c,402,463,'Task-incremental',10,'Times-Bold','center')
left=150;pw=145;gap=25;top=444;step=19.2;bottom=top-18*step
for j,stream in enumerate(['ClassIL','TaskIL']):
 x=left+j*(pw+gap)
 for tick in [0,25,50,75,100]:
  xx=x+pw*tick/100;line(c,xx,bottom-8,xx,top+8,'#E1E1E1',.4);text(c,xx,bottom-22,tick,8,align='center')
 text(c,x+pw/2,bottom-37,'Accuracy (%)',9,align='center')
rows=[]
for i,arm in enumerate(arms):
 y=top-i*step
 if i in [4,9,17]:line(c,12,y+step/2,CW-8,y+step/2,'#9A9A9A',.7)
 text(c,12,y-3,labels[arm],8.4,'Times-Italic' if i>=17 else 'Times-Roman')
 color=HexColor('#333333' if i<4 else ('#2374AB' if i<9 or i>=17 else '#278263'))
 for j,stream in enumerate(['ClassIL','TaskIL']):
  s=lookup[arm,stream]
  for k,(seed,corr,val) in enumerate(zip(s['seeds'],s['correct_vector'],s['macro_accuracy_vector'])):
   yy=y+(0 if seed is None else [-3,-1.5,0,1.5,3][k]);marker(c,left+j*(pw+gap)+pw*val,yy,2.0,color,1 if seed is None else 0,seed is None)
   rows.append(dict(arm=arm,stream=stream,seed='' if seed is None else seed,correct=corr,N=s['N_per_seed'],accuracy=val,deterministic=seed is None,sensitivity=arm in ['MLP_EWC_L1','MLP_EWC_L100']))
text(c,CW/2,31,'Circles: seeds 101, 211, 307, 401, 503 (vertical offset only for visibility).',8.7,align='center')
text(c,CW/2,17,'Filled squares: deterministic reference once. Final two rows are sensitivity settings.',8.7,align='center')
finish(c,p);writecsv(name+'_source',rows);assert len(rows)==174
CAPTIONS[name]='Final out-of-fold forced accuracy for all 15 new arms and the four previously accepted simple references. Every seeded method retains all five starts (101, 211, 307, 401, 503); vertical point offsets distinguish coincident values and have no numerical meaning. A deterministic prototype or nearest-record reference is shown once, rather than repeated as five independent results. Class-incremental accuracy is balanced over four classes with 26 averages each; task-incremental accuracy is the macro average over three pairs with 52 views each. EWC lambda 10 is the prespecified primary setting; lambda 1 and 100 remain sensitivity rows. CE and coded networks contain 32 and 52 parameters, respectively, so cross-block comparisons concern complete procedures. Points and seed ranges are descriptive, not confidence intervals.'

selected=['MLP_NAIVE','MLP_ER','MLP_EWC_L10','MLP_LWF','WH8_STATIC_REPLAY','WH8_NEG_REPLAY']
def trajectories(stream,number):
 name=f'A29_Figure_{number}_'+stream+'_acquisition_retention';c,p=create(name,5.5)
 text(c,CW/2,379,stream+': current and earlier-unit accuracy',12,'Times-Bold','center')
 rows=[]
 for i,arm in enumerate(selected):
  col=i%3;row=i//3;px=38+col*160;py=231-row*161;bw=126;bh=112
  text(c,px+bw/2,py+bh+17,labels[arm],9.3,'Times-Bold','center')
  X,Y=axes(c,(px,py,bw,bh),(.8,3.2),(0,100),[1,2,3],[0,50,100],ticksize=7.3)
  if row==1:text(c,px+bw/2,py-25,'Learning stage',8.3,align='center')
  for si,seed in enumerate(SEEDS):
   for kind,color in [('current','#2374AB'),('old','#C56616')]:
    vals=[]
    for stage in [1,2,3]:
     m=next(m for m in R['pooled_stage'] if (m['arm'],m['seed'],m['stream'],m['stage'])==(arm,seed,stream,stage))
     v=m[kind+'_accuracy'];rows.append(dict(arm=arm,stream=stream,seed=seed,stage=stage,unit_set=kind,correct=m[kind+'_correct'],N=m[kind+'_N'],accuracy='' if v is None else v))
     if v is not None:vals.append((stage,100*v))
    for (x1,y1),(x2,y2) in zip(vals,vals[1:]):
     c.setStrokeColor(lighten(color,.30));c.setLineWidth(.55);c.line(X(x1),Y(y1),X(x2),Y(y2))
    for st,v in vals:marker(c,X(st),Y(v),1.8,HexColor(color),si,False)
 text(c,CW/2,28,'Blue: current units. Orange: earlier units. Earlier-unit accuracy is undefined at stage 1.',8.4,align='center')
 for i,seed in enumerate(SEEDS):
  x=90+i*70;marker(c,x,12,2,black,i,False);text(c,x+7,9,seed,8)
 finish(c,p);writecsv(name+'_source',rows);assert len(rows)==180
 CAPTIONS[name]=('Stage-wise current-unit and earlier-unit accuracy for six prespecified comparisons in the '+stream+' stream. Each symbol follows one of the five retained seeds; colors distinguish current from earlier units. Counts are pooled over whole held-out groups before division, rather than averaged equally over unequal fold sizes. Earlier-unit accuracy has no denominator at the first stage and is not plotted as zero. '+('Current units are the current candidate pair, including the recurring PLA+ anchor; earlier units are previously acquired pairs. ' if stream=='TaskIL' else 'The first current set contains PLA+ and ABS; later current sets contain the newly introduced class, while earlier sets retain all previously introduced classes. ')+'The accompanying point table includes the exact numerator and denominator for every value. Lines identify stage trajectories and do not imply observations between learning stages.')
trajectories('ClassIL',4);trajectories('TaskIL',5)

name='A29_Figure_6_native_final_coverage_points';c,p=create(name,5.5)
text(c,CW/2,379,'Native final-stage answer coverage and conditional error',12,'Times-Bold','center')
methods=['MLP_ER','MLP_EWC_L10','WH8_NEG_REPLAY'];rows=[];curves=0
for ri,stream in enumerate(['ClassIL','TaskIL']):
 for ci,arm in enumerate(methods):
  px=38+ci*160;py=231-ri*161;bw=122;bh=112
  text(c,px+bw/2,py+bh+20,labels[arm],9.2,'Times-Bold','center');text(c,px+bw/2,py+bh+8,stream,8.2,align='center')
  X,Y=axes(c,(px,py,bw,bh),(0,100),(0,100),[0,50,100],[0,50,100],ticksize=7.1)
  if ri==1:text(c,px+bw/2,py-25,'Answered (%)',8.2,align='center')
  if ci==0:
   c.saveState();c.translate(px-29,py+bh/2);c.rotate(90);text(c,0,0,'Error among answers (%)',8.1,align='center');c.restoreState()
  for ref in R['coverage']['pins']:
   f=Path(ref['path']);f=f if f.is_absolute() else A/f
   if f.name!=f'{arm}_101_{stream}_stage3.json' and not any(f.name==f'{arm}_{s}_{stream}_stage3.json' for s in SEEDS[1:]):continue
   assert sha(f)==ref['sha256'];q=read(f);curves+=1
   for stepid,s in enumerate(q['steps']):
    rows.append(dict(arm=arm,stream=stream,fold=q['fold'],seed=q['seed'],step=stepid,threshold='' if s['threshold'] is None else s['threshold'],threshold_infinity=s['threshold_is_positive_infinity'],C=s['C'],W=s['W'],A=s['A'],F=s['F'],N=s['N'],answered=s['answered'],coverage=s['coverage'],conditional_error='' if s['conditional_error'] is None else s['conditional_error']))
    if s['conditional_error'] is not None:
     marker(c,X(100*s['coverage']),Y(100*s['conditional_error']),.80,lighten(COLORS[q['fold']-1],.18),SEEDS.index(q['seed']),False)
for i in range(5):
 x=48+i*88;marker(c,x,28,2,HexColor(COLORS[i]),0);text(c,x+6,25,'Fold '+str(i+1),7.8)
text(c,CW/2,11,'All 25 fold-seed curves per panel; only attainable complete gap-block cutpoints are plotted.',8.2,align='center')
finish(c,p);writecsv(name+'_source',rows);assert curves==150
CAPTIONS[name]='Native final-stage coverage diagnostics for replay, primary EWC and Walsh-negotiated replay, shown separately for the two evaluation streams. Each panel retains all 25 fold-seed curves. Color identifies the held-out fold; point shapes follow the same seed order as Figures 4 and 5. Only actual complete gap-block cutpoints are plotted, with no interpolated operating points or selected best threshold. At each point, answer coverage is (C+W)/N and conditional error is W/(C+W), with correct answers C, wrong answers W, unselected valid decisions A, and failures F satisfying C+W+A+F=N. The empty answered set has undefined conditional error and is retained in the source table without an error-coordinate point. Scores from different folds or decoder families are never pooled into a common ranking. These are descriptive diagnostics rather than calibrated risk guarantees.'

(OUT/'A29_figure_captions_v1.json').write_text(json.dumps({'schema':'A29_MANUSCRIPT_FIGURE_CAPTIONS_V1','captions':CAPTIONS,'source_report_sha256':sha(REPORT),'source_csv_sha256':sha(DATA),'new_training':0},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if POPPLER:
 for p in sorted(OUT.glob('*.pdf')):
  z=subprocess.run([str(POPPLER),'-r','150','-singlefile','-png',str(p),str(p.with_suffix(''))],capture_output=True,timeout=30)
  assert z.returncode==0 and p.with_suffix('.png').is_file(),(p,z.stderr)
files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.iterdir()) if p.is_file()]
assert sum(p['bytes'] for p in files)<32*1024*1024
receipt={'schema':'A29_SIX_SCIENTIFIC_FIGURE_BUILD_V1','status':'BUILT_FROM_S1_SAVED_NUMBERS_USER_VISUAL_REVIEW_PENDING','figures':6,'PDFs':6,'PNGs':len(list(OUT.glob('*.png'))),'point_tables':6,'coverage_curves':150,'coverage_point_rows':len(rows),'F3points':174,'source_report_sha256':sha(REPORT),'source_data_sha256':sha(DATA),'generator':Path(__file__).name,'generator_sha256':sha(__file__),'output_total_bytes':sum(x['bytes'] for x in files),'files':files,'new_training':0,'MATLAB':0,'marker_repeated':False}
(OUT/'A29_figure_build_receipt_v1.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=True,indent=2))
