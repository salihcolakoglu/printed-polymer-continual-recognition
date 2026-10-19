function result=a29cl_controls(outDir)
% New artificial controls only; never opens the published source CSV/results.
assert(isfolder(outDir),'A29CL:Output','Launcher creates a new output directory.');
assert(~isfile(fullfile(outDir,'controls.json')),'A29CL:Consumed','Fresh control output required.');
result=struct('schema','A29CL_PRETRAINING_CONTROLS_V1','status','STARTED', ...
 'source_training',0,'source_predictions',0,'source_csv_read',false, ...
 'matlab_version',version,'checks',{{}},'cases',struct());
timer=tic;
try
 X=[-1 .2;.3 -.7;1.1 .9;.5 -1.2]; y=[1;3;1;3]; empty=struct('theta',{},'F',{});
 s=a29cl_math('init',4,101); t=s.theta+.03*sin((1:32)');
 s8=a29cl_math('init',8,101); t8=s8.theta+.02*cos((1:52)');
 % Save every operand needed by a different numerical implementation.
 result.cases.fixture=struct('X',X,'y',y,'theta',t,'theta8',t8);
 for seed=[101 211 307 401 503]
  a=a29cl_math('init',4,seed);b=a29cl_math('init',8,seed);
  must(isequal(a.theta(1:12),b.theta(1:12)),'shared trunk initialization');
  must(all(a.m==0)&&all(a.v==0)&&a.t==0,'initial optimizer');
  [W1,b1,W2,b2]=a29cl_math('unpack',a.theta,4);
  must(isequal(a.theta,a29cl_math('pack',W1,b1,W2,b2)),'column-major roundtrip');
 end
 cases={};
 for mask={[1 3],[1 2 3 4]}
  m=mask{1};[L,g]=a29cl_math('ce',t,X,y,m,4);
  fd=central(@(v)a29cl_math('ce',v,X,y,m,4),t);
  close(g,fd,2e-7,2e-5,'CE all32parameter FD');
  cases{end+1}=struct('theta',t,'Z',X,'y',y,'mask',m,'loss',L,'gradient',g,'FD',fd); %#ok<AGROW>
 end
 result.cases.CE=cases;
 % Global all-seen CE must remain meaningful on a one-new-class batch.
 [L,g]=a29cl_math('ce',t,X,4*ones(4,1),[1 2 3 4],4);
 must(L>0&&norm(g)>1e-6,'singleton-label batch with multiclass loss');
 [~,pairg]=a29cl_math('ce',t,X,y,[1 3],4);
 must(all(pairg(17:20)==0)&&all(pairg(25:28)==0)&&pairg(30)==0&&pairg(32)==0,'masked output data gradients zero');
 OH=[eye(4) zeros(4)];H=1;while size(H,1)<8,H=[H H;H -H];end;WH=(1-H(2:5,:))/2;
 result.cases.codebooks=struct('OH8',OH,'WH8',WH);
 bcases={};
 for C={OH,WH}
  hard=C{1}(y,:);pre=a29cl_math('forward',s8.theta,X,8);q=.5*hard+.5*sigmoid(pre);
  for Q={hard,q}
   target=Q{1};[L,g]=a29cl_math('bce',t8,X,target,8);
   fd=central(@(v)a29cl_math('bce',v,X,target,8),t8);
   close(g,fd,2e-7,2e-5,'BCE all52parameter FD');
   bcases{end+1}=struct('theta',t8,'Z',X,'q',target,'loss',L,'gradient',g,'FD',fd); %#ok<AGROW>
  end
 end
 result.cases.BCE=bcases;
 % Detached own teacher, T2, full old-pair sum, and old-union ClassIL.
 kd={};terms=struct('kind',{},'Z',{},'target',{},'mask',{},'T',{},'weight',{});
 for m={[1 2],[1 3],[1 2 3]}
  mask=m{1};pred=a29cl_math('forward',s.theta,X,4);q=softmax(pred(:,mask)/2);
  [L,g]=a29cl_math('softce',t,X,q,mask,2,4);
  fd=central(@(v)a29cl_math('softce',v,X,q,mask,2,4),t);
  close(g,fd,2e-7,2e-5,'KD all32parameter FD');
  [~,g0]=a29cl_math('softce',s.theta,X,q,mask,2,4);close(g0,zeros(32,1),1e-13,1e-12,'teacher self-KD gradient0');
  kd{end+1}=struct('theta',t,'teacher',s.theta,'Z',X,'q',q,'mask',mask,'T',2,'loss',L,'gradient',g,'FD',fd); %#ok<AGROW>
  if numel(mask)==2,terms(end+1)=term('softce',X,q,mask,2,1);end %#ok<AGROW>
 end
 result.cases.KD=kd;
 [Lboth,gboth]=a29cl_math('objective',t,terms,4,empty,0);
 [Lreg,greg]=a29cl_math('objective',t,terms([]),4,empty,0);
 close(Lboth-Lreg,kd{1}.loss+kd{2}.loss,1e-13,1e-12,'full old-pair KD sum');
 close(gboth-greg,kd{1}.gradient+kd{2}.gradient,1e-13,1e-12,'KD sum gradient noT2');
 must(norm(gboth-greg)>1e-5,'old-pair preservation nonzero');
 % Independent finite differences of log probabilities (not core CE grads).
 fcases={};
 for m={[1 3],[1 2 3]}
  mask=m{1};[F,work]=a29cl_math('fisher',t,X,mask,4);Ffd=zeros(32,1);
  logits=a29cl_math('forward',t,X,4);p=softmax(logits(:,mask));
  for i=1:4
   for k=1:numel(mask)
    glog=central(@(v)logprob(v,X(i,:),mask,k),t);
    Ffd=Ffd+p(i,k)*(glog.^2)/4;
   end
  end
  close(F,Ffd,3e-7,4e-5,'model expected Fisher independent logp FD');
  empirical=zeros(32,1);meanG=zeros(32,1);
  for i=1:4
   [~,g]=a29cl_math('ce',t,X(i,:),mask(1),mask,4);
   empirical=empirical+g.^2/4;meanG=meanG+g/4;
  end
  must(norm(F-empirical)>1e-6&&norm(F-meanG.^2)>1e-6,'Fisher distinct empirical/batch-square');
  must(work.implementation_forward_rows==4&&work.class_gradient_terms==4*numel(mask),'cached exact Fisher work');
  fcases{end+1}=struct('theta',t,'Z',X,'mask',mask,'F',F,'FD_reference',Ffd,'empirical',empirical,'batch_square',meanG.^2,'work',work); %#ok<AGROW>
 end
 result.cases.Fisher=fcases;
 anchors(1)=struct('theta',t-.2*cos((1:32)'),'F',.01+(1:32)'/320);
 anchors(2)=struct('theta',t+.1*sin((1:32)'),'F',.02+(32:-1:1)'/160);
 cur=term('ce',X,y,[1 3],1,1);ewc={};
 for lam=[1 10 100]
  [L,g,c]=a29cl_math('objective',t,cur,4,anchors,lam);
  fd=central(@(v)a29cl_math('objective',v,cur,4,anchors,lam),t);
  close(g,fd,2e-7,2e-5,'two-anchor EWC+L2+CE FD');
  E=0;eg=zeros(32,1);
  for a=1:2,E=E+lam/2*sum(anchors(a).F.*(t-anchors(a).theta).^2);eg=eg+lam*anchors(a).F.*(t-anchors(a).theta);end
  close(c.EWC,E,1e-13,1e-12,'separate quadratic sum');
  [L0,g0]=a29cl_math('objective',t,cur,4,empty,0);
  close(L-L0,E,1e-12,1e-11,'penalty added once');close(g-g0,eg,1e-12,1e-11,'anchor gradient');
  bad=lam*((anchors(1).F+anchors(2).F).*(t-anchors(2).theta));
  must(norm(eg-bad)>1e-3,'detect summedF latestanchor error');
  ewc{end+1}=struct('theta',t,'Z',X,'y',y,'mask',[1 3],'anchors',anchors,'lambda',lam,'loss',L,'gradient',g,'FD',fd,'components',c); %#ok<AGROW>
 end
 result.cases.EWC=ewc;
 [W1,~,W2,~]=a29cl_math('unpack',t,4);rg=[1e-4*W1(:);zeros(4,1);1e-4*W2(:);zeros(4,1)];
 close(greg,rg,1e-14,1e-12,'weight-only global L2 exact');close(Lreg,1e-4/2*(sum(W1(:).^2)+sum(W2(:).^2)),1e-14,1e-12,'L2 half factor');
 % Stable finite extremes; hidden weights fixed while output biases move.
 large=t;large(29:32)=[1000;-1000;900;-900];[L,g]=a29cl_math('ce',large,X,y,[1 2 3 4],4);must(isfinite(L)&&all(isfinite(g)),'stable extremeCE');
 large8=t8;large8(45:52)=1000*(-1).^(1:8)';[L,g]=a29cl_math('bce',large8,X,WH(y,:),8);must(isfinite(L)&&all(isfinite(g)),'stable extremeBCE');
 % Independent scalar-array Adam, then split-stage carry equivalence.
 a=s;oracle=s;steps={};
 for k=1:3
  g=sin((1:32)'*k)/10;oracle.m=.9*oracle.m+.1*g;oracle.v=.999*oracle.v+.001*g.^2;oracle.t=oracle.t+uint64(1);n=double(oracle.t);
  oracle.theta=oracle.theta-.02*(oracle.m/(1-.9^n))./(sqrt(oracle.v/(1-.999^n))+1e-8);
  before=a;a=a29cl_math('step',a,g);close(a.theta,oracle.theta,1e-13,1e-12,'Adam independent sequence');
  steps{end+1}=struct('before',before,'g',g,'after',a); %#ok<AGROW>
 end
 result.cases.Adam=steps;
 spec=struct('terms',cur,'D',4,'anchors',empty,'ewc_lambda',0,'updates',3,'control_only',true);
 [full,receipt]=a29cl_stage(s,spec);spec.updates=2;[part,r2]=a29cl_stage(s,spec);spec.updates=1;[part,r1]=a29cl_stage(part,spec);
 must(isequal(full,part)&&full.t==3,'stagecarry including Adam');must(isequal(cur,receipt.terms),'frozen input/cache unchanged');
 result.cases.stage=struct('full',receipt,'split_first',r2,'split_second',r1);
 % Same seeded student branch never modifies teacher/cache copies.
 q=kd{1}.q;teacher=s;beforeq=q;beforeteacher=teacher;
 spec.terms=term('softce',X,q,[1 2],2,1);spec.updates=3;[~,tr]=a29cl_stage(s,spec);
 must(isequal(q,beforeq)&&isequal(teacher,beforeteacher)&&isequal(tr.terms.target,q),'teacher/cache stop-gradient and freezing');
 % Exact-tie/native-gap/failure rows; no tolerance tie policy.
 zero4=zeros(32,1);zero8=zeros(52,1);
 d=a29cl_math('decode',zero4,X,4,[1 3],[]);must(all(d.label==1)&all(d.tie)&all(d.gap==0),'CE exactties lowestglobalID');
 for C={OH,WH}
  d=a29cl_math('decode',zero8,X,8,[1 3],C{1});must(all(d.label==1)&all(d.tie)&all(d.gap==0),'code exactties MSE');
 end
 near=zero4;near(31)=eps(1);d=a29cl_math('decode',near,X,4,[1 3],[]);must(all(~d.tie)&&all(d.label==3),'near tie not arbitrarytolerance');
 badX=X;badX(2,1)=nan;d=a29cl_math('decode',t,badX,4,[1 3],[]);must(numel(d.label)==4&&d.failure(2)&&isnan(d.label(2)),'failed query retained');
 result.cases.decode=struct('theta',t,'Z',badX,'eligible',[1 3],'result',d);
 must(caught(@()a29cl_math('ce',t,X,y,1,4)),'reject singleton softmax');
 must(caught(@()a29cl_math('ce',t,X,y,[3 1],4)),'reject unorderedmask');
 must(caught(@()a29cl_math('ce',t,X,y,[1 1],4)),'rejectduplicatemask');
 badanchors=anchors;badanchors(1).F(1)=-1;must(caught(@()a29cl_math('objective',t,cur,4,badanchors,1)),'negativeFisher rejected');
 % Whole-gap blocks and a known acquisition/retention trajectory.
 cp={struct('truth',1,'label',1,'gap',0,'valid',true,'failure',false), ...
  struct('truth',2,'label',1,'gap',.1,'valid',true,'failure',false), ...
  struct('truth',3,'label',3,'gap',.1,'valid',true,'failure',false), ...
  struct('truth',4,'label',nan,'gap',.2,'valid',false,'failure',true)};
 cov=a29cl_metrics('coverage',cp);cs=cov.steps;
 must(numel(cs)==3&&isequal([cs.C],[0 1 2])&&isequal([cs.W],[0 1 1])&&isequal([cs.A],[3 1 0])&&all([cs.F]==1),'coverage whole tie blocks C/W/A/F');
 must(isnan(cs(1).conditional_error)&&cs(2).conditional_error==.5&&abs(cs(3).conditional_error-1/3)<1e-14,'coverage zeroanswer null and error denominators');
 mp={};correctByStage={[2 4],[3 2 2],[4 1 3 4]};
 for st=1:3
  corrects=correctByStage{st};
  for yy=1:st+1
   for rr=1:4
    label=yy;if rr>corrects(yy),label=1+mod(yy,st+1);end
    mp{end+1}=struct('fold',1,'seed',101,'arm','KNOWN_METRIC_FIXTURE','stream','ClassIL','stage',st,'task',0,'parentrow',sprintf('METRIC_CLASS%d_ROW%d',yy,rr),'truth',yy,'label',label,'gap',.1,'valid',true,'failure',false,'eligible',1:st+1); %#ok<AGROW>
   end
  end
 end
 ms=a29cl_metrics('summarize',mp);rtx=ms.retention{1};
 close(rtx.BWT,0,1e-14,0,'known BWT');close(rtx.signed_forgetting,1/12,1e-14,0,'known signedforgetting allows negativeunits');
 must(~ms.complete_source_OOF,'artificial partial trajectory notcompleteOOF');
 result.cases.metrics=struct('coverage_operands',{cp},'coverage',cov,'trajectory_operands',{mp},'trajectory',ms);
 % Serializer roundtrip and structural bound exercise extreme finite values.
 roundtrip=struct('v',[-realmax,realmin,eps,pi,0], 'id',uint64(104), 'undefined',nan);
 text=jsonencode(roundtrip);decoded=jsondecode(text);
 close(decoded.v(:),roundtrip.v(:),0,0,'binary64 JSON numerical roundtrip');
 must(decoded.id==104&&(isempty(decoded.undefined)||(isscalar(decoded.undefined)&&isnan(decoded.undefined))),'JSON integer/null mapping');
 must(numel(unicode2native(text,'UTF-8'))<=a29cl_output('bound',roundtrip),'independent structural JSON bytebound');
 result.cases.serialization=struct('value',roundtrip,'text',text,'bound',a29cl_output('bound',roundtrip));
 dry=a29cl_run();must(dry.source_training==0,'default runner metadataonly nofit');
 must(caught(@()a29cl_run(struct('mode','source'))),'sourcefit missingfreeze failsclosed beforeloader');
 % Resource microbenchmark: artificial worst-shape terms, no source rows.
 Z=[sin((1:84)') cos((1:84)')];Y=1+mod((0:83)',4);teach=softmax(a29cl_math('forward',s.theta,Z,4)/2);
 worst=[term('ce',Z,Y,1:4,1,.5),term('ce',Z(1:8,:),Y(1:8),1:4,1,.5),term('softce',Z,teach,1:4,2,1),term('softce',Z,teach,1:4,2,1)];
 spec=struct('terms',worst,'D',4,'anchors',anchors,'ewc_lambda',10,'updates',20,'control_only',true);[~,rt]=a29cl_stage(s,spec);
 start=tic;[~,fw]=a29cl_math('fisher',s.theta,Z(1:42,:),1:4,4);fsec=toc(start);
 Ctarget=WH(Y(1:42),:);spec=struct('terms',term('bce',Z(1:42,:),Ctarget,[],1,1),'D',8,'anchors',empty,'ewc_lambda',0,'updates',20,'control_only',true);[~,rb]=a29cl_stage(s8,spec);
 result.cases.resources=struct('CE_worst20_seconds',rt.elapsed_seconds,'CODE20_seconds',rb.elapsed_seconds,'Fisher42x4_seconds',fsec,'Fisher_work',fw,'source_training',0,'scope','Artificial microbenchmark; projection is estimate, hard watchdog still required.');
 % End-to-end artificial loader/orchestrator controls, no source CSV path.
 result.cases.data=a29cl_data('controls');
 pipe=a29cl_run(struct('mode','control_fixture','updates',3,'folds',1,'seeds',101));
 result.pipeline=pipe;
 result.cases.output_proof=a29cl_output('proof');
 result.cases.output_check=a29cl_output('check',pipe);
 result.summary=a29cl_metrics('summarize',pipe);
 result.status='CONTROL_PRODUCER_COMPLETE_PENDING_DIFFERENT_SAVED_AUDIT';
 result.elapsed_seconds=toc(timer);
catch err
 result.status='CONTROL_FAILED';result.error=struct('identifier',err.identifier,'message',err.message);result.elapsed_seconds=toc(timer);
 save_result();rethrow(err)
end
save_result();
 function must(value,name)
  if ~isscalar(value)||~value,error('A29CL:ControlFailure','%s',name);end
  result.checks{end+1}=struct('name',name,'passed',true);
 end
 function close(a,b,ab,rel,name)
  must(isequal(size(a),size(b))&&all(isfinite(a(:)))&&all(isfinite(b(:)))&&all(abs(a(:)-b(:))<=ab+rel*max(abs(a(:)),abs(b(:)))),name);
 end
 function save_result()
  % Outputcheck before writing; NaN/null only for explicit failed queries.
  text=jsonencode(result);bytes=unicode2native(text,'UTF-8');
  assert(numel(bytes)<16*1024*1024,'A29CL:OutputCap','Control JSON exceeds fixed16MiB cap.');
  f=fopen(fullfile(outDir,'controls.json'),'wb');assert(f>=0);cleaner=onCleanup(@()fclose(f));fwrite(f,bytes,'uint8');clear cleaner
 end
end
function a=term(kind,Z,target,mask,T,weight)
a=struct('kind',kind,'Z',Z,'target',target,'mask',mask,'T',T,'weight',weight);
end
function g=central(fun,theta)
g=zeros(size(theta));
for j=1:numel(theta),h=1e-5*max(1,abs(theta(j)));a=theta;b=theta;a(j)=a(j)+h;b(j)=b(j)-h;g(j)=(fun(a)-fun(b))/(2*h);end
end
function q=softmax(x)
x=x-max(x,[],2);q=exp(x)./sum(exp(x),2);
end
function q=sigmoid(x)
q=zeros(size(x));p=x>=0;q(p)=1./(1+exp(-x(p)));e=exp(x(~p));q(~p)=e./(1+e);
end
function l=logprob(theta,x,mask,k)
a=a29cl_math('forward',theta,x,4);v=a(mask);v=v-max(v);l=v(k)-log(sum(exp(v)));
end
function b=caught(fun)
try,fun();b=false;catch err,b=startsWith(err.identifier,'A29:')||startsWith(err.identifier,'A29CL:');end
end
