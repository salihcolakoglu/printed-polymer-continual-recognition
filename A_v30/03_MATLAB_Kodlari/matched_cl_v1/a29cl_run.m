function result = a29cl_run(config)
% A29CL_RUN Matched-CL orchestration; source training CLOSED without run freeze.
% mode defaults to metadata_only. control_fixture uses artificial values only.
% Source execution needs a later independently reviewed exact run freeze.
% This implementation does not accept a generic token as source admission.
if nargin==0,config=struct;end
mode=char(getdefault(config,'mode','metadata_only'));
A29=fileparts(fileparts(fileparts(mfilename('fullpath'))));
protocolpath=fullfile(A29,'02_Protokol','A29_Matched_CL_Protocol_v1.json');
linkpath=fullfile(A29,'02_Protokol','A29_Matched_CL_Input_Links_v1.json');
assert(strcmp(file_sha(protocolpath),'a875157c99f98b62ef6a9be2c7fdb194fafb8dcad000de2a86af6947ae89008e'),'A29:ProtocolHash','Effective protocol pin mismatch.');
assert(strcmp(file_sha(linkpath),'e96afb68762fc773e0f64021acf6a59cce9f55d257b6c1a8048525cc9e211a49'),'A29:InputHash','Input link pin mismatch.');
P=jsondecode(fileread(protocolpath));
if strcmp(mode,'metadata_only')
 result=struct('schema','A29_MATCHED_CL_DRY_V1','source_training',0,'catalog',a29cl_data('catalog',linkpath),'plan',P.finite_execution_design);return
end
if ~any(strcmp(mode,{'control_fixture','source'}))
 error('A29:RunMode','Unknown run mode.');
end
updates=double(getdefault(config,'updates',3));folds=double(getdefault(config,'folds',1));seeds=double(getdefault(config,'seeds',101));
if strcmp(mode,'control_fixture')
 assert(isscalar(updates)&&ismember(updates,1:3),'A29:ControlUpdates','Artificial control accepts1..3 updates only.');
 assert(isequal(folds,1)&&isequal(seeds,101),'A29:ControlScope','This finite artificial control is fold1 seed101 only.');
 Dset=a29cl_data('fixture');
else
 verify_admission(config,A29,protocolpath,linkpath,updates,folds,seeds);
 Dset=a29cl_data('prepare',fullfile(A29,'02_Veriler','A_source_average_records_v1.csv'),linkpath);
end
result=engine(P,Dset,updates,folds,seeds);
if Dset.isfixture
 result.kind='SOURCE_FREE_ARTIFICIAL_IMPLEMENTATION_CONTROL_NOT_SCIENTIFIC_RESULT';result.source_training=0;result.source_predictions=0;
else
 result.kind='EXPOSED_SOURCE_AVERAGE_MATCHED_CL_COMPARISON';result.source_training=result.counts.unique_stagefits;result.source_predictions=result.counts.predictionrows;
end
result.physical_validation=0;
if isfield(config,'outDir')&&~isempty(config.outDir)
 outdir=char(config.outDir);assert(isfolder(outdir),'A29:OutputDir','Create explicit empty output directory before control.');
 if Dset.isfixture,name='a29cl_fixture_result_v1.json';cap=16*1024*1024;else,name=sprintf('a29cl_source_fold%d_v1.json',folds);cap=64*1024*1024;end
 path=fullfile(outdir,name);assert(~isfile(path),'A29:NoOverwrite','Output exists.');
 a29cl_output('write',result,path,cap);
end
end
function R=engine(P,data,updates,folds,seeds)
R=struct('schema','A29_MATCHED_CL_RESULT_V1','updates_per_stage',updates,'folds',folds,'seeds',seeds,'data_admission',data.admission,'normalizers',data.normalizers,'initial_states',{{}},'unique_fits',{{}},'logical_stages',{{}},'predictions',{{}},'Fisher_jobs',{{}},'codebooks',P.code_negotiation);
fitcache=containers.Map('KeyType','char','ValueType','double');fcache=containers.Map('KeyType','char','ValueType','double');
arms=P.arms;streams={'TaskIL','ClassIL'};
for fold=reshape(folds,1,[])
 for seed=reshape(seeds,1,[])
  init4=a29cl_math('init',4,seed);init8=a29cl_math('init',8,seed);
  R.initial_states{end+1}=struct('fold',fold,'seed',seed,'D',4,'state',init4);R.initial_states{end+1}=struct('fold',fold,'seed',seed,'D',8,'state',init8);
  for ai=1:numel(arms)
   arm=getarm(arms,ai);D=4;if strcmp(arm.block,'CODE8'),D=8;end
   C=[];if D==8,C=P.code_negotiation.(arm.codebook);end
   for si=1:2
    stream=streams{si};state=init4;if D==8,state=init8;end
    B=a29cl_data('replay_empty');anchors=struct('theta',{},'F',{});
    for stage=1:3
     cur=data.current(fold,stream,stage);
     % The optimizer receives only cur.Z/y and declared past state below.
     % Evaluation query handles are called only after the stage update.
     teacher=[];qcache={};oldmasks={};replay=a29cl_data('replay_view',B,cur);
     if D==4&&strcmp(arm.mechanism,'joint')
      cur=data.joint(fold,stage);state=init4;
     end
     incoming=state;terms={};useReplay=(D==4&&strcmp(arm.mechanism,'replay'))||(D==8&&arm.replay);
     if D==4
      terms{1}=term('ce',cur.Z,cur.y,cur.eligible,1,1);
      if useReplay&&~isempty(replay.y)
       terms{1}.weight=.5;
       if strcmp(stream,'TaskIL')
        % Each first-admission pair retains its own CE mask. Weight by
        % its fraction of the entire eligible memory, not by task count.
        for task=reshape(unique(double(replay.firsttask)),1,[])
         idx=double(replay.firsttask)==task;
         terms{end+1}=term('ce',replay.Z(idx,:),replay.y(idx),[1,task+1],1,.5*sum(idx)/numel(replay.y)); %#ok<AGROW>
        end
       else
        terms{end+1}=term('ce',replay.Z,replay.y,1:stage+1,1,.5);
       end
      end
      if strcmp(arm.mechanism,'lwf')&&stage>1
       teacher=incoming.theta;oldmasks=old_masks(stream,stage);
       logits=a29cl_math('forward',teacher,cur.Z,D);
       for k=1:numel(oldmasks)
        q=softmax_local(logits(:,oldmasks{k})/2);qcache{end+1}=q;
        terms{end+1}=term('softce',cur.Z,q,oldmasks{k},2,1); %#ok<AGROW>
       end
      end
     else
      q=C(cur.y,:);
      if arm.negotiation
       teacher=incoming.theta;logits=a29cl_math('forward',teacher,cur.Z,D);
       a=P.code_negotiation.alpha(stage);q=(1-a)*q+a*sigmoid_local(logits);qcache={q};
      end
      terms{1}=term('bce',cur.Z,q,[],1,1);
      if useReplay&&~isempty(replay.y)
       terms{1}.weight=.5;terms{end+1}=term('bce',replay.Z,C(replay.y,:),[],1,.5);
      end
     end
     lambda=0;activeAnchors=struct('theta',{},'F',{});
     if D==4&&strcmp(arm.mechanism,'ewc')&&stage>1,lambda=arm.ewc_lambda;activeAnchors=anchors;end
     terms=[terms{:}];
     spec=struct('D',D,'terms',terms,'anchors',activeAnchors,'ewc_lambda',lambda,'updates',updates,'control_only',logical(data.isfixture));
     % Hash the entire numerical job. No scientific-result based alias.
     memoryparents=zeros(0,1,'uint64');if useReplay,memoryparents=replay.parentindex;end
     key=a29cl_data('hash',jsonencode(struct('fold',fold,'seed',seed,'current_parentindex',cur.parentindex,'memory_parentindex',memoryparents,'incoming',incoming,'spec',spec)));
     aliased=isKey(fitcache,key);
     if aliased
      fitid=fitcache(key);saved=R.unique_fits{fitid};state=saved.final_state;receipt=saved.receipt;
      assert(isequaln(saved.incoming_state,incoming)&&isequaln(saved.spec,spec)&&isequal(saved.current_parentindex,cur.parentindex)&&isequal(saved.memory_parentindex,memoryparents),'A29:AliasExact','Alias hash matched but exact job differed.');
     else
      [state,receipt]=a29cl_stage(incoming,spec);fitid=numel(R.unique_fits)+1;fitcache(key)=fitid;
      R.unique_fits{fitid}=struct('job_id',fitid,'fold',fold,'seed',seed,'D',D,'job_sha256',key,'current_parentindex',cur.parentindex,'memory_parentindex',memoryparents,'incoming_state',incoming,'final_state',state,'spec',spec,'receipt',receipt);
     end
     fisherid=0;F=[];
     if D==4&&strcmp(arm.mechanism,'ewc')&&stage<3
      fkey=a29cl_data('hash',jsonencode(struct('fold',fold,'seed',seed,'theta',state.theta,'Z',cur.Z,'mask',cur.eligible,'D',D)));
      if isKey(fcache,fkey)
       fisherid=fcache(fkey);F=R.Fisher_jobs{fisherid}.F;
      else
       [F,work]=a29cl_math('fisher',state.theta,cur.Z,cur.eligible,D);fisherid=numel(R.Fisher_jobs)+1;fcache(fkey)=fisherid;
       R.Fisher_jobs{fisherid}=struct('job_id',fisherid,'job_sha256',fkey,'fold',fold,'seed',seed,'theta',state.theta,'Z',cur.Z,'mask',cur.eligible,'F',F,'work',work);
      end
      anchors(end+1)=struct('theta',state.theta,'F',F);
     end
     Bbefore=B;admission=[];
     if useReplay
      post=a29cl_data('replay_commit',B,cur,fold,stage);B=post.buffer;admission=post.receipt;
     end
     logicalid=numel(R.logical_stages)+1;
     R.logical_stages{logicalid}=struct('id',logicalid,'fold',fold,'seed',seed,'arm',arm.id,'primary',arm.primary,'stream',stream,'stage',stage,'job_id',fitid,'aliased',aliased,'current_parentindex',cur.parentindex,'current_parentrow',{cur.parentrow},'teacher',teacher,'qcache',{qcache},'old_masks',{oldmasks},'replay_before',Bbefore,'effective_replay',replay,'replay_after',B,'replay_admission',admission,'Fisher_job_id',fisherid,'new_Fisher',F,'retained_anchors',anchors,'logical_cost',cost_card(D,cur,replay,useReplay,arm,stage,updates,oldmasks,teacher));
     query=data.query(fold,stream,stage);
     % Heldout labels exist in this evaluator scope, never optimizer terms.
     if strcmp(stream,'TaskIL'),tasklist=1:stage;else,tasklist=0;end
     for task=tasklist
      if task>0,rows=find(query.task==task);eligible=[1,task+1];else,rows=(1:numel(query.y))';eligible=1:stage+1;end
      decoded=a29cl_math('decode',state.theta,query.Z(rows,:),D,eligible,C);
      for j=1:numel(rows)
       k=rows(j);record=struct('fold',fold,'seed',seed,'arm',arm.id,'primary',arm.primary,'stream',stream,'stage',stage,'task',task,'parentindex',query.parentindex(k),'parentrow',query.parentrow{k},'group_order',query.group_order(k),'truth',query.y(k),'eligible',eligible,'job_id',fitid,'logical_stage_id',logicalid,'Z',query.Z(k,:),'label',decoded.label(j),'gap',decoded.gap(j),'tie',decoded.tie(j),'valid',decoded.valid(j),'failure',decoded.failure(j),'failure_reason',decoded.failure_reason{j},'scores',decoded.scores(j,:),'distances',decoded.distances(j,:),'outputs',decoded.outputs(j,:),'logits',decoded.logits(j,:));
       R.predictions{end+1}=record;
      end
     end
    end
   end
  end
 end
end
failurecount=sum(cellfun(@(p)p.failure,R.predictions));R.status='PRODUCER_COMPLETE_PENDING_DIFFERENT_SAVED_AUDIT';if failurecount>0,R.status='TECHNICAL_HOLD_PREDICTION_FAILURES_RETAINED';end
R.failure_count=failurecount;
R.counts=struct('unique_stagefits',numel(R.unique_fits),'updates',numel(R.unique_fits)*updates,'logical_stageinstances',numel(R.logical_stages),'predictionrows',numel(R.predictions),'Fisher_jobs',numel(R.Fisher_jobs));
n=numel(folds)*numel(seeds);g=sum(arrayfun(@(f)sum([data.catalog.outer_fold]==f)/4,folds));
expectedpred=15*numel(seeds)*g*21;
assert(R.counts.unique_stagefits==63*n&&R.counts.logical_stageinstances==90*n&&R.counts.predictionrows==expectedpred&&(R.counts.Fisher_jobs==7*n||(data.isfixture&&updates==1&&R.counts.Fisher_jobs<=7*n)),'A29:ExecutionCounts','Exact job/alias/population counts failed.');
end
function t=term(kind,Z,target,mask,T,weight)
t=struct('kind',kind,'Z',Z,'target',target,'mask',mask,'T',double(T),'weight',double(weight));
end
function m=old_masks(stream,s)
if strcmp(stream,'TaskIL'),m=arrayfun(@(k)[1,k+1],1:s-1,'UniformOutput',false);else,m={1:s};end
end
function c=cost_card(D,cur,B,useReplay,arm,s,u,masks,teacher)
P=12+5*D;n=numel(cur.y);r=0;if useReplay,r=numel(B.y);end
c=struct('parameters',P,'optimizer_updates',u,'current_gradient_rows',n*u,'replay_gradient_rows',r*u,'prediction_payload_bytes',8*P+40+256*(D==8),'Adam_bytes',16*P+8,'teacher_forward_rows',n*~isempty(teacher),'KD_mask_terms',numel(masks)*u,'KD_student_gradient_rows',n*u*numel(masks),'replay_numeric_bytes',0,'EWC_retained_numeric_bytes',0,'teacher_bytes',8*numel(teacher),'qcache_bytes',0,'Fisher_input_rows',0,'Fisher_class_gradient_terms',0);
if useReplay,c.replay_numeric_bytes=40*2*(s+1);end
if D==4&&strcmp(arm.mechanism,'ewc')
 c.EWC_retained_numeric_bytes=16*P*min(s,2);
 if s<3,c.Fisher_input_rows=n;c.Fisher_class_gradient_terms=n*numel(cur.eligible);end
end
if D==8&&arm.negotiation,c.qcache_bytes=8*n*8;elseif ~isempty(masks),c.qcache_bytes=8*n*sum(cellfun(@numel,masks));end
c.meaning='Standalone logical stage work; scalar-array payload is not RAM. Shared producer jobs do not reduce algorithmic logical cost.';
end
function out=getarm(a,i)
if iscell(a),out=a{i};else,out=a(i);end
end
function y=sigmoid_local(x)
y=zeros(size(x));pos=x>=0;y(pos)=1./(1+exp(-x(pos)));e=exp(x(~pos));y(~pos)=e./(1+e);
end
function p=softmax_local(a)
a=a-max(a,[],2);e=exp(a);p=e./sum(e,2);
end
function x=getdefault(s,k,v)
if isfield(s,k),x=s.(k);else,x=v;end
end
function h=file_sha(path)
f=fopen(path,'rb');assert(f>=0,'A29:ReadPin','Cannot read pinned protocol.');cl=onCleanup(@()fclose(f));b=fread(f,Inf,'*uint8');md=java.security.MessageDigest.getInstance('SHA-256');md.update(typecast(b,'int8'));h=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2)',1,[])); %#ok<NASGU>
end

function verify_admission(config,A29,protocolpath,linkpath,updates,folds,seeds)
% Scientific handoff gate, not a security boundary against a human who can
% forge/edit code and files. No admission file is supplied by this package.
assert(isfield(config,'freeze_path')&&isfield(config,'freeze_sha256')&&isfield(config,'admission_id'),'A29:SourceTrainingClosed','Exact reviewed source run freeze is required.');
assert(isfile(config.freeze_path)&&strcmp(file_sha(config.freeze_path),config.freeze_sha256),'A29:FreezeHash','Run freeze hash mismatch.');
F=jsondecode(fileread(config.freeze_path));
assert(strcmp(F.status,'ADMITTED_MATCHED_CL_SOURCE_RUN')&&strcmp(F.admission_id,config.admission_id),'A29:FreezeStatus','Source run has no matching admission.');
assert(strcmp(F.protocol_sha256,file_sha(protocolpath))&&strcmp(F.input_links_sha256,file_sha(linkpath)),'A29:FreezeProtocol','Run protocol/input pins mismatch.');
assert(strcmp(F.root,A29)&&strcmp(F.mode,'source'),'A29:FreezeRoot','Run root/mode mismatch.');
assert(isscalar(folds)&&ismember(folds,1:5)&&F.fold==folds&&updates==600&&F.updates==600&&isequal(reshape(seeds,1,[]),[101,211,307,401,503])&&isequal(reshape(F.seeds,1,[]),seeds),'A29:FreezeScope','Run fold/seed/update scope mismatch.');
assert(isfield(config,'outDir')&&strcmp(F.out_dir,char(config.outDir))&&isfolder(config.outDir),'A29:FreezeOutput','Output scope mismatch.');
assert(F.wall_seconds==600&&F.output_cap_bytes==64*1024*1024&&strcmp(F.code_review_status,'PASS')&&strcmp(F.controls_status,'PASS')&&strcmp(F.resource_review_status,'PASS'),'A29:FreezeGates','Required review/control/resource gates missing.');
base=fileparts(mfilename('fullpath'));needed={'a29cl_data.m','a29cl_run.m','a29cl_math.m','a29cl_stage.m','a29cl_output.m'};pins=F.code_pins;
for k=1:numel(needed)
 idx=find(strcmp({pins.name},needed{k}));assert(isscalar(idx),'A29:FreezeCodePin','Required code pin missing.');
 assert(strcmp(file_sha(fullfile(base,needed{k})),pins(idx).sha256),'A29:FreezeCodeHash','Exact run code changed.');
end
% PASS text alone is insufficient: each role has a SHA-pinned acceptance
% proof card binding the same protocol, input links and complete code pin set.
% These proof cards are intentionally absent from this pretraining package.
assert(isfield(F,'proof_cards')&&isstruct(F.proof_cards),'A29:FreezeProofCards','Three exact acceptance proof cards are required.');
roles={'code_review','controls','resource'};cards=F.proof_cards;
assert(numel(cards)==3&&all(isfield(cards,{'role','path','sha256'})),'A29:FreezeProofSchema','Exactly three role/path/SHA proof references are required.');
for k=1:numel(roles)
 idx=find(strcmp({cards.role},roles{k}));assert(isscalar(idx),'A29:FreezeProofRole','Duplicate or missing proof role.');
 ref=cards(idx);assert(isfile(ref.path)&&strcmp(file_sha(ref.path),ref.sha256),'A29:FreezeProofHash','Acceptance proof file changed or is absent.');
 Q=jsondecode(fileread(ref.path));
 fields={'schema','status','role','protocol_sha256','input_links_sha256','code_pins','ledger_event_id','evidence'};
 assert(isstruct(Q)&&isscalar(Q)&&all(isfield(Q,fields)),'A29:FreezeProofSchema','Incomplete acceptance proof card.');
 assert(strcmp(Q.schema,'A29_MATCHED_CL_SOURCE_ADMISSION_PROOF_V1')&&strcmp(Q.status,'PASS')&&strcmp(Q.role,roles{k}),'A29:FreezeProofStatus','Proof role/status/schema mismatch.');
 assert(strcmp(Q.protocol_sha256,F.protocol_sha256)&&strcmp(Q.input_links_sha256,F.input_links_sha256),'A29:FreezeProofScope','Proof protocol/input scope mismatch.');
 assert(~isempty(regexp(Q.ledger_event_id,'^SD-[0-9]+$','once')),'A29:FreezeProofLedger','Proof must identify its recorded acceptance event.');
 same_code_pins(Q.code_pins,pins);
 E=Q.evidence;assert(isstruct(E)&&~isempty(E)&&all(isfield(E,{'path','sha256'})),'A29:FreezeProofEvidence','Proof must bind its review/control/resource evidence.');
 for j=1:numel(E)
  assert(isfile(E(j).path)&&strcmp(file_sha(E(j).path),E(j).sha256),'A29:FreezeProofEvidenceHash','Underlying proof evidence changed or is absent.');
 end
end
% Launcher/watchdog and result-audit paths are evidence requirements, not an
% assertion that this process itself can enforce OS wall-clock termination.
assert(~isempty(F.owned_watchdog_evidence)&&~isempty(F.output_bound_evidence),'A29:FreezeResourceEvidence','Finite launcher/output-bound evidence required.');
end

function same_code_pins(a,b)
% Explicit schema is code_pins[{name,sha256}], not a permissive pins alias.
assert(isstruct(a)&&isstruct(b)&&numel(a)==numel(b)&&all(isfield(a,{'name','sha256'}))&&all(isfield(b,{'name','sha256'})),'A29:FreezeProofCode','Proof code pin schema/count mismatch.');
assert(numel(unique(string({a.name})))==numel(a)&&numel(unique(string({b.name})))==numel(b),'A29:FreezeProofCode','Duplicate code pin names.');
for k=1:numel(b)
 idx=find(strcmp({a.name},b(k).name));assert(isscalar(idx)&&strcmp(a(idx).sha256,b(k).sha256),'A29:FreezeProofCode','Proof exact code pins differ from admitted run.');
end
end
