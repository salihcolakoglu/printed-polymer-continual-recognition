function result = a29_run_baselines_v3(versionRoot,outDir,runCfg)
% A29_RUN_BASELINES_V2 Admitted source-run producer. Authoring is not execution.
% All numerical calls require the root's explicit source-run admission and pins.
% Outputs: complete mathematical JSON, 17-digit prediction CSV and run receipt.
overall=tic;versionRoot=canonical(versionRoot);outDir=canonical(outDir);
assert(startsWith(lower(outDir),lower([versionRoot filesep])), ...
 'A29:OutputScope','Output must be a child of the A29 version folder.');
assert(isstruct(runCfg) && isfield(runCfg,'source_run_authorized') && ...
 isequal(runCfg.source_run_authorized,true) && ...
 strcmp(runCfg.admission_token,'A29_ADMITTED_SOURCE_RUN'), ...
 'A29:NotAdmitted','Root source-run admission required; do not execute during authoring.');
assert(runCfg.max_seconds>0 && runCfg.max_json_bytes>0 && ...
 runCfg.max_total_output_bytes>0,'A29:Budget','Finite positive output/time budgets required.');
if isfolder(outDir)
 d=dir(outDir);assert(~any(~ismember({d.name},{'.','..'})), ...
  'A29:ExistingOutput','Existing output is preserved; use a new empty run directory.');
else,mkdir(outDir);end
jobs=struct([]);aliases=struct([]);det=struct([]);preds=struct([]);norms=struct([]);
result=struct();started=times();meta=struct('started',started,'admission_token',runCfg.admission_token);
try
 corePath=fullfile(versionRoot,'03_MATLAB_Kodlari','a29_core_v1.m');
 producerPath=[mfilename('fullpath') '.m'];
 assert(strcmp(canonical(producerPath),canonical(fullfile(versionRoot,'03_MATLAB_Kodlari','a29_run_baselines_v3.m'))), ...
  'A29:ProducerPath','Unexpected producer location.');
 assert(strcmp(hashFile(corePath),runCfg.expected_core_sha256),'A29:CorePin','Core hash mismatch.');
 assert(strcmp(hashFile(producerPath),runCfg.expected_producer_sha256),'A29:ProducerPin','Producer hash mismatch.');
 protocolPath=fullfile(versionRoot,'02_Protokol','A29_Baseline_Protocol_v2.json');
 assert(strcmp(hashFile(protocolPath),runCfg.expected_protocol_sha256),'A29:ProtocolPin','Protocol hash mismatch.');
 P=jsondecode(fileread(protocolPath));
 assert(strcmp(P.id,'A29_BASELINE_GROUP5_FIXEDORDER_V2'),'A29:ProtocolID','Wrong protocol.');
 assert(~isempty(runCfg.checker_pins),'A29:CheckerPins','Independent checker pins required.');
 for j=1:numel(runCfg.checker_pins)
  pin=runCfg.checker_pins(j);assert(strcmp(hashFile(pin.path),pin.sha256), ...
   'A29:CheckerPin','Checker hash mismatch.');
 end
 pin=runCfg.control_receipt_pin;
 assert(strcmp(hashFile(pin.path),pin.sha256),'A29:ControlPin','Accepted control receipt hash mismatch.');
 cardPath=fullfile(versionRoot,strrep(P.source.fold_admission_card.path,'/',filesep));
 assert(strcmp(hashFile(cardPath),P.source.fold_admission_card.sha256), ...
  'A29:AdmissionPin','Source/fold admission hash mismatch.');
 card=jsondecode(fileread(cardPath));
 sourcePath=fullfile(versionRoot,strrep(P.source.input,'/',filesep));
 d=dir(sourcePath);assert(isscalar(d) && d.bytes==P.source.bytes,'A29:SourceBytes','Source byte count mismatch.');
 assert(strcmp(hashFile(sourcePath),P.source.sha256),'A29:SourcePin','Source hash mismatch.');
 addpath(fileparts(corePath));
 T=readtable(sourcePath,'TextType','string','VariableNamingRule','preserve');
 validateSource(T,card);T=sortrows(T,{'group_order','label'});
 Xraw=double([T.E_MPa T.Hardness_ShoreD]);Y=double(T.label);
 assert(all(isfinite(Xraw),'all'),'A29:SourceFinite','Nonfinite permitted measurement.');
 folds=mod(double(T.group_order),5)+1;
 cfg=a29_core_v1('default_config');
 assert(cfg.updates==P.linear_training.updates_per_stage && cfg.lambda==1e-4, ...
  'A29:CoreConfig','Default core training config differs from protocol.');
 seedList=double(P.linear_training.seeds(:)');
 pairs={[1 2],[1 3],[1 4]};newClasses={[1 2],3,4};
 meta=struct('status','RUNNING','started',started,'software',version, ...
  'matlab_release',version('-release'),'computer',computer,'executor_pid',feature('getpid'), ...
  'protocol_path',protocolPath,'protocol_sha256',hashFile(protocolPath), ...
  'source_path',sourcePath,'source_sha256',hashFile(sourcePath), ...
  'admission_path',cardPath,'admission_sha256',hashFile(cardPath), ...
  'core_path',corePath,'core_sha256',hashFile(corePath), ...
  'producer_path',producerPath,'producer_sha256',hashFile(producerPath), ...
  'checker_pins',runCfg.checker_pins,'control_receipt_pin',runCfg.control_receipt_pin, ...
  'all_records_previously_inspected',true,'new_blind_test',false, ...
  'independent_review_accepted',false,'logical_counter_note', ...
  'Protocol100three-stage series means linear series;120including20deterministic series.');
 catalog=repmat(struct('parentrow','','group','','group_order',0,'label',0,'outer_fold',0),height(T),1);
 for i=1:height(T)
  catalog(i,1)=struct('parentrow',char(T.row_id(i)),'group',char(T.condition_group(i)), ...
   'group_order',double(T.group_order(i)),'label',Y(i),'outer_fold',folds(i));
 end
 for f=1:5
  budget(overall,runCfg);train=folds~=f;held=folds==f;
  first=find(train & ismember(Y,[1 2]));n=a29_core_v1('fit_normalizer',Xraw(first,:),cellstr(T.row_id(first)));
  assert(n.n==P.outer_groups.first_stage_train_rows(f),'A29:FirstCount','Wrong first-stage normalizer count.');
  n.fold=f;norms=appendStruct(norms,n); %#ok<AGROW>
  Z=a29_core_v1('normalize',Xraw,n);
  proto=a29_core_v1('init_proto');nn=a29_core_v1('init_nn');
  admitted=[];
  for stage=1:3
   budget(overall,runCfg);new=find(train & ismember(Y,newClasses{stage}));
   admitted=sort([admitted(:);new(:)]);seen=1:(stage+1);
   proto=a29_core_v1('admit_proto',proto,Z(new,:),Y(new));
   nn=a29_core_v1('admit_nn',nn,Z(new,:),Y(new),cellstr(T.row_id(new)));
   % Reorder retained NN memory canonically; do not reread old feature rows.
   [found,sourcePositions]=ismember(string(nn.row_ids),T.row_id);
   assert(all(found),'A29:NNIdentity','UnknownretainedNNparent.');
   [~,referenceOrder]=sort(sourcePositions);
   nn.X=nn.X(referenceOrder,:);nn.y=nn.y(referenceOrder);
   nn.row_ids=nn.row_ids(referenceOrder);
   batch=nan(4,2);
   for k=seen
    r=find(train & Y==k);sumv=zeros(1,2);
    for j=1:numel(r),sumv=sumv+Z(r(j),:);end
    batch(k,:)=sumv/numel(r);
   end
   pc=nan(4,2);for k=seen,pc(k,:)=proto.sumS(k,:)/double(proto.countn(k));end
   delta=abs(pc(seen,:)-batch(seen,:));tol=1e-12+1e-10*abs(batch(seen,:));
   assert(all(delta<=tol,'all'),'A29:CentroidIdentity','Incremental/batch centroid mismatch.');
   sid=sprintf('D_f%d_s%d',f,stage);
   for stream={'TIL','CIL'}
    preds=appendViews(preds,T,Z,held,stream{1},stage,'PROTO',NaN,sid,proto,'proto',pairs,batch);
    preds=appendViews(preds,T,Z,held,stream{1},stage,'NN1_FULL',NaN,sid,nn,'nn',pairs,[]);
   end
   ds=struct('state_id',sid,'fold',f,'stage',stage,'normalizer_fold',f, ...
    'admitted_row_ids',{cellstr(T.row_id(admitted))},'proto',proto,'nn',nn, ...
    'batch_centroid',batch,'batch_identity',struct('max_centroid_absdiff',max(delta,[],'all'), ...
    'centroid_within_tolerance',true,'query_labels_ties_checked',true));
   det=appendStruct(det,ds); %#ok<AGROW>
  end
  for seed=seedList
   budget(overall,runCfg);initial=a29_core_v1('init_linear',seed);
   initModel=struct('W',initial.W,'b',initial.b);
   id=sprintf('J_f%d_q%d_common_s1',f,seed);
   job=trainJob(id,f,seed,1,'common_stage1','',first,[1 2],initial,initModel,T,Z,Y,cfg);
   jobs=appendStruct(jobs,job); %#ok<AGROW>
   seqT=job.end_state;seqC=job.end_state;priorT=id;priorC=id;
   for stream={'TIL','CIL'}
    for method={'LIN_SEQ','LIN_JOINT_GLOBAL'}
     aliases=addAlias(aliases,f,stream{1},1,method{1},seed,id);
     preds=appendViews(preds,T,Z,held,stream{1},1,method{1},seed,id,job.end_state,'linear',pairs,[]);
    end
   end
   for stage=2:3
    budget(overall,runCfg);seen=1:(stage+1);
    rowsT=find(train & ismember(Y,pairs{stage}));
    idT=sprintf('J_f%d_q%d_seq_TIL_s%d',f,seed,stage);
    jobT=trainJob(idT,f,seed,stage,'seq_TIL',priorT,rowsT,pairs{stage},seqT,initModel,T,Z,Y,cfg);
    jobs=appendStruct(jobs,jobT);seqT=jobT.end_state;priorT=idT; %#ok<AGROW>
    rowsC=find(train & Y==stage+1);
    idC=sprintf('J_f%d_q%d_seq_CIL_s%d',f,seed,stage);
    jobC=trainJob(idC,f,seed,stage,'seq_CIL',priorC,rowsC,seen,seqC,initModel,T,Z,Y,cfg);
    jobs=appendStruct(jobs,jobC);seqC=jobC.end_state;priorC=idC; %#ok<AGROW>
    fresh=a29_core_v1('init_linear',seed);allRows=find(train & ismember(Y,seen));
    idJ=sprintf('J_f%d_q%d_joint_global_s%d',f,seed,stage);
    jobJ=trainJob(idJ,f,seed,stage,'joint_global','',allRows,seen,fresh,initModel,T,Z,Y,cfg);
    jobs=appendStruct(jobs,jobJ); %#ok<AGROW>
    aliases=addAlias(aliases,f,'TIL',stage,'LIN_SEQ',seed,idT);
    preds=appendViews(preds,T,Z,held,'TIL',stage,'LIN_SEQ',seed,idT,seqT,'linear',pairs,[]);
    aliases=addAlias(aliases,f,'CIL',stage,'LIN_SEQ',seed,idC);
    preds=appendViews(preds,T,Z,held,'CIL',stage,'LIN_SEQ',seed,idC,seqC,'linear',pairs,[]);
    for stream={'TIL','CIL'}
     aliases=addAlias(aliases,f,stream{1},stage,'LIN_JOINT_GLOBAL',seed,idJ);
     preds=appendViews(preds,T,Z,held,stream{1},stage,'LIN_JOINT_GLOBAL',seed,idJ,jobJ.end_state,'linear',pairs,[]);
    end
   end
  end
  progress=struct('status','RUNNING','completed_folds',f,'linear_jobs',numel(jobs), ...
   'logical_predictions',numel(preds),'elapsed_seconds',toc(overall),'time',times());
  writeJson(fullfile(outDir,'a29_progress_v1.json'),progress,runCfg.max_json_bytes);
 end
 assert(numel(jobs)==175 && numel(aliases)==300 && numel(det)==15 && numel(norms)==5, ...
  'A29:StateCounts','Unexpected unique/logical state counts.');
 assert(sum(arrayfun(@(j)j.trace.updates,jobs))==105000,'A29:UpdateCount','Expected105000actualupdates.');
 assert(numel(preds)==6552,'A29:PredictionCount','Expected6552logicalheldout predictions.');
 budget(overall,runCfg);summaries=summarize(preds,seedList);costs=costSummary(preds,T,folds,seedList);
 failCount=sum(strcmp({preds.status},'failure'));
 meta.status='PRODUCER_COMPLETED_PENDING_INDEPENDENT_REVIEW';
 if failCount>0,meta.status='COMPLETE_WITH_PREDICTION_FAILURE_HOLD';end
 meta.completed=true;meta.independent_review_accepted=false;meta.failure_prediction_rows=failCount;
 meta.unique_linear_fits=numel(jobs);meta.unique_linear_updates=sum(arrayfun(@(j)j.trace.updates,jobs));
 meta.logical_linear_stage_instances=numel(aliases);meta.logical_prediction_rows=numel(preds);
 meta.logical_linear_three_stage_series=100;meta.logical_all_method_three_stage_series=120;
 meta.logical_population_count=numel(summaries.population_counts);
 meta.elapsed_before_serialization_seconds=toc(overall);meta.completed_numerics=times();
 meta.timing_scope='Fullproducerincludingserializationinhandoffreceipt; thisfieldendsofnumerics.';
 result=struct('schema','A29_BASELINE_RESULT_V1','metadata',meta,'config',cfg, ...
  'source_record_catalog',catalog,'normalizers',norms,'linear_jobs',jobs,'aliases',aliases, ...
  'deterministic_states',det,'predictions',preds,'summaries',summaries,'costs',costs);
 csvPath=fullfile(outDir,'a29_predictions_v1.csv');writeCSV(csvPath,preds);
 jsonPath=fullfile(outDir,'a29_baseline_result_v1.json');writeJson(jsonPath,result,runCfg.max_json_bytes);
 assert(sumFiles(outDir)<=runCfg.max_total_output_bytes,'A29:OutputCap','Totaloutputbytecapexceeded.');
 receipt=struct('schema','A29_PRODUCER_RECEIPT_V1','status',meta.status,'complete',true, ...
  'started',started,'finished',times(),'actual_elapsed_seconds',toc(overall), ...
  'unique_linear_fits',175,'unique_linear_updates',105000,'logical_predictions',6552, ...
  'independent_review_accepted',false,'result_path',jsonPath,'result_sha256',hashFile(jsonPath), ...
  'result_bytes',fileBytes(jsonPath),'csv_path',csvPath,'csv_sha256',hashFile(csvPath), ...
  'csv_bytes',fileBytes(csvPath),'pins',meta,'run_config',runCfg);
 writeJson(fullfile(outDir,'a29_producer_receipt_v1.json'),receipt,runCfg.max_json_bytes);
 assert(sumFiles(outDir)<=runCfg.max_total_output_bytes,'A29:OutputCap','Totaloutputbytecapexceeded.');
catch err
 failure=struct('schema','A29_PRODUCER_FAILURE_V1','status','FAILED_PRESERVED', ...
  'started',started,'finished',times(),'actual_elapsed_seconds',toc(overall), ...
  'identifier',err.identifier,'message',err.message,'stack',err.stack, ...
  'complete',false,'linear_jobs_returned',numel(jobs),'logical_predictions_returned',numel(preds), ...
  'metadata',meta,'no_science_acceptance',true);
 try,writeJson(fullfile(outDir,'a29_FAILED_receipt_v1.json'),failure,runCfg.max_json_bytes);catch,disp(failure);end
 partial=struct('schema','A29_PARTIAL_RESULT_V1','metadata',failure,'normalizers',norms, ...
  'linear_jobs',jobs,'aliases',aliases,'deterministic_states',det,'predictions',preds);
 try,writeJson(fullfile(outDir,'a29_PARTIAL_result_v1.json'),partial,runCfg.max_json_bytes);catch,disp('PartialJSONcouldnotbewrittenwithinbudget.');end
 rethrow(err);
end
end

function validateSource(T,card)
expected=cellstr(string(card.source.headers_exact_order(:)))';
assert(isequal(T.Properties.VariableNames,expected),'A29:Headers','ExactCSVheadersrequired.');
assert(height(T)==104 && numel(unique(T.row_id))==104,'A29:Parents','104uniqueparentsrequired.');
g=double(T.group_order);y=double(T.label);
assert(all(g==fix(g)) && isequal(unique(g)',0:25) && all(ismember(y,1:4)), ...
 'A29:GroupOrder','Original0..25orderandfourlabelsrequired.');
assert(numel(unique(T.condition_group))==26,'A29:Groups','26wholegroupsrequired.');
for j=1:numel(card.whole_group_assignments)
 a=card.whole_group_assignments(j);r=find(T.condition_group==string(a.condition_group));
 assert(numel(r)==4 && all(g(r)==a.group_order_source_0based) && ...
  isequal(sort(y(r))',[1 2 3 4]),'A29:GroupMap','Admissiongroupmismatch.');
 for k=1:4
  q=r(y(r)==k);parent=a.parent_records([a.parent_records.label]==k);
  assert(isscalar(q) && isscalar(parent) && T.row_id(q)==string(parent.row_id), ...
   'A29:ParentMap','Exactclassparentmismatch.');
 end
 assert(mod(a.group_order_source_0based,5)+1==a.outer_fold,'A29:FoldMap','Foldmismatch.');
end
end

function job=trainJob(id,fold,seed,stage,kind,prior,rows,eligible,start,init,T,Z,Y,cfg)
timer=tic;[finish,trace]=a29_core_v1('train_linear',start,Z(rows,:),Y(rows),eligible,cfg);
job=struct('job_id',id,'fold',fold,'seed',seed,'stage',stage,'kind',kind, ...
 'start_job_id',prior,'training_row_ids',{cellstr(T.row_id(rows))}, ...
 'eligible',double(eligible(:)'),'initial_model',init,'start_state',start, ...
 'end_state',finish,'trace',trace,'elapsed_seconds',toc(timer));
end

function a=addAlias(a,f,stream,stage,method,seed,id)
x=struct('fold',f,'stream',stream,'stage',stage,'method',method,'seed',seed,'job_id',id);
a=appendStruct(a,x);
end

function rows=appendViews(rows,T,Z,held,stream,stage,method,seed,sid,state,kind,pairs,batch)
if strcmp(stream,'TIL'),taskIDs=1:stage;else,taskIDs=NaN;end
for task=taskIDs
 if isnan(task),eligible=1:(stage+1);else,eligible=pairs{task};end
 q=find(held & ismember(double(T.label),eligible));o=a29_core_v1('predict',kind,state,Z(q,:),eligible);
 if strcmp(kind,'proto')
  ref=nan(numel(q),4);
  for k=eligible,ref(:,k)=-sum((Z(q,:)-batch(k,:)).^2,2);end
  refDecision=a29_core_v1('decode',ref,eligible);
  assert(isequaln(o.prediction,refDecision.prediction) && isequal(o.tie,refDecision.tie), ...
   'A29:BatchDecision','Batchcentroidquerylabels/tiesdiffer.');
  delta=abs(o.scores(:,eligible)-ref(:,eligible));tol=1e-12+1e-10*abs(ref(:,eligible));
  assert(all(delta<=tol,'all'),'A29:BatchScores','Batchcentroidscoresdiffer.');
 end
 for j=1:numel(q)
  i=q(j);x=struct('prediction_id',sprintf('P_%05d',numel(rows)+1), ...
   'fold',mod(double(T.group_order(i)),5)+1,'stream',stream,'stage',stage, ...
   'method',method,'seed',seed,'parentrow',char(T.row_id(i)), ...
   'group',char(T.condition_group(i)),'group_order',double(T.group_order(i)), ...
   'task',task,'truth',double(T.label(i)),'eligible',double(eligible), ...
   'scores',o.scores(j,:),'logits',o.logits(j,:),'distances',o.distances(j,:), ...
   'gap',o.gap(j),'prediction',o.prediction(j),'tie',o.tie(j), ...
   'status',o.status{j},'failure',o.failure{j},'state_id',sid);
  rows=appendStruct(rows,x); %#ok<AGROW>
 end
end
end

function out=summarize(P,seeds)
pop=struct([]);units=struct([]);curves=struct([]);conf=struct([]);ret=struct([]);
methods={'PROTO','NN1_FULL','LIN_SEQ','LIN_JOINT_GLOBAL'};
for f=1:5
 for st={'TIL','CIL'}
  stream=st{1};
  for mi=1:4
   method=methods{mi};ss=seeds;if mi<=2,ss=NaN;end
   for seed=ss
    for stage=1:3
     mask=match(P,f,stream,method,seed,stage);R=P(mask);assert(~isempty(R),'A29:Population','Missingpopulation.');
     valid=strcmp({R.status},'valid');correct=valid & [R.prediction]==[R.truth];
     if strcmp(stream,'TIL'),unitID=[R.task];maxUnit=stage;intro=1:stage;
     else,unitID=[R.truth];maxUnit=stage+1;intro=[1 1 2 3];end
     acc=zeros(1,maxUnit);current=false(size(unitID));old=current;
     for u=1:maxUnit
      chosen=unitID==u;N=sum(chosen);C=sum(correct(chosen));F=sum(~valid(chosen));acc(u)=C/N;
      U=context(f,stream,stage,method,seed);U.unit_id=u;U.intro_stage=intro(u);
      U.N=N;U.correct=C;U.failure=F;U.accuracy=acc(u);units=appendStruct(units,U); %#ok<AGROW>
      if intro(u)==stage,current=current|chosen;else,old=old|chosen;end
     end
     C=context(f,stream,stage,method,seed);C.N=numel(R);C.forced_correct=sum(correct);
     C.valid_wrong=sum(valid & ~correct);C.failure=sum(~valid);C.forced_accuracy=sum(correct)/numel(R);
     C.unit_macro_accuracy=mean(acc);C.current_N=sum(current);C.current_correct=sum(correct & current);
     C.current_accuracy=C.current_correct/C.current_N;C.old_N=sum(old);C.old_correct=sum(correct & old);
     C.old_accuracy=NaN;if C.old_N>0,C.old_accuracy=C.old_correct/C.old_N;end
     C.ties=sum([R.tie]);pop=appendStruct(pop,C); %#ok<AGROW>
     V=context(f,stream,stage,method,seed);V.cuts=a29_core_v1('coverage',R);curves=appendStruct(curves,V); %#ok<AGROW>
     if strcmp(stream,'CIL'),tasks=NaN;else,tasks=1:stage;end
     for task=tasks
      if isnan(task),Q=R;labels=1:(stage+1);else,Q=R([R.task]==task);labels=[1 task+1];end
      M=zeros(numel(labels));F=zeros(1,numel(labels));
      for j=1:numel(Q)
       tr=find(labels==Q(j).truth);
       if strcmp(Q(j).status,'failure'),F(tr)=F(tr)+1;
       else,pr=find(labels==Q(j).prediction);M(tr,pr)=M(tr,pr)+1;end
      end
      K=context(f,stream,stage,method,seed);K.task=task;K.labels=labels;K.matrix=M;K.failure_by_truth=F;
      conf=appendStruct(conf,K); %#ok<AGROW>
     end
    end
    for stage=1:3
     if strcmp(stream,'TIL'),oldUnits=1:(stage-1);intro=1:3;
     else,intro=[1 1 2 3];oldUnits=find(intro<stage);end
     bwt=[];forget=[];
     for u=oldUnits
      final=unitAccuracy(units,f,stream,method,seed,stage,u);
      first=unitAccuracy(units,f,stream,method,seed,intro(u),u);earlier=[];
      for l=intro(u):(stage-1),earlier(end+1)=unitAccuracy(units,f,stream,method,seed,l,u);end %#ok<AGROW>
      bwt(end+1)=final-first;forget(end+1)=max(earlier)-final; %#ok<AGROW>
     end
     B=context(f,stream,stage,method,seed);B.old_unit_ids=oldUnits;B.BWT=NaN;B.signed_forgetting=NaN;
     if ~isempty(bwt),B.BWT=mean(bwt);B.signed_forgetting=mean(forget);end
     ret=appendStruct(ret,B); %#ok<AGROW>
    end
   end
  end
 end
end
final=struct([]);
for st={'TIL','CIL'}
 for mi=1:4
  method=methods{mi};ss=seeds;if mi<=2,ss=NaN;end
  for seed=ss
   ix=strcmp({P.stream},st{1}) & strcmp({P.method},method) & seedEqual([P.seed],seed) & [P.stage]==3;
   R=P(ix);valid=strcmp({R.status},'valid');cor=valid & [R.prediction]==[R.truth];
   if strcmp(st{1},'CIL'),uid=[R.truth];K=4;else,uid=[R.task];K=3;end
   den=zeros(1,K);correct=den;for u=1:K,den(u)=sum(uid==u);correct(u)=sum(cor & uid==u);end
   x=struct('stream',st{1},'method',method,'seed',seed,'N',numel(R), ...
    'unique_parents',numel(unique({R.parentrow})),'groups',numel(unique({R.group})), ...
    'unit_N',den,'unit_correct',correct,'unit_macro_accuracy',mean(correct./den), ...
    'correct',sum(cor),'valid_wrong',sum(valid & ~cor),'failure',sum(~valid), ...
    'forced_accuracy',sum(cor)/numel(R));final=appendStruct(final,x); %#ok<AGROW>
  end
 end
end
paired=struct([]);groups=unique({P.group});
for st={'TIL','CIL'}
 for gi=1:numel(groups)
  for contrast=1:3
   if contrast==1,ss=NaN;left='PROTO';right='NN1_FULL';
   elseif contrast==2,ss=seeds;left='LIN_SEQ';right='PROTO';
   else,ss=seeds;left='LIN_JOINT_GLOBAL';right='LIN_SEQ';end
   for seed=ss
    lseed=seed;rseed=seed;if strcmp(right,'PROTO'),rseed=NaN;end
    base=strcmp({P.stream},st{1}) & strcmp({P.group},groups{gi}) & [P.stage]==3;
    L=P(base & strcmp({P.method},left) & seedEqual([P.seed],lseed));
    R=P(base & strcmp({P.method},right) & seedEqual([P.seed],rseed));
    assert(numel(L)==numel(R) && ~isempty(L),'A29:PairedPopulation','Pairedgroupmismatch.');
    lc=sum(strcmp({L.status},'valid') & [L.prediction]==[L.truth]);
    rc=sum(strcmp({R.status},'valid') & [R.prediction]==[R.truth]);
    x=struct('stream',st{1},'group',groups{gi},'fold',L(1).fold,'left',left,'right',right, ...
     'seed',seed,'N',numel(L),'left_correct',lc,'right_correct',rc,'accuracy_difference',(lc-rc)/numel(L));
    paired=appendStruct(paired,x); %#ok<AGROW>
   end
  end
 end
end
out=struct('population_counts',pop,'unit_metrics',units,'confusions',conf, ...
 'coverage_curves',curves,'retention',ret,'final_oof',final,'paired_group_differences',paired, ...
 'scope','ExploratoryOOFcounts;noinferentialtestsorphysicalconfidenceintervals.');
end

function C=costSummary(P,T,folds,seeds)
C=struct([]);methods={'PROTO','NN1_FULL','LIN_SEQ','LIN_JOINT_GLOBAL'};
for f=1:5
 g=numel(unique(T.group_order(folds~=f)));
 for st={'TIL','CIL'}
  for mi=1:4
   ss=seeds;if mi<=2,ss=NaN;end
   for seed=ss
    for stage=1:3
     R=P(match(P,f,st{1},methods{mi},seed,stage));n=numel(R);seen=stage+1;refs=g*seen;
     x=context(f,st{1},stage,methods{mi},seed);
     x.normalizer_bytes=32;x.active_mask_bytes=4;x.vocabulary_bytes=4;
     x.prototypes=0;x.raw_retained_records=0;x.parameter_count=0;x.optimizer_bytes=0;
     x.training_history_bytes=0;x.updates_this_stage=0;x.cumulative_updates=0;
     x.gradient_examples_this_stage=0;x.cumulative_gradient_examples=0;
     x.admission_rows_cumulative=0;x.distance_comparisons=0;x.affine_outputs=0;
     candidateCounts=arrayfun(@(a)numel(a.eligible),R);
     if mi==1
      x.prototypes=seen;x.prediction_state_bytes=32+4+4+64+32;
      x.admission_rows_cumulative=refs;x.distance_comparisons=sum(candidateCounts);
     elseif mi==2
      x.raw_retained_records=refs;x.prediction_state_bytes=32+4+4+refs*24;
      x.admission_rows_cumulative=refs;x.distance_comparisons=sum(candidateCounts)*g;
     else
      x.parameter_count=12;x.prediction_state_bytes=32+4+4+96;x.optimizer_bytes=200;
      x.updates_this_stage=600;x.cumulative_updates=600*stage;x.affine_outputs=n*4;
      if mi==3
       if strcmp(st{1},'TIL'),now=2*g;cumulative=2*g*stage;
       else,now=g;if stage==1,now=2*g;end;cumulative=g*(stage+1);end
      else
       now=refs;cumulative=g*sum(2:seen);x.raw_retained_records=refs;x.training_history_bytes=refs*24;
      end
      x.gradient_examples_this_stage=600*now;x.cumulative_gradient_examples=600*cumulative;
     end
     x.total_persistent_training_bytes=x.prediction_state_bytes+x.optimizer_bytes+x.training_history_bytes;
     x.query_views=n;x.audit_ID_bytes_excluded=true;x.saved_container_bytes_excluded=true;
     x.peak_RAM_measured=false;x.logical_standalone_cost=true;C=appendStruct(C,x); %#ok<AGROW>
    end
   end
  end
 end
end
end

function rows=appendStruct(rows,item)
assert(isstruct(rows) && isstruct(item) && isscalar(item), ...
 'A29:AppendType','Struct array and scalar struct item required.');
if isempty(rows)
 rows=item;
else
 assert(isequal(fieldnames(rows),fieldnames(item)), ...
  'A29:AppendFields','Appended record fields/order must match.');
 rows(end+1,1)=item;
end
end

function x=context(f,stream,stage,method,seed)
x=struct('fold',f,'stream',stream,'stage',stage,'method',method,'seed',seed);
end
function m=match(P,f,stream,method,seed,stage)
m=[P.fold]==f & strcmp({P.stream},stream) & strcmp({P.method},method) & ...
 seedEqual([P.seed],seed) & [P.stage]==stage;
end
function m=seedEqual(values,seed)
if isnan(seed),m=isnan(values);else,m=values==seed;end
end
function a=unitAccuracy(U,f,stream,method,seed,stage,unit)
m=[U.fold]==f & strcmp({U.stream},stream) & strcmp({U.method},method) & ...
 seedEqual([U.seed],seed) & [U.stage]==stage & [U.unit_id]==unit;
assert(sum(m)==1,'A29:UnitLookup','Missing/duplicateunitmetric.');a=U(m).accuracy;
end
function budget(timer,cfg)
assert(toc(timer)<=cfg.max_seconds,'A29:TimeCap','Producerelapsedtimebudgetexceeded.');
end
function t=times()
t=struct('utc',char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss.SSSXXX')), ...
 'istanbul',char(datetime('now','TimeZone','Europe/Istanbul','Format','yyyy-MM-dd''T''HH:mm:ss.SSSXXX')));
end
function p=canonical(p)
p=char(java.io.File(char(p)).getCanonicalPath());
end
function h=hashFile(p)
f=fopen(p,'rb');assert(f>=0,'A29:PinFile','Cannotreadpinnedfile:%s',p);c=onCleanup(@()fclose(f));
b=fread(f,Inf,'*uint8');md=java.security.MessageDigest.getInstance('SHA-256');md.update(typecast(b(:),'int8'));
h=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2)',1,[]));
end
function n=fileBytes(p)
d=dir(p);n=d.bytes;
end
function n=sumFiles(folder)
d=dir(folder);n=sum([d(~[d.isdir]).bytes]);
end
function writeJson(p,obj,cap)
t=jsonencode(obj,'ConvertInfAndNaN',true);b=unicode2native(t,'UTF-8');
assert(numel(b)<=cap,'A29:JSONCap','JSONexceedsdeclaredbytecap.');
tmp=[p '.tmp.' char(java.util.UUID.randomUUID())];f=fopen(tmp,'wb');
assert(f>=0,'A29:Write','CannotcreatetemporaryJSON.');c=onCleanup(@()fclose(f));
assert(fwrite(f,b,'uint8')==numel(b),'A29:Write','ShortJSONwrite.');clear c;
[ok,msg]=movefile(tmp,p,'f');assert(ok,'A29:AtomicRename','%s',msg);
end
function writeCSV(p,R)
tmp=[p '.tmp.' char(java.util.UUID.randomUUID())];f=fopen(tmp,'w','n','UTF-8');
assert(f>=0,'A29:Write','CannotcreatepredictionCSV.');c=onCleanup(@()fclose(f));
header={'prediction_id','fold','stream','stage','method','seed','parentrow','group','group_order','task','truth', ...
 'eligible1','eligible2','eligible3','eligible4','score1','score2','score3','score4', ...
 'logit1','logit2','logit3','logit4','distance1','distance2','distance3','distance4', ...
 'gap','prediction','tie','status','failure','state_id'};
fprintf(f,'%s\n',strjoin(header,','));
for i=1:numel(R)
 r=R(i);eligible=double(ismember(1:4,r.eligible));
 values={quote(r.prediction_id),num(r.fold),quote(r.stream),num(r.stage),quote(r.method),num(r.seed), ...
  quote(r.parentrow),quote(r.group),num(r.group_order),num(r.task),num(r.truth)};
 for v=[eligible,r.scores,r.logits,r.distances,r.gap,r.prediction,double(r.tie)],values{end+1}=num(v);end %#ok<AGROW>
 values=[values,{quote(r.status),quote(r.failure),quote(r.state_id)}];fprintf(f,'%s\n',strjoin(values,','));
end
clear c;[ok,msg]=movefile(tmp,p,'f');assert(ok,'A29:AtomicRename','%s',msg);
end
function s=num(x)
if isnan(x),s='';else,s=sprintf('%.17g',x);end
end
function s=quote(x)
s=['"' strrep(char(x),'"','""') '"'];
end
