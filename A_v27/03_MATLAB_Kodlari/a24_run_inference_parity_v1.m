function report = a24_run_inference_parity_v1(learningDir,outputDir)
% One future inference-only producer; launch AFTER different saved-learning review.
% Root caller owns a single MATLAB process with a300-second maximum.
% JSON checkpoint weights and original kernel reference are compared to SLX.
versionRoot=fileparts(fileparts(mfilename('fullpath')));
assert(~isfolder(outputDir) && ~isfile(outputDir),'A24SLX:ExistingOutput','Output directory must be new.');
name='a24_A_inference_v1';
modelPath=fullfile(versionRoot,'04_Simulink_Modelleri',[name,'.slx']);
bankPath=fullfile(versionRoot,'05_Ogrenilmis_Modeller','a24_final_stage_inference_bank_v1.mat');
assert(~isfile(modelPath) && ~isfile(bankPath) && ~bdIsLoaded(name), ...
 'A24SLX:ExistingArtifacts','Preserve prior artifacts; no automatic repeat.');
mkdir(outputDir);started=tic;oldPath=path;pathCleanup=onCleanup(@()path(oldPath)); %#ok<NASGU>
addpath(fullfile(versionRoot,'03_MATLAB_Kodlari'),fullfile(versionRoot,'03_MATLAB_Kodlari','utility'));
report=struct('schema','A24_NEW_SIMULINK_PARITY_RESULT_V1','status','RUNNING', ...
 'checkpoints_completed',0,'decision_views_completed',0,'new_training',0, ...
 'new_physical_validation',0,'record_ticks_are_physical_time',false, ...
 'fresh_blind_test',false,'old_A_retrained',false,'B_source_join',false);
rows=struct([]);summaries=struct([]);example=struct();inferenceBank=cell(55,1);
try
 protocol=jsondecode(fileread(fullfile(versionRoot,'06_Protokoller','A24_new_inference_parity_protocol_v1.json')));
 assert(protocol.expected_checkpoints==55 && protocol.expected_decision_views==1040 && protocol.stage==3, ...
  'A24SLX:Protocol','Unexpected prospective scope.');tol=protocol.numeric_tolerance;
 producer=jsondecode(fileread(fullfile(learningDir,'execution_receipt_v1.json')));
 assert(strcmp(producer.status,'COMPLETED_PRODUCER_PENDING_DIFFERENT_SAVED_REVIEW'), ...
  'A24SLX:LearningIncomplete','Expected completed learning producer; independent acceptance is a caller prerequisite.');
 report.learning_producer_status=producer.status;
 % One JSON weights read. No old A/B MAT payload is reopened.
 rawBank=jsondecode(fileread(fullfile(learningDir,'checkpoints_v1.json')));
 if iscell(rawBank)
  allBank=rawBank;
 else
  allBank=cell(numel(rawBank),1);for j=1:numel(rawBank),allBank{j}=rawBank(j);end
 end
 assert(numel(allBank)==165,'A24SLX:BankPopulation','Expected165 stage checkpoints.');
 finalBank=allBank(cellfun(@(cp)cp.stage==3,allBank));
 assert(numel(finalBank)==55,'A24SLX:BankPopulation','Expected55 final-stage checkpoints.');
 categories={'TaskIncremental','ClassIncremental'};
 codeMethods={'onehot_no_negotiation','onehot_negotiated','walsh_no_negotiation','walsh_negotiated'};
 ceMethods={'sequential_no_replay','sequential_replay','joint_seen_reference'};seeds=[101,211,307,401,503];
 expectedIDs=strings(55,1);pos=0;
 for ci=1:2
  if ci==1,methods=codeMethods;else,methods=[codeMethods,ceMethods];end
  for mi=1:numel(methods)
   for si=1:5
    pos=pos+1;expectedIDs(pos)=sprintf('%s_%s_seed%d_stage3',categories{ci},methods{mi},seeds(si));
   end
  end
 end
 actualIDs=strings(55,1);for j=1:55,actualIDs(j)=string(finalBank{j}.id);end
 assert(numel(unique(actualIDs))==55 && isequal(sort(actualIDs),sort(expectedIDs)), ...
  'A24SLX:BankIdentity','Final checkpoint method/stream/seed identities changed.');
 [found,order]=ismember(expectedIDs,actualIDs);assert(all(found),'A24SLX:BankIdentity','Missing checkpoint.');finalBank=finalBank(order);
 normalizer=jsondecode(fileread(fullfile(learningDir,'normalizer_v1.json')));
 mu=reshape(normalizer.mean,1,2);sd=reshape(normalizer.sample_sd,1,2);
 assert(all(isfinite(mu)) && all(isfinite(sd)) && all(sd>0),'A24SLX:Scale','Invalid frozen TRAIN scale.');
 T=readtable(fullfile(versionRoot,'02_Veriler','A_source_average_records_v1.csv'),'TextType','string');
 assert(height(T)==104 && numel(unique(T.row_id))==104,'A24SLX:Source','Expected104 published average records.');
 T=T(T.partition=="REPORTING_DEVELOPMENT",:);
 assert(height(T)==16 && numel(unique(T.condition_group))==4,'A24SLX:Reporting','Expected16 averages in4 groups.');
 assert(all(isfinite(T.E_MPa)) && all(isfinite(T.Hardness_ShoreD)),'A24SLX:Inputs','Measurements must be finite.');
 pairs=[1,2;1,3;1,4];QT=T([],:);QT.evaluation_task=zeros(0,1);
 for task=1:3
  R=T(ismember(T.label,pairs(task,:)),:);R.evaluation_task=repmat(task,height(R),1);QT=[QT;R]; %#ok<AGROW>
 end
 QC=T;QC.evaluation_task=zeros(16,1);
 assert(height(QT)==24 && numel(unique(QT.row_id))==16,'A24SLX:Views','Expected24 TaskIL views.');
 D=readtable(fullfile(learningDir,'decisions_v1.csv'),'TextType','string');
 D=D(D.stage==3 & D.partition=="REPORTING_DEVELOPMENT",:);
 assert(height(D)==1040,'A24SLX:SavedPopulation','Expected1040 saved final reporting decisions.');
 assert(isequal(sort(unique(D.checkpoint_id)),sort(expectedIDs)),'A24SLX:SavedPopulation','Saved checkpoint set changed.');
 a24_build_inference_slx_v1(versionRoot);load_system(modelPath);
 modelCleanup=onCleanup(@()close_system(name,0)); %#ok<NASGU>
 ws=get_param(name,'ModelWorkspace');
 for ci=1:55
  cp=finalBank{ci};assert(cp.stage==3 && isequal(reshape(cp.seen_classes,1,[]),1:4), ...
   'A24SLX:SeenClasses','Expected final stage with four introduced classes.');
  model=canonicalModel(cp.model);[p,C,kind,scoreKind]=packCheckpoint(cp,model);
  assert(all(abs(reshape(cp.normalizer.mean,1,2)-mu)<=1e-12) && ...
   all(abs(reshape(cp.normalizer.sample_sd,1,2)-sd)<=1e-12),'A24SLX:Scale','Checkpoint normalizer changed.');
  if strcmp(cp.category,'TaskIncremental')
   Q=QT;mask=false(24,4);
   for j=1:24,mask(j,pairs(Q.evaluation_task(j),:))=true;end
   assert(kind==1,'A24SLX:Stream','No CE checkpoint belongs to TaskIL in this protocol.');
  else
   assert(strcmp(cp.category,'ClassIncremental'),'A24SLX:Stream','Unknown stream.');Q=QC;mask=true(16,4);
  end
  n=height(Q);U=[Q.E_MPa,Q.Hardness_ShoreD,double(mask)];tick=(0:n-1)';X=(U(:,1:2)-mu)./sd;
  saved=D(D.checkpoint_id==string(cp.id),:);
  queryKeys=string(Q.evaluation_task)+"|"+Q.row_id;savedKeys=string(saved.evaluation_task)+"|"+saved.row_id;
  assert(height(saved)==n && numel(unique(savedKeys))==n,'A24SLX:SavedKeys','Saved view keys must be unique.');
  [matched,loc]=ismember(queryKeys,savedKeys);assert(all(matched),'A24SLX:SavedKeys','Missing saved view.');saved=saved(loc,:);
  assert(isequal(double(saved.true_label),double(Q.label)) && all(saved.score_kind==string(scoreKind)) && ...
   isequal(double(saved.candidate_count),sum(mask,2)),'A24SLX:SavedMetadata','Saved labels/score kind/eligibility mismatch.');
  reference=kernelReference(model,X,C,kind,mask);
  core=zeros(n,18);for j=1:n,core(j,:)=a24_inference_numeric_v1(U(j,:),p,mu,sd,C,kind);end
  coreError=compareTrace(core,reference,tol);
  assignin(ws,'a24_inputs',timeseries(U,tick));assignin(ws,'a24_weights',p);
  assignin(ws,'a24_mean',mu);assignin(ws,'a24_scale',sd);assignin(ws,'a24_dictionary',C);assignin(ws,'a24_kind',kind);
  simOut=sim(name,'StopTime',num2str(n-1),'ReturnWorkspaceOutputs','on');
  raw=simOut.get('a24_trace');simTick=simOut.get('tout');
  [trace,layout]=normalizeTrace(raw,n);
  assert(isequal(simTick(:),tick),'A24SLX:Ticks','Simulink record ticks changed.');
  slxError=compareTrace(trace,reference,tol);coreSlxError=compareTrace(trace,core,tol);
  savedNumeric=[double(saved.score_1),double(saved.score_2),double(saved.score_3),double(saved.score_4),double(saved.raw_gap)];
  savedError=compareNumeric(trace(:,11:15),savedNumeric,tol);
  assert(isequal(trace(:,16),double(saved.forced_label)) && isequal(logical(trace(:,17)),boolColumn(saved.exact_tie)), ...
   'A24SLX:SavedDecisions','Simulink forced decisions/ties differ from saved CSV.');
  correct=trace(:,16)==double(Q.label);
  assert(isequal(correct,boolColumn(saved.correct)) && all(trace(:,18)==1),'A24SLX:SavedOutcomes','Saved correctness or execution status changed.');
  metrics=outcomeMetrics(trace,Q.label);expected=outcomeMetrics(reference,Q.label);
  assert(isequaln(metrics,expected),'A24SLX:Metrics','Full-cohort/confusion metrics changed.');
  maxError=max([coreError,slxError,coreSlxError,savedError],[],2);
  summary=struct('checkpoint_id',cp.id,'category',cp.category,'method',cp.method,'seed',cp.seed, ...
   'stage',3,'views',n,'unique_source_rows',numel(unique(Q.row_id)),'condition_groups',numel(unique(Q.condition_group)), ...
   'score_kind',scoreKind,'parameter_count',8+4+4*model.K+model.K,'raw_trace_shape',size(raw), ...
   'normalized_trace_layout',layout,'max_abs_numeric_error',max(maxError),'metrics',metrics);
  if isempty(summaries),summaries=summary;else,summaries(end+1)=summary;end %#ok<AGROW>
  for j=1:n
   e=struct('checkpoint_id',cp.id,'category',cp.category,'method',cp.method,'seed',cp.seed,'stage',3, ...
    'evaluation_task',Q.evaluation_task(j),'row_id',char(Q.row_id(j)),'condition_group',char(Q.condition_group(j)), ...
    'true_label',Q.label(j),'forced_label',trace(j,16),'correct',correct(j),'exact_tie',logical(trace(j,17)), ...
    'score_kind',scoreKind,'score_1',trace(j,11),'score_2',trace(j,12),'score_3',trace(j,13),'score_4',trace(j,14), ...
    'raw_gap',trace(j,15),'candidate_count',sum(mask(j,:)),'status',trace(j,18), ...
    'normalized_E',trace(j,1),'normalized_hardness',trace(j,2),'activation_1',trace(j,3), ...
    'activation_2',trace(j,4),'activation_3',trace(j,5),'activation_4',trace(j,6), ...
    'activation_5',trace(j,7),'activation_6',trace(j,8),'activation_7',trace(j,9),'activation_8',trace(j,10), ...
    'max_abs_numeric_error',maxError(j));
   if isempty(rows),rows=e;else,rows(end+1)=e;end %#ok<AGROW>
  end
  inferenceBank{ci}=struct('id',cp.id,'category',cp.category,'method',cp.method,'seed',cp.seed,'stage',3, ...
   'model',model,'weights52',p,'codebook4x8',C,'kind',kind,'score_kind',scoreKind);
  report.checkpoints_completed=ci;report.decision_views_completed=numel(rows);
  if ci==1
   example=struct('checkpoint_id',cp.id,'category',cp.category,'method',cp.method,'seed',cp.seed, ...
    'stage',3,'weights',p,'dictionary',C,'kind',kind,'inputs',U,'tick',tick);
   rawFirst=raw;save(fullfile(outputDir,'a24_first_raw_trace_v1.mat'),'rawFirst','tick','-v7');
  end
 end
 assert(numel(rows)==1040 && numel(summaries)==55,'A24SLX:Population','Final parity population changed.');
 assert(strcmp(example.checkpoint_id,'TaskIncremental_onehot_no_negotiation_seed101_stage3'), ...
  'A24SLX:Example','The declared first example must not be performance selected.');
 assignin(ws,'a24_inputs',timeseries(example.inputs,example.tick));assignin(ws,'a24_weights',example.weights);
 assignin(ws,'a24_mean',mu);assignin(ws,'a24_scale',sd);assignin(ws,'a24_dictionary',example.dictionary);assignin(ws,'a24_kind',example.kind);
 set_param(name,'StopTime','23','Description', ...
  'Frozen A24 TaskIL onehot/no-negotiation/seed101/stage3 example;24 reporting views of16 average records. Tick=record order, not physical time. Full55-checkpoint1040-view parity is separate. No online training, new physical validation or blind test.');
 save_system(name,modelPath);
 save(bankPath,'inferenceBank','normalizer','seeds','-v7.3');
 report.status='PASS';report.model_path=modelPath;report.final_stage_bank_path=bankPath;report.learning_directory=learningDir;
 report.demo_checkpoint=rmfield(example,{'weights','dictionary','inputs','tick'});report.numeric_tolerance=tol;
 report.checkpoints=summaries;report.elapsed_seconds=toc(started);
 report.population=struct('TaskIL_checkpoints',20,'TaskIL_views',480,'ClassIL_checkpoints',35,'ClassIL_views',560, ...
  'unique_reporting_averages',16,'unique_reporting_condition_groups',4);
 writetable(struct2table(rows),fullfile(outputDir,'a24_simulink_decisions_v1.csv'));
 writeJson(fullfile(outputDir,'a24_simulink_parity_result_v1.json'),report);
catch err
 report.status='FAILED';report.error_identifier=err.identifier;report.error_message=err.message;
 report.checkpoints=summaries;report.elapsed_seconds=toc(started);
 if ~isempty(rows),writetable(struct2table(rows),fullfile(outputDir,'a24_completed_decisions_partial_v1.csv'));end
 writeJson(fullfile(outputDir,'a24_simulink_parity_result_v1.json'),report);rethrow(err);
end
end

function model=canonicalModel(model)
assert(strcmp(model.arch,'mlp_tanh') && model.d==2 && model.h==4 && ismember(model.K,[4,8]), ...
 'A24SLX:Architecture','Expected code2-4-8 or CE2-4-4 tanh MLP.');
assert(isequal(size(model.W1),[2,4]) && isequal(size(model.W2),[4,model.K]) && ...
 numel(model.b1)==4 && numel(model.b2)==model.K,'A24SLX:Parameters','Invalid JSON weight dimensions.');
% JSON one-dimensional arrays decode as columns; restoring row biases is serialization-only.
model.b1=reshape(model.b1,1,4);model.b2=reshape(model.b2,1,model.K);
assert(all(isfinite([model.W1(:);model.b1(:);model.W2(:);model.b2(:)])), ...
 'A24SLX:Parameters','Nonfinite frozen weight.');
end

function [p,C,kind,scoreKind]=packCheckpoint(cp,model)
p=zeros(52,1);values=[model.W1(:);model.b1(:);model.W2(:);model.b2(:)];p(1:numel(values))=values;
if strcmp(cp.coder,'CE')
 assert(model.K==4 && numel(values)==32 && isempty(cp.codebook),'A24SLX:Codebook','Unexpected CE schema.');
 C=zeros(4,8);kind=2;scoreKind='seen_class_softmax';
else
 assert(model.K==8 && numel(values)==52 && ismember(string(cp.coder),["onehot","walsh"]), ...
  'A24SLX:Codebook','Unexpected code schema.');
 C=a24_codebook(cp.coder);assert(isequal(cp.codebook,C),'A24SLX:Codebook','Saved fixed dictionary changed.');
 kind=1;scoreKind='negative_mean_squared_code_distance';
end
end

function R=kernelReference(model,X,C,kind,mask)
n=size(X,1);activation=nan(n,8);
if kind==1
 activation=a24_sigmoid(r04nn_logits(model,X));
 [forced,distance,gap,tie]=a24_decode(activation,C,mask);scores=-distance;
else
 P=a24_ce_predict(model,X,mask);activation(:,1:4)=P;scores=P;scores(~mask)=NaN;
 masked=P;masked(~mask)=-Inf;[ordered,order]=sort(masked,2,'descend');
 forced=order(:,1);gap=ordered(:,1)-ordered(:,2);tie=ordered(:,1)==ordered(:,2);
end
R=[X,activation,scores,gap,forced,double(tie),ones(n,1)];
end

function [trace,layout]=normalizeTrace(raw,n)
assert(isnumeric(raw) && isreal(raw),'A24SLX:TraceShape','Trace must be a real numeric Array.');
if isequal(size(raw),[n,18])
 trace=raw;layout='ticks_by18';
elseif isequal(size(raw),[1,18,n])
 trace=reshape(permute(raw,[3,2,1]),n,18);layout='row_matrix_by_tick';
elseif isequal(size(raw),[18,1,n])
 trace=reshape(permute(raw,[3,1,2]),n,18);layout='column_matrix_by_tick';
else
 error('A24SLX:TraceShape','Expected N-by18,1-by18-byN or18-by1-byN; actual%s.',mat2str(size(raw)));
end
end

function errorPerRow=compareTrace(actual,expected,tol)
assert(isequal(size(actual),size(expected)) && size(actual,2)==18,'A24SLX:TraceShape','Trace width must be18.');
errorPerRow=compareNumeric(actual(:,1:15),expected(:,1:15),tol);
assert(isequal(actual(:,16:18),expected(:,16:18)),'A24SLX:DecisionParity','Forced/tie/status differ exactly.');
end

function errorPerRow=compareNumeric(actual,expected,tol)
assert(isequal(size(actual),size(expected)) && isequal(isnan(actual),isnan(expected)), ...
 'A24SLX:NumericParity','Numeric dimensions or inactive NaN mask changed.');
active=~isnan(expected);assert(all(isfinite(actual(active))) && all(isfinite(expected(active))), ...
 'A24SLX:NumericParity','Active outputs must be finite.');
delta=zeros(size(actual));delta(active)=abs(actual(active)-expected(active));
assert(all(delta(active)<=tol+tol*abs(expected(active))),'A24SLX:NumericParity','Tolerance exceeded.');
errorPerRow=max(delta,[],2);
end

function b=boolColumn(value)
if islogical(value),b=value(:);elseif isnumeric(value)
 assert(all(value==0 | value==1),'A24SLX:Boolean','Boolean values must be0/1.');b=logical(value(:));
else
 value=lower(string(value(:)));assert(all(ismember(value,["true","false","1","0"])), ...
  'A24SLX:Boolean','Unrecognized saved Boolean.');b=value=="true" | value=="1";
end
end

function m=outcomeMetrics(trace,truth)
truth=double(truth(:));forced=trace(:,16);status=trace(:,18);valid=status==1;
confusion=accumarray([truth(valid),forced(valid)],1,[4,4]);n=numel(truth);correct=sum(valid & forced==truth);
classCount=accumarray(truth,1,[4,1]);classCorrect=diag(confusion);present=classCount>0;
m=struct('views',n,'forced_correct',correct,'forced_wrong',sum(valid & forced~=truth), ...
 'failure',sum(~valid),'exact_ties',sum(trace(:,17)==1),'forced_accuracy',correct/n, ...
 'class_balanced_accuracy',mean(classCorrect(present)./classCount(present)), ...
 'confusion_truth_by_forced',confusion,'class_view_counts',classCount);
assert(m.forced_correct+m.forced_wrong+m.failure==n,'A24SLX:Counts','No failed or wrong view may be removed.');
end

function writeJson(file,value)
fid=fopen(file,'w','n','UTF-8');assert(fid~=-1,'A24SLX:Write','Output cannot be opened.');
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(value,'PrettyPrint',true,'ConvertInfAndNaN',true));
end
