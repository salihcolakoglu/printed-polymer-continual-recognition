function run_r04_source_experiment_v2
% One frozen exploratory source-record package. No tuning or random physics.
here=fileparts(mfilename('fullpath'));
addpath(fullfile(here,'matlab_utility'));
out=fullfile(here,'source_results_v2');
assert(~isfolder(out),'R04:NoOverwrite','Result directory already exists. Do not silently rerun.');
reviewPath=fullfile(here,'review','launch_acceptance_v2.json');
assert(isfile(reviewPath),'R04:Review','Independent launch acceptance required.');
review=jsondecode(fileread(reviewPath));
assert(strcmp(review.status,'PASS_FOR_ONE_FROZEN_TECHNICAL_REPAIR_SOURCE_RUN'),'R04:Review','Launch not accepted.');
mkdir(out);
started=datetime('now','TimeZone','UTC'); wall=tic;
protocol=jsondecode(fileread(fullfile(here,'prospective_protocol_v1.json')));
T=readtable(fullfile(here,'staged_source_records_v1.csv'),'TextType','string');
assert(height(T)==104 && numel(unique(T.row_id))==104);
assert(numel(unique(T.condition_group))==26);
assert(all(isfinite(T.E_MPa)) && all(isfinite(T.Hardness_ShoreD)));
Xraw=[T.E_MPa,T.Hardness_ShoreD]; Y=double(T.label);
pairs=[1,2;1,3;1,4]; K=4;
train=strcmp(T.partition,'TRAIN');
firstRows=find(train & ismember(Y,pairs(1,:)));
assert(numel(firstRows)==36);
mu=mean(Xraw(firstRows,:),1); sd=std(Xraw(firstRows,:),0,1);
constantChannels=sd==0; sd(constantChannels)=1;
X=(Xraw-mu)./sd;
normalizer=struct('mean',mu,'sample_sd',sd,'constant_channels',constantChannels, ...
 'source_row_ids',T.row_id(firstRows),'source_condition_groups',unique(T.condition_group(firstRows)), ...
 'fit_population','T1 TRAIN36 only; same scale for all stages/architectures/arms');
writeJson(fullfile(out,'normalizer_v1.json'),normalizer);
opt=struct('eta',.02,'beta1',.9,'beta2',.999,'epsilon',1e-8);
lambda=1e-4; steps=600; seeds=[101,211,307,401,503];
assert(protocol.training.full_batch_steps_per_stage==steps);
assert(isequal(protocol.training.seeds(:)',seeds));
assert(protocol.decoder.margin_threshold==.15);
architectures={'linear','mlp_tanh'}; paramCounts=[12,32];
arms={'sequential_no_replay','sequential_replay','joint_seen_reference'};
decisions=struct([]); ledger=struct([]); summaries=struct([]); checkpoints=cell(2,5,3,3);
for ai=1:2
 for si=1:5
  arch=architectures{ai}; seed=seeds(si); h=4*(ai==2);
  initial=r04nn_init(arch,2,K,h,seed);
  assert(numel(r04nn_pack(initial))==paramCounts(ai));
  for armIndex=1:3
   arm=arms{armIndex}; model=initial; state=[]; previousFailure=false;
   for stage=1:3
    assert(toc(wall)<1200,'R04:TimeLimit','Frozen package wall time limit exceeded.');
    current=find(train & ismember(Y,pairs(stage,:)));
    ids=current; taskIds=repmat(stage,numel(current),1); w=ones(numel(current),1)/numel(current);
    if strcmp(arm,'sequential_replay') && stage>1
     memory=[]; memoryTasks=[];
     for oldTask=1:stage-1
      for lab=pairs(oldTask,:)
       eligible=find(train & Y==lab);
       [~,order]=sort(T.group_order(eligible)); eligible=eligible(order);
       memory=[memory;eligible(1:2)]; %#ok<AGROW>
       memoryTasks=[memoryTasks;repmat(oldTask,2,1)]; %#ok<AGROW>
      end
     end
     ids=[current;memory]; taskIds=[taskIds;memoryTasks];
     w=[ones(numel(current),1)/numel(current);.5*ones(numel(memory),1)/numel(memory)];
    elseif strcmp(arm,'joint_seen_reference')
     model=initial; state=[]; ids=[]; taskIds=[];
     for knownTask=1:stage
      r=find(train & ismember(Y,pairs(knownTask,:)));
      ids=[ids;r];taskIds=[taskIds;repmat(knownTask,numel(r),1)]; %#ok<AGROW>
     end
     w=ones(numel(ids),1)/numel(ids);
    end
    mask=false(numel(ids),K);
    for t=1:stage, mask(taskIds==t,pairs(t,:))=true; end
    success=~previousFailure; errorText=''; lossBefore=NaN; lossAfter=NaN; updates=0;
    completedGradientCalls=0; diagnosticGradientCalls=0;
    if success
     try
      [lossBefore,~]=r04nn_lossgrad(model,X(ids,:),Y(ids),mask,w,lambda);
      completedGradientCalls=completedGradientCalls+1; diagnosticGradientCalls=diagnosticGradientCalls+1;
      for update=1:steps
       [~,g]=r04nn_lossgrad(model,X(ids,:),Y(ids),mask,w,lambda);
       completedGradientCalls=completedGradientCalls+1;
       [model,state]=r04nn_adam(model,g,state,opt); updates=update;
      end
      [lossAfter,~]=r04nn_lossgrad(model,X(ids,:),Y(ids),mask,w,lambda);
      completedGradientCalls=completedGradientCalls+1; diagnosticGradientCalls=diagnosticGradientCalls+1;
     catch ex
      success=false; previousFailure=true; errorText=getReport(ex,'extended','hyperlinks','off');
     end
    else
     errorText='Not executed after prior stage training failure; all planned query views retained as failures.';
    end
    checkpoint=struct('model',model,'adam_state',state,'stage',stage,'architecture',arch,'arm',arm, ...
      'seed',seed,'train_row_indices',ids,'train_task_ids',taskIds,'row_masks',mask,'input_weights',w,'success',success);
    checkpoints{ai,si,armIndex,stage}=checkpoint;
    L=struct('architecture',arch,'arm',arm,'seed',seed,'stage',stage,'success',success, ...
      'error',errorText,'loss_before',lossBefore,'loss_after',lossAfter,'updates_executed',updates, ...
      'optimizer_update_gradient_examples',updates*numel(ids), ...
      'diagnostic_gradient_examples',diagnosticGradientCalls*numel(ids), ...
      'total_completed_gradient_examples',completedGradientCalls*numel(ids), ...
      'training_task_views_per_update',numel(ids), ...
      'unique_training_source_rows',numel(unique(T.row_id(ids))), ...
      'unique_training_condition_groups',numel(unique(T.condition_group(ids))), ...
      'parameter_count',numel(r04nn_pack(model)), 'adam_step',adamStep(state));
    if isempty(ledger),ledger=L;else,ledger(end+1)=L;end %#ok<AGROW>
    writeJson(fullfile(out,'partial_training_ledger_v1.json'),ledger);
    preservePartial(out,checkpoints,normalizer,protocol,ledger,decisions,summaries);
    for task=1:3
     partNames={'TRAIN','FIT_DEVELOPMENT','REPORTING_DEVELOPMENT'};
     for pi=1:3
      r=find(strcmp(T.partition,partNames{pi}) & ismember(Y,pairs(task,:)));
      queryMask=false(numel(r),K);queryMask(:,pairs(task,:))=true;
      if success
       P=r04nn_predict(model,X(r,:),queryMask); reason='';
      else
       P=nan(numel(r),K);reason='training_failure';
      end
      D=makeDecisions(T,Y,r,P,pairs(task,:),arch,arm,seed,stage,task,partNames{pi},reason);
      if isempty(decisions),decisions=D;else,decisions=[decisions,D];end %#ok<AGROW>
      S=summarize(D);if isempty(summaries),summaries=S;else,summaries(end+1)=S;end %#ok<AGROW>
     end
    end
    writeJson(fullfile(out,'partial_training_ledger_v1.json'),ledger);
    preservePartial(out,checkpoints,normalizer,protocol,ledger,decisions,summaries);
   end
  end
 end
end
% A single deterministic nearest baseline; no optimization seed replication.
for stage=1:3
 pool=find(train & ismember(Y,unique(pairs(1:stage,:))));
 for task=1:3
  partNames={'TRAIN','FIT_DEVELOPMENT','REPORTING_DEVELOPMENT'};
  for pi=1:3
   r=find(strcmp(T.partition,partNames{pi}) & ismember(Y,pairs(task,:)));
   if all(ismember(pairs(task,:),Y(pool)))
    distances=zeros(numel(r),2);
    for c=1:2
     refs=pool(Y(pool)==pairs(task,c));
     delta=reshape(X(r,:),numel(r),1,2)-reshape(X(refs,:),1,numel(refs),2);
     distances(:,c)=min(sum(delta.^2,3),[],2);
    end
    score=-distances;score=score-max(score,[],2);p2=exp(score);p2=p2./sum(p2,2);
    P=zeros(numel(r),K);P(:,pairs(task,:))=p2;reason='';
   else
    P=nan(numel(r),K);reason='unseen_candidate';
   end
   D=makeDecisions(T,Y,r,P,pairs(task,:),'nearest','seen_reference',0,stage,task,partNames{pi},reason);
   decisions=[decisions,D];S=summarize(D);if isempty(summaries),summaries=S;else,summaries(end+1)=S;end %#ok<AGROW>
  end
 end
 preservePartial(out,checkpoints,normalizer,protocol,ledger,decisions,summaries);
end
assert(numel(ledger)==90,'R04:Ledger','All 90 planned training-stage entries must be retained.');
assert(numel(decisions)==14508,'R04:Population','Whole frozen decision population changed.');
assert(numel(summaries)==837,'R04:Summary','Expected all architecture/arm/seed/stage/task/partition summaries.');
writeJson(fullfile(out,'training_ledger_v1.json'),ledger);
writeJson(fullfile(out,'decisions_v1.json'),decisions);
writeJson(fullfile(out,'task_stage_summaries_v1.json'),summaries);
writetable(struct2table(decisions),fullfile(out,'decision_rows_v1.csv'));
save(fullfile(out,'models_and_optimizer_states_v1.mat'),'checkpoints','normalizer','protocol','ledger','-v7.3');
receipt=struct('status','COMPLETED_FROZEN_EXPLORATORY_SOURCE_PACKAGE_NOT_PHYSICAL_VALIDATION', ...
 'started_at_utc',char(started),'finished_at_utc',char(datetime('now','TimeZone','UTC')), ...
 'elapsed_seconds',toc(wall),'matlab_version',version,'matlab_release',version('-release'), ...
 'source_records',104,'source_condition_groups',26,'training_stage_entries',numel(ledger), ...
 'successful_training_stages',sum([ledger.success]),'failed_training_stages',sum(~[ledger.success]), ...
 'decision_views',numel(decisions),'summary_rows',numel(summaries), ...
 'new_blind_test',false,'physical_validation',false,'model_assisted_CL',false);
writeJson(fullfile(out,'execution_receipt_v1.json'),receipt);disp(jsonencode(receipt));
end

function n=adamStep(state)
if isempty(state),n=0;else,n=state.t;end
end

function D=makeDecisions(T,Y,r,P,pair,arch,arm,seed,stage,task,partition,reason)
D=struct([]);
for j=1:numel(r)
 scores=P(j,pair); forced=0; defined=false; margin=NaN; selected=0;
 if strcmp(reason,'unseen_candidate')
  outcome='nonselection';
 elseif any(~isfinite(scores))
  outcome='failure';
 else
  [~,winner]=max(scores); forced=pair(winner);defined=true;margin=abs(diff(scores));
  if margin<.15 || scores(1)==scores(2)
   outcome='nonselection';
  else
   selected=forced;
   if selected==Y(r(j)),outcome='selected_correct';else,outcome='selected_wrong';end
  end
 end
 d=struct('architecture',arch,'arm',arm,'seed',seed,'stage',stage,'task',task, ...
  'partition',partition,'source_row_id',char(T.row_id(r(j))), ...
  'condition_group',char(T.condition_group(r(j))),'label',Y(r(j)), ...
  'candidate_a',pair(1),'candidate_b',pair(2),'score_a',scores(1),'score_b',scores(2), ...
  'margin',margin,'selected_label',selected,'forced_argmax',forced, ...
  'forced_argmax_defined',defined,'forced_argmax_correct',defined && forced==Y(r(j)), ...
  'outcome',outcome,'reason',reason,'task_role',taskRole(stage,task));
 if isempty(D),D=d;else,D(end+1)=d;end %#ok<AGROW>
end
end

function role=taskRole(stage,task)
if task>stage,role='future_task_diagnostic';elseif task==stage,role='new_task';else,role='old_task';end
end

function S=summarize(D)
d=D(1);o={D.outcome};N=numel(D);correct=sum(strcmp(o,'selected_correct'));wrong=sum(strcmp(o,'selected_wrong'));
S=struct('architecture',d.architecture,'arm',d.arm,'seed',d.seed,'stage',d.stage,'task',d.task, ...
 'partition',d.partition,'task_role',d.task_role,'decision_views',N, ...
 'unique_original_rows',numel(unique({D.source_row_id})),'unique_condition_groups',numel(unique({D.condition_group})), ...
 'selected_correct',correct,'selected_wrong',wrong,'nonselection',sum(strcmp(o,'nonselection')), ...
 'failure',sum(strcmp(o,'failure')),'forced_argmax_correct',sum([D.forced_argmax_correct]), ...
 'forced_argmax_defined',sum([D.forced_argmax_defined]),'selected_correct_fraction',correct/N, ...
 'coverage',(correct+wrong)/N,'selective_error',wrong/max(1,correct+wrong), ...
 'selective_error_defined',correct+wrong>0);
assert(S.selected_correct+S.selected_wrong+S.nonselection+S.failure==N);
end

function writeJson(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid~=-1,'R04:Write','Could not open output');
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(value,'PrettyPrint',true,'ConvertInfAndNaN',true));
end

function preservePartial(out,checkpoints,normalizer,protocol,ledger,decisions,summaries)
% Commit after every completed training/nearest stage. Preserve last valid
% checkpoint if a later save is interrupted; do not rely on end-of-run RAM.
temp=fullfile(out,'partial_package_pending_v1.mat');
target=fullfile(out,'partial_package_v1.mat');
save(temp,'checkpoints','normalizer','protocol','ledger','decisions','summaries','-v7.3');
[ok,message]=movefile(temp,target,'f');
assert(ok,'R04:PartialCommit','Partial checkpoint commit failed: %s',message);
end
