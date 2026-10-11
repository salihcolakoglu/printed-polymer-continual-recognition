function run_a24_learning_v1
% New, explicitly defined exploratory A extension. Original A is not rerun.
here=fileparts(mfilename('fullpath')); root=fileparts(here);
addpath(here,fullfile(here,'utility'));
out=fullfile(root,'07_Sonuclar','learning_run_v1');
assert(~isfolder(out),'A24:NoOverwrite','Output exists; inspect, never silently rerun.');
accept=jsondecode(fileread(fullfile(root,'08_Inceleme','A24_launch_acceptance_v1.json')));
assert(strcmp(accept.status,'PASS_FOR_ONE_A24_LEARNING_RUN'),'A24:Review','Different technical acceptance required.');
protocol=jsondecode(fileread(fullfile(root,'06_Protokoller','A24_prospective_protocol_v3.json')));
assert(protocol.training.updates_per_stage==600);
assert(isequal(protocol.training.seeds(:)',[101 211 307 401 503]));
T=readtable(fullfile(root,'02_Veriler','A_source_average_records_v1.csv'),'TextType','string');
assert(height(T)==104 && numel(unique(T.row_id))==104 && numel(unique(T.condition_group))==26);
Y=double(T.label); Xraw=[T.E_MPa T.Hardness_ShoreD];
assert(all(isfinite(Xraw),'all') && all(ismember(Y,1:4)));
train=strcmp(T.partition,'TRAIN'); first=find(train & ismember(Y,[1 2]));
assert(numel(first)==36); mu=mean(Xraw(first,:),1); sd=std(Xraw(first,:),0,1);
constant=sd==0;sd(constant)=1;X=(Xraw-mu)./sd;
normalizer=struct('mean',mu,'sample_sd',sd,'constant_channels',constant, ...
 'fit_row_ids',T.row_id(first),'fit_population','First-stage TRAIN36; fixed in both streams.');
opt=struct('eta',.02,'beta1',.9,'beta2',.999,'epsilon',1e-8);
lambda=1e-4;steps=600;seeds=[101 211 307 401 503];alphas=[.5 2/3 .75];
codeMethods={'onehot_no_negotiation','onehot_negotiated','walsh_no_negotiation','walsh_negotiated'};
ceMethods={'sequential_no_replay','sequential_replay','joint_seen_reference'};
streams={'TaskIncremental','ClassIncremental'};
bank={};ledger=struct([]);decisions=struct([]); mkdir(out);
started=char(datetime('now','TimeZone','UTC'));wall=tic;
for ci=1:2
 category=streams{ci}; methods=codeMethods;
 if ci==2,methods=[methods ceMethods];end
 for mi=1:numel(methods)
  method=methods{mi};isCode=mi<=4;negotiated=ismember(mi,[2 4]);
  if isCode
   D=8;if mi<=2,coder='onehot';else,coder='walsh';end
   C=a24_codebook(coder);
  else,D=4;coder='CE';C=[];end
  for seed=seeds
   initial=r04nn_init('mlp_tanh',2,D,4,seed);model=initial;state=[];
   for stage=1:3
    assert(toc(wall)<900,'A24:TimeLimit','Frozen run wall limit exceeded.');
    if ci==1,pairs=[1 2;1 3;1 4];currentClasses=pairs(stage,:);else
     if stage==1,currentClasses=[1 2];else,currentClasses=stage+1;end
    end
    seen=1:(stage+1);current=find(train & ismember(Y,currentClasses));
    ids=current;w=ones(numel(ids),1);memory=[];
    if strcmp(method,'sequential_replay') && stage>1
     for old=1:stage
      eligible=find(train & Y==old);[~,order]=sort(T.group_order(eligible));
      memory=[memory;eligible(order(1:2))]; %#ok<AGROW>
     end
     assert(numel(unique(memory))==numel(memory));
     ids=[current;memory];w=[ones(numel(current),1)/numel(current);.5*ones(numel(memory),1)/numel(memory)];
    elseif strcmp(method,'joint_seen_reference')
     model=initial;state=[];ids=find(train & ismember(Y,seen));w=ones(numel(ids),1);
    end
    preTask=model; teacher=[];targets=[];alpha=0;teacherExamples=0;
    if isCode
     assert(isequal(ids,current));targets=C(Y(ids),:);
     if negotiated
      alpha=alphas(stage);teacher=a24_sigmoid(r04nn_logits(preTask,X(current,:)));
      targets=(1-alpha)*targets+alpha*teacher;teacherExamples=numel(current);
     end
     [lossBefore,~]=a24_code_lossgrad(model,X(ids,:),targets,w,lambda);
    else
     mask=false(numel(ids),4);mask(:,seen)=true;
     [lossBefore,~]=a24_ce_lossgrad(model,X(ids,:),Y(ids),mask,w,lambda);
    end
    for update=1:steps
     if isCode,[~,g]=a24_code_lossgrad(model,X(ids,:),targets,w,lambda);
     else,[~,g]=a24_ce_lossgrad(model,X(ids,:),Y(ids),mask,w,lambda);end
     [model,state]=r04nn_adam(model,g,state,opt);
    end
    if isCode,[lossAfter,~]=a24_code_lossgrad(model,X(ids,:),targets,w,lambda);
    else,[lossAfter,~]=a24_ce_lossgrad(model,X(ids,:),Y(ids),mask,w,lambda);end
    id=sprintf('%s_%s_seed%d_stage%d',category,method,seed,stage);
    cp=struct('id',id,'category',category,'method',method,'coder',coder,'seed',seed,'stage',stage, ...
     'model',model,'adam_state',state,'pre_task_model',preTask,'codebook',C, ...
     'alpha',alpha,'frozen_teacher',teacher,'frozen_targets',targets, ...
     'current_row_ids',T.row_id(current),'training_row_ids',T.row_id(ids),'memory_row_ids',T.row_id(memory), ...
     'training_weights',w/sum(w),'seen_classes',seen,'normalizer',normalizer);
    bank{end+1}=cp; %#ok<AGROW>
    L=struct('id',id,'category',category,'method',method,'seed',seed,'stage',stage, ...
     'alpha',alpha,'updates',steps,'loss_before',lossBefore,'loss_after',lossAfter, ...
     'parameter_count',numel(r04nn_pack(model)),'parameter_bytes',8*numel(r04nn_pack(model)), ...
     'optimizer_m_v_bytes',16*numel(r04nn_pack(model)),'codebook_bytes',8*numel(C), ...
     'optimizer_t',state.t,'current_records',numel(current),'memory_records',numel(memory), ...
     'training_records_per_update',numel(ids),'training_unique_records',numel(unique(ids)), ...
     'training_groups',numel(unique(T.condition_group(ids))), ...
     'update_gradient_examples',steps*numel(ids),'diagnostic_gradient_examples',2*numel(ids), ...
     'teacher_forward_examples',teacherExamples,'elapsed_run_seconds',toc(wall));
    if isempty(ledger),ledger=L;else,ledger(end+1)=L;end %#ok<AGROW>
    if ci==1
     for task=1:3
      rows=find(ismember(Y,pairs(task,:)));
      E=evaluate(cp,T,X,Y,rows,pairs(task,:),task,isCode);
      if isempty(decisions),decisions=E;else,decisions=[decisions E];end %#ok<AGROW>
     end
    else
     rows=find(ismember(Y,seen));E=evaluate(cp,T,X,Y,rows,seen,0,isCode);
     if isempty(decisions),decisions=E;else,decisions=[decisions E];end %#ok<AGROW>
    end
    save(fullfile(out,'partial_models_v1.mat'),'bank','ledger','normalizer','protocol','-v7.3');
    fprintf('A24 stage %d/165 complete: %s\n',numel(ledger),id);
   end
  end
 end
end
assert(numel(ledger)==165 && sum([ledger.updates])==99000);
assert(numel(decisions)==17550,'A24:Population','Prospective decision population changed.');
writetable(struct2table(decisions),fullfile(out,'decisions_v1.csv'));
writetable(struct2table(ledger),fullfile(out,'training_ledger_v1.csv'));
writeJson(fullfile(out,'checkpoints_v1.json'),bank);
writeJson(fullfile(out,'normalizer_v1.json'),normalizer);
save(fullfile(out,'models_and_optimizer_states_v1.mat'),'bank','ledger','normalizer','protocol','-v7.3');
receipt=struct('status','COMPLETED_PRODUCER_PENDING_DIFFERENT_SAVED_REVIEW', ...
 'started_at_utc',started,'finished_at_utc',char(datetime('now','TimeZone','UTC')), ...
 'elapsed_seconds',toc(wall),'MATLAB_version',version,'stage_entries',numel(ledger), ...
 'updates',sum([ledger.updates]),'decisions',numel(decisions),'source_average_records',104, ...
 'new_blind_test',false,'physical_validation',false,'new_Simulink',false);
writeJson(fullfile(out,'execution_receipt_v1.json'),receipt);disp(jsonencode(receipt));
end

function E=evaluate(cp,T,X,Y,rows,candidates,task,isCode)
if isCode
 P=a24_sigmoid(r04nn_logits(cp.model,X(rows,:)));
 [forced,distance,gap,tie]=a24_decode(P,cp.codebook,candidates);
 score=-distance;kind='negative_mean_squared_code_distance';
else
 mask=false(numel(rows),4);mask(:,candidates)=true;
 score=a24_ce_predict(cp.model,X(rows,:),mask);
 [ss,ord]=sort(score(:,candidates),2,'descend');forced=candidates(ord(:,1));forced=forced(:);
 gap=ss(:,1)-ss(:,2);tie=ss(:,1)==ss(:,2);kind='seen_class_softmax';
end
E=struct([]);
for j=1:numel(rows)
 y=Y(rows(j));introduced=max(1,y-1);
 if task>0
  unit=task;if task>cp.stage,role='future_task_diagnostic';elseif task==cp.stage,role='current_task';else,role='old_task';end
 else
  unit=introduced;if introduced==cp.stage,role='new_class';else,role='old_class';end
 end
 scores=score(j,:);scores(~ismember(1:4,candidates))=NaN;
 e=struct('checkpoint_id',cp.id,'category',cp.category,'method',cp.method,'seed',cp.seed,'stage',cp.stage, ...
  'evaluation_task',task,'acquisition_unit',unit,'role',role,'partition',char(T.partition(rows(j))), ...
  'row_id',char(T.row_id(rows(j))),'condition_group',char(T.condition_group(rows(j))), ...
  'true_label',y,'forced_label',forced(j),'correct',forced(j)==y,'exact_tie',tie(j), ...
  'score_kind',kind,'score_1',scores(1),'score_2',scores(2),'score_3',scores(3),'score_4',scores(4), ...
  'raw_gap',gap(j),'candidate_count',numel(candidates));
 if isempty(E),E=e;else,E(end+1)=e;end %#ok<AGROW>
end
end

function writeJson(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid~=-1,'A24:Write','Output cannot be opened.');
c=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(value,'PrettyPrint',true,'ConvertInfAndNaN',true));
end
