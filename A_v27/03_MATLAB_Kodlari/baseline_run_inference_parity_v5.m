function report = baseline_run_inference_parity_v5(originalPublicRoot,outputDir)
% One actual future producer: no training, no refit, no B sources.
% Original root is the complete A_v23_Public_Release_v2 folder.
% outputDir must be new. Root launcher owns and bounds this MATLAB process.
versionRoot=fileparts(fileparts(mfilename('fullpath')));
assert(~isfolder(outputDir) && ~isfile(outputDir),'Baseline:ExistingOutput','Output must be new.');
modelName='baseline_A_inference_v5';
modelPath=fullfile(versionRoot,'04_Simulink_Modelleri',[modelName,'.slx']);
bankPath=fullfile(versionRoot,'05_Ogrenilmis_Modeller','baseline_A_final_stage_models_v1.mat');
assert(~isfile(modelPath) && ~isfile(bankPath) && ~bdIsLoaded(modelName), ...
 'Baseline:ExistingArtifacts','Preserve previous artifacts; do not silently repeat.');
mkdir(outputDir); started=tic; oldPath=path;
pathCleanup=onCleanup(@()path(oldPath)); %#ok<NASGU>
addpath(fullfile(versionRoot,'03_MATLAB_Kodlari'));
old=fullfile(originalPublicRoot,'03_Hesaplama_Eki','recorded_package_v2');
addpath(fullfile(old,'matlab_utility'));
report=struct('schema','A_BASELINE_SIMULINK_PARITY_RESULT_V1','status','RUNNING', ...
 'checkpoints_completed',0,'decision_views_completed',0,'new_training',0, ...
 'new_physical_validation',0,'record_ticks_are_physical_time',false, ...
 'source_prior_pins_are_fresh_checks',false);
rows=struct([]); checkpoints=struct([]);
try
 protocol=jsondecode(fileread(fullfile(versionRoot,'06_Protokoller','baseline_simulink_parity_protocol_v1.json')));
 assert(protocol.expected_checkpoints==30 && protocol.expected_decision_views==720 && protocol.stage==3, ...
  'Baseline:Protocol','Unexpected baseline scope.');
 tol=protocol.numeric_tolerance;
 T=readtable(fullfile(versionRoot,'02_Veriler','baseline_A_source_records_v23_0.csv'),'TextType','string');
 assert(height(T)==104 && numel(unique(T.row_id))==104,'Baseline:Source','Expected104 unique published average records.');
 T=T(T.partition=="REPORTING_DEVELOPMENT",:);
 assert(height(T)==16 && numel(unique(T.condition_group))==4,'Baseline:Reporting','Expected16 averages in4 source condition groups.');
 assert(all(isfinite(T.E_MPa)) && all(isfinite(T.Hardness_ShoreD)),'Baseline:Source','Inputs must be finite.');
 pairs=[1,2;1,3;1,4]; Q=T([],:); Q.task=zeros(0,1);
 for task=1:3
  R=T(ismember(T.label,pairs(task,:)),:); R.task=repmat(task,height(R),1); Q=[Q;R]; %#ok<AGROW>
 end
 assert(height(Q)==24 && numel(unique(Q.row_id))==16,'Baseline:Views','Expected24 views of16 averages.');
 U=[Q.E_MPa,Q.Hardness_ShoreD,pairs(Q.task,:)]; tick=(0:23)';
 mask=false(24,4);for j=1:24,mask(j,pairs(Q.task(j),:))=true;end
 % Exactly one semantic MAT read. No large-payload reread/hash in this driver.
 B=load(fullfile(old,'source_results_v2','models_and_optimizer_states_v1.mat'),'checkpoints','normalizer');
 assert(isequal(size(B.checkpoints),[2,5,3,3]),'Baseline:Bank','Checkpoint schema changed.');
 mu=reshape(B.normalizer.mean,1,2);sd=reshape(B.normalizer.sample_sd,1,2);
 assert(all(isfinite(mu)) && all(isfinite(sd)) && all(sd>0),'Baseline:Scale','Invalid frozen scale.');
 expectedMu=reshape(protocol.frozen_normalizer.mean,1,2);expectedSd=reshape(protocol.frozen_normalizer.sample_sd,1,2);
 assert(all(abs(mu-expectedMu)<=1e-12) && all(abs(sd-expectedSd)<=1e-12),'Baseline:Scale','Frozen TRAIN36 scale changed.');
 X=([Q.E_MPa,Q.Hardness_ShoreD]-mu)./sd;
 D=readtable(fullfile(old,'source_results_v2','decision_rows_v1.csv'),'TextType','string');
 D=D(D.stage==3 & D.partition=="REPORTING_DEVELOPMENT" & ismember(D.architecture,["linear","mlp_tanh"]),:);
 assert(height(D)==720,'Baseline:SavedPopulation','Expected720 archived learned decisions.');
 baseline_build_inference_slx_v5(versionRoot);load_system(modelPath);
 modelCleanup=onCleanup(@()close_system(modelName,0)); %#ok<NASGU>
 ws=get_param(modelName,'ModelWorkspace');
 archs={'linear','mlp_tanh'};arms={'sequential_no_replay','sequential_replay','joint_seen_reference'};
 seeds=[101,211,307,401,503]; bank=cell(2,5,3); example=struct();
 for ai=1:2
  for si=1:5
   for armIndex=1:3
    cp=B.checkpoints{ai,si,armIndex,3};
    assert(cp.success && cp.stage==3 && strcmp(cp.architecture,archs{ai}) && ...
     strcmp(cp.arm,arms{armIndex}) && cp.seed==seeds(si),'Baseline:Checkpoint','Checkpoint identity/success mismatch.');
    assert(cp.model.d==2 && cp.model.K==4,'Baseline:Dimensions','Unexpected input/output width.');
    if ai==2,assert(cp.model.h==4,'Baseline:Dimensions','Expected4 hidden units.');end
    p=packParameters(cp.model,ai);bank{ai,si,armIndex}=cp.model;
    saved=D(D.architecture==string(archs{ai}) & D.arm==string(arms{armIndex}) & D.seed==seeds(si),:);
    queryKeys=string(Q.task)+"|"+Q.row_id;savedKeys=string(saved.task)+"|"+saved.source_row_id;
    assert(height(saved)==24 && numel(unique(savedKeys))==24,'Baseline:SavedKeys','Archived checkpoint keys must be unique24.');
    [matched,loc]=ismember(queryKeys,savedKeys);assert(all(matched),'Baseline:SavedKeys','Missing archived query view.');
    saved=saved(loc,:);
    assert(isequal(double(saved.label),double(Q.label)) && isequal(double(saved.candidate_a),U(:,3)) && ...
     isequal(double(saved.candidate_b),U(:,4)),'Baseline:SavedIdentity','Labels/pairs mismatch.');
    [P,forced,Z]=r04nn_predict(cp.model,X,mask);
    margin=abs(P(:,1)-P(sub2ind(size(P),(1:24)',U(:,4))));
    selected=forced;selected(margin<0.15 | margin==0)=0;status=double(selected~=0);
    reference=[X,Z,P,margin,forced,selected,status];
    assignin(ws,'baseline_inputs',timeseries(U,tick));assignin(ws,'baseline_p',p);
    assignin(ws,'baseline_mu',mu);assignin(ws,'baseline_sd',sd);assignin(ws,'baseline_arch',ai);
    simOut=sim(modelName,'StopTime','23','ReturnWorkspaceOutputs','on');
    trace=simOut.get('baseline_trace');simTick=simOut.get('tout');
    rawSize=size(trace);
    writeJson(fullfile(outputDir,'last_raw_trace_shape_v1.json'),struct('trace_size',rawSize,'time_size',size(simTick)));
    if isequal(rawSize,[14,1,24]) || isequal(rawSize,[1,14,24]),trace=squeeze(trace)';end
    assert(isnumeric(trace) && isequal(size(trace),[24,14]) && isequal(simTick(:),tick), ...
     'Baseline:TraceShape','Expected24 ordered ticks by14 numeric fields.');
    assert(all(isfinite(trace),'all'),'Baseline:Nonfinite','Finite archived inputs must give finite outputs.');
    delta=abs(trace-reference);numericOK=delta(:,1:11)<=tol+tol*abs(reference(:,1:11));
    assert(all(numericOK,'all') && isequal(trace(:,12:14),reference(:,12:14)), ...
     'Baseline:NumericParity','Simulink differs from independent legacy MATLAB reference.');
    actual=outcomes(trace(:,13),trace(:,14),Q.label);
    scoreB=trace(sub2ind(size(trace),(1:24)',6+U(:,4)));
    assert(all(abs(trace(:,7)-saved.score_a)<=tol+tol*abs(saved.score_a)) && ...
     all(abs(scoreB-saved.score_b)<=tol+tol*abs(saved.score_b)) && ...
     all(abs(trace(:,11)-saved.margin)<=tol+tol*abs(saved.margin)), ...
     'Baseline:SavedScores','Simulink scores differ from archived CSV.');
    assert(isequal(trace(:,12),double(saved.forced_argmax)) && all(logical(saved.forced_argmax_defined)) && ...
     isequal(trace(:,13),double(saved.selected_label)) && isequal(actual,string(saved.outcome)), ...
     'Baseline:SavedDecisions','Simulink decisions differ from archived CSV.');
    measured=metrics(actual);expected=metrics(string(saved.outcome));
    assert(isequaln(measured,expected),'Baseline:Metrics','Aggregated outcome metrics differ.');
    r=struct('architecture',archs{ai},'arm',arms{armIndex},'seed',seeds(si),'stage',3, ...
     'views',24,'unique_source_rows',16,'unique_condition_groups',4,'max_abs_numeric_error',max(delta,[],'all'),'metrics',measured);
    if isempty(checkpoints),checkpoints=r;else,checkpoints(end+1)=r;end %#ok<AGROW>
    for j=1:24
     row=struct('architecture',archs{ai},'arm',arms{armIndex},'seed',seeds(si),'stage',3, ...
      'task',Q.task(j),'source_row_id',char(Q.row_id(j)),'condition_group',char(Q.condition_group(j)), ...
      'label',Q.label(j),'candidate_a',U(j,3),'candidate_b',U(j,4),'score_a',trace(j,7),'score_b',scoreB(j), ...
      'margin',trace(j,11),'forced_argmax',trace(j,12),'selected_label',trace(j,13), ...
      'outcome',char(actual(j)),'max_abs_numeric_error',max(delta(j,:)));
     if isempty(rows),rows=row;else,rows(end+1)=row;end %#ok<AGROW>
    end
    report.checkpoints_completed=numel(checkpoints);report.decision_views_completed=numel(rows);
    if report.checkpoints_completed==1,example=struct('parameters',p,'architecture',ai,'seed',seeds(si),'arm',arms{armIndex});end
   end
  end
 end
 assert(numel(checkpoints)==30 && numel(rows)==720,'Baseline:Population','Final parity population changed.');
 % The delivered SLX opens with a declared first checkpoint, never best-seed selection.
 assignin(ws,'baseline_inputs',timeseries(U,tick));assignin(ws,'baseline_p',example.parameters);
 assignin(ws,'baseline_mu',mu);assignin(ws,'baseline_sd',sd);assignin(ws,'baseline_arch',example.architecture);
 set_param(modelName,'Description','Frozen A inference: linear/no-replay/seed101/stage3 demonstration with24 reporting views. Tick=record order, not physical time. Full30-checkpoint parity is in the separate result. No online learning.');
 save_system(modelName,modelPath);
 normalizer=B.normalizer;save(bankPath,'bank','normalizer','seeds','archs','arms','-v7.3');
 report.status='PASS';report.model_path=modelPath;report.final_stage_bank_path=bankPath;
 report.demo_checkpoint=rmfield(example,'parameters');report.numeric_tolerance=tol;
 report.source_prior_pins=protocol.source_prior_pins;report.elapsed_seconds=toc(started);
 report.checkpoints=checkpoints;
 writetable(struct2table(rows),fullfile(outputDir,'baseline_simulink_decisions_v1.csv'));
 writeJson(fullfile(outputDir,'baseline_simulink_parity_result_v1.json'),report);
catch errorInfo
 report.status='FAILED';report.error_identifier=errorInfo.identifier;report.error_message=errorInfo.message;
 report.elapsed_seconds=toc(started);report.checkpoints=checkpoints;
 if ~isempty(rows),writetable(struct2table(rows),fullfile(outputDir,'baseline_completed_decisions_partial_v1.csv'));end
 writeJson(fullfile(outputDir,'baseline_simulink_parity_result_v1.json'),report);
 rethrow(errorInfo);
end
end

function p=packParameters(m,ai)
p=zeros(32,1);
if ai==1
 assert(strcmp(m.arch,'linear'),'Baseline:Architecture','Linear architecture mismatch.');
 p(1:12)=[m.W(:);m.b(:)];
else
 assert(strcmp(m.arch,'mlp_tanh'),'Baseline:Architecture','Tanh architecture mismatch.');
 p(:)=[m.W1(:);m.b1(:);m.W2(:);m.b2(:)];
end
assert(all(isfinite(p)),'Baseline:Parameters','Nonfinite frozen weights.');
end

function o=outcomes(selected,status,truth)
o=repmat("nonselection",numel(selected),1);o(status<0)="failure";
o(status>0 & selected==truth)="selected_correct";o(status>0 & selected~=truth)="selected_wrong";
end

function m=metrics(o)
n=numel(o);c=sum(o=="selected_correct");w=sum(o=="selected_wrong");
m=struct('views',n,'selected_correct',c,'selected_wrong',w, ...
 'nonselection',sum(o=="nonselection"),'failure',sum(o=="failure"), ...
 'selected_correct_fraction',c/n,'coverage',(c+w)/n,'selective_error',w/max(1,c+w),'selective_error_defined',c+w>0);
assert(m.selected_correct+m.selected_wrong+m.nonselection+m.failure==n,'Baseline:Counts','Outcome population changed.');
end

function writeJson(file,value)
fid=fopen(file,'w','n','UTF-8');assert(fid~=-1,'Baseline:Write','Could not write result.');
c=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(value,'PrettyPrint',true,'ConvertInfAndNaN',true));
end
