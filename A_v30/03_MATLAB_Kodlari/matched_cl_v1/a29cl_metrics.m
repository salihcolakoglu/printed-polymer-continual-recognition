function out = a29cl_metrics(action,input)
%A29CL_METRICS Pure fixed-population A29 descriptive metrics; no file access.
% coverage(predictionCells) -> native-score whole-tie-block C/W/A/F curve.
% summarize(resultPipelineOrPredictionCells) -> fold/stage and pooled/stage
% counts, class/task units, acquisition, old accuracy, BWT and signed forgetting.
% NaN means undefined/unavailable (JSON null), never a zero substitute.
if ~(ischar(action) || (isstring(action) && isscalar(action)))
    error('A29CL:Metrics','Metric action must be scalar text.');
end
switch char(action)
    case 'coverage'
        out = coverage(cells(input));
    case 'summarize'
        out = summarize(input);
    otherwise
        error('A29CL:Metrics','Unknown metric action.');
end
end

function out = coverage(P)
N = numel(P);
correct = false(N,1);valid = false(N,1);gaps = nan(N,1);
for k=1:N
    [valid(k),correct(k),gaps(k)] = decision(P{k});
end
levels = sort(unique(gaps(valid)),'descend');
thresholds = [Inf;levels(:)];
steps = repmat(struct('threshold',Inf,'threshold_is_infinite',true, ...
    'C',0,'W',0,'A',0,'F',0,'N',N,'answered',0, ...
    'coverage',0,'conditional_error',NaN),numel(thresholds),1);
for k=1:numel(thresholds)
    selected = valid & gaps>=thresholds(k);
    C = sum(selected & correct);
    W = sum(selected & ~correct);
    A = sum(valid & ~selected);
    F = sum(~valid);
    answered = C+W;
    conditional_error = NaN;
    if answered>0
        conditional_error = W/answered;
    end
    fraction = NaN;
    if N>0
        fraction = answered/N;
    end
    steps(k) = struct('threshold',thresholds(k), ...
        'threshold_is_infinite',isinf(thresholds(k)), ...
        'C',C,'W',W,'A',A,'F',F,'N',N,'answered',answered, ...
        'coverage',fraction,'conditional_error',conditional_error);
end
out = struct('schema','A29CL_COVERAGE_V1','N',N,'valid_count',sum(valid), ...
    'failure_count',sum(~valid),'steps',steps, ...
    'threshold_selection_uses_truth',false,'whole_exact_gap_tie_blocks',true, ...
    'calibrated_risk_guarantee',false);
end

function out = summarize(input)
kind = 'UNKNOWN_PREDICTION_PROVENANCE';
if isstruct(input) && isscalar(input) && isfield(input,'predictions')
    P = cells(input.predictions);
    if isfield(input,'kind')
        kind = char(input.kind);
    end
else
    P = cells(input);
end
N = numel(P);
meta = repmat(struct('fold',0,'seed',0,'arm','','stream','','stage',0),N,1);
foldkeys = cell(N,1);poolkeys = cell(N,1);rowkeys = cell(N,1);
for k=1:N
    r = P{k};
    need(r,{'fold','seed','arm','stream','stage','task','parentrow','truth'});
    if ~(isscalar(r.fold) && isfinite(r.fold) && any(r.fold==1:5) ...
            && isscalar(r.seed) && isfinite(r.seed) && r.seed>=0 && r.seed==fix(r.seed) ...
            && isscalar(r.stage) && isfinite(r.stage) && any(r.stage==1:3) ...
            && isscalar(r.task) && isfinite(r.task) && r.task==fix(r.task) ...
            && isscalar(r.truth) && isfinite(r.truth) && any(r.truth==1:4))
        error('A29CL:Metrics','Invalid prediction metadata.');
    end
    stream = char(r.stream);arm = char(r.arm);
    if ~(strcmp(stream,'TaskIL') || strcmp(stream,'ClassIL'))
        error('A29CL:Metrics','Unknown evaluation stream.');
    end
    if strcmp(stream,'TaskIL') && ~(r.task>=1 && r.task<=r.stage)
        error('A29CL:Metrics','TaskIL task view must be acquired.');
    elseif strcmp(stream,'ClassIL') && r.task~=0
        error('A29CL:Metrics','ClassIL query must not carry a task ID.');
    end
    meta(k) = struct('fold',double(r.fold),'seed',double(r.seed),'arm',arm, ...
        'stream',stream,'stage',double(r.stage));
    foldkeys{k} = sprintf('%d|%d|%s|%s|%d',r.fold,r.seed,arm,stream,r.stage);
    poolkeys{k} = sprintf('%d|%s|%s|%d',r.seed,arm,stream,r.stage);
    rowkeys{k} = sprintf('%s|%d|%s',foldkeys{k},r.task,char(r.parentrow));
end
if numel(unique(rowkeys))~=N
    error('A29CL:Metrics','Duplicate logical prediction key; refusing a changed denominator.');
end
fold_stage = {};pooled_stage = {};curves = {};
[~,~,fg] = unique(foldkeys);
[~,~,pg] = unique(poolkeys);
for k=1:max([0;fg])
    idx = find(fg==k);
    m = meta(idx(1));
    block = stage_metric(P(idx),m,kind,false);
    fold_stage{end+1} = block; %#ok<AGROW>
    curves{end+1} = struct('fold',m.fold,'seed',m.seed,'arm',m.arm, ...
        'stream',m.stream,'stage',m.stage,'curve',coverage(P(idx))); %#ok<AGROW>
end
for k=1:max([0;pg])
    idx = find(pg==k);
    m = meta(idx(1));m.fold=0;
    pooled_stage{end+1} = stage_metric(P(idx),m,kind,true); %#ok<AGROW>
end
retention = trajectories(pooled_stage);
seed_summary = seed_ranges(pooled_stage);
folds = unique([meta.fold]);
all_complete = ~isempty(pooled_stage);
for k=1:numel(pooled_stage)
    all_complete = all_complete && pooled_stage{k}.complete_expected_OOF;
end
complete = strcmp(kind,'EXPOSED_SOURCE_AVERAGE_MATCHED_CL_COMPARISON') ...
    && isequal(folds,1:5) && all_complete;
if complete
    population_status = 'COMPLETE_EXPOSED_SOURCE_OOF_NOT_BLIND_PHYSICAL_VALIDATION';
elseif strcmp(kind,'SOURCE_FREE_ARTIFICIAL_IMPLEMENTATION_CONTROL_NOT_SCIENTIFIC_RESULT')
    population_status = 'SOURCE_FREE_ARTIFICIAL_CONTROL_NOT_SCIENTIFIC_OOF';
else
    population_status = 'PARTIAL_OR_UNVERIFIED_POPULATION_NOT_COMPLETE_OOF';
end
out = struct('schema','A29CL_SUMMARY_V1','kind',kind, ...
    'N_logical_views',N,'folds_present',folds, ...
    'complete_source_OOF',complete,'population_status',population_status, ...
    'fold_stage',{fold_stage},'pooled_stage',{pooled_stage}, ...
    'retention',{retention},'coverage',{curves},'seed_summary',{seed_summary}, ...
    'scope','Fixed exposed source-group views, seeds descriptive; no inferential p-values or physical confidence interval.', ...
    'score_cross_fold_pooling',false,'seed_selection',false, ...
    'physical_validation',0);
end

function m = stage_metric(P,key,kind,pooled)
N = numel(P);correct = false(N,1);valid = false(N,1);
truth = zeros(N,1);task = zeros(N,1);parents = cell(N,1);folds = zeros(N,1);
confusion = zeros(4,4);failure_by_truth = zeros(4,1);
for k=1:N
    r=P{k};[valid(k),correct(k)] = decision(r);
    truth(k)=r.truth;task(k)=r.task;parents{k}=char(r.parentrow);folds(k)=r.fold;
    if valid(k)
        confusion(r.truth,r.label)=confusion(r.truth,r.label)+1;
    else
        failure_by_truth(r.truth)=failure_by_truth(r.truth)+1;
    end
end
if strcmp(key.stream,'TaskIL')
    ids = 1:key.stage;unit_values=task;intro=ids;
    current_ids=key.stage;
else
    ids=1:key.stage+1;unit_values=truth;intro=max(1,ids-1);
    if key.stage==1
        current_ids=[1,2];
    else
        current_ids=key.stage+1;
    end
end
units = repmat(struct('id',0,'introduction_stage',0,'N',0,'correct',0, ...
    'failure',0,'accuracy',NaN,'expected_full_OOF_N',0,'confusion',[], ...
    'eligible_ids',[],'failure_by_truth',[]),numel(ids),1);
for k=1:numel(ids)
    sel=unit_values==ids(k);
    n=sum(sel);c=sum(sel & correct);f=sum(sel & ~valid);
    accuracy=ratio(c,n);
    if strcmp(key.stream,'TaskIL')
        eligible=[1,ids(k)+1];expected=52;
    else
        eligible=ids(k);expected=26;
    end
    units(k)=struct('id',ids(k),'introduction_stage',intro(k),'N',n,'correct',c, ...
        'failure',f,'accuracy',accuracy,'expected_full_OOF_N',expected, ...
        'confusion',confusion(eligible,eligible),'eligible_ids',eligible, ...
        'failure_by_truth',failure_by_truth(eligible));
    if strcmp(key.stream,'TaskIL')
        % A repeated anchor's confusion must belong only to this task view.
        uc=zeros(2,2);uf=zeros(2,1);
        for j=find(sel)'
            ti=find(eligible==truth(j),1);
            if valid(j)
                pi=find(eligible==P{j}.label,1);
                if isempty(ti) || isempty(pi)
                    error('A29CL:Metrics','Task view truth/prediction outside its pair.');
                end
                uc(ti,pi)=uc(ti,pi)+1;
            else
                if isempty(ti), error('A29CL:Metrics','Task truth outside pair.'); end
                uf(ti)=uf(ti)+1;
            end
        end
        units(k).confusion=uc;units(k).failure_by_truth=uf;
    end
end
current = ismember(unit_values,current_ids);
old = ismember(unit_values,ids(intro<key.stage));
full_counts = all([units.N]==[units.expected_full_OOF_N]);
expected_parents = 26*(key.stage+1);
if strcmp(key.stream,'TaskIL')
    expected_N=52*key.stage;
else
    expected_N=26*(key.stage+1);
end
allfolds=isequal(unique(folds)',1:5);
complete = pooled && allfolds && full_counts && N==expected_N ...
    && numel(unique(parents))==expected_parents;
% Complete source provenance is checked additionally by summarize's pipelinekind.
macro=NaN;
if all(isfinite([units.accuracy]))
    macro=mean([units.accuracy]);
end
m=struct('fold',key.fold,'seed',key.seed,'arm',key.arm,'stream',key.stream, ...
    'stage',key.stage,'kind',kind,'pooled_OOF_counts',pooled, ...
    'folds_present',unique(folds)','complete_expected_OOF',complete, ...
    'N',N,'correct',sum(correct),'wrong_valid',sum(valid & ~correct), ...
    'failure',sum(~valid),'forced_accuracy',ratio(sum(correct),N), ...
    'unique_source_parents',numel(unique(parents)), ...
    'units',units,'macro_accuracy',macro, ...
    'class_balanced_accuracy',NaN,'task_macro_accuracy',NaN, ...
    'current_N',sum(current),'current_correct',sum(current & correct), ...
    'current_accuracy',ratio(sum(current & correct),sum(current)), ...
    'old_N',sum(old),'old_correct',sum(old & correct), ...
    'old_accuracy',ratio(sum(old & correct),sum(old)), ...
    'confusion',confusion,'failure_by_truth',failure_by_truth, ...
    'expected_full_OOF_N',expected_N);
if strcmp(key.stream,'TaskIL')
    m.task_macro_accuracy=macro;
else
    m.class_balanced_accuracy=macro;
end
end

function out = trajectories(S)
out={};keys=cell(numel(S),1);
for k=1:numel(S)
    r=S{k};keys{k}=sprintf('%d|%s|%s',r.seed,r.arm,r.stream);
end
[~,~,groups]=unique(keys);
for k=1:max([0;groups])
    B=S(groups==k);first=B{1};
    if strcmp(first.stream,'TaskIL')
        ids=[1,2];intro=[1,2];
    else
        ids=[1,2,3];intro=[1,1,2];
    end
    acq=nan(size(ids));fin=nan(size(ids));peak=nan(size(ids));
    for j=1:numel(ids)
        acq(j)=unit_at(B,intro(j),ids(j));
        fin(j)=unit_at(B,3,ids(j));
        earlier=nan(1,3-intro(j));
        for stage=intro(j):2
            earlier(stage-intro(j)+1)=unit_at(B,stage,ids(j));
        end
        if all(isfinite(earlier)),peak(j)=max(earlier);end
    end
    bwt=fin-acq;forgetting=peak-fin;
    mean_bwt=NaN;mean_forgetting=NaN;
    if all(isfinite(bwt)),mean_bwt=mean(bwt);end
    if all(isfinite(forgetting)),mean_forgetting=mean(forgetting);end
    complete=true;
    for j=1:numel(B),complete=complete && B{j}.complete_expected_OOF;end
    complete=complete && isequal(sort(cellfun(@(r)r.stage,B)),1:3);
    out{end+1}=struct('seed',first.seed,'arm',first.arm,'stream',first.stream, ...
        'old_unit_ids',ids,'introduction_stages',intro,'acquisition_accuracy',acq, ...
        'final_accuracy',fin,'peak_earlier_accuracy',peak,'BWT_by_unit',bwt, ...
        'signed_forgetting_by_unit',forgetting,'BWT',mean_bwt, ...
        'signed_forgetting',mean_forgetting,'complete_expected_OOF',complete, ...
        'stage1_BWT',NaN,'stage1_signed_forgetting',NaN); %#ok<AGROW>
end
end

function out=seed_ranges(S)
out={};keys=cell(numel(S),1);
for k=1:numel(S)
    r=S{k};keys{k}=sprintf('%s|%s|%d',r.arm,r.stream,r.stage);
end
[~,~,groups]=unique(keys);
expected=[101,211,307,401,503];
for k=1:max([0;groups])
    B=S(groups==k);first=B{1};
    seeds=cellfun(@(r)r.seed,B);values=cellfun(@(r)r.macro_accuracy,B);
    [seeds,order]=sort(seeds);values=values(order);
    all_five=isequal(seeds(:)',expected) && all(isfinite(values));
    ranges=[NaN,NaN,NaN];
    if all_five,ranges=[min(values),median(values),max(values)];end
    out{end+1}=struct('arm',first.arm,'stream',first.stream,'stage',first.stage, ...
        'expected_seed_ids',expected,'available_seed_ids',seeds(:)', ...
        'available_macro_accuracy',values(:)','all_five_seeds_present',all_five, ...
        'minimum',ranges(1),'median',ranges(2),'maximum',ranges(3), ...
        'descriptive_only',true); %#ok<AGROW>
end
end

function accuracy=unit_at(B,stage,id)
accuracy=NaN;
for k=1:numel(B)
    r=B{k};
    if r.stage==stage
        for j=1:numel(r.units)
            if r.units(j).id==id
                accuracy=r.units(j).accuracy;return
            end
        end
    end
end
end

function [valid,correct,gap]=decision(r)
need(r,{'truth','label','gap'});
if ~(isnumeric(r.truth) && isreal(r.truth) && isscalar(r.truth) ...
        && isfinite(r.truth) && any(r.truth==1:4))
    error('A29CL:Metrics','Truth must be one finite global label 1..4.');
end
valid=isnumeric(r.label) && isreal(r.label) && isscalar(r.label) ...
    && isfinite(r.label) && any(r.label==1:4) ...
    && isnumeric(r.gap) && isreal(r.gap) && isscalar(r.gap) ...
    && isfinite(r.gap) && r.gap>=0;
if isfield(r,'valid'),valid=valid && isequal(r.valid,true);end
if isfield(r,'failure'),valid=valid && isequal(r.failure,false);end
if valid && isfield(r,'eligible'),valid=any(r.label==r.eligible);end
gap=NaN;if valid,gap=double(r.gap);end
correct=valid && r.label==r.truth;
end

function out=cells(input)
if iscell(input)
    out=input(:);
elseif isstruct(input)
    out=num2cell(input(:));
else
    error('A29CL:Metrics','Predictions must be a struct array or cell array.');
end
for k=1:numel(out)
    if ~isstruct(out{k}) || ~isscalar(out{k})
        error('A29CL:Metrics','Each prediction must be a scalar struct.');
    end
end
end

function need(r,names)
if ~isstruct(r) || ~isscalar(r) || ~all(isfield(r,names))
    error('A29CL:Metrics','Missing required prediction fields.');
end
end

function x=ratio(a,b)
x=NaN;if b>0,x=a/b;end
end
