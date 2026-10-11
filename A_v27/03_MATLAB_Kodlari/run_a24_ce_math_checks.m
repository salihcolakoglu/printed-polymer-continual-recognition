function report = run_a24_ce_math_checks(outputPath)
% New general-mask CE checks only; exactly two 32-parameter fixtures.
% No old A controls, source records, training, or model fitting are run here.
if nargin<1, outputPath=''; end
here=fileparts(mfilename('fullpath'));
addpath(here,fullfile(here,'utility'));
items=struct('id',{},'passed',{},'detail',{});
names={'W1','b1','W2','b2'}; lambda=1e-4; step=1e-6;
for fixture=1:2
    if fixture==1
        model=r04nn_init('mlp_tanh',2,4,4,41);
        X=[-.8,.2;.4,-.3;.7,.9]; Y=[3;3;3];
        mask=repmat(logical([1,1,1,0]),3,1); w=[1;1;1];
    else
        model=r04nn_init('mlp_tanh',2,4,4,43);
        X=[.3,-.9;-.6,.2;.5,.7]; Y=[4;1;3];
        mask=true(3,4); w=[1;.25;2];
    end
    [loss,g,aux]=a24_ce_lossgrad(model,X,Y,mask,w,lambda);
    maxError=0; parameterCount=0;
    for a=1:numel(names)
        name=names{a};
        for j=1:numel(model.(name))
            plus=model; minus=model;
            plus.(name)(j)=plus.(name)(j)+step;
            minus.(name)(j)=minus.(name)(j)-step;
            lp=a24_ce_lossgrad(plus,X,Y,mask,w,lambda);
            lm=a24_ce_lossgrad(minus,X,Y,mask,w,lambda);
            numeric=(lp-lm)/(2*step);
            maxError=max(maxError,abs(numeric-g.(name)(j)));
            parameterCount=parameterCount+1;
        end
    end
    record(sprintf('ce_central_difference_%d',fixture), ...
        parameterCount==32 && maxError<5e-7, ...
        sprintf('32 parameters; %d eligible classes; max absolute error %.17g.', ...
        sum(mask(1,:)),maxError));
    record(sprintf('ce_mask_probability_%d',fixture), ...
        all(aux.P(~mask)==0) && max(abs(sum(aux.P,2)-1))<1e-14 && ...
        all(aux.row_logit_gradient(~mask)==0), ...
        'Inactive probabilities and logit gradients are zero; active probabilities sum to one.');
    [scaledLoss,scaledGrad]=a24_ce_lossgrad(model,X,Y,mask,7*w,lambda);
    weightError=0;
    for a=1:numel(names)
        weightError=max(weightError,max(abs(g.(names{a})-scaledGrad.(names{a})),[],'all'));
    end
    record(sprintf('ce_weight_normalization_%d',fixture), ...
        abs(loss-scaledLoss)<1e-12 && weightError<1e-12, ...
        'Current/replay row weights normalize once over the complete batch.');
    [loss0,g0]=a24_ce_lossgrad(model,X,Y,mask,w,0);
    expectedPenalty=.5*lambda*(sum(model.W1.^2,'all')+sum(model.W2.^2,'all'));
    decayError=max([max(abs(g.W1-g0.W1-lambda*model.W1),[],'all'), ...
        max(abs(g.W2-g0.W2-lambda*model.W2),[],'all'), ...
        max(abs(g.b1-g0.b1),[],'all'),max(abs(g.b2-g0.b2),[],'all')]);
    record(sprintf('ce_weight_only_decay_%d',fixture), ...
        abs(loss-loss0-expectedPenalty)<1e-12 && decayError<1e-12, ...
        'The L2 penalty and gradient affect original weight matrices only, never biases.');
end
Z=[1000,999,-1000,10;-900,500,1000,-1000];
mask=logical([1,1,0,0;0,1,1,1]);
[P,logden,mx]=a24_seen_softmax(Z,mask);
record('ce_extreme_logits_2_and_3_masks',all(isfinite(P),'all') && ...
    all(isfinite(logden)) && all(isfinite(mx)) && all(P(~mask)==0) && ...
    max(abs(sum(P,2)-1))<1e-14, ...
    'Stable two-class and three-class softmax at extreme finite logits.');
zeroModel=model;
for a=1:numel(names), zeroModel.(names{a})=zeros(size(zeroModel.(names{a}))); end
[P,pred,Z]=a24_ce_predict(zeroModel,[0,0;1,-1],mask);
record('ce_predict_eligibility_ties',isequal(pred,[1;2]) && all(Z(:)==0) && ...
    all(P(~mask)==0), ...
    'Zero-logit exact ties choose the smallest eligible global label per row.');
rejected=false;
try
    a24_ce_lossgrad(model,[0,0],4,logical([1,1,1,0]),1,lambda);
catch err
    rejected=strcmp(err.identifier,'a24:TargetMask');
end
record('ce_ineligible_target_rejected',rejected, ...
    'A target outside its seen-class mask is rejected before gradient use.');
report=struct('status','NOT_EVALUATED','scope','NEW_A24_GENERAL_MASK_CE_ONLY', ...
    'central_difference_fixtures',2,'parameters_per_fixture',32, ...
    'source_record_training_runs',0,'checks',{items}, ...
    'check_count',numel(items),'failures',sum(~[items.passed]));
if report.failures==0, report.status='PASS'; else, report.status='FAIL'; end
if ~isempty(outputPath)
    fid=fopen(outputPath,'w','n','UTF-8');
    assert(fid~=-1,'a24:Output','Cannot open requested result path.');
    cleaner=onCleanup(@()fclose(fid)); %#ok<NASGU>
    fprintf(fid,'%s\n',jsonencode(report,'PrettyPrint',true));
end
assert(report.failures==0,'a24:MathChecks','General-mask CE checks failed.');

    function record(id,passed,detail)
        items(end+1)=struct('id',id,'passed',logical(passed),'detail',detail); %#ok<AGROW>
    end
end
