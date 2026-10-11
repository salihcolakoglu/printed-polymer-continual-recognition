function report = run_a24_new_math_checks(outputPath)
% New coded-loss/decode checks ONLY. Authoring does not execute this file.
% Exactly two seeded central-difference fixtures; no source-record training.
% The teacher fixture checks fixed numeric targets, not driver call timing.
if nargin<1, outputPath=''; end
here = fileparts(mfilename('fullpath'));
addpath(here,fullfile(here,'utility'));
items = struct('id',{},'passed',{},'detail',{});
C1 = a24_codebook('onehot8'); Cw = a24_codebook('walsh8');
record('dictionary_shapes',isequal(size(C1),[4,8]) && isequal(size(Cw),[4,8]), ...
    'Both dictionaries use the same four global labels and eight output bits.');
record('onehot_padding',isequal(C1,[eye(4),zeros(4,4)]), ...
    'Ordinary target control has one active bit and four trailing zero bits.');
record('walsh_balance',all(sum(Cw,2)==4), ...
    'Chosen nonconstant Hadamard rows have four active bits each.');
Q = 2*Cw-1;
record('walsh_centered_orthogonality',isequal(Q*Q',8*eye(4)), ...
    'Orthogonality refers to centered bipolar rows, not the binary rows.');
pairHamming = zeros(4);
for a=1:4
    for b=1:4, pairHamming(a,b)=sum(Cw(a,:)~=Cw(b,:)); end
end
record('walsh_pair_distance',all(pairHamming(~eye(4))==4), ...
    'Each distinct Walsh row differs at four of eight coordinates.');
extreme = a24_sigmoid([-1000,-20,0,20,1000]);
record('sigmoid_extremes',all(isfinite(extreme)) && extreme(1)==0 && ...
    extreme(3)==.5 && extreme(5)==1 && max(abs(extreme+fliplr(extreme)-1))<1e-14, ...
    'Extreme logits remain finite and the sigmoid retains reflection symmetry.');
for fixture=1:2
    if fixture==1
        model=r04nn_init('mlp_tanh',2,8,4,17);
        X=[-.7,.2;.3,-.4;.8,.6]; w=[1;2;3]; C=C1;
    else
        model=r04nn_init('mlp_tanh',2,8,4,29);
        X=[.9,-.2;-.5,.4;.1,-.8]; w=[0;4;1]; C=Cw;
    end
    pBefore=a24_sigmoid(r04nn_logits(model,X));
    targets=.5*C([1,2,3],:)+.5*pBefore;
    lambda=1e-4;
    [loss,g,aux]=a24_code_lossgrad(model,X,targets,w,lambda);
    maxError=0; names={'W1','b1','W2','b2'}; step=1e-6;
    for a=1:numel(names)
        name=names{a};
        for j=1:numel(model.(name))
            plus=model; minus=model;
            plus.(name)(j)=plus.(name)(j)+step;
            minus.(name)(j)=minus.(name)(j)-step;
            lp=a24_code_lossgrad(plus,X,targets,w,lambda);
            lm=a24_code_lossgrad(minus,X,targets,w,lambda);
            numeric=(lp-lm)/(2*step);
            maxError=max(maxError,abs(numeric-g.(name)(j)));
        end
    end
    record(sprintf('central_difference_%d',fixture),maxError<5e-7, ...
        sprintf('All 52 parameters; frozen targets; maximum absolute error %.17g.',maxError));
    [scaledLoss,scaledGrad]=a24_code_lossgrad(model,X,targets,7*w,lambda);
    gradScaleError=0;
    for a=1:numel(names)
        gradScaleError=max(gradScaleError,max(abs(g.(names{a})-scaledGrad.(names{a})),[],'all'));
    end
    record(sprintf('weight_normalization_%d',fixture), ...
        abs(loss-scaledLoss)<1e-12 && gradScaleError<1e-12 && abs(sum(aux.normalized_weights)-1)<1e-14, ...
        'Multiplying all row weights by a constant preserves loss and gradients.');
    frozen=targets; changed=model; changed.b2=changed.b2+.2;
    a24_code_lossgrad(changed,X,targets,w,lambda);
    recomputed=.5*C([1,2,3],:)+.5*a24_sigmoid(r04nn_logits(changed,X));
    record(sprintf('fixed_target_expectation_%d',fixture), ...
        isequal(targets,frozen) && max(abs(recomputed-targets),[],'all')>1e-6, ...
        'Task-start target array remains fixed although later model predictions change; driver timing is checked separately.');
end
for dictionary=1:2
    if dictionary==1, C=C1; else, C=Cw; end
    [forced,distances,gap,tied]=a24_decode(C,C,1:4);
    own=distances(sub2ind([4,4],(1:4)',(1:4)'));
    record(sprintf('exact_code_decode_%d',dictionary), ...
        isequal(forced,(1:4)') && all(own==0) && all(gap>0) && ~any(tied), ...
        'Each exact code selects its own global label with zero distance.');
end
[forced,distances,gap,tied]=a24_decode([C1(1,:);C1(3,:)],C1, ...
    logical([0,1,0,0;0,0,0,1]));
record('query_specific_mask',isequal(forced,[2;4]) && all(isnan(gap)) && ...
    ~any(tied) && all(isnan(distances([1,2,4,5,6,7]))), ...
    'A singleton eligible label is selected; inactive distances and undefined gap serialize as null.');
mid=(C1(1,:)+C1(2,:))/2;
[forced,distances,gap,tied]=a24_decode(mid,C1,[2,1]);
record('exact_tie_smallest_label',forced==1 && tied && gap==0 && ...
    all(isnan(distances(3:4))), ...
    'Exact ties are flagged and choose the smallest eligible global label, independent of supplied order.');
alpha=.5; schedule=zeros(1,3);
for j=1:3, schedule(j)=alpha; alpha=alpha/(2*alpha-alpha^2); end
record('alpha_orientation_schedule',max(abs(schedule-[.5,2/3,.75]))<1e-14, ...
    'alpha weights the frozen prediction: target=(1-alpha)*code+alpha*prediction.');
report=struct('status','NOT_EVALUATED','scope','NEW_A24_MATH_ONLY', ...
    'central_difference_fixtures',2,'source_record_training_runs',0, ...
    'driver_timing_verified',false,'checks',{items}, ...
    'check_count',numel(items),'failures',sum(~[items.passed]));
if report.failures==0, report.status='PASS'; else, report.status='FAIL'; end
if ~isempty(outputPath)
    fid=fopen(outputPath,'w','n','UTF-8');
    assert(fid~=-1,'a24:Output','Cannot open requested result path.');
    cleaner=onCleanup(@()fclose(fid)); %#ok<NASGU>
    fprintf(fid,'%s\n',jsonencode(report,'PrettyPrint',true));
end
assert(report.failures==0,'a24:MathChecks','New math checks failed.');

    function record(id,passed,detail)
        items(end+1)=struct('id',id,'passed',logical(passed),'detail',detail); %#ok<AGROW>
    end
end
