function [state,receipt] = a29cl_stage(state,spec)
%A29CL_STAGE Bounded pure full-batch Adam stage, no loader or file access.
% Production spec.updates=600. Source-free controls may use 1..599 only with
% spec.control_only=true. This flag is descriptive, not source admission.
% Required spec: terms,D,anchors,ewc_lambda,updates. terms and anchors are
% immutable MATLAB value arrays, constructed before the first update.
required = {'terms','D','anchors','ewc_lambda','updates'};
if ~isstruct(spec) || ~isscalar(spec) || ~all(isfield(spec,required))
    error('A29CL:Specification','Incomplete scalar stage specification.');
end
if ~(isnumeric(spec.updates) && isreal(spec.updates) && isscalar(spec.updates) ...
        && isfinite(spec.updates) && spec.updates==fix(spec.updates) ...
        && spec.updates>=1 && spec.updates<=600)
    error('A29CL:Budget','Stage update count must be an integer in 1..600.');
end
control_only = isfield(spec,'control_only') && isequal(spec.control_only,true);
if spec.updates~=600 && ~control_only
    error('A29CL:Budget','Short stages require explicit source-free control_only=true.');
end
if ~isstruct(spec.terms) || ~isstruct(spec.anchors)
    error('A29CL:Specification','Stage terms/anchors must be struct arrays.');
end
% MATLAB pass-by-value preserves incoming state and all detached target caches.
incoming = state;
terms = spec.terms;
anchors = spec.anchors;
rows_per_update = 0;
softmax_rows_per_update = 0;
bce_bits_per_update = 0;
for k=1:numel(terms)
    if ~all(isfield(terms(k),{'kind','Z','target','mask','T','weight'}))
        error('A29CL:Specification','Incomplete term.');
    end
    rows_per_update = rows_per_update+size(terms(k).Z,1);
    switch char(terms(k).kind)
        case {'ce','softce'}
            softmax_rows_per_update = softmax_rows_per_update+size(terms(k).Z,1);
        case 'bce'
            bce_bits_per_update = bce_bits_per_update+size(terms(k).Z,1)*spec.D;
        otherwise
            error('A29CL:Specification','Unknown term kind.');
    end
end
clock_start = tic;
first = struct();
last = struct();
for u=1:double(spec.updates)
    [loss,g,components] = a29cl_math('objective',state.theta,terms, ...
        spec.D,anchors,spec.ewc_lambda);
    if u==1 || u==spec.updates
        checkpoint = struct('update_in_stage',u,'pre_state',state, ...
            'loss',loss,'gradient',g,'components',components);
        if u==1
            first = checkpoint;
        end
        if u==spec.updates
            last = checkpoint;
        end
    end
    state = a29cl_math('step',state,g);
end
elapsed = toc(clock_start);
receipt = struct('schema','A29CL_STAGE_RECEIPT_V1','control_only',control_only, ...
    'incoming_state',incoming,'final_state',state,'D',spec.D, ...
    'updates',double(spec.updates),'first',first,'last',last, ...
    'terms',terms,'anchors',anchors,'ewc_lambda',spec.ewc_lambda, ...
    'elapsed_seconds',elapsed,'finite',true, ...
    'work',struct('optimizer_updates',double(spec.updates), ...
    'gradient_example_rows',double(spec.updates)*rows_per_update, ...
    'softmax_example_rows',double(spec.updates)*softmax_rows_per_update, ...
    'BCE_bit_terms',double(spec.updates)*bce_bits_per_update, ...
    'EWC_parameter_anchor_products',double(spec.updates)*numel(state.theta)*numel(anchors), ...
    'source_reads',0,'teacher_forward_rows_inside_stage',0, ...
    'cached_targets_recomputed',false));
end
