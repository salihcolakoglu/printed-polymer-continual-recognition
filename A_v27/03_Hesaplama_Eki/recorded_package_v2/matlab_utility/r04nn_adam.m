function [model,state] = r04nn_adam(model,g,state,opt)
% Explicit state must be passed to every update, including later stages.
if nargin<4 || isempty(opt), opt=struct('eta',.02,'beta1',.9,'beta2',.999,'epsilon',1e-8); end
validateattributes(opt.eta,{'double'},{'scalar','finite','positive'});
validateattributes(opt.beta1,{'double'},{'scalar','>=',0,'<',1});
validateattributes(opt.beta2,{'double'},{'scalar','>=',0,'<',1});
validateattributes(opt.epsilon,{'double'},{'scalar','finite','positive'});
names=r04nn_param_names(model);
if isempty(state)
    state=struct('t',0,'m',struct(),'v',struct());
    for j=1:numel(names), state.m.(names{j})=zeros(size(model.(names{j}))); state.v.(names{j})=zeros(size(model.(names{j}))); end
end
validateattributes(state.t,{'numeric'},{'scalar','integer','nonnegative'});
state.t=state.t+1;
for j=1:numel(names)
    n=names{j}; assert(isequal(size(g.(n)),size(model.(n))) && isequal(size(state.m.(n)),size(model.(n))) && isequal(size(state.v.(n)),size(model.(n))),'r04nn:Dimensions','Adam parameter/state dimension mismatch.');
    validateattributes(g.(n),{'double'},{'real','finite'});
    state.m.(n)=opt.beta1*state.m.(n)+(1-opt.beta1)*g.(n);
    state.v.(n)=opt.beta2*state.v.(n)+(1-opt.beta2)*(g.(n).^2);
    mhat=state.m.(n)/(1-opt.beta1^state.t); vhat=state.v.(n)/(1-opt.beta2^state.t);
    model.(n)=model.(n)-opt.eta*mhat./(sqrt(vhat)+opt.epsilon);
end
end
