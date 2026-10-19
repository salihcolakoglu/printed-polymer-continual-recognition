function varargout = a29cl_math(action, varargin)
%A29CL_MATH Pure binary64 kernels for A29 matched-CL protocol v1.
% Independently authored source-informed implementation, not an execution gate.
% No file, source-data, persistent-cache, network or toolbox access.
% theta = [W1(:); b1(:); W2(:); b2(:)] in MATLAB column-major order.
% API: init(D,seed); forward(theta,Z,D); ce(theta,Z,y,mask,D);
% softce(theta,Z,q,mask,T,D); bce(theta,Z,q,D); fisher(theta,Z,mask,D);
% objective(theta,terms,D,anchors,lambda); step(state,g);
% decode(theta,Z,D,eligible,codebook). See accompanying implementation notes.
if ~(ischar(action) || (isstring(action) && isscalar(action)))
    error('A29CL:Action','Action must be scalar text.');
end
switch char(action)
    case 'init'
        [varargout{1:nargout}] = initialize(varargin{:});
    case 'unpack'
        [varargout{1:nargout}] = unpack(varargin{:});
    case 'pack'
        [varargout{1:nargout}] = pack(varargin{:});
    case 'forward'
        [varargout{1:nargout}] = forward(varargin{:});
    case 'ce'
        [varargout{1:nargout}] = ce(varargin{:});
    case 'softce'
        [varargout{1:nargout}] = softce(varargin{:});
    case 'bce'
        [varargout{1:nargout}] = bce(varargin{:});
    case 'fisher'
        [varargout{1:nargout}] = fisher(varargin{:});
    case 'objective'
        [varargout{1:nargout}] = objective(varargin{:});
    case 'step'
        [varargout{1:nargout}] = adam_step(varargin{:});
    case 'decode'
        [varargout{1:nargout}] = decode(varargin{:});
    otherwise
        error('A29CL:Action','Unknown math action: %s',char(action));
end
end

function state = initialize(D,seed)
valid_D(D);
if ~(isnumeric(seed) && isscalar(seed) && isreal(seed) && isfinite(seed) ...
        && seed >= 0 && seed == fix(seed) && seed <= 2^32-1)
    error('A29CL:Seed','Seed must be an integer in the MATLAB twister range.');
end
rng(double(seed),'twister');
W1 = (2*rand(2,4)-1)*sqrt(6/(2+4));
W2 = (2*rand(4,D)-1)*sqrt(6/(4+D));
theta = [W1(:); zeros(4,1); W2(:); zeros(D,1)];
state = struct('theta',theta,'m',zeros(size(theta)),'v',zeros(size(theta)), ...
    't',uint64(0));
end

function [logits,h] = forward(theta,Z,D)
[W1,b1,W2,b2] = unpack(theta,D);
valid_Z(Z);
h = tanh(bsxfun(@plus,Z*W1,b1));
logits = bsxfun(@plus,h*W2,b2);
if any(~isfinite(logits(:))) || any(~isfinite(h(:)))
    error('A29CL:Nonfinite','Nonfinite network activation.');
end
end

function [loss,g] = ce(theta,Z,y,mask,D)
mask = valid_mask(mask,D);
N = size(Z,1);
if ~(isnumeric(y) && isreal(y) && numel(y)==N && all(isfinite(y(:))) ...
        && all(y(:)==fix(y(:))))
    error('A29CL:Target','Hard target must have one finite global label per row.');
end
y = double(y(:));
[found,local] = ismember(y,mask);
if ~all(found)
    error('A29CL:Mask','Every hard target must be eligible under its CE mask.');
end
[logits,h] = forward(theta,Z,D);
a = logits(:,mask);
[logp,p] = logsoftmax(a);
idx = sub2ind(size(a),(1:N)',local);
loss = -sum(logp(idx))/N;
d = p;
d(idx) = d(idx)-1;
d = d/N;
dlogits = zeros(N,D);
dlogits(:,mask) = d;
g = backprop(theta,Z,h,dlogits,D);
finite_loss_gradient(loss,g);
end

function [loss,g] = softce(theta,Z,q,mask,T,D)
mask = valid_mask(mask,D);
if ~(isa(T,'double') && isscalar(T) && isreal(T) && isfinite(T) && T>0)
    error('A29CL:Temperature','Temperature must be positive finite binary64.');
end
[logits,h] = forward(theta,Z,D);
N = size(Z,1);
valid_q(q,N,numel(mask));
if any(abs(sum(q,2)-1)>1e-12)
    error('A29CL:Target','KD target rows must sum to one (absolute tolerance 1e-12).');
end
[logp,p] = logsoftmax(logits(:,mask)/T);
loss = -sum(sum(q.*logp,2))/N;
dlogits = zeros(N,D);
% q is immutable data: the derivative is only through the student logits.
% No explicit T^2 factor; no extra averaging over old tasks here.
dlogits(:,mask) = (bsxfun(@times,p,sum(q,2))-q)/(N*T);
g = backprop(theta,Z,h,dlogits,D);
finite_loss_gradient(loss,g);
end

function [loss,g] = bce(theta,Z,q,D)
if D~=8
    error('A29CL:Width','Protocol coded BCE requires eight outputs.');
end
[logits,h] = forward(theta,Z,D);
N = size(Z,1);
valid_q(q,N,D);
% Stable softplus(x)-q*x. No probability or target clipping.
values = max(logits,0)-q.*logits+log1p(exp(-abs(logits)));
loss = sum(values(:))/(N*D);
dlogits = (sigmoid(logits)-q)/(N*D);
g = backprop(theta,Z,h,dlogits,D);
finite_loss_gradient(loss,g);
end

function [F,work] = fisher(theta,Z,mask,D)
if D~=4
    error('A29CL:Width','Protocol EWC model Fisher requires four outputs.');
end
mask = valid_mask(mask,D);
[logits,h] = forward(theta,Z,D);
[~,p] = logsoftmax(logits(:,mask));
N = size(Z,1);
F = zeros(numel(theta),1);
% Exact categorical model expectation, not an observed-label Fisher and not
% the squared minibatch gradient. Reuse the single batched forward cache;
for i = 1:N
    for k = 1:numel(mask)
        d = p(i,:);
        d(k) = d(k)-1;
        dlogits = zeros(1,D);
        dlogits(mask) = d;
        % Negative log-likelihood gradient; squaring removes its minus sign.
        g = backprop(theta,Z(i,:),h(i,:),dlogits,D);
        F = F+p(i,k)*(g.^2)/N;
    end
end
if any(~isfinite(F)) || any(F<0)
    error('A29CL:Nonfinite','Invalid expected Fisher diagonal.');
end
work = struct('model_input_rows',N,'class_gradient_terms',N*numel(mask), ...
    'implementation_forward_rows',N, ...
    'includes_L2',false,'includes_penalty',false,'uses_observed_labels',false);
end

function [loss,g,components] = objective(theta,terms,D,anchors,ewc_lambda)
[W1,~,W2,~] = unpack(theta,D);
if ~isstruct(terms) || ~isstruct(anchors)
    error('A29CL:Specification','Terms and anchors must be struct arrays, including empty arrays.');
end
if ~(isa(ewc_lambda,'double') && isscalar(ewc_lambda) && isreal(ewc_lambda) ...
        && isfinite(ewc_lambda) && ewc_lambda>=0)
    error('A29CL:Specification','EWC coefficient must be nonnegative finite binary64.');
end
loss = 0;
g = zeros(size(theta));
items = repmat(struct('index',0,'kind','','weight',0,'unweighted_loss',0, ...
    'weighted_loss',0,'rows',0,'mask_width',0),numel(terms),1);
for k = 1:numel(terms)
    term = terms(k);
    required(term,{'kind','Z','target','mask','T','weight'});
    if ~(isa(term.weight,'double') && isscalar(term.weight) && isreal(term.weight) ...
            && isfinite(term.weight) && term.weight>=0)
        error('A29CL:Specification','Every term weight must be finite nonnegative binary64.');
    end
    switch char(term.kind)
        case 'ce'
            [value,grad] = ce(theta,term.Z,term.target,term.mask,D);
        case 'softce'
            [value,grad] = softce(theta,term.Z,term.target,term.mask,term.T,D);
        case 'bce'
            [value,grad] = bce(theta,term.Z,term.target,D);
        otherwise
            error('A29CL:Specification','Unknown objective term kind.');
    end
    loss = loss+term.weight*value;
    g = g+term.weight*grad;
    items(k) = struct('index',k,'kind',char(term.kind),'weight',term.weight, ...
        'unweighted_loss',value,'weighted_loss',term.weight*value, ...
        'rows',size(term.Z,1),'mask_width',numel(term.mask));
end
lambda_L2 = 1e-4;
L2 = lambda_L2/2*(sum(W1(:).^2)+sum(W2(:).^2));
L2grad = [lambda_L2*W1(:);zeros(4,1);lambda_L2*W2(:);zeros(D,1)];
penalty = 0;
pengrad = zeros(size(theta));
for k = 1:numel(anchors)
    required(anchors(k),{'theta','F'});
    ref = anchors(k).theta;
    F = anchors(k).F;
    if ~(isa(ref,'double') && iscolumn(ref) && isequal(size(ref),size(theta)) ...
            && isreal(ref) && all(isfinite(ref)) && isa(F,'double') ...
            && iscolumn(F) && isequal(size(F),size(theta)) && isreal(F) ...
            && all(isfinite(F)) && all(F>=0))
        error('A29CL:Anchor','Every EWC anchor must contain finite theta and nonnegative same-sized Fisher.');
    end
    delta = theta-ref;
    penalty = penalty+ewc_lambda/2*sum(F.*delta.^2);
    pengrad = pengrad+ewc_lambda*F.*delta;
end
loss = loss+L2+penalty;
g = g+L2grad+pengrad;
finite_loss_gradient(loss,g);
components = struct('terms',items,'L2',L2,'EWC',penalty,'total',loss, ...
    'anchor_count',numel(anchors),'lambda_EWC',ewc_lambda,'lambda_L2',lambda_L2);
end

function state = adam_step(state,g)
required(state,{'theta','m','v','t'});
P = numel(state.theta);
if ~(isa(state.theta,'double') && iscolumn(state.theta) && isreal(state.theta) ...
        && all(isfinite(state.theta)) && isa(state.m,'double') ...
        && isequal(size(state.m),size(state.theta)) && isreal(state.m) ...
        && all(isfinite(state.m)) && isa(state.v,'double') ...
        && isequal(size(state.v),size(state.theta)) && isreal(state.v) ...
        && all(isfinite(state.v)) && all(state.v>=0) && isa(g,'double') ...
        && iscolumn(g) && numel(g)==P && isreal(g) && all(isfinite(g)))
    error('A29CL:State','Invalid Adam state or gradient.');
end
if ~(isa(state.t,'uint64') && isscalar(state.t) && state.t<uint64(1000000))
    error('A29CL:State','Adam step counter must be bounded scalar uint64.');
end
state.m = 0.9*state.m+(1-0.9)*g;
state.v = 0.999*state.v+(1-0.999)*(g.^2);
state.t = state.t+uint64(1);
t = double(state.t);
state.theta = state.theta-0.02*(state.m/(1-0.9^t))./(sqrt(state.v/(1-0.999^t))+1e-8);
if any(~isfinite(state.theta)) || any(~isfinite(state.m)) || any(~isfinite(state.v))
    error('A29CL:Nonfinite','Nonfinite Adam update.');
end
end

function out = decode(theta,Z,D,eligible,codebook)
valid_D(D);
eligible = valid_mask(eligible,D);
if D==8 && any(eligible>4)
    error('A29CL:Mask','Coded decoder eligible labels must be global IDs 1..4.');
end
if ~(isa(Z,'double') && isreal(Z) && ismatrix(Z) && size(Z,2)==2 && size(Z,1)>=1)
    error('A29CL:Input','Z must be a nonempty real binary64 N by 2 matrix.');
end
P = 12+5*D;
if ~(isa(theta,'double') && isreal(theta) && iscolumn(theta) && numel(theta)==P)
    error('A29CL:State','Invalid decoder theta shape or type.');
end
N = size(Z,1);
K = numel(eligible);
out = struct('label',nan(N,1),'gap',nan(N,1),'tie',false(N,1), ...
    'valid',false(N,1),'failure',true(N,1),'failure_reason',{repmat({''},N,1)}, ...
    'scores',nan(N,K),'distances',nan(N,K),'logits',nan(N,D), ...
    'outputs',nan(N,D),'eligible',eligible,'higher_is_better',true);
if D==8
    if ~(isa(codebook,'double') && isreal(codebook) && isequal(size(codebook),[4,8]) ...
            && all(isfinite(codebook(:))) && all(codebook(:)==0 | codebook(:)==1))
        error('A29CL:Codebook','Coded decoder requires a finite binary64 binary 4 by 8 codebook.');
    end
elseif ~isempty(codebook)
    error('A29CL:Codebook','CE decoder must not receive a codebook.');
end
if any(~isfinite(theta))
    out.failure_reason(:) = {'nonfinite_state'};
    return
end
% Keep every query in the output; nonfinite rows become explicit failures.
for i=1:N
    if any(~isfinite(Z(i,:)))
        out.failure_reason{i} = 'nonfinite_query';
        continue
    end
    try
        a = forward(theta,Z(i,:),D);
    catch err
        if strcmp(err.identifier,'A29CL:Nonfinite')
            out.failure_reason{i} = 'nonfinite_activation';
            continue
        end
        rethrow(err);
    end
    out.logits(i,:) = a;
    if D==4
        [~,p] = logsoftmax(a(eligible));
        score = p;
        out.outputs(i,eligible) = p;
    else
        v = sigmoid(a);
        out.outputs(i,:) = v;
        delta = bsxfun(@minus,codebook(eligible,:),v);
        distance = sum(delta.^2,2)'/8;
        out.distances(i,:) = distance;
        score = -distance;
    end
    if any(~isfinite(score))
        out.failure_reason{i} = 'nonfinite_candidate';
        continue
    end
    out.scores(i,:) = score;
    best = max(score);
    tied = find(score==best);
    % Eligible is canonical ascending ID order; exact ties choose lowest ID.
    out.label(i) = eligible(tied(1));
    out.tie(i) = numel(tied)>1;
    ranked = sort(score,'descend');
    out.gap(i) = ranked(1)-ranked(2);
    out.valid(i) = true;
    out.failure(i) = false;
end
end

function g = backprop(theta,Z,h,dlogits,D)
[~,~,W2,~] = unpack(theta,D);
gW2 = h'*dlogits;
gb2 = sum(dlogits,1);
dh = (dlogits*W2').*(1-h.^2);
gW1 = Z'*dh;
gb1 = sum(dh,1);
g = [gW1(:);gb1(:);gW2(:);gb2(:)];
end

function [W1,b1,W2,b2] = unpack(theta,D)
valid_D(D);
if ~(isa(theta,'double') && isreal(theta) && iscolumn(theta) ...
        && numel(theta)==12+5*D && all(isfinite(theta)))
    error('A29CL:State','theta must be finite binary64 column with 12+5D parameters.');
end
W1 = reshape(theta(1:8),2,4);
b1 = reshape(theta(9:12),1,4);
W2 = reshape(theta(13:12+4*D),4,D);
b2 = reshape(theta(13+4*D:end),1,D);
end

function theta = pack(W1,b1,W2,b2)
D = size(W2,2);
valid_D(D);
if ~(isa(W1,'double') && isreal(W1) && isequal(size(W1),[2,4]) ...
        && isa(b1,'double') && isreal(b1) && isequal(size(b1),[1,4]) ...
        && isa(W2,'double') && isreal(W2) && isequal(size(W2),[4,D]) ...
        && isa(b2,'double') && isreal(b2) && isequal(size(b2),[1,D]))
    error('A29CL:State','Invalid array shapes/types for theta packing.');
end
theta = [W1(:);b1(:);W2(:);b2(:)];
if any(~isfinite(theta))
    error('A29CL:Nonfinite','Nonfinite parameter array.');
end
end

function valid_D(D)
if ~(isnumeric(D) && isreal(D) && isscalar(D) && isfinite(D) && any(D==[4,8]))
    error('A29CL:Width','Only protocol widths D=4 or D=8 are admitted.');
end
end

function valid_Z(Z)
if ~(isa(Z,'double') && isreal(Z) && ismatrix(Z) && size(Z,2)==2 ...
        && size(Z,1)>=1 && all(isfinite(Z(:))))
    error('A29CL:Input','Z must be nonempty finite real binary64 N by 2.');
end
end

function mask = valid_mask(mask,D)
valid_D(D);
if ~(isnumeric(mask) && isreal(mask) && isvector(mask) && numel(mask)>=2 ...
        && all(isfinite(mask(:))) && all(mask(:)==fix(mask(:))) ...
        && all(mask(:)>=1) && all(mask(:)<=min(D,4)))
    error('A29CL:Mask','Mask must contain at least two distinct global IDs in 1..4.');
end
mask = double(mask(:)');
if numel(unique(mask))~=numel(mask) || ~isequal(mask,sort(mask))
    error('A29CL:Mask','Mask must be unique ascending global IDs.');
end
end

function valid_q(q,N,K)
if ~(isa(q,'double') && isreal(q) && isequal(size(q),[N,K]) ...
        && all(isfinite(q(:))) && all(q(:)>=0) && all(q(:)<=1))
    error('A29CL:Target','Soft target must be a finite binary64 N by K matrix in [0,1].');
end
end

function [logp,p] = logsoftmax(a)
m = max(a,[],2);
z = bsxfun(@minus,a,m);
logp = bsxfun(@minus,z,log(sum(exp(z),2)));
p = exp(logp);
end

function s = sigmoid(a)
s = zeros(size(a));
pos = a>=0;
s(pos) = 1./(1+exp(-a(pos)));
e = exp(a(~pos));
s(~pos) = e./(1+e);
end

function finite_loss_gradient(loss,g)
if ~isfinite(loss) || any(~isfinite(g))
    error('A29CL:Nonfinite','Nonfinite objective or gradient.');
end
end

function required(s,names)
if ~isstruct(s) || ~isscalar(s) || ~all(isfield(s,names))
    error('A29CL:Specification','Missing required scalar struct fields.');
end
end
