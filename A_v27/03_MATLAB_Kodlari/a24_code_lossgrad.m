function [loss,g,aux] = a24_code_lossgrad(model,X,targets,w,lambda)
% Weighted BCE, averaged over ALL eight output bits, plus weight-only L2.
% targets are supplied numeric constants. The driver must construct them
% ONCE at task start from the pre-task network and keep them fixed.
% No target recomputation, previous examples, or future inputs occur here.
[Z,H] = r04nn_logits(model,X);
assert(strcmp(model.arch,'mlp_tanh') && model.K==8, ...
    'a24:Architecture','Expected the declared eight-output tanh MLP.');
N = size(X,1); D = size(Z,2);
validateattributes(targets,{'double'}, ...
    {'size',[N,D],'real','finite','>=',0,'<=',1});
validateattributes(w,{'double'}, ...
    {'column','numel',N,'real','finite','nonnegative'});
validateattributes(lambda,{'double'},{'scalar','real','finite','nonnegative'});
weightMass = sum(w);
assert(isfinite(weightMass) && weightMass>0, ...
    'a24:Weights','Weight mass must be positive and finite.');
wn = w/weightMass;
P = a24_sigmoid(Z);
bitLoss = max(Z,0)-targets.*Z+log1p(exp(-abs(Z)));
perRow = mean(bitLoss,2);
dataLoss = sum(wn.*perRow);
dZ = ((P-targets)/D).*wn;
dHidden = (dZ*model.W2').*(1-H.^2);
penalty = .5*lambda*(sum(model.W1.^2,'all')+sum(model.W2.^2,'all'));
g = struct('W1',X'*dHidden+lambda*model.W1,'b1',sum(dHidden,1), ...
    'W2',H'*dZ+lambda*model.W2,'b2',sum(dZ,1));
loss = dataLoss+penalty;
aux = struct('P',P,'logits',Z,'normalized_weights',wn, ...
    'data_loss',dataLoss,'weight_penalty',penalty, ...
    'per_row_bce',perRow,'row_logit_gradient',dZ);
assert(isfinite(loss) && all(isfinite(dZ),'all'), ...
    'a24:Nonfinite','Nonfinite coded loss or logit gradient.');
end
