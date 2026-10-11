function [loss,g,aux] = a24_ce_lossgrad(model,X,Y,mask,w,lambda)
% Same ABI as r04nn_lossgrad; only the eligible-class count is generalized.
% Weighted per-row masked CE plus .5*lambda*sum(weight matrices.^2).
% Weights normalize over the complete supplied batch; biases are unpenalized.
[Z,H] = r04nn_logits(model,X);
[P,logden,mx] = a24_seen_softmax(Z,mask);
N=size(X,1); K=model.K;
validateattributes(Y,{'numeric'}, ...
    {'column','numel',N,'real','finite','integer','>=',1,'<=',K});
validateattributes(w,{'double'}, ...
    {'column','numel',N,'real','finite','nonnegative'});
validateattributes(lambda,{'double'},{'scalar','real','finite','nonnegative'});
idx=sub2ind([N,K],(1:N)',double(Y));
assert(all(mask(idx)),'a24:TargetMask','Each target must be eligible in its row.');
weightMass=sum(w);
assert(isfinite(weightMass) && weightMass>0, ...
    'a24:Weights','Weight mass must be positive and finite.');
wn=w/weightMass;
ce=logden+mx-Z(idx);
dataLoss=sum(wn.*ce);
dZ=P; dZ(idx)=dZ(idx)-1; dZ=dZ.*wn;
switch model.arch
    case 'linear'
        penalty=.5*lambda*sum(model.W.^2,'all');
        g=struct('W',X'*dZ+lambda*model.W,'b',sum(dZ,1));
    case 'mlp_tanh'
        penalty=.5*lambda*(sum(model.W1.^2,'all')+sum(model.W2.^2,'all'));
        dH=(dZ*model.W2').*(1-H.^2);
        g=struct('W1',X'*dH+lambda*model.W1,'b1',sum(dH,1), ...
            'W2',H'*dZ+lambda*model.W2,'b2',sum(dZ,1));
    otherwise
        error('a24:Architecture','Unknown architecture.');
end
loss=dataLoss+penalty;
aux=struct('P',P,'logits',Z,'normalized_weights',wn, ...
    'data_loss',dataLoss,'weight_penalty',penalty,'per_row_ce',ce, ...
    'row_logit_gradient',dZ);
assert(isfinite(loss) && all(isfinite(dZ),'all'), ...
    'a24:Nonfinite','Nonfinite CE loss or gradient.');
end
