function [loss,g,aux] = r04nn_lossgrad(model,X,Y,mask,w,lambda)
% Weighted per-row task-masked CE + .5*lambda*sum(weight matrices.^2).
% Input weights are normalized once over the COMPLETE supplied batch.
[Z,H]=r04nn_logits(model,X); [P,logden,mx]=r04nn_pair_softmax(Z,mask);
N=size(X,1); K=model.K;
validateattributes(Y,{'numeric'},{'column','numel',N,'integer','>=',1,'<=',K});
validateattributes(w,{'double'},{'column','numel',N,'real','finite','nonnegative'});
validateattributes(lambda,{'double'},{'scalar','real','finite','nonnegative'});
idx=sub2ind([N,K],(1:N)',double(Y));
assert(all(mask(idx)),'r04nn:TargetMask','Every target must be allowed by its own row mask.');
weightMass=sum(w); assert(isfinite(weightMass) && weightMass>0,'r04nn:Weights','Weight mass must be positive and finite.');
w=w/weightMass; ce=logden+mx-Z(idx); dataLoss=sum(w.*ce);
dZ=P; dZ(idx)=dZ(idx)-1; dZ=dZ.*w;
switch model.arch
    case 'linear'
        penalty=.5*lambda*sum(model.W.^2,'all');
        g=struct('W',X'*dZ+lambda*model.W,'b',sum(dZ,1));
    case 'mlp_tanh'
        penalty=.5*lambda*(sum(model.W1.^2,'all')+sum(model.W2.^2,'all'));
        dH=(dZ*model.W2').*(1-H.^2);
        g=struct('W1',X'*dH+lambda*model.W1,'b1',sum(dH,1), ...
                 'W2',H'*dZ+lambda*model.W2,'b2',sum(dZ,1));
end
loss=dataLoss+penalty;
aux=struct('P',P,'logits',Z,'normalized_weights',w,'data_loss',dataLoss, ...
           'weight_penalty',penalty,'per_row_ce',ce,'row_logit_gradient',dZ);
assert(isfinite(loss),'r04nn:Nonfinite','Nonfinite loss.');
end
