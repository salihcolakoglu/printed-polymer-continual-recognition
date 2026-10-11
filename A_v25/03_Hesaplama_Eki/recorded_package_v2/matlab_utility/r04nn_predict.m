function [P,pred,Z] = r04nn_predict(model,X,mask)
Z=r04nn_logits(model,X); P=r04nn_pair_softmax(Z,mask);
[~,pred]=max(P,[],2); % deterministic MATLAB lowest-index tie convention
end
