function [P,pred,Z] = a24_ce_predict(model,X,mask)
% Same output ABI as r04nn_predict, with arbitrary nonempty seen-class masks.
Z=r04nn_logits(model,X);
P=a24_seen_softmax(Z,mask);
[~,pred]=max(P,[],2); % exact ties choose the smallest eligible global label
end
