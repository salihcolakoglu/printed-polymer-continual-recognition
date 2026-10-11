function [P,logden,maxlogit] = a24_seen_softmax(Z,mask)
% Stable softmax over each row's eligible seen classes.
% Unlike the preserved r04nn_pair_softmax, this supports 2/3/4-class masks.
% A nonempty singleton mask is also mathematically defined.
validateattributes(Z,{'double'},{'2d','real','finite','nonempty'});
assert(islogical(mask) && isequal(size(mask),size(Z)), ...
    'a24:Mask','Mask must be logical and match N-by-K logits.');
assert(all(any(mask,2)),'a24:Mask','Every row needs an eligible class.');
masked = Z; masked(~mask)=-Inf;
maxlogit = max(masked,[],2);
ex = exp(masked-maxlogit);
denominator = sum(ex,2);
P = ex./denominator;
logden = log(denominator);
assert(all(isfinite(P),'all') && all(isfinite(logden)), ...
    'a24:Nonfinite','Nonfinite masked softmax output.');
end
