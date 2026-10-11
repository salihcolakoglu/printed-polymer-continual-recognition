function [P,logden,maxlogit] = r04nn_pair_softmax(Z,mask)
% Exactly two allowed classes PER ROW. Excluded logits have zero probability.
validateattributes(Z,{'double'},{'2d','real','finite'});
assert(islogical(mask) && isequal(size(mask),size(Z)),'r04nn:Mask','Mask must be logical and match N-by-K logits.');
assert(all(sum(mask,2)==2),'r04nn:Mask','Each row must allow exactly two classes.');
Zm=Z; Zm(~mask)=-Inf; maxlogit=max(Zm,[],2);
ex=exp(Zm-maxlogit); denominator=sum(ex,2);
P=ex./denominator; logden=log(denominator);
end
