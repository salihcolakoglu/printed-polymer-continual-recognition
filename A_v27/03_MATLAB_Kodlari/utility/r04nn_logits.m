function [Z,H] = r04nn_logits(model,X)
validateattributes(X,{'double'},{'2d','real','finite'});
assert(size(X,2)==model.d,'r04nn:Dimensions','X must be N-by-d.');
assert(size(X,1)>0,'r04nn:Dimensions','Empty batches are invalid.');
names=r04nn_param_names(model);
for j=1:numel(names), validateattributes(model.(names{j}),{'double'},{'real','finite'}); end
switch model.arch
    case 'linear'
        assert(isequal(size(model.W),[model.d,model.K]) && isequal(size(model.b),[1,model.K]),'r04nn:Dimensions','Invalid linear parameters.');
        H=[]; Z=X*model.W+model.b;
    case 'mlp_tanh'
        assert(isequal(size(model.W1),[model.d,model.h]) && isequal(size(model.b1),[1,model.h]) && isequal(size(model.W2),[model.h,model.K]) && isequal(size(model.b2),[1,model.K]),'r04nn:Dimensions','Invalid MLP parameters.');
        H=tanh(X*model.W1+model.b1); Z=H*model.W2+model.b2;
end
assert(all(isfinite(Z),'all'),'r04nn:Nonfinite','Nonfinite logits.');
end
