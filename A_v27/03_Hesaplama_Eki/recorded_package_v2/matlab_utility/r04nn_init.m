function model = r04nn_init(arch,d,K,h,seed)
% Scoped rng(seed,'twister') Xavier uniform; restores caller RNG state.
validateattributes(d,{'numeric'},{'scalar','integer','positive'});
validateattributes(K,{'numeric'},{'scalar','integer','>=',2});
validateattributes(seed,{'numeric'},{'scalar','integer','nonnegative'});
arch=char(arch); oldRng=rng; restoreRng=onCleanup(@()rng(oldRng)); %#ok<NASGU>
rng(seed,'twister');
model=struct('arch',arch,'d',double(d),'K',double(K),'h',double(h));
switch arch
    case 'linear'
        assert(h==0,'r04nn:Architecture','Linear architecture requires h=0.');
        limit=sqrt(6/(d+K)); model.W=(2*rand(d,K)-1)*limit; model.b=zeros(1,K);
    case 'mlp_tanh'
        validateattributes(h,{'numeric'},{'scalar','integer','positive'});
        limit1=sqrt(6/(d+h)); limit2=sqrt(6/(h+K));
        model.W1=(2*rand(d,h)-1)*limit1; model.b1=zeros(1,h);
        model.W2=(2*rand(h,K)-1)*limit2; model.b2=zeros(1,K);
    otherwise
        error('r04nn:Architecture','Unknown architecture.');
end
end
