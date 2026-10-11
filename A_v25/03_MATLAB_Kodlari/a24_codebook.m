function C = a24_codebook(kind)
% A v24 controlled code dictionaries: four labels, eight output bits.
% Original MATLAB adaptation; source and licence notes are in 01_Kaynaklar.
% Dictionary row = stable GLOBAL label (1..4), never a task-local label.
kind = lower(char(kind));
switch kind
    case {'onehot','onehot8','padded_onehot8'}
        C = [eye(4), zeros(4,4)];
    case {'walsh','walsh8','balanced_hadamard8'}
        H = 1;
        for j = 1:3
            H = [H,H; H,-H]; %#ok<AGROW>
        end
        C = (1-H(2:5,:))/2;
    otherwise
        error('a24:Codebook','Unknown dictionary: %s.',kind);
end
assert(isequal(size(C),[4,8]) && all(C(:)==0 | C(:)==1), ...
    'a24:Codebook','Expected a binary 4-by-8 dictionary.');
end
