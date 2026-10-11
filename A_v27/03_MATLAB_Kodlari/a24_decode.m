function [forced,distances,rawgap,tieflag] = a24_decode(P,C,eligible)
% Nearest code by mean squared Euclidean distance; NOT probabilities.
% eligible: numeric label vector (1..4), logical 1-by-4 mask, or N-by-4 mask.
% Inactive distances are NaN for JSON-null serialization; internal Inf is
% used only to exclude labels from minimization. Exact ties choose the
% smallest eligible GLOBAL label. rawgap is second-best minus best distance.
% A singleton eligible set has rawgap=NaN; no selection threshold is applied.
validateattributes(P,{'double'},{'2d','real','finite','>=',0,'<=',1});
validateattributes(C,{'double'},{'size',[4,8],'real','finite','>=',0,'<=',1});
assert(size(P,2)==8 && size(P,1)>0, ...
    'a24:Dimensions','P must be nonempty N-by-8.');
N = size(P,1);
if islogical(eligible)
    if isequal(size(eligible),[1,4])
        mask = repmat(eligible,N,1);
    else
        assert(isequal(size(eligible),[N,4]), ...
            'a24:Mask','Logical eligibility must be 1-by-4 or N-by-4.');
        mask = eligible;
    end
else
    validateattributes(eligible,{'numeric'}, ...
        {'vector','nonempty','real','finite','integer','>=',1,'<=',4});
    assert(numel(unique(eligible))==numel(eligible), ...
        'a24:Mask','Eligible labels must be unique.');
    mask = false(N,4);
    mask(:,double(eligible(:))) = true;
end
assert(all(any(mask,2)),'a24:Mask','Every query needs an eligible label.');
allDistances = zeros(N,4);
for k = 1:4
    allDistances(:,k) = mean((P-C(k,:)).^2,2);
end
masked = allDistances;
masked(~mask) = Inf;
[best,forced] = min(masked,[],2);
tieflag = sum(mask & (masked==best),2)>1;
ordered = sort(masked,2,'ascend');
rawgap = ordered(:,2)-ordered(:,1);
rawgap(sum(mask,2)==1) = NaN;
distances = allDistances;
distances(~mask) = NaN;
end
