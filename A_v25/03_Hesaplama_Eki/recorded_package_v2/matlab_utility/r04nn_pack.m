function [v,layout] = r04nn_pack(s,names)
% Pass model alone; pass gradient with explicit parameter names.
if nargin<2, names=r04nn_param_names(s); end
v=[]; layout=struct('name',{},'shape',{},'first',{},'last',{});
for j=1:numel(names)
    value=s.(names{j}); first=numel(v)+1; v=[v;value(:)]; %#ok<AGROW>
    layout(j)=struct('name',names{j},'shape',size(value),'first',first,'last',numel(v));
end
end
