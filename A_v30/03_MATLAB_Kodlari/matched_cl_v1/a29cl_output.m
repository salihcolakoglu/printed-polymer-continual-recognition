function out=a29cl_output(action,varargin)
% Bounded serialization: no output grows with 600 update histories.
% Bound counts every numerical leaf as 26 ASCII bytes plus array syntax.
% This covers finite IEEE binary64/uint64, booleans and explicit nulls.
switch action
 case 'bound',out=bound(varargin{1});
 case 'proof',out=proof();
 case 'check',out=check_result(varargin{1});
 case 'write'
  value=varargin{1};path=char(varargin{2});cap=varargin{3};
  assert(~isfile(path),'A29CL:OutputConsumed','No output overwrite.');
  receipt=check_result(value);txt=jsonencode(value,'ConvertInfAndNaN',true);bytes=unicode2native(txt,'UTF-8');
  assert(numel(bytes)<=receipt.actual_structure_upper_bytes&&numel(bytes)<=cap,'A29CL:SerializeBound','Serialized bytes exceed prospective bound.');
  f=fopen(path,'wb');assert(f>=0,'A29CL:Write','Cannot open exact output.');c=onCleanup(@()fclose(f));fwrite(f,bytes,'uint8');clear c
  out=struct('encoded_bytes',numel(bytes),'actual_structure_upper_bytes',receipt.actual_structure_upper_bytes,'prospective_result_upper_bytes',receipt.prospective_result_upper_bytes);
 otherwise,error('A29CL:OutputAction','Unknown output operation.');
end
end
function p=proof()
limits=struct('unique_fits',[315,65536],'logical_stages',[450,49152], ...
 'predictions',[9450,2048],'Fisher_jobs',[35,16384],'initial_states',[10,8192]);
names=fieldnames(limits);total=524288; % all other top-level fields and keys
for i=1:numel(names),x=limits.(names{i});total=total+x(1)*x(2)+x(1)+2;end
p=struct('schema','A29CL_RESULT_BYTE_PROOF_V1','limits',limits, ...
 'metadata_max_bytes',524288,'result_upper_bytes',total, ...
 'stdout_hard_cap_bytes',2*1024*1024,'auxiliary_files_reserve_bytes',256*1024, ...
 'aggregate_directory_upper_bytes',total+2*1024*1024+256*1024, ...
 'directory_cap_bytes',64*1024*1024, ...
 'scope','Hard schema/structural caps; oversize causes HOLD before resultfile write. ASCII text and bounded numeric/cardinality rules mandatory. Not proof that arbitrary future extensions fit.');
assert(p.aggregate_directory_upper_bytes<p.directory_cap_bytes,'A29CL:Proof','Insufficient output reserve.');
end
function out=check_result(value)
p=proof();names=fieldnames(p.limits);
assert(isstruct(value)&&isscalar(value)&&all(isfield(value,names)),'A29CL:ResultSchema','Incomplete bounded result schema.');
actual=bound(value);meta=rmfield(value,names);mb=bound(meta);
assert(mb<=p.metadata_max_bytes,'A29CL:MetadataCap','Top-level metadata exceeds declared structural cap.');
for i=1:numel(names)
 items=value.(names{i});lim=p.limits.(names{i});
 assert(iscell(items)&&numel(items)<=lim(1),'A29CL:CardinalityCap','Record count exceeds declared per-fold bound.');
 for j=1:numel(items)
  assert(bound(items{j})<=lim(2),'A29CL:RecordCap','Individual audit record exceeds declared bound.');
 end
end
assert(actual<=p.result_upper_bytes,'A29CL:ResultCap','Result structural bound failed.');
out=struct('actual_structure_upper_bytes',actual,'prospective_result_upper_bytes',p.result_upper_bytes,'metadata_upper_bytes',mb);
end
function n=bound(v)
% No jsonencode here: independent structural byte accounting.
if ischar(v)
 assert(isrow(v)||isempty(v),'A29CL:TextShape','Only character rows are serialized.');
 a=double(v);assert(all(a>=32&a<=126),'A29CL:TextDomain','Only printable ASCII in numerical audit artifacts.');
 n=2+numel(a)+sum(a==34|a==92);return
end
if isstring(v)
 if isscalar(v),n=bound(char(v));return;end
 n=bound(cellstr(v));return
end
if isnumeric(v)||islogical(v)
 assert(isreal(v)&&ndims(v)<=2,'A29CL:NumberDomain','Real scalar/vector/matrix only.');
 N=numel(v);if N==0,n=2;return;end
 width=26;if islogical(v),width=5;end
 if isscalar(v),n=width;elseif isvector(v),n=2+width*N+(N-1);else,r=size(v,1);n=2+2*r+width*N+(N-r)+(r-1);end
 return
end
if iscell(v)
 assert(isvector(v)||isempty(v),'A29CL:CellShape','Only vector cell arrays in bounded schema.');
 n=2+max(0,numel(v)-1);for j=1:numel(v),n=n+bound(v{j});end;return
end
if isstruct(v)
 if isempty(v),n=2;return;end
 assert(isvector(v),'A29CL:StructShape','Only vector struct arrays in bounded schema.');
 if ~isscalar(v),n=2+numel(v)-1;for j=1:numel(v),n=n+bound(v(j));end;return;end
 keys=fieldnames(v);n=2+max(0,numel(keys)-1);
 for j=1:numel(keys),n=n+bound(keys{j})+1+bound(v.(keys{j}));end
 return
end
error('A29CL:SerializationType','Unsupported serializer type.');
end
