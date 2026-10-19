function out = a29cl_data(action,varargin)
% A29CL_DATA Declared data visibility, metadata replay and synthetic fixture.
% MIT: original A29 MATLAB implementation. No source training is performed here.
% Predictor payload is Z(:,1:2) only. Handles are software boundaries, not a
% security sandbox against a malicious caller inspecting MATLAB closures.
switch lower(char(action))
 case 'catalog'
  L=readlinks(varargin{1}); C=L.source_record_catalog; validate_catalog(C);
  out=struct('schema','A29_CATALOG_V1','rows',C,'normalizers',L.normalizers,'source_features_read',false);
 case 'prepare'
  out=prepare_source(varargin{1},varargin{2});
 case 'fixture'
  if isempty(varargin),out=fixture();else,out=fixture(varargin{1});end
 case 'controls'
  out=data_controls();
 case 'replay_empty'
  out=emptybuffer();
 case 'replay_view'
  B=varargin{1};cur=varargin{2};keep=~ismember(B.parentindex,cur.parentindex);
  out=slice_buffer(B,find(keep));
 case 'replay_commit'
  [out,receipt]=commit(varargin{1},varargin{2},varargin{3},varargin{4});
  out=struct('buffer',out,'receipt',receipt);
 case 'hash'
  out=sha_text(char(varargin{1}));
 otherwise
  error('A29:DataAction','Unknown data operation.');
end
end
function L=readlinks(x)
if isstruct(x),L=x;else,L=jsondecode(fileread(char(x)));end
assert(strcmp(L.schema,'A29_MATCHED_CL_INPUT_LINKS_V1'),'A29:InputSchema','Wrong input links.');
end
function validate_catalog(C)
assert(numel(C)==104,'A29:CatalogSize','Expected104 parents.');
assert(numel(unique(string({C.parentrow})))==104,'A29:DuplicateParent','Duplicate parent ID.');
a=[C.group_order]';y=[C.label]';f=[C.outer_fold]';
assert(isequal(a,repelem((0:25)',4))&&isequal(y,repmat((1:4)',26,1)),'A29:CatalogOrder','Expected group/label canonical order.');
assert(isequal(f,1+mod(a,5)),'A29:FoldMap','Fold mismatch.');
for g=0:25
 assert(numel(unique(string({C(a==g).group})))==1,'A29:SplitGroup','Group split.');
end
assert(numel(unique(string({C.group})))==26,'A29:GroupCount','26 source groups required.');
end
function out=prepare_source(csvPath,links)
L=readlinks(links);C=L.source_record_catalog;validate_catalog(C);
assert(endsWith(strrep(char(csvPath),'\','/'),'02_Veriler/A_source_average_records_v1.csv'),'A29:SourcePath','Only pinned source CSV accepted.');
P=L.pins;idx=find(strcmp({P.path},'02_Veriler/A_source_average_records_v1.csv'));
assert(isscalar(idx),'A29:SourcePin','Missing source pin.');
info=dir(csvPath);assert(isscalar(info)&&info.bytes==P(idx).bytes,'A29:SourceBytes','Source bytes mismatch.');
assert(strcmp(file_sha(csvPath),P(idx).sha256),'A29:SourceHash','Source hash mismatch.');
T=readtable(csvPath,'TextType','string','VariableNamingRule','preserve');
assert(height(T)==104,'A29:SourceRows','Source row count.');
assert(isequal(string(T.row_id),string({C.parentrow})'),'A29:SourceIDs','Source catalog ID order mismatch.');
assert(isequal(double(T.label),[C.label]')&&isequal(double(T.group_order),[C.group_order]'),'A29:SourceMetadata','Source label/group order mismatch.');
assert(isequal(string(T.condition_group),string({C.group})'),'A29:SourceGroups','Source group ID mismatch.');
X=double([T.E_MPa,T.Hardness_ShoreD]);assert(all(isfinite(X),'all'),'A29:NonfiniteX','Nonfinite source input.');
N=L.normalizers;normalizer_receipts=cell(5,1);
for f=1:5
 n=N(f);sel=[C.outer_fold]'~=f & [C.label]'<=2;
 assert(n.fold==f&&n.n==sum(sel),'A29:NormalizerN','Scaler membership/count mismatch.');
 assert(isequal(string(n.train_row_ids(:)),string({C(sel).parentrow})'),'A29:NormalizerIDs','Scaler TRAIN IDs mismatch.');
 mu=mean(X(sel,:),1);sd=std(X(sel,:),0,1);savedmu=reshape(n.mean,1,2);savedsd=reshape(n.sd,1,2);
 assert(all(isfinite(savedmu))&&all(isfinite(savedsd))&&all(savedsd>0),'A29:ScalerFinite','Invalid saved scaler.');
 assert(all(abs(mu-savedmu)<=1e-12*max(1,abs(savedmu)))&&all(abs(sd-savedsd)<=1e-12*max(1,abs(savedsd))),'A29:ScalerMismatch','Saved scaler recomputation mismatch.');
 normalizer_receipts{f}=struct('fold',f,'n',sum(sel),'mean_delta',mu-savedmu,'sd_delta',sd-savedsd,'saved_values_used',true);
end
receipt=struct('kind','SOURCE_ADMISSION_ONLY_NO_TRAINING','source_sha256',file_sha(csvPath),'source_rows',104,'source_groups',26,'normalizer_checks',{normalizer_receipts});
out=make_handles(C,X,N,false,receipt);
end
function out=fixture(options)
if nargin==0,options=struct;end
% Artificial values carry no inference about physical material distributions.
C=repmat(struct('parentrow','','group','','group_order',0,'label',0,'outer_fold',0),104,1);X=zeros(104,2);
for g=0:25
 for y=1:4
  j=4*g+y;C(j)=struct('parentrow',sprintf('ARTIFICIAL_G%02d_Y%d',g,y),'group',sprintf('ARTIFICIAL_GROUP_%02d',g),'group_order',g,'label',y,'outer_fold',1+mod(g,5));
  X(j,:)=[g/7+y/3,(-1)^y*(g+1)/9+y/5];
 end
end
if isfield(options,'current_point_shift'),X(6,:)=X(6,:)+double(options.current_point_shift);end
if isfield(options,'future_shift'),idx=[C.label]'==4;X(idx,:)=X(idx,:)+double(options.future_shift);end
if isfield(options,'heldout_shift'),idx=[C.outer_fold]'==1;X(idx,:)=X(idx,:)+double(options.heldout_shift);end
N=repmat(struct('mean',[],'sd',[],'n',0,'train_row_ids',{{}},'fold',0),5,1);
for f=1:5
 sel=[C.outer_fold]'~=f & [C.label]'<=2;N(f)=struct('mean',mean(X(sel,:),1),'sd',std(X(sel,:),0,1),'n',sum(sel),'train_row_ids',{{C(sel).parentrow}},'fold',f);
end
if isfield(options,'invalid_sd')&&options.invalid_sd,N(1).sd(1)=0;end
out=make_handles(C,X,N,true,struct('kind','ARTIFICIAL_FIXTURE','source_rows_read',0,'physical_validation',0));
end
function out=make_handles(C,X,N,isfixture,receipt)
% X is private to loader scope; the stage learner receives only returned Z.
out=struct('schema','A29_DATA_CAPABILITIES_V1','isfixture',isfixture,'catalog',C,'normalizers',N,'admission',receipt,'current',@current,'joint',@joint,'query',@query);
 function b=current(f,stream,s)
  check(f,stream,s);labs=current_labels(stream,s);
  rows=find([C.outer_fold]'~=f & ismember([C.label]',labs));
  b=batch(rows,f);b.current_labels=labs;b.stage=s;b.stream=char(stream);
  b.eligible=curmask(stream,s);b.kind='CURRENT_TRAIN';
 end
 function b=joint(f,s)
  check(f,'ClassIL',s);rows=find([C.outer_fold]'~=f & [C.label]'<=s+1);
  b=batch(rows,f);b.current_labels=1:s+1;b.stage=s;b.stream='JOINT_GLOBAL';b.eligible=1:s+1;b.kind='JOINT_TRAIN_UNIQUE_PARENTS';
 end
 function b=query(f,stream,s)
  check(f,stream,s);rows=[];tasks=[];
  if strcmp(stream,'TaskIL')
   for t=1:s
    r=find([C.outer_fold]'==f & ismember([C.label]',[1,t+1]));rows=[rows;r];tasks=[tasks;repmat(t,numel(r),1)]; %#ok<AGROW>
   end
  else
   rows=find([C.outer_fold]'==f & [C.label]'<=s+1);tasks=zeros(numel(rows),1);
  end
  b=batch(rows,f);b.task=tasks;b.stage=s;b.stream=char(stream);b.kind='EVALUATION_ONLY';
 end
 function b=batch(rows,f)
  rows=rows(:);mu=reshape(N(f).mean,1,2);sd=reshape(N(f).sd,1,2);
  assert(all(isfinite(mu))&&all(isfinite(sd))&&all(sd>0),'A29:DataScaler','Invalid frozen scaler; no fallback permitted.');
  b=struct('Z',(X(rows,:)-mu)./sd,'y',[C(rows).label]','parentindex',uint64(rows),'parentrow',{{C(rows).parentrow}},'group_order',[C(rows).group_order]','outer_fold',[C(rows).outer_fold]');
 end
end
function check(f,stream,s)
assert(isscalar(f)&&ismember(f,1:5)&&isscalar(s)&&ismember(s,1:3),'A29:FoldStage','Invalid fold/stage.');
assert(any(strcmp(stream,{'TaskIL','ClassIL'})),'A29:Stream','Invalid stream.');
end
function labs=current_labels(stream,s)
if strcmp(stream,'TaskIL'),labs=[1,s+1];elseif s==1,labs=[1,2];else,labs=s+1;end
end
function labs=curmask(stream,s)
if strcmp(stream,'TaskIL'),labs=[1,s+1];else,labs=1:s+1;end
end
function B=emptybuffer()
B=struct('Z',zeros(0,2),'y',zeros(0,1),'firsttask',zeros(0,1,'uint64'),'parentindex',zeros(0,1,'uint64'),'parentrow',{{}},'selection_hash',{{}});
end
function out=slice_buffer(B,k)
out=B;out.Z=B.Z(k,:);out.y=B.y(k);out.firsttask=B.firsttask(k);out.parentindex=B.parentindex(k);out.parentrow=B.parentrow(k);out.selection_hash=B.selection_hash(k);
end
function [B,r]=commit(B,cur,fold,stage)
assert(strcmp(cur.kind,'CURRENT_TRAIN'),'A29:ReplaySource','Only current TRAIN may admit replay.');
oldclasses=unique(B.y);newclasses=setdiff(unique(cur.y),oldclasses);r=struct('admitted',{{}},'retained_previous_classes',oldclasses','hash_policy','A29_REPLAY_V1|fold|label|exactparentrow','source','CURRENT_FIRST_ARRIVAL_ONLY');
for y=reshape(newclasses,1,[])
 expected=1;if y>=3,expected=y-1;end
 assert(stage==expected,'A29:ReplayFirstArrival','Missing earlier class memory; old class cannot be reselected.');
 idx=find(cur.y==y);assert(numel(idx)>=2,'A29:ReplayQuota','Insufficient current parents.');keys=strings(numel(idx),2);
 for j=1:numel(idx)
  keys(j,1)=sha_text(sprintf('A29_REPLAY_V1|%d|%d|%s',fold,y,cur.parentrow{idx(j)}));keys(j,2)=string(cur.parentrow{idx(j)});
 end
 [~,order]=sortrows(keys,[1,2]);chosen=idx(order(1:2));
 for j=1:2
  k=chosen(j);hh=char(keys(order(j),1));B.Z(end+1,:)=cur.Z(k,:);B.y(end+1,1)=y;B.firsttask(end+1,1)=uint64(expected);B.parentindex(end+1,1)=cur.parentindex(k);B.parentrow{end+1,1}=cur.parentrow{k};B.selection_hash{end+1,1}=hh;
  r.admitted{end+1,1}=struct('label',y,'parentindex',cur.parentindex(k),'parentrow',cur.parentrow{k},'selection_hash',hh,'firsttask',expected);
 end
end
assert(numel(unique(B.parentindex))==numel(B.y)&&numel(B.y)==2*(stage+1)&&numel(B.y)<=8,'A29:ReplayInvariant','Replay uniqueness/capacity/count failure.');
r.post_stage_size=numel(B.y);r.numeric_payload_bytes=numel(B.y)*40;
end
function h=sha_text(s)
h=sha_bytes(unicode2native(s,'UTF-8'));
end
function h=file_sha(path)
f=fopen(path,'rb');assert(f>=0,'A29:ReadHash','Cannot read pinned file.');c=onCleanup(@()fclose(f));b=fread(f,Inf,'*uint8');h=sha_bytes(b); %#ok<NASGU>
end
function h=sha_bytes(b)
md=java.security.MessageDigest.getInstance('SHA-256');md.update(typecast(uint8(b(:)),'int8'));h=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2)',1,[]));
end

function r=data_controls()
% Source-free counterfactual checks exercise real data/replay functions.
D=fixture();future=fixture(struct('future_shift',1000));held=fixture(struct('heldout_shift',500));bad=fixture(struct('invalid_sd',true));
cases={};C=D.catalog;validate_catalog(C);
for f=1:5
 cur=D.current(f,'TaskIL',1);query=D.query(f,'ClassIL',3);
 assert(isempty(intersect(cur.parentindex,query.parentindex)),'A29:ControlHeldout','TRAIN/heldout parent overlap.');
 assert(all(cur.outer_fold~=f)&&all(query.outer_fold==f),'A29:ControlFold','Fold isolation.');
 assert(all(ismember(cur.y,[1,2]))&&numel(cur.y)==D.normalizers(f).n,'A29:ControlNormalizer','Stage1 membership.');
 assert(max(abs(mean(cur.Z,1)))<1e-12&&max(abs(std(cur.Z,0,1)-1))<1e-12,'A29:ControlNormalizer','Stage1-only standardization.');
 cases{end+1}=struct('name',sprintf('fold%d_wholegroup_scaler_membership',f),'pass',true,'train_stage1',numel(cur.y),'heldout_parents',numel(query.y));
end
for sn={'TaskIL','ClassIL'}
 stream=sn{1};B=emptybuffer();Bp=emptybuffer();effective=zeros(1,3);post=zeros(1,3);
 for stage=1:3
  cur=D.current(1,stream,stage);mutated=held.current(1,stream,stage);
  assert(isequaln(cur,mutated),'A29:ControlHeldMutation','Heldout feature perturbation reached current loader.');
  if stage<3
   fut=future.current(1,stream,stage);assert(isequaln(cur,fut),'A29:ControlFutureMutation','Unintroduced class4 features reached current loader.');
  end
  view=a29cl_data('replay_view',B,cur);effective(stage)=numel(view.y);
  assert(isempty(intersect(view.parentindex,cur.parentindex)),'A29:ControlReplayDedup','Memory/current duplicated parent.');
  [B,admit]=commit(B,cur,1,stage);post(stage)=numel(B.y);
  curp=future.current(1,stream,stage);[Bp,admitp]=commit(Bp,curp,1,stage);
  assert(isequal(B.parentindex,Bp.parentindex)&&isequal(B.selection_hash,Bp.selection_hash)&&isequal(admit,admitp),'A29:ControlFeatureSelection','Replay selection depends on features.');
  assert(all(B.firsttask==uint64(max(1,B.y-1))),'A29:ControlReplayTag','Wrong first-admission task tag.');
 end
 want=[0,2,4];if strcmp(stream,'ClassIL'),want=[0,4,6];end
 assert(isequal(effective,want)&&isequal(post,[4,6,8]),'A29:ControlReplayCounts','Replay cardinality mismatch.');
 cases{end+1}=struct('name',[stream,'_current_only_counterfactual_and_replay'],'pass',true,'effective',effective,'post',post,'feature_independent_hash_selection',true);
end
q=D.query(1,'ClassIL',3);qh=held.query(1,'ClassIL',3);qf=future.query(1,'ClassIL',3);
assert(~isequal(q.Z,qh.Z)&&~isequal(q.Z(q.y==4,:),qf.Z(qf.y==4,:)),'A29:ControlPerturbation','Counterfactual perturbation had no effect anywhere.');
assert(isequal(q.Z(q.y~=4,:),qf.Z(qf.y~=4,:)),'A29:ControlFutureSupport','Future perturbation changed earlier classes.');
cases{end+1}=struct('name','counterfactual_operands_effective_only_in_declared_scope','pass',true,'heldout_shift',500,'future_class4_shift',1000);
failed=false;try,bad.current(1,'TaskIL',1);catch e,failed=strcmp(e.identifier,'A29:DataScaler');end
assert(failed,'A29:ControlScalerFail','ZeroSD did not fail closed.');
cases{end+1}=struct('name','invalid_saved_scaler_failstop','pass',true,'zero_sd_rejected',failed);
B=emptybuffer();cur=D.current(1,'TaskIL',1);[B,~]=commit(B,cur,1,1);again=B;[again,~]=commit(again,D.current(1,'TaskIL',2),1,2);
assert(isequal(B.parentindex(1:4),again.parentindex(1:4)),'A29:ControlNoReselection','Returning anchor memory changed.');
cases{end+1}=struct('name','old_anchor_not_reselected','pass',true,'retained_old_parent_count',4);
Cbad=C;Cbad(2).outer_fold=5;failed=false;try,validate_catalog(Cbad);catch e,failed=strcmp(e.identifier,'A29:FoldMap');end
assert(failed,'A29:ControlCatalogFail','Split group catalog accepted.');
cases{end+1}=struct('name','invalid_fold_group_catalog_rejected','pass',true,'rejected',failed);
changed=fixture(struct('current_point_shift',50));basecur=D.current(1,'TaskIL',1);newcur=changed.current(1,'TaskIL',1);
assert(~isequal(basecur.Z,newcur.Z),'A29:ControlCurrentSignal','A current TRAIN perturbation had no effect.');
cases{end+1}=struct('name','current_train_perturbation_visible','pass',true,'point_shift',50);
cur=basecur;perm=numel(cur.y):-1:1;permuted=cur;permuted.Z=cur.Z(perm,:);permuted.y=cur.y(perm);permuted.parentindex=cur.parentindex(perm);permuted.parentrow=cur.parentrow(perm);permuted.group_order=cur.group_order(perm);permuted.outer_fold=cur.outer_fold(perm);
[A,~]=commit(emptybuffer(),cur,1,1);[B,~]=commit(emptybuffer(),permuted,1,1);
assert(isequaln(A,B),'A29:ControlHashOrder','Replay selected payload depends on current input iteration order.');
cases{end+1}=struct('name','replay_hash_ranking_permutation_invariant','pass',true,'reversed_current_rows',numel(perm));
k=find(cur.y==1,1);tiny=cur;tiny.Z=cur.Z(k,:);tiny.y=cur.y(k);tiny.parentindex=cur.parentindex(k);tiny.parentrow=cur.parentrow(k);tiny.group_order=cur.group_order(k);tiny.outer_fold=cur.outer_fold(k);
failed=false;try,commit(emptybuffer(),tiny,1,1);catch e,failed=strcmp(e.identifier,'A29:ReplayQuota');end
assert(failed,'A29:ControlQuotaFail','Replay with fewer than2currentparents accepted.');
cases{end+1}=struct('name','replay_insufficient_quota_rejected','pass',true,'available_parents',1,'required_parents',2);
r=struct('schema','A29_DATA_CONTROLS_V1','status','PASS','case_count',numel(cases),'cases',{cases},'source_rows_read',0,'source_training',0,'meaning','Artificial code/data-interface checks only; not source experiment or physical validation.');
end
