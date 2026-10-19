function varargout = a29_core_v1(action,varargin)
% A29_CORE_V1 Source-free numerical primitives. No file or source-data access.
% Linear tensors: W,mW,vW are 2x4; b,mb,vb are 1x4; t is uint64.
% Eligible classes: global ID vector or a 1x4 logical mask, common to all rows.
switch lower(char(action))
 case 'default_config'
  varargout{1}=struct('lambda',1e-4,'updates',600,'lr',.02, ...
   'beta1',.9,'beta2',.999,'epsilon',1e-8);
 case 'init_linear',varargout{1}=initLinear(varargin{:});
 case 'fit_normalizer',varargout{1}=fitNormalizer(varargin{:});
 case 'normalize',varargout{1}=normalizeRows(varargin{:});
 case 'lossgrad',[varargout{1:nargout}]=lossgrad(varargin{:});
 case 'adam',varargout{1}=adamStep(varargin{:});
 case 'train_linear',[varargout{1:nargout}]=trainLinear(varargin{:});
 case 'init_proto'
  varargout{1}=struct('sumS',zeros(4,2),'countn',zeros(1,4,'uint64'), ...
   'active',false(1,4));
 case 'admit_proto',varargout{1}=admitProto(varargin{:});
 case 'init_nn'
  varargout{1}=struct('X',zeros(0,2),'y',zeros(0,1),'row_ids',{{}}, ...
   'active',false(1,4));
 case 'admit_nn',varargout{1}=admitNN(varargin{:});
 case 'predict',varargout{1}=predict(varargin{:});
 case 'decode',varargout{1}=decode(varargin{:});
 case 'coverage',varargout{1}=coverage(varargin{:});
 otherwise,error('A29:Action','Unknown core action: %s',char(action));
end
end

function s=initLinear(seed)
validateattributes(seed,{'numeric'},{'scalar','integer','nonnegative','finite'});
rng(double(seed),'twister');
s=struct('W',(2*rand(2,4)-1)*sqrt(6/(2+4)),'b',zeros(1,4), ...
 'mW',zeros(2,4),'vW',zeros(2,4),'mb',zeros(1,4),'vb',zeros(1,4), ...
 't',uint64(0));
end

function n=fitNormalizer(X,row_ids)
validX(X);N=size(X,1);
assert(N>=2,'A29:NormalizerCount','At least two first-stage TRAIN rows required.');
ids=cellstr(string(row_ids(:)));
assert(numel(ids)==N && numel(unique(ids))==N,'A29:NormalizerIDs','Unique IDs must match rows.');
mu=sum(X,1)/N;sd=sqrt(sum((X-mu).^2,1)/(N-1));
assert(all(isfinite(mu)) && all(isfinite(sd)) && all(sd>0), ...
 'A29:NormalizerSTOP','Nonfinite or nonpositive sample SD: no replacement allowed.');
n=struct('mean',mu,'sd',sd,'n',N,'train_row_ids',{ids});
end

function Z=normalizeRows(X,n)
validX(X);
assert(isequal(size(n.mean),[1 2]) && isequal(size(n.sd),[1 2]) && ...
 all(isfinite(n.mean)) && all(isfinite(n.sd)) && all(n.sd>0), ...
 'A29:NormalizerSTOP','Invalid frozen normalizer.');
Z=(X-n.mean)./n.sd;
assert(all(isfinite(Z),'all'),'A29:NormalizeNonfinite','Nonfinite normalized inputs.');
end

function [loss,g,aux]=lossgrad(s,X,y,eligible,lambda)
validX(X);validLinear(s);c=classes(eligible);N=size(X,1);y=double(y(:));
assert(N>0 && numel(y)==N && all(ismember(y,c)), ...
 'A29:LossTargets','Targets must belong to the common eligible set.');
validateattributes(lambda,{'double'},{'scalar','finite','nonnegative'});
L=X*s.W+s.b;E=L(:,c);mx=max(E,[],2);ex=exp(E-mx);P=zeros(N,4);
P(:,c)=ex./sum(ex,2);lse=mx+log(sum(ex,2));
idx=sub2ind([N 4],(1:N)',y);
dataLoss=sum(lse-L(idx))/N;penalty=.5*lambda*sum(s.W.^2,'all');
D=P;D(idx)=D(idx)-1;D=D/N;
g=struct('dW',X'*D+lambda*s.W,'db',sum(D,1));loss=dataLoss+penalty;
assert(isfinite(loss) && all(isfinite(g.dW),'all') && all(isfinite(g.db)), ...
 'A29:LossNonfinite','Nonfinite loss or gradient.');
aux=struct('logits',L,'probabilities',P,'data_loss',dataLoss,'weight_penalty',penalty);
end

function s=adamStep(s,g,cfg)
validLinear(s);validCfg(cfg);
assert(isequal(size(g.dW),[2 4]) && isequal(size(g.db),[1 4]) && ...
 all(isfinite(g.dW),'all') && all(isfinite(g.db)), ...
 'A29:Gradient','Gradient shape or finiteness invalid.');
s.t=s.t+uint64(1);t=double(s.t);
s.mW=cfg.beta1*s.mW+(1-cfg.beta1)*g.dW;
s.vW=cfg.beta2*s.vW+(1-cfg.beta2)*g.dW.^2;
s.mb=cfg.beta1*s.mb+(1-cfg.beta1)*g.db;
s.vb=cfg.beta2*s.vb+(1-cfg.beta2)*g.db.^2;
s.W=s.W-cfg.lr*(s.mW/(1-cfg.beta1^t))./(sqrt(s.vW/(1-cfg.beta2^t))+cfg.epsilon);
s.b=s.b-cfg.lr*(s.mb/(1-cfg.beta1^t))./(sqrt(s.vb/(1-cfg.beta2^t))+cfg.epsilon);
validLinear(s);
end

function [s,tr]=trainLinear(s,X,y,eligible,cfg)
validCfg(cfg);assert(cfg.updates>=1 && cfg.updates==fix(cfg.updates), ...
 'A29:Updates','Positive integer update count required.');
timer=tic;[before,~,~]=lossgrad(s,X,y,eligible,cfg.lambda);
first=struct();last=struct();
for u=1:cfg.updates
 pre=s;[value,g]=lossgrad(pre,X,y,eligible,cfg.lambda);s=adamStep(pre,g,cfg);
 snap=struct('before',pre,'gradient',g,'loss',value,'after',s);
 if u==1,first=snap;end
 if u==cfg.updates,last=snap;end
end
[after,~,~]=lossgrad(s,X,y,eligible,cfg.lambda);
tr=struct('loss_before',before,'loss_after',after,'updates',cfg.updates, ...
 'gradient_examples',cfg.updates*size(X,1),'first_step',first, ...
 'last_step',last,'elapsed_seconds',toc(timer));
end

function s=admitProto(s,X,y)
validX(X);y=double(y(:));
assert(size(X,1)==numel(y) && all(ismember(y,1:4)), ...
 'A29:Admission','Invalid admission labels/row count.');
new=unique(y)';
assert(~any(s.active(new)),'A29:RepeatedClass','An active class may not be admitted again.');
for i=1:size(X,1)
 k=y(i);s.sumS(k,:)=s.sumS(k,:)+X(i,:);s.countn(k)=s.countn(k)+uint64(1);
end
s.active=s.countn>0;
assert(all(isfinite(s.sumS),'all'),'A29:ProtoNonfinite','Nonfinite sums.');
end

function s=admitNN(s,X,y,row_ids)
validX(X);y=double(y(:));ids=cellstr(string(row_ids(:)));
assert(size(X,1)==numel(y) && numel(y)==numel(ids) && all(ismember(y,1:4)), ...
 'A29:Admission','Invalid NN admission shapes/labels.');
assert(~any(s.active(unique(y))),'A29:RepeatedClass','An active class may not be admitted again.');
assert(numel(unique(ids))==numel(ids) && ~any(ismember(ids,s.row_ids)), ...
 'A29:RepeatedRow','NN references must be unique source parents.');
s.X=[s.X;X];s.y=[s.y;y];s.row_ids=[s.row_ids(:);ids];s.active(unique(y))=true;
end

function out=predict(kind,s,X,eligible)
assert(isa(X,'double') && ismatrix(X) && size(X,2)==2, ...
 'A29:PredictShape','Prediction inputs must be Nx2 binary64.');
c=classes(eligible);N=size(X,1);scores=nan(N,4);L=nan(N,4);D=nan(N,4);
ok=all(isfinite(X),2);kind=lower(char(kind));
switch kind
 case 'linear'
  good=all(isfinite(s.W),'all') && all(isfinite(s.b)) && ...
   isequal(size(s.W),[2 4]) && isequal(size(s.b),[1 4]);
  if good && any(ok)
   L(ok,:)=X(ok,:)*s.W+s.b;v=L(ok,c);v=v-max(v,[],2);ex=exp(v);
   scores(ok,c)=ex./sum(ex,2);
  end
 case 'proto'
  good=all(s.active(c)) && all(s.countn(c)>0) && all(isfinite(s.sumS),'all');
  if good
   for k=c
    center=s.sumS(k,:)/double(s.countn(k));D(ok,k)=sum((X(ok,:)-center).^2,2);
   end
   scores(:,c)=-D(:,c);
  end
 case 'nn'
  good=all(s.active(c)) && all(isfinite(s.X),'all') && all(ismember(c,s.y));
  if good
   q=find(ok)';
   for k=c
    refs=s.X(s.y==k,:);
    for i=q,D(i,k)=min(sum((refs-X(i,:)).^2,2));end
   end
   scores(:,c)=-D(:,c);
  end
 otherwise,error('A29:PredictKind','Unknown prediction kind.');
end
L(:,setdiff(1:4,c))=NaN;
out=decode(scores,c);out.scores=scores;out.logits=L;out.distances=D;
for i=1:N
 if ~ok(i),out.failure{i}='nonfinite_query';
 elseif ~good,out.failure{i}='nonfinite_state_or_missing_candidate';end
end
end

function out=decode(scores,eligible)
assert(isa(scores,'double') && ismatrix(scores) && size(scores,2)==4, ...
 'A29:DecodeShape','Scores must be Nx4 binary64.');
c=classes(eligible);N=size(scores,1);
out=struct('prediction',nan(N,1),'gap',nan(N,1),'tie',false(N,1), ...
 'status',{repmat({'failure'},N,1)},'failure',{repmat({'nonfinite_eligible_score'},N,1)});
for i=1:N
 v=scores(i,c);
 if any(~isfinite(v)),continue;end
 best=max(v);winners=c(v==best);vsort=sort(v,'descend');
 out.prediction(i)=min(winners);out.tie(i)=numel(winners)>1;
 out.gap(i)=vsort(1)-vsort(2);out.status{i}='valid';out.failure{i}='';
end
end

function cuts=coverage(rows)
assert(isstruct(rows) && ~isempty(rows),'A29:CoverageRows','Nonempty prediction structs required.');
N=numel(rows);g=[rows.gap];pred=[rows.prediction];truth=[rows.truth];
valid=strcmp({rows.status},'valid') & isfinite(g) & isfinite(pred);
thresholds=[Inf,sort(unique(g(valid)),'descend')];
cuts=struct('kind',{},'threshold',{},'C',{},'W',{},'A',{},'F',{}, ...
 'N',{},'answered',{},'coverage',{},'conditional_error',{},'total_unanswered',{});
for j=1:numel(thresholds)
 answered=valid & g>=thresholds(j);C=sum(answered & pred==truth);W=sum(answered & pred~=truth);
 A=sum(valid & ~answered);F=sum(~valid);nAnswered=C+W;err=NaN;
 if nAnswered>0,err=W/nAnswered;end
 kind='finite';if j==1,kind='empty';end
 cuts(j)=struct('kind',kind,'threshold',thresholds(j),'C',C,'W',W, ...
  'A',A,'F',F,'N',N,'answered',nAnswered,'coverage',nAnswered/N, ...
  'conditional_error',err,'total_unanswered',A+F);
 assert(C+W+A+F==N,'A29:CoverageCount','Disjoint cells must sum to N.');
end
end

function c=classes(v)
if islogical(v)
 assert(isequal(size(v),[1 4]),'A29:EligibleShape','Logical mask must be1x4.');c=find(v);
else
 assert(isnumeric(v) && isvector(v),'A29:EligibleShape','Eligible class IDs required.');
 c=double(v(:)');assert(numel(unique(c))==numel(c),'A29:EligibleDuplicate','Duplicate candidates.');
 c=sort(c);
end
assert(numel(c)>=2 && all(ismember(c,1:4)), ...
 'A29:Eligible','Two to four distinct global candidates required.');
end

function validX(X)
assert(isa(X,'double') && ismatrix(X) && size(X,2)==2 && all(isfinite(X),'all'), ...
 'A29:Input','Finite Nx2 binary64 input required.');
end

function validLinear(s)
fields={'W','b','mW','vW','mb','vb'};shapes={[2 4],[1 4],[2 4],[2 4],[1 4],[1 4]};
for j=1:numel(fields)
 v=s.(fields{j});assert(isa(v,'double') && isequal(size(v),shapes{j}) && all(isfinite(v),'all'), ...
  'A29:LinearState','Invalid linear tensor: %s',fields{j});
end
assert(isa(s.t,'uint64') && isscalar(s.t) && all(s.vW>=0,'all') && all(s.vb>=0), ...
 'A29:AdamState','Invalid optimizer step or squared moments.');
end

function validCfg(c)
need={'lambda','updates','lr','beta1','beta2','epsilon'};
assert(all(isfield(c,need)),'A29:Config','Complete configuration required.');
assert(isfinite(c.lr) && c.lr>0 && isfinite(c.epsilon) && c.epsilon>0 && ...
 c.beta1>=0 && c.beta1<1 && c.beta2>=0 && c.beta2<1 && ...
 isfinite(c.lambda) && c.lambda>=0,'A29:Config','Invalid optimizer configuration.');
end
