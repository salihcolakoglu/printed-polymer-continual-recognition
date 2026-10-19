function y = a30_inference_chart_v1(u,theta,meta,mu,sd,refs,books)
%#codegen
% Independent inference only. Tick indexes queries, not physical time.
% u=[raw E, raw ShoreD, logical model selector, four eligibility controls].
% meta columns: kind(1 LIN,2 CE,3 CODE,4 PROTO,5 NN1),D,book,snapshot,nref.
% No truth, parent identity, recorded prediction or optimizer enters chart.
y=nan(1,35);
y(30)=0; y(31:34)=u(4:7); y(35)=u(3);
mid=u(3);
if ~isfinite(mid) || mid~=floor(mid) || mid<1 || mid>size(theta,1),return;end
for j=1:4
 if ~isfinite(u(j+3)) || ~(u(j+3)==0 || u(j+3)==1),return;end
end
if sum(u(4:7))<2,return;end
for j=1:2
 if ~isfinite(u(j)) || ~isfinite(mu(j)) || ~isfinite(sd(j)) || sd(j)<=0,return;end
 y(j)=(u(j)-mu(j))/sd(j);
 if ~isfinite(y(j)),return;end
end
k=meta(mid,1); D=meta(mid,2); book=meta(mid,3);
if k<1 || k>5 || k~=floor(k),return;end
if k==1 || k==4,plen=12;elseif k==2,plen=32;elseif k==3,plen=52;else,plen=0;end
for j=1:plen,if ~isfinite(theta(mid,j)),return;end,end
score=nan(1,4); logit=nan(1,8); output=nan(1,8); dist=nan(1,4);
if k==1
 for c=1:4
  if u(c+3)==1
   logit(c)=y(1)*theta(mid,2*c-1)+y(2)*theta(mid,2*c)+theta(mid,8+c);
  end
 end
elseif k==2 || k==3
 if (k==2 && D~=4) || (k==3 && D~=8),return;end
 h=zeros(1,4);
 for j=1:4,h(j)=tanh(y(1)*theta(mid,2*j-1)+y(2)*theta(mid,2*j)+theta(mid,8+j));end
 for c=1:8
  if c<=D
   a=0;
   for j=1:4,a=a+h(j)*theta(mid,12+4*(c-1)+j);end
   logit(c)=a+theta(mid,12+4*D+c);
   if ~isfinite(logit(c)),return;end
  end
 end
end
if k==1 || k==2
 mx=-Inf;
 for c=1:4,if u(c+3)==1,mx=max(mx,logit(c));end,end
 denom=0;
 for c=1:4,if u(c+3)==1,denom=denom+exp(logit(c)-mx);end,end
 for c=1:4
  if u(c+3)==1
   if k==2,score(c)=exp((logit(c)-mx)-log(denom));else,score(c)=exp(logit(c)-mx)/denom;end
   output(c)=score(c);
  end
 end
elseif k==3
 if book~=floor(book) || book<1 || book>2,return;end
 for j=1:8
  if logit(j)>=0,output(j)=1/(1+exp(-logit(j)));else,e=exp(logit(j));output(j)=e/(1+e);end
 end
 for c=1:4
  if u(c+3)==1
   d=0;
   for j=1:8,b=books((book-1)*4+c,j);if ~(b==0 || b==1),return;end,d=d+(output(j)-b)^2;end
   dist(c)=d/8;score(c)=-dist(c);
  end
 end
elseif k==4
 for c=1:4
  if u(c+3)==1
   n=theta(mid,8+c);if n<=0 || n~=floor(n),return;end
   a=y(1)-theta(mid,c)/n;b=y(2)-theta(mid,4+c)/n;
   dist(c)=a*a+b*b;score(c)=-dist(c);
  end
 end
elseif k==5
 snap=meta(mid,4); n=meta(mid,5);
 if snap~=floor(snap) || snap<1 || snap>3 || n~=floor(n) || n<1 || n>84,return;end
 % Reject the entire malformed retained bank, never omit a bad reference.
 for j=1:84
  if j<=n
   ix=(snap-1)*84+j;
   if ~isfinite(refs(ix,1)) || ~isfinite(refs(ix,2)) || ~isfinite(refs(ix,3)) || refs(ix,3)~=floor(refs(ix,3)) || refs(ix,3)<1 || refs(ix,3)>4,return;end
  end
 end
 for c=1:4
  if u(c+3)==1
   d=Inf;
   for j=1:84
    if j<=n
     ix=(snap-1)*84+j;
     if refs(ix,3)==c
      a=y(1)-refs(ix,1);b=y(2)-refs(ix,2);d=min(d,a*a+b*b);
     end
    end
   end
   dist(c)=d;score(c)=-d;
  end
 end
end
best=-Inf; second=-Inf; label=NaN; ties=0;
for c=1:4
 if u(c+3)==1
  v=score(c);if ~isfinite(v),return;end
  if v>best,second=best;best=v;label=c;ties=1;
  elseif v==best,second=best;ties=ties+1;
  elseif v>second,second=v;end
 end
end
y(3:10)=logit;y(11:18)=output;y(19:22)=score;y(23:26)=dist;
y(27)=label;y(28)=best-second;y(29)=double(ties>1);y(30)=1;
end
