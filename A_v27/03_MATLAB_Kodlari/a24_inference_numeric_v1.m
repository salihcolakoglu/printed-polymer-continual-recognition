function y = a24_inference_numeric_v1(u,p,mu,sd,C,kind)
%#codegen
% Frozen A24 recognition. u=[E,ShoreD,eligible1..4]; kind1=code,2=CE.
% Fixed52 weight slots, codebook4x8. No truth/row/model identity input.
% y18=[normalized2,activation8,active_scores4,gap,forced,tie,status].
% Scores: code=-mean squared distance; CE=eligible softmax.
% CE unused activation slots and inactive class scores are NaN.
y=nan(1,18);y(16)=0;y(17)=0;y(18)=-1;
u=reshape(u,1,6);p=reshape(p,52,1);mu=reshape(mu,1,2);sd=reshape(sd,1,2);
if any(~isfinite([u,p',mu,sd])) || any(sd<=0) || ~(kind==1 || kind==2)
 return;
end
mask=u(3:6);
if any(~(mask==0 | mask==1)) || sum(mask)<2,return;end
if kind==1 && (any(~isfinite(C(:))) || any(~(C(:)==0 | C(:)==1)))
 return;
end
x=(u(1:2)-mu)./sd;
h=tanh(x*reshape(p(1:8),2,4)+reshape(p(9:12),1,4));
activation=nan(1,8);score=-inf(1,4);
if kind==1
 logits8=h*reshape(p(13:44),4,8)+reshape(p(45:52),1,8);
 if any(~isfinite(logits8)),return;end
 for b=1:8
  if logits8(b)>=0
   activation(b)=1/(1+exp(-logits8(b)));
  else
   eb=exp(logits8(b));activation(b)=eb/(1+eb);
  end
 end
 for k=1:4
  if mask(k)==1,score(k)=-sum((activation-C(k,:)).^2)/8;end
 end
else
 % CE uses its original contiguous32 slots; slots33..52 are unused zeros.
 logits4=h*reshape(p(13:28),4,4)+reshape(p(29:32),1,4);
 if any(~isfinite(logits4)),return;end
 top=-inf;
 for k=1:4,if mask(k)==1 && logits4(k)>top,top=logits4(k);end,end
 ex=zeros(1,4);
 for k=1:4,if mask(k)==1,ex(k)=exp(logits4(k)-top);end,end
 probability=ex/sum(ex);activation(1:4)=probability;
 for k=1:4,if mask(k)==1,score(k)=probability(k);end,end
end
best=-inf;second=-inf;forced=0;
for k=1:4
 if mask(k)==1
  if score(k)>best
   second=best;best=score(k);forced=k;
  elseif score(k)>second
   second=score(k);
  end
 end
end
tieCount=0;
for k=1:4,if mask(k)==1 && score(k)==best,tieCount=tieCount+1;end,end
if ~isfinite(best) || ~isfinite(second) || any(~isfinite(x)),return;end
y(1:2)=x;y(3:10)=activation;
for k=1:4,if mask(k)==1,y(10+k)=score(k);end,end
y(15)=best-second;y(16)=forced;y(17)=double(tieCount>1);y(18)=1;
end
