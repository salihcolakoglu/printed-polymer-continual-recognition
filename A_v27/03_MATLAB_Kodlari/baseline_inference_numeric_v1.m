function y = baseline_inference_numeric_v1(u,p,mu,sd,arch)
%#codegen
% u: [E_MPa, ShoreD, candidate_a, candidate_b]. Pair is control, not a feature.
% p: fixed32 parameter slots. arch1 linear; arch2 two-four-four tanh.
% y: Xnorm2, logits4, probabilities4, margin, forced, selected, status.
% status1 selected,0 nonselection,-1 failure. True label never enters here.
y=nan(1,14); y(12:13)=0; y(14)=-1;
if any(~isfinite(u(:))) || any(~isfinite(p(:))) || ...
        any(~isfinite(mu(:))) || any(~isfinite(sd(:))) || any(sd(:)<=0)
    return;
end
a=u(3); b=u(4);
if a~=1 || b~=floor(b) || b<2 || b>4 || (arch~=1 && arch~=2)
    return;
end
x=[(u(1)-mu(1))/sd(1),(u(2)-mu(2))/sd(2)];
if arch==1
    W=reshape(p(1:8),2,4); bias=reshape(p(9:12),1,4);
    z=x*W+bias;
else
    W1=reshape(p(1:8),2,4); b1=reshape(p(9:12),1,4);
    W2=reshape(p(13:28),4,4); b2=reshape(p(29:32),1,4);
    z=tanh(x*W1+b1)*W2+b2;
end
if any(~isfinite(z)), return; end
m=max(z(a),z(b)); ea=exp(z(a)-m); eb=exp(z(b)-m);
P=zeros(1,4); P(a)=ea/(ea+eb); P(b)=eb/(ea+eb);
margin=abs(P(a)-P(b)); forced=a;
if P(b)>P(a), forced=b; end % accepted pairs ascending; tie -> smallest index
selected=0; status=0;
if margin>=0.15 && P(a)~=P(b), selected=forced; status=1; end
y=[x,z,P,margin,forced,selected,status];
end
