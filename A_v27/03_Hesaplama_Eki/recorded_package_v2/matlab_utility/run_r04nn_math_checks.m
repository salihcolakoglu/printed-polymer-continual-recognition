function receipt = run_r04nn_math_checks(outdir)
% New toy-only checks. No downloaded source records are loaded.
if nargin<1, outdir=fileparts(mfilename('fullpath')); end
checks=struct('name',{},'pass',{},'value',{},'threshold',{});
receipt=struct('status','RUNNING','scope','generic toy mathematical implementation proof only', ...
    'source_training_runs',0,'source_files_loaded',0,'physical_validation',false, ...
    'matlab_version',version,'started_at_utc',char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss.SSSXXX')), ...
    'checks',checks);
try
    s=RandStream('mt19937ar','Seed',271828); X=randn(s,7,3);
    pairs=[1 2;2 3;4 5;1 4;2 5;3 4;1 5]; mask=false(7,5);
    for n=1:7, mask(n,pairs(n,:))=true; end
    Y=[1;3;5;4;2;3;5]; w=[1;2;0;3;4;2;1]; w=w/sum(w);
    arches={'linear','mlp_tanh'};
    for a=1:2
        if a==1, h=0; else, h=4; end
        m=r04nn_init(arches{a},3,5,h,271829+a);
        [~,g]=r04nn_lossgrad(m,X,Y,mask,w,.07);
        theta=r04nn_pack(m); ga=r04nn_pack(g,r04nn_param_names(m)); gn=zeros(size(theta)); epsFD=1e-6;
        for j=1:numel(theta)
            p=theta; p(j)=p(j)+epsFD; q=theta; q(j)=q(j)-epsFD;
            lp=r04nn_lossgrad(r04nn_unpack(m,p),X,Y,mask,w,.07);
            lq=r04nn_lossgrad(r04nn_unpack(m,q),X,Y,mask,w,.07); gn(j)=(lp-lq)/(2*epsFD);
        end
        addcheck(['mixed_mask_regularized_finite_difference_' arches{a}],max(abs(ga-gn)),2e-7);
        [l1,g1]=r04nn_lossgrad(m,X,Y,mask,w,.07);
        [l2,g2]=r04nn_lossgrad(m,X,Y,mask,17*w,.07);
        [l3,g3]=r04nn_lossgrad(m,[X;X],[Y;Y],[mask;mask],[w/2;w/2],.07);
        v1=r04nn_pack(g1,r04nn_param_names(m)); v2=r04nn_pack(g2,r04nn_param_names(m)); v3=r04nn_pack(g3,r04nn_param_names(m));
        addcheck(['weight_scale_and_duplication_invariance_' arches{a}],max([abs(l1-l2);abs(l1-l3);abs(v1-v2);abs(v1-v3)]),1e-12);
        mask2=false(7,5); mask2(:,1:2)=true; Y2=[1;2;1;2;1;2;1];
        [~,gx]=r04nn_lossgrad(m,X,Y2,mask2,w,0.0);
        if a==1, excluded=[reshape(gx.W(:,3:5),[],1);gx.b(3:5)']; else, excluded=[reshape(gx.W2(:,3:5),[],1);gx.b2(3:5)']; end
        addcheck(['excluded_class_gradient_exact_zero_' arches{a}],max(abs(excluded)),0);
        [~,g0]=r04nn_lossgrad(m,X,Y,mask,w,0.0);
        names=r04nn_param_names(m); rr=[];
        for j=1:numel(names)
            n=names{j}; expected=zeros(size(m.(n))); if startsWith(n,'W'), expected=.07*m.(n); end
            rr=[rr;reshape(g.(n)-g0.(n)-expected,[],1)]; %#ok<AGROW>
        end
        addcheck(['weights_only_L2_no_bias_penalty_' arches{a}],max(abs(rr)),1e-13);
    end
    m=r04nn_init('linear',1,4,0,9); m.W=[log(3),0,-1000,1000]; m.b=zeros(1,4);
    [l,g,aux]=r04nn_lossgrad(m,1,1,logical([1 1 0 0]),1,0.0);
    addcheck('binary_logistic_exact_known_case',max([abs(aux.P-[.75 .25 0 0]),abs(l-log(4/3)),abs(g.W-[-.25 .25 0 0]),abs(g.b-[-.25 .25 0 0])]),1e-14);
    m2=m; m2.W(3:4)=[1e6,-1e6];
    [l2,~,aux2]=r04nn_lossgrad(m2,1,1,logical([1 1 0 0]),1,0.0);
    addcheck('future_head_logit_invariance',max([abs(l-l2),abs(aux.P-aux2.P)]),0);
    m=r04nn_init('linear',1,2,0,7); m.W=[1 2]; m.b=[3 4];
    g=struct('W',[.1 -.2],'b',[.3 -.4]); opt=struct('eta',.02,'beta1',.9,'beta2',.999,'epsilon',1e-8);
    [m1,st1]=r04nn_adam(m,g,[],opt); names=r04nn_param_names(m); errs=[];
    for j=1:numel(names), n=names{j}; expected=m.(n)-opt.eta*g.(n)./(abs(g.(n))+opt.epsilon); errs=[errs;reshape(m1.(n)-expected,[],1)]; end %#ok<AGROW>
    addcheck('Adam_first_update_closed_form',max(abs(errs)),1e-13);
    g2=struct('W',[-.05 .4],'b',[.1 .2]); [m2,st2]=r04nn_adam(m1,g2,st1,opt); errs=[];
    for j=1:numel(names)
        n=names{j}; mm=.9*(.1*g.(n))+.1*g2.(n); vv=.999*(.001*g.(n).^2)+.001*g2.(n).^2;
        expected=m1.(n)-.02*(mm/(1-.9^2))./(sqrt(vv/(1-.999^2))+1e-8);
        errs=[errs;reshape(m2.(n)-expected,[],1)]; %#ok<AGROW>
    end
    addcheck('Adam_persistent_second_update_closed_form',max([abs(errs);abs(st2.t-2)]),1e-13);
    [reset,~]=r04nn_adam(m1,g2,[],opt); gap=max(abs(r04nn_pack(reset)-r04nn_pack(m2)));
    addcheck('Adam_state_carry_is_not_reset',double(gap<=1e-5),0);
    expecterror(@()r04nn_lossgrad(m,1,2,logical([1 0]),1,0.0),'invalid_mask_rejected');
    expecterror(@()r04nn_lossgrad(m,1,1,logical([1 1]),0,0.0),'zero_weight_mass_rejected');
    receipt.status='PASS'; receipt.checks=checks; receipt.check_count=numel(checks);
    receipt.ended_at_utc=char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss.SSSXXX'));
    writeReceipt(); fprintf('NEW_MATH_CHECKS_PASS %d\n',numel(checks));
catch ME
    receipt.status='FAIL'; receipt.checks=checks; receipt.check_count=numel(checks);
    receipt.error=struct('identifier',ME.identifier,'message',ME.message,'stack',ME.stack);
    receipt.ended_at_utc=char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss.SSSXXX'));
    writeReceipt(); rethrow(ME);
end
    function addcheck(name,value,threshold)
        pass=isfinite(value) && value<=threshold;
        checks(end+1)=struct('name',name,'pass',pass,'value',value,'threshold',threshold);
        assert(pass,'r04nn:MathCheckFailed','Math check failed: %s; value %.17g > %.17g',name,value,threshold);
    end
    function expecterror(fun,name)
        caught=false; try, fun(); catch, caught=true; end
        addcheck(name,double(~caught),0);
    end
    function writeReceipt()
        fid=fopen(fullfile(outdir,'math_checks_receipt_v1.json'),'w','n','UTF-8');
        assert(fid>=0,'r04nn:Receipt','Cannot write receipt.'); cleaner=onCleanup(@()fclose(fid));
        fprintf(fid,'%s\n',jsonencode(receipt,'PrettyPrint',true)); %#ok<NASGU>
    end
end
