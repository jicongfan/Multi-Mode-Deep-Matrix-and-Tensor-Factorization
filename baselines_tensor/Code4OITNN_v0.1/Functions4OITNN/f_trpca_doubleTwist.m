function memo = f_trpca_doubleTwist(obs,opts,memo)
sz = size(obs.tY);
K = length(sz);
lamS=opts.para.lambdaS;
rho = opts.para.rho;
nu = opts.para.nu;

if isfield(opts.para,'vW')==0
    weights=ones(K,1)/K;
else
    weights=opts.para.vW;
end

opR3D=@(X,k)f_3DReshape(X,k);
opR3Di=@(X,k)f_3DReshapeInverse(X,sz,k);

tM=obs.tY;
% shortcuts
normTruthL=norm(double(memo.truthL(:)));
normTruthS=norm(double(memo.truthS(:)));
tL=zeros(sz);
cK=cell(K,1);
cY=cell(K,1);
tY1=zeros(sz);
tS=zeros(sz);
tT=zeros(sz);
tY2=zeros(sz);
for k=1:2:K
    cK{k}=opR3D(tY1,k);
    cY{k}=opR3D(tY1,k);
end
sumL_=zeros(sz);
sumY=zeros(sz);

fprintf('+++++++++f_trpca_doubleTwist+++++++\n')
sz
for iter=1:opts.MAX_ITER_OUT
    % old point
    cLold=cK; tLold=tL; 
    Sold=tS; Told=tT; 
    
    % temp variables
    fval=0;
    sumL_=0*sumL_;  sumY=0*sumY;
    
    % ++++ Update L and S++++
    tT_ = rho*tT+tY2; tM_=rho*tM-tY1; tmp = rho*(1+ 2*2);
    for k=1:2:K
        sumL_ = sumL_+opR3Di(rho*cK{k}+cY{k},k);
    end
    
    tS = 2*tM_ + (2+1)*tT_-sumL_; 
    tS = tS/tmp;
    
    tL =  2*sumL_ + tM_ - tT_;    
    tL = tL/tmp;
    % ++++ Update L  and S++++
    
    % ++++ Update K_k, T, K ++++
    for k=1:2:K
        tau = weights(k)/rho;
        [cK{k},fk] = f_prox_TNN( opR3D(tL,k) - cY{k}/rho, tau);
        fval = fval+weights(k)*fk;
    end
    tau = lamS/rho;
    tT = f_prox_l1(tS-tY2/rho,tau);
    % ++++ Update K_k, T, K ++++

    % ++++ Print state & Check convergence ++++ 
    infE = 0;
    % variable convergence
    infE=max(infE, h_inf_norm(tS-Sold));
    infE=max(infE,h_inf_norm(tT-Told));
    infE = max(infE, h_inf_norm(tL-tLold));
    for k=1:2:K
        infE = max(infE, h_inf_norm(cK{k}-cLold{k}));
    end
    % constraint convergence
    infE=max(infE,h_inf_norm(tT-tS));
    for k=1:2:K
        infE = max( infE, h_inf_norm( opR3D(tL,k)-cK{k} ) );
    end
    
    memo.iter=iter;
    memo.fval(iter)=fval; 
    memo.rho(iter)=rho; 
    memo.eps(iter)=infE;
    
    memo.rseL(iter)=h_tnorm(double( tL-memo.truthL))/normTruthL;
    memo.rseS(iter)=h_tnorm(double( tT-memo.truthS ))/normTruthS;

    %MSE
    memo.F2errorL(iter)=power( h_tnorm(double( tL-memo.truthL)), 2 );
    memo.F2errorS(iter)=power( h_tnorm(double( tT-memo.truthS)), 2 );
    
    memo.psnr(iter)=h_Psnr(memo.truthL,tL);

    if K == 3
        memo.ssim(iter)=h_SSIM(memo.truthL,tL);
    end
    
    % Print iteration state
    if opts.verbose  && mod(iter,20)==0
    fprintf('++%d:  eps=%0.2e, rseL=%0.2e, rho=%0.2e, psnr=%2.3f \n', ...
                iter, memo.eps(iter),memo.rseL(iter),memo.rho(iter),memo.psnr(iter));
    end
    
    if ( memo.eps(iter) <opts.MAX_EPS ) 
        fprintf('Stopped:%d:  eps=%0.2e, rseL=%0.2e, rho=%0.2e, psnr=%2.3f \n', ...
                iter, memo.eps(iter),memo.rseL(iter),memo.rho(iter),memo.psnr(iter));
        break;
    end
    % ++++ Print state & Check convergence ++++ 
    
    
    % ++++ Dual variables ++++
    tY1 = tY1 +rho*(tL+tS-tM);
    tY2 = tY2 +rho*(tT-tS);
    % W_k
    for k=1:2:K
        cY{k}=cY{k}+rho*( cK{k} - opR3D(tL,k) );
    end
    % ++++ Dual variables ++++
    
    % rho
    rho=min(rho*nu,opts.MAX_RHO);
end
memo.Lhat = tL;
memo.Shat = tS;
memo.vTubalRank=zeros(K,1);
for k=1:2:K
memo.vTubalRank(k)=  f_tubal_rank(cK{k});
end
end
