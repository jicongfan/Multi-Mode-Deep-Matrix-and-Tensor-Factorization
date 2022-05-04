function memo = f_rtd_OITNN_O_tune_rho(obs,opts,memo)
% L: the tensor to compute OITNN
% vW: the weight vector
sz = size(obs.tY);
K = length(sz);
lamL=opts.para.lambdaL;
lamS=opts.para.lambdaS;

vRho = opts.para.vRho;
rho = opts.para.rho;
vNu = opts.para.vNu;
nu = opts.para.nu;


if isfield(opts.para,'vW')==0
    weights=ones(K,1)/K;
else
    weights=opts.para.vW;
end

opR3D=@(X,k)f_3DReshape(X,k);
opR3Di=@(X,k)f_3DReshapeInverse(X,sz,k);

tY=obs.tY;
% shortcuts
normTruthL=norm(double(memo.truthL(:)));
normTruthS=norm(double(memo.truthS(:)));
tL=tY;
cK=cell(K,1);
cY=cell(K,1);
%tW=zeros(sz);
tS=zeros(sz);
tR=zeros(sz);
tZ=zeros(sz);
%tK=zeros(sz);
for k=1:K
    cK{k}=opR3D(tS,k);
    cY{k}=opR3D(tS,k);
end
%sumK=zeros(sz);

fprintf('+++++++++f_rtd_OITNN_O_tune_rho+++++++\n')
sz
for iter=1:opts.MAX_ITER_OUT
    % old point
    cKold=cK; tLold=tL; 
    Sold=tS; Rold=tR; 
    
    % temp variables
    fval=0;    
    % ++++ Update L and S++++
    sumRho=0;
    tK_=tY;  
    tR_ = rho*tR+tY-tZ; 
    for k=1:K
        sumRho = sumRho+vRho(k);
        tK_ = tK_+opR3Di(vRho(k)*cK{k}-cY{k},k);
    end
    tmp = sumRho+rho+rho*sumRho;
    tL = ((1+rho)*tK_-tR_)/tmp;
    tS = ((1+sumRho)*tR_ -tK_)/tmp;
    % ++++ Update L  and S++++
    
    % ++++ Update K_k, T, K ++++
    for k=1:K
        tau = lamL*weights(k)/vRho(k);
        [cK{k},fk] = f_prox_TNN( opR3D(tL,k) + cY{k}/vRho(k), tau);
        fk = fk*weights(k);
        fval = fval+weights(k)*fk;
    end
    tau = lamS/rho;
    tR = f_prox_l1(tS+tZ/rho,tau);
    
    % ++++ Update K_k, T, K ++++

    % ++++ Print state & Check convergence ++++ 
    infE = 0;
    % variable convergence
    infE=max(infE, h_inf_norm(tS-Sold));
    infE=max(infE,h_inf_norm(tR-Rold));
    infE = max(infE, h_inf_norm(tL-tLold));
    for k=1:K
        infE = max(infE, h_inf_norm(cK{k}-cKold{k}));
    end
    
    % constraint convergence
    infE=max(infE,h_inf_norm(tR-tS));
    for k=1:K
        infE = max( infE, h_inf_norm( opR3D(tL,k)-cK{k} ) );
    end
    
    memo.iter=iter;
    memo.fval(iter)=fval; 
    memo.rho(iter)=rho; 
    memo.eps(iter)=infE;
    
    memo.rseL(iter)=h_tnorm(double( tL-memo.truthL))/normTruthL;
    memo.rseS(iter)=h_tnorm(double( tR-memo.truthS ))/normTruthS;

    %MSE
    memo.F2errorL(iter)=power( h_tnorm(double( tL-memo.truthL)), 2 );
    memo.F2errorS(iter)=power( h_tnorm(double( tR-memo.truthS)), 2 );
    
    memo.psnr(iter)=h_Psnr(memo.truthL,tL);
    if opts.verbose  && mod(iter,5)==0    
        fprintf('++%d:  epsL=%0.2e, rseL=%0.2e, rho=%0.2e, \n\t PSNR=%2.3f, ssim=%0.4f\n', ...
                iter, memo.eps(iter),memo.rseL(iter),memo.rho(iter),memo.psnr(iter),memo.ssim(iter));
    end
    
    if ( memo.eps(iter) <opts.MAX_EPS && iter>10) 
        fprintf('Stopped:%d:  epsL=%0.2e, rseL=%0.2e, rho=%0.2e,\n\t PSNR=%2.3f, ssim=%0.4f\n', ...
                iter, memo.eps(iter),memo.rseL(iter),memo.rho(iter),memo.psnr(iter),memo.ssim(iter));
        break;
    end
    % ++++ Print state & Check convergence ++++ 
    
    
    % ++++ Dual variables ++++
    tZ = tZ + rho*(tS-tR);
    % W_k
    for k=1:K
        cY{k}=cY{k}+vRho(k)*( opR3D(tL,k)-cK{k});
    end
    % ++++ Dual variables ++++
    
    % rho
    rho=min(rho*nu,opts.MAX_RHO);
    for k=1:K
        vRho(k)=min(vRho(k)*vNu(k),opts.MAX_RHO);
    end
end
memo.Lhat = tL;
memo.Shat = tS;
memo.vOverlapRank=zeros(K,1);
for k=1:K
memo.vOverlapRank(k)=f_tubal_rank(cK{k});
end
end
