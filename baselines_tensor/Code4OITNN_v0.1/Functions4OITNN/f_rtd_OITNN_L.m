function memo = f_rtd_OITNN_L(obs,opts,memo)
% L: the tensor to compute OITNN
% vW: the weight vector
sz = size(obs.tY);
K = length(sz);
lamL=opts.para.lambdaL;
lamS=opts.para.lambdaS;
alpha=opts.para.alpha;
rho = opts.para.rho;
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
cL=cell(K,1);
cK=cell(K,1);
cY=cell(K,1);
tW=zeros(sz);
tS=zeros(sz);
tT=zeros(sz);
tZ=zeros(sz);
tK=zeros(sz);
for k=1:K
    cL{k}=zeros(sz);
    cK{k}=opR3D(tW,k);
    cY{k}=opR3D(tW,k);
end
sumK_=zeros(sz);
sumW=zeros(sz);

fprintf('+++++++++f_rtd_OITNN_L+++++++\n')
sz
for iter=1:opts.MAX_ITER_OUT
    % old point
    cKold=cK; cLold=cL; 
    Kold=tK; Sold=tS; Told=tT; 
    
    % temp variables
    fval=0;
    sumK_=0*sumK_;  sumW=0*sumW;
    
    % ++++ Update L_k and S++++
    tT_ = tT+tZ/rho; tK_=tK+tW/rho; tmp = K+ (1+K)*(1+rho);
    for k=1:K
        sumK_ = sumK_+opR3Di(cK{k}+cY{k}/rho,k);
    end
    
    tS = (K+1)*tY + (K+rho+K*rho)*tT_ - K*tK_-sumK_; 
    tS = tS/tmp;
    
    sumL = K*(1+rho)*tK_ + (1+rho)*sumK_ + K*tY - K*tT_;    
    sumL = sumL/tmp;
    
    for k=1:K
        cL{k} = rho*tK_ + rho*opR3Di(cK{k}+cY{k}/rho,k) + tY ...
                     - (1+rho)*sumL - tS;
        cL{k} = cL{k}/rho;
    end
    % ++++ Update L_k  and S++++
    
    % ++++ Update K_k, T, K ++++
    dK=0;
    for k=1:K
        tau = lamL*weights(k)/rho;
        [cK{k},fk] = f_prox_TNN( opR3D(cL{k},k) - cY{k}/rho, tau);
        fk = fk*weights(k);
        fval = fval+weights(k)*fk;
        dK = max(dK, h_inf_norm(cK{k}-cKold{k}));
    end
    tau = lamS/rho;
    tT = f_prox_l1(tS-tZ/rho,tau);
    T_tmp=sumL-tW/rho;
    tK=sign(T_tmp).*min(abs(T_tmp),alpha);
    % ++++ Update K_k, T, K ++++

    % ++++ Print state & Check convergence ++++ 
    infE = 0;
    % variable convergence
    infE=max(infE, h_inf_norm(tS-Sold));
    infE=max(infE,h_inf_norm(tT-Told));
    infE=max(infE,h_inf_norm(tK-Kold));
    for k=1:K
        infE = max(infE, h_inf_norm(cL{k}-cLold{k}));
        infE = max(infE, h_inf_norm(cK{k}-cKold{k}));
    end
    % constraint convergence
    infE=max(infE,h_inf_norm(tT-tS));
    infE=max(infE,h_inf_norm(tK-sumL));
    for k=1:K
        infE = max( infE, h_inf_norm( opR3D(cL{k},k)-cK{k} ) );
    end
    
    memo.iter=iter;
    memo.fval(iter)=fval; 
    memo.rho(iter)=rho; 
    memo.eps(iter)=infE;
    
    memo.rseL(iter)=h_tnorm(double( sumL-memo.truthL ))/normTruthL;
    memo.rseS(iter)=h_tnorm(double( tT-memo.truthS ))/normTruthS;

    %MSE
    memo.F2errorL(iter)=power( h_tnorm(double( sumL-memo.truthL)), 2 );
    memo.F2errorS(iter)=power( h_tnorm(double( tT-memo.truthS)), 2 );
    
    memo.psnr(iter)=h_Psnr(memo.truthL,sumL);
    if K == 3 && opts.verbose
        memo.ssim(iter)=h_SSIM(memo.truthL,sumL);
%         fprintf('++%d fL1:fL2: fL3=%0.2f:%0.2f:%0.2f, fS=%0.2f \n', iter, ...
%             h_tnorm(cK{1})/normTruthL,...
%             h_tnorm(cK{2})/normTruthL,h_tnorm(cK{3})/normTruthL, h_tnorm(tT)/normTruthS);
    end
    % Print iteration state
%     if opts.showImg
%          figure(10086);
%         clf;
%         hold on;
%         imshow(sumL);
%         txt=sprintf('latent %d: psnr = %0.2f, %0.2f:%0.2f:%0.2f,%0.2f',iter,memo.psnr(iter),...
%             h_tnorm(cK{1})/normTruthL,...
%             h_tnorm(cK{2})/normTruthL,h_tnorm(cK{3})/normTruthL, h_tnorm(tT-memo.truthS)/normTruthS);
%         title(txt);
%         pause(0.01);
%     end
    if opts.verbose  && mod(iter,5)==0
    fprintf('++%d:  epsL=%0.2e, rseL=%0.2e, rho=%0.2e,\n\t PSNR=%2.3f, ssim=%0.4f\n', ...
                iter, memo.eps(iter),memo.rseL(iter),memo.rho(iter),memo.psnr(iter),memo.ssim(iter));
    end
    
    if ( memo.eps(iter) <opts.MAX_EPS ) && ( iter > 10 )
        fprintf('Stopped:%d:  epsL=%0.2e, rseL=%0.2e, rho=%0.2e,\n\r PSNR=%2.3f, ssim=%0.4f\n', ...
                iter, memo.eps(iter),memo.rseL(iter),memo.rho(iter),memo.psnr(iter),memo.ssim(iter));
        break;
    end
    % ++++ Print state & Check convergence ++++ 
    
    
    % ++++ Dual variables ++++
    tW = tW +rho*(tK-sumL);
    tZ = tZ +rho*(tT-tS);
    % W_k
    for k=1:K
        cY{k}=cY{k}+rho*( cK{k} - opR3D(cL{k},k) );
    end
    % ++++ Dual variables ++++
    
    % rho
    rho=min(rho*nu,opts.MAX_RHO);
end
memo.cLhat=cK;
memo.Lhat = sumL;
memo.Shat = tS;
memo.vLatentRank=zeros(K,1);
for k=1:K
memo.vLatentRank(k)=f_tubal_rank(cK{k});
end
end
