function memo = f_rtd_TNN(obs,opts,memo)
sz = size(obs.tY);

lamL=opts.para.lambdaL;
lamS=opts.para.lambdaS;
alpha=opts.para.alpha;
rho = opts.para.rho;
nu = opts.para.nu;

K0 =1;
tY=obs.tY;
% shortcuts
normTruthL=norm(double(memo.truthL(:)));
normTruthS=norm(double(memo.truthS(:)));
tL=zeros(sz);
tH=zeros(sz);
tY1=zeros(sz);
tY2=zeros(sz);
tS=zeros(sz);
tT=zeros(sz);
tY3=zeros(sz);
tK=zeros(sz);


fprintf('+++++++++f_rtd_tnn+++++++\n')
sz
for iter=1:opts.MAX_ITER_OUT
    % old point
    Hold=tH; Lold=tL;
    Kold=tK; Sold=tS; Told=tT;
    

    % ++++ Update L and S++++
    tT_ = tT+tY3/rho; tK_=tK+tY2/rho; tmp = 1+ (1+K0)*(1+rho);
    tH_ = tH+tY1/rho;   
    
    tS = (K0+1)*tY + (K0+rho+K0*rho)*tT_ - tK_-tH_;
    tS = tS/tmp;
    
    tL = (1+rho)*tK_ + (1+rho)*tH_ + tY - tT_;
    tL = tL/tmp;
    % ++++ Update L  and S++++
    
    % ++++ Update H, T, K ++++
    tau = lamL/rho;
    [tH,~] = f_prox_TNN( tL- tY1/rho, tau);
    
    tau = lamS/rho;
    tT = f_prox_l1(tS-tY3/rho,tau);
    
    T_tmp=tL-tY2/rho;
    tK=sign(T_tmp).*min(abs(T_tmp),alpha);
    % ++++ Update H, T, K ++++
    
    % ++++ Print state & Check convergence ++++
    infE = 0;
    % variable convergence
    infE=max(infE, h_inf_norm(tS-Sold));
    infE=max(infE,h_inf_norm(tT-Told));
    infE=max(infE,h_inf_norm(tK-Kold));
    infE = max(infE, h_inf_norm(tL-Lold));
    
    infE = max(infE, h_inf_norm(tH-Hold));
    % constraint convergence
    infE = max( infE, h_inf_norm(tL-tH) );
    infE=max(infE,h_inf_norm(tT-tS));
    infE=max(infE,h_inf_norm(tK-tL));
    
    
    memo.iter=iter;
    memo.rho(iter)=rho;
    memo.eps(iter)=infE;
    
    memo.rseL(iter)=h_tnorm(double( tL-memo.truthL))/normTruthL;
    memo.rseS(iter)=h_tnorm(double( tT-memo.truthS ))/normTruthS;
    
    %MSE
    memo.F2errorL(iter)=power( h_tnorm(double( tL-memo.truthL)), 2 );
    memo.F2errorS(iter)=power( h_tnorm(double( tT-memo.truthS)), 2 );
    memo.psnr(iter)=h_Psnr(memo.truthL,tL);
    
    
    % Print iteration state
    if opts.verbose  && mod(iter,2)==0
        fprintf('++%d:  epsL=%0.2e, rseL=%0.2e, rho=%0.2e,\n\t PSNR=%2.3f, ssim=%0.4f\n', ...
            iter, memo.eps(iter),memo.rseL(iter),memo.rho(iter),memo.psnr(iter),memo.ssim(iter));
    end
    
    if ( memo.eps(iter) <opts.MAX_EPS )
        fprintf('Stopped:%d:  epsL=%0.2e, rseL=%0.2e, rho=%0.2e,\n\t  PSNR=%2.3f, ssim=%0.4f\n', ...
            iter, memo.eps(iter),memo.rseL(iter),memo.rho(iter),memo.psnr(iter),memo.ssim(iter));
        break;
    end
    % ++++ Print state & Check convergence ++++
    
    
    % ++++ Dual variables ++++
    tY1 = tY1 + rho*(tH-tL);
    tY2 = tY2 + rho*(tK-tL);
    tY3 = tY3 + rho*(tT-tS);
    % ++++ Dual variables ++++
    
    % rho
    rho=min(rho*nu,opts.MAX_RHO);
end
memo.Lhat = tL;
memo.Shat = tS;
memo.TubalRank=f_tubal_rank(tH);
end
