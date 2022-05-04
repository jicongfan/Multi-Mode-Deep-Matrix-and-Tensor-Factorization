function memo=f_tc_OITNN_L(obs,opts,memo)
% parameters
alpha=opts.para.alpha; rho=opts.para.rho; nu=opts.para.nu;

% shortcuts
normTruth=norm(double(memo.truth(:)));
K=length(obs.tsize);
% tensor variables X
X=zeros(obs.tsize);
cM=cell(K,1); 
cX=cell(K,1);
cY=cell(K,1); 
Z=zeros(obs.tsize); 
for k=1:K
    cM{k}=X; 
    cX{k}=X;
    cY{k}=X;
end
SumM=X;
SumY=X;

fprintf('++++f_tc_OITNN_L++++\n');
tSize=obs.tsize
for iter=1:opts.MAX_ITER_OUT
    oldX=X; fval=0;

    SumM=0*SumM;
    SumY=0*SumY;
    
    % M_i
    for k=1:K
        M=f_KDArray2ThreeD(cX{k}-cY{k}/rho, k);
        tau=alpha(k)/rho;
        [M,fk]=f_prox_TNN(M,tau);
        cM{k}=f_3DArray2KD(M,tSize,k);
        SumY=SumY+cY{k};
        SumM=SumM+cM{k};
        fval=fval+alpha(k)*fk;
    end
    
    % X_i
    SumX=(SumY+rho*SumM+K*rho*X+K*Z)/(K+1)/rho;
    for k=1:K
        cX{k}= (rho*X+Z+rho*cM{k}+cY{k}-rho*SumX)/rho ;
    end
    
    % X
    X=(rho*SumX-Z)/rho;
    X(obs.idx)=obs.y;
        
    % Record 
    memo.iter=iter;
    memo.fval(iter)=fval; 
    memo.rho(iter)=rho; 
    memo.eps(iter)=norm(double( X(:)-oldX(:) ))/( norm(double(oldX(:)))+eps );
    memo.err(iter)=norm(double( X(:)-memo.truth(:) ))/normTruth;
    memo.psnr(iter)=h_Psnr(memo.truth(:), X(:));
    % Print iteration state
    if opts.verbose &&  mod(iter,1)==0
    fprintf('++%d: psnr=%0.2f, eps=%0.2e, err=%0.2e, rho=%0.2e \n', ...
                iter,memo.psnr(iter),memo.eps(iter),memo.err(iter),memo.rho(iter));
    end
    
    if  (memo.eps(iter) <opts.MAX_EPS && iter>140)
        fprintf('Stopped:%d: psnr=%0.2f, eps=%0.2e, err=%0.2e,rho=%0.2e\n', ...
                iter,memo.psnr(iter),memo.eps(iter),memo.err(iter),memo.rho(iter));
        memo.T_hat=X;
        break;
    end
    
    % Y_i
    for k=1:K
        cY{k}=cY{k}+rho*( cM{k}-cX{k} );
    end
    % Z
    Z=Z+rho*(X-SumX);
    
    % rho
    rho=min(rho*nu,opts.MAX_RHO);
end
memo.T_hat=X;
memo.cT=cM;
