function memo=f_tc_OITNN_O(obs,opts,memo)
% parameters
alpha=opts.para.alpha; 
vRho=opts.para.vRho; 
vNu=opts.para.vNu;

% shortcuts
normTruth=norm(double(memo.truth(:)));
K=length(obs.tsize);
% tensor variables X, 
X=zeros(obs.tsize);
cM=cell(K,1);
cY=cell(K,1);
for k=1:K
    cM{k}=X; 
    cY{k}=X;
end
SumM=X;
SumY=X;

fprintf('++++f_tc_OITNN_O++++\n');
tSize=obs.tsize
for iter=1:opts.MAX_ITER_OUT
    oldX=X; fval=0;

    SumM=0*SumM;
    SumY=0*SumY;
    
    for k=1:K
        rho=vRho(k);
        M=f_KDArray2ThreeD(X-cY{k}/rho,k);
        tau=alpha(k)/rho;
        [M,fk]=f_prox_TNN(M,tau);
        cM{k}=f_3DArray2KD(M,tSize,k);
        SumY=SumY+cY{k};
        SumM=SumM+cM{k}*rho;
        fval=fval+alpha(k)*fk;
    end
    sumRho=0;
    for k=1:K
        sumRho=vRho(k)+sumRho;
    end
    X=(SumY+SumM)/sumRho;
    X(obs.idx)=obs.y;
        
    % Record 
    memo.iter=iter;
    memo.fval(iter)=fval; 
    memo.rho(iter)=vRho(2); 
    memo.eps(iter)=norm(double( X(:)-oldX(:) ))/( norm(double(oldX(:)))+eps );
    %memo.err(iter)=sum( double( L(:)-memo.truth(:) ).*double( L(:)-memo.truth(:) ) );
    memo.err(iter)=norm(double( X(:)-memo.truth(:) ))/normTruth;
    memo.psnr(iter)=h_Psnr(memo.truth(:), X(:));
    % Print iteration state
    if opts.verbose &&  mod(iter,1)==0
    fprintf('++%d: psnr=%0.2f, eps=%0.2e, err=%0.2e, rho=%0.2e \n', ...
                iter,memo.psnr(iter),memo.eps(iter),memo.err(iter),memo.rho(iter));
    end
    
    if ( memo.eps(iter) <opts.MAX_EPS ) && ( iter > 50 )
        fprintf('Stopped:%d: psnr=%0.2f, eps=%0.2e, err=%0.2e,rho=%0.2e\n', ...
                iter,memo.psnr(iter),memo.eps(iter),memo.err(iter),memo.rho(iter));
        memo.T_hat=X;
        break;
    end
       
    for k=1:K
        cY{k}=cY{k}+vRho(k)*( cM{k}-X );
        vRho(k)=min(vRho(k)*vNu(k),opts.MAX_RHO);
    end
    
    
end
memo.T_hat=X;
