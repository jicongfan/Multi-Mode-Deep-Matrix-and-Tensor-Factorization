
function memo=f_tc_TNN(obs,opts,memo)
% parameters
rho=opts.para.rho; 
nu=opts.para.nu;

% shortcuts
normTruth=norm(double(memo.truth(:)));
%nModes=length(obs.tsize);
d1=obs.tsize(1);
d2=obs.tsize(2);
d3=obs.tsize(3);
% tensor variables L, 
L=zeros(obs.tsize);
% decoupling variable to F-norm
L1=L;

% Lagrangian Multipliers
Y1=L; Y2=L;

% Mask Tensor and Observation Tensor
B=zeros(obs.tsize); Y=zeros(obs.tsize);
B(obs.idx)=1; Y(obs.idx)=obs.y;

% Tensor of all ones

fprintf('++++f_tc_TNN++++\n');

for iter=1:opts.MAX_ITER_OUT
    oldL=L; 
    
    %+++++Update L1++++++++
    % L1 ==> F-norm decoupler
    L1=(Y1+rho*L-B.*Y2+rho*Y)./(rho*B+rho);
    %+++++Update L1++++++++
     
    %+++++Update L++++++++
    % L ==> TNN proximator
    L_tmp=L1-Y1/rho;
    [L,fval]=f_prox_TNN(L_tmp,1/rho);
    %+++++Update L++++++++    
    
    % Record 
    memo.iter=iter;
    memo.fval(iter)=fval; 
    memo.rho(iter)=rho; 
    memo.eps(iter)=norm(double( L(:)-oldL(:) ))/( norm(double(oldL(:)))+eps );
    memo.err(iter)=norm(double( L(:)-memo.truth(:) ))/normTruth;
    memo.psnr(iter)=h_Psnr(memo.truth(:), L(:));
    % Print iteration state
    if opts.verbose &&  mod(iter,5)==0
    fprintf('++%d: psnr=%0.2f, eps=%0.2e, err=%0.2e, rho=%0.2e size=(%d,%d,%d)\n', ...
                iter,memo.psnr(iter),memo.eps(iter),memo.err(iter),memo.rho(iter),d1,d2,d3);
    end
    
    if ( memo.eps(iter) <opts.MAX_EPS ) && ( iter > 10 )
        fprintf('Stopped:%d: psnr=%0.2f eps=%0.2e, err=%0.2e,rho=%0.2e\n', ...
                iter,memo.psnr(iter),memo.eps(iter),memo.err(iter),memo.rho(iter));
        memo.T_hat=L;
        break;
    end
    
    %+++++Update Y1 and Y2++++++++
    Y1=Y1+rho*(L-L1);
    Y2=Y2+rho*B.*(L1-Y);
    %+++++Update Y1 and Y2++++++++    
    
    rho=min(rho*nu,opts.MAX_RHO);
end
memo.T_hat=L;
