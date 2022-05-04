function [Xr,M]=MC_DMF_OSE(NN,X,M,lambda,maxiter)
% min |X-f(Z)|^2+|W|^2
Z=NN.a{end-1}';
act_f=NN.activation_func{end};
disp('Training neural networks for DMFMC ......')
NN_MF.maxiter=2000;
[W]=ose_optimization_rprop(X,M,Z,act_f,lambda,maxiter);
switch act_f
    case 'sigm'
        Xr=sigm(W*Z).*(1-M)+X.*M;
end
end
%%
function [W]=ose_optimization_rprop(X,M,Z,act_f,lambda,maxiter)
p.verbosity = 3;                    % Increase verbosity to print something [0~3]
p.MaxIter   = maxiter;          
p.d_Obj     = 1e-5;
p.method    = 'IRprop+';          
p.display   = 0;
% W=randn(size(X,1),size(Z,1));
W=zeros(size(X,1),size(Z,1));
[m,n]=size(W);
w=W(:);
V.act_f=act_f;
V.lambda=lambda;
V.M=M;
V.Z=Z;
V.X=X;
V.sw=size(W);
[w,f,EXITFLAG,STATS] = rprop(@fg_ose,w,p,V);
W=reshape(w,V.sw);
end
%%
function [f,g]=fg_ose(w,V)
W=reshape(w,V.sw);
switch V.act_f
    case 'sigm'
        Y=sigm(W*V.Z);
        EM=V.M.*(V.X-Y);
        f=0.5*sum(EM(:).^2)/size(Y,2)+V.lambda*0.5*sum(W(:).^2);
        gW=-EM.*Y.*(1-Y)*V.Z'/size(Y,2)+V.lambda*W;
        g=gW(:);
end
end
