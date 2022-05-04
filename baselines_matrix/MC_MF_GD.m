function [Xr,A,B] = MC_MF_GD(X,M,d,lambda,options)
disp('MC_MF ...')
if ~isfield(options,'maxiter')
    options.maxiter=1000;
end
if ~isfield(options,'tol')
    options.tol=1e-6;
end
if ~isfield(options,'solver')
    options.solver='IRprop+';
end
% initialization
[m,n]=size(X);
[U,S,V]=svds(X,d);
s=diag(S);
A=U*diag(s.^0.5);
B=diag(s.^0.5)*V';
MF.m=m;
MF.n=n;
MF.d=d;
MF.X=X;
MF.M=M;
MF.lambda=lambda;
% opt
y=[A(:);B(:)];
if strcmp(options.solver,'IRprop+')
    p.verbosity = 3;                    % Increase verbosity to print something [0~3]
    p.MaxIter   = options.maxiter;          
    p.d_Obj     = options.tol;
    p.method    = 'IRprop+';          
    p.display   = 0;
    [y,f,EXITFLAG,STATS] = rprop(@fg,y,p,MF);
else   
    opt.Method=options.solver;
    opt.MaxIter=options.maxiter;
    opt.optTol=options.tol;
    opt.progTol=options.tol;
    opt.LS_type=0;
    [y,loss]=minFunc(@fg,y,opt,MF);
end
A=reshape(y(1:m*d),m,d);
B=reshape(y(m*d+1:end),d,n);
Xh=A*B;
Xr=M.*X+(1-M).*Xh;
end
%%
function [f,g]=fg(y,MF)
A=reshape(y(1:MF.m*MF.d),MF.m,MF.d);
B=reshape(y(MF.m*MF.d+1:end),MF.d,MF.n);
f=0.5*norm(MF.M.*(MF.X-A*B),'fro')^2+0.5*MF.lambda*(norm(A,'fro')^2+norm(B,'fro')^2);
gA=-(MF.M.*(MF.X-A*B))*B';
gB=-A'*(MF.M.*(MF.X-A*B));
g=[gA(:);gB(:)];
end
