function G=update_G(G,X,M,U,lambda,maxiter)
M1=tenmat(M,1)';
s_G=size(G);
G1=tenmat(G,1);
X1=tenmat(X,1)';
N=ndims(X);
BN=U{end};
if N>2
    for n=N-1:-1:2
        BN=kron(BN,U{n});
    end
end
%    
p.verbosity = 0;                  
p.MaxIter   = maxiter;          
p.d_Obj     = 1e-5;
p.method    = 'IRprop+';          
p.display   = 0;

y=G.data(:);
par.X=X1.data;
par.M=M1.data;
par.A=U{1};
par.B=BN';
par.sG1=size(G1);
par.lambda=lambda;

%[y,f,EXITFLAG,STATS] = rprop(@fg_G,y,p,par);
[f,g]=fg_G(y,par);
y=y-0.01*g;
G1=reshape(y,size(G1));
G=tensor(G1,s_G);

end
%%
function [f,g]=fg_G(y,par)
C=reshape(y,par.sG1);
f=0.5*norm(par.M.*(par.X-par.A*C*par.B),'fro')^2+0.5*par.lambda*norm(C,'fro')^2;
g=-par.A'*(par.M.*(par.X-par.A*C*par.B))*par.B'+par.lambda*C;
g=g(:);
end