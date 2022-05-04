function [X,A,Z]=MC_ALS(X0,M,alpha,d,maxIter)%%% 
%%% 0.5*|X-AZ| + 0.5*|A|2 + 0.5*|Z|2
[m,n]=size(X0);
%
[U,S,V]=svd(X0);
U=U(:,1:d);
S=S(1:d,1:d);
V=V(:,1:d);
s=diag(S);
A=U*diag(s.^0.5);
Z=diag(s.^0.5)*V';
% A=randn(m,d);
% Z=zeros(d,n);
X=X0;
% Z=randn(d,n);
rho=5;
e=1e-6;
%
iter=0;
a0=1;
cc=1;
obj_old=Inf;
%
while iter<maxIter
    iter=iter+1;
    Z_new=inv(A'*A+alpha*eye(d))*A'*X;
    A_new=X*Z_new'*inv(Z_new*Z_new'+alpha*eye(d));
    X_new=A_new*Z_new;
    X_new=X_new.*~M+X0.*M;
    %
    stopC=max([norm(Z_new-Z,'fro') norm(A_new-A,'fro') norm(X_new-X,'fro')]);
    %
    isstopC=stopC<e;
    if mod(iter,50)==0||isstopC||iter==1
        obj=0.5*norm(X_new-A_new*Z_new,'fro')^2+0.5*alpha*(norm(A_new,'fro')^2+norm(Z_new,'fro')^2);
        disp(['iteration=' num2str(iter) '/' num2str(maxIter)...
            '  f=' num2str(obj) '  stopC=' num2str(stopC)])
    end
    if isstopC
        disp('converged')
        break;
    end
%     obj_old=obj;
    Z=Z_new;
    A=A_new;
    X=X_new;   
end
X=X0.*M+(A*Z).*(1-M);     
end



