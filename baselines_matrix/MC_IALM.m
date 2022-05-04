function [A,E]=MC_IALM(D,M)
% initialize
[m,n]=size(D);
Y = zeros(m,n);
A = Y;
E=Y;
% Iteration 
iter = 0;
converged = false;
mu=0.1;
r=1.01;
maxiter=500;
e1=1e-6;
e2=1e-6;
normD=norm(D,'fro');
while ~converged&&iter<maxiter
    iter=iter+1;
   temp=D-E+Y/mu;
   A_new=solve_NuclearNorm(temp,mu);
   E_new=(D-A_new+Y/mu).*~M;
   Y=Y+mu*(D-A_new-E_new);
   stopC1=norm(D-A_new-E_new,'fro')/normD;
   stopC2=max(norm(A_new-A,'fro'),norm(E_new-E,'fro'))/normD;
   isstopC=stopC1<e1&&stopC2<e2;
   if isstopC
        disp('converged')
        break;
   end
   if mod(iter,20)==0||isstopC
%       disp(['iteration=' num2str(iter) '/' num2str(maxiter) '  rankX=' num2str(rank(A_new,1e-3*norm(A_new,2))) '  mu=' num2str(mu)])
        disp(['iteration=' num2str(iter) '/' num2str(maxiter) '  mu=' num2str(mu) ' stopC1=' num2str(stopC1) '  stopC2=' num2str(stopC2) '  ......'])
   end  
   mu=min(mu*r,1e5);
   A=A_new;
   E=E_new;
end
end
%%
function [Z]=solve_NuclearNorm(L,mu)
if min(size(L))<2000
    [U,sigma,V] = svd(L,'econ');
else
    [U,sigma,V] = lansvd(L,0);
end
    sigma = diag(sigma);
    svp = length(find(sigma>1/mu));
    if svp>=1
        sigma = sigma(1:svp)-1/mu;
    else
        svp = 1;
        sigma = 0;
    end
    Z=U(:,1:svp)*diag(sigma)*V(:,1:svp)';
end


