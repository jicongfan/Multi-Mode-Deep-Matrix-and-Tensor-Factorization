function [J,x]=opt_Adam(f,x,V,opt)
alpha=opt.alpha;
maxiter=opt.maxiter;
beta_1=0.9;
beta_2=0.999;
e=1e-8;
%
m=0;
v=0;
t=0;
for i=1:maxiter
    t=t+1;
    [J(i),g] = feval(f,x,V);
    m=beta_1.*m+(1-beta_1).*g;
    v=beta_2.*v+(1-beta_2).*(g.^2);
    m1=m./(1-beta_1^t);
    v1=v./(1-beta_2^t);
    x_new=x-alpha*m./(v.^0.5+e);
    if max(abs(x_new-x))<1e-6
        break;
    end
    x=x_new;
    if mod(t,10)==0
        disp(['iter=' num2str(t)  '/' num2str(maxiter) '  fun_val=' num2str(J(i)) '  alpha=' num2str(alpha)])
    end
%     if t>1
%         if J(t)>J(t-1)
%             alpha=alpha*0.8;
%         else
%             alpha=alpha*1.1;
%         end
%     end
end
end
