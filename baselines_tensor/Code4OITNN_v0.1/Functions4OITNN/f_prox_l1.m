function [x,f]=f_prox_l1(Y,rho)
x=sign(double(Y)).*max(abs(double(Y))-rho,0);
f=sum(abs(x(:)));