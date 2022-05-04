function [M,S,U,V]=f_prox_NN(M,tau,nSV)
[U,S,V]=svd(double(M),'econ');
S=max(abs(S)-tau,0);
M=U*S*V';
end
