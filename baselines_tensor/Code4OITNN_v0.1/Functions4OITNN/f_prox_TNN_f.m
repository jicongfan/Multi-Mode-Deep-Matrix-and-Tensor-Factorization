function [X, objV] = f_prox_TNN_f(Y,rho)
% proximal function of TNN
% f_prox_TNN_f: without IFFT
[~,~,n3] = size(Y);

[U,S,V]=f_tsvd_f(Y);

[S,objV] = f_prox_l1(S, rho);

X = Y*0;
mid = ceil((n3+1)/2);
X(:,:,1) = U(:,:,1)*S(:,:,1)*V(:,:,1)';
for i = 2:mid
    X(:,:,i) = U(:,:,i)*S(:,:,i)*V(:,:,i)';
    X(:,:,n3+2-i) = conj(X(:,:,i));
end


