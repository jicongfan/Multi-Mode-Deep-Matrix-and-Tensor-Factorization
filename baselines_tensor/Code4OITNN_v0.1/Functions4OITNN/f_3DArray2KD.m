function X=f_3DArray2KD(X,sz,k)
K=length(sz);
idx_aug = [K,1:K,1:K];
R = idx_aug(k);
vC = idx_aug( (k+2) : (K-3+k+2) );

idx_p = [R, vC, k];

%C = prod(sz(vC));

X = reshape(X, [sz(R),sz(vC),sz(k)]);

idx_new = zeros(K:1);
for kk = 1:K
    idx_new(kk) = find(idx_p==kk);
end
X = permute(X,idx_new);

