function X=f_3DReshapeInverse(X,sz,k)
K=length(sz);
idx_aug = [1:K,1:K];
R = idx_aug(k);
vC = idx_aug( (k+2) : (K-3+k+2) );
t = idx_aug(k+1);
idx_p = [R, vC, t];

%C = prod(sz(vC));

X = reshape(X, [sz(R),sz(vC),sz(t)]);

idx_new = zeros(K:1);
for kk = 1:K
    idx_new(kk) = find(idx_p==kk);
end
X = permute(X,idx_new);