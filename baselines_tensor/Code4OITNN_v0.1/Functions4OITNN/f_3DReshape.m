function X=f_3DReshape(X,k)
sz = size(X);
K=length(sz);
idx_aug = [1:K,1:K];
R = idx_aug(k);
vC = idx_aug( (k+2) : (K-3+k+2) );
t = idx_aug(k+1);
idx_p = [R, vC, t];
X = permute(X,idx_p);
C = prod(sz(vC));
X = reshape(X, [sz(R), C, sz(t)]);
