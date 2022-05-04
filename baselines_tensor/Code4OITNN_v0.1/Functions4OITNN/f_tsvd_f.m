function [U,S,V] = f_tsvd_f(X)
sz =size(X);
X = fft(X,[],3);
r = min(sz(1),sz(2));
U = zeros(sz(1),r,sz(3));
S = zeros(r,r,sz(3));
V = zeros(sz(2),r,sz(3));
iMid = floor(sz(3)/2)+1;
for i=1:iMid
    [U(:,:,i),S(:,:,i),V(:,:,i)]=svd(X(:,:,i),'econ');
end
for i=iMid+1:sz(3)
    k = sz(3)+2-i;
    U(:,:,i)=conj(U(:,:,k));
    S(:,:,i)=S(:,:,k);
    V(:,:,i)=conj(V(:,:,k));
end
% U = ifft(U,[],3);
% S = ifft(S,[],3);
% V = ifft(V,[],3);