function M=f_fold(M, sz,rInd)
K=length(sz);
R=rInd;
vIdx = ones(1,K); 
vIdx(R)=0;
C = find(vIdx);

M = reshape(M,[sz(R), sz(C)]);

order_new = 1:K;
for i=1:K
    order_new(i) = find([R C]==i);
end
M = permute(M,order_new);
end