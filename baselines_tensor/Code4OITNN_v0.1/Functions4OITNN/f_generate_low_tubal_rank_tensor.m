function L=f_generate_low_tubal_rank_tensor(tSize,rTubal,maxValue)
m=tSize(1); n=tSize(2); k=tSize(3); r=rTubal;
P=randn(m,r,k); Q=randn(r,n,k);
L=tprod(P,Q);
%normL_inf=max(abs(L(:)));
%Spiky Normalized
%L=L/normL_inf*maxValue;
