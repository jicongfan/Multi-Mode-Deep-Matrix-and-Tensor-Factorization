function [vObs, vIndex]=f_P_Rand_Omega(M,obsRatio)
N=numel(M);
idx=randperm(N);
k=max( 1,ceil(N*obsRatio) );
vIndex=idx(1:k);
vObs=M(vIndex);
