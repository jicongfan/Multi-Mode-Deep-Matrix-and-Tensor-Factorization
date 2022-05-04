function[X_new,sigma_new,svp]= ReweightedPro2TraceNorm(Z,tau,w)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% min: 1/2*||Z-X||^2 + tau*\sum_i |sigma_i(X)|*w_i
% w is a sigular value vector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[S, V, D, ~] = MySVD(Z);
SigmaZ= diag(V);
svp =length(find(SigmaZ > tau));
sigma_new = GeneralizedThresholding(SigmaZ(1:svp), w(1:svp), tau);
if svp==0
    X_new=zeros(size(Z));
else
    X_new   =S(:, 1:svp) * diag(sigma_new) * (D(:,1:svp))';
end

end



