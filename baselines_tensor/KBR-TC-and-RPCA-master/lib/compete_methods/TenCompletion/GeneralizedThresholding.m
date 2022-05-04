function [A_new] = GeneralizedThresholding(A, W, tau)
A_sign = sign(A);
A_new = abs(A) - tau * abs(W);
A_new(A_new < 0) = 0;
A_new = A_new .* A_sign;
end