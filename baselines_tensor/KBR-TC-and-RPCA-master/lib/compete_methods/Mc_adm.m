%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADM algorithm: tensor completion
% paper: Tensor completion for estimating missing values in visual data
% date: 05-22-2011
% min_X: \sum_i \alpha_i \|X_{i(i)}\|_*
% s.t.:  X_\Omega = T_\Omega
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [X, errList, lowrank_part] = Mc_adm(T, Omega, beta, maxIter, epsilon, X)

if nargin < 6
    X = T;
    X(logical(1-Omega)) = mean(T(Omega));
end
dim = size(T);
Omega = Unfold(Omega, dim, 1);
X = Unfold(X, dim, 1);

errList = zeros(maxIter, 1);
Y = zeros(dim);
Y = Unfold(Y, dim, 1);

M = Y;

normT = norm(T(:));

for k = 1: maxIter
%     if mod(k, 20) == 0
%         fprintf('Mc: iterations = %d   difference=%f \n', k, errList(k-1));
%     end
    beta = beta * 1.1;
    
    % update Y
    [Y, temp_n] = Pro2TraceNorm(X - M/beta, 1/beta);
      
    % update X
    lastX = X;
    X = Y + M/beta;
    X(Omega) = T(Omega);
    
    % update M
    M = M + beta * (Y - X);
 
    % compute the error
    errList(k) = norm(X(:)-lastX(:)) / normT;
    if errList(k) < epsilon && k>=80
        break;
    end
end

X = Fold(X, dim, 1);

lowrank_part = temp_n;

errList = errList(1:k);
% fprintf('Mc: ends: total iterations = %d   difference=%f  \n\n', k, errList(k));

