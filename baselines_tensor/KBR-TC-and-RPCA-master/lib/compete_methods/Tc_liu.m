%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADM algorithm: tensor completion
% paper: Tensor completion for estimating missing values in visual data
% date: 05-22-2011
% min_X: \sum_i \alpha_i \|X_{i(i)}\|_*
% s.t.:  X_\Omega = T_\Omega
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [X, errList, lowrank_part] = Tc_liu(T, Omega, alpha, beta, maxIter, epsilon, X)

if nargin < 7
    X = T;
    X(logical(1-Omega)) = mean(T(Omega));
end

errList = zeros(maxIter, 1);
dim = size(T);
Y = cell(ndims(T), 1);
M = Y;

normT = norm(T(:));
for i = 1:ndims(T)
    Y{i} = X;
    M{i} = zeros(dim);
end

Msum = zeros(dim);
Ysum = zeros(dim);
lowrank_part = [];
temp_lowrank_part = zeros(3, 1);
for k = 1: maxIter
%     if mod(k, 20) == 0
%         fprintf('Liu: iterations = %d   difference=%f \n', k, errList(k-1));
%     end
    beta = beta * 1.05;
    
    % update Y
    Msum = 0*Msum;
    Ysum = 0*Ysum;
    
    for i = 1:ndims(T)
        [temp_Y, temp_n] = Pro2TraceNorm(Unfold(X-M{i}/beta, dim, i), alpha(i)/beta);
        Y{i} = Fold(temp_Y, dim, i); 
        Msum = Msum + M{i};
        Ysum = Ysum + Y{i}; 
        temp_lowrank_part(i) = temp_n;
    end
    lowrank_part = [lowrank_part, temp_lowrank_part];
       
    % update X
    %X(logical(1-Omega)) = ( Msum(logical(1-Omega)) + beta*Ysum(logical(1-Omega)) ) / (ndims(T)*beta);
    lastX = X;
    X = (Msum + beta*Ysum) / (ndims(T)*beta);
    X(Omega) = T(Omega);
    
    % update M
    for i = 1:ndims(T)
        M{i} = M{i} + beta*(Y{i} - X);
    end
    
    % compute the error
    errList(k) = norm(X(:)-lastX(:)) / normT;
    if errList(k) < epsilon 
        break;
    end
end

errList = errList(1:k);
% fprintf('Liu: ends: total iterations = %d   difference=%f  \n\n', k, errList(k));

