%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADM algorithm: tensor completion
% paper: Tensor completion for estimating missing values in visual data
% date: 05-22-2011
% min_X: \sum_i \alpha_i \|X_{i(i)}\|_mcp
% s.t.:  X_\Omega = T_\Omega
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [X, errList,lowrank_part] = McpLRTC(T, Omega, alpha, beta, maxIter, epsilon, X)


%% initilizing the X

[X, errList] = HaLRTC(T, Omega, alpha, beta, maxIter, epsilon);
dim = size(X);
Sigma_old =cell(ndims(X), 1); 
gamma1=5;

for outier=1:2
    
for i = 1:ndims(X)
    
     [Stemp, Sigma_old{i}, Dtemp] = MySVD(Unfold(X, dim, i));
      Sigma_old{i} = max(1-Sigma_old{i}/gamma1,0);
     
end
%% one-step ALM

errList = zeros(maxIter, 1);
dim = size(T);
Y = cell(ndims(T), 1); % Y : spliting variable
M = Y; % M : dual variable

normT = norm(T(:));
for i = 1:ndims(T)
    Y{i} = X;
    M{i} = zeros(dim);
end

Msum = zeros(dim);
Ysum = zeros(dim);
lowrank_part = [];
temp_n        = zeros(ndims(T),1);
for k = 1: maxIter
%     if mod(k, 20) == 0
%         fprintf('McpLRTC: iterations = %d   difference=%f\n', k, errList(k-1));
%     end
    beta = beta * 1.05;
    
    % update Y
    Msum = 0*Msum;
    Ysum = 0*Ysum;
 
    for i = 1:ndims(T)
        w=diag(Sigma_old{i});
        [temp,~,temp_n(i)] = ReweightedPro2TraceNorm( Unfold(X-M{i}/beta, dim, i), alpha(i)/beta, w );
        Y{i} = Fold( temp, dim,  i);
        Msum = Msum + M{i};
        Ysum = Ysum + Y{i};
    end
    
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
    lowrank_part = [lowrank_part, temp_n];
%     if errList(k) < epsilon
%         break;
%      end
end

end

errList = errList(1:k);
% fprintf('McpLRTC ends: total iterations = %d   difference=%f\n\n', k, errList(k));
