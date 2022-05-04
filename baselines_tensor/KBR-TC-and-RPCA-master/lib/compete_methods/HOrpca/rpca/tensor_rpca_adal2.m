function results = tensor_rpca_adal2( data, params )
% Solve
%       min_{X,E} \sum_i ||X_(i)||_* + \lambda1*||E||_1
%       s.t. X + E = T

%  converse to
%       min_{X1,X2,...,XN,E} \sum_i ||Xi_(i)||_* + \lambda*||E||_1
%       s.t. Xi + E = T,i=1,2,...,N
%            
% data.T
% params
% X, V are cell arrays of tensors.
%
% Algorithm: ADAL

tic;
T = data.T;
N = length( size(params.X0) );
X = cell( 1, N );
U = cell( 1, N );
V = cell( 1, N );
for i = 1:N
    X{i} = params.X0;
    V{i} = params.V0{i};
end
E = params.E0;

lambda = params.lambda;
mu = params.mu1;
verbose = params.verbose;
T_mat=tenmat(T,1);
tnorm = norm(T_mat);

for iter = 1:params.max_iter
    % solve X_i's
    for i = 1:N
        [X{i}, junk, U{i}] = tensor_shrinkage( T+mu*V{i}-E, mu, i );
    end
    
    % solve E
    Ep = E;
    D = cell( 1, N );
    for i = 1:N
        D{i} = X{i} - T - mu*V{i};
    end
    P = -(1/N)*ten_sum_all(D);
    E = shrinkage_t( P, lambda*mu/N );
    
    % compute optimality stats
    pres = 0;
    tdiff = cell( 1, N );
    for i = 1:N
        tdiff{i} = X{i} + E - T;
    end
    pres = tensor_array_norm(tdiff);
    
    Ediff = E - Ep;
    pres = pres / (sqrt(N)*tnorm);
    dres = norm(tenmat(Ediff,1)) / norm(tenmat(Ep,1));
    Y = ten_sum_all(X) / N;
    rel_err = norm(tenmat(Y-data.X,1)) / norm(tenmat(data.X,1));
    
    % print
    if verbose
        if mod(iter,20)==0
    fprintf('Iter: %d,   pinf: %3.2e,   dinf: %3.2e,    rel_err: %3.2e\n', iter, pres, dres, rel_err );
        end
    end
    
    if max(pres, dres) < params.opt_tol
%     if pres < params.opt_tol
        break;
    end
    
    % update Lagrange multipliers
    for i = 1:N
        V{i} = V{i} - tdiff{i}/mu;
    end
    
%     if rem(iter,10) == 0
%         mu = max(mu*0.9, params.mu_min);
%     end
end

results.X = Y;
results.E = E;
results.V = V;
results.T = T;
results.U = U;
results.iter = iter;
results.cpu = toc;
results.mu = mu;
results.lambda = lambda;



end
