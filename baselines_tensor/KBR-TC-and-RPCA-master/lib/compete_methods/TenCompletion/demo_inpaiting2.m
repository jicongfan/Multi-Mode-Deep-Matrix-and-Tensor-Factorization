clear;
close all; 

% addpath('mylib/');
% addpath(genpath('C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data'));

T0 = double(imread('org1.png'));
[m,n,l]=size(T0);
Omega=randperm(m*n*l,fix(0.1*m*n*l));

T=zeros(m,n,l);
T(Omega)=T0(Omega);


%alpha = [1, 1, 1e-3];
alpha =[1,1,1];
alpha = alpha / sum(alpha);


maxIter = 300;
epsilon = 1e-5;

% "X" returns the estimation, 
% "errList" returns the list of the relative difference of outputs of two neighbor iterations 

%% High Accuracy LRTC (solve the original problem, HaLRTC algorithm in the paper)
rho = 1e-6;
[X_H, errList_H] = HaLRTC(...
    T, ...                       % a tensor whose elements in Omega are used for estimating missing value
    Omega,...               % the index set indicating the obeserved elements
    alpha,...                  % the coefficient of the objective function, i.e., \|X\|_* := \alpha_i \|X_{i(i)}\|_* 
    rho,...                      % the initial value of the parameter; it should be small enough  
    maxIter,...               % the maximum iterations
    epsilon...                 % the tolerance of the relative difference of outputs of two neighbor iterations 
    );

norm(X_H(:)-T0(:),'fro')/norm(T0(:),'fro')

%% Mcp LRTC
epsilon=1e-3
[X_Mcp, errList_Mcp] = McpLRTC(...
    T,...
    Omega,...
    alpha,...
    rho,...
    maxIter, ...
    epsilon);
norm(X_Mcp(:)-T0(:),'fro')/norm(T0(:),'fro')

