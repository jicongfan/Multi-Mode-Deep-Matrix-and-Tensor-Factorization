
close all;
addpath(genpath('C:\Users\admin\Desktop\新建文件夹\tensor_toolbox'));
addpath('mylib/');
% 50*50*50 rank=20, SR=0.3
% 100*100*100 rank=20 SR=0.5 

I1=20;
I2=20;
I3=20;
I4=20;
I5=20;

r=4;
sr=0.2;

core = tensor(randn(r,r,r,r,r),[r r r r r]); %<-- The core tensor.
U = {randn(I1,r), randn(I2,r), randn(I3,r),randn(I4,r),randn(I5,r)}; %<-- The matrices.
X = ttensor(core,U); %<-- Create the ttensor.


X1=double(X);
Num=I1*I2*I3*I4*I5;
Omega=randperm(Num,fix(Num*sr));
T=zeros(I1,I2,I3,I4,I5);
T(Omega)=X1(Omega);
   
alpha = [1, 1, 1,1,1];
alpha = alpha / sum(alpha);
maxIter = 400;

%% 
epsilon = 1e-5;
rho = 1e-9;
[X_H, errList_H] = HaLRTC(...
    T, ...                       % a tensor whose elements in Omega are used for estimating missing value
    Omega,...               % the index set indicating the obeserved elements
    alpha,...                  % the coefficient of the objective function, i.e., \|X\|_* := \alpha_i \|X_{i(i)}\|_* 
    rho,...                      % the initial value of the parameter; it should be small enough  
    maxIter,...               % the maximum iterations
    epsilon...                 % the tolerance of the relative difference of outputs of two neighbor iterations 
    );

results_H=norm(X_H(:)-X1(:),'fro')/norm(X1(:),'fro')

%%
epsilon=1e-6;
[X_Mcp, errList_Mcp] = McpLRTC(...
    T,...
    Omega,...
    alpha,...
    rho,...
    maxIter, ...
    epsilon);

results_Mcp=norm(X_Mcp(:)-X1(:),'fro')/norm(X1(:),'fro')
%% 
[X_Scad, errList_Mcp] = SCADLRTC(...
    T,...
    Omega,...
    alpha,...
    rho,...
    maxIter, ...
    epsilon);

results_Scad=norm(X_Scad(:)-X1(:),'fro')/norm(X1(:),'fro')
 



