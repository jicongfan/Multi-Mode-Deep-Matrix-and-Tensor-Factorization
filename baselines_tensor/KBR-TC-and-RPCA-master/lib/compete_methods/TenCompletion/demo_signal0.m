
close all;
% addpath(genpath('C:\Users\admin\Desktop\�½��ļ���\tensor_toolbox'));
% addpath('mylib/');

I=50;
r=25;
samprato=[0.1: 0.03: 0.6];
len =length(samprato);

results_H=zeros(1,len);
results_Mcp=zeros(1,len);
results_Scad=zeros(1,len);

core = tensor(rand(r,r,r),[r r r]); %<-- The core tensor.
U = {rand(I,r), rand(I,r), rand(I,r)}; %<-- The matrices.
X = ttensor(core,U); %<-- Create the ttensor.
X1=double(X);

for q=1:len


sr=samprato(q);
Num=I*I*I;
Omega=randperm(Num,fix(Num*sr));
T=zeros(I,I,I);
T(Omega)=X1(Omega);
   
alpha = [1, 1, 1];
alpha = alpha / sum(alpha);
maxIter =400;

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

results_H(q)=norm(X_H(:)-X1(:),'fro')/norm(X1(:),'fro')

%%
epsilon=1e-6;
[X_Mcp, errList_Mcp] = McpLRTC(...
    T,...
    Omega,...
    alpha,...
    rho,...
    maxIter, ...
    epsilon);

results_Mcp(q)=norm(X_Mcp(:)-X1(:),'fro')/norm(X1(:),'fro')
%% 
[X_Scad, errList_Mcp] = SCADLRTC(...
    T,...
    Omega,...
    alpha,...
    rho,...
    maxIter, ...
    epsilon);

results_Scad(q)=norm(X_Scad(:)-X1(:),'fro')/norm(X1(:),'fro')
   end



