%   clear 
  close all;
  addpath(genpath('C:\Users\admin\Desktop\新建文件夹\TensorCompletion'));
%   allfiles=dir('C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\tomato\org\');
allfiles=dir('C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\Video\tomato\org\');
%   allfiles=dir('C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\ocean\org\');%
%   allfiles=dir('C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\Hyper_Multi_Spectral\hyperspectral image sequence\org\');
%   allfiles=dir('\Users\admin\Desktop\新建文件夹\TensorCompletion\data\brainMRI\org\');
% allfiles=dir('C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\mutilspectral\superballs_ms\superballs_ms\');
path='C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\Video\tomato\org\';
% path='C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\mutilspectral\superballs_ms\superballs_ms\';
%   path='\Users\admin\Desktop\新建文件夹\TensorCompletion\data\brainMRI\org\'
%  path ='C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\Hyper_Multi_Spectral\hyperspectral image sequence\org\';
%   path='C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\ocean\org\';
%   path='C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\tomato\org\';
  
  fullnm0=strcat(path,allfiles(4).name);
  [m,n,l]=size(imread(fullnm0));
  num=length(allfiles)-3;
  T0=zeros(m,n,l,num);
  kk=1;
for i=1:length(allfiles)
     name=allfiles(i).name;
     [pathstr1, name1, ext1] = fileparts(name);
     if strcmp(ext1,'.png')
         
    
         fullnm=strcat(path,name);
         temp=imread(fullnm);
         T0(:,:,:,kk)=temp;
         kk=kk+1;
         
     end
end

  
Num=n*m*l*(num);
Omega=randperm(Num,fix(0.1*Num));
T=ones(m,n,l,num)*255;
T(Omega)=T0(Omega);

% a=T(:,:,:,1);
% figure,imshow(uint8(a));
% aa=T0(:,:,:,1)
% figure,imshow(uint8(aa));


%%
alpha =[1,1,1,1];
alpha = alpha / sum(alpha);


maxIter = 200;
epsilon = 1e-8;

% "X" returns the estimation, 
% "errList" returns the list of the relative difference of outputs of two neighbor iterations 
% High Accuracy LRTC (solve the original problem, HaLRTC algorithm in the paper)
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

% epsilon=1e-3
[X_Mcp, errList_Mcp] = McpLRTC(...
    T,...
    Omega,...
    alpha,...
    rho,...
    maxIter, ...
    epsilon);
norm(X_Mcp(:)-T0(:),'fro')/norm(T0(:),'fro')

%% SCAD LRTC
[X_Scad, errList_Mcp] = SCADLRTC(...
    T,...
    Omega,...
    alpha,...
    rho,...
    maxIter, ...
    epsilon);
norm(X_Scad(:)-T0(:),'fro')/norm(T0(:),'fro')






