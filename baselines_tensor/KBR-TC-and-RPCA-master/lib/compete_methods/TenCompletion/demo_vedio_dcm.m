%   clear 
  close all;
  addpath(genpath('C:\Users\XieQ\Documents\MATLAB\BoYiA\TensorCompletion'));
%   allfiles=dir('C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\tomato\org\');
%   allfiles=dir('C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\ocean\org\');%
%   allfiles=dir('C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\hyperspectral
%   image sequence\org\');
%   allfiles=dir('\Users\admin\Desktop\新建文件夹\TensorCompletion\data\brainMRI\org\');
% allfiles =dir('C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\INCISIX1\');
%  allfiles=dir('C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\Medical_image\KNIX\Knee (R)\Loc (Right) - 1\');
% allfiles=dir('C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\DIASTOLIX\CorCTALow  2.0  B25f 0-95%\')
% % allfiles=dir('C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\ANONYMIZE\BRAIN\Ax PD&T2 FSE - 4\');
% allfiles=dir('C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\Medical_image\INCISIX1\');
% path='C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\Medical_image\INCISIX1\';
allfiles=dir('C:\Users\XieQ\Documents\MATLAB\BoYiA\TensorCompletion\data\Medical_image\MRIX\MRIX LUMBAR\Thoracic; Lumbar\Ax T2 frFSE S - 5\');
path='C:\Users\XieQ\Documents\MATLAB\BoYiA\TensorCompletion\data\Medical_image\MRIX\MRIX LUMBAR\Thoracic; Lumbar\Ax T2 frFSE S - 5\';

% allfiles=dir('C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\BRAINIX\T1-3D-FFE-C - 801\');
% allfiles=dir('C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\PNEUMATIX\PNEUMATIX\Cardiovascular Heart-Cardiac Function\fl3d-cor\');
% allfiles=dir('C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\MRIX\MRIX LUMBAR\Thoracic; Lumbar\Ax T2 frFSE S - 5\');
% allfiles=dir('C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\PANORAMIX\PANORAMIX\PANORAMIX\Abdomen 1ABD_PEL_AAA\Abd-Pel w-c  3.0  B30f\');
% path='C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\PANORAMIX\PANORAMIX\PANORAMIX\Abdomen 1ABD_PEL_AAA\Abd-Pel w-c  3.0  B30f\';
% path='C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\MRIX\MRIX LUMBAR\Thoracic; Lumbar\Ax T2 frFSE S - 5\';

% path='C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\PNEUMATIX\PNEUMATIX\Cardiovascular Heart-Cardiac Function\fl3d-cor\';
% path='C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\BRAINIX\T1-3D-FFE-C - 801\';
% path='C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\ANONYMIZE\BRAIN\Ax PD&T2 FSE - 4\';
% path='C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\DIASTOLIX\CorCTALow  2.0  B25f 0-95%\';
% path='C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\Medical_image\KNIX\Knee (R)\Loc (Right) - 1\';
% path='C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\INCISIX1\';
%   path='\Users\admin\Desktop\新建文件夹\TensorCompletion\data\brainMRI\org\'
%  path ='C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\hyperspectral image sequence\org\';
%   path='C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\ocean\org\';
%   path='C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\tomato\org\';
  
  fullnm0=strcat(path,allfiles(4).name);
  [m,n]=size(dicomread(fullnm0));
  num=length(allfiles)-2;
  T0=zeros(m,n,num);
  kk=1;
for i=1:length(allfiles)
     name=allfiles(i).name;
     [pathstr1, name1, ext1] = fileparts(name);
     if strcmp(ext1,'.dcm')
         
    
         fullnm=strcat(path,name);
         temp=dicomread(fullnm);
         T0(:,:,kk)=temp;
         kk=kk+1;
         
     end
end

  
Num=n*m*(num);
Omega=randperm(Num,fix(0.3*Num));
T=ones(m,n,num)*max(T0(:)); % 65535
T(Omega)=T0(Omega);

% a=T(:,:,:,1);
% figure,imshow(uint8(a));
% aa=T0(:,:,:,1)
% figure,imshow(uint8(aa));


%%
alpha =[1,1,1];
alpha = alpha / sum(alpha);


maxIter =200;% 300;
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






