xyloObj =  VideoReader('C:\Users\admin\Desktop\新建文件夹\TensorCompletion\data\Video\tennis_sif.avi');
% xyloObj = mmreader('xylophone.mpg');
nFrames = xyloObj.NumberOfFrames;
vidHeight = xyloObj.Height;
vidWidth = xyloObj.Width;

% Preallocate movie structure.

for i=1:nFrames
    mov(i) =struct('cdata', zeros(vidHeight, vidWidth, 3, 'uint8'),...
           'colormap', []);
end
% Read one frame at a time.
for k = 1 : nFrames
    mov(k).cdata = read(xyloObj, k);
end

% Size a figure based on the video's width and height.
% hf = figure;
% set(hf, 'position', [150 150 vidWidth vidHeight])

% Play back the movie once at the video's frame rate.
% movie(hf, mov, 1, xyloObj.FrameRate);


% Convert the movie into double
T0=zeros(vidHeight,vidWidth,3,nFrames);
for k=1:nFrames
    T0(:,:,:,k)=mov(k).cdata;
end

%
Num=vidHeight*vidWidth*3*nFrames;
Omega=randperm(Num,fix(0.5*Num));
T=ones(vidHeight,vidWidth,3,nFrames)*255;
T(Omega)=T0(Omega);

% a=T(:,:,:,1);
% figure,imshow(uint8(a));
% aa=T0(:,:,:,1)
% figure,imshow(uint8(aa));


%%
alpha =[1,1,1,1];
alpha = alpha / sum(alpha);


maxIter = 150;
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

