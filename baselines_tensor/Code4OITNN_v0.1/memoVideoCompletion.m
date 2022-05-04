clear; close all; clc;
addpath('./Functions4OITNN');

%% ++++++++++++++Experiment setting++++++++++
obsRatio=0.15;
noiseSignalRatio=0;
methodList={'TC:OITNN-O','TC:OITNN-L','TC:TNN'};
isMethodOn=[1,1,1];


%% ++++++++ Load Video++++++++++
load('akiyo_qcif_RGB_30.mat');

% We use a small number of frames in this demo
nFrames=8;
T= T(:,:,:,1:nFrames);

% Tensor recovery performs better after this permutation
T = permute(T,[1 4 3 2]);
sz=size(T); 

%+++++++++++++Signal Generation+++++++++++++
signalMag=h_tnorm(T)/sqrt(prod(sz));
sigma=noiseSignalRatio*signalMag;
obs.tsize=sz;
[obs.y,obs.idx]=f_P_Rand_Omega(double(T),obsRatio);

vE = randn(size(obs.y))*sigma;
obs.y=obs.y+vE;
T_miss=zeros(sz);
T_miss(obs.idx)=obs.y;
E = zeros(sz);
E(obs.idx)=vE;
%+++++++++++++Signal Generation+++++++++++++

%% TC: OITNN-O
iMethod=1;
if isMethodOn(iMethod)
    %++++++Model Parameters++++++
    % The parameters are not optimal
    % but it takes long time to tune    
    vW=[0.1,10,0.1,0.1];
    vW=vW/sum(vW);
    %++++++Model Parameters++++++
    
    %++++++Algorithm Paramters+++++
    vRho=[1e-8 1e-4 1e-8 1e-8];
    vNu=[1 1.1 1 1];
    %++++++Algorithm Paramters+++++
    
    %+++++Algorithm options+++++
    optsOITNNO.MAX_ITER_OUT=300;
    optsOITNNO.MAX_RHO=1e5;
    optsOITNNO.MAX_EPS=1e-4;
    optsOITNNO.verbose=1;
    optsOITNNO.para.alpha=vW;
    optsOITNNO.para.vRho=vRho;
    optsOITNNO.para.vNu=vNu;
    %+++++Algorithm options+++++
    
    %+++++construct memo+++++
    memoOITNNO=h_construct_memo_v2(optsOITNNO);
    memoOITNNO.truth=T;
    %+++++construct memo+++++
    
    %++++++++++++++Run++++++++++++++
    tic
    memoOITNNO=f_tc_OITNN_O(obs,optsOITNNO,memoOITNNO);
    toc
    %++++++++++++++Run++++++++++++++
end
%------------------------------------

%% TC: OITNN-L
iMethod=2;
if isMethodOn(iMethod)
    %++++++Model Parameters++++++
    % The parameters are not optimal
    % but it takes long time to tune      
    vW=[1 1e-3 1 1];
    vW=vW/sum(vW);
    %++++++Model Parameters++++++
    
    %++++++Algorithm Paramters+++++
    rho=1e-10; nu=1.05;
    %++++++Algorithm Paramters+++++
    
    %+++++Algorithm options+++++
    optsOITNNL.MAX_ITER_OUT=200;
    optsOITNNL.MAX_RHO=1e5;
    optsOITNNL.MAX_EPS=1e-6;
    optsOITNNL.verbose=1;
    optsOITNNL.para.alpha=vW;
    optsOITNNL.para.rho=rho;
    optsOITNNL.para.nu=nu;
    %+++++Algorithm options+++++
    
    %+++++construct memo+++++
    memoOITNNL=h_construct_memo_v2(optsOITNNL);
    memoOITNNL.truth=T;
    %+++++construct memo+++++
    
    %++++++++++++++Run++++++++++++++
    tic
    memoOITNNL=f_tc_OITNN_L(obs,optsOITNNL,memoOITNNL);
    toc
    %++++++++++++++Run++++++++++++++
end

%% TC: TNN
iMethod=3;
if isMethodOn(iMethod)
    %++++++Algorithm Paramters+++++
    rho=1e-4; nu=1.1;
    %++++++Algorithm Paramters+++++
    
    %+++++Algorithm options+++++
    optTNN.MAX_ITER_OUT=200;
    optTNN.MAX_RHO=1e5;
    optTNN.MAX_EPS=1e-4;
    optTNN.verbose=1;
    optTNN.para.rho=rho;
    optTNN.para.nu=nu;
    optTNN.para.lmb1=1;
    %+++++Algorithm options+++++
    
    hatTNN = T*0;
    TobsNN = T*0;
    TobsNN(obs.idx)=obs.y;
    BNN = T*0;
    BNN(obs.idx)=1;

    memoTNN=h_construct_memo_v2(optTNN);
    for iF=1:3
        memoTNN.truth=double(squeeze(T(:,:,iF,:)));
        Bnn=squeeze(BNN(:,:,iF,:));
        idx = find(Bnn>0);
        Mobs=squeeze(TobsNN(:,:,iF,:));
        obsTNN.y = Mobs(idx);
        obsTNN.idx = idx;
        obsTNN.tsize = size(Bnn);
        %-------------Memo----------------
        
        %------------------------------------
        tic
        memoTNN=f_tc_TNN(obsTNN,optTNN,memoTNN);
        toc
        %-----------------------------------
        hatTNN(:,:,iF,:)=memoTNN.T_hat;
    end
    T_hat=hatTNN;
    %------------------------------------
end



