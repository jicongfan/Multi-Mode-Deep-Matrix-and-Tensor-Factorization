clear; close all; clc;
addpath('./Functions4OITNN');
%% ++++++++++++++Experiment setting++++++++++
nImages=9;
nObsRatioSetting=2;
methodList={'TC:OITNN-O','TC:OITNN-L','TC:TNN'};
isMethodOn=[1,1,1];
%++++++++++++++Experiment setting++++++++++

%% ++++++++++++Run algorithms on each image++++++++++++++
for iImg=1:nImages
    imgID=iImg;
    imgFolder='./Image4Test/';
    imgPrefix='';
    imgPath=[imgFolder  imgPrefix num2str(imgID) '.jpg'];
    for iRatio=1:nObsRatioSetting
        
        observationRatio=iRatio*0.1;
        noiseSignalRatio=0;
        
        T=double(imread(imgPath));
        sz=size(T); 
        
        % +++Observation+++
        signalMag=h_tnorm(T)/sqrt(prod(sz));
        sigma=noiseSignalRatio*signalMag;
        obs.tsize=sz;
        [obs.y,obs.idx]=f_P_Rand_Omega(double(T),observationRatio);
        vE = randn(size(obs.y))*sigma;
        obs.y=obs.y+vE;
        T_miss=zeros(sz);
        T_miss(obs.idx)=obs.y;
        E = zeros(sz);
        E(obs.idx)=vE;
        % +++Observation+++

        %% TC: OITNN-O
        iMethod=1;
        if isMethodOn(iMethod)
            
            %++++++Model Parameters++++++
            vW=[1, 1, 1];
            vW=vW/sum(vW);
            %++++++Model Parameters++++++
            
            %++++++Algorithm Paramters+++++
            vRho=1e-8*[1,1,1];
            vNu=1.1*[1,1,1];
            %++++++Algorithm Paramters+++++
            
            %+++++Algorithm options+++++
            optsOverlap.MAX_ITER_OUT=300;
            optsOverlap.MAX_RHO=1e5;
            optsOverlap.MAX_EPS=1e-4;
            optsOverlap.verbose=1;
            optsOverlap.para.alpha=vW;
            optsOverlap.para.vRho=vRho;
            optsOverlap.para.vNu=vNu;
            %+++++Algorithm options+++++
            
            %+++++construct memo+++++
            memoOverlap=h_construct_memo_v2(optsOverlap);
            memoOverlap.truth=T;
            %+++++construct memo+++++
            
            %++++++++++++++Run++++++++++++++
            tic
            memoOverlap=f_tc_OITNN_O(obs,optsOverlap,memoOverlap);
            toc
            %++++++++++++++Run++++++++++++++
        end
        %------------------------------------
        
        %% TC: OITNN-L
        % -------------Options-------------
        iMethod=2;
        if isMethodOn(iMethod)
            
            %++++++Model Parameters++++++
            vV=[1 1 1];
            vV=vV/sum(vV);
            %++++++Model Parameters++++++
            
            %++++++Algorithm Paramters+++++
            rho=3e-8; nu=1.1;
            %++++++Algorithm Paramters+++++
            
            %+++++Algorithm options+++++
            optsLatent.MAX_ITER_OUT=200;
            optsLatent.MAX_RHO=1e5;
            optsLatent.MAX_EPS=1e-6;
            optsLatent.verbose=1;
            optsLatent.para.alpha=vV;
            optsLatent.para.rho=rho;
            optsLatent.para.nu=nu;
            %+++++Algorithm options+++++
            
            %+++++construct memo+++++
            memoLatent=h_construct_memo_v2(optsLatent);
            memoLatent.truth=T;
            %+++++construct memo+++++
            
            %++++++++++++++Run++++++++++++++            
            tic
            memoLatent=f_tc_OITNN_L(obs,optsLatent,memoLatent);
            toc
            %++++++++++++++Run++++++++++++++
        end
        
        %% TC: TNN
        iMethod=3;
        if isMethodOn(iMethod)
            
            %++++++Algorithm Parameters++++++
            rho=4e-5;
            nu=1.1;
            %++++++Algorithm Parameters++++++
            
            %+++++Algorithm options+++++
            optTNN.MAX_ITER_OUT=100;
            optTNN.MAX_RHO=1e5;
            optTNN.MAX_EPS=1e-4;
            optTNN.verbose=1;
            optTNN.para.rho=rho;
            optTNN.para.nu=nu;
            %+++++Algorithm options+++++
            
            %+++++construct memo+++++
            memoTNN=h_construct_memo_v2(optTNN);
            memoTNN.truth=double(T);
            %+++++construct memo+++++
            
            %++++++++++++++Run++++++++++++++
            tic
            memoTNN=f_tc_TNN(obs,optTNN,memoTNN);
            toc
            %++++++++++++++Run++++++++++++++ 
        end
    end
end
% ++++++++++++Run Algorithms++++++++++++++
