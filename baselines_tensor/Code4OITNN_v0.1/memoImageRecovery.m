clear; close all; clc;
addpath('./Functions4OITNN');
%% ++++++++++++++Experiment setting++++++++++
sparseRatio=0.3;
NoiseSignalRatio=0.2;

methodList={'RTD:OITNN-O','RTD:OITNN-L','RTD:TNN'};
isMethodOn=[1 1 1];

nImages = 9;
imgFolder='./Image4Test/';
imgPrefix='';
%++++++++++++++Experiment setting++++++++++

%% ++++++++++++Run algorithms on each image++++++++++++++
for imgID=1:nImages
    imgPath=[imgFolder imgPrefix num2str(imgID) '.jpg'];
     
    %% +++Signal and Noise Generation+++
    T=double(imread(imgPath));
    L=T/h_inf_norm(T);
    
    sz=size(L);K=length(sz);D=prod(sz);
    d1=sz(1);d2=sz(2);d3=sz(3);
    % L_infty norm
    alphaL=max( abs(L(:)));
    alphaS=alphaL;
    % Sparse corruption observation mask
    B=rand(sz)<sparseRatio;
    Ttmp1=rand(sz)<0.5;
    Ttmp2=ones(sz);
    S=(Ttmp2-2*Ttmp1)*alphaS;
    S=S.*B;
    % Gaussian noise
    sigma=NoiseSignalRatio*h_tnorm(L)/sqrt(D);
    G=randn(sz)*sigma;
    % The observed tensor
    Y=L+S+G;
    % +++Signal and Noise Generation+++
        
    %%  RTD: OITNN-O
    iModel =1;
    if isMethodOn(iModel)
        
        %++++++Model Parameters++++++
        % The parameters may be not optimal
        % but it takes long time to tune
        ClO=8e0;
        CsO=1*ClO/sqrt(sz(1)*sz(3));
        lamL=ClO;
        lamS=CsO;
        vW = [0.1 1 0.1];
        vW=vW/sum(vW);
        %++++++Model Parameters++++++
        
        %++++++Algorithm Paramters+++++ 
        rho=1e0; nu=1;
        %++++++Algorithm Paramters+++++
        
        % +++Observation+++
        obs.tY=Y;
        % +++Observation+++
        
        %+++++Algorithm options+++++
        optsO.para.lambdaL=lamL;
        optsO.para.lambdaS=lamS;
        optsO.para.alpha=alphaL;
        optsO.para.rho=rho;
        optsO.para.nu=nu;
        optsO.para.vW=vW;
        optsO.MAX_ITER_OUT=500;
        optsO.MAX_RHO=1e10;
        optsO.MAX_EPS=5e-4;
        optsO.verbose=1;
        %+++++Algorithm options+++++
        
        %+++++construct memo+++++
        memoO=h_construct_memo_v2(optsO);
        memoO.truthL=L;
        memoO.truthS=S;
        optsO.showImg=0;
        %+++++construct memo+++++
        
        %++++++++++++++Run++++++++++++++
        t=clock;
        memoO=f_rtd_OITNN_O(obs,optsO,memoO);
        t=etime(clock,t);
        %++++++++++++++Run++++++++++++++
    end
    
    %%  RTD: OITNN-L
    iModel =2;
    if isMethodOn( iModel)
        
        % ++++++Model Parameters++++++
        % The parameters may be not optimal
        % but it takes long time to tune
        CSigma=0.6; %0.25
        ClL=CSigma*19; %1.7e1; 20;
        CsL=CSigma*0.00654321/0.11; % 0.06
        lamL=0;
        for k=1:K
            lamL=max(lamL,f_tensor_spectral_norm(f_3DReshape(G,k)));
        end

        lamL=ClL*lamL;
        lamS=(f_l_inf_norm(G) +  2*K*alphaS);
        lamS=CsL*lamS;
        
        vV = [1 0.0288 1];
        vV=vV/sum(vV);
        %++++++Model Parameters++++++
        
        
        %++++++Algorithm Paramters+++++   
        rho=1e0; nu=1;
        %++++++Algorithm Paramters+++++   
        
        % +++Observation+++
        obs.tY=Y;
        % +++Observation+++
        
        %+++++Algorithm options+++++
        optsL.obs=obs;
        optsL.para.lambdaL=lamL;
        optsL.para.lambdaS=lamS;
        optsL.para.alpha=alphaL;
        optsL.para.rho=rho;
        optsL.para.nu=nu;
        optsL.para.vW=vV;
        optsL.MAX_ITER_OUT=300;
        optsL.MAX_RHO=1e10;
        optsL.MAX_EPS=3e-4;
        optsL.verbose=1;
        optsL.showImg=0;
        %+++++Algorithm options+++++
        
        %+++++construct memo+++++
        memoL=h_construct_memo_v2(optsL);
        memoL.truthL=L;
        memoL.truthS=S;
        %+++++construct memo+++++
        
        %++++++++++++++Run++++++++++++++
        t=clock;
        memoL=f_rtd_OITNN_L(obs,optsL,memoL);
        t=etime(clock,t);
        %++++++++++++++Run++++++++++++++
    end
    
    %% RTD: TNN
    iModel =3;
    if isMethodOn( iModel)
        %++++++Model Parameters++++++
        CTNN=5e0;
        lamL=CTNN;
        lamS=lamL/sqrt(sz(1)*sz(3));
        %++++++Model Parameters++++++
        
        
        %++++++Algorithm Paramters+++++ 
        rho=1e-3; nu=1.1;
        %++++++Algorithm Paramters+++++ 
        
        % +++Observation+++
        obs.tY=Y;
        % +++Observation+++
        
        %+++++Algorithm options+++++
        opts.obs=obs;
        optsTNN.para.lambdaL=lamL;
        optsTNN.para.lambdaS=lamS;
        optsTNN.para.alpha=alphaL;
        optsTNN.para.rho=rho;
        optsTNN.para.nu=nu;
        optsTNN.MAX_ITER_OUT=300;
        optsTNN.MAX_RHO=1e10;
        optsTNN.MAX_EPS=1e-4;
        optsTNN.verbose=1;
        optsTNN.showImg=0;
        %+++++Algorithm options+++++
        
        %+++++construct memo+++++
        memoTNN=h_construct_memo_v2(optsTNN);
        memoTNN.truthL=L;
        memoTNN.truthS=S;
        %+++++construct memo+++++
        
        %++++++++++++++Run++++++++++++++
        t=clock;
        memoTNN=f_rtd_TNN(obs,optsTNN,memoTNN);
        t=etime(clock,t);
        %++++++++++++++Run++++++++++++++ 
    end

end
