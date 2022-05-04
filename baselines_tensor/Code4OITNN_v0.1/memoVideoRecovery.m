clear; close all; clc;
addpath('./Functions4OITNN');
%% ++++++++++++++Experiment setting++++++++++
sparseRatio=0.15;
noiseSignalRatio=0.15;

methodList={'RTD:OITNN-O','RTD:OITNN-L','RTD:TNN'};
isMethodOn=[1 1 1];
%++++++++++++++Experiment setting++++++++++

%% ++++++++ Load Video++++++++++
load('akiyo_qcif_RGB_30.mat');

% We use a small number of frames in this demo
nFrame = 8;
T= T(:,:,:,1:nFrame);

% Tensor recovery performs better after this permutation
T = permute(T,[1 4 3 2]); 

sz =size(T); D=prod(sz);
d1 = sz(1); d2 = sz(2); d3 = sz(3); d4 = sz(4);
%++++++++ Load Video++++++++++

%% +++Signal and Noise Generation+++
infT=h_inf_norm(T); L=T/infT;

% L_infty norm
alphaL=max( abs(L(:))); alphaS=alphaL;

% Sparse corruption observation mask
B=rand(sz)<sparseRatio;
Ttmp1=rand(sz)<0.5; Ttmp2=ones(sz);
S=(Ttmp2-2*Ttmp1)*alphaS;
S=S.*B;

% Gaussian noise
sigma=noiseSignalRatio*h_tnorm(L)/sqrt(D);
G=randn(sz)*sigma;

% The observed tensor
Y=L+S+G;
% +++Signal and Noise Generation+++

%%  RTD: OITNN-O
iModel =1;
if isMethodOn(iModel)
    
    %++++++Model Parameters++++++
    % The parameters are not optimal
    % but it takes long time to tune      
    vW=[1,0.01,0.01,0.01];
    vW=vW/sum(vW);
    
    lamO=1e0;
    S2L=1/sqrt(max(d1*d2,d2*d3*d4));
    muO=lamO*S2L;
    %++++++Model Parameters++++++
    
    %++++++Algorithm Paramters+++++     
    vRho=[1e-4 1e-8 1e-8 1e-8];
    vNu=[1.1 1 1 1];
    rho=1e-4;
    nu=1.1;
    %++++++Algorithm Paramters+++++  
    
    % +++Observation+++
    obs.tY=Y;
    % +++Observation+++
    
    %+++++Algorithm options+++++
    optsOverlap.para.lambdaL=lamO;
    optsOverlap.para.lambdaS=muO;
    optsOverlap.para.alpha=alphaL;
    optsOverlap.para.vRho=vRho;
    optsOverlap.para.vNu=vNu;
    optsOverlap.para.rho=rho;
    optsOverlap.para.nu=nu;
    optsOverlap.para.vW=vW;
    optsOverlap.MAX_ITER_OUT=100;
    optsOverlap.MAX_RHO=1e10;
    optsOverlap.MAX_EPS=5e-4;
    optsOverlap.verbose=1;
    optsOverlap.showImg=0;
    %+++++Algorithm options+++++
    
    %+++++construct memo+++++
    memoOverlap=h_construct_memo_v2(optsOverlap);
    memoOverlap.truthL=L;
    memoOverlap.truthS=S;
    %+++++construct memo+++++
    
    %++++++++++++++Run++++++++++++++
    t=clock;
    memoOverlap=f_rtd_OITNN_O_tune_rho(obs,optsOverlap,memoOverlap);
    t=etime(clock,t);
    %++++++++++++++Run++++++++++++++
    
end

%%  RTD: OITNN-L
iModel =2;
if isMethodOn(iModel)
    %++++++Model Parameters++++++
    % The parameters are not optimal
    % but it takes long time to tune  
    v1 = sqrt(max(d1*d2,d2*d3*d4));
    v2 = sqrt(max(d2*d3,d3*d1*d4));
    v3 = sqrt(max(d3*d4,d4*d1*d2));
    v4 = sqrt(max(d4*d1,d1*d2*d3));
    vV=[v1 v2 v3 v4];
    muL=1e-5;
    lamL=muL;
    %++++++Model Parameters++++++
    
    %++++++Algorithm Paramters+++++    
    rho=1e-12; nu=1.1;
    %++++++Algorithm Paramters+++++
    

    
    % +++Observation+++
    obs.tY=Y;
    % +++Observation+++
    
    %+++++Algorithm options+++++
    optsLatent.obs=obs;
    optsLatent.para.lambdaL=lamL;
    optsLatent.para.lambdaS=muL;
    optsLatent.para.alpha=alphaL;
    optsLatent.para.rho=rho;
    optsLatent.para.nu=nu;
    optsLatent.para.vW=vV;
    optsLatent.MAX_ITER_OUT=800;
    optsLatent.MAX_RHO=1e10;
    optsLatent.MAX_EPS=1e-8;
    optsLatent.verbose=1;
    optsLatent.showImg=0;
    %+++++Algorithm options+++++
    
    %+++++construct memo+++++
    memoLatent=h_construct_memo_v2(optsLatent);
    memoLatent.truthL=L;
    memoLatent.truthS=S;
    %+++++construct memo+++++
    
    %++++++++++++++Run++++++++++++++
    t=clock;
    memoLatent=f_rtd_OITNN_L(obs,optsLatent,memoLatent);
    t=etime(clock,t);
    %++++++++++++++Run++++++++++++++
    memo=memoLatent;
end

%% RTD: TNN
iModel =3;
if isMethodOn(iModel)
    
    %++++++Model Parameters++++++
    CTNN_SvL=1/sqrt(d1*d4);
    lamL=8e0;
    muL=lamL*CTNN_SvL;
    %++++++Model Parameters++++++
    
    %++++++Algorithm Paramters+++++    
    rho=1e0; nu=1;
    %++++++Algorithm Paramters+++++   
    
    iChannel=3;
    % +++Observation+++
    obs.tY=squeeze(Y(:,:,iChannel,:));
    % +++Observation+++
    
    %+++++Algorithm options+++++
    opts.obs=obs;
    optsTNN.para.lambdaL=lamL;
    optsTNN.para.lambdaS=muL;
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
    memoTNN.truthL=squeeze(L(:,:,iChannel,:));
    memoTNN.truthS=squeeze(S(:,:,iChannel,:));
    %+++++construct memo+++++
    
    %++++++++++++++Run++++++++++++++
    t=clock;
    memoTNN=f_rtd_TNN(obs,optsTNN,memoTNN);
    t=etime(clock,t);
    %++++++++++++++Run++++++++++++++
end
