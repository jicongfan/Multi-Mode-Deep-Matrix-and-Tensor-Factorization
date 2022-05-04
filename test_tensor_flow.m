clc
clear all
warning off
%rng(1)
load('td_flow.mat');
X=X/max(X(:));
mr=[0.97];
%%
for qq=1:length(mr)
for pp=1:1
X=tensor(X);
M=zeros(size(X));
miss_ratio=mr(qq);
M(randperm(prod(size(M)),round(prod(size(M))*(1-miss_ratio))))=1;
M=tensor(M,size(X));
Xm=times(X,M);
k=0;
% % %%
% % disp('LRTC...')
% % k=k+1;  alg_name{k}='HaLRTC';
% % alpha=size(X);  alpha=alpha/sum(alpha);
% % rho = 5*1e-3; maxIter=200;  epsilon=1e-5;
% % tic
% % % [X_H, errList_H] = HaLRTC(Xm.data,logical(M.data),alpha,rho,maxIter,epsilon,Xm.data);
% % [X_H, errList] = FaLRTC(Xm.data,logical(M.data), alpha, 1, 1, 1, maxIter, epsilon);
% % T(k)=toc;
% % Xr{k}=tensor(X_H);
% % %% TenALS 2014
% % % disp('TenALS...')
% % k=k+1;  alg_name{k}='TenALS';
% % tic
% % r_init=1;% 2 5 10
% % [V{1} V{2} V{3} s dist] = TenALS(Xm.data, M.data, r_init, 1, 20, 1e-4);% iter 100 or 20
% % T(k)=toc;
% % Xr{k}=full(ktensor(s,V));
% % %% TMac Inverse Problems and Imaging 2015
% % disp('Tmac...')
% % k=k+1;  alg_name{k}='TMac';
% % r_init=10;
% % opts = [];  opts.maxit = 1000;   opts.tol = 1e-5; 
% % opts.Mtr = M;   opts.alpha_adj = 0;
% % opts.rank_adj = 1*ones(1,3);   opts.rank_min = 10*ones(1,3);    opts.rank_max = [5 20 20];
% % Mknown=find(M.data==1); data=Xm.data;   data=data(Mknown);
% % EstCoreNway=[1 1 1];
% % tic
% % [X_dec,Y_dec,Out_dec] = TMac(data,Mknown,size(M),EstCoreNway,opts);
% % Xrec = zeros(size(M));
% % for i = 1:3
% %     Xrec = Xrec+Out_dec.alpha(i)*Fold(X_dec{i}*Y_dec{i},size(M),i);
% % end
% % Xr{k}=tensor(Xrec);
% % %% KBR-TC TPAMI 2017
% % disp('KBR-TC...')
% % k=k+1;  alg_name{k}='KBR-TC';
% % Par2.tol     = 1e-4;    Par2.maxIter = 300;    Par2.maxSubiter = 1;%lambda
% % Par2.rho = 1.05;    Par2.mu = 1e-5; Par2.lambda = 0.01; % 0.01 is even better
% % Par2.rank=size(X);
% % X_01=Xm.data;vmin=min(X_01(:));vmax=max(X_01(:));X_01=(X_01-vmin)/(vmax-vmin);
% % tic;
% % X_KBR = KBR_TC(X_01, logical(M.data), Par2);X_KBR=X_KBR*(vmax-vmin)+vmin;
% % Xr{k}=tensor(X_KBR);
% % T(k) = toc;
% % %%
% % disp('Tensor ring...')
% % k=k+1;
% % r=5*ones(1,3); % TR-rank 
% % maxiter=500; % maxiter 300~500
% % tol=1e-6; % 1e-6~1e-8
% % Lambda=1e1; % usually 1~10
% % ro=1.1; % 1~1.5
% % K=1e-2; % 1e-1~1e0 
% % [Xr{k},~,~]=TRLRF(Xm.data,M.data,r,maxiter,K,ro,Lambda,tol);
% % %%
% % k=k+1;
% % Omega=find(M.data==1);
% % obs.y=Xm(Omega);
% % obs.idx=Omega;
% % obs.tsize=size(M);
% % vW=[1, 1, 1];
% % vW=vW/sum(vW)/10000;
% % vRho=1e-8*[1,1,1];
% % vNu=1.1*[1,1,1];
% % optsOverlap.MAX_ITER_OUT=300;
% % optsOverlap.MAX_RHO=1e5;
% % optsOverlap.MAX_EPS=1e-4;
% % optsOverlap.verbose=1;
% % optsOverlap.para.alpha=vW;
% % optsOverlap.para.vRho=vRho;
% % optsOverlap.para.vNu=vNu;
% % memoOverlap=h_construct_memo_v2(optsOverlap);
% % memoOverlap.truth=X.data;
% % memoOverlap=f_tc_OITNN_O(obs,optsOverlap,memoOverlap);
% % Xr{k}=memoOverlap.T_hat;
%%
disp('DMTF...')
k=k+1; 
s_G=[10 20 20];
s_NN{1}=[5 s_G(1) size(Xm,1)];
s_NN{2}=[10 s_G(2) size(Xm,2)];
s_NN{3}=[10 s_G(3) size(Xm,3)];
lambda_w=[1 1 10]*0.001;
lambda_z=[1 1 10]*0.001;
lambda_c=[0.001];
options.maxiter=3000;
%
[Xr{k},G,U,loss,NN] = M2DMTF_tensor(Xm,M,s_G,s_NN,lambda_w,lambda_z,lambda_c,options);
%%
for i=1:length(Xr)
if ~isempty(Xr{i})
re(pp,i)=norm(times(minus(Xr{i},X),minus(1,M)))/norm(times(X,minus(1,M)))*100;
end
end
end

end

