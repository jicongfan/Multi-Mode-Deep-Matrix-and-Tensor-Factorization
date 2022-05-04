clc
clear all
load('ml100k.mat');
M00=zeros(size(X));
M00(X~=0)=1;
sM=sum(M00,3);
cs=sum(sM);
ic=find(cs>50);
Xt=X(:,ic,:);
rs=sum(sM');
is=find(rs>50);
Xt=Xt(is,:,:);
X=Xt;
% rng(10)
for q=1:1
miss_ratio=0.7;
M_org=zeros(size(X));
M_org(X~=0)=1;
idm=find(M_org==1);
idt=sort(randperm(length(idm),round(length(idm)*(1-miss_ratio))));
idms=idm(idt);
M=zeros(size(X));
M(idms)=1;
M=tensor(M,size(M));
%
Xm=times(X,M);
%
r_init=20;%round(max(size(M))*mean(M(:)));
prod(size(M))*mean(M(:))/sum(size(M))
k=0;
%%
disp('LRTC...')
k=k+1;  alg_name{k}='HaLRTC';
alpha=size(X);  alpha=alpha/sum(alpha);
rho = 5*1e-3; maxIter=100;  epsilon=1e-4;
tic
% [X_H, errList_H] = HaLRTC(Xm.data,logical(M.data),alpha,rho,maxIter,epsilon,Xm.data);
[X_H, errList] = FaLRTC(Xm.data,logical(M.data), alpha, 1, 1, 1, maxIter, epsilon);
T(k)=toc;
Xr{k}=tensor(X_H);
%% TenALS 2014
disp('TenALS...')
k=k+1;  alg_name{k}='TenALS';
tic
r_init=3;
[V{1} V{2} V{3} s dist] = TenALS(Xm.data, M.data, r_init, 1, 100, 1e-5);
T(k)=toc;
Xr{k}=full(ktensor(s,V));
%% TMac Inverse Problems and Imaging 2015
disp('Tmac...')
k=k+1;  alg_name{k}='TMac';
r_init=10;
opts = [];  opts.maxit = 100;   opts.tol = 1e-5; 
opts.Mtr = M;   opts.alpha_adj = 0;
opts.rank_adj = 1*ones(1,3);   opts.rank_min = 10*ones(1,3);    opts.rank_max = [30 30 150];
Mknown=find(M.data==1); data=Xm.data;   data=data(Mknown);
EstCoreNway=[1 1 1];
tic
[X_dec,Y_dec,Out_dec] = TMac(data,Mknown,size(M),EstCoreNway,opts);
Xrec = zeros(size(M));
for i = 1:3
    Xrec = Xrec+Out_dec.alpha(i)*Fold(X_dec{i}*Y_dec{i},size(M),i);
end
T(k)=toc;
Xr{k}=tensor(Xrec);
%% KBR-TC TPAMI 2017
disp('KBR-TC...')
k=k+1;  alg_name{k}='KBR-TC';
Par2.tol     = 1e-4;    Par2.maxIter = 200;    Par2.maxSubiter = 1;%lambda
Par2.rho = 1.05;    Par2.mu = 1e-5; Par2.lambda = 0.1; % 0.01 is even better
Par2.rank=size(X);
X_01=Xm.data;vmin=min(X_01(:));vmax=max(X_01(:));X_01=(X_01-vmin)/(vmax-vmin);
tic;
X_KBR = KBR_TC(X_01, logical(M.data), Par2);X_KBR=X_KBR*(vmax-vmin)+vmin;
Xr{k}=tensor(X_KBR);
T(k) = toc;
%%
disp('Tensor ring...')
k=k+1;
r=[1 1 1]*3; % TR-rank 
maxiter=200; % maxiter 300~500
tol=1e-6; % 1e-6~1e-8
Lambda=1e2; % usually 1~10
ro=1.1; % 1~1.5
K=1e-1; % 1e-1~1e0 
tic
[Xr{k},~,~]=TRLRF(Xm.data,M.data,r,maxiter,K,ro,Lambda,tol);
Xr{k}=tensor(Xr{k});
T(k)=toc;
%%
k=k+1;
tic
Omega=find(M.data==1);
obs.y=Xm(Omega);
obs.idx=Omega;
obs.tsize=size(M);
vW=[1, 1, 1];
vW=vW/sum(vW)/10000;
vRho=1e-8*[1,1,1];
vNu=1.1*[1,1,1];
optsOverlap.MAX_ITER_OUT=100;
optsOverlap.MAX_RHO=1e5;
optsOverlap.MAX_EPS=1e-4;
optsOverlap.verbose=1;
optsOverlap.para.alpha=vW;
optsOverlap.para.vRho=vRho;
optsOverlap.para.vNu=vNu;
memoOverlap=h_construct_memo_v2(optsOverlap);
memoOverlap.truth=X;
memoOverlap=f_tc_OITNN_O(obs,optsOverlap,memoOverlap);
Xr{k}=memoOverlap.T_hat;
Xr{k}=tensor(Xr{k});
T(k)=toc;
%%
disp('M2DMTF...')
k=k+1; 
tic
s_G=[20 20 10];
s_NN{1}=[5 s_G(1) size(Xm,1)];
s_NN{2}=[5 s_G(2) size(Xm,2)];
s_NN{3}=[5 s_G(3) size(Xm,3)];
lambda_w=[1 1 1]*1;
lambda_z=[1 1 1]*1;
lambda_c=[100];
options.maxiter=2000;
%
[Xr{k},G,U,loss,NN] = M2DMTF_tensor(Xm,M,s_G,s_NN,lambda_w,lambda_z,lambda_c,options);
T(k)=toc;
%%
for i=1:length(Xr)
    Mr=M_org-M;
    if ~isempty(Xr{i})
        Xr_data=Xr{i}.data;
        re(q,i)=sum(((Xr_data(:)-X(:)).*Mr(:)).^2)^0.5/sum(((X(:)).*Mr(:)).^2)^0.5*100;
        re_mae(q,i)=sum(abs(Xr{i}.data(:)-X(:)).*Mr(:))/sum(abs(X(:).*Mr(:)))*100;
        rmse(q,i)=(sum(((Xr_data(:)-X(:)).*Mr(:)).^2)/sum(Mr(:)))^0.5;
    end
end
end
