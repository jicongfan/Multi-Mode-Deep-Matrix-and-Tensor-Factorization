clc
clear all
mr=[0.98];% missing rate
for qq=1:length(mr)
for pp=1:1
%rng(99)
r=2;
z1=unifrnd(-1,1,r,50);
z2=unifrnd(-1,1,r,50);
z3=unifrnd(-1,1,r,50);
v=1;
A=sigm(unifrnd(-v,v,20,10)*sin(unifrnd(-v,v,10,r)*z1));
B=exp(unifrnd(-v,v,20,10)*sigm(unifrnd(-v,v,10,r)*z2));
C=cos(unifrnd(-v,v,20,10)*exp(unifrnd(-v,v,10,r)*z3));
sA=svd(A);
sB=svd(B);
sC=svd(C);
d=size(A,1);
G=tensor(randn(d,d,d));
U{1}=A';U{2}=B';U{3}=C';
X=full(ttensor(G,U));
M=zeros(size(X));
miss_ratio=mr(qq);
M(randperm(prod(size(M)),round(prod(size(M))*(1-miss_ratio))))=1;
M=tensor(M,size(X));
Xm=times(X,M);
k=0;
%% HaLRTC or FaLRTC
disp('LRTC...')
k=k+1;
alpha=[1 1 1];
rho = 5*1e-3; maxIter=500;  epsilon=1e-5;
tic
% [X_H, errList_H] = HaLRTC(Xm.data,logical(M.data),alpha,rho,maxIter,epsilon,Xm.data);
[X_H, errList] = FaLRTC(Xm.data,logical(M.data), alpha, 1, 1, 1, maxIter, epsilon);
T(k)=toc;
Xr{k}=tensor(X_H);
%% TenALS
disp('TenALS...')
k=k+1;
tic
r_init=5;
[V{1} V{2} V{3} s dist] = TenALS(Xm.data, M.data, r_init, 1, 100, 1e-5);
T(k)=toc;
Xr{k}=full(ktensor(s,V));
%% TMac Inverse Problems and Imaging 2015
disp('Tmac...')
k=k+1;
tic
r_init=10;
opts = [];  opts.maxit = 500;   opts.tol = -1e-5; 
opts.Mtr = M;   opts.alpha_adj = 0;
opts.rank_adj = 1*ones(1,3);   opts.rank_min = 1*ones(1,3);    opts.rank_max = r_init*ones(1,3);
Mknown=find(M.data==1); data=Xm.data;   data=data(Mknown);
EstCoreNway=ones(1,3)*1;
tic
[X_dec,Y_dec,Out_dec] = TMac(data,Mknown,size(M),EstCoreNway,opts);
Xrec = zeros(size(M));
for i = 1:3
    Xrec = Xrec+Out_dec.alpha(i)*Fold(X_dec{i}*Y_dec{i},size(M),i);
end
Xr{k}=tensor(Xrec);
T(k)=toc;
%% KBR-TC TPAMI 2017
disp('KBR-TC...')
k=k+1;
Par2.tol     = 1e-4;    Par2.maxIter = 300;    Par2.maxSubiter = 1;%lambda
Par2.rho = 1.05;    Par2.mu = 1e-5; Par2.lambda = 0.1; % 0.01 is even better
X_01=Xm.data;vmin=min(X_01(:));vmax=max(X_01(:));X_01=(X_01-vmin)/(vmax-vmin);
tic;
X_KBR = KBR_TC(X_01, logical(M.data), Par2);X_KBR=X_KBR*(vmax-vmin)+vmin;
Xr{k}=tensor(X_KBR);
T(k) = toc;
%% Tensor ring latent
disp('Tensor ring...')
k=k+1;
tic
r=3*ones(1,3); % TR-rank 
maxiter=500; % maxiter 300~500
tol=1e-6; % 1e-6~1e-8
Lambda=1e-3; % usually 1~10
ro=1.1; % 1~1.5
K=1e-2; % 1e-1~1e0 
[Xr{k},~,~]=TRLRF(Xm.data,M.data,r,maxiter,K,ro,Lambda,tol);
T(k)=toc;
%% OITNN
k=k+1;
tic
Omega=find(M.data==1);
obs.y=Xm(Omega);
obs.idx=Omega;
obs.tsize=size(M);
vW=[1, 1, 1];
vW=vW/sum(vW)/100;
vRho=1e-8*[1,1,1];
vNu=1.1*[1,1,1];
optsOverlap.MAX_ITER_OUT=300;
optsOverlap.MAX_RHO=1e5;
optsOverlap.MAX_EPS=1e-4;
optsOverlap.verbose=1;
optsOverlap.para.alpha=vW;
optsOverlap.para.vRho=vRho;
optsOverlap.para.vNu=vNu;
memoOverlap=h_construct_memo_v2(optsOverlap);
memoOverlap.truth=X.data;
memoOverlap=f_tc_OITNN_O(obs,optsOverlap,memoOverlap);
Xr{k}=memoOverlap.T_hat;
T(k)=toc;
%% M2DMTF
disp('M2DMTF...')
k=k+1; 
s_G=[20 20 20]; r=3;
s_NN{1}=[r 10 s_G(1) size(Xm,1)];
s_NN{2}=[r 10 s_G(2) size(Xm,2)];
s_NN{3}=[r 10 s_G(3) size(Xm,3)];
lambda_w=[1 1 1]*1;
lambda_z=[1 1 1]*1;
lambda_c=[1];
options.maxiter=1000;
% 
tic
[Xr{k},G,U,loss,NN] = M2DMTF_tensor(Xm,M,s_G,s_NN,lambda_w,lambda_z,lambda_c,options);
T(k)=toc;
%%
for i=1:length(Xr)
    if ~isempty(Xr{i})
    re(pp,i)=norm(times(minus(Xr{i},X),minus(1,M)))/norm(times(X,minus(1,M)))*100;
    end
end
end
me(qq,:)=mean(re);
sd(qq,:)=std(re);
end
