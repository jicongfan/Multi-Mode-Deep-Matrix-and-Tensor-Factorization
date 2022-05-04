clc
clear all
mr=[0.95];
rng(100)
for qq=1:length(mr)
for pp=1:1
r=2;
z1=unifrnd(-1,1,r,100);
z2=unifrnd(-1,1,r,100);
v=1;
A=unifrnd(-v,v,30,20)*sigm(unifrnd(-v,v,20,10)*sin(unifrnd(-v,v,10,r)*z1));
B=unifrnd(-v,v,30,20)*exp(unifrnd(-v,v,20,10)*sigm(unifrnd(-v,v,10,r)*z2));
X=A'*B;
miss_rate=mr(qq);
M=zeros(size(X));
M(randperm(prod(size(M)),round(prod(size(M))*(1-miss_rate))))=1;
Xm=X.*M;
k=0;
%% low-rank matrix factorization (MF)
k=k+1;
opt_mf.maxiter=2000;
opt_mf.solver='IRprop+';
[Xr{k}]=MC_MF_GD(Xm,M,5,0.01,opt_mf);%%% v=1: 5 0.01; v=2: 5 0.01;
%% nuclear norm minimization (NNM) 
k=k+1;
[Xr{k},E]=MC_IALM(Xm,M);
%% FGSR
k=k+1;
opt_fgsr.d=rank(X);
opt_fgsr.regul_B='L21';
opt_fgsr.tol=1e-4;
opt_fgsr.lambda=0.01;opt_fgsr.maxiter=2000;
[Xr{k}]=MC_FGSR_PALM(Xm,M,opt_fgsr);
%% DMF
k=k+1;
s=[r 10 20 size(X,2)];% input size, hidden size 1, ..., output size
opt.Wp=0.01;
opt.Zp=0.01;
opt.maxiter=3000;
opt.activation_func={'tanh_opt','tanh_opt','linear'};
[X_DMF,NN_MF]=MC_DMF(Xm',M',s,opt);
Xr{k}=X_DMF';
%%
disp('DMTF...')
k=k+1; 
s_G=[20 20];r=3;
s_NN{1}=[r 10 s_G(1) size(Xm,1)];
s_NN{2}=[r 10 s_G(2) size(Xm,2)];
lambda=[1 1 1]*1;
options.maxiter=2000;
% 
[Xr{k},G,U,loss,NN] = M2DMTF_matrix(Xm,M,s_G,s_NN,lambda(1),lambda(2),lambda(3),options);

%%
for i=1:length(Xr)
    if ~isempty(Xr{i})
    re(pp,i)=norm((1-M).*(Xr{i}-X),'fro')/norm((1-M).*X,'fro')*100;
    end
end
end
end