clc
clear all
rng(1)
vp=[0.5];
for pp=1:length(vp)
%%
for jj=1:1
load('movielen100k.mat');
X0=X;
M_org=double(X0~=0);
RRR=sum(M_org(:))/prod(size(M_org))
X=X0;
[nr,nc]=size(X);
missrate=vp(pp);

% miss data
M_t=ones(nr,nc);
% random mask
idx=find(M_org(:)==1);
lidx=length(idx);
temp=randperm(lidx,round(lidx*missrate));
M_t(idx(temp))=0;
%
M=M_t.*M_org;
Xo=X.*M;
%Xo(M==0)=3;
k=0;
%% low-rank matrix factorization
k=k+1;
opt_mf.maxiter=1000;
opt_mf.solver='IRprop+';
%[Xr{k},A,B]=MC_MF_GD(Xo,M,10,1,opt_mf);%%% 
[Xr{k},A,Z]=MC_ALS(Xo,M,5,1,2000);
%% FGSR 
k=k+1;
optionsf.d=50;
optionsf.regul_B='L21';optionsf.maxiter=2000;
optionsf.lambda=0.015;% 0.007
[Xr{k}]=MC_FGSR_PALM(Xo,M,optionsf);
%% DMF
k=k+1;
s=[10 50 100 size(Xo,1)];
options.Wp=0.1;
options.Zp=20;
options.maxiter=1000;
options.activation_func={'tanh_opt','tanh_opt','linear'};
[X_DMF,NN_MF]=MC_DMF(Xo',M',s,options);
Xr{k}=X_DMF';
%%
k=k+1;
s_G=[10 10]*10;
s_NN{1}=[10 50 s_G(1) size(X,1)];
s_NN{2}=[10 50 s_G(2) size(X,2)];
lambda=[1 1 1]*1;
options_M2DMF.maxiter=2000;
options_M2DMF.optimizer='iprop+';
%
[Xr{k},G,U,loss,NN] = M2DMTF_matrix(Xo,M,s_G,s_NN,lambda(1),lambda(2),lambda(3),options_M2DMF);
%%
for i=1:length(Xr)
MM=(~M).*M_org;
Xc=Xr{i};
E=(Xc-X0).*MM;
RMSE(jj,i,pp)=sqrt(sum((E(:)).^2)/sum(MM(:)));
RE(jj,i,pp)=sqrt(sum((E(:)).^2))/sqrt(sum((X(:).*MM(:)).^2));
end
end
%
end




