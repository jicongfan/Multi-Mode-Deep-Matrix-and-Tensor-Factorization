function [Xr,G,U,STATS,NN] = MWDMTF_GD(X,M,s_G,s_NN,lambda_w,lambda_z,lambda_c,options)
disp('MW-DMTF ...')
if ~isfield(options,'maxiter')
    options.maxiter=100;
end
if ~isfield(options,'tol_outer')
    options.tol_outer=1e-5;
end
if ~isfield(options,'tol_inner')
    options.tol_inner=1e-5;
end
if ~isfield(options,'activation_func')
    for i=1:length(s_NN)
        for j=1:length(s_NN{i})-2
            options.act_func{i}{j}='tanh_opt';%tanh_opt
        end
        options.act_func{i}{j+1}='linear';
    end
end
N = ndims(X);
options
% initialization
% G=tensor(ones(s_G));
% G=tensor(zeros(s_G))*0;
% G=tensor(randn(s_G))*0;
for n=1:N
%     m=size(X,n);
%     temp=randn(m,s_G(n));
%     U{n}=temp./repmat((sum(temp.^2).^0.5),m,1)*0;
    Xu=tenmat(X,n);
    [u,~,~]=svd(Xu.data,'econ');
    U{n}=u(:,1:s_G(n))*1;
end
G=ttm(X,{U{1}',U{2}',U{3}'})*0;
% T=full(ttensor(G,U));
% G=G*(norm(X.data(:))/mean(M.data(:)))/norm(T.data(:));
for n = 1:N
    m=size(X,n);
    Z=zeros(m,s_NN{n}(1));
    Z=randn(m,s_NN{n}(1))*0.01;%
    optNN{n}.activation_func=options.act_func{n};
    NN_MF{n}=NN_MF_setup(s_NN{n},optNN{n});
    NN_MF{n}.s=s_NN{n};
    NN_MF{n}.Z=Z;
    NN_MF{n}.m=s_G(n);
    NN_MF{n}.n=m;
    NN_MF{n}.d=s_NN{n}(1);
    NN_MF{n}.Wp=lambda_w(n);
    NN_MF{n}.Zp=lambda_z(n);
    NN_MF{n}.MC=1;
end
G1=tenmat(G,1);
X1=tenmat(X,1);
M1=tenmat(M,1);
NN.M1=M1.data;
NN.X1=X1.data;
s_G=size(G);
NN.sG1=size(G1);
NN.s_G=s_G;
NN.X=X;
NN.M=M;
NN.lambda=lambda_c;
NN.Wp=lambda_w;
NN.Zp=lambda_z;
for n=1:N
    Mn=tenmat(NN.M,n);
    NN_MF{n}.M=Mn.data;
    Xn=tenmat(NN.X,n);
    NN_MF{n}.X=Xn.data;
    P=tenmat(ttm(G,U,[1:n-1 n+1:N]),n)';
    NN_MF{n}.W{end}=P.data;
    NN_MF{n} = NN_MF_ff(NN_MF{n},NN_MF{n}.Z);
    U{n}=NN_MF{n}.a{end-1};
end
NN.F=NN_MF;
%%  
p.verbosity = 3;                    % Increase verbosity to print something [0~3]
p.MaxIter   = options.maxiter;          
p.d_Obj     = 1e-5;
p.method    = 'IRprop+';          
p.display   = 0;
y=[];
for n=1:N
    w=[];
for i=1:length(NN_MF{n}.W)-1
    w=[w;NN_MF{n}.W{i}(:)];
end
z=NN_MF{n}.Z(:);
y=[y;z;w];
end
y=[y;G.data(:)];
[y,f,EXITFLAG,STATS] = rprop(@fg,y,p,NN);
STATS=STATS.error;
% opt.alpha=0.001;
% opt.maxiter=options.maxiter; 
% [J,y]=opt_Adam(@fg,y,NN,opt);
% STATS=[];
% opt.Method='lbfgs';
% opt.MaxIter=options.maxiter;
% opt.optTol=1e-6;
% opt.progTol=1e-6;
% opt.LS_type=0;
% [y,loss]=minFunc(@fg,y,opt,NN);
% for i=1:1000
%     [f,g_all]=fg(y,NN);
%     y=y-g_all*1e-6;
%     %[i f]
% end
%%%%%%%%
[U,G,NN_MF]=get_UG(y,NN);
Xh=ttensor(G,U);
Xr=times(M,X)+times(1-M,Xh);
NN.F=NN_MF;
end
%% 
function [U,G,NN_MF]=get_UG(y,NN)
np=0;
for n=1:length(NN.F)
    NN_MF{n}=NN.F{n};
    Mn=tenmat(NN.M,n);
    NN_MF{n}.M=Mn.data;
    Xn=tenmat(NN.X,n);
    NN_MF{n}.X=Xn.data;
    lz=NN_MF{n}.d*NN_MF{n}.n;
    NN_MF{n}.Z=reshape(y(np+1:np+lz),NN_MF{n}.n,NN_MF{n}.d);
    t=lz;
    for i=1:length(NN_MF{n}.W)-1
        [a,b]=size(NN_MF{n}.W{i});
        NN_MF{n}.W{i}=reshape(y(np+t+1:np+t+a*b),a,b);
        t=t+a*b;
    end
    np=np+t;
end
for n=1:length(NN.F)
    NN_MF{n} = NN_MF_ff(NN_MF{n},NN_MF{n}.Z);
    U{n}=NN_MF{n}.a{end-1};
end
% G
C=reshape(y(np+1:end),NN.sG1);
G=tensor(C,NN.s_G);
end
%%
function [f,g_all]=fg(y,NN)
N=ndims(NN.X);
np=0;
sum_w=0;
sum_z=0;
for n=1:length(NN.F)
    NN_MF{n}=NN.F{n};
     Mn=tenmat(NN.M,n);
     NN_MF{n}.M=Mn.data;
     Xn=tenmat(NN.X,n);
     NN_MF{n}.X=Xn.data;
    lz=NN_MF{n}.d*NN_MF{n}.n;
    NN_MF{n}.Z=reshape(y(np+1:np+lz),NN_MF{n}.n,NN_MF{n}.d);
    t=lz;
    for i=1:length(NN_MF{n}.W)-1
        [a,b]=size(NN_MF{n}.W{i});
        NN_MF{n}.W{i}=reshape(y(np+t+1:np+t+a*b),a,b);
        t=t+a*b;
    end
    np=np+t;
    NN_MF{n} = NN_MF_ff(NN_MF{n},NN_MF{n}.Z);
    U{n}=NN_MF{n}.a{end-1};
end
C=reshape(y(np+1:end),NN.sG1);
G=tensor(C,NN.s_G);
np=0;
mn=(NN_MF{1}.n*NN_MF{2}.n*NN_MF{3}.n)^(1/3);
for n=1:length(NN.F)
    P=tenmat(ttm(G,U,[1:n-1 n+1:N]),n)';
    NN_MF{n}.W{end}=P.data;
    NN_MF{n} = NN_MF_ff(NN_MF{n},NN_MF{n}.Z);
    U{n}=NN_MF{n}.a{end-1};
    NN_MF{n} = NN_MF_bp(NN_MF{n});
    gW=[];
    for i=1:length(NN_MF{n}.W)-1
        wt=NN_MF{n}.W{i}(:,2:end);
        sum_w=sum_w+sum(wt(:).^2)*mn*NN_MF{n}.Wp;
        dW=NN_MF{n}.dW{i}+mn*NN_MF{n}.Wp*[zeros(size(NN_MF{n}.W{i},1),1) NN_MF{n}.W{i}(:,2:end)];
        gW=[gW;dW(:)];
%         gW=[gW;NN_MF.dW{i}(:)+myNN.weight_penalty_L2*myNN.W{i}(:)];
    end
    gZ=NN_MF{n}.dZ+NN_MF{n}.Zp*NN_MF{n}.Z;
    gZ=gZ(:);
    g{n}=[gZ;gW];
    np=np+length(g{n});
    sum_z=sum_z+sum(NN_MF{n}.Z(:).^2)*NN_MF{n}.Zp;
end
g_all=[];
for n=1:length(NN.F)   
    g_all=[g_all;g{n}];
end
% G
N=ndims(NN.X);
BN=U{end};
if N>2
    for n=N-1:-1:2
        BN=kron(BN,U{n});
    end
end
C=reshape(y(np+1:end),NN.sG1);
A=U{1};
%[size(NN.X1) size(A) size(C) size(BN)]
gC=-A'*(NN.M1.*(NN.X1-A*C*BN'))*BN+mn*NN.lambda*C;
gC=gC(:);
g_all=[g_all;gC];
f=0.5*norm(NN.M1.*(NN.X1-A*C*BN'),'fro')^2+0.5*sum_w+0.5*sum_z+0.5*mn*NN.lambda*norm(C,'fro')^2;
end
