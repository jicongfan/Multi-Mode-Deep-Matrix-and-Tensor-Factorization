function [Xr,C,A,loss,NN] = M2DMTF_matrix(X,M,s_G,s_NN,lambda_w,lambda_z,lambda_c,options)
disp('MW-DMTF ...')
if ~isfield(options,'maxiter')
    options.maxiter=1000;
end
if ~isfield(options,'optimizer')
    options.optimizer='iprop+';
end
if ~isfield(options,'activation_func')
    for i=1:length(s_NN)
        for j=1:length(s_NN{i})-2
            options.act_func{i}{j}='tanh_opt';
        end
        options.act_func{i}{j+1}='linear';
    end
end
N = ndims(X);
options
%% initialization
G=eye(s_G);
[u,s,v]=svds(X,s_G(1));
A=randn(size(X,1),size(G,1));
B=randn(size(G,2),size(X,2));
for n = 1:2
    m=size(X,n);
    Z=randn(m,s_NN{n}(1));
    optNN{n}.activation_func=options.act_func{n};
    NN_MF{n}=NN_MF_setup(s_NN{n},optNN{n});
    NN_MF{n}.s=s_NN{n};
    NN_MF{n}.Z=Z;
    NN_MF{n}.m=s_G(n);
    NN_MF{n}.n=m;
    NN_MF{n}.d=s_NN{n}(1);
    NN_MF{n}.Wp=lambda_w;
    NN_MF{n}.Zp=lambda_z;
    NN_MF{n}.MC=1;
end
NN.sG=size(G);
NN.X=X;
NN.M=M;
NN.lambda=lambda_c;
NN.Wp=lambda_w;
NN.Zp=lambda_z;
NN_MF{1}.M=M;
NN_MF{1}.X=X;
NN_MF{1}.W{end}=(G*B)';
NN_MF{1} = NN_MF_ff(NN_MF{1},NN_MF{1}.Z);
A=NN_MF{1}.a{end-1};
NN_MF{2}.M=M';
NN_MF{2}.X=X';
NN_MF{2}.W{end}=A*G;
NN_MF{2} = NN_MF_ff(NN_MF{2},NN_MF{2}.Z);
B=NN_MF{2}.a{end-1};B=B';
NN.F=NN_MF;
%%  optimization
y=[];
for n=1:2
    w=[];
for i=1:length(NN_MF{n}.W)-1
    w=[w;NN_MF{n}.W{i}(:)];
end
z=NN_MF{n}.Z(:);
y=[y;z;w];
end
y=[y;G(:)];
STATS=[];
switch options.optimizer
    case 'iprop+'
        p.verbosity = 3; % Increase verbosity to print something [0~3]
        p.MaxIter   = options.maxiter;          
        p.d_Obj     = 1e-5;
        p.method    = 'IRprop+';          
        p.display   = 0;
        [y,f,EXITFLAG,STATS] = rprop(@fg,y,p,NN);
        loss=STATS.error(:);
    case 'adam'
        opt.alpha=0.01;% may need to tune
        opt.maxiter=options.maxiter;
        [loss,y]=opt_Adam(@fg,y,NN,opt);
        loss=loss(:);
    case 'lbfgs'
        opt.Method='lbfgs';
        opt.MaxIter=options.maxiter;
        opt.MaxFunEvals=options.maxiter*2;
        opt.optTol=1e-6;
        opt.progTol=1e-6;
        opt.LS_type=0;
        [y,f,~,STATS]=minFunc(@fg,y,opt,NN);
        loss=STATS.trace.fval(:);
end
[A,B,C]=get_UG(y,NN);
Xh=A*C*B;
Xr=M.*X+(1-M).*Xh;
end
%% 
function [A,B,C]=get_UG(y,NN)
np=0;
    NN_MF{1}=NN.F{1};
    lz=NN_MF{1}.d*NN_MF{1}.n;
    NN_MF{1}.Z=reshape(y(np+1:np+lz),NN_MF{1}.n,NN_MF{1}.d);
    t=lz;
    for i=1:length(NN_MF{1}.W)-1
        [a,b]=size(NN_MF{1}.W{i});
        NN_MF{1}.W{i}=reshape(y(np+t+1:np+t+a*b),a,b);
        t=t+a*b;
    end
    np=np+t;
        NN_MF{2}=NN.F{2};
    lz=NN_MF{2}.d*NN_MF{2}.n;
    NN_MF{2}.Z=reshape(y(np+1:np+lz),NN_MF{2}.n,NN_MF{2}.d);
    t=lz;
    for i=1:length(NN_MF{2}.W)-1
        [a,b]=size(NN_MF{2}.W{i});
        NN_MF{2}.W{i}=reshape(y(np+t+1:np+t+a*b),a,b);
        t=t+a*b;
    end
    np=np+t;
for n=1:2
    NN_MF{n} = NN_MF_ff(NN_MF{n},NN_MF{n}.Z);
end
A=NN_MF{1}.a{end-1};
B=NN_MF{2}.a{end-1};B=B';
C=reshape(y(np+1:end),NN.sG);
end
%%
function [f,g_all]=fg(y,NN)
N=ndims(NN.X);
np=0;
sum_w=0;
sum_z=0;
for n=1:length(NN.F)
    NN_MF{n}=NN.F{n};
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
end
A=NN_MF{1}.a{end-1};
B=NN_MF{2}.a{end-1};B=B';
C=reshape(y(np+1:end),NN.sG);
np=0;
mn=sqrt(NN_MF{1}.n*NN_MF{2}.n);
ss=0;
for n=1:length(NN.F)
    if n==1
        P=(C*B)';
    else
        P=A*C;
    end
    NN_MF{n}.W{end}=P;
    NN_MF{n} = NN_MF_ff(NN_MF{n},NN_MF{n}.Z);
    NN_MF{n} = NN_MF_bp(NN_MF{n});
    gW=[];
    for i=1:length(NN_MF{n}.W)-1
        wt=NN_MF{n}.W{i}(:,2:end);
        sum_w=sum_w+sum(wt(:).^2);
        dW=NN_MF{n}.dW{i}+mn*NN_MF{n}.Wp*[zeros(size(NN_MF{n}.W{i},1),1) NN_MF{n}.W{i}(:,2:end)];
        gW=[gW;dW(:)];
    end

    gZ=NN_MF{n}.dZ+NN_MF{n}.Zp*NN_MF{n}.Z;
    gZ=gZ(:);
    g{n}=[gZ;gW];
    np=np+length(g{n});
    sum_z=sum_z+sum(NN_MF{n}.Z(:).^2);
end
g_all=[];
for n=1:length(NN.F)   
    g_all=[g_all;g{n}];
end
% G
C=reshape(y(np+1:end),NN.sG);
gC=-A'*(NN.M.*(NN.X-A*C*B))*B'+mn*NN.lambda*C;
gC=gC(:);
g_all=[g_all;gC];
f=0.5*norm(NN.M.*(NN.X-(A*C*B)),'fro')^2+0.5*mn*NN.Wp*sum_w+0.5*NN.Zp*sum_z+0.5*mn*NN.lambda*norm(C,'fro')^2;
end
