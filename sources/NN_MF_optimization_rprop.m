function NN_MF=NN_MF_optimization_rprop(NN_MF)
p.verbosity = 0;                    % Increase verbosity to print something [0~3]
p.MaxIter   = NN_MF.maxiter;          
p.d_Obj     = 1e-5;
p.method    = 'IRprop+';          
p.display   = 0;

w=[];
for i=1:length(NN_MF.W)-1
    w=[w;NN_MF.W{i}(:)];
end
z=NN_MF.Z(:);
y=[z;w];
% [y,f,EXITFLAG,STATS] = rprop(@fg_DMFMC,y,p,NN_MF);
lz=NN_MF.d*NN_MF.n;
%
[f,g]=fg_DMFMC(y,NN_MF);
y=y-0.001*g;
Z=reshape(y(1:lz),NN_MF.n,NN_MF.d);
NN_MF.Z=Z;
t=lz+1;
for i=1:length(NN_MF.W)-1
    [a,b]=size(NN_MF.W{i});
    NN_MF.W{i}=reshape(y(t:t+a*b-1),a,b);
    t=t+a*b;
end
NN_MF=NN_MF_ff(NN_MF,Z);
%
end