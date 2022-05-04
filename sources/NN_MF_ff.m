function NN_MF=NN_MF_ff(NN_MF,Z)
Z=[ones(NN_MF.n,1) Z];% add bias 1
L=length(NN_MF.s);
NN_MF.a{1}=Z;
for i=2:L
    switch NN_MF.activation_func{i-1}
        case 'sigm'
            NN_MF.a{i}=sigm(NN_MF.a{i-1}*NN_MF.W{i-1}');
        case 'tanh_opt'
            NN_MF.a{i}=tanh_opt(NN_MF.a{i-1}*NN_MF.W{i-1}');
        case 'linear'
            NN_MF.a{i}=NN_MF.a{i-1}*NN_MF.W{i-1}';
        case 'relu'
            NN_MF.a{i}=max(NN_MF.a{i-1}*NN_MF.W{i-1}',0);
    end
    if i<L-1
        NN_MF.a{i}=[ones(NN_MF.n,1) NN_MF.a{i}];
    end
end
% pedictive error and value of loss function
NN_MF.e=NN_MF.X-NN_MF.a{L};
if NN_MF.MC==1
    NN_MF.e=NN_MF.e.*NN_MF.M;
end
switch NN_MF.activation_func{end}
    case {'sigm', 'relu', 'linear','tanh_opt'}
        NN_MF.loss=1/2*sum(sum(NN_MF.e.^2));
%     case 'softmax'
%         NN_MF.loss=-sum(sum(Y.* log(NN_MF.a{L})))/NN_MF.n;
end
%
end