function NN_MF=NN_MF_bp(NN_MF)
L=length(NN_MF.s);
switch NN_MF.activation_func{end}
    case 'sigm'
        d{L}=-NN_MF.e.*(NN_MF.a{L}.*(1-NN_MF.a{L}));
    case 'tanh_opt'
        d{L}=-NN_MF.e.*(1.7159*2/3 *(1-1/(1.7159)^2*NN_MF.a{L}.^2));
    case {'softmax','linear'}
        d{L}=-NN_MF.e;
    case 'relu'
        d{L}=-NN_MF.e;        
end
for i=L-1:-1:2
    switch NN_MF.activation_func{i-1}
        case 'sigm'
            d_act=NN_MF.a{i}.*(1-NN_MF.a{i});
        case 'linear'
            d_act=1;
        case 'tanh_opt'
            d_act=1.7159*2/3 *(1-1/(1.7159)^2*NN_MF.a{i}.^2);
        case 'relu'
            d_act=max(sign(NN_MF.a{i}),0);            
    end
    if i+1==L||i+1==L-1 % in this case in d{n} there is not the bias term to be removed
        d{i} = (d{i + 1} * NN_MF.W{i}).* d_act; % Bishop (5.56)
    else % in this case in d{i} the bias term has to be removed
        d{i} = (d{i + 1}(:,2:end) * NN_MF.W{i}) .* d_act;
    end
end
%
for i = 1:(L - 2)
    if i+1==L-1
        NN_MF.dW{i} = (d{i + 1}' * NN_MF.a{i});
    else
        NN_MF.dW{i} = (d{i + 1}(:,2:end)' * NN_MF.a{i});      
    end
end
% dZ
if size(d{2},2)>size(NN_MF.W{1},1)
    NN_MF.dZ=d{2}(:,2:end)*NN_MF.W{1}(:,2:end);
else
    NN_MF.dZ=d{2}(:,1:end)*NN_MF.W{1}(:,2:end);
end
%NN_MF.dZ=NN_MF.dZ/NN_MF.n;
end