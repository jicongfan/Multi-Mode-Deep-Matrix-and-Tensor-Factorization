function NN_MF=NN_MF_setup(s,options)
if length(options.activation_func)~=(length(s)-1)
    error('The number of layers does not match the number of activation functions!')
end
NN_MF.activation_func=options.activation_func;
NN_MF.layer=length(s)-1;
for i=1:NN_MF.layer
    if i<NN_MF.layer
        NN_MF.W{i}=(rand(s(i+1),s(i)+1)-0.5)*2*4*sqrt(6/(s(i+1)+s(i)));
    else
        NN_MF.W{i}=(rand(s(i+1),s(i))-0.5)*2*4*sqrt(6/(s(i+1)+s(i)));
    end
    %size(NN_MF.W{i})
end
end