function Y=f_trans(X)
[n1,n2,n3]=size(X);
Y = zeros(n2,n1,n3);
Y(:,:,1)=X(:,:,1)';
for i=2:n3
    Y(:,:,i)=X(:,:,n3+2-i)';
end
end