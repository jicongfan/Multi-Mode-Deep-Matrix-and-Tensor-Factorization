function Z=f_t_prod(X,Y)
d1=size(X,1);d3=size(X,3);d4=size(Y,2);

if size(X,2) ~= size(Y,1)
    error('Cannot t-product for dimensional dismatching!');
end

X = fft(X,[],3); Y = fft(Y,[],3);

Z = zeros(d1,d4,d3);
mid = floor((2+d3)/2);
for k=1:mid
    Z(:,:,k)=squeeze(X(:,:,k))*squeeze(Y(:,:,k));
end
for k=mid+1:d3
    Z(:,:,k)=conj(Z(:,:,2+d3-k));
end
Z=ifft(Z,[],3);