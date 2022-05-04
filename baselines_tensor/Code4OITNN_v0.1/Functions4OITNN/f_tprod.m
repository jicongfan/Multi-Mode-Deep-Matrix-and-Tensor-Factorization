function C = f_tprod(A,B)

[n1,~,n3] = size(A);
m = size(B,2);
A = fft(A,[],3);
B = fft(B,[],3);
C = zeros(n1,m,n3);

% first frontal slice
C(:,:,1) = A(:,:,1)*B(:,:,1);
% i=2,...,halfn3
mid = floor(n3/2)+1;
for i = 2 : mid
    C(:,:,i) = A(:,:,i)*B(:,:,i);
    C(:,:,n3+2-i) = conj(C(:,:,i));
end

C = ifft(C,[],3);


