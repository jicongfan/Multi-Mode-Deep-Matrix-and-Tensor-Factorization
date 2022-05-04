function Cf = f_tprod_f(A,B)

[n1,~,n3] = size(A);
m = size(B,2);
A = fft(A,[],3);
B = fft(B,[],3);
Cf = zeros(n1,m,n3);

% first frontal slice
Cf(:,:,1) = A(:,:,1)*B(:,:,1);
% i=2,...,halfn3
mid = floor(n3/2)+1;
for i = 2 : mid
    Cf(:,:,i) = A(:,:,i)*B(:,:,i);
    Cf(:,:,n3+2-i) = conj(Cf(:,:,i));
end



