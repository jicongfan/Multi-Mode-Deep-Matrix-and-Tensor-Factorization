function X = LRTV_TC(T,Omega)

%==========================================================================
% Solving
%   || X ||_* + lambda * \sum_j || X_j ||_TV 
%       X_Omega = T_Omega 
% D ---> input noisy data
% X ---> denoising result
% 
% by Qi Xie
%==========================================================================
X         = T.*Omega;
X(~Omega) = mean(T(Omega));
Y = X;
r = 3;
sizeT = size(T);
lambda = 10*sqrt(r/(100*prod(sizeT)));
rho = 1.05;
mu = 1e-2;

LamT  = zeros(sizeT);
LamY  = zeros(sizeT);
LamM  = zeros(sizeT);
LamZ  = zeros([sizeT,2]);
DX    = zeros([sizeT,2]);
fft1D = zeros([sizeT,2]);
for i = 1:2
    fft1D(:,:,:,i)    = permute((psf2otf([+1, -1], sizeT([i:2,1:i-1,3]))),[2-i+2:2,1:2-i+1,3]);
end
allfftn =  sum(fft1D.*conj(fft1D),4);

for i = 1:100
    
    % Update M
    tempX    =   Unfold(Y+LamM/mu,sizeT,3)';
    [U,S,V]  =   svd(tempX,'econ');
    M        =   Fold(V*diag(max(diag(S)-1/mu,0))*U',sizeT,3);
    
    % Update Z
    for j  = 1:2
        DX(:,:,:,j)  = real(ifft2(fft1D(:,:,:,j).*fft2(X)));
    end
    Z = max(min(0, DX+LamZ/mu+lambda/mu), DX+LamZ/mu-lambda/mu);
    
    % Update X
    fftnDZ = zeros(sizeT);
    for j = 1:2
        fftnDZ = fftnDZ + conj(fft1D(:,:,:,j)).*fft2(mu*Z(:,:,:,j)-LamZ(:,:,:,j));
    end
    X = real(ifft2( ( fft2(mu*Y -LamY)  + fftnDZ  ) ./ ( mu+ mu*allfftn) ));

    % Update Y
    Y(Omega) = (T(Omega) +X(Omega)+M(Omega)-(LamT(Omega) + LamM(Omega) -LamY(Omega) )/mu   )/3;
    Y(~Omega) = ( X(~Omega)+M(~Omega)-( LamM(~Omega) -LamY(~Omega) )/mu   )/2;
    
    
    % Update mutiplayers
    LamM = LamM  +  mu*(Y-M);
    LamY = LamY  +  mu*(X-Y);
    LamT(Omega) = LamT(Omega)  +  mu*(Y(Omega)-T(Omega));
    LamZ = LamZ  +  mu*(DX-Z);
    
    mu = mu*rho;
    
%     figure(5)
%     subplot(121); imshow(Y(:,:,1),[]);title(['band 1 of iter ' num2str(i)]);
%     subplot(122); imshow(Y(:,:,31),[]);title(['band 31 of iter ' num2str(i)]);
%     pause(0.1)
end
end


