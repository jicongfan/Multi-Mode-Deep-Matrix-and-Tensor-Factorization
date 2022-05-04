function normSp=f_tensor_spectral_norm(X3d)
[~,~,nTubes]=size(X3d);
Xf3d=fft(X3d,[],3);
vSpNorm=zeros(1,nTubes);
for iTube=1:nTubes
    % May be accelerated!
    vSpNorm(iTube)=norm(Xf3d(:,:,iTube));
end
normSp=max(vSpNorm);


