function mse=h_mse(X,Y)
D=X-Y;
mse=norm(double(D(:)));
mse=mse*mse/numel(double(D));
