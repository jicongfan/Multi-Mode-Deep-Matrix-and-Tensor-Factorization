function mse=f_error_mse(A,B)
E=double(A-B);
N=numel(E);
err=norm(E(:));
mse=err*err/N;
