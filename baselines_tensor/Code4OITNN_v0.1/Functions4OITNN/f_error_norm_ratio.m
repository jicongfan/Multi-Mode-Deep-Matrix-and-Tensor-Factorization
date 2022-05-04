function ratio=f_error_norm_ratio(est,truth)
est=double(est(:));
truth=double(truth(:));
ratio=norm(est-truth)/norm( truth);
