function r_t = f_tubal_rank(T)
sz = size(T);
d3=sz(3);
i_mid=(2+d3)/2;
r_t = 0;
T = fft(T,[],3);
for i=1:i_mid
    r_t = max(r_t,rank(T(:,:,i)));
end
