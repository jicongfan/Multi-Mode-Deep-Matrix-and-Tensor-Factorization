function v2=h_interpolating_vector(v,maxV)
% interpolating with noise
n=length(v);
v2=zeros(2*n,0);
for i=1:n
    v2(2*i-1)=v(i);
    if i< n
        v2(2*i)=(v(i+1)+v(i))/2;
    else
        v2(2*i)=2*v2(2*i-1)-v2(2*i-3);
    end
end
idx=find(v2>maxV);
v2(idx)=maxV;
end
