function fr=FR(I,r,sr)
[m,n]=size(I);
I_temp=zeros(1,n);
for i=1:n
    temp=I;
    temp(i)=[];
    I_temp(i)=prod(temp);
end

N=prod(I);
fr=r*(I_temp+I-r)./(N*sr);
fr=mean(fr);

return;
end
