function [val, ssimap] = h_SSIM(X,Y)
X=double(X);
Y=double(Y);
  if ndims(X) == 3 
      if size(X,3) == 3
        x = 0.299*X(:,:,1) + 0.587*X(:,:,2) + 0.114*X(:,:,3);
        y = 0.299*Y(:,:,1) + 0.587*Y(:,:,2) + 0.114*Y(:,:,3);
      else
          x = mean(X,3);
          y = mean(Y,3);
      end
  else
    x = X;
    y = Y;
  end
  [val, ssimap] = f_ssim2d(x,y);
