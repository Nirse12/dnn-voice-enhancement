function [dis]=segIS(x,a,alpha)
%
% usage		[dis]=dist[x,a,alpha]
% input		a = true lpc parameters
%		alpha = true gain
%		x = corrupted samples
% output	dis = Itakura-Saito Distortion measure
%

[m,n] = size(a);
if m > n , a = a'; end;

[m,n] = size(x);
if m < n , x = x'; end;

p = length(a)-1;
N = length(x);

for k = 0:p

  y = x;
  y =[ y(k+1:length(x));zeros(k,1)];
  r(k+1) = x'*y;

end; % for

R = toeplitz (r)/N;
sig = det (R) / det (R(1:length(R)-1,1:length(R)-1));
dis = (a*R*a'/alpha^2) - log(sig/alpha^2) - 1.;
