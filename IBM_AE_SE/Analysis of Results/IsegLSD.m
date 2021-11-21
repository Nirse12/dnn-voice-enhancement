function [d,Ex] = IsegLSD(x,y)
% Log Spectral Distance
M = 256; %200;%256;
dM = 0.25*M; %M; %0.25*M;
RdB = 50;		% range for the spectrum
X = stft(x,M,dM,1);
Y = stft(y,M,dM,1);
Ex = sum(abs(X).^2);
logXa = 20*log10(abs(X));
logYa = 20*log10(abs(Y));
max_logXa = max(logXa(:));
logXa(find(logXa<max_logXa-RdB)) = max_logXa-RdB;
max_logYa = max(logYa(:));
logYa(find(logYa<max_logYa-RdB)) = max_logYa-RdB;
d = mean((logXa-logYa).^2).^0.5;
d = d';

th = 1e-4*max(Ex);
%d(find(Ex < th)) = nan;

d=nanmedian(d);


function y = stft(x,nfft,dM,dN,wintype)
% stft : Short Time Fourier Transform
% ***************************************************************@
% Inputs: 
%    x,     	signal;
%    nfft,  	window length;
%    dM,			sampling step in Time;
%    dN,			sampling step in Frequency;
%    wintype,	window type;
% Usage:
%    Y=stft(x,nfft,dM,dN,wintype);
% Defaults:
%    wintype='Hamming';
%    dN = 1;
%    dM = 0.5*nfft;
%    nfft = minimum of 256 and the length of the signal;
% Notes:
%    B = STFT(A,NFFT,dM,dN,WINTYPE) calculates the STFT
%    for the signal in vector A.  The signal is split into
%    overlapping segments, each of which are windowed and then
%    Fourier transformed to produce an estimate of the
%    short-term frequency content of the signal.

% Copyright (c) 2000. Dr Israel Cohen. 
% All rights reserved. Created  17/12/00.
% ***************************************************************@
x = x(:); % make a column vector for ease later
nx = length(x);
if nargin == 1
	nfft = min(nx,256);
end
if nargin < 3
   dM = 0.5*nfft;
   dN = 1;
end
if nargin < 5
	wintype = 'hanning'; %'boxcar';%'hamming';
end
win = eval([wintype,sprintf('(%g)',nfft)]);
win = win/sum(win.^2)^0.5;
ncol = fix((nx-nfft)/dM+1);
y = zeros(nfft,ncol);
colindex = 1 + (0:(ncol-1))*dM;
rowindex = (1:nfft)';
[C,R] = meshgrid(colindex-1,rowindex);
y(:) = x(C+R);
%y(:) = x(rowindex(:,ones(1,ncol))+colindex(ones(nfft,1),:)-1);
%y(:) = win(:,ones(1,ncol)).*y;
y(:) = repmat(win,1,ncol).*y;
N = nfft/dN;
for k = 1:dN-1 %time aliasing to reduce frequency resolution
   y(1:N,:) = y(1:N,:) + y((1:N)+k*N,:);
end
y = y(1:N,:);
y = fft(y);
y = y(1:N/2+1,:);
