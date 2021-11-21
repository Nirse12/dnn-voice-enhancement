function e=dist_lsd(x,dx,param)

% Description:
%   Log-Spectral-Distortion: computes the LSD between input
%   signals. We assume that the signals are power-aligned!
% Usage:
%    e=dist_lsd(x,dx,M,L,wintype)
% Inputs:
%    x - reference signal
%    dx - estimated signal
%    param.fftsize - FFT resolution (default - 512)
%    param.M - frame size (default 512)
%    param.L - frame step (default 256)
%    param.wintype - analysis window type (default 'hamming')
% Outputs:
%    e - LSD distance matrix

if nargin<3
    % default parameters
    M=512;
    L=256;
    wintype='hamming';
    fftsize=512;
else
    fftsize=param.fftsize;
    M=param.M;
    L=param.L;
    wintype=param.wintype;
end


% STFT
X=meas.stft_analysis(x,M,L,wintype,fftsize);
dX=meas.stft_analysis(dx,M,L,wintype,fftsize);

% calculate the log-spectrum
Lx=meas.absdb(X);
Ldx=meas.absdb(dX);

% eliminate to dynamic range of 60dB
minval=max([Lx(:);Ldx(:)])-60;
Lx=max(Lx,minval);
Ldx=max(Ldx,minval);

% calculate the LSD
e=Lx-Ldx;
e=abs(e).^2;

end