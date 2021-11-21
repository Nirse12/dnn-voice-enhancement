function ds=dist_wsnr(x,y,fs,param)

% Frequency-weighted segmental SNR
%
% [d, e] = my_wsnr(x, y, fs, param) calculates frequency-weighted segmental SNR of X
% with reference to Y.
%
% Written and distributed by the REVERB challenge organizers on 1 July, 2013
% Inquiries to the challenge organizers (REVERB-challenge@lab.ntt.co.jp)

% parameters
if nargin<4
    param=struct('frame',0.032,...
        'shift',0.016,...
        'window','boxcar',...
        'numband',23);
end

% Normalization
x=x/sqrt(sum(x.^2));
y=y/sqrt(sum(y.^2));

% STFT parameters
frame=fix(param.frame*fs);
shift=fix(param.shift*fs);
fftpt=2^nextpow2(frame);

% add zeros for integer number of frames
x=[x;zeros(shift*ceil(length(x)/shift)-length(x),1)];
y=[y;zeros(shift*ceil(length(y)/shift)-length(y),1)];

% Spectrum amplitude
X=meas.stft_analysis(x,frame,shift,param.window,fftpt);
Y=meas.stft_analysis(y,frame,shift,param.window,fftpt);
X=abs(X);
Y=abs(Y);
[num_freq,~]=size(X);

% Mel-scale frequency warping
melmat=meas.fft2melmx(fftpt,fs,param.numband,1,0,fs/2,1,1);
melmat=melmat(:,1:num_freq);
X=melmat*X;
Y=melmat*Y;

% Calculate WSNR
W=power(Y,0.2);
E=X-Y;

% remove outliers
Ws=sum(W,1);
ny=size(E,1);
ds=10*W.*log10((Y.^2)./(E.^2))./Ws(ones(ny,1),:);
%     ds=min(ds,35);
%     ds=max(ds,-10);
