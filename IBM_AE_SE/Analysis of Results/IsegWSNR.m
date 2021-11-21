function [seg_wsnr,Ex] = IsegWSNR(clean_speech, processed_speech)
% Segmental SNR

clean_speech = clean_speech(:);
processed_speech = processed_speech(:);
M = 512; %256; %200       % Size of analysis window
dM = 0.5*M; % M % 0.25*M;
win = hanning(M);%boxcar(M);%hanning(M);
lenx = length(clean_speech);
Nframes = floor((lenx-M+dM)/dM);
% The segmental SNR is clamped to range between 35dB and -10dB
MIN_SNR = -inf; % -inf; %-10;		   % minimum SNR in dB
MAX_SNR =  35; % inf;   %35;		   % maximum SNR in dB
[Filters,I] = MakeWeightFilters(16e3);

seg_wsnr = zeros(Nframes,1);
Ex = zeros(Nframes,1);

start = 0;

for k = 1:Nframes

    xwin = win.*clean_speech(start+(1:M));
    ywin = win.*processed_speech(start+(1:M));
    seg_wsnr(k) = segWSNR(xwin,ywin-xwin,Filters,I,0);%10*log10(sum(xwin.^2)/sum((xwin-ywin).^2));
    start = start + dM;
    Ex(k) = sum(xwin.^2);

end

seg_wsnr = max(seg_wsnr,MIN_SNR);
seg_wsnr = min(seg_wsnr,MAX_SNR);

th = 1e-4*max(Ex);
%seg_wsnr(find(Ex < th)) = nan;


