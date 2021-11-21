function seg_snr=IsegSNR(clean_speech, processed_speech)
% Segmental SNR

clean_speech = clean_speech(:);
processed_speech = processed_speech(:);
M = 256; %200;%256;       % Size of analysis window
dM = 0.25*M; %M; %0.25*M;
win = hanning(M);%boxcar(M);%hanning(M);
lenx = length(clean_speech);
Nframes = floor((lenx-M+dM)/dM);
% The segmental SNR is clamped to range between 35dB and -10dB
MIN_SNR = -10;		   % minimum SNR in dB
MAX_SNR =  35;		   % maximum SNR in dB
seg_snr = zeros(Nframes,1);
Ex = zeros(Nframes,1);
start = 0;
for k = 1:Nframes
    xwin = win.*clean_speech(start+(1:M));
    ywin = win.*processed_speech(start+(1:M));
    seg_snr(k) = 10*log10(sum(xwin.^2)/sum((xwin-ywin).^2));
    start = start + dM;
    Ex(k) = sum(xwin.^2);

end
seg_snr = max(seg_snr,MIN_SNR);
seg_snr = min(seg_snr,MAX_SNR);

th = 1e-4*max(Ex);
seg_snr(find(Ex < th)) = nan;



