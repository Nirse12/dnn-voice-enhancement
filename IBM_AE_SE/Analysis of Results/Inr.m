function [NR,d,Ex] = nr(x,y,z)
% Noise Reduction
% th=1e-4;
x = x(:);
y = y(:);
z = z(:);

M = 256;
dM = 0.25*M;
win = hanning(M);
lenx = length(x);
Nframes = floor((lenx-M+dM)/dM);
% The segmental SNR is clamped to range between 35dB and -10dB
Ex = zeros(Nframes,1);
NR = zeros(Nframes,1);
start = 0;
for k = 1:Nframes
    xwin = win.*x(start+(1:M));
    ywin = win.*y(start+(1:M));
    zwin = win.*z(start+(1:M));
    Ex(k) = sum(xwin.^2);
    NR(k) = 10*log10(sum(zwin.^2)/sum(ywin.^2));
    start = start+dM;
end
th = 5e-4*max(Ex);
NR(find(Ex>th)) = nan;
d = nanmedian(NR(find(Ex<th)));
