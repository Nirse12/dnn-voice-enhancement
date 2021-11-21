function [SNR_weighted2,SNR] = segWSNR(S,N,Filters,I,clipping);
%function [SNR_weighted2,SNR] = SNR_weighted2(S,N,fs);
%
%calculates a weighted SNR according to the band importance function
%
%INPUTS:
%	S				=	filtered clean speech signal
%	N				=  filtered noise signal
%	fs				=	sampling frequency
%						if fs <= 9 kHz, the 14 band 1/3 octave band 
%					             (ANSI S1.1-1986 standard) values	 are used
%						else the band importance values in the SII standard 
%						(ANSI S3.5-1997) are used
%	clipping		=	if '1' the SNRs are limited to the interval [0,30] dB
%						if '0' the SNRs are not limited to [0,30] dB

n = length(Filters);
SNR = zeros(1,n);

%calculation of the SNR values in the different bands
for i = 1:n

   Ep(i) = rmsdb(filter(Filters(i).b,Filters(i).a,S));
   
   %noise spectrum level -> should include the noncorrelated as well as the 
   %correlated noise (i.e. reverberation). Here, only the uncorrelated noise is
   %taken into account (-> only valid in case of small reverberation)!! 
   
   Np(i) = rmsdb(filter(Filters(i).b,Filters(i).a,N));
   SNR(i) = Ep(i)-Np(i);
   if clipping 
      SNR(i) = min(max(0,SNR(i)),30);
   end
end

SNR_weighted2 = I*SNR';

function [y] = rmsdb( x );
% Root Mean Square
%	if x is a vector, y = rms( x ) returns the root mean square
%	value of the elements of x.
%
%	if x is a matrix, y = rms( x ) returns the root mean square
%	value of each column of x.

[r,c] = size( x );
if c == 1,
   y = 10 * log10( x'*x / length(x) );
elseif r == 1,
   y = 10 * log10( x*x' / length(x) );
else
   y = 10 * log10( sum( x .* x ) / length(x) );
end


