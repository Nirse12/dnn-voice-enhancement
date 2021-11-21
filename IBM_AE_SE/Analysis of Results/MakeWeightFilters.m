function [Filters,I] = MakeWeightFilters(fs);
%Filters = MakeWeightFilters(fs);
%calculates Filters for weighted SNR according to the band importance function
%
%INPUTS:
%	fs				=	sampling frequency
%						if fs <= 9 kHz, the 14 band 1/3 octave band 
%					             (ANSI S1.1-1986 standard) values	 are used
%						else the band importance values in the SII standard 
%						(ANSI S3.5-1997) are used


%Band importance function
r=2^(1/6);
if fs/2 > 4500;
   I=[83 95 150 289 440 578 653 711 818 844 882 898 868 844 771 527 364 185]*1e-4;
   F=[160 200 250 315 400 500 630 800 1000 1250 1600 2000 2500 3150 4000 5000 6300 8000];
   f2=F*r;
   n=sum(fs/2>f2);
	F=F(1:n);
   I=I(1:n);
	%disp([num2str(n) '/18 banden, max: ' num2str(sum(I)*100) '%']);
else
   I=[128 320 320 447 447 639 639 767 959 1182 1214 1086 1086 757]*1e-4;
   F=[200 250 315 400 500 630 800 1000 1250 1600 2000 2500 3150 4000];
   f2=F*r;
	n=sum(fs/2>f2);
	F=F(1:n);
   I=I(1:n);
   %disp([num2str(n) '/14 banden, max: ' num2str(sum(I)*100) '%']);
end

Filters = struct('a',[],'b',[]);
for i=1:n
   [Filters(i).b,Filters(i).a]	 = oct3dsgn(F(i),fs,3);	
end

function [B,A] = oct3dsgn(Fc,Fs,N); 
% OCT3DSGN  Design of a one-third-octave filter.
%    [B,A] = OCT3DSGN(Fc,Fs,N) designs a digital 1/3-octave filter with 
%    center frequency Fc for sampling frequency Fs. 
%    The filter is designed according to the Order-N specification 
%    of the ANSI S1.1-1986 standard. Default value for N is 3. 
%    Warning: for meaningful design results, center frequency used
%    should preferably be in range Fs/200 < Fc < Fs/5.
%    Usage of the filter: Y = FILTER(B,A,X). 
%
%    Requires the Signal Processing Toolbox. 
%
%    See also OCT3SPEC, OCTDSGN, OCTSPEC.

% Author: Christophe Couvreur, Faculte Polytechnique de Mons (Belgium)
%         couvreur@thor.fpms.ac.be
% Last modification: Aug. 25, 1997, 2:00pm.

% References: 
%    [1] ANSI S1.1-1986 (ASA 65-1986): Specifications for
%        Octave-Band and Fractional-Octave-Band Analog and
%        Digital Filters, 1993.

if (nargin > 3) | (nargin < 2)
  error('Invalide number of arguments.');
end
if (nargin == 2)
  N = 3; 
end
if (Fc > 0.88*(Fs/2))
  error('Design not possible. Check frequencies.');
end
  
% Design Butterworth 2Nth-order one-third-octave filter 
% Note: BUTTER is based on a bilinear transformation, as suggested in [1]. 
pi = 3.14159265358979;
f1 = Fc/(2^(1/6)); 
f2 = Fc*(2^(1/6)); 
Qr = Fc/(f2-f1); 
Qd = (pi/2/N)/(sin(pi/2/N))*Qr;
alpha = (1 + sqrt(1+4*Qd^2))/2/Qd; 
W1 = Fc/(Fs/2)/alpha; 
W2 = Fc/(Fs/2)*alpha;
[B,A] = butter(N,[W1,W2]); 


