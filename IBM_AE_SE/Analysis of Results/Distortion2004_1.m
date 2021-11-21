function [NL,SNR,SEGSNRmed,ISmed] = Distortion2004_1(z,s,sestHMM,sestMIX,sestKEM)

seg_leng = 200;
p = 10;
L = fix(length(s)/seg_leng);

S = reshape(s(1:L*seg_leng),seg_leng,L)';
Z = reshape(z(1:L*seg_leng),seg_leng,L)';
SEST_MIX = reshape(sestMIX(1:L*seg_leng),seg_leng,L)';
SEST_HMM = reshape(sestHMM(1:L*seg_leng),seg_leng,L)';
SEST_KEM = reshape(sestKEM(1:L*seg_leng),seg_leng,L)';

Tnr = L*seg_leng-3000:L*seg_leng;
Tsnr = 1:L*seg_leng-3000;


NL = 10*log10(var([z(Tnr)' sestHMM(Tnr)' sestMIX(Tnr)' sestKEM(Tnr)']));
SNR = 10*log10([var(s(Tsnr))/var(z(Tsnr)-s(Tsnr))...
		var(s(Tsnr))/var(sestHMM(Tsnr)-s(Tsnr))...
		var(s(Tsnr))/var(sestMIX(Tsnr)-s(Tsnr))...
		var(s(Tsnr))/var(sestKEM(Tsnr)-s(Tsnr))...
	       ]);

for m = 1:size(S,1)

  [a,g] = aryule(S(m,:),p);
  ISz(m) = segIS(Z(m,:),a,sqrt(g));
  SNRz(m) = segSNR(Z(m,:),S(m,:)+eps);
  ISest_HMM(m) = segIS(SEST_HMM(m,:),a,sqrt(g));
  SNRest_HMM(m) = segSNR(SEST_HMM(m,:),S(m,:)+eps);
  ISest_MIX(m) = segIS(SEST_MIX(m,:),a,sqrt(g));
  SNRest_MIX(m) = segSNR(SEST_MIX(m,:),S(m,:)+eps);
  ISest_KEM(m) = segIS(SEST_KEM(m,:),a,sqrt(g));
  SNRest_KEM(m) = segSNR(SEST_KEM(m,:),S(m,:)+eps);
  
end;
SEGSNRmed = nanmedian([SNRz' SNRest_HMM' SNRest_MIX' SNRest_KEM']);
ISmed = nanmedian([ISz' ISest_HMM' ISest_MIX' ISest_KEM']);




