z = wavread('fjrp1-sx352SPEECHsnr_10_Nsy');
s = wavread('fjrp1-sx352SPEECHsnr_10_Cln');
sestOLS = wavread('fjrp1-sx352SPEECHsnr_10_OLS');
sestMIX = wavread('fjrp1-sx352SPEECHsnr_10_MIX');
sestKEM = wavread('fjrp1-sx352SPEECHsnr_10_KEM');

L = min([length(s) length(z) length(sestOLS)  length(sestMIX) length(sestKEM)]);

s = s(1:L);
z = z(1:L);
sestKEM = sestKEM(1:L);
sestOLS = sestOLS(1:L);
sestMIX = sestMIX(1:L);

% --------------

SNR_seg_ZZZ = nanmedian(IsegSNR(s,z));
SNR_seg_MIX = nanmedian(IsegSNR(s,sestMIX));
SNR_seg_KEM = nanmedian(IsegSNR(s,sestKEM));
SNR_seg_OLS = nanmedian(IsegSNR(s,sestOLS));


WSNR_seg_ZZZ = nanmedian(IsegWSNR(s,z));
WSNR_seg_MIX = nanmedian(IsegWSNR(s,sestMIX));
WSNR_seg_KEM = nanmedian(IsegWSNR(s,sestKEM));
WSNR_seg_OLS = nanmedian(IsegWSNR(s,sestOLS));


LSD_seg_ZZZ = nanmedian(IsegLSD(s,z));
LSD_seg_MIX = nanmedian(IsegLSD(s,sestMIX));
LSD_seg_KEM = nanmedian(IsegLSD(s,sestKEM));
LSD_seg_OLS = nanmedian(IsegLSD(s,sestOLS));


NL_seg_ZZZ = nanmedian(Inr(s,s,z));
NL_seg_MIX = nanmedian(Inr(s,s,sestMIX));
NL_seg_KEM = nanmedian(Inr(s,s,sestKEM));
NL_seg_OLS = nanmedian(Inr(s,s,sestOLS));

T = 1:length(s);
SNR_tot_ZZZ = 10*log10(var(s(T))/var(s(T)'-z(T)'));
SNR_tot_MIX = 10*log10(var(s(T))/var(s(T)'-sestMIX(T)'));
SNR_tot_KEM = 10*log10(var(s(T))/var(s(T)'-sestKEM(T)'));
SNR_tot_OLS = 10*log10(var(s(T))/var(s(T)'-sestOLS(T)'));


