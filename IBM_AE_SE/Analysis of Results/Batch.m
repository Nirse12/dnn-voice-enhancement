z=wavread('fjrp1-sx352SPEECHsnr_10_Nsy');
s=wavread('fjrp1-sx352SPEECHsnr_10_Cln');
sestOLS=wavread('fjrp1-sx352SPEECHsnr_10_OLS');
sestMIX=wavread('fjrp1-sx352SPEECHsnr_10_MIX');
sestKEM=wavread('fjrp1-sx352SPEECHsnr_10_KEM');


[NL,SNR,SEGSNRmed,ISmed] = Distortion2004_1(z,s,sestOLS,sestMIX,sestKEM);