function []=meas_example()

% move to the directory where the EXE is
curr_path=mfilename('fullpath');
tmp=strfind(curr_path,'\');
meas_path=curr_path(1:tmp(end)-1);
orig_dir=cd(meas_path);

% load clean, noisy and processed signals
[sig_cln,fs]=audioread('male_short_TIMIT1_clean.wav');
[sig_noisy,~]=audioread('male_short_TIMIT1_revrb.wav');
[sig_processed,~]=audioread('male_short_TIMIT1_rkemd.wav');

% move back to the original directory
cd(orig_dir);

% Cepstral Distance (CD)
ds_cd_in=meas.dist_cd(sig_cln,sig_noisy,fs);
ds_cd_out=meas.dist_cd(sig_cln,sig_processed,fs);
disp('Cepstral Distance (CD):');
disp(['      In: ' num2str(mean(ds_cd_in)) ' Out: ' num2str(mean(ds_cd_out))]);

% Log Likelihood Ratio (LLR)
ds_lpc_in=meas.dist_lpc(sig_noisy,sig_cln,fs);
ds_lpc_out=meas.dist_lpc(sig_processed,sig_cln,fs);
disp('Log Likelihood Ratio (LLR):');
disp(['      In: ' num2str(mean(ds_lpc_in)) ' Out: ' num2str(mean(ds_lpc_out))]);

% Log Spectral Distortion (LSD)
ds_lsd_in=meas.dist_lsd(sig_cln,sig_noisy);
ds_lsd_out=meas.dist_lsd(sig_cln,sig_processed);
disp('Log Spectral Distortion (LSD):');
disp(['      In: ' num2str(mean(ds_lsd_in(:))) ' Out: ' num2str(mean(ds_lsd_out(:)))]);

% Weighted SNR (WSNR)
ds_wsnr_in=meas.dist_wsnr(sig_cln,sig_noisy,fs);
ds_wsnr_out=meas.dist_wsnr(sig_cln,sig_processed,fs);
disp('Weighted SNR (WSNR):');
disp(['      In: ' num2str(mean(ds_wsnr_in(:))) ' Out: ' num2str(mean(ds_wsnr_out(:)))]);

% PESQ
ds_pesq_in=meas.dist_pesq_wrap(...
    [meas_path '\male_short_TIMIT1_clean.wav'],...
    [meas_path '\male_short_TIMIT1_revrb.wav'],fs);
ds_pesq_out=meas.dist_pesq_wrap(...
    [meas_path '\male_short_TIMIT1_clean.wav'],...
    [meas_path '\male_short_TIMIT1_rkemd.wav'],fs);
disp('PESQ:');
disp(['      In: ' num2str(mean(ds_pesq_in)) ' Out: ' num2str(mean(ds_pesq_out))]);
