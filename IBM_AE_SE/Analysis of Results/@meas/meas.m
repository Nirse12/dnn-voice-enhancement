classdef meas
    
    % TODO - write a summary of this class
    
    properties
    end
    
    methods
    end
    
    methods(Static)
        
        []=meas_example();
        [moslqo,rawmos]=dist_pesq_wrap(clean,degraded,fs,mode)
        [d,e]=dist_cd(x,y,fs,param);
        dist=dist_lpc(eval,clean,fs,param);
        e=dist_lsd(x,dx,M,L,wintype,fftsize);
        ds=dist_wsnr(x,y,fs,param);
        y=stft_analysis(x,K,dM,wintype,fftsize,cut_negative_freqs);
        x=stft_synthesis(Y,K,dM,wintype,fftsize);
        [X,x]=stft_frambld(x,M,L);
        [wts,binfrqs] = fft2melmx(nfft, sr, nfilts, width, minfrq, maxfrq, htkmel, constamp);
        p = calcpesq(tgtname, refname, pesqexe)
        % get the dB value of the data - 10 X log_10 of the absolute value
        function y=absdb(x), y=10*log10(abs(x)); end
        
    end
    
end