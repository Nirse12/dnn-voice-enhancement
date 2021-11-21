    



    function []=plotfreq(sig,fs,K,L,wintype,RdB)
    
    sig=sig/max(abs(sig));
    Sig=20*log10(abs(stft_analysis(sig,K,L,1,wintype,1))); % signal spectrum
    
    if nargin<6
        RdB=60; % range for the spectrum plot
    end

    maxval=max(Sig(:));
    Sig=Sig-maxval;
    Sig(Sig<maxval-RdB)=maxval-RdB; % arbitrarily keep 50dB range    
    xaxis=1/fs*(1:size(Sig,2))';
    yaxis=fs/2/K/1000*(0:K)';
    imagesc(xaxis,yaxis,Sig);
    axis('xy');
%     colormap('hot');