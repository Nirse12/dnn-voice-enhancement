function []=plotfreqtime(sig,fs,K,L,wintype,RdB)
    % DESCRIPTION
    %   Plot a signal in time domain and STFT domain
    % Usage:
    %   plotfreqtime(sig,fs,K,L,wintype)
    % Inputs:
    %   sig - signal in time domain
    %   fs - sampling frequency
    %   K - number of samples in STFT frame
    %   L -  sampling step in time (number of samples)
    %   wintype - analysis window type
    %   RdB - dynamic range in dB (default - 60dB)
    
    sig=sig/max(abs(sig));
    Sig=20*log10(abs(stft_analysis(sig,K,L,1,wintype,1))); % signal spectrum
    
    % parameters
    if nargin<6
        RdB=60; % range for the spectrum plot
    end
    fontsize=24;
    pos1=[0.08 0.37 0.8 0.563]; % spectrum plot position
    pos2=[0.08 0.15 0.8 0.2]; % time domain plot position
    pos3=[0.888 0.15 0.05 0.785]; % colorbar position
    
    pos4=[0.08 0.37 0.8 0.563/2];
    pos5=[0.08 0.37*2 0.8 0.563/2];
    pos6=[0.08 0.37/2 0.8 0.563/2];
    % plot signal spectrum
    maxval=max(Sig(:));
    Sig(Sig<maxval-RdB)=maxval-RdB; % arbitrarily keep 50dB range
    h(1)=subplot(3,1,1:2);
    xaxis=1/fs*(1:size(Sig,2))';
    yaxis=fs/2/K/1000*(0:K)';
    imagesc(xaxis,yaxis,Sig);
    ylabel('Frequency [KHz]','fontsize',fontsize);
    set(h(1),'ydir','normal','XTick',[]);
    h(3)=colorbar; % add colorbar
    
    
    % plot in time-domain
    h(2)=subplot(3,2,5);
    tvec=1/fs*(1:length(sig));
    plot(tvec,sig);
    xlabel('Time [Sec]','fontsize',fontsize);
    ylabel('Amplitude','fontsize',fontsize);
    tlen=floor(tvec(end));
    %     xtick=0:tlen/5:tlen;
    xtick=0:.5:tlen;
    ytick=[-1 0 1];
    set(h(2),'YTick',ytick,'XTick',xtick,'xlim',[tvec(1) tvec(end)],'ylim', [-1.1 1.1]);
    %ylim_time=get(h(2),'ylim');
    %set(h(2),'YTick',[-1 0 1],'ylim',ylim_time);
    
    set(h(1),'position',pos1,'fontsize',fontsize);
    set(h(2),'position',pos2,'fontsize',fontsize);
    set(h(3),'position',pos3,'fontsize',fontsize);