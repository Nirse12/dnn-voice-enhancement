function y=stft_analysis(x,K,dM,wintype,fftsize,cut_negative_freqs)
    
    % Usage
    %    Y=stft_analysis(x,K,dM,dN,wintype,cut_negative_freqs)
    % Example
    %    Y=stft_analysis(x,512,256,1,'hamming',1)
    % Inputs
    %    x - signal
    %    K - window length
    %    dM - sampling step in Time
    %    fftsize - FFT resolution (fftsize-K is the length of zero padding)
    %    wintype - window type
    %    cut - whether to cut STFT of real signals
    % Defaults
    %    cut='cut'
    %    wintype='Hanning'
    %    dN=1
    %    dM=0.5*K
    %    K=minimum of 256 and the length of the signal
    % Notes
    %    B=STFT(A,NFFT,dM,dN,WINTYPE) calculates the STFT
    %    for the signal in vector A.  The signal is split into
    %    overlapping segments, each of which are windowed and then
    %    Fourier transformed to produce an estimate of the
    %    short-term frequency content of the signal.
    
    x=x(:); % column vector
    
    % default values
    if nargin<5, wintype='Hamming'; end
    if nargin<6, fftsize=K; end
    if nargin<7, cut_negative_freqs=1; end
    
    % synthesis window
    if ischar(wintype) % wintype is a string - the window type name
        if exist(wintype)
            wins=eval([lower(wintype),sprintf('(%g)',K)]);
            if fftsize>K
                wins=[wins;zeros(fftsize-K,1)];
            end
        else
            error(['Undefined window type: ',wintype])
        end
    else % wintype is not a string but a vector
        wins=wintype;
    end
    
    % arrange the signals in the desired frame structure
    y=meas.stft_frambld(x,fftsize,dM);
    
    % Apply the window to the array of offset signal segments.
    % find analysis window for the above synthesis window
    win=wins;
    win=get_analysis_window(win,fftsize,dM);
    y=win(:,ones(1,size(y,2))).*y;
    
    % FFT on the columns
    y=fft(y);
    
    % real signals need only half of the STFT map
    if ~any(any(imag(x))) && cut_negative_freqs
        y=y(1:fftsize/2+1,:);
    end
    
end


function ana_win=get_analysis_window(win,fftsize,dM)
    
    % calculate the analysis window according to the desired synthesis
    % window
    
    win2=win.^2;
    W0=win2(1:dM);
    for k=dM:dM:fftsize-1
        swin2=lnshift(win2,k);
        W0=W0+swin2(1:dM);
    end
    W0=mean(W0)^0.5;
    win=win/W0;
    Cwin=sum(win.^2)^0.5;
    ana_win=win/Cwin;
    
end


function y = lnshift(x,t)
    
    % lnshift: t circular left shift of 1-d signal
    %  Usage
    %    y = lnshift(x,t)
    %  Inputs
    %    x   1-d signal
    %  Outputs
    %    y   1-d signal
    %        y(i) = x(i+t) for i+t < n
    %	 		y(i) = x(i+t-n) else
    
    szX=size(x);
    if szX(1)>1
        n=szX(1);
        y=[x((1+t):n); x(1:t)];
    else
        n=szX(2);
        y=[x((1+t):n) x(1:t)];
    end
    
end