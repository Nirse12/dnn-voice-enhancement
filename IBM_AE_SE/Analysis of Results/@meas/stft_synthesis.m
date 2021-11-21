function x=stft_synthesis(Y,K,dM,wintype,fftsize)
    
    % Usage
    %    x=stft_synthesis(Y,K,dM,dN,wintype)
    % Example
    %    x=stft_synthesis(Y,512,256,1,'hamming');
    % Inputs
    %    Y - stft of x
    %    K - window length
    %    dM - sampling step in Time
    %    fftsize - FFT resolution (fftsize-K is the length of zero padding)
    %    wintype - window type
    % Ouputs
    %    x - signal
    % Defaults
    %    wintype='Hamming'
    %    dM=0.5*K
    %    K=2*(size(Y,1)-1)
    
    % default values
    if nargin<3, dM=0.5*K; end
    if nargin<4, wintype='Hamming'; end
    if nargin<5, fftsize=K; end
    
    % synthesis window
    if ischar(wintype)
        if exist(wintype)
            win=eval([lower(wintype),sprintf('(%g)',K)]);
            if fftsize>K
                win=[win;zeros(fftsize-K,1)];
            end
        else
            error(['Undefined window type: ',wintype])
        end
    else % wintype is not a string
        win=wintype;
    end
    
    % rebuild the synthesis window (is it necessary?!)
    ana_win=get_analysis_window(win,fftsize,dM);
    win=biorwin(ana_win,dM);
    
    % extend the anti-symmetric range of the spectum
    % In case that the number of frequency bins is K/2+1
    if size(Y,1)~=fftsize
        Y(fftsize/2+2:fftsize,:)=conj(Y(fftsize/2:-1:2,:));
    end
    
    % Computes IDFT for each column of Y
    Y=ifft(Y);  %Y=real(ifft(Y));
    
    % Apply the synthesis window
    ncol=size(Y,2);
    Y=win(:,ones(1,ncol)).*Y;
    
    % Overlapp & add
    x=zeros((ncol-1)*dM+fftsize,1);
    idx=(1:fftsize)';
    start=0;
    for l=1:ncol
        x(start+idx)=x(start+idx)+Y(:,l);
        start=start+dM;
    end
    
end


function win=biorwin(wins,dM)
    
    % Inputs:
    %    wins - synthesis window;
    %    dM - sampling step in Time;
    % Output:
    %    win - analysis window;
    % Usage:
    %    win=biorwin(wins,dM);
    % Defaults:
    %    noverlap=length(wins)/2;
    
    wins=wins(:);
    L=length(wins);
    win=zeros(L,1);
    
    for k=1:dM
        H=wins(k:dM:L);
        win(k:dM:L)=pinv(H);
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
    
    % lnshift -- t circular left shift of 1-d signal
    %  Usage
    %    y = lnshift(x,t)
    %  Inputs
    %    x   1-d signal
    %  Outputs
    %    y   1-d signal:
    %         y(i) = x(i+t)      ;  for i+t < n
    %	 	  y(i) = x(i+t-n)   ;  else
    
    szX=size(x);
    if szX(1)>1
        n=szX(1);
        y=[x((1+t):n); x(1:t)];
    else
        n=szX(2);
        y=[x((1+t):n) x(1:t)];
    end
    
end