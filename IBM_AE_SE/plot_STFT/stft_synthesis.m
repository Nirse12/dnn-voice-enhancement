function x=stft_synthesis(Y,nfft,dM,dN,wintype)
    % Usage
    %    x=istft(Y,nfft,dM,dN,wintype)
    % Example
    %    x=stft_synthesis(Y,512,256,1,'hamming');
    % Inputs
    %    Y     		stft of x
    %    nfft  		window length
    %    dM			sampling step in Time
    %    dN			sampling step in Frequency
    %    wintype	window type
    % Ouputs
    %    x     		signal
    % Defaults
    %    wintype='Hanning'
    %    dN=1
    %    dM=0.5*nfft
    %    nfft=2*(size(Y,1)-1)
    
    
    if nargin == 1
        nfft = 2*(size(Y,1)-1);
    end
    if nargin < 3
        dM = 0.5*nfft;
        dN = 1;
    end
    if nargin < 5
        wintype = 'Hamming';
    end
    
    if exist(wintype)
        win=eval([lower(wintype),sprintf('(%g)',nfft)]);
    else
        error(['Undefined window type: ',wintype])
    end
    
    win2=win.^2;
    W0=win2(1:dM);
    for k=dM:dM:nfft-1
        swin2=lnshift(win2,k);
        W0=W0+swin2(1:dM);
    end
    W0=mean(W0)^0.5;
    win=win/W0;
    Cwin=sum(win.^2)^0.5;
    win=win/Cwin;
    
    win = biorwin(win,dM,dN);
    % win = win*norm(wina);
    
    
    N=nfft/dN;
    %extend the anti-symmetric range of the spectum
    %In case that the number of frequency bins is nfft/2+1
    if size(Y,1) ~= nfft
        Y(N/2+2:N,:)=conj(Y(N/2:-1:2,:));
    end
    
    % Computes IDFT for each column of Y
    Y = ifft(Y);  %Y=real(ifft(Y));
    Y=Y((1:N)'*ones(1,dN),:);
    
    % Apply the synthesis window
    ncol=size(Y,2);
    Y = win(:,ones(1,ncol)).*Y;
    
    % Overlapp & add
    x=zeros((ncol-1)*dM+nfft,1);
    idx=(1:nfft)';
    start=0;
    for l=1:ncol
        x(start+idx)=x(start+idx)+Y(:,l);
        start=start+dM;
    end
    
    %Cancelling the artificial delay at the beginning an end of the input signal x[n] (see stft_new)
    %Note that we're not cancelling the delay at the end of the signal (because
    %we don't have nx - the length of x[n] before the zeros padding). Hence,
    %the reconstructed signal will 'suffer' from zeros padding at its end
    %delay1 = (ceil(nfft/dM)-1)*dM;
    delay1 = 0;
    % delay1 = nfft-1;
    x = x(delay1+1:end);
    
end

function y = lnshift(x,t)
    % lnshift -- t circular left shift of 1-d signal
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


function win=biorwin(wins,dM,dN)
    % Inputs:
    %    wins,     synthesis window;
    %    dM,			sampling step in Time;
    %    dN,			sampling step in Frequency;
    % Output:
    %    win,      analysis window;
    % Usage:
    %    win=biorwin(wins,dM,dN);
    % Defaults:
    %    noverlap=length(wins)/2;
    
    wins=wins(:);
    L=length(wins);
    N=L/dN;
    win=zeros(L,1);
    mu=zeros(2*dN-1,1);
    mu(1)=1;
    %mu(1)=1/N;
    
    for k=1:dM
        %     H=zeros(2*dN-1,ceil(L/dM));
        %     for q=0:2*dN-2
        %         h=shiftcir(wins,q*N);
        %         H(q+1,:)=h(k:dM:L)';
        %     end
        H = wins(k:dM:L);
        win(k:dM:L)=pinv(H)*mu;
    end
    %win=win/max(win);
    
end