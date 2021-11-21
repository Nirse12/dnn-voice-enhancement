function ana_win = get_analysis_window(win,nfft,dM)
    
    win2=win.^2;
    W0=win2(1:dM);
    for k=dM:dM:nfft-1
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