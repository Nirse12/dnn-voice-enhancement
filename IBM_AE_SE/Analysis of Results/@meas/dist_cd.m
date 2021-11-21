function ds=dist_cd(x,y,fs,param)
    
    %% dist_cd
    
    % Cepstral distance between two signals
    %
    % [d, e] = dist_cd(x, y, fs, param) calculates cepstral distance between
    % two one-dimensional signals specified by X and Y.
    %
    % Written and distributed by the REVERB challenge organizers on 1 July, 2013
    % Inquiries to the challenge organizers (REVERB-challenge@lab.ntt.co.jp)
    %
    % Updated by Boaz Schwartz, April 2014 (renamed "dist_cd")
    
    %% parameters
    
    if nargin<4
        param=struct('frame',0.032,...
        'shift',0.016,...
        'window',@boxcar,...
        'timdif',0.0,...
        'order',24);
    end
    
    %% Align and normalize power
    
    % cut the longer signal
    if length(x)>length(y)
        x=x(1:length(y));
    else
        y=y(1:length(x));
    end
    
    % normalize power
    if isfield(param,'cmn')
        if ~strcmp(param.cmn, 'y')
            x=x/sqrt(sum(x.^2));
            y=y/sqrt(sum(y.^2));
        end
    end
    
    %% Framing
    
    % Calculate the number of frames
    frame=fix(param.frame*fs);
    shift=fix(param.shift*fs);
    num_sample=length(x);
    num_frame=fix((num_sample-frame+shift)/shift);
    
    % analysis window
    win=window(param.window,frame);
    % frames indexes
    idx=repmat((1:frame)',1,num_frame)+...
        repmat((0:num_frame-1)*shift,frame,1);
    
    % segmented signals
    X=bsxfun(@times,x(idx),win);
    Y=bsxfun(@times,y(idx),win);
    
    %% Cepstrum analysis
    
    % cepstrum transform
    ceps_x=realceps(X);
    ceps_y=realceps(Y);
    
    % cut the desired bands
    ceps_x=ceps_x(1:param.order+1,:);
    ceps_y=ceps_y(1:param.order+1,:);
    
    % Perform cepstral mean normalization.
    if isfield(param,'cmn')
        if strcmp(param.cmn,'y')
            ceps_x=bsxfun(@minus,ceps_x,mean(ceps_x,2));
            ceps_y=bsxfun(@minus,ceps_y,mean(ceps_y,2));
        end
    end
    
    
    %% Calculate the cepstral distances
    
    % difference in the cepstrum domain
    err=(ceps_x-ceps_y).^2;
    % average score per frame
    ds=10/log(10)*sqrt(2*sum(err(2:end,:),1)+err(1, :));
    % remove outliers
    ds=min(ds,10);
    ds=max(ds,0);    
    
end


function c = realceps(x, flr)
    
    % REALCEPS
    % Real-valued cepstral coefficients of real-valued sequences
    %
    % C = REALCEPS(X, FLR) calculates real-valued cepstral coefficients
    % real-valued sequences specified by X. Each column of Y contains the
    % cepstral coefficients of the corresponding column of X.
    
    % Check input arguments
    if nargin<2
        flr=-100;
    end
    
    % Calculate the power spectra of the input frames
    pt=2^nextpow2(size(x,1));
    Px=abs(fft(x,pt));
    
    % Perform flooring.
    flr=max(Px(:))*10^(flr/20);
    Px=max(Px,flr);
    
    % Calculate the cepstral coefficients
    c=real(ifft(log(Px)));
    
end