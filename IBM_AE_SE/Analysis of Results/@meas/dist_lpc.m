function dist=dist_lpc(eval,clean,fs,param)
    
    % my_lpcllr
    % LPC-based distance measures
    %
    % dist=lpc_dist(eval,clean,fs,param)
    %
    % Written and distributed by the REVERB challenge organizers on 1 July, 2013
    % Inquiries to the challenge organizers (REVERB-challenge@lab.ntt.co.jp).
    %
    % Updated by Boaz Schwartz 8 Jan, 2014, to contain also
    % Likelihood-Ratio (LR), and the Itakura-Saito measure.
    
    % parameters
    if nargin<4
        param=struct('frame',0.032,...
        'shift',0.016,...
        'window',@boxcar,...
        'lpcorder',12,...
        'type','LLR');
    end
    
    % framimg parameters
    frame=fix(param.frame*fs);
    shift=fix(param.shift*fs);
    
    % add zeros for integer number of frames
    eval=[eval;zeros(shift*ceil(length(eval)/shift)-length(eval),1)];
    clean=[clean;zeros(shift*ceil(length(clean)/shift)-length(clean),1)];
    
    % number of frames
    num_sample=length(eval);
    num_frame=fix((num_sample-frame+shift)/shift);
    
    % Break up the signals into frames
    win=window(param.window,frame);
    idx=repmat((1:frame)',1,num_frame)+repmat((0:num_frame-1)*shift,frame,1);
    Eval=bsxfun(@times,eval(idx),win);
    Clean=bsxfun(@times,clean(idx),win);
    
    % FFT
    Eval=fft(Eval,2^nextpow2(2*frame-1));
    Clean=fft(Clean,2^nextpow2(2*frame-1));
    
    % auto-correlation
    Reval=ifft(abs(Eval).^2);
    Reval=Reval./frame;
    Reval=real(Reval);
    Rclean=ifft(abs(Clean).^2);
    Rclean=Rclean./frame;
    Rclean=real(Rclean);
    
    % LPC parameters
    [Aeval,~]=levinson(Reval,param.lpcorder);
    Aeval=real(Aeval');
    [Aclean,~]=levinson(Rclean,param.lpcorder);
    Aclean=real(Aclean');
    
    % Calculate LLR for each frame
    dist=zeros(num_frame, 1);
    for n=1:num_frame
        
        R=toeplitz(Rclean(1:param.lpcorder+1,n));
        
        num=Aeval(:,n)'*R*Aeval(:, n);
        den=Aclean(:, n)'*R*Aclean(:, n);
        
        switch param.type
            
            case 'LR' % Likelihood Ratio
                dist(n)=num/den-1;
                
            case 'LLR' % Log Likelihood Ratio
                dist(n)=log(num/den);
                
            case 'IS' % Itakura-Saito
                geval=Aeval(:,n)'*Reval(1:param.lpcorder+1,n);
                gclean=Aclean(:,n)'*Rclean(1:param.lpcorder+1,n);
                dist(n)=gclean*num/(geval*den)+log(geval/gclean)-1;
                
        end
        
    end
    
    % % %     % outlier removal
    % % %     ds=sort(ds);
    % % %     ds=ds(1:ceil(num_frame*0.95));
    % % %     ds=min(ds,2);
    % % %     ds=max(ds,0);
    % % %
    % % %     % mean and median
    % % %     d=mean(ds);
    % % %     e=median(ds);
    