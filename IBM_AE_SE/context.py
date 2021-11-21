import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid import make_axes_locatable

def context (ar,pre_win,aftr_win):
    #add context frames from the past and/or from the future
    add_befor=np.zeros((pre_win,ar.shape[1]))
    add_after=np.zeros((aftr_win,ar.shape[1]))
    ar= np.vstack((add_befor,ar,add_after))
    d=[np.concatenate((ar[i-pre_win:i+aftr_win+1]) ) for i in np.arange(pre_win,len(ar)-aftr_win)]
    return np.array(d)
    
def norm (ar):
    #mean=0, std=1
    ar-=np.mean(ar,axis=0)
    ar/=np.std(ar,axis=0)
    return np.array(ar)

def log_stft(z,K=512,overlap=0.75,RdB=80):
    K_int=K/2+1
    K_int=int(K_int)
    sub_num=1/(1-overlap)-1
    SEG_NO=np.fix(len(z)/(K*(1-overlap)))-sub_num
    SEG_NO=int(SEG_NO)
    Z=np.zeros((K_int,SEG_NO))
    P=np.zeros((K,SEG_NO))
    for seg in np.arange(1,SEG_NO+1):
        time_cal=np.arange((seg-1)*K*(1-overlap)+1,(seg-1)*K*(1-overlap)+K+1)-1
        time_cal=time_cal.astype('int')
        V=np.fft.fft(z[time_cal]*np.append(np.hanning(K-1),0))
        P[:,seg-1]=np.angle(V).reshape(512,)
        time_freq=np.arange(1,K_int+1)-1
        time_freq=time_freq.astype('int')
        Z[:,seg-1]=np.log(np.abs(V[time_freq])+eps)
    return (Z,P)
    
#def stft(z,K=512,overlap=0.75,RdB=80):
#    K_int=K/2+1
#    K_int=int(K_int)
#    sub_num=1/(1-overlap)-1
#    SEG_NO=np.fix(len(z)/(K*(1-overlap)))-sub_num
#    SEG_NO=int(SEG_NO)
#    Z=np.zeros((K_int,SEG_NO))
#    P=np.zeros((K,SEG_NO))
#    for seg in np.arange(1,SEG_NO+1):
#        time_cal=np.arange((seg-1)*K*(1-overlap)+1,(seg-1)*K*(1-overlap)+K+1)-1
#        time_cal=time_cal.astype('int')
#        V=np.fft.fft(z[time_cal]*np.append(np.hanning(K-1),0))
#        P[:,seg-1]=np.angle(V).reshape(512,)
#        time_freq=np.arange(1,K_int+1)-1
#        time_freq=time_freq.astype('int')
#        Z[:,seg-1]=V[time_freq]
#        
#    Z=20*np.log10(np.abs(Z))
#    maxval=np.max(Z) 
#    Z[Z<maxval-RdB]=maxval-RdB
#    return(Z,P)

def pure_stft(z,K=512,overlap=0.75,RdB=80):
    K_int=K/2+1
    K_int=int(K_int)
    
    sub_num=1/(1-overlap)-1
    SEG_NO=np.fix(len(z)/(K*(1-overlap)))-sub_num
    SEG_NO=int(SEG_NO)
    Z=np.zeros((K_int,SEG_NO),dtype=complex)
    P=np.zeros((K,SEG_NO))
    for seg in np.arange(1,SEG_NO+1):
        time_cal=np.arange((seg-1)*K*(1-overlap)+1,(seg-1)*K*(1-overlap)+K+1)-1
        time_cal=time_cal.astype('int')
        V=np.fft.fft(z[time_cal]*np.append(np.hanning(K-1),0))
        P[:,seg-1]=np.angle(V).reshape(K,)
        time_freq=np.arange(1,K_int+1)-1
        time_freq=time_freq.astype('int')
        Z[:,seg-1]=V[time_freq]
    return Z
 
def plot_freqtime(z=1,fs=16000,K=512,overlap=0.75,stft_sig=None,title=' '): 
    RdB=80
    if stft_sig is None:
        z=z/np.max(np.abs(z))
        (stft_sig,P)=stft(z,K,overlap)
#        Z=20*np.log10(np.abs(Z_stft))
#        maxval=np.max(Z) 
#        Z[Z<maxval-RdB]=maxval-RdB
#        stft_sig=Z
#    plt.subplot(2,1,1)
    total_time=0.032+len(stft_sig.T)*0.008#*1/fs*len(z)
    x_nums=np.linspace(1,np.floor(total_time),np.floor(total_time))*(1/(1-overlap))*fs/K
    my_xaxis=np.linspace(1,np.floor(total_time),np.floor(total_time))
    my_xaxis=my_xaxis.astype('int')
      
    plt.figure(title)
    plt.rc('text',usetex=True)
    plt.rc('font',family='serif')
    plt.rcParams.update({'font.size' : 22})
    ax=plt.gca()
    im=ax.imshow(stft_sig, aspect='auto')
    if len(stft_sig)==257:
        y_nums=np.linspace(0,(K/2),5)
        my_yaxis=['0','2','4','6','8']
        plt.yticks(y_nums,my_yaxis)
        plt.ylabel('Frequency [KHz]')
        ax.invert_yaxis()
#    else:
##      else:
#        y_nums=[1,3,4]#np.linspace(1,len(stft_sig)-1,len(stft_sig))
#        my_yaxisstr='my_yaxis=['
#        for i in range(len(stft_sig)):
#            my_yaxisstr=my_yaxisstr+'"'+str(i+1)+'",'
#        my_yaxisstr=my_yaxisstr[:-1]
#        my_yaxisstr=my_yaxisstr+']'
#        exec(my_yaxisstr)
##        my_yaxis=['0','2','4','6','8']
#        ytick='plt.yticks(y_nums,my_yaxis)'
#        exec(ytick)
##        plt.ylabel('Frequency [KHz]')
#        ax.invert_yaxis()
    
    plt.xticks(x_nums,my_xaxis)
#    plt.title('Enhanced signal')
    
    plt.xlabel('Time [sec]')
    plt.ylabel('Experts index')
#    plt.title(title)
    
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im)
#    plt.subplot(2,1,2)
#    plt.plot(z)
    plt.show()
    return 

def stft(x, nfft=512, overlap=0.75, flag='true'):
    x = x.reshape(x.size,1) # make a column vector for ease later
    nx = x.size
    dM = int(nfft*(1-overlap))
    dN = 1

    window=np.hanning
    win = window(nfft)
    win = win.reshape(win.size,1)
    #find analysis window for the above synthesis window
    #figure out number of columns for offsetting the signal
    #this may truncate the last portion of the signal since we'd
    #rather not append zeros unnecessarily - also makes the fancy
    #indexing that follows more difficult.
    ncol = int(np.fix((nx-nfft)/dM+1))
    y = np.zeros((nfft,ncol))
    #now stuff x into columns of y with the proper offset
    #should be able to do this with fancy indexing!
    colindex = np.arange(0,ncol)*dM
    colindex = colindex.reshape(1,ncol)
    rowindex = (np.arange(0,nfft)).reshape(nfft,1)
    rowindex_rep = (np.tile(rowindex,(1,ncol))).astype(int)
    colindex_rep = (np.tile(colindex,(nfft,1))).astype(int)
    indx = rowindex_rep + colindex_rep
    
    for ii in range(0,nfft):
        idx = (indx[ii,:]).reshape(indx.shape[1])
        y[ii,:] = (x[idx]).reshape(len(idx))

    #Apply the window to the array of offset signal segments.
    win_rep = np.tile(win,(1,ncol))
    y = np.multiply(win_rep,y)
    #now fft y which does the columns
    N = int(nfft/dN)
    for k in range(1,dN):
       y[0:N,:] = y[0:N,:] + y[np.arange((0,N))+k*N,:]
    
    y = y[0:N,:]
    y = np.fft.fft(y,axis=0)
    
    if flag=='true':
        if not np.any(x.imag):
             y = y[0:int(N/2)+1,:]
    Y=y.T
    return Y