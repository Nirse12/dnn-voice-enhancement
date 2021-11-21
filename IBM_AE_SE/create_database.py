from context import *
import glob,os
import soundfile as sf
import numpy as np

#from  context import *
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.special import erf
from python_speech_features import mfcc,delta
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import model_from_json
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.io.wavfile as wav
from keras.models import load_model
#import Enh_mixture_of_deep_experts as soft

eps=2.2204*np.exp(-16)
ubuntu_flag=0

noise=['Mix']#,'Babble', ]#,'Babble''Car','Speech',]
SNR=[-5,0,5,10]

max_sentences=2000

        
Clean=[]
Noisy=np.zeros((1800000,257*10))
Targets_IRM=np.zeros((1800000,257))
Targets_IBM=np.zeros((1800000,257))
Targets_OLD_IBM=np.zeros((1800000,257))
Targets_clean_log_spec=np.zeros((1800000,257))
MFCC_mat=np.zeros((1800000,39*10))
        
TIMIT=0
        

clean_train_path='/data/Database/WSJ1/Train'
noise_path='/data/Database/Nonspeech/Rand_noises.wav'
#v,f=sf.read(noise_path)
v,f=sf.read(noise_path)

speakers_dir=os.listdir(clean_train_path)

test_indx=1
#for snr in SNR:
startt=0
while test_indx<max_sentences:
    
    snr=SNR[np.random.randint(len(SNR))]
    #    for i in range (8):
    i=np.random.randint(1,8+1)
#    a=dir_name+'/DR'+str(i)
    b=os.listdir(clean_train_path)
    #    for j in range(len(b)):
    j=np.random.randint(len(b))
    c=clean_train_path + '/%s' % b[j]
    cc=os.listdir(c)
    
    d=glob.glob(c+'/*.wav')
    
    l=np.random.randint(len(d))
    
    
    s,fs=sf.read(d[l])
    if len(s)<257*30:
        continue
        
    noise_name=noise[np.random.randint(len(noise))]
    
    if noise_name == 'Mix':
        v,f=sf.read(noise_path)
        if len(s)>len(v):
            s=s[0:len(v)-384]
        start_i=np.random.randint(len(v)-len(s))
        end_i=start_i+len(s)
        v=v[start_i:end_i]
        g=np.sqrt(10**(-snr/10)*np.std(s)**2/np.std(v)**2)
        v=g*v
        t=np.random.randn(v.shape[0],)
        v=v+t*0.001
        
        
    if noise_name == 'White':
        v=np.random.randn(s.shape[0],1)
        g=np.sqrt(10**(-snr/10)*np.std(s)**2/np.std(v)**2)
        v=g*v
        v=v.reshape(s.shape)
        
    if noise_name == 'Speech':
        v,f=sf.read('/data/notebooks/mixture_of_deep_experts/Noises/Speech.wav')
        start_i=np.random.randint(len(v)-len(s))
        end_i=start_i+len(s)
        v=v[start_i:end_i]
        g=np.sqrt(10**(-snr/10)*np.std(s)**2/np.std(v)**2)
        v=g*v
    
    if noise_name == 'Babble':
        v,f=sf.read('/data/notebooks/mixture_of_deep_experts/Noises/Babble.wav')
        start_i=np.random.randint(len(v)-len(s))
        end_i=start_i+len(s)
        v=v[start_i:end_i]
        g=np.sqrt(10**(-snr/10)*np.std(s)**2/np.std(v)**2)
        v=g*v
    
    if noise_name == 'Car':
        v,f=sf.read('/data/notebooks/mixture_of_deep_experts/Noises/Car.wav')
        start_i=np.random.randint(len(v)-len(s))
        end_i=start_i+len(s)
        v=v[start_i:end_i]
        g=np.sqrt(10**(-snr/10)*np.std(s)**2/np.std(v)**2)
        v=g*v
    
    if noise_name == 'Factory':
        v,f=sf.read('/data/notebooks/mixture_of_deep_experts/Noises/Factory.wav')
        start_i=np.random.randint(len(v)-len(s))
        end_i=start_i+len(s)
        v=v[start_i:end_i]
        g=np.sqrt(10**(-snr/10)*np.std(s)**2/np.std(v)**2)
        v=g*v
        
    if noise_name == 'Room':
        v,f=sf.read('/data/notebooks/mixture_of_deep_experts/Noises/Room.wav')
        start_i=np.random.randint(len(v)-len(s))
        end_i=start_i+len(s)
        v=v[start_i:end_i]
        g=np.sqrt(10**(-snr/10)*np.std(s)**2/np.std(v)**2)
        v=g*v
        
    z=s+v
    
    fs=16000
    K=512
    overlap=0.75
                
    zz=z
    zz=zz/np.std(zz)
    
    S_stft=np.abs(stft(s,K,overlap))
    V_stft=np.abs(stft(v,K,overlap))
    Z_stft=np.abs(stft(zz,K,overlap))
    
    if len(S_stft)<30:
        continue
    
    temp_IRM=np.sqrt((S_stft**2)/(V_stft**2+S_stft**2))
#    plt.figure();plt.imshow(np.log10(S_stft.T[::-1]),aspect='auto');plt.title('Clean')
#    plt.figure();plt.imshow(np.log10(V_stft.T[::-1]),aspect='auto');plt.title('Noise')
#    plt.figure();plt.imshow(np.log10(Z_stft.T[::-1]),aspect='auto');plt.title('Noisy')
    IRM=temp_IRM
    OLD_IBM=S_stft>V_stft
#    plt.figure();plt.imshow(temp_IRM.T[::-1],aspect='auto')
    log_spec=np.log10(S_stft+eps)
    log_spec=(log_spec-log_spec.min())/(log_spec.max()-log_spec.min())
    IBM1=log_spec>0.6#=np.log10(np.abs(v_stft))
    IBM2=log_spec>0.7
 
    
    IBM=np.zeros_like(IBM2)
    IBM[:,0:75]=IBM2[:,0:75]
    IBM[:,75:256]=IBM1[:,75:256]
#    plt.figure();plt.imshow((OLD_IBM.T[::-1]),aspect='auto');plt.title('IBM')
    
    C=mfcc(zz,fs,winlen=K/fs,winstep=K/fs*(1-overlap),nfft=K)
    d_f=delta(C,2)
    dd_f=delta(d_f,2)
    C=np.concatenate((C,d_f,dd_f),axis=1)
    C=C[:-1]
    
    if len(C)>len(Z_stft):
        C=C[:-1]
    if len(C)<len(Z_stft):
        Z_stft=Z_stft[:-1]
    if len(C)!=len(Z_stft): 
        continue
    
    temp_noisy=np.log10(Z_stft+eps)
    temp_noisy=norm(temp_noisy)
    temp_noisy=context(temp_noisy,5,4)
    
    temp_mfcc=norm(C)
    temp_mfcc=context(temp_mfcc,5,4)
    
    endd=startt+temp_noisy.shape[0]
    
    if len(temp_IRM)!=len(temp_noisy): continue

    if endd>Noisy.shape[0]: break

    Noisy[startt:endd,:]=temp_noisy
    MFCC_mat[startt:endd,:]=temp_mfcc
    Targets_IRM[startt:endd,:]=IRM
    Targets_IBM[startt:endd,:]=IBM
    Targets_OLD_IBM[startt:endd,:]=OLD_IBM
    Targets_clean_log_spec[startt:endd,:]=np.log10(S_stft+eps)
#    Noisy=np.concatenate((Noisy,temp_noisy),0)
#    MFCC_mat=np.concatenate((MFCC_mat,temp_mfcc),0)
    startt=endd+1
    
#    Targets_IRM=np.concatenate((Targets_IRM,IRM),0)
#    Targets_IBM=np.concatenate((Targets_IBM,IBM),0)
#    Targets_OLD_IBM=np.concatenate((Targets_OLD_IBM,OLD_IBM),0)
#    Targets_clean_log_spec=np.concatenate((Targets_clean_log_spec,np.log10(S_stft+eps)),0)
    print('file {} / {}'.format(test_indx,max_sentences))
    test_indx+=1

Noisy=Noisy[0:startt-1]
MFCC_mat=MFCC_mat[0:startt-1]
Targets_IRM=Targets_IRM[0:startt-1]
Targets_IBM=Targets_IBM[0:startt-1]
Targets_OLD_IBM=Targets_OLD_IBM[0:startt-1]
Targets_clean_log_spec=Targets_clean_log_spec[0:startt-1]
#Clean=Targets_clean_log_spec                    
#X_train=np.log10(Noisy+eps)

#X_train=norm(X_train)
#X_train=context(X_train,4,4)

#MFCC_mat=norm(MFCC_mat)
#MFCC_mat=context(MFCC_mat,4,4)

#Y_train=Targets_IRM
#Targets_clean_log_spec=(Targets_clean_log_spec-Targets_clean_log_spec.min())/(Targets_clean_log_spec.max()-Targets_clean_log_spec.min())
#mfcc_f=MFCC_mat



#sio.savemat('/data/notebooks/mixture_of_deep_experts/Database_as_test_conditions.mat', {'Targets_IBM':Targets_IBM,'Targets_IRM':Targets_IRM,'Targets_OLD_IBM':Targets_OLD_IBM,'Targets_clean_log_spec':Targets_clean_log_spec,'Noisy':Noisy,'MFCC_mat':MFCC_mat})
#np.savez('/data/notebooks/mixture_of_deep_experts/Database_as_test_conditions',Targets_IBM=Targets_IBM,Targets_IRM=Targets_IRM,Targets_OLD_IBM=Targets_OLD_IBM,Targets_clean_log_spec=Targets_clean_log_spec,Noisy=Noisy,MFCC_mat=MFCC_mat)
##Mat=sio.loadmat('/data/notebooks/mixture_of_deep_experts/Database.mat')
#Mat=sio.loadmat('/data/notebooks/mixture_of_deep_experts/Database_as_test_conditions.mat')
##Mat=sio.loadmat('/data/notebooks/mixture_of_deep_experts/Database_sngl_noise.mat')
#for i in  Mat:
#   a=i+'=Mat["'+i+'"]'
#   exec(a)            
#   
#X_train=context(X_train,4,4)
