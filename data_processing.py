import numpy as np
import librosa as lr
import random
import librosa.display
import os
import scipy as sp

eps = 2.2204*np.exp(-16)

# get_list_of_files
# ==============================================================================================
def get_list_of_files(path):
    clean_audio_list = get_list_of_wav_files(path)
    random.shuffle(clean_audio_list)
    return clean_audio_list


# get_sample_from_file
# ==============================================================================================
def get_audio_from_path(audio_path):
    audio, sr = lr.load(audio_path, 16000)
    return audio.astype('float32'), sr


# get_noise_segment
# ==============================================================================================
def get_noise_segment(noise_audio, clean_length):  # assune same sr
    random_noise_start = int(random.random() * (len(noise_audio) - clean_length))  # asume noise is larger then clean
    noise_segment = noise_audio[random_noise_start:random_noise_start + clean_length]
    noise_segment = noise_segment / np.std(noise_segment)
    return noise_segment


# multiply_noise_by_SNR
# ==============================================================================================
def multiply_noise_by_SNR(clean_signal, noise_segment, desire_snr):
    # Import scipy
    if len(noise_segment) < len(clean_signal):
        clean_signal = clean_signal[:len(noise_segment)]
    # Signal power in data from wav file
    psig = clean_signal.var()

    # For 10 dB SNR, calculate linear SNR (SNR = 10Log10(Psig/Pnoise)
    g = np.sqrt(psig/(10**(desire_snr/10)))

    # Find required noise power
    snr_segments = g * noise_segment

    # Add noise to signal
    return snr_segments + clean_signal, clean_signal


# my_stft
# ==============================================================================================
def my_stft(audio_input):
    K = 512
    hann_win = np.append(np.hanning(K - 1), 0)
    # if not np.isfinite(audio_input).all():
    #     audio_input = np.nan_to_num(audio_input)
    stft = librosa.stft(audio_input, n_fft=K, window=hann_win, hop_length=128, win_length=K)
    actual_phase = np.angle(stft)
    stft = np.log(np.abs(stft)+eps)
    return stft, actual_phase


# get_list_of_wav_files
# ==============================================================================================
def get_list_of_wav_files(path_to_files):
    list_of_files = []
    for file in os.listdir(path_to_files):
        if file.endswith(".wav"):
            list_of_files.append(os.path.join(path_to_files, file))
    return list_of_files

# y_stft
# ==============================================================================================
def y_stft(z, K, overlap):
    sub_num = 1 / (1 - overlap) - 1
    SEG_NO = np.fix(len(z) / (K * (1 - overlap))) - sub_num
    Z = np.zeros((int(K / 2 + 1), int(SEG_NO)))
    P = np.zeros((int(K / 2 + 1), int(SEG_NO)))
    for seg in np.arange(1, SEG_NO + 1):
        time_cal = np.arange((seg - 1) * K * (1 - overlap) + 1,
                             (seg - 1) * K * (1 - overlap) + K + 1) - 1
        time_cal = time_cal.astype('int')
        V = np.fft.fft(z[time_cal] * np.append(np.hanning(K - 1), 0))
        time_freq = np.arange(1, K / 2 + 1 + 1) - 1
        time_freq = time_freq.astype('int')
        P[:, int(seg - 1)] = np.angle(V[time_freq])
        Z[:, int(seg - 1)] = np.abs(V[time_freq])
    # return P, Z
    return P, np.log(Z + eps)


# istft
# ==============================================================================================
def istft(A, P, synt_win):
    '''
    Returns the ISTFT of a (K/2+1) x TIME signal
    param: A is the log-magnitude of the STFT, dimensions are (K/2+1) x TIME.
    param: P is the phase of the STFT, dimensions are (K/2+1) X TIME.
    param: synt_win is the synthesis window of length K.
    '''
    # s_est=np.zeros((len(z),1))
    K = 512
    overlap = 0.75

    # Switch back from log-magnitude to magnitude
    A = np.exp(A)
    A[0:3] *= 0.001

    # Create the a KxTIME STFT from the (K/2+1)xTIME STFT
    A_inv, P_inv = A[::-1], P[::-1]
    A_inv, P_inv = A_inv[1:-1], P_inv[1:-1]
    P_full = np.concatenate([P, -P_inv], 0)
    A_full = np.concatenate([A, A_inv], 0)
    # A_full, P_full = A_full.numpy(), P_full.numpy()

    # Attach the phase with the magnitude
    S = 50 * A_full * np.exp(1j * P_full)
    S = np.real(np.fft.ifft(S, axis=0))
    S = synt_win * S

    SEG_NO = S.shape[1]
    long = 0.008 * 16000 * SEG_NO + K
    long = int(long)
    s_est = np.zeros((long, 1))
    for seg in np.arange(1, SEG_NO + 1):
        time_cal = np.arange((seg - 1) * K * (1 - overlap) + 1, (seg - 1) * K * (1 - overlap) + K + 1) - 1
        time_cal = time_cal.astype('int64')
        s_est[time_cal, :] = s_est[time_cal, :] + np.expand_dims(S[:, seg - 1], axis=1)

    return s_est.squeeze()

