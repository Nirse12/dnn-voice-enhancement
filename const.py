import torch

# PATH_TO_CLEAN = "nir_voice"
PATH_TO_CLEAN = "clean_train_dataset"
PATH_TO_MERGED_NOISE = "merged_noise/noisemerged.wav"
PATH_TO_NOISE = "merged_noise/train_noise_data/Rand_noises_train.wav"
PATH_TO_NARROWBAND_NOISE = "merged_noise/yochai_dataset/narrowband_noises_test.wav"
PATH_TO_WIDEBAND_NOISE = "merged_noise/yochai_dataset/Rand_noises_test.wav"
SYNTH_WIN = "synt_win.mat"
PATH = './dnn_net.pth'
file_limit = 1
noise_path = "noise_test"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
