import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
from data_processing import *
from sklearn.preprocessing import scale
from const import *
import random

eps = 2.2204 * np.exp(-16)


def make_context(noisy, context_mat):
    noisy_pad = np.zeros((257, 2))
    noisy = np.append(noisy, noisy_pad, axis=1)
    noisy = np.insert(noisy, 0, 0, axis=1)
    noisy = np.insert(noisy, 1, 0, axis=1)
    iter_range = (len(np.transpose(noisy)))
    alist = []
    for t in range(2, iter_range - 2):
        if t % 1000 == 0:
            print("{}/{} Context frames loaded.".format(t, iter_range - 4))
        X = noisy[:, [t - 2, t - 1, t, t + 1, t + 2]]
        X = X.flatten()
        alist.extend(X)
    context_mat = np.reshape(alist, (iter_range - 4, 5 * 257))
    return np.transpose(context_mat)


def irm(clean, noise):
    mask = np.copy(clean)
    rows = mask.shape[0]
    cols = mask.shape[1]
    for x in range(0, rows):
        for y in range(0, cols):
            mask[x, y] = np.sqrt(clean[x, y] ** 2 / (clean[x, y] ** 2 + noise[x, y] ** 2))
    plt.figure()
    librosa.display.specshow(mask)
    plt.colorbar()
    plt.savefig('irm.png')
    plt.close()
    return mask


def ibm(clean, noise):
    mask = np.copy(clean)
    mask[(clean) ** 2 > (noise) ** 2] = 1
    mask[(clean) ** 2 < (noise) ** 2] = 0
    plt.figure()
    librosa.display.specshow(mask)
    plt.colorbar()
    plt.savefig('ibm.png')
    plt.close()
    return mask


class EnhanceDataset(Dataset):
    def __init__(self, list_of_shuffled_files, noise_path, file_limit):
        list_of_shuffled_files = random.sample(list_of_shuffled_files, file_limit)
        self.snr_list = [-5, 0, 5]
        print("---------------------------------")
        self.clean_stft = np.empty(shape=(257, 0))
        self.noise_stft = np.empty(shape=(257, 0))
        self.noisy_stft = np.empty(shape=(257, 0))
        self.ibm_context = np.empty(shape=(5 * 257, 0))
        self.irm_context = np.empty(shape=(5 * 257, 0))
        self.spec_context = np.empty(shape=(5 * 257, 0))
        self.list_of_shuffled_files = list_of_shuffled_files
        noise_sample, _ = get_audio_from_path(noise_path)
        for _ in range(1):
            for i, clean_file in enumerate(list_of_shuffled_files):
                print("Noise shape: ", np.shape(noise_sample))
                print()
                print("---------------------------------")
                print("Loadng file number: {}/{}".format(i + 1, len(self.list_of_shuffled_files)))
                self.clean_sample, clean_sr = get_audio_from_path(clean_file)
                cut_noise_sample = get_noise_segment(noise_sample, len(self.clean_sample))

                print("Clean sample is {} seconds".format(self.clean_sample.shape[0] / clean_sr))
                print("Length of clean sample: ", np.shape(self.clean_sample))
                self.noisy_sample, self.clean_sample = multiply_noise_by_SNR(self.clean_sample, cut_noise_sample,
                                                                        random.choice(self.snr_list))
                self.noisy_sample = self.noisy_sample / np.std(self.noisy_sample)

                # Perform STFT
                self.phase_clean, clean_audio_stft = y_stft(self.clean_sample, 512, 0.75)
                self.phase_noise, noise_audio_stft = y_stft(cut_noise_sample, 512, 0.75)
                self.phase_noisy, noisy_audio_stft = y_stft(self.noisy_sample, 512, 0.75)

                # Spectrogram plot
                plt.figure()
                librosa.display.specshow(clean_audio_stft)
                plt.colorbar()
                plt.savefig('clean.png')
                plt.close()
                # noise_audio_stft = (noise_audio_stft - noise_audio_stft.mean(axis=0)) / noise_audio_stft.std(axis=0)

                # Spectrogram plot
                plt.figure()
                librosa.display.specshow(noise_audio_stft)
                plt.colorbar()
                plt.savefig('noise.png')
                plt.close()
                # noisy_audio_stft = np.cbrt(noisy_audio_stft)
                # noisy_audio_stft = (noisy_audio_stft - noisy_audio_stft.mean(axis=0)) / noisy_audio_stft.std(axis=0)
                # print(noisy_audio_stft)

                # Append to a list
                self.clean_stft = np.append(self.clean_stft, clean_audio_stft, axis=1)
                self.noise_stft = np.append(self.noise_stft, noise_audio_stft, axis=1)
                self.noisy_stft = np.append(self.noisy_stft, noisy_audio_stft, axis=1)

                print("Noisy shape: ", self.noisy_stft.shape)
        self.noisy_stft = (self.noisy_stft - self.noisy_stft.mean(axis=0)) / self.noisy_stft.std(axis=0)
        self._ibm = ibm(self.clean_stft, self.noise_stft)
        self._irm = irm(self.clean_stft, self.noise_stft)
        # Noisy processing

        plt.figure()
        librosa.display.specshow(self.noisy_stft)
        plt.colorbar()
        plt.savefig('noisy.png')
        plt.close()

        self.spec_context = make_context(noisy=self.noisy_stft, context_mat=self.spec_context)
        print("Dataset shape: ", self.spec_context.shape)

    def __getitem__(self, i):
        return np.transpose(self.spec_context[:, i]), np.transpose(self.clean_stft[:, i]), np.transpose(
            self._ibm[:, i]), np.transpose(self._irm[:, i])

    def __len__(self):
        return (self.spec_context.shape[1])



