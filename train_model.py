import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import data_processing
import torch
import torch.nn as nn
import torch.optim as optim
import EnhanceDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import utils
import DNN_Model
import librosa as lr
import gc
import scipy.io
from const import *
import pickle

# -----------------------------------------------------------------------------------

synt_win = scipy.io.loadmat(SYNTH_WIN)
synt_win = synt_win['synt_win']

gc.collect()
torch.cuda.empty_cache()

epochs = 200

lr = 0.0001
optimizer_type = 'Adam'  # Adam, RMSprop, SGD
batch_size = 128
data_files = 0
diff_snr = 1

clean_audio_list = data_processing.get_list_of_files(PATH_TO_CLEAN)
noise_audio_list = PATH_TO_NOISE

# Neural Net Parameters
input_size = 1285
# Create a model
model = DNN_Model.NeuralNetwork(input_size=input_size).to(device)
# -----------------------------------------------------------------------------------
# Loss function
loss_fn = nn.MSELoss()
loss_bin = nn.BCEWithLogitsLoss()

# Optimizer type
if optimizer_type == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=lr)
elif optimizer_type == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif optimizer_type == 'RMSProp':
    optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-08)
else:
    NotImplementedError("optimizer not implemented")


# data_torch = EnhanceDataset.EnhanceDataset(list_of_shuffled_files=clean_audio_list, noise_path=noise_audio_list, file_limit=file_limit)


# with open('train1000_notlog.pickle', "wb") as f:
#     pickle.dump(data_torch, f)
#     print("Saved training data to: {}".format(f))

with open('train1000.pickle', 'rb') as f:
    data_torch = pickle.load(f)
    print("Loaded training data from: {}".format(f))


lengths = [int(len(data_torch)*0.8), int(len(data_torch) - int(len(data_torch)*0.8))]

train_set, val_set = torch.utils.data.random_split(data_torch, lengths)

X_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
X_val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

for t in range(epochs):
    # print("Epoch: ", t)
    utils.train_loop(X_train_loader, X_val_loader, model, loss_fn, loss_bin, optimizer, device, t)
    # utils.test_loop(X_train_loader, model, loss_fn, device)


    PATH = './dnn_net_1000_new_log.pt'
    # print("Saving net parameters to path: {}".format(PATH))
    torch.save(model.state_dict(), PATH)









