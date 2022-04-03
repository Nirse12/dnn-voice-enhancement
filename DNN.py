# not related to the project
# not related to the project
# not related to the project
# not related to the project
# not related to the project
# not related to the project

import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection as model_selection
from data_processing import *
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from scipy.io import wavfile
import train_dataset

device = torch.device("cuda")

optimizer_type = 'SGD'  # Adam, RMSprop,SGD
epochs = 5
training_file_size = 10  # will change to 10K but need to check how to het whole file size
training_file_path = "training"
train_size = 0.8
test_size = 0.2

training_Data = []
for each in range(1, training_file_size):
    train_audio, sr = get_audio_from_path(training_file_path)
    #print(train_audio)
    training_Data.append(train_audio)

# separate
train_set, test_set = model_selection.train_test_split(training_Data, train_size=train_size, test_size=test_size)
train_set, val_set = model_selection.train_test_split(train_set, train_size=train_size, test_size=test_size)
# add batch


print("train set: ", np.shape(train_set[0]))
# print("test set: ", test_set)
#print(len(val_set))
#print(np.shape(val_set))
# print(len(trainset[1]))


# define the network
layer1_size = 1024
layer2_size = 1024
layer3_size = 1024
length_stft = len(train_set[1])
input_length = int(5 * (1 + len(train_set[1]) / 2))
print(input_length)


class MyNetFC(nn.Module):
    def __init__(self, layer1_size=layer1_size, layer2_size=layer2_size, layer3_size=layer3_size, p=0):
        super(MyNetFC, self).__init__()
        # hidden 1:
        self.fc1 = nn.Linear(input_length, layer1_size)
        self.ReLU = nn.ReLU()
        # hidden 2:
        self.fc2 = nn.Linear(layer1_size, layer2_size)
        self.ReLU = nn.ReLU()
        # hidden 3:
        self.fc3 = nn.Linear(layer2_size, layer3_size)
        self.ReLU = nn.ReLU()

        self.y1 = nn.Linear(layer2_size, length_stft)  # for clean spec
        self.ReLU = nn.ReLU()

        self.y2 = nn.Linear(layer2_size, length_stft)  # for IBM
        self.ReLU = nn.Sigmoid()

        self.y3 = nn.Linear(layer2_size, length_stft)  # for IRM
        self.ReLU = nn.Sigmoid()

        # MLP
        self.MLP = nn.Linear(3 * length_stft, length_stft)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        # first hidden:
        x = self.fc1(x)
        x = self.ReLU(x)

        # second hidden:
        x = self.fc2(x)
        x = self.ReLU(x)

        # third hidden:
        x = self.fc3(x)
        x = self.ReLU(x)

        # for each y
        y1 = self.y1(x)
        y2 = self.y2(x)
        y3 = self.y3(x)
        y4 = self.y4(y1, y2, y3)  # check syntax

        return y1, y2, y3, y4


"""#### Training"""

# create an instance of our model
model = MyNetFC().to(device)
# loss criterion
criterion = nn.CrossEntropyLoss()

# optimizer type
if optimizer_type == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=lr)
elif optimizer_type == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif optimizer_type == 'RMSProp':
    optimizer = optim.Adam(model.parameters(), lr=lr, alpha=0.99, eps=1e-08)
else:
    NotImplementedError("optimizer not implemented")

total_train_loss = np.zeros((epochs, 1))
total_val_loss = np.zeros((epochs, 1))
total_val_accuracy = np.zeros((epochs, 1))
total_train_accuracy = np.zeros((epochs, 1))

# https://towardsdatascience.com/logistic-regression-on-mnist-with-pytorch-b048327f8d19
iter = 0
for epoch in range(epochs):  # loop over the dataset multiple times

    train_loss = 0
    train_acc = 0

    for i, (images, lables) in enumerate(train_set):
        data = torch.flatten(images, start_dim=1)
        images, lables = data.to(device), lables.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # Logistic Regression on the train data
        loss = criterion(outputs, lables)
        loss.backward()
        optimizer.step()

        iter += 1
        if iter % 500 == 0:
            # calculate Accuracy
            correct = 0
            total = 0

        # accuracy & loss
        train_loss += loss.item()
        predicted = torch.max(outputs.data, 1)[1]
        train_acc += (predicted == lables).sum()  # number of true predictions

    model.eval()  # Logistic Regression on the valuation data
    val_loss = 0
    val_acc = 0

    total_train_accuracy[epoch] = 100 * train_acc / len(train_set)
    total_val_accuracy[epoch] = 100 * val_acc / len(val_set)
    total_train_loss[epoch] = train_loss / len(train_set)
    total_val_loss[epoch] = val_loss / len(val_set)
    print("Epoch number: {}/{} \n Loss: {}. Train Accuracy: {}. Val Accuracy: {}.\n".format(epoch, epochs,
                                                                                            total_train_loss[epoch],
                                                                                            total_train_accuracy[epoch],
                                                                                            total_val_accuracy[epoch]))

# test
test_accuracy = 0
total = 0

print('Accuracy of Logistic Regression on test images is: %d %%' % (100 * test_accuracy / total))
# Plots

plt.figure(1)
plt.plot(total_train_loss)
plt.plot(total_val_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['train loss', 'val loss'])

plt.figure(2)
plt.plot(total_train_accuracy)
plt.plot(total_val_accuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train data', 'val data'])

plt.show()
