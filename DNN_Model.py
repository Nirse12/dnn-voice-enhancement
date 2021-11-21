from torch import nn
import torch.nn.functional as F
import torch



class NeuralNetwork(nn.Module):
    def __init__(self, input_size=5*257, hidden_size=1024, stft_size=257):
        super(NeuralNetwork, self).__init__()
        # hidden 1:
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        # hidden 2:
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        # hidden 3:
        self.fc3 = nn.Linear(hidden_size, stft_size)
        self.bn3 = nn.BatchNorm1d(stft_size)

        self.tanh = nn.Tanh()

        self.dropout = nn.Dropout(0.2)

        self.sgm = nn.Sigmoid()

        self.mlp = nn.Sequential(
            nn.Linear(stft_size * 3, 1600),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.BatchNorm1d(stft_size),
            nn.Linear(1600, 1600),
            # nn.ReLU(),
            nn.Linear(1600, stft_size),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # x = self.bn1(x)
        # x = F.relu(self.fc2(x))
        # x = self.dropout(x)

        # x = self.bn2(x)
        x = (self.fc3(x))
        # x = self.dropout(x)

        spec = (x)

        ibm = self.sgm(x)
        irm = self.sgm(x)

        ibm_input = torch.mul(spec, ibm) + torch.mul((1 - ibm), (spec - 1.4))
        irm_input = torch.mul(spec, irm) + torch.mul((1 - irm), (spec - 1.4))

        # ibm_input = torch.mul(spec, ibm)
        # irm_input = torch.mul(spec, irm)

        # ibm_input = spec + ibm
        # irm_input = spec + irm

        total = torch.cat((spec, ibm_input, irm_input), 1)

        total = (self.mlp((total)))

        return total, spec, ibm, irm

        # python file pickle save variable


