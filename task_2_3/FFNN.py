import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch



class FFNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FFNN, self).__init__()

        # Linear function 1: vocab_size --> 500
        self.fc1 = nn.Linear(input_dim, 700)
        # Non-linearity 1
        self.relu1 = nn.ReLU()

        # Linear function 2: 500 --> 500
        self.fc2 = nn.Linear(700, 500)
        # Non-linearity 2
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(500, 300)
        # Non-linearity 3
        self.relu3 = nn.ReLU()

        # Linear function 3 (readout): 500 --> 3
        self.fc4 = nn.Linear(300, output_dim)

    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)

        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.relu2(out)

        # Linear function 3
        out = self.fc3(out)
        # Non-linearity 3
        out = self.relu3(out)

        # Linear function 3 (readout)
        out = self.fc4(out)

        return F.softmax(out, dim=1)

    @staticmethod
    def get_loss_function():
        return nn.CrossEntropyLoss()

    def get_optimizer(self):
        #return optim.SGD(self.parameters(), lr=1e-4, weight_decay=1e-5)
        return torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
