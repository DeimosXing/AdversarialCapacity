import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, size = [2, 4, 100]):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, size[0], 3, 1)
        self.conv2 = nn.Conv2d(size[0], size[1], 3, 1)
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(size[2], size[2]//2)
        self.fc2 = nn.Linear(size[2]//2, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def Cnn_2_4():
    return CNN([2, 4, 100])

def Cnn_4_8():
    return CNN([4, 8, 200])

def Cnn_8_16():
    return CNN([8, 16, 400])
