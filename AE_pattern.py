import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd, adv_train_grad
from adversarialbox.utils import to_var, pred_batch, test

from models import LeNet5
import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import log2


class NetMul(nn.Module):
    def __init__(self, n):
        super(NetMul, self).__init__()
        self.conv1 = nn.Conv2d(1, n, 3, 1)
        self.conv2 = nn.Conv2d(n, 2 * n, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(144 * 2 * n, 32 * int(log2(n)))
        self.fc2 = nn.Linear(32 * int(log2(n)), 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# parameters initialization
device = torch.device("cuda")
alpha = 0.8
epsilon = 0.01
pred = 0
pred_adv = 2
dataset_size = 10000
index = 0

# load pre-trained net
n = 32
net = NetMul(n)
net.load_state_dict(torch.load("models/mnist/std_" + str(n) + ".pt"))
net.to(device)
net.eval()

# Data loaders
train_dataset = datasets.MNIST(root='../data/',train=True, download=True,
    transform=transforms.ToTensor())
loader_train = torch.utils.data.DataLoader(train_dataset,
    batch_size=128, shuffle=False)

test_dataset = datasets.MNIST(root='../data/', train=False, download=True,
    transform=transforms.ToTensor())
loader_test = torch.utils.data.DataLoader(test_dataset,
    batch_size=1000, shuffle=True)

# Adversarial training setup
# adversary = FGSMAttack(epsilon=0.1)
adversary = LinfPGDAttack(epsilon=epsilon)

# Train the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

for t, (x, y) in enumerate(loader_train):
    print(y.shape)
    x_var, y_var = to_var(x), to_var(y.long())
    loss = criterion(net(x_var), y_var)
    y_pred = pred_batch(x, net)
    x_adv, grad = adv_train_grad(x, y_pred, net, criterion, adversary)
    _, grad1 = adv_train_grad(x_adv, y_pred, net, criterion, adversary)
    x_adv_var = to_var(x_adv)
    scores = net(x_var)
    scores_adv = net(x_adv_var)

    print(scores.data.cpu()[index])
    print(scores.data.cpu().max(1)[1][index].detach().numpy())
    print(scores_adv.data.cpu()[index])
    print(scores_adv.data.cpu().max(1)[1][index].detach().numpy())

    grad = grad[index][0]
    grad1 = grad1[index][0]
    scale = torch.max(torch.abs(grad))
    scale1 = torch.max(torch.abs(grad1))
    print("scale:   ", scale)
    print("scale1:   ", scale1)
    grad = grad / scale
    grad1 = grad1 / scale
    print(torch.max(grad))
    print(torch.min(grad))

    red = torch.clamp(grad, 0, 1)
    blue = torch.clamp(grad, -1, 0)
    blue = torch.abs(blue)
    red = red.cpu().detach().numpy()
    blue = blue.cpu().detach().numpy()
    g = np.zeros((28, 28), dtype='float32')
    pattern = cv2.merge([red, g, blue])
    plt.imshow(pattern)
    plt.show()

    red = torch.clamp(grad1, 0, 1)
    blue = torch.clamp(grad1, -1, 0)
    blue = torch.abs(blue)
    red = red.cpu().detach().numpy()
    blue = blue.cpu().detach().numpy()
    g = np.zeros((28, 28), dtype='float32')
    pattern = cv2.merge([red, g, blue])
    plt.imshow(pattern)
    plt.show()

    img = x[index][0].cpu().detach().numpy()
    plt.imshow(img, cmap='gray')
    plt.show()
    img_adv = x_adv[index][0].cpu().detach().numpy()
    plt.imshow(img_adv, cmap='gray')
    plt.show()
    break
