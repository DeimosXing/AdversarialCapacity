import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test

from models import LeNet5
import numpy as np
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


device = device = torch.device("cuda")

# Hyper-parameters
param = {
    'batch_size': 128,
    'test_batch_size': 1000,
    'num_epochs': 15,
    'delay': 10,
    'learning_rate': 1e-3,
    'weight_decay': 5e-4,
}
PATH = 'models/std.pkl'


def adv_test(model, loader, blackbox=False, hold_out_size=None):
    """
    Check model accuracy on model based on loader (train or test)
    """
    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)

    if blackbox:
        num_samples -= hold_out_size

    for x, y in loader:
        x_var = to_var(x, volatile=True)
        # x_adv = adv_train(x, y, model, nn.CrossEntropyLoss(), LinfPGDAttack())
        x_adv = FGSM_train_rnd(x, y, model, nn.CrossEntropyLoss(), FGSMAttack())
        x_adv_var = to_var(x_adv, volatile=True)
        scores = model(x_adv_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct)/float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the adv data'
        % (num_correct, num_samples, 100 * acc))

    return acc

n = 32
net = NetMul(n)
net.load_state_dict(torch.load("models/mnist/std_" + str(n) + ".pt"))
net.to(device)
net.eval()

test_dataset = datasets.MNIST(root='../data/', train=False, download=True,
    transform=transforms.ToTensor())
loader_test = torch.utils.data.DataLoader(test_dataset,
    batch_size=param['test_batch_size'], shuffle=True)

for x, y in loader_test:
    x_adv = FGSM_train_rnd(x, y, net, nn.CrossEntropyLoss(), FGSMAttack())
    # x_adv = adv_train(x, y, net, nn.CrossEntropyLoss(), LinfPGDAttack())
    x_adv_var = x_adv.to(device)
    scores = net(x_adv_var)
    x_var = x.to(device)
    scores1 = net(x_var)
    # print(scores.data.cpu()[1].shape)
    _, preds = scores.data.cpu().max(1)
    _, preds1 = scores1.data.cpu().max(1)
    t = 0
    for i in range(90):
        break
        # if preds[i] != preds1[i] and y[i] == 7:
        # if y[i] == preds1[i] == 7 and preds[i] == 4:
        # if y[i] == 2:
        if preds1[i] == y[i] and preds[i] != preds1[i] and torch.abs(scores1.data.cpu()[i][preds1[i]] - scores1.data.cpu()[i][preds[i]]) < 1:
            t = i
            break
    img = np.array(x_adv[t], dtype='float')
    img1 = np.array(x[t], dtype='float')
    img = img.reshape((28, 28))
    img1 = img1.reshape((28, 28))

    np.save('data/mnist_sample/tmp/2.npy', img1.reshape(-1))

    plt.imshow(img, cmap='gray')
    plt.show()
    plt.imshow(img1, cmap='gray')
    plt.show()

    print('std: ', scores1[t].cpu().detach().numpy())
    print('adv: ', scores[t].cpu().detach().numpy())
    print(preds1[t].cpu().detach().numpy())
    print(preds[t].cpu().detach().numpy())
    print(t)
    break


# adv_test(net, loader_test)
