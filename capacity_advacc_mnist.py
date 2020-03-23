import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test

import numpy as np
import matplotlib.pyplot as plt
from math import log2


# Hyper-parameters
param = {
    'batch_size': 128,
    'test_batch_size': 100,
    'num_epochs': 15,
    'delay': 10,
    'learning_rate': 1e-3,
    'weight_decay': 5e-4,
}

device = torch.device("cuda")


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


def adv_test(model, loader, blackbox=False, hold_out_size=None, epsilon=0.3):
    """
    Check model accuracy on model based on loader (train or test)
    """
    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)

    if blackbox:
        num_samples -= hold_out_size

    for x, y in loader:
        # x_var = to_var(x, volatile=True)
        # x_adv = adv_train(x, y, model, nn.CrossEntropyLoss(), LinfPGDAttack(epsilon=epsilon))
        x_adv = FGSM_train_rnd(x, y, model, nn.CrossEntropyLoss(), FGSMAttack(), epsilon_max=epsilon)
        x_adv_var = x_adv.to(device)
        scores = model(x_adv_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct)/float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the adv data'
        % (num_correct, num_samples, 100 * acc))

    return acc


test_dataset = datasets.MNIST(root='../data/', train=False, download=True,
    transform=transforms.ToTensor())
loader_test = torch.utils.data.DataLoader(test_dataset,
    batch_size=param['test_batch_size'], shuffle=True)


adv_accs = []

for n in [2, 4, 8, 16, 32, 64]:
    net = NetMul(n)
    net.load_state_dict(torch.load("models/mnist/std_" + str(n) + ".pt"))
    net.to(device)
    net.eval()

    acc = adv_test(net, loader_test, epsilon=0.1)
    adv_accs.append(acc)

np.save('data/adv_accs.npy', np.array(adv_accs))
accs = np.load("data/accs.npy")

plt.plot(range(1, 7), adv_accs)
plt.plot(range(1, 7), accs)
plt.show()
