"""
Adversarial attacks on LeNet5
"""
from time import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.utils import to_var, pred_batch, test, \
    attack_over_test_data

#from models import LeNet5

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# Hyper-parameters
param = {
    'test_batch_size': 100,
    'epsilon': 0.3,
}


# Data loaders
test_dataset = datasets.MNIST(root='../data/', train=False, download=True,
    transform=transforms.ToTensor())
loader_test = torch.utils.data.DataLoader(test_dataset, 
    batch_size=param['test_batch_size'], shuffle=False)


# Setup model to be attacked
# net = LeNet5()
net = Net()
net.load_state_dict(torch.load('/home/lilyxue/Downloads/mnist_cnn.pt'))

if torch.cuda.is_available():
    print('CUDA ensabled.')
    net.cuda()

for p in net.parameters():
    p.requires_grad = False
net.eval()

test(net, loader_test)


# Adversarial attack
adversary = FGSMAttack(net, param['epsilon'])
# adversary = LinfPGDAttack(net, random_start=False)


t0 = time()
attack_over_test_data(net, adversary, param, loader_test)
print('{}s eclipsed.'.format(time()-t0))
