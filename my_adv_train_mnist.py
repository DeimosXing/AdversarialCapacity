import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test

from math import log2
import numpy as np
import matplotlib.pyplot as plt


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
alpha = 1.0
epsilon = 0.2
pred = 2
pred_adv = 7
dataset_size = 10000

# model initialization
model = torch.nn.Linear(dataset_size, 1, bias=False).to(device)
for param in model.parameters():
    param.data = torch.abs(param.data) / 10
    # print(torch.min(param.data))
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer = optim.Adadelta(model.parameters(), lr=0.1)

# load pre-trained net
n = 32
net = NetMul(n)
net.load_state_dict(torch.load("models/mnist/std_" + str(n) + ".pt"))
net.to(device)
net.eval()

# load sample image
sample_2 = np.load("data/mnist_sample/2_7.npy")
img = sample_2.reshape((28, 28))
plt.imshow(img, cmap='gray')
plt.show()

sample_2 = sample_2.reshape((1, 1, 28, 28))
sample_2 = torch.from_numpy(sample_2).to(device).float()


# load mnist data
test_dataset = datasets.MNIST(root='../data/', train=False, download=True,
    transform=transforms.ToTensor())
loader_test = torch.utils.data.DataLoader(test_dataset, dataset_size, shuffle=True)
for x, y in loader_test:
    x = x.view(dataset_size, -1)
    x = torch.t(x)
    # print(x.shape)
    # print(y.shape)
    break

x_var = x.to(device)

loss = 100
loss_min = 100
"""
for _ in range(10000):
    model.zero_grad()

    scale = torch.sum(model.weight)
    # for param in model.parameters():
    #     scale = torch.sum(param)
    t = torch.clamp(model(x_var) / scale, 0, 1)
    t = torch.reshape(t, sample_2.shape)
    new_img = torch.clamp(sample_2 + t, 0, 1)
    new_img = torch.where(new_img > sample_2 + epsilon, sample_2 + epsilon, new_img)
    new_img = torch.where(new_img < sample_2 - epsilon, sample_2 - epsilon, new_img)
    output = net(new_img)

    l = loss
    loss = alpha * output[0][2] - (1 - alpha) * output[0][3]
    loss.backward()
    for param in model.parameters():
        param.data.add_(-0.01 * param.grad.data)
    if (_ + 1) % 10 == 0:
        print(l - loss)
        print("softmax of 2:", output[0][2])
        print("softmax of 3:", output[0][3])
        print(_ + 1)
    if torch.abs(l - loss) < 1e-7:
        break
"""
for _ in range(2000):
    optimizer.zero_grad()

    scale = torch.sum(model.weight)
    # for param in model.parameters():
    #     scale = torch.sum(param)
    t = torch.clamp(model(x_var) / scale, 0, 1)
    t = torch.reshape(t, sample_2.shape)
    new_img = torch.clamp(sample_2 + t, 0, 1)
    new_img = torch.where(new_img > sample_2 + epsilon, sample_2 + epsilon, new_img)
    new_img = torch.where(new_img < sample_2 - epsilon, sample_2 - epsilon, new_img)
    output = net(new_img)
    if _ == 0:
        print(output)

    l = loss
    loss = alpha * output[0][pred] - (1 - alpha) * output[0][pred_adv]
    if loss < loss_min:
        loss_min = loss
        img_min = new_img
        output_min = output
    loss.backward()
    optimizer.step()
    for p in model.parameters():
        p.data = torch.clamp(p.data, 0, 1)

    if (_ + 1) % 10 == 0:
        print(l - loss)
        print("softmax of ", pred, ': ', output[0][pred])
        print("softmax of ", pred_adv, ': ', output[0][pred_adv])
        print(_ + 1)


print(output[0])
print("softmax of ", pred, ': ', output[0][pred])
print("softmax of ", pred_adv, ': ', output[0][pred_adv])

img = new_img.cpu().detach().numpy().reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.show()

print(output_min)
img = img_min.cpu().detach().numpy().reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.show()

for param in model.parameters():
    print(torch.min(param.data))
