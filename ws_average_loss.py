import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd, adv_train_grad
from adversarialbox.utils import to_var, pred_batch, test

from math import log2
import numpy as np
import matplotlib.pyplot as plt
import cv2


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


# ==============================parameters initialization
device = torch.device("cuda")
alpha = 1.0
epsilon = 0.2
pred = 3
pred_adv = 7
dataset_size = 10000
lr = 0.05
find_pattern = True
sample_num = 100
iter_times = 1000

# ==============================model initialization
model = torch.nn.Linear(dataset_size, 1, bias=False).to(device)
for param in model.parameters():
    param.data = torch.abs(param.data)
    param.data = param.data / torch.sum(param.data) / 1000
    # param.data = param.data / 1000
    # print(torch.min(param.data))
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer = optim.Adadelta(model.parameters(), lr=lr)

# ===============================load pre-trained net
n = 32
net = NetMul(n)
# net.load_state_dict(torch.load("models/mnist/std_" + str(n) + ".pt"))
net.load_state_dict(torch.load("models/mnist/std_32.pt"))
net.to(device)
net.eval()

# ===============================load sample image
sample = np.load("data/mnist_sample/3.npy")
img = sample.reshape((28, 28))
plt.imshow(img, cmap='gray')
plt.show()

sample = sample.reshape((1, 1, 28, 28))
sample = torch.from_numpy(sample).to(device).float()


# ================================load mnist data
train_dataset = datasets.MNIST(root='../data/', train=True, download=True,
    transform=transforms.ToTensor())
loader_train = torch.utils.data.DataLoader(train_dataset, dataset_size, shuffle=True)
for x, y in loader_train:
    x = x.view(dataset_size, -1)
    x = torch.t(x)
    label = y
    # print(x.shape)
    # print(y.shape)
    break

test_dataset = datasets.MNIST(root='../data/', train=False, download=True,
    transform=transforms.ToTensor())
loader_test = torch.utils.data.DataLoader(test_dataset, sample_num, shuffle=True)

for x1, y1 in loader_test:
    samples = x1.view(sample_num, -1)
    # samples = torch.t(samples)
    preds = y1
    # print(x.shape)
    # print(y.shape)
    break

print(samples[0].shape)
print(x.shape)
x_var = x.to(device)

# ==================================
blocks_num = 28  # per row
block_width = int(28 / blocks_num)
total_blocks = blocks_num * blocks_num
seq = list(range(total_blocks))
np.random.shuffle(seq)
s = [0] * 784
for i in range(len(seq)):
    nnn = seq[i]
    d1 = int(block_width * (i % blocks_num))
    d2 = int(784 / blocks_num * (i // blocks_num))
    d3 = int(block_width * (nnn % blocks_num))
    d4 = int(784 / blocks_num * (nnn // blocks_num))
    for j in range(block_width):
        s[28 * j + d1 + d2:28 * j + d1 + d2 + block_width] = list(range(28 * j + d3 + d4, 28 * j + d3 + d4 + block_width))
x_var = x_var[s, :]
tmp = x_var[:, 0]
tmp = tmp.cpu().detach().numpy().reshape(28, 28)
plt.imshow(tmp, cmap='gray', vmin=0, vmax=1)
plt.show()
# ====================================

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

# =======================optimize
sum_loss_origin = 0
sum_loss_ws = 0
sum_loss_adv = 0

for __ in range(sample_num):
    sample = samples[__].to(device).float().view(1, 1, 28, 28)
    pred = int(preds[__].detach().numpy())
    print(pred)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    loss = 100
    loss_min = 100

    for param in model.parameters():
        param.data = torch.abs(param.data)
        param.data = param.data / torch.sum(param.data) / 1000

    for _ in range(iter_times):

        optimizer.zero_grad()

        if False:
            if _ % 2000 == 0 and _ < 50000:
                for param in model.parameters():
                    param.data = param.data / 2

        scale = torch.sum(model.weight)
        # t = torch.clamp(model(x_var) / scale, 0, 1)
        noise = model(x_var)
        # noise = model(x_var) / scale

        # print("min: ", torch.min(model(x_var)).cpu())
        noise = torch.reshape(noise, sample.shape)

        new_img = torch.clamp(sample + noise, 0, 1)
        new_img = torch.where(new_img > sample + epsilon, sample + epsilon, new_img)
        new_img = torch.where(new_img < sample - epsilon, sample - epsilon, new_img)

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
        if False:
            for p in model.parameters():
                p.data = torch.clamp(p.data, 0, 1)

        if (_ + 1) % 100 == 0:
            # print(l - loss)
            # print("softmax of ", pred, ': ', output[0][pred])
            # print("softmax of ", pred_adv, ': ', output[0][pred_adv])
            print(_ + 1)

    # print(output[0])
    # print("softmax of ", pred, ': ', output[0][pred])
    # print("softmax of ", pred_adv, ': ', output[0][pred_adv])

    img = new_img.cpu().detach().numpy().reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.show()

    print("best:", output_min[0])
    img = img_min.cpu().detach().numpy().reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.show()

    loss_ws = output_min[0][pred]

    # adversary = LinfPGDAttack(epsilon=epsilon)
    adversary = LinfPGDAttack(epsilon=epsilon, k=1000, a=0.001)
    # adversary = FGSMAttack(epsilon=epsilon)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    x = sample.reshape(1, 1, 28, 28).cpu()

    y = torch.tensor([pred])
    x_var1, y_var = to_var(x), to_var(y.long())
    x_adv, grad = adv_train_grad(x, y, net, criterion, adversary)
    x_adv_var = to_var(x_adv)
    scores = net(x_var1)
    scores_adv = net(x_adv_var)

    loss_origin = scores.data.cpu()[0].detach().numpy()[pred]
    loss_adv = scores_adv.data.cpu()[0][pred]

    print(loss_origin)
    print(loss_ws)
    print(loss_adv)
    sum_loss_origin += loss_origin
    sum_loss_ws += loss_ws
    sum_loss_adv += loss_adv

print(sum_loss_origin / sample_num)
print(sum_loss_ws / sample_num)
print(sum_loss_adv / sample_num)