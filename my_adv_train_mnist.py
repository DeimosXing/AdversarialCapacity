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
pred = 1
pred_adv = 7
dataset_size = 10000
lr = 0.05
find_pattern = True
iter_times = 2000

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
net.load_state_dict(torch.load("models/mnist/std_" + str(n) + ".pt"))
# net.load_state_dict(torch.load("models/mnist/fgsm-10-20.pkl"))
net.to(device)
net.eval()

# ===============================load sample image
sample = np.load("data/mnist_sample/1.npy")
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

# =======================optimize
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
    # ----TODO: find noise_scale
    t1 = torch.zeros(28 * 28).reshape(sample.shape)
    t2 = torch.zeros(28 * 28).reshape(sample.shape) + 1
    t3 = sample - epsilon
    t4 = sample + epsilon
    s = list()
    s.append(torch.max(noise / (t2.to(device) - sample)).cpu())
    s.append(torch.max(noise / (t4.to(device) - sample)).cpu())
    s.append(torch.max(noise / (t1.to(device) - sample)).cpu())
    s.append(torch.max(noise / (t3.to(device) - sample)).cpu())
    # noise = noise / max(s))
    new_img = torch.clamp(sample + noise, 0, 1)
    new_img = torch.where(new_img > sample + epsilon, sample + epsilon, new_img)
    new_img = torch.where(new_img < sample - epsilon, sample - epsilon, new_img)
    # new_img = sample + noise
    # print(torch.min(noise))
    # print(torch.max(noise))
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

print("best:", output_min[0])
img = img_min.cpu().detach().numpy().reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.show()


# ========================find pattern
if find_pattern:
    # Train the model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    index = 0

    # Adversarial training setup
    # adversary = FGSMAttack(epsilon=epsilon)
    # adversary = LinfPGDAttack(epsilon=epsilon, k=2000, a=0.001)
    adversary = LinfPGDAttack(epsilon=epsilon)

    x = sample.reshape(1, 1, 28, 28).cpu()
    y = torch.tensor([pred])
    x_var, y_var = to_var(x), to_var(y.long())
    x_adv, grad = adv_train_grad(x, y, net, criterion, adversary)
    x_adv_var = to_var(x_adv)
    scores = net(x_var)
    scores_adv = net(x_adv_var)

    print("origin image:", scores.data.cpu()[index].detach().numpy())
    print(scores.data.cpu().max(1)[1][index].detach().numpy())
    # print("FSGM adv image:", scores_adv.data.cpu()[index])
    print("PGD adv image:", scores_adv.data.cpu()[index])
    print(scores_adv.data.cpu().max(1)[1][index].detach().numpy())

    grad = grad[index][0]
    scale = torch.max(torch.abs(grad))
    grad = grad / scale
    red = torch.clamp(grad, 0, 1)
    blue = torch.clamp(grad, -1, 0)
    blue = torch.abs(blue)
    red = red.cpu().detach().numpy()
    blue = blue.cpu().detach().numpy()
    g = np.zeros((28, 28), dtype='float32')
    pattern = cv2.merge([red, g, blue])
    plt.imshow(pattern)
    plt.show()

    img_adv = x_adv[index][0].cpu().detach().numpy()
    plt.imshow(img_adv, cmap='gray')
    plt.show()

    # print(torch.min(model.weight))
    # print(torch.max(model.weight))
    weight = model.weight.cpu().detach().numpy()

    plt.hist(x=weight[0], bins=200)
    plt.show()

# average_weight = [np.mean([weight[0][_] for _ in range(dataset_size) if (label[_] == i and np.abs(weight[0][_]) > 0.01)]) for i in range(10)]
# print(np.array(average_weight) * 1000)
average_weight = [np.mean([weight[0][_] for _ in range(dataset_size) if (label[_] == i and np.abs(weight[0][_]) >= 0.0)]) for i in range(10)]
print(np.array(average_weight) * 1000)
