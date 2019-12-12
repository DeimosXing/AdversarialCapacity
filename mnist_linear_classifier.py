from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from adversarialbox.utils import to_var


class Normalizer:
    # f* = (f - mean) * weight
    def __init__(self, len):
        self.len = len
        self.means = torch.randn(len)
        self.variances = torch.randn(len)

    def update(self, x):
        self.means = np.mean(x.detach().numpy(), axis=0)
        self.variances = np.sum((x.detach().numpy() - self.means) ** 2, axis=0) / x.shape[1]

    def normalize(self, x):
        # this function returns a numpy array
        return (x.detach().numpy() - self.means) / self.variances ** 0.5

    def normalize_tensor(self, x):
        return (x - torch.tensor(self.means)) / torch.tensor(self.variances) ** 0.5


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, 1)
        self.conv2 = nn.Conv2d(2, 4, 3, 1)
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(576, 20)
        # self.fc2 = nn.Linear(128, 10)
        self.fc2 = nn.Linear(20, 1)
        self.normalizer = Normalizer(20)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        # TODO: normalize x-------------
        # print(x.shape)
        self.normalizer.update(x)
        # ------------------------------
        # x = self.dropout2(x)
        # x = self.normalizer.normalize_tensor(x)
        x = self.fc2(x).view(-1)
        # output = F.log_softmax(x, dim=1)
        return x

    def forward_to_features(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).float()
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float()
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += F.mse_loss(output, target, reduction='sum').item()
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred = ((output + 1) / 2).round() * 2 - 1
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def calculate_usefulness(args, model, device, test_loader):
    model.eval()
    mean = 0
    leng = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float()
            leng += data.shape[0]
            features_output = model.normalizer.normalize(model.forward_to_features(data))
            print(features_output.shape)
            print(target.shape)
            mean += np.sum(features_output.T * target.detach().numpy(), axis=1)
            print(mean.shape)

    mean = mean / leng
    return mean


def pgd_attack(model, X_nat, y, feature_id, epsilon=0.3, k=40, a=0.01, rand=True):
    # input one X each time
    if rand:
        X = X_nat + np.random.uniform(-epsilon, epsilon,
                                      X_nat.shape).astype('float32')
    else:
        X = np.copy(X_nat)

    for i in range(k):
        X_var = to_var(torch.from_numpy(X), requires_grad=True)
        y_var = to_var(torch.tensor(y))

        features_output = model.normalizer.normalize_tensor(model.forward_to_features(X_var))[:, feature_id]
        loss = y_var * features_output
        loss.backward()
        grad = X_var.grad.data.cpu().numpy()

        X += a * np.sign(grad)

        X = np.clip(X, X_nat - epsilon, X_nat + epsilon)
        X = np.clip(X, 0, 1)  # ensure valid pixel range

    return X


def robust_usefulness(args, model, device, test_loader):
    model.eval()
    mean = 0
    leng = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float()
            leng += data.shape[0]
            data1 = pgd_attack(model, data, target)
            features_output = model.normalizer.normalize(model.forward_to_features(data))

            mean += np.sum(features_output.T * target.detach().numpy(), axis=1)

    mean = mean / leng
    return mean


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    dataset_mnist_train = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.0,), (0.1,))
                   ]))
    dataset_mnist_train_3and7 = [(element[0], 1.0) for element in dataset_mnist_train if element[1] == 3]
    dataset_mnist_train_3and7.extend(
        [(element[0], -1.0) for element in dataset_mnist_train if element[1] == 7])
    train_loader = torch.utils.data.DataLoader(
        dataset_mnist_train_3and7,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    dataset_mnist_test = datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (0.1,))
    ]))
    dataset_mnist_test_3and7 = [(element[0], 1.0) for element in dataset_mnist_test if element[1] == 3]
    dataset_mnist_test_3and7.extend(
        [(element[0], -1.0) for element in dataset_mnist_test if element[1] == 7])
    test_loader = torch.utils.data.DataLoader(
        dataset_mnist_test_3and7,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()

    if args.save_model:
        # torch.save(model.state_dict(), "mnist_cnn.pt")
        torch.save(model, "mnist_3and7.pt")


if __name__ == '__main__':
    main()
