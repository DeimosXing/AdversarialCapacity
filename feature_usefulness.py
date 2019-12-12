from __future__ import print_function
import argparse
import torch
from mnist_linear_classifier import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


def main():
    # settings
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
    parser.add_argument('--model-path', type=str, default='mnist_3and7.pt', metavar='P',
                        help='model path (default: mnist_3and7.pt)')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    model = Net()
    # model.load_state_dict(torch.load('mnist_cnn.pt'))
    model = torch.load(args.model_path)
    model.eval()

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

    print(model.normalizer.means)
    features_usefulness = calculate_usefulness(args, model, device, test_loader)
    print(features_usefulness)

    np.set_printoptions(suppress=True)
    print('model usefulness ',
          model.fc2.weight.detach().numpy() * model.normalizer.variances ** 0.5 * features_usefulness)
    # print('model bias',
    #       model.normalizer.means * model.normalizer.variances ** 0.5 * model.fc2.weight.detach().numpy() + model.fc2.bias.detach().numpy())
    # print(model.normalizer.means)
    # print(features_usefulness + model.normalizer.means)
    # print(model.normalizer.variances)
    # print(model.fc2.bias)
    # print(model.fc2.weight)
    print(model.parameters())

    # pgd_attack(model, )


if __name__ == '__main__':
    main()
