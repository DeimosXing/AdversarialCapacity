"""
Adversarially train CNN on MNIST
"""

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
import os

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test
from nets.mnist_nets import Cnn_2_4, Cnn_4_8, Cnn_8_16

parser = argparse.ArgumentParser(description='PyTorch MNIST adversarial training')
parser.add_argument('--num-epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='N')
parser.add_argument('--delay', type=int, default=10, metavar='N',
                    help=' (default: 10)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3)')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='weight_decay (default: 5e-4)')
parser.add_argument('--attacker', type=str, default='fgsm',
                    help='adversarial attacker to train (default: fgsm)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--save-path', type=str, default='.',
                    help='path to save ckpt and results')

def train(net, param, loader_train, adversary):
    net.train()
    # Train the model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=param['lr'], momentum=0.9,
        weight_decay=param['weight_decay'])

    for epoch in range(param['num_epochs']):
        print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))
        for t, (x, y) in enumerate(loader_train):

            x_var, y_var = to_var(x), to_var(y.long())
            loss = criterion(net(x_var), y_var)

            # adversarial training
            if epoch+1 > param['delay']:
                # use predicted label to prevent label leaking
                y_pred = pred_batch(x, net)
                x_adv = adv_train(x, y_pred, net, criterion, adversary)
                x_adv_var = to_var(x_adv)
                loss_adv = criterion(net(x_adv_var), y_var)
                loss = (loss + loss_adv) / 2

            if (t + 1) % 100 == 0:
                print('t = %d, loss = %.8f' % (t + 1, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def main():
    args = parser.parse_args()
    param = {k: v for k, v in args._get_kwargs()}
    # param = {
    #     'batch_size': 128,
    #     'test_batch_size': 100,
    #     'num_epochs': 15,
    #     'delay': 10,
    #     'lr': 1e-3,
    #     'weight_decay': 5e-4,
    #     'attacker':'fgsm',
    #     'cuda':False,
    #     'save_path' : 'models'
    # }
    device = torch.device("cuda" if param['cuda'] else "cpu")
    use_cuda = args.cuda and torch.cuda.is_available()
    # Data loaders
    train_dataset = datasets.MNIST(root='../data/', train=True, download=True,
                                   transform=transforms.ToTensor())
    loader_train = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=param['batch_size'], shuffle=True)
    test_dataset = datasets.MNIST(root='../data/', train=False, download=True,
                                  transform=transforms.ToTensor())
    loader_test = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=param['test_batch_size'], shuffle=True)

    save_dir = param['save_path']
    ckpt_save_dir = os.path.join(save_dir, 'ckpts')
    # log_save_dir = os.path.join(save_dir, 'logs')
    if not os.path.exists(ckpt_save_dir):
        os.mkdir(ckpt_save_dir)
    # Adversarial training setup
    if param['attacker'] == 'fgsm':
        adversary = FGSMAttack(epsilon=0.3)
    elif param['attacker'] == 'pgd':
        adversary = LinfPGDAttack()
    else:
        raise RuntimeError('invalid attacker')
    # Setup the model
    for capacity, Net in enumerate([Cnn_2_4, Cnn_4_8, Cnn_8_16]):
        model = Net().to(device)
        if use_cuda:
            device_ids = range(torch.cuda.device_count())
            if device_ids > 1:
                # Data parallel if # gpu > 1
                model = torch.nn.DaraParallel(model)
        train(model, param, loader_train, adversary)
        test(model, loader_test)
        torch.save(model.state_dict(), os.path.join(ckpt_save_dir, 'adv_trained_{}_{}.pkl'.format(param['attacker'], capacity)))

if __name__ == '__main__':
    main()
