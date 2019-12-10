"""
Adversarial attacks on LeNet5
"""
from time import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
#add higher directory to the module path
import sys
sys.path.append('..')

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.utils import to_var, pred_batch, test, attack_over_test_data
from nets.mnist_nets import *
from dataset_utils import dataset_two_cls
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='PyTorch MNIST adversarial training')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='N')
parser.add_argument('--epsilon', type=float, default=0.3)
parser.add_argument('--model-path', type=str, default='models',
                    help='path to saved ckpt')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--attacker', type=str, default='fgsm',
                    help='adversarial attacker to train (default: fgsm)')

def main():
    args = parser.parse_args()
    param = {k: v for k, v in args._get_kwargs()}
    # Hyper-parameters
    #param = {
    #    'test_batch_size': 100,
    #    'epsilon': 0.3,
    #}
    device = torch.device("cuda" if param['cuda'] else "cpu")
    use_cuda = args.cuda and torch.cuda.is_available()
    # Data loaders
    mnist_testset = datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    #mnist_testset = dataset_two_cls(mnist_testset, 3, 7, False)
    loader_test = torch.utils.data.DataLoader(
        mnist_testset,
        batch_size=param['test_batch_size'], shuffle=True)

    # Setup model to be attacked
    for capacity, Net in enumerate([Cnn_2_4, Cnn_4_8, Cnn_8_16]):
        model = Net().to(device)
        if use_cuda:
            device_ids = torch.cuda.device_count()
            if device_ids > 1:
                # Data parallel if # gpu > 1
                model = torch.nn.DataParallel(model)
        model_path = os.path.join(param['model_path'], param['attacker'], 'ckpts', 'adv_trained_{}_{}.pkl'.format(param['attacker'], capacity))
        if not os.path.exists(model_path):
            print(model_path)
            raise RuntimeError('Pretrained model doesn\'t exist, check your model path')
        model.load_state_dict(torch.load(model_path))
        # for p in model.parameters():
        #     p.requires_grad = False
        model.eval()
        test(model, loader_test)
        # Adversarial attack
        if param['attacker'] == 'fgsm':
            adversary = FGSMAttack(model, param['epsilon'])
        elif param['attacker'] == 'pgd':
            adversary = LinfPGDAttack(model, random_start=False)
        else:
            raise RuntimeError('invalid attacker')
        t0 = time()
        attack_over_test_data(model, adversary, param, loader_test)
        print('{}s eclipsed.'.format(time() - t0))


if __name__ == '__main__':
    main()
