from __future__ import print_function
import argparse
import torch
import mnist_linear_classifier
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

def main():
    model = mnist_linear_classifier.Net()
    model.load_state_dict(torch.load('mnist_cnn.pt'))
    model.eval()


if __name__ == '__main__':
    main()