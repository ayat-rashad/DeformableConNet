from __future__ import print_function
import argparse

from cnn import *
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

from torch.utils.data import Dataset
from datasets.loadDataset import load_dataset

from torch.autograd import Variable
import torch.nn.functional as F

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    ''' 
        Get the data
        Replace with load_dataset function
    '''
    
    transform = transforms.Compose([#transforms.Resize((256,256)),
                                    transforms.ToTensor()
                                    #,transforms.Normalize((0.1307,), (0.3081,))
                                   ])
    
    # TODO: replace train_data and test_data with load_dataset(...)
    #train_data = datasets.MNIST(root='/home/space/datasets/',
    #                            train=True,
    #                            download=True,
    #                            transform=transform)
    #test_data = datasets.MNIST(root='/home/space/datasets/',
    #                            train=False,
    #                            download=True,
    #                            transform=transform)
    
    train_data, val_data = train_data, val_data = load_dataset('voc2007', transform=transform, type='detection')
    #train_data, val_data = train_data, val_data = load_dataset('voc2012', transform=transform, type='detection')
    
    # For Segmentation
    #train_data, val_data = train_data, val_data = load_dataset('voc2007', transform=transform, type='segmentation')

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, **kwargs)

    model = CNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Start training
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, val_loader)

    if args.save_model:
        model_name = "model.pt" 
        torch.save(model.state_dict(), model_name)


if __name__ == '__main__':
    main()