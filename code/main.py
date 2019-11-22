from __future__ import print_function
import argparse

#from cnn import *
from seg import *
from util import *

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms

from torch.autograd import Variable
import torch.nn.functional as F


def main():
    # Training settings
    task = 'segmentation2'

    parser = argparse.ArgumentParser(description='PyTorch')
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
    
    ''' 
        Get the data
        Replace with load_data function
    '''
    
    train_loader, test_loader = get_voc_data()
    
    

    if task == 'segmentation':
        model = get_cnn_seg().to(device)
        
    elif task == 'segmentation2':
        model = get_seg_model().to(device)
        
    else:
        model = CNN().to(device)
        
    # model = nn.DataParallel(model)
        
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Start training
    for epoch in range(1, args.epochs + 1):
        # uncomment this and set pretrained to False for training
        #train(args, model, device, train_loader, optimizer, epoch)
        confmat = test(args, model, device, test_loader)
        print(confmat)

    if args.save_model:
        model_name = "model.pt" 
        torch.save(model.state_dict(), model_name)


if __name__ == '__main__':
    main()