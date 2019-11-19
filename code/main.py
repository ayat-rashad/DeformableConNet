from __future__ import print_function
import argparse

#from cnn import *
from seg import *

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms

from torch.autograd import Variable
import torch.nn.functional as F


def main():
    # Training settings
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

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    ''' 
        Get the data
        Replace with load_data function
    '''
    
    # train_loader, test_loader = load_data(dataset= ....)
    
    transformations = transforms.Compose([
    #transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        #(0.1307,), (0.3081,))
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    ])
    
    shuffle = False
    task = 'segmentation'
    
    dataset_train = datasets.VOCSegmentation('../data', year='2008', image_set="train", download=True,
                       transforms=transformations)
    dataset_test = datasets.VOCSegmentation('../data', year='2008', image_set="val", download=True,
                       transforms=transformations)
    
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=4,
        **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.test_batch_size,
        num_workers=4,
        shuffle=shuffle, **kwargs)

    if task == 'segmentation':
        model = get_cnn_seg().to(device)
    else:
        model = CNN().to(device)
        
    # model = nn.DataParallel(model)
        
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Start training
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if args.save_model:
        model_name = "model.pt" 
        torch.save(model.state_dict(), model_name)


if __name__ == '__main__':
    main()