#from __future__ import print_function
import argparse
import os

#from cnn import *
from seg import *
import util

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

from torch.utils.data import Dataset
from datasets.loadDataset import load_dataset
import visualize as viz
from collate import pad_collate

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
    parser.add_argument('--layers', type=int, default=0, metavar='L',
                        help='How many deformable layers')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs =  {}
    

    # Get Data For Segmentation
    #train_loader, test_loader = util.get_voc_data(batch_size=10, test_batch_size=1, year='2008',
    #                                          root='../data', download=False)
    
    transformations = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor()
    #    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #)
    ])
    
    def proc_img(img):
        arr = np.array(img)
        arr.dtype = np.int8
        return arr

    trgt_transformations = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.Lambda(proc_img),
    transforms.ToTensor()
    ])

    train_data,test_data = load_dataset('voc2012', type='segmentation', transform=transformations,
                                      target_transform=trgt_transformations)

    train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=10,
            #sampler=torch.utils.data.sampler.RandomSampler(tr),
            num_workers=4,
            drop_last=True
            )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        num_workers=4,
        drop_last=True,
        shuffle=False
    )   


    if task == 'segmentation':
        model = get_cnn_seg().to(device)
        
    elif task == 'segmentation2':
        model = get_seg_model(replace_layers='dconv', n_layers=args.layers).to(device)
        
    else:
        model = CNN().to(device)
        
    if torch.cuda.device_count() > 1:        
        model = nn.DataParallel(model)
        
    # Get the parameters to optimize
    feature_extract = True
    params_to_update = None

    if feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        
    optimizer = optim.SGD(params_to_update, lr=args.lr, momentum=args.momentum)
    
    print "Starting Training for Deformable Convolutional Layers:", args.layers
    # Start training
    for epoch in range(1, args.epochs + 1):
        # uncomment this and set pretrained to False for training
        train(args, model, device, train_loader, optimizer, epoch)
        print "Finished Training"
        
        confmat = test(args, model, device, test_loader)
        print "Testing Results"
        print(confmat) 

        
    if args.save_model:
        model_name = "model.pt" 
        torch.save(model.state_dict(), model_name)


if __name__ == '__main__':
    main()
