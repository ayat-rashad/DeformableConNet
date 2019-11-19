import numpy as np
import torch
import torchvision
#import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

from torch.autograd import Variable
import torch.nn.functional as F


def get_cnn_seg():        
        # Get pretrained ResNet model 
        model = models.resnet101(pretrained=True)  # backbone
     
        '''
        # possible models: 
        model = models.segmentation.fcn_resnet101(pretrained=False) 
        '''
        
        # Turn off training for their parameters
        for param in model.parameters():
            param.requires_grad = False
            
        classifier_input = model.fc.in_features
        
        # Build a new classifier
        classifier = nn.Sequential(nn.Linear(classifier_input, n_classes),
                                   #nn.ReLU(),
                                   #nn.Linear(1024, n_classes),
                                   #nn.ReLU(),
                                   #nn.Linear(512, n_classes),
                                   nn.LogSoftmax(dim=1)
                                  )
        # Replace default classifier with new classifier
        # model.classifier = classifier
        model.fc = classifier
        
        return model
        
                
        
def train(args, model, device, train_loader, optimizer, epoch, phase='train', crt=
         'cent'):
    if phase == 'train':
        model.train()
    #else:
    #    model.eval()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        #loss = F.nll_loss(output, target)
        if crt == 'cent':
            criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        
        if phase == 'train':
            loss.backward()
            optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, crt='cent'):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            if crt == 'cent':
                criterion = nn.CrossEntropyLoss()
            
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    
