import numpy as np
import torch
import torchvision
#import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

from torch.autograd import Variable
import torch.nn.functional as F


'''from coco_utils import get_coco, get_coco_kp

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import train_one_epoch, evaluate
'''

from util import *



def get_cnn_seg(n_classes=21):        
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
        '''classifier = nn.Sequential(nn.Linear(classifier_input, n_classes),
                                   #nn.ReLU(),
                                   #nn.Linear(1024, n_classes),
                                   #nn.ReLU(),
                                   #nn.Linear(512, n_classes),
                                   nn.LogSoftmax(dim=1)
                                  )'''
        
        # Replace default classifier with new classifier
        # model.classifier = classifier
        #model.fc = classifier
        
        head = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                             #nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                             nn.ReLU(),
                             #Dropout(p=0.1, inplace=False),
                             nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
                                  )
        
        #model.avgpool = nn.Identity()
        model.fc = head
        
        return model
    
    
def get_seg_model(n_classes=20, model_name='deeplab', pretrained=True):        
    # Get pretrained deeplab model 
    if model_name == 'deeplab':
        model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained)

    # Turn off training for their parameters
    #for param in model.parameters():
    #    param.requires_grad = False
    
    return model
            

def criterion(inputs, target, reduction='sum'):
    losses = {}
    target = target.squeeze(axis=1).long()
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=-1, reduction=reduction)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']

                
        
def train(args, model, device, train_loader, optimizer, epoch, phase='train', crt=
         'cent'):
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = target
        optimizer.zero_grad()
        output = model(data)
        
        #loss = F.nll_loss(output, target)
        #if crt == 'cent':
        #    criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        '''if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))'''


def test(args, model, device, test_loader, crt='cent'):
    model.eval()
    test_loss = 0
    correct = 0
    confmat = ConfusionMatrix(20)
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target
            output = model(data)
            
            #if crt == 'cent':
            #    criterion = nn.CrossEntropyLoss()
            
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
            pred = output['out'].argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()
            confmat.update(target.flatten(), pred.flatten())

        confmat.reduce_from_all_processes()

    #test_loss /= len(test_loader.dataset)

    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #    test_loss, correct, len(test_loader.dataset),
    #    100. * correct / len(test_loader.dataset)))
    return confmat
    
    
