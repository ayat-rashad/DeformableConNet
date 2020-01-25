import numpy as np
import torch
import torchvision
#import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

from torch.autograd import Variable
import torch.nn.functional as F

import def_conv

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
    
    
def get_seg_model(n_classes=21, model_name='deeplab', pretrained=True, replace_layers='dconv', n_layers=1, use_cuda=True):     
    # Get pretrained deeplab model 
    if model_name == 'deeplab':
        model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained)
    
    if replace_layers is not None:
        set_parameter_requires_grad(model, True)
        deformable_groups = 1
        #kW = KH = 3
        #inC = 2048

        layers_to_replace = [
            model.classifier[0].convs[3],
            model.classifier[0].convs[2],
            model.classifier[0].convs[1],
            model.backbone.layer4[0],
            model.backbone.layer4[1],
            model.backbone.layer4[2] 
        ]
        
        conv_op = None
        
        for i in range(n_layers):
            c = layers_to_replace[i][0]
            l_inC = c.in_channels
            l_oC = c.out_channels
            l_k = c.kernel_size
            l_stride = c.stride
            l_padding = c.padding
            l_dilation = c.dilation
            l_bias = c.bias
            
            if replace_layers == 'dconv':
                conv_op = def_conv.DeformConvPack(in_channels=l_inC, out_channels=l_oC, kernel_size=l_k,
                                                  stride=l_stride, padding=1,
                                      dilation=1, bias=l_bias, im2col_step=1)
            else:
                conv_op = nn.Conv2d(in_channels=l_inC, out_channels=l_oC, kernel_size=l_k,
                                                  stride=l_stride, padding=l_padding,
                                      dilation=l_dilation, bias=l_bias)
               
            if use_cuda:
                conv_op = conv_op.cuda()
                
            layers_to_replace[i][0] = conv_op
            set_parameter_requires_grad(model.classifier, False)
            #model.classifier[0].convs[3] = layer
    
    return model
            

def criterion(inputs, target, reduction='mean'):
    losses = {}
    target = target.squeeze(1).long()
    
    for name, x in inputs.items():
        x = nn.functional.softmax(x,1)
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=-1, reduction=reduction)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']

                
        
def train(args, model, device, train_loader, optimizer, epoch, phase='train', crt='cent'):
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = target
        optimizer.zero_grad()
        output = model(data)
                
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
    confmat = ConfusionMatrix(21)
    
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
    
    
    
def set_parameter_requires_grad(model, feature_extracting):
    for param in model.parameters():
        if feature_extracting:
            param.requires_grad = False
        else:
            param.requires_grad = True
 
