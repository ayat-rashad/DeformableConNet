import numpy as np
import torch
import torch.nn as nn
import torchvision
#import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from torch.autograd import Variable
import torch.nn.functional as F

'''
TODO:
    - evaluation metrics: https://pytorch.org/blog/torchvision03/
        - mIoU
        - mAP
    - structure
'''


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        c_in1 = 3
        c_out1 = c_in2 = 20 
        c_out2 = 50
        k_size = 5
        fc_in1 = 4*4*c_out2 #k_size**2 * c_out2
        fc_in2 = fc_out1 = 500
        fc_out2 = 10
    
        self.conv1 = nn.Conv2d(c_in1, c_out1, k_size, 1)
        self.conv2 = nn.Conv2d(c_in2, c_out2, k_size, 1)
        self.fc1 = nn.Linear(fc_in1, fc_out1)
        self.fc2 = nn.Linear(fc_in2, fc_out2)

        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
