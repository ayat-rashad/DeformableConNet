import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

from torch.autograd import Variable
import torch.nn.functional as F


def get_voc_data(dataset='seg'):
    batch_size = 64
    
    transformations = transforms.Compose([
    #transforms.Resize(255),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
        #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ])
    
    if dataset == 'seg':
        dataset = datasets.VOCSegmentation('../data', image_set="train", download=True, transform=transformations)
    elif dataset == 'det':
        dataset = datasets.VOCDetection('../data', image_set="train", download=True, transform=transformations)

    train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader


def get_coco_data():
    pass