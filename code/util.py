import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

from torch.autograd import Variable
import torch.nn.functional as F


def get_voc_data(dataset='seg'):
    batch_size = 64
    shuffle = False
    
    transformations = transforms.Compose([
    #transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        #(0.1307,), (0.3081,))
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    ])
    
    trgt_transformations = transforms.Compose([
    #transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    ])
    
    dataset_train = datasets.VOCSegmentation('../data', year='2008', image_set="train", download=True,
                       transform=transformations, target_transform=trgt_transformations)
    dataset_test = datasets.VOCSegmentation('../data', year='2008', image_set="val", download=True,
                       transform=transformations, target_transform=trgt_transformations)
    
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        num_workers=4,
        shuffle=shuffle, **kwargs)   
    
    return train_loader, test_loader


def get_coco_data():
    pass


# Evaluation metrics

