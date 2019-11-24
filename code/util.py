import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

from torch.autograd import Variable
import torch.nn.functional as F


def get_voc_data(dataset='seg', batch_size = 64, test_batch_size=1, year='2008', root='../data/VOCdevkit', download=False):
    shuffle = False
    kwargs =  {}
    
    transformations = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
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
    
    dataset_train = None
    dataset_test = None
    
    if dataset == 'seg':
        dataset_train = datasets.VOCSegmentation(root, year=year, image_set="train", download=download,
                           transform=transformations, target_transform=trgt_transformations)
        dataset_test = datasets.VOCSegmentation(root, year=year, image_set="val", download=download,
                           transform=transformations, target_transform=trgt_transformations)
        
    elif dataset == 'det':
        dataset_train = datasets.VOCDetection(root, year=year, image_set="train", download=download,
                           transform=transformations, target_transform=trgt_transformations)
        dataset_test = datasets.VOCDetection(root, year=year, image_set="val", download=download,
                           transform=transformations, target_transform=trgt_transformations)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=test_batch_size,
        num_workers=4,
        shuffle=shuffle, **kwargs)   
    
    return train_loader, test_loader


def get_cityscape_data(dataset='seg', test_batch_size=1, year='2008'):
    batch_size = 64
    shuffle = False
    kwargs =  {}
    
    transformations = transforms.Compose([
    transforms.ToTensor(),
    ])
    
    trgt_transformations = transforms.Compose([
    transforms.ToTensor(),
    ])
    
    dataset_train = datasets.Cityscapes('../data', split='train', mode='fine',
                     target_type='semantic', transform=transformations, target_transform=trgt_transformations)
    dataset_test = datasets.Cityscapes('../data', split='val', mode='fine',
                     target_type='semantic', transform=transformations, target_transform=trgt_transformations)
    
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        **kwargs) 
    
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=test_batch_size,
        num_workers=4,
        shuffle=shuffle, **kwargs)   
    
    return train_loader, test_loader



def get_coco_data():
    pass


# Evaluation metrics
class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)
        
    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100)


