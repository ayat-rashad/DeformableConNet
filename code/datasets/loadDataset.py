from __future__ import print_function
import os
from datasets.VOCDataset import VOCDetection, VOCSegmentation
from datasets.COCODataset import COCODetection, COCOSegmentation

import torch
from torchvision import datasets, transforms, models

def load_dataset(dataset, transform=None, target_transform=None, type='detection'):
    '''
    Load dataset
    
    Args:
        dataset (string) ['coco', 'voc2007', 'voc2012']
        transform (callable, optional)
        type (string) : ['detection', segmentation']
        image_set (array) : (train, val) can be (train, test) as well   # Plan to add - not sure yet.
    '''
    #data_path = data_root
    #ann_path = data_root
    
    if type.lower() not in ['detection', 'segmentation']:
        raise RuntimeError('Please choose between "detection" and "segmentation" dataset')
        
    if dataset.lower() == 'voc2007':
        if type.lower() == 'detection':
            data_train = VOCDetection(root='/home/space/datasets/VOCdevkit/', 
                                    year='2007', 
                                    image_set='train', 
                                    transform=transform,
                                    target_transform=target_transform)
            data_val = VOCDetection(root='/home/space/datasets/VOCdevkit/', 
                                    year='2007', 
                                    image_set='val', 
                                    transform=transform,
                                    target_transform=target_transform)
            
        elif type.lower() == 'segmentation':
            data_train = VOCSegmentation(root='/home/space/datasets/VOCdevkit/', 
                                        year='2007', 
                                        image_set='train', 
                                        transform=transform,
                                        target_transform=target_transform)
            data_val = VOCSegmentation(root='/home/space/datasets/VOCdevkit/', 
                                        year='2007', 
                                        image_set='val', 
                                        transform=transform,
                                        target_transform=target_transform)
        else:
            raise ValueError('Operation type %s is unknown. Please choose between "detection" and "segmentation".'%type)
            
    elif dataset.lower() == 'voc2012':
        if type.lower() == 'detection':
            data_train = VOCDetection(root='/home/space/datasets/VOCdevkit/', 
                                    year='2012', 
                                    image_set='train', 
                                    transform=transform)
            data_val = VOCDetection(root='/home/space/datasets/VOCdevkit/', 
                                    year='2012', 
                                    image_set='val', 
                                    transform=transform,
                                    target_transform=target_transform)
            
        elif type.lower() == 'segmentation':
            data_train = VOCSegmentation(root='/home/space/datasets/VOCdevkit/', 
                                        year='2012', 
                                        image_set='train', 
                                        transform=transform,
                                        target_transform=target_transform)
            data_val = VOCSegmentation(root='/home/space/datasets/VOCdevkit/', 
                                        year='2012', 
                                        image_set='val', 
                                        transform=transform,
                                        target_transform=target_transform)
        else:
            raise ValueError('Operation type %s is unknown. Please choose between "detection" and "segmentation".'%type)
            
    elif dataset.lower() == 'coco':
        if type.lower() == 'detection':
            data_train = COCODetection(root='/home/space/datasets/coco/',
                                    image_set='train', 
                                    transform=transform,
                                    target_transform=target_transform)
            data_val = COCODetection(root='/home/space/datasets/coco/',
                                    image_set='val', 
                                    transform=transform,
                                    target_transform=target_transform)
            
        elif type.lower() == 'segmentation':
            data_train = COCOSegmentation(root='/home/space/datasets/coco/',
                                        image_set='train', 
                                        transform=transform,
                                        target_transform=target_transform)
            data_val = COCOSegmentation(root='/home/space/datasets/coco/',
                                        image_set='val', 
                                        transform=transform,
                                        target_transform=target_transform)
    else:
        raise ValueError('Dataset %s is not supported. Please choose between "coco", "voc2007", and "voc2012".'%dataset)
    
    
    return data_train, data_val