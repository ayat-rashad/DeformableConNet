from __future__ import print_function
import os
from datasets.VOCDataset import VOCDetection, VOCSegmentation

import torch
from torchvision import datasets, transforms, models

def load_dataset(dataset, transform=None, type='detection'):
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
            data_train = VOCDetection(root='/home/space/datasets/VOCdevkit', 
                                    year='2007', 
                                    image_set='train', 
                                    transform=transform)
            data_val = VOCDetection(root='/home/space/datasets/VOCdevkit', 
                                    year='2007', 
                                    image_set='val', 
                                    transform=transform)
            
        elif type.lower() == 'segmentation':
            data_train = VOCSegmentation(root='/home/space/datasets/VOCdevkit', 
                                        year='2007', 
                                        image_set='train', 
                                        transform=transform)
            data_val = VOCSegmentation(root='/home/space/datasets/VOCdevkit', 
                                        year='2007', 
                                        image_set='val', 
                                        transform=transform)
        else:
            raise ValueError('Operation type %s is unknown. Please choose between "detection" and "segmentation".'%type)
            
    elif dataset.lower() == 'voc2012':
        if type.lower() == 'detection':
            data_train = VOCDetection(root='/home/space/datasets/VOCdevkit', 
                                    year='2012', 
                                    image_set='train', 
                                    transform=transform)
            data_val = VOCDetection(root='/home/space/datasets/VOCdevkit', 
                                    year='2012', 
                                    image_set='val', 
                                    transform=transform)
            
        elif type.lower() == 'segmentation':
            data_train = VOCSegmentation(root='/home/space/datasets/VOCdevkit', 
                                        year='2012', 
                                        image_set='train', 
                                        transform=transform)
            data_val = VOCSegmentation(root='/home/space/datasets/VOCdevkit', 
                                        year='2012', 
                                        image_set='val', 
                                        transform=transform)
        else:
            raise ValueError('Operation type %s is unknown. Please choose between "detection" and "segmentation".'%type)
            
    elif dataset.lower() == 'coco':
        print('coco')
    else:
        raise ValueError('Dataset %s is not supported. Please choose between "coco", "voc2007", and "voc2012".'%dataset)
    
    if 'coco' in dataset:
        train_path = data_root+'coco/train2017'
        train_ann_path = data_root+'coco/annotations/instances_train2017.json'
        data_train = datasets.CocoDetection(root = train_path, 
                                            annFile = train_ann_path,
                                            transform = transform)
        
        test_path = data_root+'coco/val2017'
        test_ann_path = data_root+'coco/annotations/instances_val2017.json'
        data_test = data_train = datasets.CocoDetection(root = test_path, 
                                                        annFile = test_ann_path,
                                                        transform = transform)
    
    return data_train, data_val