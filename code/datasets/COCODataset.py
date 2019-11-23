from __future__ import print_function
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class COCODetection(Dataset):
    '''
    Coco Detection Dataset
    
    Args:
        root (string)                  : Root directory of Coco Dataset
        image_set (string)             : Select image_set to use, ['train', 'val']
        transform (callable, optional) : A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor()``
        target_transform (callable, optional) : A function/transform that takes in an output
            and returns a transformed version. E.g, ``transforms.ToTensor()``
            
    Return:
        An array of (image, label) pair.
        image (PIL image) : Dataset image
        label (string)    : Annotation informations (xmin, ymin, xmax, ymax, cls_id)
    '''
    def __init__(self, root, image_set, transform=None, target_transform=None):
        if image_set not in ['train', 'val']:
            raise RuntimeError('Image set is not found. It should be train or val.')
        
        annFile = '/home/space/datasets/coco/annotations/instances_{}2017.json'.format(image_set)
        
        self.coco = COCO(annFile)
        
        self.root_dir = root if root else '/home/space/datasets/coco/'
        self.dataset_dir = self.root_dir+image_set+'2017'
        self.transform = transform
        self.target_transform = target_transform
        
        classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
                   'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 
                   'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
                   'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 
                   'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
                   'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
                   'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 
                   'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
                   'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
                   'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 
                   'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
                   'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 
                   'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush']
        self.classes = classes
        
        self.images, self.labels = self.load_json()
        
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
        else:
            label = np.array(label)

        return image, label
    
    def load_json(self):
        items = []
        labels = []
        
        coco = self.coco
        
        img_ids = sorted(coco.getImgIds())
        for entry in coco.loadImgs(img_ids):
            fname = entry['file_name']
            abs_path = os.path.join(self.dataset_dir, fname)
            if not os.path.exists(abs_path):
                raise IOError('Image: {} not exists.'.format(abs_path))
            label = self.load_label(coco, entry)
            if not label:
                continue
            items.append(abs_path)
            labels.append(label)
        return items, labels
        
    
    def load_label(self, coco, entry):
        '''
        Parse json file and return important informations (xmin, ymin, xmax, ymax, cls_id) -> change to dict?
        '''
        entry_id = entry['id']
        ann_ids = coco.getAnnIds(imgIds=entry_id, iscrowd=None)
        objs = coco.loadAnns(ann_ids)
        
        width = entry['width']
        height = entry['height']
        
        label = []
        for obj in objs:
            cls_id = obj['category_id']-1
            bbox = obj['bbox']
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = xmin + np.maximum(0, bbox[2])
            ymax = ymin + np.maximum(0, bbox[3])
            
            label.append([xmin, ymin, xmax, ymax, cls_id])
        return label
    
class COCOSegmentation(Dataset):
    '''
    Coco Segmentation Dataset
    
    Args:
        root (string)                  : Root directory of Coco Dataset
        image_set (string)             : Select image_set to use, ['train', 'val']
        transform (callable, optional) : A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor()``
            
    Return:
        An array of (image, mask) pair.
        image (PIL image) : Dataset image
        mask  (PIL image) : Mask COCO Polygon format
    '''
    def __init__(self, root, image_set, transform=None):
        if image_set not in ['train', 'val']:
            raise RuntimeError('Image set is not found. It should be train or val.')
        
        annFile = '/home/space/datasets/coco/annotations/instances_{}2017.json'.format(image_set)
        
        self.coco = COCO(annFile)
        
        self.root_dir = root if root else '/home/space/datasets/coco/'
        self.dataset_dir = self.root_dir+image_set+'2017'
        self.transform = transform
        
        classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
                   'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 
                   'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
                   'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 
                   'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
                   'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
                   'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 
                   'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
                   'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
                   'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 
                   'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
                   'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 
                   'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush']
        self.classes = classes
        
        self.images, self.labels, self.segms = self.load_json()
        
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        segm = self.segms[idx]
        
        image = Image.open(img_file).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, segm #np.array(label), segm
    
    def load_json(self):
        items = []
        labels = []
        segms = []
        
        coco = self.coco
        
        img_ids = sorted(coco.getImgIds())
        for entry in coco.loadImgs(img_ids):
            fname = entry['file_name']
            abs_path = os.path.join(self.dataset_dir, fname)
            if not os.path.exists(abs_path):
                raise IOError('Image: {} not exists.'.format(abs_path))
            label, segm = self.load_label(coco, entry)
            if not label:
                continue
            items.append(abs_path)
            labels.append(label)
            segms.append(segm)
        return items, labels, segms
        
    
    def load_label(self, coco, entry):
        '''
        Parse json file and return important informations (xmin, ymin, xmax, ymax, cls_id) -> change to dict?
        '''
        entry_id = entry['id']
        ann_ids = coco.getAnnIds(imgIds=entry_id, iscrowd=None)
        objs = coco.loadAnns(ann_ids)
        
        width = entry['width']
        height = entry['height']
        
        label = []
        segm = []
        for obj in objs:
            if obj.get('ignore', 0) == 1:
                continue
            # crowd objs cannot be used for segmentation
            if obj.get('iscrowd', 0) == 1:
                continue
                
            cls_id = obj['category_id']-1
            bbox = obj['bbox']
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = xmin + np.maximum(0, bbox[2])
            ymax = ymin + np.maximum(0, bbox[3])
            
            segs = obj['segmentation']
            label.append([xmin, ymin, xmax, ymax, cls_id])
            segm.append([np.asarray(p).reshape(-1, 2).astype('float32') for p in segs])# if len(p) >= 6])
        return label, segm