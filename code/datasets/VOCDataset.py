from __future__ import print_function
import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

class VOCDetection(Dataset):
    '''
    Pascal VOC Detection Dataset
    
    Args:
        root      (string)             : Root directory of VOC Dataset
        year      (string)             : Dataset year, only supports 2007 and 2012
        image_set (string)             : Select image_set to use, ['train', 'val', 'trainval', 'test']
        transform (callable, optional) : A function/transform that takes in an input
            and returns a transformed version. E.g, ``transforms.ToTensor()``
        target_transform (callable, optional) : A function/transform that takes in an output
            and returns a transformed version. E.g, ``transforms.ToTensor()``
    
    Return:
        An array of (image, label) pair.
        image (PIL image/ transfomed) : Dataset image
        label (string / transfomed)    : Annotation informations (xmin, ymin, xmax, ymax, cls_id, difficult)
    '''
    def __init__(self, root, year, image_set, transform=None, target_transform=None):
        
        if year not in ['2007', '2012']:
            raise RuntimeError('Currently only support 2007 and 2012.')
        if image_set not in ['train', 'val', 'trainval', 'test']:
            raise RuntimeError('Image set is not found. It should be train, val, trainval, or test.')
        
        self.year = year
        self.root_dir = root
        self.dataset_dir = os.path.join(self.root_dir, 'VOC'+year)
        self.transform = transform
        self.target_transform = target_transform
        
        classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                   'tvmonitor')
        self.classes = classes
        
        #self.rst_dir = os.path.join(self.root_dir,'results',dataset_name,'Segmentation')
        #self.eval_dir = os.path.join(self.root_dir,'eval_result',dataset_name,'Segmentation')
        self.img_dir = os.path.join(self.dataset_dir, 'JPEGImages')
        self.ann_dir = os.path.join(self.dataset_dir, 'Annotations') # For Detection
        self.seg_dir = os.path.join(self.dataset_dir, 'SegmentationClass') # For Segmentation
        
        self.set_dir = os.path.join(self.dataset_dir, 'ImageSets', 'Main') # For Detection
        
        if not os.path.isdir(self.dataset_dir):
            raise RuntimeError('Dataset is not found or corrupted.')
        
        file_name = self.set_dir+'/'+image_set+'.txt'
        df = pd.read_csv(file_name, names=['filename'])
        self.name_list = df['filename'].values
        
        self.num_class = len(self.classes)
        self.index_map = dict(zip(self.classes, range(self.num_class)))
        
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        img_id = str(self.name_list[idx]).rjust(6, '0') if '2007' == self.year else str(self.name_list[idx])
        img_file = self.img_dir + '/' + img_id + '.jpg'
        
        image = Image.open(img_file).convert('RGB')
        
        ann_file = self.ann_dir + '/' + img_id + '.xml'

        label = self.load_label(idx)
        
        if self.transform is not None:
            image = self.transform(image)
            
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label
    
    def load_label(self, idx):
        '''
        Parse xml file and return important informations (xmin, ymin, xmax, ymax, cls_id, difficult)
        '''
        img_id = str(self.name_list[idx]).rjust(6, '0') if '2007' == self.year else str(self.name_list[idx])
        ann_file = self.ann_dir + '/' + img_id + '.xml'
        root = ET.parse(ann_file).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        label = []
        for obj in root.iter('object'):
            try:
                difficult = int(obj.find('difficult').text)
            except ValueError:
                difficult = 0
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in self.classes:
                continue
            cls_id = self.index_map[cls_name]
            xml_box = obj.find('bndbox')
            xmin = float(xml_box.find('xmin').text)
            ymin = float(xml_box.find('ymin').text)
            xmax = float(xml_box.find('xmax').text)
            ymax = float(xml_box.find('ymax').text)
            
            label.append([xmin, ymin, xmax, ymax, cls_id, difficult])
        return np.array(label)
    
class VOCSegmentation(Dataset):
    '''
    Pascal VOC Segmentation Dataset
    
    Args:
        root      (string)             : Root directory of VOC Dataset
        year      (string)             : Dataset year, only supports 2007 and 2012
        image_set (string)             : Select image_set to use, ['train', 'val', 'trainval', 'test']
        transform (callable, optional) : A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor()``
        target_transform (callable, optional) : A function/transform that takes in an output
            and returns a transformed version. E.g, ``transforms.ToTensor()``
    
    Return:
        An array of (image, mask) pair.
        image (PIL image / transfomed) : Dataset image
        mask  (PIL image / transfomed) : Mask image
    '''
    def __init__(self, root, year, image_set, transform=None, target_transform=None):
        
        if year not in ['2007', '2012']:
            raise RuntimeError('Currently only support 2007 and 2012.')
        if image_set not in ['train', 'val', 'trainval', 'test']:
            raise RuntimeError('Image set is not found. It should be train, val, trainval, or test.')
        
        self.year = year
        self.root_dir = root if root else '/home/space/datasets/VOCdevkit/'
        self.dataset_dir = os.path.join(self.root_dir, 'VOC'+year)
        self.transform = transform
        self.target_transform = target_transform
        
        classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                   'tvmonitor')
        self.classes = classes
        
        #self.rst_dir = os.path.join(self.root_dir,'results',dataset_name,'Segmentation')
        #self.eval_dir = os.path.join(self.root_dir,'eval_result',dataset_name,'Segmentation')
        self.img_dir = os.path.join(self.dataset_dir, 'JPEGImages')
        self.ann_dir = os.path.join(self.dataset_dir, 'Annotations') # For Detection
        self.seg_dir = os.path.join(self.dataset_dir, 'SegmentationClass') # For Segmentation
        
        self.set_dir = os.path.join(self.dataset_dir, 'ImageSets', 'Segmentation') # For Segmentation
        
        if not os.path.isdir(self.dataset_dir):
            raise RuntimeError('Dataset is not found or corrupted.')
        
        file_name = self.set_dir+'/'+image_set+'.txt'
        df = pd.read_csv(file_name, names=['filename'])
        self.name_list = df['filename'].values
        
        self.num_class = len(self.classes)
        self.index_map = dict(zip(self.classes, range(self.num_class)))
        
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        img_id = str(self.name_list[idx]).rjust(6, '0') if '2007' == self.year else str(self.name_list[idx])
        img_file = self.img_dir + '/' + img_id + '.jpg'
        mask_file = self.seg_dir + '/' + img_id +'.png'
        
        image = Image.open(img_file).convert('RGB')
        mask = Image.open(mask_file)
        
        if self.transform is not None:
            image = self.transform(image)
            
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return image, mask    