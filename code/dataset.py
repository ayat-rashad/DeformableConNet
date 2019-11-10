from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import os
import os.path
import tarfile
from zipfile import ZipFile
from urllib.request import urlretrieve

import torch
from torchvision import datasets, transforms, models
data_root = "../data/"

def download_and_unpack(url, filename):
    if 'coco' in url:
        download_path = data_root+'coco/'
    else: download_path = data_root'
    
    print(' Downloading '+filename)
    if not os.path.exists(download_path+filename):
        urlretrieve(url, download_path+filename)
    else:
        print(filename + ' is already downloaded')
        
    print(' Unpacking '+filename)
    try:
        if('.tar' in filename):
            with tarfile.open(download_path+filename) as tar:
                tar.extractall(path=download_path)
        elif('.zip' in filename):
            with ZipFile(download_path+filename, 'r') as z:  
                z.extractall(path=download_path)
        else:
            print(' Unknown file type on '+filename)
    finally:
        os.remove(download_path+filename)
        print(' Done')
        
def download_dataset(dataset):
    print('-- Downloading dataset {}. It might take a while --'.format(dataset))
    if(dataset == 'voc2007'):
        directory = './data/VOCdevkit/VOC2007'
        if not os.path.exists(directory):
            download_and_unpack('http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar', 'VOCtrainval_06-Nov-2007.tar')
            download_and_unpack('http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar', 'VOCtest_06-Nov-2007.tar')
        else:
            print(directory + 'exists')
    elif(dataset == 'voc2012'):
        directory = './data/VOCdevkit/VOC2012'
        if not os.path.exists(directory):
            download_and_unpack('http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar', 'VOCtrainval_11-May-2012.tar')
            download_and_unpack('http://pjreddie.com/media/files/VOC2012test.tar', 'VOCtest_11-May-2012.tar')
        else:
            print(directory + 'exists')
    elif(dataset == 'coco'):    
        directory = './data/coco/val2017'
        if not os.path.exists(directory):
            download_and_unpack('http://images.cocodataset.org/zips/val2017.zip', 'val2017.zip')
        else:
            print(directory + ' exists')

        directory = './data/coco/train2017'
        if not os.path.exists(directory):
            download_and_unpack('http://images.cocodataset.org/zips/train2017.zip', 'train2017.zip')
        else:
            print(directory + 'exists')

        directory = './data/coco/annotations'
        if not os.path.exists(directory):
            download_and_unpack('http://images.cocodataset.org/annotations/annotations_trainval2017.zip', 
                        'annotations_trainval2017.zip')
        else:
            print(directory + 'exists')

        ann_path = './data/coco/annotations/'
        filename = 'coco-labels-paper.txt'
        print('Downloading '+filename)
        if not os.path.exists(ann_path+filename):
            urlretrieve('https://raw.githubusercontent.com/amikelive/coco-labels/master/'+filename, 
                        ann_path+filename)
            print('Done')
        else:
            print(filename+' is already downloaded')
    elif(dataset == 'all'):
        download_dataset('voc2007')
        download_dataset('voc2012')
        download_dataset('coco')
    else: assert False, "Dataset is not recognize. It is neiter coco, voc2007, nor voc2012"
    
    print('-- Finish downloading dataset {}. --'.format(dataset))
        
def load_dataset(dataset):
    data_path = data_root
    ann_path = data_root
    
    if 'coco' in dataset:
        train_path = data_root+'coco/train2017'
        train_ann_path = data_root+'coco/annotations/instances_train2017.json'
        data_train = datasets.CocoDetection(root = train_path, annFile = train_ann_path)
        
        test_path = data_root+'coco/val2017'
        test_ann_path = data_root+'coco/annotations/instances_val2017.json'
        data_test = data_train = datasets.CocoDetection(root = test_path, annFile = test_ann_path)
    elif 'voc' in dataset:
        year = dataset[3:]
        assert year in ['2007', '2012'] , 'Only support voc2007 and voc2012'
        
        data_train = datasets.VOCDetection(root = data_root, year=year, image_set='train')
        data_test = datasets.VOCDetection(root = data_root, year=year, image_set='val')
    else:
        assert False, "Dataset is not recognize. It is neiter coco, voc2007, nor voc2012"
    
    return data_train, data_test

def preview_data(dataset):
    n = np.random.randint(len(dataset))
    image, label = dataset[n]
    ds = 'coco' if 'annotation' not in label else 'voc'
        
    # Get bounding boxes and their categories
    if ds == 'coco':
        with open(data_root+'coco/annotations/coco-labels-paper.txt', 'r') as f:
            coco_labels = f.read().splitlines()
        bboxes = np.zeros((len(label), 4))
        cats = np.zeros((len(label), 1), dtype='U10')
        
        for i, item in enumerate(label):
            bboxes[i] = label[i]['bbox']
        for i, item in enumerate(label):
            cats[i] = coco_labels[int(label[i]['category_id'])-1]
        
    else:
        bboxes = np.zeros((len(label['annotation']['object']), 4))
        cats = np.empty((len(label['annotation']['object']), 1), dtype='U10')
        
        if type(label['annotation']['object']) is list:
            for i, el in enumerate(label['annotation']['object']):
                cats[i] = el['name']
                bboxes[i] = list(el['bndbox'].values())
        else:
            cats[0] = label['annotation']['object']['name']
            bboxes[0] = list(label['annotation']['object']['bndbox'].values())
            
    # Get bounding box colors
    num_class = len(cats)
    colors = np.random.rand(num_class)
    cmap = plt.get_cmap('tab20')
    bbox_colors = [cmap(i) for i in colors]
    color = {}
    
    img = np.array(image)
    plt.figure()
    fig, ax = plt.subplots(1, figsize=(12,9))
    ax.imshow(img)

    for bbox, cat in zip(bboxes, cats):
        for el in bbox_colors: 
            if color.get(cat.item()) == None:
                color[cat.item()] = bbox_colors[0]
                bbox_colors.remove(bbox_colors[0])
        c = color.get(cat.item())
        
        x0 = bbox[0]
        y0 = bbox[1]
        x1 = bbox[2] if ds == 'coco' else bbox[2]-bbox[0]
        y1 = bbox[3] if ds == 'coco' else bbox[3]-bbox[1]

        box = patches.Rectangle((x0, y0), x1, y1, linewidth=4, edgecolor=c, facecolor='none')
        ax.add_patch(box)
        plt.text(x0, y0, s=cat.item(), 
                 color='black', verticalalignment='top', bbox={'color': c, 'pad': 0})

    plt.axis('off')
    plt.show()