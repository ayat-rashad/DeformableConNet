from __future__ import print_function
import os
import os.path
import tarfile
from zipfile import ZipFile
from urllib.request import urlretrieve

data_root = "/home/space/datasets/"

def download_and_unpack(url, filename):
    if 'coco' in url:
        download_path = data_root+'coco/'
    else: download_path = data_root
    
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
        directory = '/home/space/datasets/VOCdevkit/VOC2007'
        if not os.path.exists(directory):
            download_and_unpack('http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar', 'VOCtrainval_06-Nov-2007.tar')
            download_and_unpack('http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar', 'VOCtest_06-Nov-2007.tar')
        else:
            print(directory + 'exists')
    elif(dataset == 'voc2012'):
        directory = '/home/space/datasets/VOCdevkit/VOC2012'
        if not os.path.exists(directory):
            download_and_unpack('http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar', 'VOCtrainval_11-May-2012.tar')
            download_and_unpack('http://pjreddie.com/media/files/VOC2012test.tar', 'VOCtest_11-May-2012.tar')
        else:
            print(directory + 'exists')
    elif(dataset == 'coco'):    
        directory = '/home/space/datasets/coco/val2017'
        if not os.path.exists(directory):
            download_and_unpack('http://images.cocodataset.org/zips/val2017.zip', 'val2017.zip')
        else:
            print(directory + ' exists')

        directory = '/home/space/datasets/coco/train2017'
        if not os.path.exists(directory):
            download_and_unpack('http://images.cocodataset.org/zips/train2017.zip', 'train2017.zip')
        else:
            print(directory + 'exists')

        directory = '/home/space/datasets/coco/annotations'
        if not os.path.exists(directory):
            download_and_unpack('http://images.cocodataset.org/annotations/annotations_trainval2017.zip', 
                        'annotations_trainval2017.zip')
        else:
            print(directory + 'exists')

        ann_path = '/home/space/datasets/coco/annotations/'
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