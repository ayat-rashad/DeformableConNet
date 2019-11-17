from __future__ import print_function
import matplotlib.pyplot as plt
import random
import numpy as np

def show(image, outfile=None):
    '''
    Show image from tensor format [c, h, w]
    Args:
        image (Tensor)
        outfile (string) : Path to the output file
        
    Return:
        ax (Matplotlib Axes)
    '''
    img = image.permute(1,2,0)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    ax.imshow(img)
    
    if outfile != None: 
        ax.savefig(outfile)
        
    return ax

def showBbox(image, label, outfile=None, class_names=None):
    '''
    Show image with its bounding boxes
    
    Args:
        image (Tensor)
        label (string) : image label
        outfile (string) : Path to the output file
        class_names (array) : Array of class names
    
    Return:
        ax (Matplotlib Axes)
    '''
    bboxes = label[:, :4]
    class_ids = label[:, 4:5]
    ax = show(image)
    
    # Get bounding box colors
    cmap = plt.get_cmap('tab20')
    colors = dict()
    
    for i, bbox in enumerate(bboxes):
        class_id = int(class_ids[i])
        if class_id not in colors:
            colors[class_id] = cmap(random.random())
        xmin, ymin, xmax, ymax = [int(x) for x in bbox] 
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                            fill=False,
                            edgecolor=colors[class_id],
                            linewidth=3.5)
        ax.add_patch(rect)

        class_name = class_names[class_id] if class_names is not None else str(cls_id)
        # might need to add logic for score later
        score = ''

        ax.text(xmin, ymin, 
                s='{:s} {:s}'.format(class_name, score), 
                color='black',
                fontsize=11,
                verticalalignment='top', 
                bbox={'color': colors[class_id], 'pad': 0, 'alpha':0.5})
    
    if outfile != None: 
        ax.savefig(outfile)
        
    return ax   