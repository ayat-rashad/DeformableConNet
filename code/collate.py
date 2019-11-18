import torch
import numpy as np

def pad_tensor(image, pad):
    '''
    Pad an image with 0s

    Args:
        image - Image tensor to pad
        pad - Padding size

    Return:
        image - Padded image tensor
    '''
    # Tensor [c, h, w] -> dim(0) can be skipped, pad only dim(1) and dim(2)
    pad_size = list(image.shape)

    # Pad dim 1 if it is not equal pad_length
    if(pad_size[1] != pad):
        pad_size[1] = pad - image.size(1)
        image = torch.cat([image, torch.zeros(*pad_size)], dim=1)
        pad_size = list(image.shape)

    if(pad_size[2] != pad):
        pad_size[2] = pad - image.size(2)
        image = torch.cat([image, torch.zeros(*pad_size)], dim=2)
        pad_size = list(image.shape)

    return image

def pad_collate(batch):
    '''
    Pad images in batch correspond to the maximum dim in the batch

    Args:
        batch (List of (tensor, label))

    Return:
        images - A tensor of all padded images in 'batch'
        label - Label
    '''
    # Find longest sequence
    max_len = max(map(lambda b: max(b[0].shape), batch))

    # pad according to max_len
    padded_batch = []
    for b in batch:
        x, y, z = b[0].shape
        padded_tensor = pad_tensor(b[0], pad=max_len)
        padded_batch.append((padded_tensor, b[1]))

    # stack all
    images = torch.stack(list(map(lambda x: x[0], padded_batch)), dim=0)
    label = list(map(lambda x: x[1], padded_batch))
    return images, label