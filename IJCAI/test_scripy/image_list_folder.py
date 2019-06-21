#coding=utf-8
import os
import csv
import torch
from torchvision.datasets.folder import *
from scipy.misc import imresize
import numpy as np

class ImageListFolder(torch.utils.data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        images = []
        with open(os.path.join(root, 'dev.csv'), 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filepath = os.path.join(root, row['filename'])
                truelabel = int(row['trueLabel'])
                tragetedlabel = int(row['targetedLabel'])
                item = (filepath, truelabel, tragetedlabel)
                images.append(item)

        if len(images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = images
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target, attacked_target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, attacked_target

    def __len__(self):
        return len(self.imgs)

def inception_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as image:
            image = np.asarray(image)
            image = imresize(image, [299, 299]).astype(np.float32)
            image = ( image / 255.0 ) * 2.0 - 1.0
            image = Image.fromarray(np.uint8(image))
    return  image

def vgg_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as image:
            image = np.asarray(image)
            _R_MEAN = 123.68
            _G_MEAN = 116.78
            _B_MEAN = 103.94
            image = imresize(image, [299, 299]).astype(np.float32)
            image[:, :, 0] = image[:, :, 0] - _R_MEAN
            image[:, :, 1] = image[:, :, 1] - _G_MEAN
            image[:, :, 2] = image[:, :, 2] - _B_MEAN
            image = Image.fromarray(np.uint8(image))
    return image