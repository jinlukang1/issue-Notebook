import csv
import torch
import glob
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets.folder import *
import os, sys
from PIL import Image
import numpy as np
from scipy.misc import imresize

# sys.path.insert(0, '/ghome/jinlk/lib')

def preprocessor(im, model_name):
    """
    :param im:  the raw image array [height, width, channels]
    :param model_name: the network model name 
    :return: 
    """
    if 'inception' in model_name.lower():
        image = imresize(im, [224, 224]).astype(np.float32)
        image = ( image / 255.0 ) * 2.0 - 1.0
        return  image
    if 'resnet' in model_name.lower() or 'vgg' in model_name.lower():
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        image = imresize(im, [224, 224]).astype(np.float32)
        image[:, :, 0] = image[:, :, 0] - _R_MEAN
        image[:, :, 1] = image[:, :, 1] - _G_MEAN
        image[:, :, 2] = image[:, :, 2] - _B_MEAN
        return image

class Tianchi_Dataset(Dataset):
    """docstring for Tianchi_Dataset"""
    def __init__(self, datapath):
        super(Tianchi_Dataset, self).__init__()
        self.datapath = datapath
        self.all_imgs = glob.glob(self.datapath+'/*/*.jpg')
        # print(self.all_imgs)
        self.mean_arr = [0.5, 0.5, 0.5]
        self.stddev_arr = [0.5, 0.5, 0.5]
        self.normalize = transforms.Normalize(mean=self.mean_arr,
                                 std=self.stddev_arr)

        self.model_dimension = 224
        self.center_crop = 224
        self.data_transform = transforms.Compose([
            # transforms.Resize(self.model_dimension),
            # transforms.CenterCrop(self.center_crop),
            transforms.ToTensor(),
            self.normalize,
        ])

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, index):
        img_path = self.all_imgs[index]
        img = Image.open(img_path).convert('RGB')
        # img = np.asarray(img)
        # print(img.size)

        _dir, _img_name = os.path.split(img_path)
        label = int(_dir[-3:])

        inception_img = preprocessor(img, 'inception')
        vgg_img = preprocessor(img, 'vgg')

        inception_data = self.data_transform(inception_img)
        vgg_data = self.data_transform(vgg_img)

        return inception_data, vgg_data, label

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
        print(root)
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


if __name__ == '__main__':

    train_data = Tianchi_Dataset(datapath = '/gdata/jinlk/jinlukang/example/tianchi/IJCAI_2019_AAAC_train')
    train_dataloader = DataLoader(dataset = train_data, num_workers = 4, batch_size = 16, shuffle = True)

    for itr, (inception_data, vgg_data, label) in enumerate(train_dataloader):
        print("{} {}".format(vgg_data.shape, label.shape))
