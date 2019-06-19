import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from datasets.transform import *


class CityscapesDataset(Dataset):

    def __init__(self, cfg, period):
        self.dataset_dir = os.path.join(cfg.ROOT_DIR, 'data', 'Cityscapes')
        self.period = period
        self.list = os.path.join(self.dataset_dir, 'list', period + '.txt')
        self.im_names = []
        self.gt_names = []
        with open(self.list) as f:
            line = f.readline()
            while(line):
                im_name, gt_name = line.split()
                self.im_names.append(im_name)
                self.gt_names.append(gt_name)
                line = f.readline()

        self.rescale = None
        self.randomcrop = None
        self.randomscalecrop = None
        self.randomflip = None
        self.totensor = ToTensor()
        self.cfg = cfg

        if cfg.DATA_RESCALE:
            self.rescale = Rescale(cfg.DATA_RESCALE, fix=False)
        if self.period == 'train':
            if cfg.DATA_RANDOMCROP > 0:
                self.randomcrop = RandomCrop(cfg.DATA_RANDOMCROP)
            if cfg.DATA_RANDOMSCALECROP > 0:
                self.randomscalecrop = RandomScaleCrop(cfg.DATA_RANDOMSCALECROP)
            if cfg.DATA_RANDOMFLIP > 0:
                self.randomflip = RandomFlip(cfg.DATA_RANDOMFLIP)

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, idx):
        im_name = self.im_names[idx]
        gt_name = self.gt_names[idx]
        im = cv2.imread(os.path.join(self.dataset_dir, im_name))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(os.path.join(self.dataset_dir, gt_name), 0)

        sample = {'name': im_name, 'image': im, 'segmentation': gt}

        if self.period == 'train':
            if self.cfg.DATA_RANDOMFLIP > 0:
                sample = self.randomflip(sample)
            if self.cfg.DATA_RANDOMCROP > 0:
                sample = self.randomcrop(sample)
            if self.cfg.DATA_RANDOMSCALECROP > 0:
                sample = self.randomscalecrop(sample)
            if self.cfg.DATA_RESCALE:
                sample = self.rescale(sample)
        sample = self.totensor(sample)

        return sample
