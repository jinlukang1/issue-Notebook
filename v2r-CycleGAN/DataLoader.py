import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2

transform2tensor = transforms.Compose([
    transforms.ToTensor()
    ])

transform2pil = transforms.Compose([
    transforms.ToPILImage()
    ])

class License_Real(Dataset):
    def __init__(self, datapath):
        self.datapath = datapath
        print('loading data from {}'.format(self.datapath))
        self.img_list = np.load(self.datapath)
        self.size = self.img_list.shape[0]
        print('Done!')

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = self.img_list[idx, :, :, :]
        img = cv2.resize(img,(160,56),interpolation=cv2.INTER_CUBIC)
        return transform2tensor(img)#.transpose(2, 0, 1)

class License_Virtual(Dataset):
    def __init__(self, datapath):
        self.datapath = datapath
        print('loading data from {}'.format(self.datapath))
        self.img_list = np.load(self.datapath)
        self.size = self.img_list.shape[0]
        print('Done!')

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = self.img_list[idx, :, :, :]
        img = cv2.resize(img,(160,56),interpolation=cv2.INTER_CUBIC)
        return transform2tensor(img)#.transpose(2, 0, 1)

if __name__ == '__main__':
    RealDataset = License_Real('/gdata/jinlk/zhangyesheng/LPR/lab_data/real/train_im.npy')
    VirtualDatastet = License_Virtual('/gdata/jinlk/jinlukang/vLPR/npy_data/all_car_recorder_train_im.npy')
    print('RealDataset:{}, VirtualDatastet:{}'.format(len(RealDataset), len(VirtualDatastet)))