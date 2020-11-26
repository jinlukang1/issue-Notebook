import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from config import LABEL_PATH, TEST_PATH, colors, IMG_HEIGHT, IMG_WIDTH

transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

def getColorSqrCenter(label, color):
    mask = np.equal(label,color)
    mask = np.logical_and(mask[:,:,0],np.logical_and(mask[:,:,1],mask[:,:,2]))
    Y,X = np.nonzero(mask)
    x,y = int(np.mean(X)), int(np.mean(Y))
    return x,y

class KeypointsDataset(Dataset):
    def __init__(self, image_folder, label_folder, num_classes, img_height, img_width, radius, transform):
        self.num_classes = num_classes
        self.img_height = img_height
        self.img_width = img_width
        self.radius = radius         
        self.transform = transform

        self.imgs = image_folder
        self.labels = label_folder
        
        self.map_value = np.array([[np.linalg.norm([self.img_width - _x, self.img_height - _y]) 
                          for _x in range(img_width * 2)] for _y in range(img_height * 2)])
        
        self.offsets_x_value = np.array([[self.img_width - _x for _x in range(self.img_width * 2)] 
                                         for _y in range(self.img_height * 2)])
        self.offsets_y_value = np.array([[self.img_height - _y for _x in range(self.img_width * 2)] 
                                         for _y in range(self.img_height * 2)])
    
    
    def __getitem__(self, index):  
        '''
        img = np.array(Image.open(os.path.join(TEST_PATH,self.imgs[index])))
        noise = np.random.randint(low=-5,high=5,size=(IMG_HEIGHT,IMG_WIDTH,3))
        for i in range(IMG_HEIGHT):
            for j in range(IMG_HEIGHT):
                for k in range(3):
                    img[i][j][k] += noise[i][j][k]
                    
                    if(img[i][j][k]<0):
                        img[i][j][k]=0
                    if(img[i][j][k]>255):
                        img[i][j][k]=255
                    
        img *= (img > 0)
        img = img * (img <= 255) + 255 * (img > 255)
        img = img.astype(np.uint8)
        img=Image.fromarray(img)
        '''
        img = self.transform(Image.open(os.path.join(TEST_PATH,self.imgs[index])))
        keypoints = np.zeros((self.num_classes, 2))
        maps = np.zeros((self.num_classes, self.img_height, self.img_width), dtype='float32')
        offsets_x = np.zeros((self.num_classes, self.img_height, self.img_width), dtype='float32')
        offsets_y = np.zeros((self.num_classes, self.img_height, self.img_width), dtype='float32')
        for i in range(0, self.num_classes):
            gt = np.uint8(plt.imread(os.path.join(LABEL_PATH,self.labels[index])) [:,:,:3] * 255)
            x,y = (getColorSqrCenter(gt,colors["keypoints"][i]))
            keypoints[i][0] = x
            keypoints[i][1] = y

            if x == 0 and y == 0:
                maps[i] = np.zeros((self.img_height, self.img_width))
                continue
            if self.img_height - y < 0 or self.img_width - x < 0:
                continue          
            maps[i] = self.map_value[self.img_height - int(y) : self.img_height * 2 - int(y), 
                                     self.img_width  - int(x) : self.img_width * 2  - int(x)]       
            maps[i][maps[i] <= self.radius] = 1
            maps[i][maps[i] >  self.radius] = 0
            offsets_x[i] = self.offsets_x_value[self.img_height - y : self.img_height * 2 - y, self.img_width - x : self.img_width * 2 - x]
            offsets_y[i] = self.offsets_y_value[self.img_height - y : self.img_height * 2 - y, self.img_width - x : self.img_width * 2 - x]
        return img, (maps, offsets_x, offsets_y), (keypoints)
    
    def __len__(self):
        return len(self.labels)
