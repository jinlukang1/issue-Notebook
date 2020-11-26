import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

dtype = torch.float32

class Prediction:
    def __init__(self, model, num_classes, img_height, img_width, img_small_height, img_small_width):
        self.model = model
        self.num_classes = num_classes
        self.img_height  = img_height
        self.img_width   = img_width
        self.img_small_height = img_small_height
        self.img_small_width  = img_small_width
        
        self.offset_x_ij = torch.arange(0, self.img_small_width) \
            .repeat(self.img_small_height).view(1,1,self.img_small_height, self.img_small_width)
        self.offset_y_ij = torch.arange(0, self.img_small_height) \
            .repeat(self.img_small_width).view(self.img_small_width, self.img_small_height).t().contiguous() \
            .view(1,1,self.img_small_height, self.img_small_width)
        
        
        # self.offset_x_ij = self.offset_x_ij.cuda().type(torch.cuda.FloatTensor)
        # self.offset_y_ij = self.offset_y_ij.cuda().type(torch.cuda.FloatTensor)
        self.offset_x_ij = self.offset_x_ij.to(dtype = dtype)
        self.offset_y_ij = self.offset_y_ij.to(dtype = dtype)
        
        self.offset_x_add = (0 - self.offset_x_ij).view(self.img_small_height, self.img_small_width, 1, 1)
        self.offset_y_add = (0 - self.offset_y_ij).view(self.img_small_height, self.img_small_width, 1, 1)
        
        self.offset_x_ij = (self.offset_x_ij + self.offset_x_add) * self.img_width / self.img_small_width
        self.offset_y_ij = (self.offset_y_ij + self.offset_y_add) * self.img_height/ self.img_small_height
        
    def predict(self, imgs):
        # img: torch.Tensor(3, height, width) 
        if len(imgs.shape) == 4:
            imgs = imgs.view(-1, imgs.shape[1], imgs.shape[2], imgs.shape[3])    
        elif len(imgs.shape) == 3:
            imgs = imgs.view(-1, imgs.shape[0], imgs.shape[1], imgs.shape[2])
        
        device = imgs.device

        result, (maps_pred, offsets_x_pred, offsets_y_pred) = self.model.forward(Variable(imgs))
        maps_pred = maps_pred.data
        offsets_x_pred = offsets_x_pred.data.to(device = device)
        offsets_y_pred = offsets_y_pred.data.to(device = device)
        
        keypoints = torch.zeros(self.num_classes, 2)
        # keypoints = keypoints.type(torch.cuda.LongTensor)
        keypoints = keypoints.to(dtype = torch.long)

        self.offset_x_ij = self.offset_x_ij.to(device = device)
        self.offset_y_ij = self.offset_y_ij.to(device = device)
        for k in range(self.num_classes):
            offsets_x_ij = self.offset_x_ij + offsets_x_pred[0][k]
            offsets_y_ij = self.offset_y_ij + offsets_y_pred[0][k]
            distances_ij = torch.sqrt(offsets_x_ij * offsets_x_ij + offsets_y_ij * offsets_y_ij)

            distances_ij[distances_ij > 1] = 1
            distances_ij = 1 - distances_ij
            score_ij = (distances_ij * maps_pred[0][k]).sum(3).sum(2)

            v1,index_y = score_ij.max(0)
            v2,index_x = v1.max(0)
            
            keypoints[k][1] = index_y[index_x] * self.img_height / self.img_small_height
            # keypoints[k][1] = torch.floor_divide(index_y[index_x] * self.img_height, self.img_small_height)
            keypoints[k][0] = index_x * self.img_width / self.img_small_width
            # keypoints[k][0] = torch.floor_divide(index_x * self.img_width, self.img_small_width)
                
        keypoints = keypoints.view(self.num_classes, 2)
        
        maps_array = result[0]
        offsets_x_array = result[1]
        offsets_y_array = result[2]
        return (maps_array, offsets_x_array, offsets_y_array), keypoints
    