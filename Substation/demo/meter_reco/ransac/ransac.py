# -*- coding:utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
from PIL import Image
from src.prediction import Prediction
from src.model import Keypoints
from config import NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, IMG_SMALL_HEIGHT, IMG_SMALL_WIDTH, CUDA_DEVICE, ZERO_FULL, PATH, ZERO_FULL_PATH, colors,  CROPPED_LABEL_PATH, CROPPED_RECTIFIED_PATH,NUM_OF_TEST, CROPPED_TEST_PATH, num_of_dial

def getColorSqrCenter(label,color):
    mask = np.equal(label,color)
    mask = np.logical_and(mask[:,:,0],np.logical_and(mask[:,:,1],mask[:,:,2]))
    Y,X = np.nonzero(mask)
    if(X != []):
        x,y = int(np.mean(X)), int(np.mean(Y))
    else:
        x = -1
        y = -1
    return [x,y]

def rectified_judge(label,x,y):
    mask = np.equal(label,[0,0,0])
    mask = np.logical_and(mask[:,:,0],np.logical_and(mask[:,:,1],mask[:,:,2]))
    Y,X = np.nonzero(mask)
    if(X.size > x*y/3):
        return -1
    return 1

transform1 = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

def main():
    if not (os.path.exists(CROPPED_RECTIFIED_PATH)):
        os.makedirs(CROPPED_RECTIFIED_PATH)
    fnames = os.listdir(CROPPED_LABEL_PATH)
    seg_zero = plt.imread(os.path.join(ZERO_FULL_PATH,ZERO_FULL+".png"))
    seg_label = np.uint8(seg_zero[:,:,:3] * 255)
    #get keypoints from zero_full images
    dst = []
    for c in colors["keypoints"]:
        dst.append(getColorSqrCenter(seg_label,c))
    dst = np.array(dst)
    torch.cuda.set_device(CUDA_DEVICE)
    model_pred = Keypoints(NUM_CLASSES)
    model_pred = model_pred.cuda()
    model_pred.load_state_dict(torch.load(os.path.join(PATH,"model_100.pkl")))
    model_pred.eval()
    num_of_correction = 0
    for fname in fnames:
        #print(fname)
        image = cv2.imread(os.path.join(CROPPED_TEST_PATH,fname))
        for i in range(0,9,3):
        #get keypoints from label images
            cropped_im = image[i:IMG_WIDTH-i, i:IMG_HEIGHT-i]
            cropped_im = cv2.resize(cropped_im, (IMG_WIDTH, IMG_HEIGHT),interpolation = cv2.INTER_NEAREST)
            im = transform1(np.array(cropped_im))
            im = im.cuda()
            
            prediction = Prediction(model_pred, NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, IMG_SMALL_HEIGHT, IMG_SMALL_WIDTH)
            result, keypoints = prediction.predict(im)
            keypoints = keypoints.cpu().numpy()
            keypoints = np.array(keypoints)
            
            #transpose and save
            T = cv2.estimateAffine2D(keypoints, dst, ransacReprojThreshold=5)
            im_rect = cv2.warpAffine(image,T[0],(seg_zero.shape[1], seg_zero.shape[0]))
            if(rectified_judge(im_rect,seg_zero.shape[1],seg_zero.shape[0])==1):
                image = im_rect
                num_of_correction = num_of_correction + 1
                break
            
        image = image[:,:, (2, 1, 0)]
        image = Image.fromarray(image.astype(np.uint8))
        image.save(os.path.join(CROPPED_RECTIFIED_PATH,fname))
    
    print("%d dial %d correction images" %(num_of_dial,num_of_correction))

    
if __name__ == '__main__':
    main()