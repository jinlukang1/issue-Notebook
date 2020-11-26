import cv2
import numpy as np
import os

if __name__ == "__main__":
    np_img = cv2.imread('/Users/jinlukang/Desktop/SubstationDataset/00000.jpg')
    # np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    target = np.array([241, 104, 146])[146, 104, 241]
    
    # print(np_img[720][1280])
    # print(np_img[:, :, 2].shape)
    # print(np.where((np_img[:, :, 0]==target[0])&(np_img[:, :, 1]==target[1])))
    print(np.where(np_img[:, :, 0]==target[2]))
    # print(target)
    # print(np_img.shape)
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.imshow('image',np_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()