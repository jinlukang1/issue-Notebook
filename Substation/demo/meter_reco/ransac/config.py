# -*- coding:utf-8 -*-
import os
import numpy as np
colors = {}
colors["keypoints"] = np.array([[242,201,197], # center
                               [199,125,143],
                               [0,126,138],
                               [183,140,93],
                               [227,96,50],
                               [59,246,53],
                               [141,238,180],
                               [121,160,208],
                               [205,228,85],
                               [249,173,199],
                               [170,71,248]])
NUM_CLASSES = colors["keypoints"].shape[0]
IMG_HEIGHT  = 288
IMG_WIDTH   = 288
IMG_SMALL_HEIGHT = 96
IMG_SMALL_WIDTH  = 96
CUDA_DEVICE = 1
RADIUS = 20
num_of_dial = 1
NUMS_OF_IMAGE = [40500,30678,28767,39388,28595]
NUMS_OF_TEST = [3745,4572,3322,4901,2691]
ZERO_FULL_S = ["微信图片_20191101171759","微信图片_20191101171743","微信图片_20191101171737","微信图片_20191101171827","微信图片_20191101171822"]
PATHS = ["/data3/home_huxiaoming/biaoji_A","/data3/home_huxiaoming/biaoji_B","/data3/home_huxiaoming/biaoji_C","/data3/home_huxiaoming/biaoji_D","/data3/home_huxiaoming/biaoji_E"]
NUM_OF_IMAGE = NUMS_OF_IMAGE[num_of_dial-1]
NUM_OF_TEST = NUMS_OF_TEST[num_of_dial-1]
ZERO_FULL = ZERO_FULL_S[num_of_dial-1]
PATH = PATHS[num_of_dial-1]
CROP_PATH = os.path.join(PATH,"cropped")
ZERO_FULL_PATH = os.path.join(PATH, "zero_full")
TEST_PATH = os.path.join(CROP_PATH, "image")
LABEL_PATH = os.path.join(CROP_PATH,"label")
TESTPATH = os.path.join(PATH,"test")
CROPPED_TEST_PATH = os.path.join(TESTPATH,"test")
CROPPED_LABEL_PATH = os.path.join(TESTPATH,"label")
CROPPED_RECTIFIED_PATH = os.path.join(TESTPATH,"rectified")
RECTIFIED_PATH = os.path.join(PATH,"rectified")
MOVED_PATH = os.path.join(CROP_PATH,"moved")
RECTIFIED_MOVED_PATH = os.path.join(PATH,"rectified_moved")
CROP_RANGE = 3
epochs = 100
batch_size = 32