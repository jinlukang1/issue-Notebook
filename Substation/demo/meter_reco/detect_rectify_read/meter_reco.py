# -*- coding: utf8 -*-

# to run this program especially the detection module, it's necessary to change environment to open--mmlab
# use the following command 
# `source activate base`
# `conda activate open-mmlab`

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch, torchvision

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.insert(0, "../mmdetection")
from mmdet.apis import init_detector, inference_detector

sys.path.insert(0,"../ransac")
from src.prediction import Prediction
from src.model import Keypoints

# device = torch.device("cpu")
device = torch.device("cuda")
dtype = torch.float32

import utils
from utils import *

n_meter = 5

H, W = 288, 288
h, w = 96, 96

base_path = '/home/songwenlong/qt_demo/meter_reco'
rectifier_ckpts = ["rectify_model_A.pkl","rectify_model_B.pkl","rectify_model_C.pkl", "rectify_model_D.pkl","rectify_model_E.pkl"]
reader_ckpts = ["models/{}_rect_percent.pth".format(s) for s in "ABCDE"]

log_file = "{} run log.txt".format(os.path.basename(__file__).split('.')[0])

class ModelOrientResNet50(torch.nn.Module):
    '''
    '''
    def __init__(self,meter_type):
        super(ModelOrientResNet50,self).__init__()
        self.meter_type = meter_type
        resnet50 = torchvision.models.resnet50(pretrained=False)
        self.backbone = torch.nn.Sequential(*list(resnet50.children())[:-2])
        self.pool = torch.nn.MaxPool2d(kernel_size = 2)
        self.conv = torch.nn.Conv2d(in_channels = 2048, out_channels = 1024, kernel_size = 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(1024 * 2 * (4 if self.meter_type == 2 else 2),2)
        return
    def forward(self,x):
        conv = self.backbone(x)
        conv = self.relu(conv)
        conv = self.pool(conv)
        conv = self.conv(conv)
        conv = self.relu(conv)
        conv = self.pool(conv)
        #print('conv shape', conv.shape)
        fc = conv.reshape(conv.shape[0],-1)
        #print(fc.szie())
        output = self.fc(fc)
        orient = output / torch.norm(output,dim = 1,keepdim = True)
        return orient
    pass

def detect_meter(im):
    config_path = os.path.join(base_path, "work-dirs5/faster_rcnn_r50_fpn_1x_voc0712.py")
    detector_ckpt = '/data3/jinlukang/best_models_images/Models/detector_model.pth'
    detector = init_detector(config_path, detector_ckpt, device=device)
    meter_type = -1
    k = 0
    while meter_type < 0:  # may not detect anyone of the meters in the first inference
        detect_result = inference_detector(detector, im)
        highest_conf = 0
        for i in range(n_meter):
            if detect_result[i].size and detect_result[i][0, 4] > highest_conf:
                meter_type = i
                highest_conf = detect_result[i][0, 4]
                pass
            pass
        k += 1
        if k > 5:
            return im, meter_type, None
            pass
        pass
        pass
    bndbox = detect_result[meter_type][0]
    meter = im[np.int(bndbox[1]):np.int(bndbox[3]),
               np.int(bndbox[0]):np.int(bndbox[2])]
    im_anno = im.copy()
    cv2.rectangle(im_anno,(bndbox[0],bndbox[1]),(bndbox[2],bndbox[3]),(0,255,0),2)
    cv2.putText(im_anno, text = "{}".format("ABCDE"[meter_type]),
                org = (bndbox[0], np.int(bndbox[1]) - 10), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 1, color = (0, 255, 0), thickness = 2)
    return im_anno, meter, meter_type

def rectify_meter(meter, meter_type, H = 288, W = 288):
    coord_keypoints1 = np.array(coord_keypoints[meter_type])
    n_keypoints = coord_keypoints1.shape[0]
    #print(n_keypoints)
    rectifier = Keypoints(n_keypoints)
    #rectifier.load_state_dict(torch.load('rect_model_{}.pth'.format(
        #"ABCDE"[meter_type]), map_location=gpu_device))
    rectifier.load_state_dict(torch.load(os.path.join(base_path, "ransac", rectifier_ckpts[meter_type]), map_location=device))
    rectifier = rectifier.to(device=device, dtype=dtype)
    rectifier.eval()
    prediction = Prediction(rectifier, n_keypoints, H, W, h, w)
    for i in range(0, 9, 3):
        meter1 = cv2.resize(
            meter[i:meter.shape[0] - i, i:meter.shape[1] - i], (W, H))
        if meter_type > 0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            x = torch.from_numpy(np.expand_dims((meter1 - mean) / std, axis=0).transpose([0, 3, 1, 2])).to(
                device=device, dtype=dtype)
        else:
            x = torch.from_numpy(np.expand_dims(meter1, axis=0).transpose(
                [0, 3, 1, 2])).to(device=device, dtype=dtype)
        with torch.no_grad():
            result, keypoints = prediction.predict(x)
            pass
        keypoints = keypoints.cpu().numpy()
        standard_meter_seg_name = "/home/songwenlong/qt_demo/meter_reco/ransac/src/biaoji_{}_zero_full.png".format("ABCDE"[
                                                                     meter_type])
        seg_zero = plt.imread(standard_meter_seg_name)
        seg_label = np.uint8(seg_zero[:, :, :3] * 255)
        dst = []
        if meter_type == 2:
            for c in colors["keypoints"][1:]:
                dst.append(getColorSqrCenter(
                    seg_label, c, np.ones(seg_label.shape[:2])))
        else:
            for c in colors["keypoints"]:
                dst.append(getColorSqrCenter(
                    seg_label, c, np.ones(seg_label.shape[:2])))
        dst = np.array(dst)
        T, _ = cv2.estimateAffine2D(keypoints, dst, ransacReprojThreshold=5)
        meter_rect = cv2.warpAffine(
            meter1, T, (seg_zero.shape[1], seg_zero.shape[0]))
        #         num_of_correction = 0
        if rectified_judge(meter_rect, 495, 497) == 1:
            break
            pass
    if meter_type == 2:
        d = np.minimum(meter_rect.shape[0], meter_rect.shape[1] // 2)
        meter_rect = meter_rect[np.int(meter_rect.shape[0] / 2 - d / 2):np.int(meter_rect.shape[0] / 2 + d / 2),
                                np.int(meter_rect.shape[1] / 2 - d):np.int(meter_rect.shape[1] / 2 + d)]
        pass
    else:
        d = np.minimum(meter_rect.shape[0], meter_rect.shape[1])
        meter_rect = meter_rect[np.int(meter_rect.shape[0] / 2 - d / 2):np.int(meter_rect.shape[0] / 2 + d / 2),
                                np.int(meter_rect.shape[1] / 2 - d / 2):np.int(meter_rect.shape[1] / 2 + d / 2)]
        pass
    if meter_type == 2:
        meter_rect = cv2.resize(meter_rect, (H*2, W))
    else:
        meter_rect = cv2.resize(meter_rect, (H, W))
    return meter_rect

def rectified_judge(label, x, y):
    mask = np.equal(label, [0, 0, 0])
    mask = np.logical_and(mask[:, :, 0], np.logical_and(
        mask[:, :, 1], mask[:, :, 2]))
    Y, X = np.nonzero(mask)
    if X.size > x * y / 3:
        return -1
    return 1

def read_meter(meter_rect, meter_type):
    inds = [0, 0, 0, 0, 0]
    if meter_type == 2:
        meter_rect = cv2.resize(meter_rect, (512, 256))
    else:
        meter_rect = cv2.resize(meter_rect, (256, 256))
    reader = ModelOrientResNet50(meter_type)
    reader.load_state_dict(torch.load(os.path.join(base_path, "detect_rectify_read", "models/RectMeterOrientRegressResNet50_{}_{}_v5.pth".format("ABCDE"[meter_type], inds[meter_type]))))
    reader = reader.to(device=device, dtype=dtype)
    reader.eval()
    # meter_rect = cv2.resize(meter_rect, (512, 256))
    #print(meter_rect.shape, meter_rect.max())
    x = torch.from_numpy(np.expand_dims(meter_rect, axis=0).transpose(
        [0, 3, 1, 2])).to(device=device, dtype=dtype)
    with torch.no_grad():
        orient = reader(x).cpu().numpy()
        #print(orient)
        pass
    return orient_to_reading(orient, meter_type)

def detect_rectify_read_an_image(im):
    # detect the meter
    im_anno, meter, meter_type = detect_meter(im)
    if meter.max() > 1:
        meter = meter / 255
        pass

    # rectify the meter
    meter_rect = rectify_meter(meter,meter_type)

    # read the meter
    reading_pred = read_meter(meter_rect, meter_type)
    reading_pred = float(reading_pred.tolist()[0])
    
    print('meter reco done!')
    meter_rect = cv2.resize(meter_rect, (H, W))
    return im_anno, meter, meter_type, meter_rect, reading_pred

def main():
    img_path = '/home/songwenlong/qt_demo/meter_test_folder/100035.jpg'
    im = plt.imread(img_path)[:, :, :3]
    im_anno, meter, meter_type, meter_rect, reading_pred = detect_rectify_read_an_image(im)
    print(meter_type, reading_pred)
    return

if __name__ == '__main__':
    main()
