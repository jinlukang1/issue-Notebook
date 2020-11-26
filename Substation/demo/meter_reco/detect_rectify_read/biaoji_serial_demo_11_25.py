# Author: Vincent Zhang
# Mail: zhyx12@gmail.com
# ----------------------------------------------
from utils import percent_to_reading
import pickle
from utils import *
import xml.etree.ElementTree
import os
import torch, torchvision
import cv2
import numpy as np
#from src.model import Keypoints
#from src.prediction import Prediction
#from mmdet.apis import init_detector, inference_detector
import sys

path = "../demo/mmdetection"
sys.path.insert(0, path)
from mmdet.apis import init_detector, inference_detector

sys.path.insert(0, "../demo/ransac")
from src.model import Keypoints
from src.prediction import Prediction


# device_num = 3
# os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
gpu_device = 'cuda:{}'.format(0)


# meter_type = 3
# video_name = 'synthetic_2'
#
# meter_type = 1
# video_name = 'synthetic_3'
#



device = torch.device("cuda")
dtype = torch.float32

H, W = 288, 288
h, w = 96, 96

#config_path = './faster_rcnn_r50_fpn_1x.py'
#detector_ckpt = './epoch_3.pth'
#Biaoji_Video_demo/demo/detect_mixedmodel/faster_rcnn_r50_fpn_1x_voc0712.py
#config_path = '/data3/home_jinlukang/songwenlong/mmdetection0/Models-virtualdata/faster_rcnn_r50_fpn_1x_voc0712.py'
#detector_ckpt = '/data3/home_jinlukang/songwenlong/mmdetection0/Models-virtualdata/epoch_2.pth'




log_file = "{} log6.txt".format(os.path.basename(__file__).split('.')[0])

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


#reader = ModelOrientResNet50(meter_type)


#inds = [80, 116, 73, 106, 77]
#reader.load_state_dict(torch.load("/data3/home_jinlukang/pengkunfu/reading/models/RectMeterOrientRegressResNet50_{}_{}_v5.pth".format("ABCDE"[meter_type], inds[meter_type])))
# inds = [87, 107, 119, 111, 119]
# reader.load_state_dict(torch.load("/data3/home_jinlukang/pengkunfu/reading/models/RectMeterOrientRegressResNet50_{}_{}_virtual_real_v5.pth".format("ABCDE"[meter_type], inds[meter_type])))



def detect_meter(im):
    config_path = '/data3/home_jinlukang/songwenlong/mmdetection0/Models-quarter-finetuning/faster_rcnn_r50_fpn_1x_voc0712.py'
    detector_ckpt = '/data3/home_jinlukang/songwenlong/Project_biaoji_2020/best_models_images/Models/detector_model.pth'
    detector = init_detector(config_path, detector_ckpt, device=gpu_device)
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
    return meter, meter_type, bndbox


def rectified_judge(label, x, y):
    mask = np.equal(label, [0, 0, 0])
    mask = np.logical_and(mask[:, :, 0], np.logical_and(
        mask[:, :, 1], mask[:, :, 2]))
    Y, X = np.nonzero(mask)
    if X.size > x * y / 3:
        return -1
    return 1


def rectify_meter(meter, meter_type, H = 288, W = 288):
    coord_keypoints1 = np.array(coord_keypoints[meter_type])
    n_keypoints = coord_keypoints1.shape[0]
    #print(n_keypoints)
    rectifier = Keypoints(n_keypoints)
    #rectifier.load_state_dict(torch.load('rect_model_{}.pth'.format(
        #"ABCDE"[meter_type]), map_location=gpu_device))
    rectifier.load_state_dict(torch.load('/data3/home_jinlukang/songwenlong/Project_biaoji_2020/best_models_images/Models/rectify_model_{}.pkl'.format(
        "ABCDE"[meter_type]), map_location=gpu_device))
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
        standard_meter_seg_name = "../demo/ransac/biaoji_{}_zero_full.png".format("ABCDE"[
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
        meter_rect = cv2.resize(meter_rect, (H *2, W ))
    else:
        meter_rect = cv2.resize(meter_rect, (H, W))
    return meter_rect


def read_meter(meter_rect, meter_type):
    #indx2 = int(im_name.split('.')[0])
    #Img_Name = "/data3/home_jinlukang/songwenlong/data/demodata" 
    #cv2.imwrite(Img_Name  , meter_rect*255)
    # 
    #plt.imsave("test.jpg", meter_rect)
    if meter_type == 2:
        meter_rect = cv2.resize(meter_rect, (512, 256))
    else:
        meter_rect = cv2.resize(meter_rect, (256, 256))
    reader = ModelOrientResNet50(meter_type)
    reader.load_state_dict(torch.load("/data3/home_jinlukang/songwenlong/Project_biaoji_2020/best_models_images/Models/reading_model_{}.pth".format("ABCDE"[meter_type])))
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


def test_manual():
    scale_intervals = np.array([0.01,0.2,0.1,0.02,0.1])
    detect_acc = 0
    test_acc = np.zeros(2)
    meter_path = "/data3/home_jinlukang/data/待测表计数据/真实数据/demo_testdata/biaoji_{}".format("ABCDE"[meter_type])
    gt_readings = {}
    with open(os.path.join(meter_path, "{}-test-groundtruth.txt".format("ABCDE"[meter_type]))) as f:
        for line in f:
            im_name, s  = line.strip().split('\t')
            gt_readings[im_name] = float(s)
            pass
        pass
    for im_name in os.listdir(os.path.join(meter_path, "images")):
        #indx2 = int(im_name.split('.')[0])
        im_path = os.path.join(meter_path, "images", im_name)
        im = plt.imread(im_path)[:, :, :3]
        if im.max() <= 1:
            im = im * 255
            pass
        #print('img read')
        meter, current_meter_type, bndbox = detect_meter(im)
        # print('bbox {}'.format(bndbox))
        mean_val = np.array([0.406, 0.456, 0.485])
        std_val = np.array([0.225, 0.229, 0.224])
        if bndbox is not None:
            meter = meter / 255.0
            # meter = (meter - mean_val) / std_val
            # print('max val in meter {}'.format(meter.max()))
            # print("type {}".format(meter_type))
            meter_rect = rectify_meter(meter, meter_type)
            #print(meter_rect.shape)

           
            #Img_Name = "/data3/home_jinlukang/songwenlong/data/demodata" + "{}".format(range(1000)) + '.jpg'
            #cv2.imwrite(Img_Name  , meter_rect)

            # print('max val in meter {}'.format(meter_rect.max()))
            # meter_rect = (meter_rect * std_val + mean_val)
            #print(meter_rect.shape)
            reading = read_meter(meter_rect, meter_type)[0]
            pass
        reading_gt = percent_to_reading(gt_readings[im_name], meter_type)

        if ('%.3f' %reading_gt) == ('%.3f' %reading):
            print("{}: pred class = {}, "
                "gt reading = {:.3f}, "
                "pred reading = {:.3f}".format(im_name, 
                                                "ABCDE"[current_meter_type], 
                                                reading_gt, reading))
        with open(log_file,"a") as f:
            f.write("{}: pred class = {}, "
                    "gt reading = {:.3f}, "
                    "pred reading = {:.3f}\n".format(im_name, 
                                                    "ABCDE"[current_meter_type], 
                                                    reading_gt, reading))
        err = np.abs(reading_gt - reading)
        test_acc[0] += err < scale_intervals[meter_type]
        test_acc[1] += err < 2 *scale_intervals[meter_type]
        #return
        pass
    test_acc /= len(gt_readings)
    with open(log_file,'a') as f:
        f.write("reading acc: {}\n".format(test_acc))
    print("reading acc: {}", test_acc)
    return

def test_virtual():
    scale_intervals = np.array([0.01,0.2,0.1,0.02,0.1])
    data_path = "/data3/home_jinlukang/data/biaoji_data_v5"
    gt_percent_path = "/data3/home_jinlukang/pengkunfu/rectified meters v5"
    gt_percent_fname = os.path.join(gt_percent_path, "biaoji_{}".format("ABCDE"[meter_type]), "groundtruth percents.txt")
    gt_percents = {}
    with open(gt_percent_fname) as f:
        for line in f:
            fname, s = line.strip().split('\t')
            #if int(fname[-9:-4]) > 1500:
            if int(fname[-9:-4]) > 1500:
                gt_percents[fname] = float(s)
                pass
            pass
        pass
    
    test_acc = np.zeros(2)
    for fname in gt_percents:
        weather, index = fname.split('_')
        im_path = os.path.join(data_path, "biaoji_{}".format("ABCDE"[meter_type]), weather, "ori", index)
        im = plt.imread(im_path)[:, :, :3]
        if im.max() <= 1:
            im = im * 255
            pass
        meter, current_meter_type, bndbox = detect_meter(im)
        if bndbox is not None:
            meter = meter / 255.0
            meter_rect = rectify_meter(meter, meter_type)
            reading = read_meter(meter_rect, meter_type)[0]
            pass
        reading_gt = percent_to_reading(gt_percents[fname], meter_type)
        with open(log_file, "a") as f:
            f.write("{}: pred class = {}, "
                    "gt reading = {:.3f}, "
                    "pred reading = {:.3f}\n".format(fname, 
                                                    "ABCDE"[current_meter_type], 
                                                    reading_gt, reading))    
            pass
        err = np.abs(reading_gt - reading)
        test_acc[0] += err < scale_intervals[meter_type]
        test_acc[1] += err < 2 *scale_intervals[meter_type]
        pass
    test_acc /= len(gt_percents)
    print("reading acc",test_acc)
    with open(log_file, "a") as f:
        f.write("{} reading acc: {}\n".format("mixedmodel",test_acc))
        pass
    return


if __name__ == "__main__":
    # get_all_res(video_name)
    for i in range(5):
        meter_type = i
        test_manual()
    # test_virtual()
