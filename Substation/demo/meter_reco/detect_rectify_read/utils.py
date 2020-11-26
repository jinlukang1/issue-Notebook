# -*- coding:utf8 -*-
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from skimage import measure
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import xml.etree.ElementTree



n_meter = 5

num_colors = np.array([[216,42,196], # 0
                       [248,132,234], # 1
                       [252,115,56], # 2
                       [155,154,125], # 3
                       [240,252,211], # 4
                       [158,69,202], # 5
                       [176,218,180], # 6
                       [228,155,83], # 7
                       [246,233,185], # 8
                       [53,148,117], # 9
                       [217,86,122], # 10
                       [204,207,253], # 11
                       [201,176,233]]) # 12
coords = np.array([[220,1250], # bottom left color bar, the percentage of the reading, higher bit [x,y]
                   [320,1250], # lower bit
                   [420,1250], # type of the dial damage
                   [220,185], # top left color bar, distance to the meter
                   [2350,185], # top right color bar, yaw
                   [2350,1250]]) # bottom right color bar, pitch
coord_keypoints = np.array([[[1279,709],
                             [1364,880],
                             [1281,900],
                             [1196,871],
                             [1129,813],
                             [1106,729],
                             [1122,647],
                             [1180,583],
                             [1252,547],
                             [1337,551],
                             [1413,598]],
                            [[1445, 507],
                             [1396, 473],
                             [1338, 452],
                             [1280, 445],
                             [1218, 449],
                             [1169, 475],
                             [1117, 509],
                             [1502, 593],
                             [1501, 403],
                             [1053, 403],
                             [1070, 593]],
                            [[1619,626],
                             [1502,539],
                             [1357,493],
                             [1219,493],
                             [1074,538],
                             [952,622],
                             [864,803],
                             [1699,801],
                             [1701,322],
                             [864,322]],
                            [[1480,868],
                             [1520,766],
                             [1518,658],
                             [1474,562],
                             [1387,497],
                             [1282,480],
                             [1178,494],
                             [1091,562],
                             [1044,658],
                             [1043,764],
                             [1075,868]],
                            [[1544,663],
                             [1448,630],
                             [1359,615],
                             [1273,611],
                             [1189,618],
                             [1101,636],
                             [1006,670],
                             [1279,1064],
                             [1694,649],
                             [1279,248],
                             [864,649]]])

theta_ranges = np.array([[1.106,5.580],
                         [3.926,5.508],
                         [3.935,5.490],
                         [2.552,6.873],
                         [3.812,5.621]])
reading_ranges = np.array([[0.55,1],
                           [0,6],
                           [0,5],
                           [-0.1,0.9],
                           [0,3]])
colors = {}
colors["hull"] = np.array([225,255,226])
colors["face"] = np.array([230,198,213])
colors["pointer"] = np.array([255,61,243])
colors["scales"] = np.array([215,142,98])
colors["keypoints"] = np.array([[242,201,197],# max reading, 3.0
                                [199,125,143],
                               [0,126,138],
                               [183,140,93],
                               [227,96,50],
                               [59,246,53],
                               [141,238,180], # 0
                               [121,160,208], # down
                               [205,228,85], # right
                               [249,173,199], # top
                               [170,71,248]]) # left
meter_names = ["biaoji_{}".format(c) for c in "ABCDE"]


class ModelOrientation(torch.nn.Module):
    def __init__(self,meter_type,feat_nums = [3,16,32,64,128,256,512]):
        super(ModelOrientation,self).__init__()
        self.feat_nums = feat_nums
        self.meter_type = meter_type
        self.convs = torch.nn.ModuleList([torch.nn.Conv2d(in_channels = self.feat_nums[i],
                                                          out_channels = self.feat_nums[i + 1],
                                                          kernel_size = 3, stride=1, padding=1)
                                          for i in range(len(self.feat_nums) - 1)])
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm2d(num_features = self.feat_nums[i + 1])
                                        for i in range(len(self.feat_nums) - 1)])
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size = 2)
        if self.meter_type == 2:
            self.fc = torch.nn.Linear(self.feat_nums[-1] * 2 * 4,2)
            pass
        else:
            self.fc = torch.nn.Linear(self.feat_nums[-1] * 2 * 2,2)
            pass
        return
    def forward(self,x):
        conv = x
        for i in range(len(self.feat_nums) - 1):
            conv = self.convs[i](conv)
            conv = self.bns[i](conv)
            conv = self.relu(conv)
            conv = self.pool(conv)
            pass
#         return conv
        fc = conv.view([conv.shape[0],-1])
        output = self.fc(fc)
        cossin = output / torch.norm(output,dim = 1,keepdim = True)
        return cossin
    pass

class ModelPercent(torch.nn.Module):
    def __init__(self,meter_type,feat_nums = [3,16,32,64,128,256,512]):
        super(ModelPercent,self).__init__()
        self.feat_nums = feat_nums
        self.meter_type = meter_type
        self.convs = torch.nn.ModuleList([torch.nn.Conv2d(in_channels = self.feat_nums[i],
                                                          out_channels = self.feat_nums[i + 1],
                                                          kernel_size = 3, stride=1, padding=1)
                                          for i in range(len(self.feat_nums) - 1)])
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm2d(num_features = self.feat_nums[i + 1])
                                        for i in range(len(self.feat_nums) - 1)])
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size = 2)
        if self.meter_type == 2:
            self.w = torch.nn.Parameter(torch.normal(mean=0.0,
                                               std = 0.1 * torch.ones([self.feat_nums[-1] * 2 * 4,1])))
            pass
        else:
            self.w = torch.nn.Parameter(torch.normal(mean=0.0,
                                               std = 0.1 * torch.ones([self.feat_nums[-1] * 2 * 2,1])))
            pass

        return
    def forward(self,x):
        conv = x
        for i in range(len(self.feat_nums) - 1):
            conv = self.convs[i](conv)
            conv = self.bns[i](conv)
            conv = self.relu(conv)
            conv = self.pool(conv)
            pass
#         return conv
        fc = conv.view([conv.shape[0],-1])
        fc1 = fc / torch.norm(fc,dim = 1,keepdim = True)
        percents = torch.squeeze(torch.matmul(fc1,self.w / torch.norm(self.w)),dim = 1)
        return percents
    pass



def getAnno(xml_path):
    tree = xml.etree.ElementTree.parse(xml_path)
    root = tree.getroot()
    object = root.find("object")

    meter_type_str = object.find("name").text
    meter_type = ord(meter_type_str[-1]) - ord('A')

    bndbox = object.find("bndbox")
    xmin = int(bndbox.find("xmin").text)
    ymin = int(bndbox.find("ymin").text)
    xmax = int(bndbox.find("xmax").text)
    ymax = int(bndbox.find("ymax").text)
    bndbox = np.array([xmin,ymin,xmax,ymax])
    return meter_type,bndbox

def color_to_int(rgb):
    return np.where(np.sum(np.equal(rgb,num_colors),axis = 1))[0][0]

def percent_to_reading(percent,meter_type):
    reading = (reading_ranges[meter_type,1] - reading_ranges[meter_type,0]) * percent + reading_ranges[meter_type,0]
    return reading

def orient_to_reading(orient,meter_type):
    if len(orient.shape) == 1:
        theta = np.arctan2(orient[1],orient[0])
    elif len(orient.shape) == 2:
        theta = np.arctan2(orient[:,1],orient[:,0])
    n = np.floor((theta_ranges[meter_type,1] - theta) / (2 * np.pi))
    theta += 2 * np.pi * n
    reading = percent_to_reading((theta - theta_ranges[meter_type,0]) / (theta_ranges[meter_type,1] - theta_ranges[meter_type,0]),meter_type)
    return reading

def extract_seg_bar(seg):
    a = color_to_int(seg[coords[0,1],coords[0,0]])
    b = color_to_int(seg[coords[1,1],coords[1,0]])
    percent = (10 * a + b) / 100
    damage_type = color_to_int(seg[coords[2,1],coords[2,0]])
    distance = color_to_int(seg[coords[3,1],coords[3,0]])
    pitch = color_to_int(seg[coords[4,1],coords[4,0]]) - 6
    yaw = color_to_int(seg[coords[5,1],coords[5,0]]) - 6
    return percent, damage_type, distance, yaw, pitch


def getColorSqrCenter(label,color,region_mask):
    '''
    label: [H,W,3]
    color: [3]
    '''
    mask = np.equal(label,color)
    result = np.ones_like(region_mask,dtype = np.bool)
    for i in range(3):
        result = np.logical_and(result,np.logical_and(mask[:,:,i],region_mask))
        pass
    Y,X = np.nonzero(result)
    if X.size > 0 and Y.size > 0:
        x,y = np.mean(X), np.mean(Y)
    else:
        x,y = -1,-1
    return np.array([x,y])

def rgb_equal(im1,im2):
    mask = np.equal(im1,im2)
    return np.logical_and(np.logical_and(mask[:,:,0],mask[:,:,1]),mask[:,:,2])

def rectify_meter_with_gt(im,seg,meter_type,bndbox):
    X,Y = np.meshgrid(np.arange(im.shape[1]),np.arange(im.shape[0]))
    xmin,ymin,xmax,ymax = bndbox
    mask_rect = np.logical_and(np.logical_and(X >= xmin,X <= xmax),np.logical_and(Y >= ymin,Y <= ymax))
    if meter_type == 2:
        x = np.array([getColorSqrCenter(seg,c,mask_rect) for c in colors["keypoints"][1:]])
    else:
        x = np.array([getColorSqrCenter(seg,c,mask_rect) for c in colors["keypoints"]])
    if np.sum(x[:,0] > 0) < 5:
        return None

    if not meter_type == 2 or np.sum(x[-4:,:] < 0) > 0:
        x1 = x[x[:,0] > 0]
        x1 -= x1.mean(axis = 0)
        ref = np.array(coord_keypoints[meter_type])[x[:,0] > 0].astype(np.float32)
        ref -= ref.mean(axis = 0)
        t1 = np.max(np.linalg.norm(x1,axis = 1))
        t2 = np.max(np.linalg.norm(ref,axis = 1))
        x1 /= t1
        ref /= t2
        M,_ = cv2.estimateAffine2D(x1,ref)
        mask_face = rgb_equal(seg,colors["face"])
        Y,X = np.nonzero(np.logical_and(mask_face,mask_rect))
        coords_face0 = np.stack([X,Y])
        coords_face1 = np.matmul(M[:,:2],coords_face0)
        xmin1,xmax1,ymin1,ymax1 = coords_face1[0,:].min(),coords_face1[0,:].max(),coords_face1[1,:].min(),coords_face1[1,:].max()
        if xmin1 - (xmax1 - xmin1) / 2 < 0:
            M[0,2] = np.int(-(xmin1 - (xmax1 - xmin1) / 2))
        if ymin1 - (ymax1 - ymin1) / 2 < 0:
            M[1,2] = np.int(-(ymin1 - (ymax1 - ymin1) / 2))
        xmin1 += M[0,2]
        xmax1 += M[0,2]
        ymin1 += M[1,2]
        ymax1 += M[1,2]
        im_rect = cv2.warpAffine(im,M,(im.shape[1] * 2, im.shape[0] * 2))

        roi = im_rect[np.int(ymin1 - (ymax1 - ymin1) / 2):np.int(ymax1 + (ymax1 - ymin1) / 2),
                      np.int(xmin1 - (xmax1 - xmin1) / 2):np.int(xmax1 + (xmax1 - xmin1) / 2)]
    # use perspective transform
    else:
        x1 = x[x[:,0] > 0]
        # x1 -= x1.mean(axis = 0)
        ref = np.array(coord_keypoints[meter_type])[x[:,0] > 0].astype(np.float32)
        # ref -= ref.mean(axis = 0)
        ref /= 5
        size = ref[-3] - ref[-1]
        ref -= ref[-1] - size / 2
        M,_ = cv2.findHomography(x1,ref)
        roi = cv2.warpPerspective(im,M,(np.int(size[0] * 2), np.int(size[1] * 2))) # (im.shape[1] * 2, im.shape[0] * 2)
    return roi

def extract_rect_meter(meter_type):
    if not os.path.exists("rectified meters"):
        os.mkdir("rectified meters")
    if not os.path.exists(os.path.join("rectified meters",meter_names[meter_type])):
        os.mkdir(os.path.join("rectified meters",meter_names[meter_type]))
    if not os.path.exists(os.path.join("rectified meters",meter_names[meter_type],"images")):
        os.mkdir(os.path.join("rectified meters",meter_names[meter_type],"images"))
    gt_fname = os.path.join("rectified meters",meter_names[meter_type],"groundtruth.txt")
    if os.path.exists(gt_fname):
        os.remove(gt_fname)
    data_path = os.path.join("../data/biaoji_data_v4",meter_names[meter_type])
    with open(gt_fname,"a") as of:
        weathers = os.listdir(data_path)
        for weather in sorted(weathers):
            print(weather)
            fnames = os.listdir(os.path.join(data_path,weather,"ori"))
            for fname in sorted(fnames):#
                fname = fname[:-4]
                print('\t' + fname)
                im = plt.imread(os.path.join(data_path,weather,"ori",fname + ".jpg"))[:,:,:3]
                seg = plt.imread(os.path.join(data_path,weather,"seg",fname + ".jpg"))[:,:,:3]
                meter_type,bndbox = getAnno(os.path.join(data_path,weather,"anno",fname + ".xml"))
                if bndbox.sum():
                    percent,_,_,_,_ = extract_seg_bar(seg)

                    roi = rectify_meter_with_gt(im,seg,meter_type,bndbox)

                    if roi is not None:
                        im_name = "{}_{}.jpg".format(weather,fname)
                        of.write("{}\t{:.2f}\n".format(im_name,percent))
                        plt.imsave(os.path.join("rectified meters",meter_names[meter_type],"images",im_name),roi)
                pass
            # plt.figure()
            # plt.subplot(1,2,1)
            # plt.imshow(im)
            # plt.title("reading0: {:.2f}".format(reading0))
            # # plt.title("reading0: {:.3f}, reading1: {:.3f}".format(reading0,reading1))
            # plt.subplot(1,2,2)
            # plt.imshow(roi)
            # plt.show()
            pass
        pass
    return

def extract_unrect_meter(meter_type):
    if not os.path.exists("unrectified meters"):
        os.mkdir("unrectified meters")
    if not os.path.exists(os.path.join("unrectified meters",meter_names[meter_type])):
        os.mkdir(os.path.join("unrectified meters",meter_names[meter_type]))
    # if not os.path.exists(os.path.join("unrectified meters",meter_names[meter_type],"images")):
    #     os.mkdir(os.path.join("unrectified meters",meter_names[meter_type],"images"))
    if not os.path.exists(os.path.join("unrectified meters",meter_names[meter_type],"masks")):
        os.mkdir(os.path.join("unrectified meters",meter_names[meter_type],"masks"))
    # gt_fname = os.path.join("unrectified meters",meter_names[meter_type],"groundtruth.txt")
    # if os.path.exists(gt_fname):
    #     os.remove(gt_fname)
    data_path = os.path.join("../data/biaoji_data_v4",meter_names[meter_type])
    # with open(gt_fname,"a") as of:
    weathers = os.listdir(data_path)
    for weather in sorted(weathers):
        print(weather)
        fnames = os.listdir(os.path.join(data_path,weather,"ori"))
        for fname in sorted(fnames):#
            fname = fname[:-4]
            print('\t' + fname)
            im = plt.imread(os.path.join(data_path,weather,"ori",fname + ".jpg"))[:,:,:3]
            seg = plt.imread(os.path.join(data_path,weather,"seg",fname + ".jpg"))[:,:,:3]
            meter_type,bndbox = getAnno(os.path.join(data_path,weather,"anno",fname + ".xml"))
            if bndbox.sum():
                percent,_,_,_,_ = extract_seg_bar(seg)

                xmin,ymin,xmax,ymax = bndbox
                meter = im[np.int(np.maximum(0,(3 * ymin - ymax) / 2)):np.int((3 * ymax - ymin) / 2),
                           np.int(np.maximum(0,(3 * xmin - xmax) / 2)):np.int((3 * xmax - xmin) / 2)]
                mask = seg[np.int(np.maximum(0,(3 * ymin - ymax) / 2)):np.int((3 * ymax - ymin) / 2),
                           np.int(np.maximum(0,(3 * xmin - xmax) / 2)):np.int((3 * xmax - xmin) / 2)]

                if meter is not None:
                    im_name = "{}_{}.jpg".format(weather,fname)
                    # of.write("{}\t{:.2f}\n".format(im_name,percent))
                    # if not os.path.exists(os.path.join("unrectified meters",meter_names[meter_type],"images",im_name)):
                    #     plt.imsave(os.path.join("unrectified meters",meter_names[meter_type],"images",im_name),meter)
                    plt.imsave(os.path.join("unrectified meters",meter_names[meter_type],"masks",im_name[:-3] + "png"),mask)
            pass
        pass
        # pass
    return

# def compute_percent_from_mask(seg,meter_type,bndbox):
#     X,Y = np.meshgrid(np.arange(im.shape[1]),np.arange(im.shape[0]))
#     xmin,ymin,xmax,ymax = bndbox
#     mask_rect = np.logical_and(np.logical_and(X >= xmin,X <= xmax),np.logical_and(Y >= ymin,Y <= ymax))
#     if meter_type == 2:
#         x = np.array([getColorSqrCenter(seg,c,mask_rect) for c in colors["keypoints"][1:]])
#     else:
#         x = np.array([getColorSqrCenter(seg,c,mask_rect) for c in colors["keypoints"]])
#     if np.sum(x[:,0] > 0) < 5:
#         return None
#
#     if not meter_type == 2 or np.sum(x[-4:,:] < 0) > 0:
#         percent =
#     else:
#
#     return percent
#
# def percent_err_gt_mask(meter_type):
#     '''
#     compute percentage error between the groundtruth and the result computed from the pointer and krypoints
#     '''
#     error = 0
#     n = 0
#     data_path = os.path.join("../data/biaoji_data_v4",meter_names[meter_type])
#     weathers = os.listdir(data_path)
#     for weather in sorted(weathers):
#         print(weather)
#         fnames = os.listdir(os.path.join(data_path,weather,"ori"))
#         for fname in sorted(fnames):#
#             fname = fname[:-4]
#             print('\t' + fname)
#             im = plt.imread(os.path.join(data_path,weather,"ori",fname + ".jpg"))[:,:,:3]
#             seg = plt.imread(os.path.join(data_path,weather,"seg",fname + ".jpg"))[:,:,:3]
#             meter_type,bndbox = getAnno(os.path.join(data_path,weather,"anno",fname + ".xml"))
#             if bndbox.sum():
#                 percent,_,_,_,_ = extract_seg_bar(seg)
#                 percent1 = compute_percent_from_mask(seg)
#                 error += np.abs(percent - percent1)
#                 n += 1
#                 pass
#
#     return

def main():
    extract_rect_meter(meter_type = 4)
    return

if __name__ == "__main__":
    main()
