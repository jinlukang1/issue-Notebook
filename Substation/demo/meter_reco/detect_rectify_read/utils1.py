# -*- coding: utf8 -*-
import numpy as np
import matplotlib.pyplot as plt

import torch, torchvision

import os

import xml.etree.ElementTree

import cv2

data_path = "/data3/home_jinlukang/data/biaoji_data_v5"

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

scale_intervals = np.array([0.01,0.2,0.1,0.02,0.1])

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

def get_colors_of_keypoints(meter_type):
    '''
    get colors of dial tick mark keypoints for each meter, in order of ascending reading values
    note that the numbers of keypoints are different between different meters
    '''
    if meter_type == 0:
        colors_keypoint_pointer = colors["keypoints"][1:]
    elif meter_type == 2:
        colors_keypoint_pointer = colors["keypoints"][6:0:-1]
    elif meter_type == 3:
        colors_keypoint_pointer = colors["keypoints"][::-1]
    else:
        colors_keypoint_pointer = colors["keypoints"][6::-1]
    return colors_keypoint_pointer

def get_anno(xml_path):
    '''
    get meter boundingbox from .xml file
    '''
    tree = xml.etree.ElementTree.parse(xml_path)
    root = tree.getroot()
    obj = root.find("object")

    meter_type_str = obj.find("name").text
    meter_type = ord(meter_type_str[-1]) - ord('A')

    bndbox = obj.find("bndbox")
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
    if len(orient.shape) == 1: # one sample
        theta = np.arctan2(orient[1],orient[0])
    elif len(orient.shape) == 2: # array of samples
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

def get_color_sqr_center(label,color):
    '''
    get the center of the specified color square / bar
    label: [H,W,3]
    color: [3]
    '''
    result = np.prod(np.equal(label,color),axis = 2)
    Y,X = np.nonzero(result)
    if X.size > 0 and Y.size > 0:
        x,y = np.mean(X), np.mean(Y)
    else:
        x,y = -1,-1
    return np.array([x,y])

def im_rgb_equal(im1,im2):
    mask = np.equal(im1,im2)
    return np.prod(mask,axis = 2)#np.logical_and(np.logical_and(mask[:,:,0],mask[:,:,1]),mask[:,:,2])

def rectify_meter_with_gt(im,seg,meter_type,bndbox):
    '''
    rectify image with groundtruth mask and return coord of keypoints (pointer endpoints and scale keypoints)
    '''
    xmin,ymin,xmax,ymax = bndbox
    H, W = im.shape[:2]
    bndbox1 = np.array([(3 * xmin - xmax) / 2, (3 * ymin - ymax) / 2,
                        (3 * xmax - xmin) / 2, (3 * ymax - ymin) / 2])
    bndbox1[::2] = np.clip(bndbox1[::2], 0, W)
    bndbox1[1::2] = np.clip(bndbox1[1::2], 0, W)
    bndbox1 = bndbox1.astype(np.int)
    xmin,ymin,xmax,ymax = bndbox1
    meter_seg = seg[bndbox1[1]:bndbox1[3],bndbox1[0]:bndbox1[2]]
    if meter_type == 2:
        keypoints = np.array([get_color_sqr_center(meter_seg,c) for c in colors["keypoints"][1:]])
    else:
        keypoints = np.array([get_color_sqr_center(meter_seg,c) for c in colors["keypoints"]])
    mask_pointer = im_rgb_equal(meter_seg,colors["pointer"])
    y,x = np.nonzero(mask_pointer)
    if np.sum(keypoints[:,0] > 0) < 5 or y.size == 0:
        return None, None
    if meter_type == 0:
        scale_kp = keypoints[1:]
        center = keypoints[0]
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        theta = lstsq_fit_line(x,y)
        outer_endp, inner_endp = get_outer_inner_endpoint((x,y),theta,scale_kp)
        scale_pointer_keypoints = np.vstack([scale_kp,outer_endp,center])
        pass
    elif meter_type == 3:
        '''
        the mask of the pointer of meter C is a bit of wonky ...
        '''
        scale_kp = keypoints[::-1]
        H, W = meter_seg.shape[:2]
        vs = np.vstack([x - W / 2, y - H / 2]).T # [n, 2]
        d = np.sum(vs ** 2, axis = 1)
        outer_endp_ind = np.argmax(d)
        outer_endp = np.array([x[outer_endp_ind],y[outer_endp_ind]])
        
        vs /= (np.linalg.norm(vs,axis = 1,keepdims = True) + 1e-6)
        v1 = outer_endp - np.array([W / 2, H / 2])
        v1 /= (np.linalg.norm(v1) + 1e-6)
        cross_prod = np.abs(v1[1] * vs[:,0] - v1[0] * vs[:,1])
        on_the_line = cross_prod < 1e-2
        d = d[on_the_line]
        x = x[on_the_line]
        y = y[on_the_line]
        inner_endp_ind = np.argmin(d)
        inner_endp = np.array([x[inner_endp_ind],y[inner_endp_ind]])
        scale_pointer_keypoints = np.vstack([scale_kp,outer_endp,inner_endp])
        pass
    else:
        if meter_type == 2:
            scale_kp = keypoints[5::-1]
        elif meter_type in [1,4]:
            scale_kp = keypoints[6::-1]
        upper_endp_ind = np.argmin(y)
        lower_endp_ind = np.argmax(y)
        upper_endp = np.array([x[upper_endp_ind],y[upper_endp_ind]])
        lower_endp = np.array([x[lower_endp_ind],y[lower_endp_ind]])
        scale_pointer_keypoints = np.vstack([scale_kp,upper_endp,lower_endp])
        pass
    
    keypoints[keypoints[:,0] > 0] += bndbox1[:2]
    scale_pointer_keypoints += bndbox1[:2]
    # if the meter is in A,B,D,E,
    # or the meter is C but the 4 corner keypoints are not all visible,
    # then use affine transformation
    if not meter_type == 2 or np.sum(keypoints[-4:,:] < 0) > 0:
        keypoints1 = keypoints[keypoints[:,0] > 0]
        ref = np.array(coord_keypoints[meter_type])[keypoints[:,0] > 0].astype(np.float32)
        keypoints1 -= keypoints1.mean(axis = 0)
        ref -= ref.mean(axis = 0)
        t1 = np.max(np.linalg.norm(keypoints1,axis = 1))
        t2 = np.max(np.linalg.norm(ref,axis = 1))
        keypoints1 /= t1
        ref /= t2
        M,_ = cv2.estimateAffine2D(keypoints1,ref)

        mask_face = im_rgb_equal(seg,colors["face"])
        Y,X = np.nonzero(mask_face[ymin:ymax,xmin:xmax])
        coords_face0 = np.stack([X,Y]).T
        coords_face0 += bndbox1[:2]
        coords_face1 = np.matmul(M[:,:2],coords_face0.T).T
        xmin1,xmax1,ymin1,ymax1 = coords_face1[:,0].min(),coords_face1[:,0].max(),coords_face1[:,1].min(),coords_face1[:,1].max()
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
        scale_pointer_keypoints[scale_pointer_keypoints[:,0] > 0] = (np.matmul(M[:,:2], scale_pointer_keypoints[scale_pointer_keypoints[:,0] > 0].T)).T + M[:,2]
        scale_pointer_keypoints[scale_pointer_keypoints[:,0] > 0] -= np.array([np.int(xmin1 - (xmax1 - xmin1) / 2),
                                                                                np.int(ymin1 - (ymax1 - ymin1) / 2)])
        pass
    # use perspective transform
    else:
        keypoints1 = keypoints[keypoints[:,0] > 0]
        ref = np.array(coord_keypoints[meter_type])[keypoints[:,0] > 0].astype(np.float32)
        ref /= 5
        size = ref[-3] - ref[-1]
        tl = ref[-1] - size / 2
        ref -= tl
        M,_ = cv2.findHomography(keypoints1,ref)
        roi = cv2.warpPerspective(im,M,(np.int(size[0] * 2), np.int(size[1] * 2)))
        scale_pointer_keypoints[scale_pointer_keypoints[:,0] > 0] = cv2.perspectiveTransform(np.array([scale_pointer_keypoints[scale_pointer_keypoints[:,0] > 0]]), M)[0]
        pass
    return roi, scale_pointer_keypoints.astype(np.int)

def extract_rect_meter(meter_type,data_path = data_path,meter_path = "rectified meters v5"):
    if not os.path.exists(meter_path):
        os.mkdir(meter_path)
    if not os.path.exists(os.path.join(meter_path,meter_names[meter_type])):
        os.mkdir(os.path.join(meter_path,meter_names[meter_type]))
    if not os.path.exists(os.path.join(meter_path,meter_names[meter_type],"images")):
        os.mkdir(os.path.join(meter_path,meter_names[meter_type],"images"))
    if not os.path.exists(os.path.join(extract_path,meter_names[meter_type],"masks")):
        os.mkdir(os.path.join(extract_path,meter_names[meter_type],"masks"))
    gt_fname = os.path.join(meter_path,meter_names[meter_type],"groundtruth percents keypoints.txt")
    if os.path.exists(gt_fname):
        os.remove(gt_fname)
    data_path1 = os.path.join(data_path,meter_names[meter_type])
    
    weathers = os.listdir(data_path1)
    for weather in sorted(weathers):
        print(weather)
        fnames = os.listdir(os.path.join(data_path1,weather,"ori"))
        ims_to_show = np.random.choice(fnames, 5, replace = False)
        for fname in sorted(fnames):
            fname = fname[:-4]
            print('\t' + fname)
            im = plt.imread(os.path.join(data_path1,weather,"ori",fname + ".jpg"))[:,:,:3]
            seg = plt.imread(os.path.join(data_path1,weather,"seg",fname + ".jpg"))[:,:,:3]
            meter_type,bndbox = get_anno(os.path.join(data_path1,weather,"anno",fname + ".xml"))
            if bndbox.sum():
                percent,_,_,_,_ = extract_seg_bar(seg)
                roi, keypoints = rectify_meter_with_gt(im,seg,meter_type,bndbox)
                if roi is not None:
                    im_name = "{}_{}.jpg".format(weather,fname)
                    with open(gt_fname,"a") as of:
                        of.write("{}\t{:.2f}\t".format(im_name,percent))
                        for k,p in enumerate(keypoints):
                            of.write("{} {} ".format(p[0],p[1]))
                            pass
                        of.write("\n")
                        pass
                    # im_name_to_save = os.path.join(meter_path,meter_names[meter_type],"images",im_name)
                    # if not os.path.exists(im_name_to_save):
                    #     plt.imsave(im_name_to_save,roi)
                    if fname + ".jpg" in ims_to_show:
                        roi1 = plt.imread(os.path.join("/data3/home_jinlukang/pengkunfu/rectified meters v5","biaoji_{}".format("ABCDE"[meter_type]),"images","{}_{}.jpg".format(weather, fname)))[:,:,:3]
                        if roi1.max() > 1:
                            roi1 = roi1 / 255
                        for k,p in enumerate(keypoints):
                            roi1[p[1],p[0]] = np.clip((num_colors[k] if k < num_colors.shape[0] else np.array([255,0,0])) / 255, 0, 1)
                            pass                        
                        plt.imsave("rect_keypoint_sample_{}_{}_{}.png".format("ABCDE"[meter_type], weather, fname),roi1)
                    pass
                pass
            pass
        pass
    return

def extract_all_rect_meter():
    for i in range(5):
        extract_rect_meter(i)
        pass
    return

def extract_unrect_meter(meter_type,extract_path = "unrectified meters v5"):
    if not os.path.exists(extract_path):
        os.mkdir(extract_path)
    if not os.path.exists(os.path.join(extract_path,meter_names[meter_type])):
        os.mkdir(os.path.join(extract_path,meter_names[meter_type]))
    if not os.path.exists(os.path.join(extract_path,meter_names[meter_type],"images")):
        os.mkdir(os.path.join(extract_path,meter_names[meter_type],"images"))
    if not os.path.exists(os.path.join(extract_path,meter_names[meter_type],"masks")):
        os.mkdir(os.path.join(extract_path,meter_names[meter_type],"masks"))
    gt_fname = os.path.join(extract_path,meter_names[meter_type],"groundtruth.txt")
    if os.path.exists(gt_fname):
        os.remove(gt_fname)
    meter_path = os.path.join(data_path,meter_names[meter_type])
    
    weathers = os.listdir(meter_path)
    for weather in sorted(weathers):
        print(weather)
        fnames = os.listdir(os.path.join(meter_path,weather,"ori"))
        for fname in sorted(fnames):#
            fname = fname[:-4]
            print('\t' + fname)
            im = plt.imread(os.path.join(meter_path,weather,"ori",fname + ".jpg"))[:,:,:3]
            seg = plt.imread(os.path.join(meter_path,weather,"seg",fname + ".jpg"))[:,:,:3]
            meter_type,bndbox = get_anno(os.path.join(meter_path,weather,"anno",fname + ".xml"))
            if bndbox.sum():
                percent,_,_,_,_ = extract_seg_bar(seg)

                xmin,ymin,xmax,ymax = bndbox
                meter = im[np.int(np.maximum(0,(3 * ymin - ymax) / 2)):np.int((3 * ymax - ymin) / 2),
                            np.int(np.maximum(0,(3 * xmin - xmax) / 2)):np.int((3 * xmax - xmin) / 2)]
                mask = seg[np.int(np.maximum(0,(3 * ymin - ymax) / 2)):np.int((3 * ymax - ymin) / 2),
                            np.int(np.maximum(0,(3 * xmin - xmax) / 2)):np.int((3 * xmax - xmin) / 2)]

                if meter is not None:
                    im_name = "{}_{}.jpg".format(weather,fname)
                    with open(gt_fname,"a") as of:
                        of.write("{}\t{:.2f}\n".format(im_name,percent))
                        pass
                    im_name_to_save = os.path.join(extract_path,meter_names[meter_type],"images",im_name)
                    if not os.path.exists(im_name_to_save):
                        plt.imsave(im_name_to_save,meter)
                    plt.imsave(os.path.join(extract_path,meter_names[meter_type],"masks",im_name[:-3] + "png"),mask)
                    pass
                pass
            pass
        pass
    return


def lstsq_fit_line(x,y):
    '''
    x: x-coordinates, [n], n > 1
    y: y-coordinates, [n]
    theta: angle of the normal direction of line, in []
    '''
    eps = 1e-5
    x1 = x - x.mean()
    y1 = y - y.mean()
    t = np.sqrt(np.max(x1 ** 2 + y1 ** 2))
    x1 /= (t + eps)
    y1 /= (t + eps)
    cov_xy2 = 2 * np.mean(x1 * y1)
    var_xy = np.mean(x1 ** 2) - np.mean(y1 ** 2)
    theta2 = np.arctan2(cov_xy2,var_xy)
    if theta2 < 0:
        theta2 += 2 * np.pi
    theta = theta2 / 2
    if np.cos(2 * theta) * var_xy + np.sin(2 * theta) * cov_xy2 > 0:
        theta += np.pi / 2
    return theta

def get_outer_inner_endpoint(points,theta,keypoints):
    '''
    get outer endpoint and inner endpoint with respect to keypoints, by 
    inputs:


    outputs:
    '''
    x,y = points
    v = np.array([np.cos(theta + np.pi / 2),
                  np.sin(theta + np.pi / 2)])
    ps = np.array([x,y]).T
    # first, get the endpoints from inner product with the normal vector of the line
    inner_prods = np.matmul(ps,v)
    endp1 = ps[np.argmax(inner_prods)]
    endp2 = ps[np.argmin(inner_prods)]
    # then seperate inner from outer by comparing distance
    if np.min(np.sum((endp1 - keypoints) ** 2,axis = 1)) < np.min(np.sum((endp2 - keypoints) ** 2,axis = 1)):
        outer_endp,inner_endp = endp1,endp2
    else:
        outer_endp,inner_endp = endp2,endp1
    return outer_endp, inner_endp

def compute_reading_from_mask(meter_seg,meter_type):
    '''
    inputs:
        meter_seg:
        meter_type:
    output:
        reading: 
    '''
    keypoints = np.array([get_color_sqr_center(meter_seg,c) for c in get_colors_of_keypoints(meter_type)])
    
    # cos(2 * theta) * 2 * Cov(X,Y) + sin(2 * theta) * (Var(Y) - Var(X)) = 0 =>
    # (cos(2 * theta),sin(2 * theta)) = (Var(X) - Var(Y),2 * Cov(X,Y))
    pointer_mask = im_rgb_equal(meter_seg,colors["pointer"])
    y,x = np.nonzero(pointer_mask)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    theta = lstsq_fit_line(x,y)
    outer_endp, inner_endp = get_outer_inner_endpoint((x,y),theta,keypoints)

    if np.dot(outer_endp - inner_endp,np.array([np.cos(theta - np.pi / 2),np.sin(theta - np.pi / 2)])) < 0:
        theta += np.pi
    v = np.array([np.cos(theta),np.sin(theta)])
    ds = np.matmul(keypoints - outer_endp,v)
    k = -1
    for i in range(ds.shape[0] - 1):
        if ds[i] <= 0 and ds[i + 1] > 0:
            k = i
            break
    if k == -1:
        v1 = keypoints[0] - inner_endp
        v2 = keypoints[-1] - inner_endp
        v3 = outer_endp - inner_endp
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        v3 /= np.linalg.norm(v3)
        if np.dot(v1,v3) > np.dot(v2,v3):
            percent_fit = 0
            pass
        else:
            percent_fit = 1
            pass
        pass
    else:
        percent_fit = (k + (-ds[k]) / (ds[k + 1] - ds[k])) / (keypoints.shape[0] - 1)
        pass
    return percent_to_reading(percent_fit,meter_type)

def compute_systematic_err(dataset_path = data_path, err_file = "systematic err.txt"):
    for meter_type in range(5):
        with open(err_file,'a') as f:
            f.write("{}\n".format("ABCDE"[meter_type]))
        pass
        meter_path = os.path.join(dataset_path,meter_names[meter_type])
        for weather in sorted(os.listdir(meter_path)):
            errs = []
            with open(err_file,'a') as f:
                f.write("\t{}\n".format(weather))
            pass
            weather_path = os.path.join(meter_path,weather)
            im_path = os.path.join(weather_path,"ori")
            seg_path = os.path.join(weather_path,"seg")
            for im_name in sorted(os.listdir(seg_path)):
                index = im_name[:5]
                anno_path = os.path.join(weather_path,"anno",index + ".xml")
                _, bndbox = getAnno(anno_path)
                seg = plt.imread(os.path.join(seg_path,im_name))[:,:,:3]
                if seg.max() <= 1:
                    seg = np.uint8(seg * 255)
                    pass
                percent_gt, _, _, _, _ = extract_seg_bar(seg)

                xmin,ymin,xmax,ymax = bndbox
                meter_seg = seg[np.int(np.maximum(ymin - (ymax - ymin) / 2,0)):np.int(np.minimum(ymax + (ymax - ymin) / 2,seg.shape[0])),
                                np.int(np.maximum(xmin - (xmax - xmin) / 2,0)):np.int(np.minimum(xmax + (xmax - xmin) / 2,seg.shape[1]))]
                if im_rgb_equal(meter_seg,colors["pointer"]).sum() > 1:
                    percent_fit = compute_reading_from_mask(meter_seg,meter_type)
                    err0 = np.abs(percent_to_reading(percent_gt,meter_type) - percent_to_reading(percent_fit,meter_type))
                    errs.append(err0)
                    with open(err_file,'a') as f:
                        f.write("\t\t{}: reading err = {:.3f}\n".format(im_name,err0))
                    pass
                else:
                    with open(err_file,'a') as f:
                        f.write("\t\t{}: no pointer or one-point pointer mask\n".format(im_name))
                    pass
                pass

            errs = np.array(errs)
            with open(err_file,'a') as f:
                f.write("{}: mean reading err = {:.3f}\n".format("ABCDE"[meter_type],errs.mean()))
                pass
            pass
        pass
    return

def find_problematic_meters(dataset_path = data_path, output_file = "problematic meters.txt"):
    '''
    find meters without pointer mask, or with only one point of pointer mask, or with erroneous reading
    '''
    for meter_type in range(5):
        with open(output_file, "a") as f:
            f.write("{}\n".format(meter_names[meter_type]))
            pass
        meter_path = os.path.join(dataset_path,meter_names[meter_type])
        for weather in sorted(os.listdir(meter_path)):
            errs = []
            with open(output_file, "a") as f:
                f.write("\t{}\n".format(weather))
                pass
            weather_path = os.path.join(meter_path,weather)
            seg_path = os.path.join(weather_path,"seg")
            for im_name in sorted(os.listdir(seg_path)):
                index = im_name[:5]
                anno_path = os.path.join(weather_path,"anno",index + ".xml")
                _, bndbox = get_anno(anno_path)
                seg = plt.imread(os.path.join(seg_path,im_name))[:,:,:3]
                if seg.max() <= 1:
                    seg = np.uint8(seg * 255)
                    pass
                percent, _, _, _, _ = extract_seg_bar(seg)

                xmin,ymin,xmax,ymax = bndbox
                h, w = ymax - ymin, xmax - xmin
                meter_seg = seg[np.int(np.maximum(ymin - h / 2,0)):np.int(np.minimum(ymax + h / 2,seg.shape[0])),
                                np.int(np.maximum(xmin - w / 2,0)):np.int(np.minimum(xmax + w / 2,seg.shape[1]))]

                pointer_mask = im_rgb_equal(meter_seg,colors["pointer"])
                if np.sum(pointer_mask) == 0:
                    with open(output_file, "a") as f:
                        f.write("\t\t{}: no pointer mask\n".format(im_name))
                        pass
                    pass
                elif np.sum(pointer_mask) == 1:
                    with open(output_file, "a") as f:
                        f.write("\t\t{}: one-point pointer mask\n".format(im_name))
                        pass
                    pass
                else:
                    reading_fit = compute_reading_from_mask(meter_seg,meter_type)
                    reading_gt = percent_to_reading(percent,meter_type)
                    err = np.abs(reading_gt - reading_fit)
                    if err > scale_intervals[meter_type]:
                        with open(output_file, "a") as f:
                            f.write("\t\t{}: gt reading = {:.3f}, fit reading = {:.3f}\n".format(im_name, reading_gt, reading_fit))
                            pass
                        pass
                    pass
                pass
            pass
        pass
    return

def sample(meter_type,samples_path,nSample = 200):
    if not os.path.exists(os.path.join(samples_path,"ABCDE"[meter_type])):
        os.mkdir(os.path.join(samples_path,"ABCDE"[meter_type]))
        pass
    gt_reading_file = os.path.join(samples_path,"ABCDE"[meter_type],"ground_truth_reading.txt")
    if os.path.exists(gt_reading_file):
        os.remove(gt_reading_file)
        pass
    samples_name = []
    meter_path = os.path.join(data_path,"biaoji_{}".format("ABCDE"[meter_type]))
    weathers = os.listdir(meter_path)
    for w in weathers:
        im_names = [os.path.join(w,im_name) for im_name in os.listdir(os.path.join(meter_path,w,"ori")) if int(im_name[:5]) >= 1500]
        samples_name.extend(im_names)
        pass
    samples_name = np.random.choice(samples_name,nSample,replace = False)
    with open(gt_reading_file,'w') as f:
        for sample_name in samples_name:
            w,im_name = os.path.split(sample_name)
            im = plt.imread(os.path.join(meter_path,w,"ori",im_name))[:,:,:3]
            seg = plt.imread(os.path.join(meter_path,w,"seg",im_name))[:,:,:3]
            _,bndbox = get_anno(os.path.join(meter_path,w,"anno",im_name[:-3] + "xml"))
            xmin,ymin,xmax,ymax = bndbox
            meter = im[np.int(np.maximum(ymin - (ymax - ymin) / 2,0)):np.int(np.minimum(ymax + (ymax - ymin) / 2,seg.shape[0])),
                    np.int(np.maximum(xmin - (xmax - xmin) / 2,0)):np.int(np.minimum(xmax + (xmax - xmin) / 2,seg.shape[1]))]
            percent,_,_,_,_ = extract_seg_bar(seg)
            reading = percent_to_reading(percent,meter_type)
            plt.imsave(os.path.join(samples_path,"ABCDE"[meter_type],sample_name.replace('/','_')),meter)
            f.write("{}\t{:.3f}\n".format(sample_name,reading))
            pass
        pass
    return

class Model(torch.nn.Module):
    '''
    feature extractor for meter reading regression or pointer orientation regression
    '''
    def __init__(self,feat_nums = [3,16,32,64,128,256,512]):
        super(Model,self).__init__()
        self.feat_nums = feat_nums
        self.convs = torch.nn.ModuleList([torch.nn.Conv2d(in_channels = self.feat_nums[i],
                                                          out_channels = self.feat_nums[i + 1],
                                                          kernel_size = 3, stride=1, padding=1)
                                          for i in range(len(self.feat_nums) - 1)])
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm2d(num_features = self.feat_nums[i + 1])
                                        for i in range(len(self.feat_nums) - 1)])
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size = 2)
        return
    def forward(self,x):
        conv = x
        for i in range(len(self.feat_nums) - 1):
            conv = self.convs[i](conv)
            conv = self.bns[i](conv)
            conv = self.relu(conv)
            conv = self.pool(conv)
            pass
        return conv
    pass

def split_train_val(records, val_ratio = 0.1):
    '''
    '''
    N = len(records)
    np.random.shuffle(records)
    train_records = records[np.int(val_ratio * N):]
    val_records = records[:np.int(val_ratio * N)]
    return train_records, val_records

def area(bndbox):
    if len(bndbox.shape) == 1:
        return np.prod(bndbox[2:] - bndbox[:2])
    elif len(bndbox.shape) == 2:
        return np.prod(bndbox[:,2:] - bndbox[:,:2],axis = 1)

def interval_intersect(intervals1,intervals2, mode):
    '''
    input: 
        intervals1: [n1, 2]
        intervals2: [n2, 2]
        mode: "xyxy" or "xywh"
    return:
        intersect: [n1, n2]
    '''
    x11 = intervals1[:,0]
    x21 = intervals2[:,0]
    if mode == "xyxy":
        x12 = intervals1[:,1]
        x22 = intervals2[:,1]
        pass
    elif mode == "xywh":
        x12 = intervals1[:,1] + intervals1[:,0]
        x22 = intervals2[:,1] + intervals2[:,0]
        pass
    intersect = np.maximum(0, np.minimum(np.expand_dims(x12,axis = 1), np.expand_dims(x22,axis = 0)) - np.maximum(np.expand_dims(x11,axis = 1),np.expand_dims(x21, axis = 0)))
    return intersect

def compute_IoU(bndboxes1, bndboxes2, mode):
    '''
    input: 
        bndboxes1: [n1, 4]
        bndboxes2: [n2, 4]
        mode: "xyxy" or "xywh"
    return:
        IoUs: [n1, n2]
    '''
    eps = 1e-5
    n1 = bndboxes1.shape[0]
    n2 = bndboxes2.shape[0]
    if mode == "xyxy":
        w1 = bndboxes1[:,2] - bndboxes1[:,0]
        w2 = bndboxes2[:,2] - bndboxes2[:,0]
        h1 = bndboxes1[:,3] - bndboxes1[:,1]
        h2 = bndboxes2[:,3] - bndboxes2[:,1]
        pass
    elif mode == "xywh":
        w1 = bndboxes1[:,2]
        w2 = bndboxes2[:,2]
        h1 = bndboxes1[:,3]
        h2 = bndboxes2[:,3]
        pass
    S1 = w1 * h1
    S2 = w2 * h2
    
    x_intersect = interval_intersect(bndboxes1[:,0::2], bndboxes2[:,0::2], mode)
    y_intersect = interval_intersect(bndboxes1[:,1::2], bndboxes2[:,1::2], mode)
    area_intersect = x_intersect * y_intersect
    IoUs = area_intersect / (np.expand_dims(S1, axis = 1) + np.expand_dims(S2, axis = 0) - area_intersect + eps)
    return IoUs

def determine_TP(bndboxes_pred_a_sample, bndboxes_gt_a_sample, n_classes, IoUTH, mode):
    '''
    determine True Positive for boundingbox predictions of one image
    input:
        [[class1, score, x1, y1, x2, y2], [class2, score, x1, y1, x2, y2], ...]
        [n, 6]
        n can be 0
    output:
        [[[score, TP (0 or 1)], [score, TP (0 or 1)], ...], ...]
        [n_classes], [k,2]
    '''
    res = []
    for i in range(n_classes):
        TPs = []
        if bndboxes_pred_a_sample.shape[0]:
            bndboxes = bndboxes_pred_a_sample[bndboxes_pred_a_sample[:,0] == i, 1:]
            if bndboxes.shape[0]:
                inds = np.argsort(bndboxes[:,0])[::-1]
                bndboxes = bndboxes[inds]
                bndboxes_gt = bndboxes_gt_a_sample[bndboxes_gt_a_sample[:,0] == i, 1:]
                if bndboxes_gt.shape[0]:
                    IoUs = compute_IoU(bndboxes[:,1:], bndboxes_gt, mode)
                    for j in range(bndboxes.shape[0]):
                        if np.max(IoUs[j]) > IoUTH:
                            IoUs[:,np.argmax(IoUs[j])] = 0
                            TP = 1
                            pass
                        else:
                            TP = 0
                            pass
                        TPs.append([bndboxes[j,0],TP])
                        pass
                    pass
                    TPs = np.array(TPs)
                else:
                    TPs = np.array([bndboxes[:,0],np.zeros(bndboxes.shape[0])]).T
                    pass
                res.append(TPs)
                pass
            else:
                res.append(np.empty([0,2]))
                pass
            pass
        else:
            res.append(np.empty([0,2]))
            pass
        pass
    return res

def compute_AP(bndboxes_pred, bndboxes_gt, n_classes, IoUTH, mode):
    '''
    input:
        bndboxes_pred: 
        list of boundingbox predictions for each image, 
        [[[class1, score, x1, y1, x2, y2], [class2, score, x1, y1, x2, y2], ...],
         [[class1, score, x1, y1, x2, y2], [class2, score, x1, y1, x2, y2], ...],
         ...]
        bndboxes_gt: 
        list of groundtruth boundingboxes for each image
        [[[class, x1, y1, x2, y2], ...],
         [[class, x1, y1, x2, y2], ...]]
    output:
        APs: [AP_class1, AP_class2, ...]
    '''
    assert len(bndboxes_pred) == len(bndboxes_gt)
    n = len(bndboxes_pred)
    TPs = [[] for i in range(n_classes)]
    n_gts = np.zeros(n_classes)
    for k in range(n):
        TP_sample_k = determine_TP(bndboxes_pred[k], bndboxes_gt[k], n_classes, IoUTH, mode)
        for i in range(n_classes):
            TPs[i].append(TP_sample_k[i])
            if bndboxes_gt[k].shape[0]:
                n_gts[i] += np.sum(bndboxes_gt[k][:,0] == i)
                pass
            pass
        pass
    for i in range(n_classes):
        TPs[i] = np.vstack(TPs[i])
        TPs[i] = TPs[i][np.argsort(TPs[i][:,0])[::-1]]
        pass
    APs = np.zeros(n_classes)
    for i in range(n_classes):
        if TPs[i].shape[0]:
            TP_sum = np.cumsum(TPs[i][:,1])
            PR = np.vstack([TP_sum / (np.arange(TP_sum.shape[0]) + 1), TP_sum / n_gts[i]]).T
            P_max = PR[-1,0]
            for j in range(PR.shape[0] - 1,-1,-1):
                if PR[j,0] < P_max:
                    PR[j,0] = P_max
                    pass
                else:
                    P_max = PR[j,0]
                    pass
                pass
            APs[i] = PR[0,0] * PR[0,1] + np.sum((PR[1:,1] - PR[:-1,1]) * PR[:-1,0])
            pass
        pass
    return APs