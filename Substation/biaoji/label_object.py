import os
import tqdm
import numpy as np
import cv2
import glob
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import argparse
import codecs
from skimage import measure
from collections import Counter
from utils import makedirs
import threading
import time

color_list = {'biaoji': [230, 198, 213], 'biaoji_A': [143, 234, 243],
              'biaoji_B': [166, 221, 155], 'biaoji_B2': [132, 106, 235], 'biaoji_C': [150, 167, 255],
              'biaoji_D': [163, 66, 144], 'biaoji_E': [225, 255, 226]}

def prettify(elem):
    """
        Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf8')
    root = etree.fromstring(rough_string)
    return etree.tostring(root, pretty_print=True, encoding='utf-8').replace("  ".encode(), "\t".encode())


def GetPoint(np_img):
    points_list = []
    temp_img = np.zeros_like(np_img)
    for defect_color in color_list.keys():
        query_list = np.array(color_list[defect_color])
        (rows, cols) = np.where(np.sum(np.abs(np_img - query_list), axis=2) == 0)
        temp_img[rows, cols, :] = np.array(color_list[defect_color])
    try:
        [L, num] = measure.label(temp_img[:, :, 2], background = 0, connectivity=2, return_num=True)
        for q in range(num):
            [rows, cols] = np.where(L[:, :] == q+1)
            for defect_color in color_list.keys():
                if temp_img[rows[0], cols[0], 2] == np.array(color_list[defect_color][2]):
                    left, right = min(cols), max(cols)
                    up, down = min(rows), max(rows)
                    points_list.append([defect_color, left, up, right, down])
    except:
        [defect_color, left, up, right, down] = None, 0, 0, 0, 0
        points_list.append([defect_color, left, up, right, down])
    return points_list

def sort_points(points_list):
    max_area = 0
    max_area2 = 0
    fin_points = [None, 0, 0, 0, 0]
    fin_name = None
    for points in points_list:
        [defect_color, xmin, ymin, xmax, ymax] = points
        if defect_color == 'biaoji' and (xmin < 1280 and xmax > 1280):
            temp_area = (xmax - xmin) * (ymax - ymin)
            if temp_area > max_area:
                max_area = temp_area
                fin_points = [defect_color, xmin, ymin, xmax, ymax]
        else:
            temp_area2 = (xmax - xmin) * (ymax - ymin)
            if temp_area2 > max_area2:
                max_area2 = temp_area2
                fin_name = defect_color
    if fin_name == 'biaoji_B2':
        fin_name = 'biaoji_B'
    fin_points[0] = fin_name
    return [fin_points]

def GenXml(info_dict, path):
    '''
    :param info_dict:传入的字典信息
    :param path: 生成的voc格式的xml文件保存路径
    :return: None
    '''
    top = Element('annotation')
    filename = SubElement(top, 'filename')
    filename.text = info_dict['filename']
    folder = SubElement(top, 'folder')
    folder.text = 'data'
    anno_path = SubElement(top, 'path')
    anno_path.text = os.path.join('data', info_dict['filename'])
    source = SubElement(top, 'source')
    database = SubElement(source, 'database')
    database.text = 'Unknown'
    segmented = SubElement(top, 'segmented')
    segmented.text = str(0)
    size_part = SubElement(top, 'size')
    width = SubElement(size_part, 'width')
    width.text = str(info_dict['width'])
    height = SubElement(size_part, 'height')
    height.text = str(info_dict['height'])
    depth = SubElement(size_part, 'depth')
    depth.text = str(info_dict['depth'])
    for each_object in info_dict['boxlist']:
        if each_object['xmin'] == 0:
            continue
        object_item = SubElement(top, 'object')
        name = SubElement(object_item, 'name')
        name.text = str(each_object['name'])
        pose = SubElement(object_item, 'pose')
        pose.text = 'Unspecified'
        truncated = SubElement(object_item, 'truncated')
        truncated.text = str(0)
        difficult = SubElement(object_item, 'difficult')
        difficult.text = str(0)
        bndbox = SubElement(object_item, 'bndbox')
        xmin = SubElement(bndbox, 'xmin')
        xmin.text = str(each_object['xmin'])
        ymin = SubElement(bndbox, 'ymin')
        ymin.text = str(each_object['ymin'])
        xmax = SubElement(bndbox, 'xmax')
        xmax.text = str(each_object['xmax'])
        ymax = SubElement(bndbox, 'ymax')
        ymax.text = str(each_object['ymax'])
    prettifyResult = prettify(top)
    out_file = codecs.open(os.path.join(path, info_dict['filename'].rsplit('.', 1)[0]+'.xml'), 'w', encoding='utf-8')
    out_file.write(prettifyResult.decode('utf8'))
    out_file.close()

def CountColor(np_img):
    temp_list = []
    h, w, c = np_img.shape
    for hi in range(h):
        for wi in range(w):
            temp_list.append(np_img[hi, wi, 2])
    print(Counter(temp_list))

def label_object(threadName, config):
    makedirs(config.label_path)
    seg_pic_list = glob.glob(os.path.join(config.seg_path, '*.*[png|jpg|jpeg]'))
    if threadName == 'Thread1':
        reverse_set = True
    else:
        reverse_set = False
    seg_pic_list = sorted(seg_pic_list, reverse=reverse_set)
    pbar = tqdm.tqdm(seg_pic_list[:(len(seg_pic_list)//2)+1])
    for each_seg_pic in pbar:
        pbar.set_description('Img length : {}, Reverse : {}, Thread : {}'.format(len(seg_pic_list), reverse_set, threadName))
        each_seg_numpy = cv2.imread(each_seg_pic)
        # print(each_seg_pic)
        each_seg_numpy = cv2.cvtColor(each_seg_numpy, cv2.COLOR_BGR2RGB)
        h, w, c = each_seg_numpy.shape
        # CountColor(each_seg_numpy)
        temp_dict = {}
        temp_dict['boxlist'] = []
        temp_dict['filename'] = each_seg_pic.rsplit('\\', 1)[1]
        temp_dict['width'] = w
        temp_dict['height'] = h
        temp_dict['depth'] = c
        points_list = GetPoint(each_seg_numpy)
        points_list = sort_points(points_list)
        for points in points_list:
            box_temp = {}
            [defect_color, xmin, ymin, xmax, ymax] = points
            box_temp['name'] = defect_color
            box_temp['xmin'] = xmin
            box_temp['ymin'] = ymin
            box_temp['xmax'] = xmax
            box_temp['ymax'] = ymax
            if defect_color != None:
                temp_dict['boxlist'].append(box_temp)
            # print(xmin, ymin, xmax, ymax)
            GenXml(temp_dict, config.label_path)

class myThread(threading.Thread):
    def __init__(self, threadID, name, config):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.config = config
    def run(self):
        print ("Thread start ：" + self.name)
        label_object(self.name, self.config)
        print ("Thread exit ：" + self.name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_path', type=str, default=r'd:\Documents\AirSim\biaoji_A\sorted\weather0\seg', help='The path saved the segmentation img')
    parser.add_argument('--ori_path', type=str, default=r'd:\Documents\AirSim\biaoji_A\sorted\weather0\ori', help='The path saved the origin img')
    parser.add_argument('--label_path', type=str, default=r'd:\Documents\AirSim\biaoji_A\sorted\weather0\anno', help='The path saved the label xml')
    # parser.add_argument('--reverse', type=bool, default=False)

    config = parser.parse_args()
    thread1 = myThread(1, 'Thread1', config)
    thread2 = myThread(2, 'Thread2', config)

    thread1.start()
    time.sleep(2)
    thread2.start()
    thread1.join()
    thread2.join()

    # label_object(config)
