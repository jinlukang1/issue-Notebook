import os, sys
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

color_list = {'test': 167, 'test2': 140, 'test3': 136}

def prettify(elem):
    """
        Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf8')
    root = etree.fromstring(rough_string)
    return etree.tostring(root, pretty_print=True, encoding='utf-8').replace("  ".encode(), "\t".encode())

def GetPoint(np_img):
    points_list = []
    try:
        [L, num] = measure.label(np_img[:, :, 2], background = 0, connectivity=2, return_num=True)
        for q in range(num):
            [rows, cols] = np.where(L[:, :] == q+1)
            for defect_color in color_list.keys():
                if np_img[rows[0], cols[0], 2] == color_list[defect_color]:
                    left, right = min(cols), max(cols)
                    up, down = min(rows), max(rows)
                    points_list.append([defect_color, left, up, right, down])
    except:
        [defect_color, left, up, right, down] = None, 0, 0, 0, 0
        points_list.append([defect_color, left, up, right, down])
    return points_list

def GenXml(info_dict, path):
    top = Element('annotation')
    filename = SubElement(top, 'filename')
    filename.text = info_dict['filename']
    size_part = SubElement(top, 'size')
    width = SubElement(size_part, 'width')
    width.text = str(info_dict['width'])
    height = SubElement(size_part, 'height')
    height.text = str(info_dict['height'])
    depth = SubElement(size_part, 'depth')
    depth.text = str(info_dict['depth'])
    for each_object in info_dict['boxlist']:
        object_item = SubElement(top, 'object')
        name = SubElement(object_item, 'name')
        name.text = str(each_object['name'])
        bndbox = SubElement(object_item, 'bndbox')
        xmin = SubElement(bndbox, 'xmin')
        xmin.text = str(each_object['xmin'])
        ymin = SubElement(bndbox, 'ymin')
        ymin.text = str(each_object['ymin'])
        xmax = SubElement(bndbox, 'xmax')
        xmax.text = str(each_object['xmax'])
        ymax = SubElement(bndbox, 'ymax')
        ymax.text = str(each_object['ymax'])
    # tree = ElementTree(top)
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
    print(np_img[427, 1628, 2])

def main(config):
    seg_pic_list = glob.glob(os.path.join(config.seg_path, '*.*[png|jpg|jpeg]'))
    print(seg_pic_list)
    for each_seg_pic in seg_pic_list:
        each_seg_numpy = cv2.imread(each_seg_pic)
        print(each_seg_pic)
        each_seg_numpy = cv2.cvtColor(each_seg_numpy, cv2.COLOR_BGR2RGB)
        h, w, c = each_seg_numpy.shape
        CountColor(each_seg_numpy)
        temp_dict = {}
        temp_dict['boxlist'] = []
        temp_dict['filename'] = each_seg_pic.rsplit('\\', 1)[1]
        temp_dict['width'] = w
        temp_dict['height'] = h
        temp_dict['depth'] = c
        points_list = GetPoint(each_seg_numpy)
        for points in points_list:
            box_temp = {}
            [defect_color, xmin, ymin, xmax, ymax] = points
            box_temp['name'] = defect_color
            box_temp['xmin'] = xmin
            box_temp['ymin'] = ymin
            box_temp['xmax'] = xmax
            box_temp['ymax'] = ymax
            temp_dict['boxlist'].append(box_temp)
            print(xmin, ymin, xmax, ymax)
            GenXml(temp_dict, config.label_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_path', type=str, default='./', help='The path saved the segmentation img')
    parser.add_argument('--ori_path', type=str, default='./', help='The path saved the origin img')
    parser.add_argument('--label_path', type=str, default='./', help='The path saved the label xml')

    config = parser.parse_args()
    main(config)