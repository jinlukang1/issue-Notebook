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
from utils import show_classes

color_list = {'gkxfw_h': [129, 236, 90], 'gkxfw_l': [79, 167, 218], 'bjmh': [146, 104, 241], 'gjbs': [222, 195, 203],
             'xmbhyc': [227, 247, 190], 'tgywjyc': [195, 205, 254], 'hxqyfywyc': [202, 186, 214], 'jsxs': [215, 252, 228], 
             'bjbmyw': [231, 170, 231], 'jyzpl': [207 ,131, 215], 'jyzlw': [238, 185, 179], 'jyz': [183, 140, 93]}
            #  'jyz': [224, 227, 198]}

coords = [[1250, 220], # bottom left color bar, the percentage of the reading, higher bit [x,y]
           [1250, 320], # lower bit
           [1250, 420], # meter type
           [185, 220], # top left color bar, distance to the meter
           [185, 2350], # top right color bar, yaw
           [1250, 2350]] # bottom right color bar, pitch

abnormal_dict = {'normal': [216, 42, 196], 'abnormal': [248, 132, 234]}

def prettify(elem):
    """
        Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf8')
    root = etree.fromstring(rough_string)
    return etree.tostring(root, pretty_print=True, encoding='utf-8').replace("  ".encode(), "\t".encode())


def GetPoint(np_img):
    '''
    :param np_img: 语义图片
    :return: 目标颜色的连通域
    '''
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
    # print(points_list)
    return points_list

def sort_points(points_list):
    result_points = []
    for points in points_list:
        flag = True
        [defect_color, xmin, ymin, xmax, ymax] = points
        if (xmin < 1800 and xmax > 800 and ymin < 1440 and ymax > 200): # 1500 1000
            area = (xmax - xmin) * (ymax - ymin)
            if (defect_color == 'jyz' and area < 100000) or (area < 2000 and defect_color not in ['gkxfw_h', 'gkxfw_l', 'jyzpl']) or area < 20:
                continue
            for cmp_points in result_points:
                [cmp_defect_color, cmp_xmin, cmp_ymin, cmp_xmax, cmp_ymax] = cmp_points
                cmp_area = (cmp_xmax - cmp_xmin) * (cmp_ymax - cmp_ymin)
                w = max(((cmp_xmax - cmp_xmin) + (xmax - xmin) - max(cmp_xmax, xmax) + min(cmp_xmin, xmin)), 0)
                h = max(((cmp_ymax - cmp_ymin) + (ymax - ymin) - max(cmp_ymax, ymax) + min(cmp_ymin, ymin)), 0)
                inter_area = w * h
                if (inter_area / area) * (inter_area / cmp_area) >= 0.8:
                    flag = False
            if flag:
                result_points.append(points)
    
    # 细碎类型缺陷标注框合并
    combine_defeat = ['jyzlw', 'gkxfw_h', 'gkxfw_l', 'hxqyfywyc']
    fin_points = []
    for each_point in result_points:
        if each_point[0] not in combine_defeat:# and each_point[0] != 'xmbhyc': # 金属锈蚀等其他的无需有这个判断
            fin_points.append(each_point)
    for target_defeat in combine_defeat:
        xmin, ymin, xmax, ymax = 2560, 1440, 0, 0
        for each_point in result_points:
            if each_point[0] == target_defeat:
                defect_color = each_point[0]
                xmin = min(each_point[1], xmin)
                ymin = min(each_point[2], ymin)
                xmax = max(each_point[3], xmax)
                ymax = max(each_point[4], ymax)
        if xmin != 2560:
            fin_points.append([defect_color, xmin, ymin, xmax, ymax])
    
        
    return fin_points

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


def show_abnormal_state(config):
    state_result = []
    seg_pic_list = glob.glob(os.path.join(config.seg_path, '*.*[png|jpg|jpeg]'))
    for each_seg_pic in seg_pic_list:
        each_seg_numpy = cv2.imread(each_seg_pic)
        each_seg_numpy = cv2.cvtColor(each_seg_numpy, cv2.COLOR_BGR2RGB)
        # print(each_seg_numpy.shape)
        abnormal_state = each_seg_numpy[1250, 220]
        temp = list(abnormal_state)
        if temp == [216, 42, 196]:
            state_result.append('normal')
        elif temp == [248, 132, 234]:
            state_result.append('abnormal')
        else:
            state_result.append('nothing')
    print(Counter(state_result))
    return state_result



if __name__ == '__main__':
    # 单独的标注策略：
    # 1.部分标注jyz需要限制jyz的大小面积
    # 2.部分类别的标注需要去除其中所含有的箱体
    # 3.gkxfw的小面积物块需要保存
    # 4.考虑一部分物体未能标注上，需要对可标注的物体的范围做扩宽
    # 5.jyzlw和gkxfw的多物块拼接到一起
    # 6.jyzlw标注的时候，jyz会被lw分裂成不同的连通域
    # jyzpl jyz限制100000 去除箱体标注
    # jyzlw jyz限制300000 去除箱体标注
    # bjmh jyz限制50000 去除箱体标注
    items = ['weather2', 'weather6'] # ['jyzlw', 'gjbs', 'jyzpl', 'bjmh', 'gkxfw', 'bjbmyw']
    for item in items:
        print(item + ' start!')
        path = r'D:\Documents\AirSim\defeat_test_sorted\\' + item + r'\\'

        parser = argparse.ArgumentParser()
        parser.add_argument('--seg_path', type=str, default = path + 'seg', help='The path saved the segmentation img')
        parser.add_argument('--ori_path', type=str, default = path + 'ori', help='The path saved the origin img')
        parser.add_argument('--label_path', type=str, default = path + 'anno', help='The path saved the label xml')
        # parser.add_argument('--reverse', type=bool, default=False)

        config = parser.parse_args()

        thread1 = myThread(1, 'Thread1', config)
        thread2 = myThread(2, 'Thread2', config)

        thread1.start()
        time.sleep(2)
        thread2.start()
        thread1.join()
        thread2.join()

        usable_file_list = show_classes(config.label_path)
        print(len(usable_file_list))
        show_abnormal_state(config)

