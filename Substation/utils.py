import glob
import os
import shutil
import tqdm
import random
import numpy as np
import cv2
from xml.etree import ElementTree as ET
from collections import Counter

num_colors = [[216,42,196], # 0
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
            [201,176,233]] # 12

coords = [[1250, 220], # bottom left color bar, the percentage of the reading, higher bit [x,y]
           [1250, 320], # lower bit
           [185, 220], # top left color bar, distance to the meter
           [185, 2350], # top right color bar, yaw
           [1250, 2350]] # bottom right color bar, pitch

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def SortFiles(source_path, target_path):
    All_fileslist = glob.glob(os.path.join(source_path, '*.png'))
    Ori_fileslist = [x for x in All_fileslist if '_0_0' in x]
    Seg_fileslist = [x for x in All_fileslist if '_0_5' in x]
    makedirs(os.path.join(target_path, 'ori'))
    makedirs(os.path.join(target_path, 'seg'))
    for file in tqdm.tqdm(Ori_fileslist):
        file_name = file.rsplit('\\', 1)[1]
        shutil.copyfile(file, os.path.join(target_path, 'ori', file_name))

    for file in tqdm.tqdm(Seg_fileslist):
        file_name = file.rsplit('\\', 1)[1]
        shutil.copyfile(file, os.path.join(target_path, 'seg', file_name))
    print('Done!')

def RenameFiles(path):
    All_fileslist = glob.glob(os.path.join(path, '*.*[png|jpg|jpeg|xml]'))
    All_fileslist.sort()
    for file in All_fileslist:
        new_filename = os.path.join(file.rsplit('\\', 1)[0], str(All_fileslist.index(file)).zfill(5)+'.xml')
        os.rename(file, new_filename)
    print('Done!')

def SortToVOC(path):
    random.seed(55)
    anno_list = glob.glob(os.path.join(path, 'anno', '*.xml'))
    img_list = glob.glob(os.path.join(path, 'ori', '*.png'))
    print(len(img_list))
    random.shuffle(img_list)
    train_img_list = img_list[:int(0.8*len(img_list))]
    test_img_list = img_list[int(0.8*len(img_list)):]
    train_img_list.sort()
    test_img_list.sort()
    makedirs(os.path.join(path, 'VOC_format_dataset', 'Annotations'))
    makedirs(os.path.join(path, 'VOC_format_dataset', 'JPEGImages'))
    makedirs(os.path.join(path, 'VOC_format_dataset', 'ImageSets', 'Main'))
    # for img in img_list:
    #     shutil.copyfile(img, os.path.join(path, 'VOC_format_dataset', 'JPEGImages', img.rsplit('\\')[-1]))
    # for anno in anno_list:
    #     shutil.copyfile(anno, os.path.join(path, 'VOC_format_dataset', 'Annotations', anno.rsplit('\\')[-1]))
    with open(os.path.join(path, 'VOC_format_dataset', 'ImageSets', 'Main', 'trainval.txt'), 'w') as f:
        for train_img in train_img_list:
            train_img_name = train_img.rsplit('\\')[-1].rsplit('.')[0]
            f.write(train_img_name)
            f.write('\n')
    with open(os.path.join(path, 'VOC_format_dataset', 'ImageSets', 'Main', 'test.txt'), 'w') as f:
        for test_img in test_img_list:
            test_img_name = test_img.rsplit('\\')[-1].rsplit('.')[0]
            f.write(test_img_name)
            f.write('\n')

def judge_data(np_img):
    flag = True
    img_distance = np_img[coords[2][0], coords[2][1]]
    img_yaw = np_img[coords[3][0], coords[3][1]]
    img_pitch = np_img[coords[4][0], coords[4]][1]
    if not img_distance.tolist() in num_colors:
        flag = False
    if not img_yaw.tolist() in num_colors:
        flag = False
    if not img_pitch.tolist() in num_colors:
        flag = False
    if flag == False:
        return False
    else:
        return True


def wash_data(seg_path):
    # 按照judge_data中的规则清洗数据
    bad_datalist = []
    seg_list = glob.glob(os.path.join(seg_path, '*.*[png|jpg|jpeg]'))
    for each_seg in tqdm.tqdm(seg_list):
        # file_name = each_seg.rsplit('\\', 1)[1]
        each_seg_numpy = cv2.imread(each_seg)
        each_seg_numpy = cv2.cvtColor(each_seg_numpy, cv2.COLOR_BGR2RGB)
        if not judge_data(each_seg_numpy):
            bad_datalist.append(each_seg)
    print(bad_datalist)
    for bad_data in bad_datalist:
        try:
            os.remove(bad_data)
            os.remove(bad_data.replace('seg', 'ori'))
            os.remove(bad_data.replace('seg', 'anno').replace('png', 'xml'))
        except:
            print('{} not exist'.format(bad_data))

def show_classes(anno_path):
    # 显示数据集中的各个表计的类别
    anno_list = glob.glob(os.path.join(anno_path, '*.xml'))
    class_list = []
    for each_anno in anno_list:
        tree = ET.parse(each_anno)
        annotation = tree.getroot()
        for box in annotation.iter('object'):
            for attrib in box:
                if attrib.tag == 'name':
                    class_list.append(attrib.text)
    print(Counter(class_list))

def conbine_data(path):
    # 将多个需要合并的数据文件放在path下
    all_img_list = glob.glob(os.path.join(path, '*', '*', '*.jpg'))
    all_img_list.sort()
    all_anno_list = glob.glob(os.path.join(path, '*', '*', '*.xml'))
    all_anno_list.sort()
    print('img: {}, anno: {}'.format(len(all_img_list), len(all_anno_list)))
    target_path = os.path.join(path, 'conbined_data')
    makedirs(os.path.join(target_path, 'ori'))
    makedirs(os.path.join(target_path, 'anno'))
    for each_img in tqdm.tqdm(all_img_list):
        shutil.copyfile(each_img, os.path.join(target_path, 'ori', str(all_img_list.index(each_img)).zfill(5)+'.jpg'))
    for each_anno in tqdm.tqdm(all_anno_list):
        shutil.copyfile(each_anno, os.path.join(target_path, 'anno', str(all_anno_list.index(each_anno)).zfill(5)+'.xml'))
    print('Done!')

def temp(path):
    all_files = glob.glob(os.path.join(path, '*.jpg'))
    print(len(all_files))
    for each_file in all_files:
        os.rename(each_file, each_file.replace('jpg', 'xml'))
    print('Done!')

if __name__ == '__main__':
    # SortFiles(r'D:\Documents\AirSim\images', r'D:\Documents\AirSim\Timages')
    # RenameFiles(r'D:\Documents\AirSim\Timages\ori')
    # RenameFiles(r'D:\Documents\AirSim\Timages\seg')
    # wash_data(r'D:\Documents\AirSim\Timages\seg')
    # show_classes(r'D:\Documents\AirSim\Timages\anno')
    temp('./Annotations')

    # SortToVOC(r'd:\Documents\AirSim\Timages')
