import glob
import os
import shutil
import tqdm
import random
import numpy as np
import cv2
from xml.etree import ElementTree as ET
from collections import Counter
import xml.dom.minidom

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
           [1250, 420], # meter type
           [185, 220], # top left color bar, distance to the meter
           [185, 2350], # top right color bar, yaw
           [1250, 2350]] # bottom right color bar, pitch

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass

def SortFiles(source_path, target_path):
    All_fileslist = glob.glob(os.path.join(source_path, '*.jpg'))
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

def RenameFiles(path, last_str):
    All_fileslist = glob.glob(os.path.join(path, '*.*[png|jpg|jpeg|xml]'))
    All_fileslist.sort()
    for file in All_fileslist:
        new_filename = os.path.join(file.rsplit('\\', 1)[0], str(All_fileslist.index(file)).zfill(5)+last_str)
        os.rename(file, new_filename)
    print(path + ' Done!')

def SortToVOC(path):
    random.seed(55)
    anno_list = glob.glob(os.path.join(path, 'anno', '*.xml'))
    img_list = glob.glob(os.path.join(path, 'ori', '*.jpg'))
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

def sort_real_data(source_path, target_path):
    xml_list = glob.glob(os.path.join(source_path, '*', '*.xml'))
    makedirs(os.path.join(target_path, 'anno'))
    makedirs(os.path.join(target_path, 'ori'))
    for each_xml in tqdm.tqdm(xml_list):
        xml_name = each_xml.rsplit('\\')[-1]
        shutil.copyfile(each_xml, os.path.join(target_path, 'anno', xml_name))
        each_jpg = each_xml.replace('xml', 'JPG')
        jpg_name = each_jpg.rsplit('\\')[-1]
        shutil.copyfile(each_jpg, os.path.join(target_path, 'ori', jpg_name))
    print('Done!')

def judge_data(np_img):
    flag = True
    img_higher = np_img[coords[0][0], coords[0][1]]
    img_lower = np_img[coords[1][0], coords[1][1]]
    img_meter = np_img[coords[2][0], coords[2][1]]
    img_distance = np_img[coords[3][0], coords[3][1]]
    img_yaw = np_img[coords[4][0], coords[4][1]]
    img_pitch = np_img[coords[5][0], coords[5][1]]
    judge_list = [img_higher, img_lower, img_meter, img_distance, img_yaw, img_pitch]
    for condition in judge_list:
        if not condition.tolist() in num_colors:
            flag = False
    if not flag:
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
    print(len(bad_datalist))
    # print(bad_datalist)
    for bad_data in bad_datalist:
        try:
            os.remove(bad_data)
            os.remove(bad_data.replace('seg', 'ori'))
            # os.remove(bad_data.replace('seg', 'anno').replace('jpg', 'xml'))
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
    all_img_list = glob.glob(os.path.join(path, '*', 'ori', '*.jpg'))
    all_img_list.sort()
    all_seg_list = glob.glob(os.path.join(path, '*', 'seg', '*.jpg'))
    all_seg_list.sort()
    all_anno_list = glob.glob(os.path.join(path, '*', 'anno', '*.xml'))
    all_anno_list.sort()
    print('img: {}, anno: {}'.format(len(all_img_list), len(all_anno_list)))
    target_path = os.path.join(path, 'conbined_data')
    makedirs(os.path.join(target_path, 'ori'))
    makedirs(os.path.join(target_path, 'seg'))
    makedirs(os.path.join(target_path, 'anno'))
    for each_img in tqdm.tqdm(all_img_list):
        shutil.copyfile(each_img, os.path.join(target_path, 'ori', str(all_img_list.index(each_img)).zfill(5)+'.jpg'))
    for each_img in tqdm.tqdm(all_seg_list):
        shutil.copyfile(each_img, os.path.join(target_path, 'seg', str(all_seg_list.index(each_img)).zfill(5)+'.jpg'))
    for each_anno in tqdm.tqdm(all_anno_list):
        shutil.copyfile(each_anno, os.path.join(target_path, 'anno', str(all_anno_list.index(each_anno)).zfill(5)+'.xml'))
    print('Done!')

def trans_name(path):
    all_files = glob.glob(os.path.join(path, '*.jpg'))
    print(len(all_files))
    for each_file in all_files:
        os.rename(each_file, each_file.replace('jpg', 'xml'))
    print('Done!')

def temp_txt(path):
    all_files = glob.glob(os.path.join(path, '*', '*.xml'))
    with open(os.path.join(path, 'test.txt'), 'w') as f:
        for test_img in all_files:
            test_img_name = test_img.rsplit('\\')[-1].rsplit('.')[0]
            f.write(test_img_name)
            f.write('\n')

def gen_each_train_test(path):
    img_list = glob.glob(os.path.join(path, 'ori', '*.jpg'))
    print(len(img_list))
    img_list.sort()
    train_img_list = img_list[:1500]
    test_img_list = img_list[1500:]
    train_img_list.sort()
    test_img_list.sort()
    with open(os.path.join(path, 'trainval.txt'), 'w') as f:
        for train_img in train_img_list:
            train_img_name = train_img.rsplit('\\')[-1].rsplit('.')[0]
            f.write(train_img_name)
            f.write('\n')
    with open(os.path.join(path, 'test.txt'), 'w') as f:
        for test_img in test_img_list:
            test_img_name = test_img.rsplit('\\')[-1].rsplit('.')[0]
            f.write(test_img_name)
            f.write('\n')

def get_test_set(list_file, ori_path, seg_path, anno_path, target_path):
    makedirs(os.path.join(target_path, 'ori'))
    makedirs(os.path.join(target_path, 'seg'))
    makedirs(os.path.join(target_path, 'anno'))
    test_file_list = []
    with open(list_file, 'r') as f:
        for line in f:
            test_file_list.append(line.strip())
    print(test_file_list[0])
    ori_list = glob.glob(os.path.join(ori_path, '*.jpg'))
    seg_list = glob.glob(os.path.join(seg_path, '*.jpg'))
    anno_list = glob.glob(os.path.join(anno_path, '*.xml'))
    for each_name in tqdm.tqdm(test_file_list):
        for each_img in ori_list:
            if '/'+each_name in each_img:
                shutil.copyfile(each_img, os.path.join(target_path, 'ori', each_name+'.jpg'))
        for each_img in seg_list:
            if '/'+each_name in each_img:
                shutil.copyfile(each_img, os.path.join(target_path, 'seg', each_name+'.jpg'))
        for each_img in anno_list:
            if '/'+each_name in each_img:
                shutil.copyfile(each_img, os.path.join(target_path, 'anno', each_name+'.xml'))

def gen_voc_txt(dataset_path, new_test_path):
    all_test_txt = glob.glob(os.path.join(dataset_path, '*', '*', 'test.txt'))
    all_trainval_txt = glob.glob(os.path.join(dataset_path, '*', '*', 'trainval.txt'))
    with open(os.path.join(new_test_path, 'all_test.txt'), 'w') as fw:
        for each_txt in all_test_txt:
            each_test_path = os.path.join('biaoji_data_v4', each_txt.split('\\')[-3], each_txt.split('\\')[-2])
            with open(each_txt, 'r') as fr:
                for line in fr:
                    fw.write(os.path.join(each_test_path, line))
    with open(os.path.join(new_test_path, 'all_trainval.txt'), 'w') as fw:
        for each_txt in all_trainval_txt:
            each_test_path = os.path.join('biaoji_data_v4', each_txt.split('\\')[-3], each_txt.split('\\')[-2])
            with open(each_txt, 'r') as fr:
                for line in fr:
                    fw.write(os.path.join(each_test_path, line))

def copy_all_xml(dataset_path):
    all_xml_list = glob.glob(os.path.join(dataset_path, '*', '*', 'anno', '*.xml'))
    print(len(all_xml_list))
    for each_xml in tqdm.tqdm(all_xml_list):
        shutil.copyfile(each_xml, each_xml.replace('anno', 'ori'))

def wash_xml(anno_list):
    bad_list = []
    for each_anno in tqdm.tqdm(anno_list):
        dom = xml.dom.minidom.parse(each_anno)
        root = dom.documentElement
        xmin = root.getElementsByTagName('xmin')
        if len(xmin) == 0:
            bad_list.append(each_anno)
            continue
        if xmin[0].firstChild.data == str(0):
            bad_list.append(each_anno)
    print(len(bad_list))
    print(bad_list)
    for each_bad in bad_list:
        os.remove(each_bad)
        os.remove(each_bad.replace('anno', 'ori').replace('xml', 'jpg'))
        os.remove(each_bad.replace('anno', 'seg').replace('xml', 'jpg'))

def del_file(path):
    ori_list = glob.glob(os.path.join(path, 'ori', '*.jpg'))
    seg_list = glob.glob(os.path.join(path, 'seg', '*.jpg'))
    anno_list = glob.glob(os.path.join(path, 'anno', '*.xml'))
    ori_list.sort()
    seg_list.sort()
    anno_list.sort()
    print(len(ori_list), len(seg_list), len(anno_list))
    for file in ori_list[2100:]:
        os.remove(file)
    for file in seg_list[2100:]:
        os.remove(file)
    for file in anno_list[2100:]:
        os.remove(file)
    ori_list = glob.glob(os.path.join(path, 'ori', '*.jpg'))
    seg_list = glob.glob(os.path.join(path, 'seg', '*.jpg'))
    anno_list = glob.glob(os.path.join(path, 'anno', '*.xml'))
    print(len(ori_list), len(seg_list), len(anno_list))


if __name__ == '__main__':
    # SortFiles(r'D:\Documents\AirSim\images', r'D:\Documents\AirSim\Timages')
    # RenameFiles(r'D:\Documents\AirSim\Timages\ori', '.jpg')
    # RenameFiles(r'D:\Documents\AirSim\Timages\seg', '.jpg')
    # sort_real_data(r'C:\Users\Administrator\Desktop\变电站巡检项目\本地资料\表计照片', r'C:\Users\Administrator\Desktop\biaoji')
    # wash_data(r'E:\Timages\seg')
    # show_classes(r'D:\Documents\AirSim\biaoji_v2\biaoji_B\anno')
    # temp('./Annotations')
    # random.seed(55)
    # for i in range(9):
    #     gen_each_train_test(r'd:\Documents\AirSim\Biaoji_Dataset\biaoji_E\weather' + str(i))
    # list_file = '/data3/home_jinlukang/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
    # ori_path = '/data3/home_jinlukang/data/VOCdevkit/VOC2007/JPEGImages'
    # anno_path = '/data3/home_jinlukang/data/VOCdevkit/VOC2007/Annotations'
    # seg_path = '/data3/home_jinlukang/data/biaoji_data_v3/conbined_data/seg'
    # target_path = '/data3/home_jinlukang/data/biaoji_data_v3/test_set'
    # get_test_set(list_file, ori_path, seg_path, anno_path, target_path)

    # SortToVOC(r'd:\Documents\AirSim\Timages')
    dataset_path = r'd:\Documents\AirSim\Biaoji_Dataset'
    new_txt_path = r'd:\Documents\AirSim\Biaoji_Dataset'
    gen_voc_txt(dataset_path, new_txt_path)
    # copy_all_xml(dataset_path)
    # conbine_data(r'd:\Documents\AirSim\Biaoji_Dataset\111')
    # path = r'd:\Documents\AirSim\Biaoji_Dataset\biaoji_E'
    # a = glob.glob(os.path.join(path, r'*\anno\*.xml'))
    # wash_xml(a)

    # del_file(r'd:\Documents\AirSim\Biaoji_Dataset\biaoji_C\weather0')
    # del_file(r'd:\Documents\AirSim\Biaoji_Dataset\biaoji_C\weather1')
    # del_file(r'd:\Documents\AirSim\Biaoji_Dataset\biaoji_C\weather2')
    # del_file(r'd:\Documents\AirSim\Biaoji_Dataset\biaoji_C\weather3')
    # del_file(r'd:\Documents\AirSim\Biaoji_Dataset\biaoji_C\weather4')
    # del_file(r'd:\Documents\AirSim\Biaoji_Dataset\biaoji_C\weather5')
    # del_file(r'd:\Documents\AirSim\Biaoji_Dataset\biaoji_C\weather6')
    # del_file(r'd:\Documents\AirSim\Biaoji_Dataset\biaoji_C\weather7')
    # del_file(r'd:\Documents\AirSim\Biaoji_Dataset\biaoji_C\weather8')


    # path_list = glob.glob(r'd:\Documents\AirSim\biaoji_D\sorted/*')
    # print(path_list)
    # for path in path_list:
    #     del_file(path)
    #     anno_list = glob.glob(os.path.join(path, 'anno/*.xml'))
    #     wash_xml(anno_list)

    # anno_path_list = glob.glob(r'd:\Documents\AirSim\Biaoji_Dataset\biaoji_C/*/anno')
    # anno_path_list.sort()
    # print(anno_path_list)
    # for path in anno_path_list:
    #     RenameFiles(path, '.xml')
    #
    # anno_path_list = glob.glob(r'd:\Documents\AirSim\Biaoji_Dataset\biaoji_C/*/seg')
    # anno_path_list.sort()
    # print(anno_path_list)
    # for path in anno_path_list:
    #     RenameFiles(path, '.jpg')
    #
    # anno_path_list = glob.glob(r'd:\Documents\AirSim\Biaoji_Dataset\biaoji_C/*/ori')
    # anno_path_list.sort()
    # print(anno_path_list)
    # for path in anno_path_list:
    #     RenameFiles(path, '.jpg')