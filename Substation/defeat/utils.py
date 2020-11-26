import glob
import os
import shutil
import tqdm
import random
import numpy as np
import cv2
import json
from PIL import Image
from xml.etree import ElementTree as ET
from collections import Counter
import xml.dom.minidom

num_colors = [[216, 42, 196],  # 0
              [248, 132, 234],  # 1
              [252, 115, 56],  # 2
              [155, 154, 125],  # 3
              [240, 252, 211],  # 4
              [158, 69, 202],  # 5
              [176, 218, 180],  # 6
              [228, 155, 83],  # 7
              [246, 233, 185],  # 8
              [53, 148, 117],  # 9
              [217, 86, 122],  # 10
              [204, 207, 253],  # 11
              [201, 176, 233]]  # 12

coords = [[1250, 220], # bottom left color bar, the percentage of the reading, higher bit [x,y]
           [1250, 320], # lower bit
           [1250, 420], # meter type
           [185, 220], # top left color bar, distance to the meter
           [185, 2350], # top right color bar, yaw
           [1250, 2350]] # bottom right color bar, pitch

abnormal_dict = {'normal': [216, 42, 196], 'abnormal': [248, 132, 234]}

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
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

def show_classes(anno_path):
    # 显示数据集中的各个表计的类别
    anno_list = glob.glob(os.path.join(anno_path, '*.xml'))
    class_list = []
    file_list = []
    for each_anno in anno_list:
        tree = ET.parse(each_anno)
        annotation = tree.getroot()
        for box in annotation.iter('object'):
            for attrib in box:
                if attrib.tag == 'name':
                    class_list.append(attrib.text)
                    file_list.append(each_anno)
    print(Counter(class_list))
    return list(set(file_list))

def combine_labels(ori_path, seg_path, anno_path, mode='ori'):
    # 将待测试的图像数据合并，后续通过labelimg工具进行查看
    limit_num = 100
    makedirs(ori_path.replace('ori', 'combined'))
    all_ori = glob.glob(os.path.join(ori_path, '*.jpg'))[:limit_num]
    all_seg = glob.glob(os.path.join(seg_path, '*.jpg'))[:limit_num]
    all_anno = glob.glob(os.path.join(anno_path, '*.xml'))[:limit_num]
    if mode=='ori':
        for each_ori in all_ori:
            shutil.copyfile(each_ori, each_ori.replace('ori', 'combined'))
        for each_anno in all_anno:
            shutil.copyfile(each_anno, each_anno.replace('anno', 'combined'))
    if mode=='seg':
        for each_seg in all_seg:
            shutil.copyfile(each_seg, each_seg.replace('seg', 'combined'))
        for each_anno in all_anno:
            shutil.copyfile(each_anno, each_anno.replace('anno', 'combined'))

def judge_data(np_img):
    flag = True
    img_status = np_img[coords[0][0], coords[0][1]]
    judge_list = [img_status]
    for condition in judge_list:
        if not condition.tolist() in list(abnormal_dict.values()):
            flag = False
    return flag

def file_filter(ori_path, seg_path, anno_path):
    # 过滤无法使用的数据
    bad_filelist = []
    seg_list = glob.glob(os.path.join(seg_path, '*.*[png|jpg|jpeg]'))
    for each_seg in tqdm.tqdm(seg_list):
        each_seg_numpy = cv2.imread(each_seg)
        each_seg_numpy = cv2.cvtColor(each_seg_numpy, cv2.COLOR_BGR2RGB)
        if not judge_data(each_seg_numpy):
            bad_filelist.append(each_seg)
    print('bad file num : {}/{}'.format(len(bad_filelist), len(seg_list)))
    print(bad_filelist)
    # for bad_file in bad_filelist:
    #     try:
    #         os.remove(bad_file)
    #         os.remove(bad_file.replace('seg', 'ori'))
    #     except:
    #         print('{} not exist'.format(bad_file))

def encode_img(ori_path, target_path):
    img = Image.open(ori_path)
    img = img.convert("RGB")
    img.save(target_path, quality=95)

def encode_folder_img(ori_path):
    target_folder = ori_path.replace('weather', 'weather_low')
    makedirs(os.path.join(target_folder, 'ori'))
    img_path_list = glob.glob(os.path.join(ori_path, 'ori', '*.*[png|jpg|jpeg]'))
    for img_path in tqdm.tqdm(img_path_list):
        target_img_path = img_path.replace('weather', 'weather_low')
        # print(target_img_path, img_path)
        encode_img(img_path, target_img_path)
    
def count_folder_images(seg_path, distance_res, yaw_res, pitch_res):
    seg_list = glob.glob(os.path.join(seg_path, '*.*[png|jpg|jpeg]'))
    for seg in tqdm.tqdm(seg_list):
        seg_img = cv2.imread(seg)
        seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
        try:
            distance_res.append(num_colors.index(seg_img[coords[3][0], coords[3][1]].tolist()))
            yaw_res.append(num_colors.index(seg_img[coords[4][0], coords[4][1]].tolist()))
            pitch_res.append(num_colors.index(seg_img[coords[5][0], coords[5][1]].tolist()))
        except:
            distance_res.append(-1)
            yaw_res.append(-1)
            pitch_res.append(-1)
    # print(Counter(distance_res))
    # print(Counter(yaw_res))
    # print(Counter(pitch_res))
    # return distance_res, yaw_res, pitch_res
        
def copy_seg_anno(file_path):
    target_folder = file_path.replace('weather', 'weather_low')
    makedirs(os.path.join(target_folder, 'seg'))
    makedirs(os.path.join(target_folder, 'anno'))
    seg_paths = glob.glob(os.path.join(file_path, 'seg', '*.*[png|jpg|jpeg]'))
    anno_paths = glob.glob(os.path.join(file_path, 'anno', '*.xml'))
    for seg_path in tqdm.tqdm(seg_paths):
        shutil.copyfile(seg_path, seg_path.replace('weather', 'weather_low'))
    for anno_path in tqdm.tqdm(anno_paths):
        shutil.copyfile(anno_path, anno_path.replace('weather', 'weather_low'))


if __name__ == "__main__":
    # 数据分组整理
    # items = ['weather2', 'weather6'] # ['bjbmyw', 'jsxs', 'bjmh', 'gjbs', 'hxqyfywyc', 'tgywjyc', 'xmbhyc']
    # for item in items:
    #     source_path = r'd:\Documents\AirSim\defeat_test\\' + item
    #     target_path = r'd:\Documents\AirSim\defeat_test_sorted\\' + item
    #     SortFiles(source_path, target_path)
    #     RenameFiles(target_path+'\\ori', '.jpg')
    #     RenameFiles(target_path+'\\seg', '.jpg')

    # 数据压缩
    defects = ['hxqyfywyc_sorted']# ['bjmh_sorted', 'bjbmyw_sorted', 'gjbs_sorted', 'gkxfw_sorted', 'hxqyfywyc_sorted', 'jsxs_sorted', 'jyzlw_sorted', 'jyzpl_sorted', 'tgywjyc_sorted', 'xmbhyc_sorted']
    for defect in defects:
        items = ['weather0']# ['weather0', 'weather1', 'weather2', 'weather3', 'weather4', 'weather5', 'weather6', 'weather7', 'weather8'] # ['bjbmyw', 'jsxs', 'bjmh', 'gjbs', 'hxqyfywyc', 'tgywjyc', 'xmbhyc']
        for item in items:
            ori_path = os.path.join(r'd:\Documents\AirSim', defect, item)
            encode_folder_img(ori_path)
            copy_seg_anno(ori_path)

    # 数据统计
    # items = ['weather0', 'weather1', 'weather2', 'weather3', 'weather4', 'weather5', 'weather6', 'weather7', 'weather8'] # ['bjbmyw', 'jsxs', 'bjmh', 'gjbs', 'hxqyfywyc', 'tgywjyc', 'xmbhyc']
    # distance_res = []
    # yaw_res = []
    # pitch_res = []
    # for item in items:        
    #     seg_path = os.path.join(r'G:\2020\Defect_Dataset_V1\xmbhyc_sorted', item, 'seg')
    #     count_folder_images(seg_path, distance_res, yaw_res, pitch_res)
    # print(Counter(distance_res))
    # print(Counter(yaw_res))
    # print(Counter(pitch_res))

    # seg_path = r'G:\2020\Defect_Dataset_V1\xmbhyc_sorted\weather0\seg'
    # count_folder_images(seg_path)

    # defects = ['xmbhyc_sorted', 'tgywjyc_sorted', 'jyzpl_sorted', 'jyzlw_sorted', 'jsxs_sorted', 'hxqyfywyc_sorted', 'gkxfw_sorted', 'gjbs_sorted', 'bjmh_sorted', 'bjbmyw_sorted']
    # items = ['weather0', 'weather1', 'weather2', 'weather3', 'weather4', 'weather5', 'weather6', 'weather7', 'weather8']
    # result_dict = {}
    # for defect in defects:
    #     defect_dict = {}
    #     distance_res = []
    #     yaw_res = []
    #     pitch_res = []
    #     for item in items:        
    #         seg_path = os.path.join(r'G:\2020\Defect_Dataset_V1', defect, item, 'seg')
    #         count_folder_images(seg_path, distance_res, yaw_res, pitch_res)
    #     defect_dict['distance'] = dict(Counter(distance_res))
    #     defect_dict['yaw'] = dict(Counter(yaw_res))
    #     defect_dict['pitch'] = dict(Counter(pitch_res))
    #     result_dict[defect] = defect_dict
    #     print(Counter(distance_res))
    #     print(Counter(yaw_res))
    #     print(Counter(pitch_res))
    # with open('./result.json', 'w') as f:
    #     json.dump(result_dict, f)

    # 数据标注验证
    # item = 'gkxfw'
    # ori_path = '/Users/lukangjin/Desktop/SubstationDataset/Dataset/sorted/' + item + '/ori'
    # seg_path = '/Users/lukangjin/Desktop/SubstationDataset/Dataset/sorted/' + item + '/seg'
    # anno_path = '/Users/lukangjin/Desktop/SubstationDataset/Dataset/sorted/' + item + '/anno'
    # combine_labels(ori_path, seg_path, anno_path)

    # 数据清洗
    # item = 'gjbs'
    # ori_path = r'D:\Documents\AirSim\defeat_test\\' + item + '\\ori'
    # seg_path = r'D:\Documents\AirSim\defeat_test\\' + item + '\\seg'
    # anno_path = r'D:\Documents\AirSim\defeat_test\\' + item + '\\anno'
    # file_filter(ori_path, seg_path, anno_path)