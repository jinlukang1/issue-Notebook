import json
import os
import numpy as np
import cv2
import glob
import tqdm
from xml.etree import ElementTree as ET

color_list = {'p1': [242,201,197], 'p2': [199,125,143], 'p3': [0,126,138], 'p4': [183,140,93], 'p5': [227,96,50],
              'p6': [59,246,53], 'p7': [141,238,180], 'p8': [121,160,208], 'p9': [205,228,85], 'p10': [249,173,199],
              'p11': [151,202,240]}

num_colors = [[216,42,196], [248,132,234], [252,115,56], [155,154,125], [240,252,211],
            [158,69,202], [176,218,180], [228,155,83], [246,233,185], [53,148,117], [217,86,122],
            [204,207,253], [201,176,233]]

coords = [[1250, 220], [1250, 320]]

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(path + ' is maked!')

def GetIndications(seg_path):
    seg_img = cv2.imread(seg_path)
    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
    higher_bit = seg_img[coords[0][0], coords[0][1]]
    lower_bit = seg_img[coords[1][0], coords[1][1]]
    if higher_bit.tolist() in num_colors:
        higher = num_colors.index(higher_bit.tolist()) * 10
    else:
        higher = -1
    if lower_bit.tolist() in num_colors:
        lower = num_colors.index(lower_bit.tolist())
    else:
        lower = -1
    return 100 - (higher + lower)

def GetPoints(np_img, colors_list):
    point_list = []
    for each_point in colors_list.keys():
        [rows, cols] = np.where(np_img[:, :, 2] == colors_list[each_point][-1])
        if len(rows) == 0:
            fin_row = -1
            fin_col = -1
        else:
            fin_col = int(np.mean(cols))
            fin_row = int(np.mean(rows))
        point_list.append([fin_row, fin_col])
    return point_list

def GetDetectImg(xml_path, seg_path, ori_path):
    seg_img = cv2.imread(seg_path)
    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
    ori_img = cv2.imread(ori_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    tree = ET.parse(xml_path)
    annotation = tree.getroot()
    xmin, ymin, xmax, ymax = 0, 0, 0, 0
    for box in annotation:
        if box.tag == 'object':
            for bndbox in box:
                if bndbox.tag == 'bndbox':
                    for point in bndbox:
                        if point.tag == 'xmin':
                            xmin = int(point.text)
                        if point.tag == 'ymin':
                            ymin = int(point.text)
                        if point.tag == 'xmax':
                            xmax = int(point.text)
                        if point.tag == 'ymax':
                            ymax = int(point.text)
    if xmax == 0:
        raise ValueError
    else:
        return seg_img[ymin:ymax, xmin:xmax], ori_img[ymin:ymax, xmin:xmax]

def GenJson(indications, point_list, ori_path, detect_ori, json_path):
    h, w, c = detect_ori.shape
    data_dict = {}
    data_dict['imagePath'] = ori_path.rsplit('\\', 1)[1]
    data_dict['imageData'] = None
    data_dict['indications'] = indications
    data_dict['shapes'] = []
    data_dict['imageWidth'] = w
    data_dict['imageHeight'] = h
    data_dict['flags'] = {}
    data_dict['lineColor'] = [0, 255, 0, 128]
    data_dict['fillColor'] = [255, 0, 0, 128]
    data_dict['version'] = '3.16.7'
    for each_point in point_list:
        temp_dict = {}
        temp_dict['line_color'] = None
        temp_dict['shape_type'] = 'polygon'
        temp_dict['points'] = each_point
        temp_dict['flags'] = {}
        temp_dict['fill_color'] = None
        temp_dict['label'] = 'p'+str(point_list.index(each_point))
        data_dict['shapes'].append(temp_dict)
    file_name = ori_path.rsplit('\\', 1)[1].replace('png', 'json')
    with open(os.path.join(json_path, file_name), 'w') as f:
        f.write(json.dumps(data_dict, sort_keys=True, indent=4, separators=(',', ':')))

def GenDatasetAnno(dataset_path):
    seg_file_list = glob.glob(os.path.join(dataset_path, 'seg', '*.png'))
    makedirs(os.path.join(dataset_path, 'anno2'))
    makedirs(os.path.join(dataset_path, 'ori2'))
    for seg_path in tqdm.tqdm(seg_file_list):
        xml_path = seg_path.replace('seg', 'anno').replace('png', 'xml')
        ori_path = seg_path.replace('seg', 'ori')
        new_ori_path = seg_path.replace('seg', 'ori2')
        json_path = seg_path.replace('seg', 'anno2').replace('png', 'json').rsplit('\\', 1)[0]
        detect_seg, detect_ori = GetDetectImg(xml_path, seg_path, ori_path)
        point_list = GetPoints(detect_seg, color_list)
        indications = GetIndications(seg_path)
        GenJson(indications, point_list, ori_path, detect_seg, json_path)
        detect_ori = cv2.cvtColor(detect_ori, cv2.COLOR_RGB2BGR)
        cv2.imwrite(new_ori_path, detect_ori)

    print('Done!')

if __name__ == '__main__':
    seg_path = './00011_seg.png'
    ori_path = './00011.png'
    xml_path = './00011.xml'
    json_path = './'

    # detect_seg, detect_ori = GetDetectImg(xml_path, seg_path, ori_path)
    # point_list = GetPoints(detect_seg, color_list)
    # indications = GetIndications(seg_path)
    # GenJson(indications, point_list, ori_path, detect_seg, json_path)
    GenDatasetAnno(r'd:\Documents\AirSim\biaoji_A')

    # detect_ori = cv2.cvtColor(detect_ori, cv2.COLOR_RGB2BGR)
    # show_img = detect_ori
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image', show_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()