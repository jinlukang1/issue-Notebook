import json
import os
import numpy as np
import cv2
import glob
import tqdm
from xml.etree import ElementTree as ET

color_list = {'p1': [242,201,197], 'p2': [199,125,143], 'p3': [0,126,138], 'p4': [183,140,93], 'p5': [227,96,50],
              'p6': [59,246,53], 'p7': [141,238,180], 'p8': [121,160,208], 'p9': [205,228,85], 'p10': [249,173,199],
              'p11': [170,71,248]}

num_colors = [[216,42,196], [248,132,234], [252,115,56], [155,154,125], [240,252,211],
            [158,69,202], [176,218,180], [228,155,83], [246,233,185], [53,148,117], [217,86,122],
            [204,207,253], [201,176,233]]

coords = [[1250, 220], # bottom left color bar, the percentage of the reading, higher bit [x,y]
           [1250, 320], # lower bit
           [1250, 420], # meter type
           [185, 220], # top left color bar, distance to the meter
           [185, 2350], # top right color bar, yaw
           [1250, 2350]] # bottom right color bar, pitch

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
        return -1
    if lower_bit.tolist() in num_colors:
        lower = num_colors.index(lower_bit.tolist())
    else:
        return -1
    return higher + lower

def GetOtherAnno(seg_path):
    seg_img = cv2.imread(seg_path)
    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
    meter = seg_img[coords[2][0], coords[2][1]]
    distance = seg_img[coords[3][0], coords[3][1]]
    yaw = seg_img[coords[4][0], coords[4][1]]
    pitch = seg_img[coords[5][0], coords[5][1]]
    if meter.tolist() in num_colors:
        meter_index = num_colors.index(meter.tolist())
    else:
        meter_index = -1
    if distance.tolist() in num_colors:
        distance_index = num_colors.index(distance.tolist())
    else:
        distance_index = -1
    if yaw.tolist() in num_colors:
        yaw_index = num_colors.index(yaw.tolist())
    else:
        yaw_index = -1
    if pitch.tolist() in num_colors:
        pitch_index = num_colors.index(pitch.tolist())
    else:
        pitch_index = -1

    return [meter_index, distance_index, yaw_index, pitch_index]

def GetPoints(np_img, colors_list, xmin, ymin):
    point_list = []
    for each_point in colors_list.keys():
        [rows, cols] = np.where(np_img[:, :, 2] == colors_list[each_point][-1])
        if len(rows) == 0:
            fin_row = -1
            fin_col = -1
        else:
            fin_col = int(np.mean(cols))
            fin_row = int(np.mean(rows))
        point_list.append([fin_col + xmin, fin_row + ymin])
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
        return seg_img[ymin:ymax, xmin:xmax], ori_img[ymin:ymax, xmin:xmax], xmin, ymin

def GenJson(indications, OtherAnnos, point_list, ori_path, detect_ori, json_path):
    h, w, c = detect_ori.shape
    data_dict = {}
    data_dict['imagePath'] = ori_path.rsplit('\\', 1)[1]
    data_dict['imageData'] = None
    data_dict['indications'] = indications
    data_dict['meter'] = OtherAnnos[0]
    data_dict['distance'] = OtherAnnos[1]
    data_dict['yaw'] = OtherAnnos[2]
    data_dict['pitch'] = OtherAnnos[3]
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
    file_name = ori_path.rsplit('\\', 1)[1].replace('jpg', 'json')
    with open(os.path.join(json_path, file_name), 'w') as f:
        f.write(json.dumps(data_dict, sort_keys=True, indent=4, separators=(',', ':')))

def GenDatasetAnno(dataset_path):
    seg_file_list = glob.glob(os.path.join(dataset_path, 'seg', '*.jpg'))
    makedirs(os.path.join(dataset_path, 'anno2'))
    makedirs(os.path.join(dataset_path, 'ori2'))
    makedirs(os.path.join(dataset_path, 'seg2'))
    for seg_path in tqdm.tqdm(seg_file_list):
        xml_path = seg_path.replace('seg', 'anno').replace('jpg', 'xml')
        ori_path = seg_path.replace('seg', 'ori')
        new_ori_path = seg_path.replace('seg', 'ori2')
        new_seg_path = seg_path.replace('seg', 'seg2')
        json_path = seg_path.replace('seg', 'anno2').replace('jpg', 'json').rsplit('\\', 1)[0]
        detect_seg, detect_ori, xmin, ymin = GetDetectImg(xml_path, seg_path, ori_path)
        point_list = GetPoints(detect_seg, color_list, xmin, ymin)
        indications = GetIndications(seg_path)
        OtherAnnos = GetOtherAnno(seg_path)
        GenJson(indications, OtherAnnos, point_list, ori_path, detect_seg, json_path)
        detect_ori = cv2.cvtColor(detect_ori, cv2.COLOR_RGB2BGR)
        detect_seg = cv2.cvtColor(detect_seg, cv2.COLOR_RGB2BGR)
        cv2.imwrite(new_ori_path, detect_ori)
        cv2.imwrite(new_seg_path, detect_seg)

    print('Done!')

if __name__ == '__main__':
    # seg_path = './00011_seg.png'
    # ori_path = './00011.png'
    # xml_path = './00011.xml'
    # json_path = './'

    # detect_seg, detect_ori = GetDetectImg(xml_path, seg_path, ori_path)
    # point_list = GetPoints(detect_seg, color_list)
    # indications = GetIndications(seg_path)
    # GenJson(indications, point_list, ori_path, detect_seg, json_path)
    # path_list = glob.glob(r'd:\Documents\AirSim\Biaoji_Dataset\*\*')
    # for each_path in path_list:
    #     GenDatasetAnno(each_path)
    GenDatasetAnno(r'd:\Documents\AirSim\Biaoji_Dataset\biaoji_C\weather0')

    # detect_ori = cv2.cvtColor(detect_ori, cv2.COLOR_RGB2BGR)
    # show_img = detect_ori
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image', show_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()