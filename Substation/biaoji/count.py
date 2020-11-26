import numpy as np
import matplotlib.pyplot as plt
import glob, os
from collections import Counter
import cv2
import tqdm

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

coords = [[1250, 220],  # bottom left color bar, the percentage of the reading, higher bit [x,y]
          [1250, 320],  # lower bit
          [1250, 420],  # meter type
          [185, 220],  # top left color bar, distance to the meter
          [185, 2350],  # top right color bar, yaw
          [1250, 2350]]  # bottom right color bar, pitch

def draw_hist(result_counter):
    print(result_counter)
    x, y = [], []
    for k, v in result_counter.items():
        x.append(k)
        y.append(v)
    plt.bar(x, y, color = 'blue', tick_label = x)
    plt.show()


def show_count(seg_list):
    high_bit_list, low_bit_list, meter_type_list, distance_type_list, yaw_type_list, pitch_type_list = [], [], [], [], [], []
    all_list = [high_bit_list, low_bit_list, meter_type_list, distance_type_list, yaw_type_list, pitch_type_list]
    for each_seg in tqdm.tqdm(seg_list):
        each_seg_numpy = cv2.imread(each_seg)
        each_seg_numpy = cv2.cvtColor(each_seg_numpy, cv2.COLOR_BGR2RGB)
        for i, each_list in enumerate(all_list):
            each_list.append(num_colors.index(each_seg_numpy[coords[i][0], coords[i][1]].tolist()))
    for each_list in all_list:
        result_counter = Counter(each_list)
        draw_hist(result_counter)

def show_weather(path):
    result_dict = {}
    weather_list = os.listdir(path)
    for each_weather in weather_list:
        result_dict[each_weather] = len(glob.glob(os.path.join(path, each_weather, 'seg\*.jpg')))
        print('{}:{}'.format(each_weather.split('\\')[-1], result_dict[each_weather]-1500))
    draw_hist(result_dict)


if __name__ == '__main__':
    # seg_path = r'd:\Documents\AirSim\Biaoji_Dataset\biaoji_D\*\seg'
    # seg_list = glob.glob(os.path.join(seg_path, '*.jpg'))
    # show_count(seg_list)

    path = r'd:\Documents\AirSim\Biaoji_Dataset\biaoji_D'
    show_weather(path)