import cv2
import os
import math
from collections import Counter
import numpy as np
import glob
from skimage import measure
from pylab import *
import json
import pprint
import copy


def draw_line(img_OpenCV, num_location):
    cv2.line(img_OpenCV, (num_location['x1'], num_location['y1']),
        (num_location['x2'], num_location['y2']), (0,0,255), 1)
    cv2.line(img_OpenCV, (num_location['x2'], num_location['y2']),
        (num_location['x3'], num_location['y3']), (0,0,255), 1)
    cv2.line(img_OpenCV, (num_location['x3'], num_location['y3']),
        (num_location['x4'], num_location['y4']), (0,0,255), 1)
    cv2.line(img_OpenCV, (num_location['x4'], num_location['y4']),
        (num_location['x1'], num_location['y1']), (0,0,255), 1)

char_table = {178:0, 109:1, 15:2, 103:3, 149:4, 11:5, 78:6, 35:7, 151:8, 143:9, 243:10, 72:11, 160:12, 214:13, 167:14, 101:15, 53:16, 3:17, 186:18, 88:19, 71:20, 236:21, 215:22, 113:23,
                    47:24, 219:25, 57:26, 227:27, 145:28, 217:29, 250:30, 208:31, 224:32, 69:33} 
province_table = {211:34, 251:35, 22:36, 176:37, 90:38, 210:39, 112:40, 56:41, 209:42, 18:43, 189:44, 10:45, 68:46,
                    213:47, 19:48, 97:49, 237:50, 4:51, 32:52, 164:53, 241:54, 16:55, 86:56, 255:57, 44:58, 46:59, 119:60, 36:61, 106:62, 84:63, 14:64}
roots = []
roots.append(r"d:\Documents\AirSim\BigFront_dataset\daytime/")
right_num = 0
wrong_num = 0
# for root in roots:
#     dirnames = os.listdir(root)
#     for dirname in dirnames:
#         realdir = root + dirname
#         image_path = realdir + "/segment"
#         anno_path = realdir + "/annotations"
#         annolist = os.listdir(anno_path)
#         print(len(annolist))
#         namelist = os.listdir(image_path)
#         print(len(namelist))
        # pbar = ProgressBar().start()

# for file_index in range(len(namelist)):
image_path = r"G:\dataset\Car_dataset\daytime\sc1_foggy\segment"
image_name = "0"+str(np.random.randint(1,1000)).zfill(4)+".png"
# image_name = "00001.png"
print(image_name)
LP_dict = {}
result = []
# image_path = r"d:\Documents\AirSim\BigFront_dataset\daytime\sc1_daytime_heavysnow"
# image_name = "segment/0"+str(np.random.randint(1,1000)).zfill(4)+".png"
# image_name = "segment/00001.png"
image = cv2.imread(os.path.join(image_path, image_name))
# print(image.shape)
[rows, cols] = np.where(image[:, :, 2] == 89)
coordinate_min = 2560+1440
coordinate_max = 0

#find plate position
for i, (row, col) in enumerate(zip(rows, cols)):
    coordinate_add = row + col
    if coordinate_add < coordinate_min:
        min_index = i
        coordinate_min = coordinate_add
    if coordinate_max < coordinate_add:
        max_index = i
        coordinate_max = coordinate_add

x1, y1 = cols[min_index], rows[min_index]
x3, y3 = cols[max_index], rows[max_index]

image_shrink = image[:, :, :]
rotated_image =  np.rot90(image_shrink)
# print(rotated_image.shape)

coordinate_min = 2560+1440
coordinate_max = 0

[rows, cols] = np.where(rotated_image[:, :, 2] == 89)
for i, (row, col) in enumerate(zip(rows, cols)):
    coordinate_add = row + col
    if coordinate_add < coordinate_min:
        min_index = i
        coordinate_min = coordinate_add
    if coordinate_max < coordinate_add:
        max_index = i
        coordinate_max = coordinate_add
x2, y2 = image_shrink.shape[1] - rows[min_index] + 1, cols[min_index]
x4, y4 = image_shrink.shape[1] - rows[max_index] + 1, cols[max_index]

LP_dict['vertices_locations'] = {'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2, 'x3':x3, 'y3':y3, 'x4':x4, 'y4':y4}

#find char position
plate_x_min = min(LP_dict['vertices_locations']['x1'], LP_dict['vertices_locations']['x4'])
plate_x_max = max(LP_dict['vertices_locations']['x2'], LP_dict['vertices_locations']['x3'])
plate_y_min = min(LP_dict['vertices_locations']['y1'], LP_dict['vertices_locations']['y2'])
plate_y_max = max(LP_dict['vertices_locations']['y3'], LP_dict['vertices_locations']['y4'])
plate_angle_h_r = 1.0*(y3-y4)/(1.0*(x3-x4))
plate_angle_v_r = 1.0*(x3-x2)/(1.0*(y3-y2))
plate_angle_d = math.degrees(plate_angle_h_r)
# print(plate_angle_d)

plate_image = image[plate_y_min:plate_y_max, plate_x_min:plate_x_max, :]

shrink_image = plate_image.copy()
show_image = plate_image.copy()

#province char
for (piex, index) in province_table.items():
    [rows, cols] = np.where(shrink_image[:, :, 2] == piex)
    # print(len(rows))
    if len(rows) < 10:
        continue
    # print(piex)
    left, right = min(cols), max(cols)
    up, down = min(rows), max(rows)
    delta_y = (right-left)*math.tan(plate_angle_h_r)
    delta_y = delta_y/2.0
    delta_x = (down-up)*math.tan(plate_angle_v_r)
    delta_x = delta_x/2.0
    show_image[up: down+1, left: right+1] = index
    cmean = sum(cols) / len(cols)
    rmean = sum(rows) / len(rows)
    area = len(cols)
    result.append({'char':index, 'x1':int(round(left-1-delta_x))+plate_x_min, 'y1':int(round(up-1-delta_y))+plate_y_min, 'x2':int(round(right+1-delta_x))+plate_x_min, 'y2':int(round(up-1+delta_y))+plate_y_min, 
                                    'x3':int(round(right+1+delta_x))+plate_x_min, 'y3':int(round(down+1+delta_y))+plate_y_min, 'x4':int(round(left-1+delta_x))+plate_x_min, 'y4':int(round(down+1-delta_y))+plate_y_min,
                                     'cmean':cmean, 'rmean':rmean, 'area':area})
#other chars
for (piex, index) in char_table.items():
    [rows, cols] = np.where(shrink_image[:, :, 2] == piex)
    if len(rows) < 10:
        continue
    # print(piex)
    temp_image = np.zeros(shrink_image.shape)
    for p in range(len(rows)):
        temp_image[rows[p], cols[p]] = 1

    [L, num] = measure.label(temp_image, background = 0, connectivity = 2, return_num = True)
    # print(num)
    for q in range(num):
        [rows, cols] = np.where(L[:, :, 2] == q+1)
        left, right = min(cols), max(cols)
        up, down = min(rows), max(rows)
        delta_y = (right-left)*math.tan(plate_angle_h_r)
        delta_y = delta_y/2.0
        delta_x = (down-up)*math.tan(plate_angle_v_r)
        delta_x = delta_x/2.0
        show_image[up: down+1, left: right+1] = index
        cmean = sum(cols) / len(cols)
        rmean = sum(rows) / len(rows)
        area = len(cols)
        result.append({'char':index, 'x1':int(round(left-0-delta_x))+plate_x_min, 'y1':int(round(up-0-delta_y))+plate_y_min, 'x2':int(round(right+0-delta_x))+plate_x_min, 'y2':int(round(up-0+delta_y))+plate_y_min, 
                                    'x3':int(round(right+0+delta_x))+plate_x_min, 'y3':int(round(down+0+delta_y))+plate_y_min, 'x4':int(round(left-0+delta_x))+plate_x_min, 'y4':int(round(down+0-delta_y))+plate_y_min,
                                     'cmean':cmean, 'rmean':rmean, 'area':area})
#mod for truck
result_sorted1 = sorted(result, key=lambda tmp:tmp['rmean'])
result_sorted2 = result_sorted1[:2]
result_sorted2 = sorted(result_sorted2, key=lambda tmp:tmp['cmean'])
result_sorted3 = result_sorted1[2:]
result_sorted3 = sorted(result_sorted3, key=lambda tmp:tmp['cmean'])

result_sorted = result_sorted2+result_sorted3


print("result num:{}".format(len(result_sorted)))

LP_dict['license_plate_number'] = {'num1':result_sorted[0]['char'], 'num2':result_sorted[1]['char'], 'num3':result_sorted[2]['char'], 'num4':result_sorted[3]['char'],
                                   'num5':result_sorted[4]['char'], 'num6':result_sorted[5]['char'], 'num7':result_sorted[6]['char']}
LP_dict['num1_locations'] = {'x1':result_sorted[0]['x1'], 'y1':result_sorted[0]['y1'], 'x2':result_sorted[0]['x2'], 'y2':result_sorted[0]['y2'],
                             'x3':result_sorted[0]['x3'], 'y3':result_sorted[0]['y3'], 'x4':result_sorted[0]['x4'], 'y4':result_sorted[0]['y4']}
LP_dict['num2_locations'] = {'x1':result_sorted[1]['x1'], 'y1':result_sorted[1]['y1'], 'x2':result_sorted[1]['x2'], 'y2':result_sorted[1]['y2'],
                             'x3':result_sorted[1]['x3'], 'y3':result_sorted[1]['y3'], 'x4':result_sorted[1]['x4'], 'y4':result_sorted[1]['y4']}
LP_dict['num3_locations'] = {'x1':result_sorted[2]['x1'], 'y1':result_sorted[2]['y1'], 'x2':result_sorted[2]['x2'], 'y2':result_sorted[2]['y2'],
                             'x3':result_sorted[2]['x3'], 'y3':result_sorted[2]['y3'], 'x4':result_sorted[2]['x4'], 'y4':result_sorted[2]['y4']}
LP_dict['num4_locations'] = {'x1':result_sorted[3]['x1'], 'y1':result_sorted[3]['y1'], 'x2':result_sorted[3]['x2'], 'y2':result_sorted[3]['y2'],
                             'x3':result_sorted[3]['x3'], 'y3':result_sorted[3]['y3'], 'x4':result_sorted[3]['x4'], 'y4':result_sorted[3]['y4']}
LP_dict['num5_locations'] = {'x1':result_sorted[4]['x1'], 'y1':result_sorted[4]['y1'], 'x2':result_sorted[4]['x2'], 'y2':result_sorted[4]['y2'],
                             'x3':result_sorted[4]['x3'], 'y3':result_sorted[4]['y3'], 'x4':result_sorted[4]['x4'], 'y4':result_sorted[4]['y4']}
LP_dict['num6_locations'] = {'x1':result_sorted[5]['x1'], 'y1':result_sorted[5]['y1'], 'x2':result_sorted[5]['x2'], 'y2':result_sorted[5]['y2'],
                             'x3':result_sorted[5]['x3'], 'y3':result_sorted[5]['y3'], 'x4':result_sorted[5]['x4'], 'y4':result_sorted[5]['y4']}
LP_dict['num7_locations'] = {'x1':result_sorted[6]['x1'], 'y1':result_sorted[6]['y1'], 'x2':result_sorted[6]['x2'], 'y2':result_sorted[6]['y2'],
                             'x3':result_sorted[6]['x3'], 'y3':result_sorted[6]['y3'], 'x4':result_sorted[6]['x4'], 'y4':result_sorted[6]['y4']}

for key in LP_dict.keys():
    if 'locations' in key:
        draw_line(image, LP_dict[key])
print(LP_dict['license_plate_number'])
# h, w, c = image.shape

img_OpenCV = image[min(LP_dict['vertices_locations']['y1'], LP_dict['vertices_locations']['y2']):max(LP_dict['vertices_locations']['y3'], LP_dict['vertices_locations']['y4']),
min(LP_dict['vertices_locations']['x1'], LP_dict['vertices_locations']['x4']):max(LP_dict['vertices_locations']['x2'], LP_dict['vertices_locations']['x3'])]
print(LP_dict['vertices_locations']['x1'], LP_dict['vertices_locations']['x3'], LP_dict['vertices_locations']['y1'], LP_dict['vertices_locations']['y3'])
# img_OpenCV = img_OpenCV[LP_dict['vertices_locations']['y1']:LP_dict['vertices_locations']['y3'], LP_dict['vertices_locations']['x1']:LP_dict['vertices_locations']['x3']]
h, w, c = img_OpenCV.shape
img_OpenCV = cv2.resize(img_OpenCV,(5*w, 5*h),interpolation=cv2.INTER_CUBIC)

# img_OpenCV = cv2.resize(image,(1920, 1080),interpolation=cv2.INTER_CUBIC)
# img_OpenCV = image.copy()
cv2.namedWindow("Image") 
cv2.imshow("Image", img_OpenCV) 
cv2.waitKey (0)
cv2.destroyAllWindows()