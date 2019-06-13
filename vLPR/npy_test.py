import numpy as np
import os
import cv2
import glob
from collections import Counter
import json

def write_txt():
    root = r'C:\Users\Administrator\Desktop\vLPR experiment\Car_reco_dataset\noontime\sc1_sunny\test'
    file = 'new_annotations'

    charfile = os.path.join(root, file)

    result = []
    anno_list = glob.glob(charfile+'/*.json')
    anno_list.sort()

    for each_anno in anno_list:
        with open(each_anno, 'r') as f:
            annotation = f.read()
            LP_dict = json.loads(annotation)
        print(LP_dict['license_plate_number'])
        with open(each_anno.replace('json', 'txt').replace('new_annotations', 'char_txt'), 'w') as f:
            for key in LP_dict['license_plate_number'].keys():
                f.write(str(LP_dict['license_plate_number'][key]))
                f.write('\n')

def count_npy():
    result = []
    npyfile = r'C:\Users\Administrator\Desktop\vLPR experiment\npy_data\all_car_recorder_train_char.npy'
    char_np = np.load(npyfile)
    print('char shape:{}'.format(char_np.shape))
    for i in range(char_np.shape[0]):
        result.append(char_np[i][0][2])
    print(Counter(result).most_common())

# for idx in range(seg_list.shape[0]-10000):
#     seg_img = seg_list[idx, :, :]

    # for xi in range(seg_img.shape[0]):
    #     for yi in range(seg_img.shape[1]):
    #         pixel_list.append(seg_img[xi, yi])
            # if seg_img[xi, yi] == 255:
            #     seg_img[xi, yi] = 0
            # else:
            #     seg_img[xi, yi] += 1
# print(Counter(pixel_list).most_common())
    # cv2.namedWindow("Image") 
    # cv2.imshow("Image", seg_img) 
    # cv2.waitKey (0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    count_npy()