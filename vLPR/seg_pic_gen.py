import os, sys
# sys.path.insert(0, '/ghome/jinlk/lib')
import glob
import json
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm



root = r'C:\Users\Administrator\Desktop\vLPR experiment\Car_reco_dataset'
# all_annotations = glob.glob('Car_test_data/new_annotations/*.json')
# all_imgs = glob.glob('Car_test_data/raw/*.png')
# select_index = np.random.randint(1,500)

def cal(x0, y0, x1, y1, x2, y2):
    if (x2-x1)*(y0-y1)-(y2-y1)*(x0-x1) > 0:
        return True
    else:
        return False


def get_num_seg(num_locations, img, num_pixel):
    num_x1 = num_locations['x1']
    num_x2 = num_locations['x2']
    num_x3 = num_locations['x3']
    num_x4 = num_locations['x4']
    num_y1 = num_locations['y1']
    num_y2 = num_locations['y2']
    num_y3 = num_locations['y3']
    num_y4 = num_locations['y4']
    h, w, c = img.shape
    # print("h:{}, w:{}, c:{}".format(h, w, c))
    for hi in range(h):
        for wi in range(w):
            if cal(wi, hi, num_x1, num_y1, num_x2, num_y2) and cal(wi, hi, num_x2, num_y2, num_x3, num_y3) and cal(wi, hi, num_x3, num_y3, num_x4, num_y4) and cal(wi, hi, num_x4, num_y4, num_x1, num_y1):
                img[hi, wi, :] = num_pixel
                if wi+1 < w:
                    img[hi, wi+1, :] = num_pixel
                if hi+1 < h:
                    img[hi+1, wi, :] = num_pixel
                if hi-1 > 0:
                    img[hi-1, wi, :] = num_pixel
    return img

def get_num_pos(num_locations, img, num_pixel):
    num_x1 = num_locations['x1']
    num_x2 = num_locations['x2']
    num_x3 = num_locations['x3']
    num_x4 = num_locations['x4']
    num_y1 = num_locations['y1']
    num_y2 = num_locations['y2']
    num_y3 = num_locations['y3']
    num_y4 = num_locations['y4']
    h, w, c = img.shape
    # print("h:{}, w:{}, c:{}".format(h, w, c))
    for hi in range(h):
        for wi in range(w):
            if cal(wi, hi, num_x1, num_y1, num_x2, num_y2) and cal(wi, hi, num_x2, num_y2, num_x3, num_y3) and cal(wi, hi, num_x3, num_y3, num_x4, num_y4) and cal(wi, hi, num_x4, num_y4, num_x1, num_y1):
                img[hi, wi, :] = num_pixel
                if wi+1 < w:
                    img[hi, wi+1, :] = num_pixel
                if hi+1 < h:
                    img[hi+1, wi, :] = num_pixel
                if hi-1 > 0:
                    img[hi-1, wi, :] = num_pixel
    return img

def makepath(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(path + ' is maked! ') 

def main():
    Data_Type = ['train', 'test', 'val']
    for Type in Data_Type:
        all_annotations = glob.glob(root+"/*/*/"+Type+"/new_annotations/*.json")
        # all_imgs = glob.glob(root+"/*/*/"+Type+"/raw/*.json")
        for each_annotation in tqdm(all_annotations):
            with open(each_annotation, "r") as f:
                data_annotation = f.read()
                LP_dict = json.loads(data_annotation)
            # print(select_index)
            img_path = each_annotation.replace('new_annotations', 'raw').replace('json', 'png')
            img_seg_opencv = cv2.imread(img_path)
            img_seg_opencv[: ,:, :] = 255
            img_pos_opencv = img_seg_opencv.copy()
            for key in LP_dict.keys():
                if 'num' in key and 'location' in key:
                    img_seg_opencv = get_num_seg(LP_dict[key], img_seg_opencv, LP_dict['license_plate_number'][key[:4]])
                    img_pos_opencv = get_num_pos(LP_dict[key], img_pos_opencv, int(key.replace('num', '').replace('_locations', '')))
            h, w, c = img_seg_opencv.shape
            # img_seg_opencv = cv2.resize(img_seg_opencv,(5*w, 5*h),interpolation=cv2.INTER_CUBIC)
            img_pos_path = img_path.replace('raw', 'pos_anno')
            img_seg_path = img_path.replace('raw', 'seg_anno')
            makepath(img_seg_path[:-9])
            makepath(img_pos_path[:-9])
            
            cv2.imwrite(img_seg_path, img_seg_opencv)
            cv2.imwrite(img_pos_path, img_pos_opencv)

            # cv2.imshow("images", img_pos_opencv)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()