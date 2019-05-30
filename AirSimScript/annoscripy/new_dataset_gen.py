import os,shutil
import json
import cv2
import numpy as np
import json
import glob
import random
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

SEED = 705
random.seed(SEED)

roots = []


# roots.append(r"G:\dataset\BigFront_dataset")
roots.append(r"G:\dataset\Bus_dataset")
roots.append(r"G:\dataset\BigNewEnergy_dataset")
# roots.append(r"G:\dataset\Car_dataset")
roots.append(r"G:\dataset\NewEnergy_dataset")
roots.append(r"G:\dataset\Truck_dataset")

# roots.append(r"G:\reco_dataset\Truck_reco_dataset")
# roots.append(r"G:\reco_dataset\BigNewEnergy_reco_dataset")
# roots.append(r"G:\reco_dataset\Bus_reco_dataset")
# roots.append(r"G:\reco_dataset\BigFront_reco_dataset")
# roots.append(r"G:\reco_dataset\Car_reco_dataset")
# roots.append(r"G:\reco_dataset\NewEnergy_reco_dataset")


filename = "0"+str(np.random.randint(1,1500)).zfill(4)+".png"
# filename = "00555.png"

def cal(vertices_locations, x, y):
    vertices_x1 = vertices_locations['x1']
    vertices_x2 = vertices_locations['x2']
    vertices_x3 = vertices_locations['x3']
    vertices_x4 = vertices_locations['x4']
    vertices_y1 = vertices_locations['y1']
    vertices_y2 = vertices_locations['y2']
    vertices_y3 = vertices_locations['y3']
    vertices_y4 = vertices_locations['y4']
    R1 = vertices_x1+(vertices_x2-vertices_x1)*x/160.0
    R2 = vertices_x4+(vertices_x3-vertices_x4)*x/160.0
    x_fin = R1*(50-y)/50.0+R2*y/50.0
    C1 = vertices_y1+(vertices_y4-vertices_y1)*y/50.0
    C2 = vertices_y2+(vertices_y3-vertices_y2)*y/50.0
    y_fin = C1*(160-x)/160.0+C2*x/160.0
    return int(round(x_fin)), int(round(y_fin))


def location_transfer(vertices_locations={}, num_location={}):#为了抹消之前四舍五入的影响简单做了些数值上的修改
    num_x1 = num_location['x1']-1.5
    num_y1 = num_location['y1']-1.5
    num_x2 = num_location['x2']+1.5
    num_y2 = num_location['y2']+1.5
    num_x1_fin, num_y1_fin = cal(vertices_locations, num_x1, num_y1)
    num_x3_fin, num_y3_fin = cal(vertices_locations, num_x2, num_y2)
    num_x2_fin, num_y2_fin = cal(vertices_locations, num_x2, num_y1)
    num_x4_fin, num_y4_fin = cal(vertices_locations, num_x1, num_y2)
    new_dict = {'x1':num_x1_fin-0, 'y1':num_y1_fin-0, 'x2':num_x2_fin+0, 'y2':num_y2_fin-0, 
    'x3':num_x3_fin+0, 'y3':num_y3_fin+0, 'x4':num_x4_fin-0, 'y4':num_y4_fin+0}
    return new_dict

def gen_bbox(vertices_locations):
    vertices_x1 = vertices_locations['x1']
    vertices_x2 = vertices_locations['x2']
    vertices_x3 = vertices_locations['x3']
    vertices_x4 = vertices_locations['x4']
    vertices_y1 = vertices_locations['y1']
    vertices_y2 = vertices_locations['y2']
    vertices_y3 = vertices_locations['y3']
    vertices_y4 = vertices_locations['y4']
    new_dict = {'x1':min(vertices_x1, vertices_x4), 'y1':min(vertices_y1, vertices_y2), 
            'x2':max(vertices_x2, vertices_x3), 'y2':max(vertices_y4, vertices_y3)}
    return new_dict

def gen_chars_location(bbox_dict, chars_dict):
    chars_dict['x1'] = chars_dict['x1'] - bbox_dict['x1']
    chars_dict['x2'] = chars_dict['x2'] - bbox_dict['x1']
    chars_dict['x3'] = chars_dict['x3'] - bbox_dict['x1']
    chars_dict['x4'] = chars_dict['x4'] - bbox_dict['x1']
    chars_dict['y1'] = chars_dict['y1'] - bbox_dict['y1']
    chars_dict['y2'] = chars_dict['y2'] - bbox_dict['y1']
    chars_dict['y3'] = chars_dict['y3'] - bbox_dict['y1']
    chars_dict['y4'] = chars_dict['y4'] - bbox_dict['y1']
    return chars_dict

def makepath(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(path + ' is maked! ') 

def new_dataset_gen():#新数据集的生成
    for root in roots:
        light_dirs = os.listdir(root)
        for light_dir in light_dirs:
            weather_dirs = os.listdir(os.path.join(root,light_dir))
            for weather_dir in weather_dirs:
                all_annotations = glob.glob(os.path.join(root,light_dir,weather_dir)+'/new_annotations/*.json')
                all_imgs = glob.glob(os.path.join(root,light_dir,weather_dir)+'/raw/*.png')
                for each_annotation in tqdm(all_annotations):
                    with open(each_annotation, "r") as f:
                        data_annotation = f.read()
                        LP_dict = json.loads(data_annotation)
                        LP_dict['bbox'] = gen_bbox(LP_dict['vertices_locations'])
                        # print(LP_dict['bbox'])
                        for key in LP_dict.keys():
                            if 'num' in key and 'location' in key:
                                # LP_dict[key] = location_transfer(vertices_locations=LP_dict['vertices_locations'], num_location=LP_dict[key])
                                LP_dict[key] = gen_chars_location(bbox_dict=LP_dict['bbox'], chars_dict=LP_dict[key])
                                # print(LP_dict[key])
                        del LP_dict['vertices_locations']
                        del LP_dict['bbox']
                    new_annotation = each_annotation.replace('dataset', 'reco_dataset')
                    makepath(new_annotation[:-10])
                    with open(new_annotation, "w") as f:
                        f.write(json.dumps(LP_dict, sort_keys=True, indent=4, separators=(',', ':')))
                # for each_img in tqdm(all_imgs):
                #     with open(each_img.replace('raw', 'annotations').replace('png', 'json'), "r") as f:
                #         data_annotation = f.read()
                #         LP_dict = json.loads(data_annotation)
                #         LP_dict['bbox'] = gen_bbox(LP_dict['vertices_locations'])
                #     cv_img = cv2.imread(each_img)
                #     cropped_img = cv_img[LP_dict['bbox']['y1']:LP_dict['bbox']['y2'], LP_dict['bbox']['x1']:LP_dict['bbox']['x2']]
                #     new_img = each_img.replace('dataset', 'reco_dataset')
                #     makepath(new_img[:-9])                   
                #     cv2.imwrite(new_img, cropped_img)

                print(weather_dir+' is done!')

def old_anno_modify():#将原来的标注进行更改并存储
    for root in roots:
        light_dirs = os.listdir(root)
        for light_dir in light_dirs:
            weather_dirs = os.listdir(os.path.join(root,light_dir))
            for weather_dir in weather_dirs:
                all_annotations = glob.glob(os.path.join(root,light_dir,weather_dir)+'/annotations/*.json')
                all_imgs = glob.glob(os.path.join(root,light_dir,weather_dir)+'/raw/*.png')
                for each_annotation in tqdm(all_annotations):
                    makepath(os.path.join(root, light_dir, weather_dir, 'final_annotations'))
                    with open(each_annotation, "r") as f:
                        data_annotation = f.read()
                        LP_dict = json.loads(data_annotation)
                        LP_dict['bbox'] = gen_bbox(LP_dict['vertices_locations'])
                        for key in LP_dict.keys():
                            if 'num' in key and 'location' in key:
                                # LP_dict['old_'+key] = LP_dict[key]
                                LP_dict[key] = location_transfer(vertices_locations=LP_dict['vertices_locations'], num_location=LP_dict[key])
                    with open(each_annotation.replace('annotations', 'final_annotations'), "w") as f:
                        f.write(json.dumps(LP_dict, sort_keys=True, indent=4, separators=(',', ':')))


# def get_annotations():
#     for root in roots:
#         light_dirs = os.listdir(root)
#         for light_dir in light_dirs:
#             weather_dirs = os.listdir(os.path.join(root,light_dir))
#                 all_annotations = glob.glob(os.path.join(root,light_dir,weather_dir)+'/annotations/*.json')
#                 all_imgs = glob.glob(os.path.join(root,light_dir,weather_dir)+'/raw/*.png')
#                 for each_img in all_imgs:
#                     makepath(os.path.join(root, light_dir, weather_dir, 'final_annotations'))
#                     each_annotation = each_img.replace('reco_dataset', 'dataset').replace('png', 'json').replace('raw', 'new_annotations')
#                     with open(each_annotation, "r") as f:
#                         data_annotation = f.read()
#                         LP_dict = json.loads(data_annotation)

def sort_dataset():#数据集划分
    for root in roots:
        light_dirs = os.listdir(root)
        for light_dir in light_dirs:
            weather_dirs = os.listdir(os.path.join(root,light_dir))
            for weather_dir in weather_dirs:
                all_annotations = glob.glob(os.path.join(root,light_dir,weather_dir)+'/new_annotations/*.json')
                all_imgs = glob.glob(os.path.join(root,light_dir,weather_dir)+'/raw/*.png')
                for dir_path in ['train', 'val', 'test']:
                    makepath(os.path.join(root, light_dir, weather_dir, dir_path, 'new_annotations'))
                    makepath(os.path.join(root, light_dir, weather_dir, dir_path, 'raw'))
                random.shuffle(all_annotations)
                alllist_train = all_annotations[:int(0.4*len(all_annotations))]
                alllist_val = all_annotations[int(0.4*len(all_annotations)):int(0.6*len(all_annotations))]
                alllist_test = all_annotations[int(0.6*len(all_annotations)):]
                for each_train_anno in tqdm(alllist_train):
                    shutil.copyfile(each_train_anno, each_train_anno.replace('new_annotations', 'train/new_annotations'))
                    each_train_img = each_train_anno.replace('new_annotations', 'raw').replace('json', 'png')
                    shutil.copyfile(each_train_img, each_train_img.replace('raw', 'train/raw'))
                for each_val_anno in tqdm(alllist_val):
                    shutil.copyfile(each_val_anno, each_val_anno.replace('new_annotations', 'val/new_annotations'))
                    each_val_img = each_val_anno.replace('new_annotations', 'raw').replace('json', 'png')
                    shutil.copyfile(each_val_img, each_val_img.replace('raw', 'val/raw'))
                for each_test_anno in tqdm(alllist_test):
                    shutil.copyfile(each_test_anno, each_test_anno.replace('new_annotations', 'test/new_annotations'))
                    each_test_img = each_test_anno.replace('new_annotations', 'raw').replace('json', 'png')
                    shutil.copyfile(each_test_img, each_test_img.replace('raw', 'test/raw'))

if __name__ == '__main__':
    new_dataset_gen()