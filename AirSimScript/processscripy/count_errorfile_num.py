#coding:utf-8
import os,shutil
import json
import random
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
# import pandas as pd
import numpy as np

SEED = 705
random.seed(SEED)

roots = []
# Target = "C:/Users/Administrator/Desktop/AirSimData/segment/"

char_table = {0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 10:"A", 11:"B",
 12:"C", 13:"D", 14:"E", 15:"F", 16:"G", 17:"H", 18:"J", 19:"K", 20:"L", 21:"M", 22:"N", 23:"P", 24:"Q",
  25:"R", 26:"S", 27:"T", 28:"U", 29:"V", 30:"W", 31:"X", 32:"Y", 33:"Z", 34:"皖", 35:"桂", 36:"贵", 37:"粤",
   38:"甘", 39:"京", 40:"冀", 41:"闽", 42:"渝", 43:"琼", 44:"吉", 45:"赣", 46:"豫", 47:"黑", 48:"湘", 49:"苏",
    50:"辽", 51:"蒙", 52:"宁", 53:"沪", 54:"浙", 55:"青", 56:"鄂", 57:"津", 58:"陕", 59:"新", 60:"鲁", 61:"云",
     62:"川", 63:"藏", 64:"晋", 66:"挂"}

# roots.append(r"d:\Documents\AirSim\BigFront_dataset\noontime/")
# roots.append(r"G:\dataset\BigFront_dataset\daytime/")
# roots.append(r"G:\dataset\BigFront_dataset\night/")
# roots.append(r"G:\dataset\BigFront_dataset\noontime/")

# roots.append(r"G:\dataset\BigNewEnergy_dataset\daytime/")
# roots.append(r"G:\dataset\BigNewEnergy_dataset\noontime/")
# roots.append(r"G:\dataset\BigNewEnergy_dataset\night/")

roots.append(r"G:\dataset\Bus_dataset\daytime/")
roots.append(r"G:\dataset\Bus_dataset\night/")
roots.append(r"G:\dataset\Bus_dataset\noontime/")

roots.append(r"G:\dataset\Truck_dataset\daytime/")
roots.append(r"G:\dataset\Truck_dataset\night/")
roots.append(r"G:\dataset\Truck_dataset\noontime/")

roots.append(r"G:\dataset\Car_dataset\daytime/")
roots.append(r"G:\dataset\Car_dataset\front/")
roots.append(r"G:\dataset\Car_dataset\night/")
roots.append(r"G:\dataset\Car_dataset\noontime/")

roots.append(r"G:\dataset\NewEnergy_dataset\daytime/")
roots.append(r"G:\dataset\NewEnergy_dataset\front/")
roots.append(r"G:\dataset\NewEnergy_dataset\night/")
roots.append(r"G:\dataset\NewEnergy_dataset\noontime/")

# roots.append(r"G:\dataset\NewEnergy_dataset\scene2\noontime/")
# roots.append(r"G:\dataset\NewEnergy_dataset\scene1\daytime/")
# roots.append(r"G:\dataset\NewEnergy_dataset\scene1\front/")
# roots.append(r"G:\dataset\NewEnergy_dataset\scene1\night/")
# roots.append(r"G:\dataset\NewEnergy_dataset\scene1\noontime/")
# roots.append(r"G:\dataset\NewEnergy_dataset\scene3/")
# roots.append(r"G:\dataset\gongjiao_dataset\scene1\daytime/")
# roots.append(r"G:\dataset\gongjiao_dataset\scene1\noontime/")
# roots.append(r"G:\dataset\gongjiao_dataset\scene1\night/")
# roots.append(r"G:\dataset\gongjiao_dataset\scene3\daytime/")
# roots.append(r"G:\dataset\gongjiao_dataset\scene3\noontime/")
# roots.append(r"G:\dataset\gongjiao_dataset\scene3\night/")
# roots.append(r"G:\dataset\gua_dataset\scene1\daytime/")
# roots.append(r"G:\dataset\gua_dataset\scene1\noontime/")
# roots.append(r"G:\dataset\gua_dataset\scene1\night/")
# roots.append(r"G:\dataset\gua_dataset\scene3\daytime/")
# roots.append(r"G:\dataset\gua_dataset\scene3\noontime/")
# roots.append(r"G:\dataset\gua_dataset\scene3\night/")





def error_count(roots=[]):
    error_files_count = 0
    error_files = []
    error_seg_files = []
    for root in roots:
        dirnames = os.listdir(root)
        for dirname in dirnames:
            count_num = 0
            realdir = root + dirname
            alllist1 = os.listdir(realdir + "/new_annotations/")
            print(len(alllist1))
            alllist2 = os.listdir(realdir + "/raw/")
            print(len(alllist2))
            for segment_pic in alllist2:
                if not segment_pic[:5]+".json" in alllist1:
                    # print(os.path.join(root,dirname,"segment",segment_pic))
                    error_files.append(os.path.join(root,dirname,"raw",segment_pic))
                    error_seg_files.append(os.path.join(root,dirname,"segment",segment_pic))
                    count_num += 1
            # print(error_files)
            if count_num == len(alllist2) - len(alllist1):
                print(dirname + " is right")
            else:
                print(dirname + " is wrong")
            error_files_count += count_num
    print("error file number is " + str(error_files_count))
    return error_files, error_seg_files

def distance_count(roots=[]):
    distances = []
    for root in roots:
        dirnames = os.listdir(root)
        for dirname in dirnames:
            realdir = os.path.join(root, dirname)
            alllist = os.listdir(realdir + "/annotations/")
            # random.shuffle(alllist)
            # alllist = alllist[:int(0.6*len(alllist))]
            for file in alllist:
                with open(os.path.join(realdir,"annotations",file), "r") as f:
                    annotations = f.read()
                    LP_dict = json.loads(annotations)
                    distances.append(LP_dict['distance'])
                    if LP_dict['distance'] == 10 and LP_dict['scene_type'] == "Car-Recorder":
                        print(realdir+file)
    # print(distances)
    return distances

def angle_count(roots=[]):
    angles = []
    for root in roots:
        dirnames = os.listdir(root)
        for dirname in dirnames:
            realdir = os.path.join(root, dirname)
            alllist = os.listdir(realdir + "/annotations/")
            random.shuffle(alllist)
            alllist = alllist[:int(0.6*len(alllist))]
            for file in alllist:
                with open(os.path.join(realdir,"annotations",file), "r") as f:
                    annotations = f.read()
                    LP_dict = json.loads(annotations)
                    angles.append(LP_dict['angle'])
    # print(angles)
    return angles

def scene_count(roots=[]):
    scenes = []
    for root in roots:
        dirnames = os.listdir(root)
        for dirname in dirnames:
            realdir = os.path.join(root, dirname)
            alllist = os.listdir(realdir + "/annotations/")
            random.shuffle(alllist)
            alllist = alllist[:int(0.6*len(alllist))]
            for file in alllist:
                with open(os.path.join(realdir,"annotations",file), "r") as f:
                    annotations = f.read()
                    LP_dict = json.loads(annotations)
                    scenes.append(LP_dict['scene_type'])
            print(realdir+" is done!")
    return scenes

def char_count(roots=[]):
    chars = []
    for root in roots:
        dirnames = os.listdir(root)
        for dirname in dirnames:
            realdir = os.path.join(root, dirname)
            alllist = os.listdir(realdir + "/annotations/")
            random.shuffle(alllist)
            alllist = alllist[:int(0.6*len(alllist))]
            for file in alllist:
                with open(os.path.join(realdir,"annotations",file), "r") as f:
                    annotations = f.read()
                    LP_dict = json.loads(annotations)
                    # for value in LP_dict['license_plate_number'].values():
                    #     chars.append(value)
                    if 'num8' in LP_dict['license_plate_number'].keys():
                        chars.append(LP_dict['license_plate_number']['num8'])
            print(realdir+" is done!")
    # print(value)
    return chars

def draw(char_dict = {}):
    x = []
    y = []
    for key, value in char_dict.items():
        x.append(key)
        y.append(value)
    plt.bar(range(len(y)), y, color = 'rgb', tick_label= x)
    # plt.figure(figsize=(60,60))
    plt.show()

def weather_count(roots=[]):
    weathers = {'sunny':0, 'lightrain':0, 'foggy':0, 'heavysnow':0, 'lightsnow':0, 'heavyrain':0, 'others':0}
    lights = {'daytime':0, 'noontime':0, 'night':0}
    for root in roots:
        dirnames = os.listdir(root)
        for dirname in dirnames:
            realdir = os.path.join(root, dirname)
            alllist = os.listdir(realdir + "/annotations/")
            random.shuffle(alllist)
            alllist = alllist[:int(0.6*len(alllist))]
            dirname.replace('_', '')
            if 'daytime' in root or 'daytime' in dirname:
                lights['daytime'] += len(alllist)
            elif 'noontime' in root or 'noontime' in dirname:
                lights['noontime'] += len(alllist)
            elif 'night' in root or 'evening' in root or 'night' in dirname:
                lights['night'] += len(alllist)
            # else:
            #     lights['others'] += len(alllist)


            if 'sunny' in dirname:
                weathers['sunny'] += len(alllist)
            elif 'lightrain' in dirname:
                weathers['lightrain'] += len(alllist)
            elif 'foggy' in dirname:
                weathers['foggy'] += len(alllist)
            elif 'heavysnow' in dirname:
                weathers['heavysnow'] += len(alllist)
            elif 'lightsnow' in dirname:
                weathers['lightsnow'] += len(alllist)
            elif 'heavyrain' in dirname:
                weathers['heavyrain'] += len(alllist)
            else:
                weathers['others'] += len(alllist)

    return lights

def rm_files(file_list):
    if len(file_list):
        i = 0
        for file in file_list:
            if 'raw' in file:
                new_file_path = file.replace('raw', 'errorfile_dir/raw')
            if 'segment' in file:
                new_file_path = file.replace('segment', 'errorfile_dir/segment')
            if not os.path.exists(new_file_path[:-9]):
                os.makedirs(new_file_path[:-9])
                print(new_file_path[:-9] + " is maked!")
            i += 1
            shutil.copyfile(file, new_file_path)
            os.remove(file)
            print("{}/{}".format(i, len(file_list)))

        #     if not os.path.exists(os.path.join(file[:-13], 'errorfile_dir')):
        #         os.makedirs(os.path.join(file[:-13], 'errorfile_dir'))
        #         print(os.path.join(file[:-13], 'errorfile_dir')+" is maked")
        # for file in file_list:
        #     i += 1
        #     shutil.copyfile(file,os.path.join(file[:-13], 'errorfile_dir', file[-9:]))
        #     os.remove(file)
        #     print("{}/{}".format(i, len(file_list)))

def findErrorAnno(realdir):
    ErrorAnno = []
    ErrorAnno.clear()

    alllist = os.listdir(realdir + "/new_annotations/")
    for file in alllist:
        with open(os.path.join(realdir,"new_annotations",file), "r") as f:
            annotations = f.read()
            LP_dict = json.loads(annotations)
            if_right = True
            temp = LP_dict['license_plate_number']['num1']
            del LP_dict['license_plate_number']['num1']
            for value in LP_dict['license_plate_number'].values():
                if value == 66:
                    value = 1 
                if temp < 34 or value > 34:
                    if_right = False
            if if_right == False:
                # print(file)
                ErrorAnno.append(os.path.join(realdir, file))
    return ErrorAnno


def rm_ErrorAnno(file_list=[]):
    print("AnnoError num is "+str(len(file_list)))
    if len(file_list):
        if not os.path.exists(os.path.join(file_list[0][:-10], 'errorAnno_dir', 'raw')):
            os.makedirs(os.path.join(file_list[0][:-10], 'errorAnno_dir', 'raw'))
            print(os.path.join(file_list[0][:-10], 'errorAnno_dir', 'raw')+" is maked")
        if not os.path.exists(os.path.join(file_list[0][:-10], 'errorAnno_dir', 'segment')):
            os.makedirs(os.path.join(file_list[0][:-10], 'errorAnno_dir', 'segment'))
            print(os.path.join(file_list[0][:-10], 'errorAnno_dir', 'segment')+" is maked")
        for file in file_list:
            print(file)
            shutil.copyfile(os.path.join(file[:-10],'raw', file[-10:-4]+ 'png'),os.path.join(file[:-10], 'errorAnno_dir', 'raw',file[-10:-4]+ 'png'))
            os.remove(os.path.join(file[:-10],'raw', file[-10:-4]+ 'png'))
            shutil.copyfile(os.path.join(file[:-10],'segment', file[-10:-4]+ 'png'),os.path.join(file[:-10], 'errorAnno_dir', 'segment',file[-10:-4]+ 'png'))
            os.remove(os.path.join(file[:-10],'segment', file[-10:-4]+ 'png'))
            shutil.copyfile(os.path.join(file[:-10],'new_annotations',file[-10:]),os.path.join(file[:-10], 'errorAnno_dir', file[-10:]))
            os.remove(os.path.join(file[:-10],'new_annotations',file[-10:]))


def file_count(roots=[]):
    a = 0
    for root in roots:
        dirnames = os.listdir(root)
        for dirname in dirnames:
            realdir = os.path.join(root, dirname)
            alllist = os.listdir(realdir + "/annotations/")
            # random.shuffle(alllist)
            # alllist = alllist[:int(0.6*len(alllist))]
            a += len(alllist)

    return a

def data_count_plugin(dict_list = []):
    a = [[0 for i in range(6)] for j in range(5)]
    # b = [[0 for i in range(6)] for j in range(5)]
    wrong_files = 0
    for LP_dict in dict_list:
        #dis 1.5
        if LP_dict['distance'] == 1.5 and LP_dict['angle'] == 0:
            a[0][0] += 1
        elif LP_dict['distance'] == 1.5 and LP_dict['angle'] == 10:
            a[0][1] += 1
        elif LP_dict['distance'] == 1.5 and LP_dict['angle'] == 20:
            a[0][2] += 1            
        elif LP_dict['distance'] == 1.5 and LP_dict['angle'] == 30:
            a[0][3] += 1
        elif LP_dict['distance'] == 1.5 and LP_dict['angle'] == 40:
            a[0][4] += 1 
        elif LP_dict['distance'] == 1.5 and LP_dict['angle'] == 60:
            a[0][5] += 1
        #dis 3
        elif LP_dict['distance'] == 3 and LP_dict['angle'] == 0:
            a[1][0] += 1
        elif LP_dict['distance'] == 3 and LP_dict['angle'] == 10:
            a[1][1] += 1
        elif LP_dict['distance'] == 3 and LP_dict['angle'] == 20:
            a[1][2] += 1            
        elif LP_dict['distance'] == 3 and LP_dict['angle'] == 30:
            a[1][3] += 1
        elif LP_dict['distance'] == 3 and LP_dict['angle'] == 40:
            a[1][4] += 1 
        elif LP_dict['distance'] == 3 and LP_dict['angle'] == 60:
            a[1][5] += 1
        #dis 5
        elif LP_dict['distance'] == 5 and LP_dict['angle'] == 0:
            a[2][0] += 1
        elif LP_dict['distance'] == 5 and LP_dict['angle'] == 10:
            a[2][1] += 1
        elif LP_dict['distance'] == 5 and LP_dict['angle'] == 20:
            a[2][2] += 1            
        elif LP_dict['distance'] == 5 and LP_dict['angle'] == 30:
            a[2][3] += 1
        elif LP_dict['distance'] == 5 and LP_dict['angle'] == 40:
            a[2][4] += 1 
        elif LP_dict['distance'] == 5 and LP_dict['angle'] == 60:
            a[2][5] += 1
        #dis 8
        elif LP_dict['distance'] == 8 and LP_dict['angle'] == 0:
            a[3][0] += 1
        elif LP_dict['distance'] == 8 and LP_dict['angle'] == 10:
            a[3][1] += 1
        elif LP_dict['distance'] == 8 and LP_dict['angle'] == 20:
            a[3][2] += 1            
        elif LP_dict['distance'] == 8 and LP_dict['angle'] == 30:
            a[3][3] += 1
        elif LP_dict['distance'] == 8 and LP_dict['angle'] == 40:
            a[3][4] += 1 
        elif LP_dict['distance'] == 8 and LP_dict['angle'] == 60:
            a[3][5] += 1
        #dis 10
        elif LP_dict['distance'] == 10 and LP_dict['angle'] == 0:
            a[4][0] += 1
        elif LP_dict['distance'] == 10 and LP_dict['angle'] == 10:
            a[4][1] += 1
        elif LP_dict['distance'] == 10 and LP_dict['angle'] == 20:
            a[4][2] += 1            
        elif LP_dict['distance'] == 10 and LP_dict['angle'] == 30:
            a[4][3] += 1
        elif LP_dict['distance'] == 10 and LP_dict['angle'] == 40:
            a[4][4] += 1 
        elif LP_dict['distance'] == 10 and LP_dict['angle'] == 60:
            a[4][5] += 1
        else:
            wrong_files += 1
    print("wrong files: ", wrong_files)
    for i in range(5):
        print("{} {} {} {} {} {}".format(a[i][0], a[i][1], a[i][2], a[i][3], a[i][4], a[i][5]))
    return a
        



def data_count(roots=[]):
    # a == [[0 for i in range(6)] for j in range(5)]
    # b = [[0 for i in range(6)] for j in range(5)]
    Car_Recorder_list = []
    Traffic_Monitor_list = []
    # data_dict1 = {}
    # data_dict2 = {}
    # for root in roots:
    #     dirnames = os.listdir(root)
    #     for dirname in dirnames:
    #         data_dict[dirname] = 0
    #         data_dict[dirname] = 0
    for root in roots:
        dirnames = os.listdir(root)
    for dirname in dirnames:
        Car_Recorder_list.clear()
        Traffic_Monitor_list.clear()
        for root in roots:
            realdir = os.path.join(root, dirname)
            alllist = os.listdir(realdir + "/annotations/")
            for file in alllist:
                with open(os.path.join(realdir,"annotations",file), "r") as f:
                    annotations = f.read()
                    LP_dict = json.loads(annotations)
                    # print(file)
                    if LP_dict['scene_type'] == "Car-Recorder":
                        Car_Recorder_list.append(LP_dict)
                    elif LP_dict['scene_type'] == "Traffic-Monitor":
                        Traffic_Monitor_list.append(LP_dict)
        print(dirname, " Car_Recorder_list")
        data_count_plugin(Car_Recorder_list)
        print(dirname, " Traffic_Monitor_list")
        data_count_plugin(Traffic_Monitor_list)

def modify(roots):
    a = 0
    for root in roots:
        dirnames = os.listdir(root)
        for dirname in dirnames:
            realdir = os.path.join(root, dirname)
            alllist = os.listdir(realdir + "/annotations/")
            a = 0
            for file in alllist:
                with open(os.path.join(realdir,"annotations",file), "rb") as f:
                    annotations = f.read()
                    LP_dict = json.loads(annotations)
                    if LP_dict['distance'] == 10 and LP_dict['scene_type'] == "Car-Recorder":
                        a += 1
                        LP_dict['distance'] = 8
                with open(realdir+"/annotations/"+file, "w") as f:
                    f.write(json.dumps(LP_dict, sort_keys=True, indent=4, separators=(',', ':')))
            print(dirname, " have ", a)




def main():
    #data_count
    # data_count(roots)
    # modify(roots)

    # check error file   
    ErrorAnno_files = []
    NoAnno_raw_files = []
    NoAnno_seg_files = []
    NoAnno_raw_files.clear()
    NoAnno_seg_files.clear()
    for root in roots:
        dirnames = os.listdir(root)
        for dirname in dirnames:
            realdir = os.path.join(root, dirname)
            ErrorAnno_files = findErrorAnno(realdir)
            print(ErrorAnno_files)
            rm_ErrorAnno(ErrorAnno_files)
        ErrorAnno_files.clear()
    NoAnno_raw_files, NoAnno_seg_files = error_count(roots)
    rm_files(NoAnno_raw_files)
    rm_files(NoAnno_seg_files)
    # count angle
    # angle_dict = dict(Counter(angle_count(roots)).most_common())
    # draw(angle_dict)
    # print(angle_dict)

    #count scene
    # scene_dict = dict(Counter(scene_count(roots)).most_common())
    # draw(scene_dict)
    # print(scene_dict)

    # distance count
    # distance_dict = dict(Counter(distance_count(roots)).most_common())
    # print(distance_dict)
    # draw(distance_dict)

    # weather count
    # weather_dict = dict(Counter(weather_count(roots)).most_common())
    # print(weather_dict)
    # draw(weather_dict)

    # char count
    # char_dict = dict(Counter(char_count(roots)).most_common(34))
    # draw(char_dict)
    # for key, value in char_dict.items():
    #     print("{}：{}".format(char_table[key], value))




if __name__ == "__main__":
    main()


    
