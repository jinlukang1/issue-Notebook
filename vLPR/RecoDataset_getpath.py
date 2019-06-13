import os
import glob
import json
from convert_data import npy_gen
        
CarType_Select = ['BigFront_reco_dataset', 'BigNewEnergy_reco_dataset', 'Car_reco_dataset', 'NewEnergy_reco_dataset', 'Bus_reco_dataset', 'Truck_reco_dataset']
LightType_Select = ['daytime', 'night', 'noontime']
WeatherType_Select = ['heavysnow', 'heavyrain', 'foggy', 'lightsnow', 'lightrain', 'sunny', 'cloudy']#others表示部分单独制作的如黄昏等数据，数据较少
AngleType_Select = ['0', '10', '20', '30', '40', '60']
DistanceType_Select = ['1.5', '3', '5', '8', '10']
SceneType_Select = ['Traffic-Monitor', 'Car-Recorder']
DataType_Select = ['train', 'val', 'test']


def GetDataPathList(CarType=CarType_Select, LightType=LightType_Select, 
            WeatherType=WeatherType_Select, AngleType=AngleType_Select, DistanceType=DistanceType_Select, SceneType=SceneType_Select, DataType=DataType_Select):
    PathList = []
    Imagelist = []
    Annolist = []
    for CarType_dir in CarType:
        LightType_dirs = os.listdir(CarType_dir)
        print(LightType_dirs)
        for LightType_dir in LightType_dirs:
            if LightType_dir in LightType:
                WeatherType_dirs = os.listdir(os.path.join(CarType_dir, LightType_dir))
                for WeatherType_dir in WeatherType_dirs:
                    for WeatherType_Select_Ele in WeatherType:
                        if WeatherType_Select_Ele in WeatherType_dir:
                            DataType_dirs = os.listdir(os.path.join(CarType_dir, LightType_dir, WeatherType_dir))
                            print(DataType_dirs)
                            for DataType_dir in DataType_dirs:
                                for DataType_Select_Ele in DataType:
                                    if DataType_Select_Ele in DataType_dir:
                                        PathList.append(os.path.join(CarType_dir, LightType_dir, WeatherType_dir, DataType_dir))
    print(PathList)
    for each_path in PathList:
        # Annolist.clear()
        all_annotations = glob.glob(each_path+'/new_annotations/*.json')
        all_imgs = glob.glob(each_path+'/raw/*.png')
        for each_annotation in all_annotations:
            with open(each_annotation, "r") as f:
                data_annotation = f.read()
                LP_dict = json.loads(data_annotation)
            for AngleType_Select_Ele in AngleType:
                if str(LP_dict['angle']) == AngleType_Select_Ele:
                    for DistanceType_Select_Ele in DistanceType:
                        if str(LP_dict['distance']) == DistanceType_Select_Ele:
                            for SceneType_Select_Ele in SceneType:
                                if str(LP_dict['scene_type']) == SceneType_Select_Ele:
                                    Annolist.append(each_annotation)
                                    Imagelist.append(each_annotation.replace('new_annotations', 'raw').replace('json', 'png'))
        print(each_path + " is done!")
    print(Imagelist, '\n', Annolist)
    with open('Imagelist.txt', 'w') as f:
        for each_image in Imagelist:
            f.write(each_image)
            f.write('\n')
    with open('Annolist.txt', 'w') as f:
        for each_annotation in Annolist:
            f.write(each_annotation)
            f.write('\n')


    
    print('All is done!')







if __name__ == '__main__':
    Your_CarType_Select = ['Car_reco_dataset']
    Your_LightType_Select = ['night', 'noontime', 'daytime']
    Your_WeatherType_Select = ['heavysnow' ,'heavyrain', 'foggy','lightsnow','lightrain' ,'sunny','cloudy']#others表示部分单独制作的如黄昏等数据，数据较少
    Your_AngleType_Select = ['0', '10', '20', '30', '40', '60']
    Your_DistanceType_Select = ['1.5', '3', '5', '8', '10']
    Your_SceneType_Select = ['Car-Recorder']
    Your_DataType_Select = ['val']
    for each_distance_selection in Your_DistanceType_Select:
        GetDataPathList(CarType=Your_CarType_Select, LightType=Your_LightType_Select, DataType=Your_DataType_Select, 
            WeatherType=Your_WeatherType_Select, AngleType=Your_AngleType_Select, DistanceType=[each_distance_selection], SceneType=Your_SceneType_Select)
        data_type = each_distance_selection+"_"+Your_DataType_Select[0]
        npy_gen(data_type = data_type)
