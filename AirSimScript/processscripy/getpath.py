import os
import glob

        
CarType_Select = ['BigFront_dataset', 'BigNewEnergy_dataset', 'Car_dataset', 'NewEnergy_dataset', 'Bus_dataset', 'Truck_dataset']
LightType_Select = ['daytime', 'night', 'noontime']
WeatherType_Select = ['heavysnow', 'heavyrain', 'foggy', 'lightsnow', 'lightrain', 'sunny', 'cloudy']#others表示部分单独制作的如黄昏等数据，数据较少
AngleType_Select = ['0', '10', '20', '30', '40', '60']
DistanceType_Select = ['1.5', '3', '5', '8', '10']
SceneType_Select = ['Traffic-Monitor', 'Car-Recorder']
Type_Select = ['Train']


def GetDataPathList(CarType=CarType_Select, LightType=LightType_Select, 
            WeatherType=WeatherType_Select, AngleType=AngleType_Select, DistanceType=DistanceType_Select, SceneType=SceneType_Select):
    PathList = []
    for CarType_dir in CarType:
        LightType_dirs = os.listdir(CarType_dir)
        for LightType_dir in LightType_dirs:
            if LightType_dir in LightType_Select:
                WeatherType_dirs = os.listdir(os.path.join(CarType_dir, LightType_dir))
                for WeatherType_dir in WeatherType_dirs:
                    for WeatherType_Select_Ele in WeatherType_Select:
                        if WeatherType_Select_Ele in WeatherType_dir:
                            print(os.path.join(CarType_dir, LightType_dir, WeatherType_dir))







if __name__ == '__main__':
    CarType_Select = ['Bus_dataset', 'Truck_dataset']
    LightType_Select = ['night', 'noontime']
    WeatherType_Select = ['heavysnow', 'heavyrain']#others表示部分单独制作的如黄昏等数据，数据较少
    # AngleType_Select = ['0', '10', '20', '30', '40', '60']
    # DistanceType_Select = ['1.5', '3', '5', '8', '10']
    # SceneType_Select = ['Traffic-Monitor', 'Car-Recorder']
    # Type_Select = ['Train']
    GetDataPathList()
