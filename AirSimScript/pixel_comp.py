#尝试失败
import cv2
import os
import numpy as np
import math
import json
from collections import Counter


PASS_SCORE = 0.02

def Car_Pixel_Num(input_pic):
	image = cv2.imread(input_pic)
	[rows, cols] = np.where(image[:, :, 2] == 41)
	return len(rows)

def Suv_Pixel_Num(input_pic):
	image = cv2.imread(input_pic)
	[rows, cols] = np.where(image[:, :, 2] == 24)
	return len(rows)

def Plate_Pixel_Num(input_pic):
	image = cv2.imread(input_pic)
	[rows, cols] = np.where(image[:, :, 2] == 89)
	return len(rows)

def No_Pass(key_num, current_num):
	if abs(key_num - current_num) < PASS_SCORE:
		#print(abs(key_num - current_num)/max(key_num, current_num))
		return False
	else:
		#print(abs(key_num - current_num)/max(key_num, current_num))
		return True

def Pixel_Num(input_pic):
	Car_Pixel = Car_Pixel_Num(input_pic)
	Suv_Pixel = Suv_Pixel_Num(input_pic)
	return Suv_Pixel, Car_Pixel

def Angle_Cal(file):
	with open(file,"r") as f:
		annotations = f.read()
		LP_dict = json.loads(annotations)
		A_x = LP_dict['vertices_locations']['x1']
		A_y = LP_dict['vertices_locations']['y1']
		B_x = LP_dict['vertices_locations']['x2']
		B_y = LP_dict['vertices_locations']['y2']
	return round((B_y-A_y)/abs(A_x-B_x), 3)

#像素分数以及序号初始化
def Counter_init(namelist):
	Car_counter = 0
	Suv_counter = 0
	for pic in namelist:
		pic_path = os.path.join(PATH, pic)
		Suv_Pixel, Car_Pixel = Pixel_Num(pic_path)
		if Car_Pixel > Suv_Pixel and Car_counter == 0:
			Car_counter = Car_Pixel
			print("Car_init_index:", namelist.index(pic))
			print("Car_init_score:", Car_counter)
		elif Car_Pixel <= Suv_Pixel and Suv_counter == 0:
			Suv_counter = Suv_Pixel
			print("Suv_init_index:", namelist.index(pic))
			print("Suv_init_score:", Suv_counter)
		if Car_counter != 0 and Suv_counter != 0:
			return Suv_counter, Car_counter

def Car_Type(file):
	with open(file,"r") as f:
		annotations = f.read()
		LP_dict = json.loads(annotations)
		A_x = LP_dict['vertices_locations']['x1']
		A_y = LP_dict['vertices_locations']['y1']
	image = cv2.imread(file.replace("annotations", "segment")[:-4] + "png")
	return image[A_y-50, A_x, 2]#图像的尺寸是1440*2560


car_count = []

# PATH = "C:/Users/Administrator/Desktop/AirSimData/scene/test/segment/"
# PATH = "E:/dataset/final_dataset/front_noontime_lightrain/segment/"
# PATH = "E:/dataset/final_dataset/scene3_noontime/segment/"
# PATH = "E:/dataset/Scene2_Data/evening/heavy_snow/segment/"
camera_count = 1

ANNO_PATH = "E:/dataset/Scene2_Data/evening/heavy_snow/"
namelist = os.listdir(ANNO_PATH+"annotations/")


Angle_Cache = Angle_Cal(os.path.join(ANNO_PATH+"annotations/", namelist[0]))
for pic_anno in namelist:
	pic_anno_path = os.path.join(ANNO_PATH+"annotations/", pic_anno)
	if Car_Type(pic_anno_path) == 41:
		Angle_Score = Angle_Cal(pic_anno_path)
		# car_count.append(Car_Type(pic_anno_path))
		if No_Pass(Angle_Score, Angle_Cache):
			Angle_Cache = Angle_Score
			camera_count += 1
			print("Camera transfer at:", namelist.index(pic_anno)+1)



# Plate_Cache = Plate_Pixel_Num(os.path.join(PATH, namelist[103]))
# print(Plate_Cache)
# Plate_Cache = Plate_Pixel_Num(os.path.join(PATH, namelist[99]))
# print(Plate_Cache)

# Suv_Cache, Car_Cache = Counter_init(namelist)
# SUV和CAR的像素分数的初始化

# for pic in namelist:
# 	#print("Now At:", namelist.index(pic)+1)
# 	pic_path = os.path.join(PATH, pic)
# 	Suv_Pixel, Car_Pixel = Pixel_Num(pic_path)
# 	if No_Pass(Suv_Pixel, Suv_Cache) and No_Pass(Car_Pixel, Car_Cache):
# 		print(abs(Suv_Pixel - Suv_Cache)/max(Suv_Pixel, Suv_Cache))
# 		print(abs(Car_Pixel - Car_Cache)/max(Car_Pixel, Car_Cache))
# 		camera_count += 1
# 		if Suv_Pixel > Car_Pixel:
# 			Suv_Cache = Suv_Pixel
# 		else:
# 			Car_Cache = Car_Pixel
# 		print("Camera transfer at:", namelist.index(pic)+1)


# print(Plate_Cache)
# for pic in namelist:
# 	pic_path = os.path.join(PATH, pic)
# 	Plate_Pixel = Plate_Pixel_Num(pic_path)
# 	if No_Pass(Plate_Pixel, Plate_Cache):
# 		print("Camera transfer at:", namelist.index(pic)+1)
# 		Plate_Cache = Plate_Pixel


print(camera_count)

# res=cv2.resize(image1,(1280,720),interpolation=cv2.INTER_CUBIC)
# cv2.namedWindow("Image") 
# cv2.imshow("Image", res)
# cv2.waitKey (0)
# cv2.destroyAllWindows()
