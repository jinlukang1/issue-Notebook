import cv2
import os
import numpy as np

PASS_SCORE = 0.02

def Car_Pixel_Num(input_pic):
	image = cv2.imread(input_pic)
# 	[rows_suv, cols_suv] = np.where(image[:, :, 2] == 24)
	[rows, cols] = np.where(image[:, :, 2] == 41)
	return len(rows)

def Suv_Pixel_Num(input_pic):
	image = cv2.imread(input_pic)
# 	[rows_suv, cols_suv] = np.where(image[:, :, 2] == 24)
	[rows, cols] = np.where(image[:, :, 2] == 24)
	return len(rows)

def No_Pass(key_num, current_num):
	if abs(key_num - current_num)/max(key_num, current_num) < PASS_SCORE:
		return False
	else:
		return True

def Pixel_Num(input_pic):
	Car_Pixel = Car_Pixel_Num(input_pic)
	Suv_Pixel = Suv_Pixel_Num(input_pic)
	return Suv_Pixel, Car_Pixel

def Counter_init(namelist):
	Car_counter = 0
	Suv_counter = 0
	for pic in namelist:
		pic_path = os.path.join(PATH, pic)
		Car_Pixel = Car_Pixel_Num(pic_path)
		Suv_Pixel = Suv_Pixel_Num(pic_path)
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



# PATH = "C:/Users/Administrator/Desktop/AirSimData/scene/test/segment/"
PATH = "E:/dataset/final_dataset/scene3_noontime/segment/"
namelist = os.listdir(PATH)
print(namelist)


camera_count = 0
# Car_pixel_cache = Car_Pixel_Num(os.path.join(path, namelist[0])) 
# Suv_Cache, Car_Cache = 
Suv_Cache, Car_Cache = Counter_init(namelist)

for pic in namelist:
	# print(Car_Pixel_Num(os.path.join(path,pic)))
	# print(Suv_Pixel_Num(os.path.join(path,pic)))
	print("Now At:", namelist.index(pic))
	pic_path = os.path.join(PATH, pic)
	Suv_Pixel, Car_Pixel = Pixel_Num(pic_path)
	if No_Pass(Suv_Pixel, Suv_Cache) and No_Pass(Car_Pixel, Car_Cache):
		camera_count += 1
		if Suv_Pixel > Car_Pixel:
			Suv_Cache = Suv_Pixel
		else:
			Car_Cache = Car_Pixel
		print("Camera transfer at:", namelist.index(pic))



	
print(camera_count)

# res=cv2.resize(image1,(1280,720),interpolation=cv2.INTER_CUBIC)
# cv2.namedWindow("Image") 
# cv2.imshow("Image", res) 
# cv2.waitKey (0)
# cv2.destroyAllWindows()
