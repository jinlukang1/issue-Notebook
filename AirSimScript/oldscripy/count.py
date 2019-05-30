import os
import json
from collections import Counter

num1_count = []
num2_count = []
num3_count = []
num4_count = []
num5_count = []
num6_count = []
num7_count = []
num_count = []
angle_count = []
distance_count = []



root1 = "D:/Documents/AirSim/Scene1_Data/daytime/"
root2 = "D:/Documents/AirSim/Scene1_Data/evening/"
root3 = "D:/Documents/AirSim/Scene1_Data/noontime/"
for root in root1,root2,root3:
	dirnames = os.listdir(root)
	for dirname in dirnames:
		realdir = root + dirname
		alllist = os.listdir(realdir + "/annotations/")
		for file in alllist:
			with open(realdir + "/annotations/"+file, "r") as f:
				annotation = f.read()
				LP_dict = json.loads(annotation)
				# num1_count.append(LP_dict['license_plate_number']['num1'])
				# num2_count.append(LP_dict['license_plate_number']['num2'])
				# num3_count.append(LP_dict['license_plate_number']['num3'])
				# num4_count.append(LP_dict['license_plate_number']['num4'])
				# num5_count.append(LP_dict['license_plate_number']['num5'])
				# num6_count.append(LP_dict['license_plate_number']['num6'])
				# num7_count.append(LP_dict['license_plate_number']['num7'])
				num_count.append(LP_dict['license_plate_number']['num1'])
				num_count.append(LP_dict['license_plate_number']['num2'])
				num_count.append(LP_dict['license_plate_number']['num3'])
				num_count.append(LP_dict['license_plate_number']['num4'])
				num_count.append(LP_dict['license_plate_number']['num5'])
				num_count.append(LP_dict['license_plate_number']['num6'])
				num_count.append(LP_dict['license_plate_number']['num7'])
				angle_count.append(LP_dict['angle'])
				distance_count.append(LP_dict['distance'])
		print(dirname + " is Done!")

# result1 = Counter(num1_count)
# result2 = Counter(num2_count)
# result3 = Counter(num3_count)
# result4 = Counter(num4_count)
# result5 = Counter(num5_count)
# result6 = Counter(num6_count)
# result7 = Counter(num7_count)
# print(result1)
# print(result2)
# print(result3)
# print(result4)
# print(result5)
# print(result6)
# print(result7)

result_num = Counter(num_count)
print(result_num)