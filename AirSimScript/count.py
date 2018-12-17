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


root1 = "D:/Documents/AirSim/Scene1_Data/daytime/"
root2 = "D:/Documents/AirSim/Scene1_Data/evening/"
root3 = "D:/Documents/AirSim/Scene1_Data/noontime/"
for root in root1,root2,root3:
	dirnames = os.listdir(root)
	for dirname in dirnames:
		realdir = root + dirname
#	alllist = os.listdir(realdir + "/Annotation/")
#	for file in alllist:
#		with open(root+file, "r") as f:
#			annotation = f.read()
#			LP_dict = json.loads(annotation)
#			num1_count.append(LP_dict['license_plate_number']['num1'])
#			num2_count.append(LP_dict['license_plate_number']['num2'])
#			num3_count.append(LP_dict['license_plate_number']['num3'])
#			num4_count.append(LP_dict['license_plate_number']['num4'])
#			num5_count.append(LP_dict['license_plate_number']['num5'])
#			num6_count.append(LP_dict['license_plate_number']['num6'])
#			num7_count.append(LP_dict['license_plate_number']['num7'])
		print(dirname + " is Done!")
#	for key in LP_dict['license_plate_number']['num1']:
#		count_dict['num1'] = count_dict.get(key, 0) + 1

	result1 = Counter(num1_count)
#	result2 = Counter(num2_count)
#	result3 = Counter(num3_count)
#	result5 = Counter(num5_count)
#	result6 = Counter(num6_count)
#	result7 = Counter(num7_count)
	print(result1)
#	print(result2)
#	print(result3)
#	print(result4)
#	print(result5)
#	print(result6)
#	print(result7)