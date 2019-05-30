import json
import os

# LP_dict['vertices_locations'] = {'x1':0.0,'x2':0.0,'x3':0.0,'x4':0.0,'y1':0.0,'y2':0.0,'y3':0.0,'y4':0.0}
# LP_dict['license_plate_number'] = {'num1':0,'num2':0,'num3':0,
# 									'num4':0,'num5':0,'num6':0,'num7':0}
# LP_dict['num1_locations'] = {'x1':0.0,'x2':0.0,'x3':0.0,'x4':0.0}
# LP_dict['num2_locations'] = {'x1':0.0,'x2':0.0,'x3':0.0,'x4':0.0}
# LP_dict['num3_locations'] = {'x1':0.0,'x2':0.0,'x3':0.0,'x4':0.0}
# LP_dict['num4_locations'] = {'x1':0.0,'x2':0.0,'x3':0.0,'x4':0.0}
# LP_dict['num5_locations'] = {'x1':0.0,'x2':0.0,'x3':0.0,'x4':0.0}
# LP_dict['num6_locations'] = {'x1':0.0,'x2':0.0,'x3':0.0,'x4':0.0}
# LP_dict['num7_locations'] = {'x1':0.0,'x2':0.0,'x3':0.0,'x4':0.0}

roots = []
# roots.append(r"d:\Documents\AirSim\dataset_done/")

# roots.append(r"G:\dataset\Big_NewEnergy\scene3\daytime/")
# roots.append(r"G:\dataset\Big_NewEnergy\scene3\noontime/")
# roots.append(r"G:\dataset\Big_NewEnergy\scene3\night/")
# roots.append(r"G:\dataset\Big_NewEnergy\scene1\daytime/")
# roots.append(r"G:\dataset\Big_NewEnergy\scene1\noontime/")
# roots.append(r"G:\dataset\Big_NewEnergy\scene1\night/")
# roots.append(r"G:\dataset\NewEnergy_dataset\front/")
# roots.append(r"G:\dataset\NewEnergy_dataset\night/")
# roots.append(r"G:\dataset\NewEnergy_dataset\noontime/")
# roots.append(r"G:\reco_dataset\NewEnergy_reco_dataset\daytime/")
# roots.append(r"G:\reco_dataset\NewEnergy_reco_dataset\front/")
# roots.append(r"G:\reco_dataset\NewEnergy_reco_dataset\night/")
roots.append(r"D:\Documents\AirSim\BigFront_dataset/")

# roots.append(r"G:\dataset\Car_dataset\Scene2\daytime/")
# roots.append(r"G:\dataset\Car_dataset\Scene2\noontime/")
# roots.append(r"G:\dataset\Car_dataset\Scene2\evening/")
# roots.append(r"G:\dataset\Car_dataset\Scene2\front/")
# roots.append(r"G:\dataset\Car_dataset\Scene1\daytime/")
# roots.append(r"G:\dataset\Car_dataset\Scene1\noontime/")
# roots.append(r"G:\dataset\Car_dataset\Scene1\evening/")
# roots.append(r"G:\dataset\Car_dataset\Scene1\front/")
# roots.append(r"G:\dataset\NewEnergy_dataset\scene2\daytime/")
# roots.append(r"G:\dataset\NewEnergy_dataset\scene2\front/")
# roots.append(r"G:\dataset\NewEnergy_dataset\scene2\night/")
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


# # distance scene1
distance1 = range(0,7)
distance2 = range(7,14)
distance3 = range(14,21)
distance4 = range(21,28)
distance5 = range(28,37)

angle1 = [3,10,17,24]
angle2 = [28,29,30]
angle3 = [2,4,9,11,16,18,23,25,31,32,33]
angle4 = [34,35,36]
angle5 = [1,5,8,12,15,19,22,26]
angle6 = [0,6,7,13,14,20,21,27]

# distance scene2

# distance1 = [0,1,2,3,4,5,6,33,34,35,36]
# distance2 = []
# distance3 = range(7,14)
# distance4 = range(14,21)
# distance5 = range(21,33)

# angle1 = [0,7,14,33]
# angle2 = [24,25,26,30,31,32]
# angle3 = [1,2,8,9,15,16,21,22,23,27,28,29,34,35]
# angle4 = []
# angle5 = [3,4,10,11,17,18,36]
# angle6 = [5,6,12,13,19,20]


for root in roots:
	dirnames = os.listdir(root)
	for dirname in dirnames:
		realdir = root + dirname
		file_num = len(os.listdir(realdir+"/annotations"))
#		os.makedirs(realdir+"/annotations/")
		for file in os.listdir(realdir+"/annotations"):
			with open(realdir+"/annotations/"+file, "r") as f:
				annotation = f.read()
				LP_dict = json.loads(annotation)

			# if LP_dict['angle'] == 34 or LP_dict['angle'] == 35:
			LP_dict['angle'] = int(int(file[:-5])/50)
			LP_dict['distance'] = int(int(file[:-5])/50)
			# print(LP_dict['angle'])





			key_num1 = LP_dict['angle']%37
			if key_num1>=0 and key_num1<=27:
				LP_dict['scene_type'] = 'Car-Recorder'
			else:
				LP_dict['scene_type'] = 'Traffic-Monitor'

			# key_num2 = LP_dict['angle']%37
			# if (key_num2>=0 and key_num2<=20) or key_num2 >=33:
			# 	LP_dict['scene_type'] = 'Car-Recorder'
			# else:
			# 	LP_dict['scene_type'] = 'Traffic-Monitor'


# distance judge
			if LP_dict['distance']%37 in distance1:
				LP_dict['distance'] = 1.5
			elif LP_dict['distance']%37 in distance2:
				LP_dict['distance'] = 3
			elif LP_dict['distance']%37 in distance3:
				LP_dict['distance'] = 5
			elif LP_dict['distance']%37 in distance4:
				LP_dict['distance'] = 8
			elif LP_dict['distance']%37 in distance5:
				LP_dict['distance'] = 10
# angle judge
			if LP_dict['angle']%37 in angle1:
				LP_dict['angle'] = 0
			elif LP_dict['angle']%37 in angle2:
				LP_dict['angle'] = 10
			elif LP_dict['angle']%37 in angle3:
				LP_dict['angle'] = 20
			elif LP_dict['angle']%37 in angle4:
				LP_dict['angle'] = 30
			elif LP_dict['angle']%37 in angle5:
				LP_dict['angle'] = 40
			elif LP_dict['angle']%37 in angle6:
				LP_dict['angle'] = 60

			# if os.listdir(realdir+"/annotations").index(file) > 1850:
			# 	#print(file)
			# 	LP_dict['distance'] = 10
			# 	LP_dict['angle'] = 20

			# print("file:{}, distance:{}".format(file,LP_dict['distance']))
			# LP_dict['distance'] = int(int(file[:-5])/100)
			# LP_dict['angle'] = int(int(file[:-5])/100)
			with open(realdir+"/annotations/"+file, "w") as f:
				f.write(json.dumps(LP_dict, sort_keys=True, indent=4, separators=(',', ':')))


		# 	with open(realdir+'/annotations/%s.json' % file[:-4], "w") as f:
		# 		LP_dict['distance'] = 1
		# 		LP_dict['angle'] = 1
		# 		f.write(json.dumps(LP_dict,sort_keys=True, indent=4, separators=(',', ' : ')))

		print(realdir+ str(file_num) + " Done!")


#with open("C:/Users/Administrator/Desktop/script/test.json", "w") as f:
#	f.write(json.dumps(LP_dict,sort_keys=True, indent=4, separators=(',', ' : ')))
