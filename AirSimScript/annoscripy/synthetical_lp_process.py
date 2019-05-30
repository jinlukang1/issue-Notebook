import cv2
import os
from collections import Counter
import numpy as np
import glob
from skimage import measure
from pylab import *
import json
import pprint

char_table = {178:0, 109:1, 15:2, 103:3, 149:4, 11:5, 78:6, 35:7, 151:8, 143:9, 243:10, 72:11, 160:12, 214:13, 167:14, 101:15, 53:16, 3:17, 186:18, 88:19, 71:20, 236:21, 215:22, 113:23,
					47:24, 219:25, 57:26, 227:27, 145:28, 217:29, 250:30, 208:31, 224:32, 69:33} 
province_table = {211:34, 251:35, 22:36, 176:37, 90:38, 210:39, 112:40, 56:41, 209:42, 18:43, 189:44, 10:45, 68:46,
					213:47, 19:48, 97:49, 237:50, 4:51, 32:52, 164:53, 241:54, 16:55, 86:56, 255:57, 44:58, 46:59, 119:60, 36:61, 106:62, 84:63, 14:64, 65:66}

path = "C:/Users/Administrator/Desktop/AirSimData/raw"
namelist = os.listdir(path)


class UserEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.int64):
			return int(obj)
		return json.JSONEncoder.default(self, obj)

def cal_overlap(input1, input2):
	width1 = float(abs(input1[1] - input1[0]))
	width2 = float(abs(input2[1] - input2[0]))
	if input1[1]<input2[0] or input1[0] > input2[1]:
		return 0
	elif input1[0] >= input2[0] and input1[1] <= input2[1]:
		return 1
	elif input2[0] >= input1[0] and input2[1] <= input1[1]:
		return 1
	else:
		a = abs(input1[1] - input2[0])
		b = abs(input1[0] - input2[1])
		overlap = max(min(a,b)/width1, min(a,b)/width2)
		return overlap
# def Quadrilateral_area(coordinates_vector):
# 	X1 = coordinates_vector[0, :]
# 	X2 = coordinates_vector[1, :]
# 	X3 = coordinates_vector[2, :]
# 	X4 = coordinates_vector[3, :]
# 	side_length1 = np.linalg.norm(X1 - X2)
# 	side_length2 = np.linalg.norm(X2 - X3)
# 	side_length3 = np.linalg.norm(X3 - X4)
# 	side_length4 = np.linalg.norm(X4 - X1)
# 	side_array = np.array([side_length1, side_length2, side_length3, side_length4])
# 	p = sum(side_array) / 2
# 	area = np.sqrt((p - side_length1) * (p - side_length2) * (p - side_length3) * (p - side_length4))
# 	return area

for i in range(len(namelist)):
	image_name = namelist[i]
	image = cv2.imread(os.path.join(path, image_name))
	print(image_name)

	LP_dict = {}
	result = []
	[rows, cols] = np.where(image[:, :, 2] == 89)
	x_add_y = 0
	coordinate_min = 1440 + 2560
	coordinate_max = 0
	min_index = 0
	max_index = 0
	for j in range(len(rows)):
		x_add_y = rows[j] + cols[j]
		if coordinate_min > x_add_y:
			min_index = j
			coordinate_min = x_add_y
		if coordinate_max < x_add_y:
			max_index = j
			coordinate_max = x_add_y
	x1, y1 = cols[min_index], rows[min_index]
	x3, y3 = cols[max_index], rows[max_index]
	image_shrink = image[:, :x1 + 500, :]

	[rows, cols] = np.where(image_shrink[:, :, 2] == 89)
	x_add_y = 0
	coordinate_min = 1440 + 2560
	coordinate_max = 0
	min_index = 0
	max_index = 0
	for j in range(len(rows)):
		x_add_y = rows[j] + cols[j]
		if coordinate_min > x_add_y:
			min_index = j
			coordinate_min = x_add_y
		if coordinate_max < x_add_y:
			max_index = j
			coordinate_max = x_add_y
	x1, y1 = cols[min_index], rows[min_index]
	x3, y3 = cols[max_index], rows[max_index]

	rotated_image =  np.rot90(image_shrink)
	[rows, cols] = np.where(rotated_image[:, :, 2] == 89)
	x_add_y = 0
	coordinate_min = 1440 + 2560
	coordinate_max = 0
	min_index = 0
	max_index = 0
	for j in range(len(rows)):
		x_add_y = rows[j] + cols[j]
		if coordinate_min > x_add_y:
			min_index = j
			coordinate_min = x_add_y
		if coordinate_max < x_add_y:
			max_index = j
			coordinate_max = x_add_y
	x2, y2 = image_shrink.shape[1] - rows[min_index] - 1, cols[min_index]
	x4, y4 = image_shrink.shape[1] - rows[max_index] - 1, cols[max_index]
	# print(x1, y1, x2, y2, x3, y3, x4, y4)
	LP_dict['vertices_locations'] = {'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2, 'x3':x3, 'y3':y3, 'x4':x4, 'y4':y4}

	coordinates = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
	# area = Quadrilateral_area(coordinates)
	# print(area)
	xmin = min(x1, x4)
	ymin = min(y1, y2)
	xmax = max(x2, x3)
	ymax = max(y3, y4)
	roi = image_shrink[ymin: ymax + 1, xmin: xmax + 1, :]
	
	pts1 = np.float32([[x1 - xmin, y1 - ymin], [x2 - xmin, y2 - ymin], [x3 - xmin, y3 - ymin], [x4 - xmin, y4 - ymin]])
	pts2 = np.float32([[0, 0], [159, 0], [159, 49], [0, 49]])
	M = cv2.getPerspectiveTransform(pts1, pts2)
	perspective_roi = cv2.warpPerspective(roi, M, (160, 50), flags = cv2.INTER_NEAREST)

	semantic_map = np.ones([50, 160]) * 255
	for (piex, index) in province_table.items():
		[rows, cols] = np.where(perspective_roi[:, :, 2] == piex)
		if len(rows) < 2:
			continue
		#print(piex)
		left = min(cols)
		right = max(cols)
		up = min(rows)
		down = max(rows)
		semantic_map[up: down, left: right] = index
		cmean = sum(cols) / len(cols)
		rmean = sum(rows) / len(rows)
		area = len(cols)
		result.append({'char':index, 'x1':left, 'y1':up, 'x2':right, 'y2':down, 'delete':0, 'cmean':cmean, 'rmean':rmean, 'area':area})
	for (piex, index) in char_table.items():
		[rows, cols] = np.where(perspective_roi[:, :, 2] == piex)
		if len(rows) < 5:
			continue
		#print(piex)
		tmp_map = np.zeros([50, 160])
		for p in range(len(rows)):
			tmp_map[rows[p], cols[p]] = 1
		[L, num] = measure.label(tmp_map, background = 0, connectivity = 2, return_num = True)
		for q in range(num):
			[rows, cols] = np.where(L == q + 1)
			left = min(cols)
			right = max(cols)
			up = min(rows)
			down = max(rows)
			semantic_map[up: down, left: right] = index
			cmean = sum(cols) / len(cols)
			rmean = sum(rows) / len(rows)
			area = len(cols)
			if area < 10:
				continue
			result.append({'char':index, 'x1':left, 'y1':up, 'x2':right, 'y2':down, 'delete':0, 'cmean':cmean, 'rmean':rmean, 'area':area})
	semantic_map = semantic_map.astype(np.uint8)

	result_sorted1 = sorted(result, key=lambda tmp:tmp['rmean'])
	result_sorted2 = result_sorted1[:2]
	result_sorted2 = sorted(result_sorted2, key=lambda tmp:tmp['cmean'])
	result_sorted3 = result_sorted1[-5:]
	result_sorted3 = sorted(result_sorted3, key=lambda tmp:tmp['cmean'])

	result_sorted = result_sorted2+result_sorted3#mod
	pprint.pprint(result_sorted)
	overlap_group = []
	for i in range(2,len(result_sorted)-1):
		overlap = cal_overlap([result_sorted[i]['x1'],result_sorted[i]['x2']],[result_sorted[i+1]['x1'],result_sorted[i+1]['x2']])
		print(overlap)
		if overlap >= 0.5:
			flag_join = 0
			print('yes')
			for j in range(len(overlap_group)):
				if i in overlap_group[j] and i+1 not in overlap_group[j]:
					overlap_group[j].append(i+1)
					flag_join = 1
			if flag_join == 0:
				overlap_group.append([])
				overlap_group[-1].append(i)
				overlap_group[-1].append(i+1)
	print(overlap_group)
	flag_merge = 0
	for i in range(len(overlap_group)):
		group = overlap_group[i]
		if flag_merge == 0:
			up = 50
			down = -1
			left = 160
			right = -1
			cmean = 0.0
			area = 0.0
			for j in range(len(group)):
				cmean += result_sorted[group[j]]['cmean'] * result_sorted[group[j]]['area']
				area += result_sorted[group[j]]['area']

				if result_sorted[group[j]]['y1'] < up:
					up = result_sorted[group[j]]['y1']
				if result_sorted[group[j]]['y2'] > down:
					down = result_sorted[group[j]]['y2']
				if result_sorted[group[j]]['x1'] < left:
					left = result_sorted[group[j]]['x1']
				if result_sorted[group[j]]['x2'] > right:
					right = result_sorted[group[j]]['x2']
				result_sorted[group[j]]['delete'] = 1
				char = result_sorted[group[j]]['char']
			cmean /= area
			result_sorted.append({'char':char, 'x1':left, 'y1':up, 'x2':right, 'y2':down, 'delete':0, 'cmean':cmean, 'rmean':rmean, 'area':area})
			pprint.pprint(result_sorted)

	result_sorted = [data for data in result_sorted if data['delete'] == 0]

	result_sorted1 = sorted(result_sorted, key=lambda tmp:tmp['rmean'])
	result_sorted2 = result_sorted1[:2]
	result_sorted2 = sorted(result_sorted2, key=lambda tmp:tmp['cmean'])
	result_sorted3 = result_sorted1[-5:]
	result_sorted3 = sorted(result_sorted3, key=lambda tmp:tmp['cmean'])

	result_sorted = result_sorted2+result_sorted3#mod
	result_sorted_temp = result_sorted
	result_sorted = result_sorted3

	mean_width = 0
	for i in range(len(result_sorted)):
		if result_sorted[i]['char'] != 1:
			mean_width += result_sorted[i]['x2'] - result_sorted[i]['x1']
	mean_width /= 5
	if len(result_sorted) < 5:
				for i in range(len(result_sorted)):
					width = result_sorted[i]['x2'] -result_sorted[i]['x1']
					if width >= 1.5 * mean_width and width <= 2.5 * mean_width:
						append_char = copy.copy(result_sorted[i])
						#print(append_char)
						result_sorted[i]['x2'] = int(result_sorted[i]['x1'] + width / 2)
						result_sorted[i]['cmean'] -= 10
						result_sorted.append(append_char)
						#print(append_char)
						result_sorted[-1]['x1'] = int(result_sorted[-1]['x2'] - width / 2)
						result_sorted[-1]['cmean'] += 10
					elif width > 2.5 * mean_width <= 3.5 * mean_width:
						append_char = copy.copy(result_sorted[i])
						result_sorted[i]['x2'] = int(result_sorted[i]['x1'] + width / 3)
						result_sorted[i]['cmean'] -= 20
						result_sorted.append(append_char)
						result_sorted[-1]['x1'] = int(result_sorted[-1]['x1'] + width / 3)
						result_sorted[-1]['x2'] = int(result_sorted[-1]['x1'] + 2*(width/3))
						result_sorted.append(append_char)
						result_sorted[-1]['x1'] = int(result_sorted[-1]['x2'] - width / 3)
						result_sorted[i]['cmean'] += 20
					elif width > 3.5 * mean_width <= 4.5 * mean_width:
						append_char = copy.copy(result_sorted[i])
						result_sorted[i]['x2'] = int(result_sorted[i]['x1'] + width / 4)
						result_sorted[i]['cmean'] -= 30
						result_sorted.append(append_char)
						result_sorted[-1]['x1'] = int(result_sorted[-1]['x1'] + width / 4)
						result_sorted[-1]['x2'] = int(result_sorted[-1]['x1'] + 2*(width/4))
						result_sorted[i]['cmean'] -= 10
						result_sorted.append(append_char)
						result_sorted[-1]['x1'] = int(result_sorted[-1]['x2'] + 2*(width/4))
						result_sorted[-1]['x2'] = int(result_sorted[-1]['x1'] + 3*(width/4))
						result_sorted[i]['cmean'] += 20
						result_sorted.append(append_char)
						result_sorted[-1]['x1'] = int(result_sorted[-1]['x2'] + 3*(width/4))
						result_sorted[i]['cmean'] += 20
					pass
	# pprint.pprint(result_sorted)
	# result_sorted = sorted(result_sorted, key=lambda tmp:tmp['cmean'])
	result_sorted1 = sorted(result_sorted_temp, key=lambda tmp:tmp['rmean'])
	result_sorted2 = result_sorted1[:2]
	result_sorted2 = sorted(result_sorted2, key=lambda tmp:tmp['cmean'])
	result_sorted3 = result_sorted
	result_sorted3 = sorted(result_sorted3, key=lambda tmp:tmp['cmean'])

	result_sorted = result_sorted2+result_sorted3#mod

	if len(result_sorted) < 7 or result_sorted[0]['char'] == 5:
		# LP_dict['error'] = 1
		continue

	fig = plt.figure()
	subplot(211)
	imshow(perspective_roi)
	title('lp')
	subplot(212)
	imshow(semantic_map)
	title('semantic_map')
	show()
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	LP_dict['license_plate_number'] = {'num1':result_sorted[0]['char'], 'num2':result_sorted[1]['char'], 'num3':result_sorted[2]['char'], 'num4':result_sorted[3]['char'],
										'num5':result_sorted[4]['char'], 'num6':result_sorted[5]['char'], 'num7':result_sorted[6]['char']}
	LP_dict['num1_locations'] = {'x1':result_sorted[0]['x1'], 'y1':result_sorted[0]['y1'], 'x2':result_sorted[0]['x2'], 'y2':result_sorted[0]['y2']}
	LP_dict['num2_locations'] = {'x1':result_sorted[1]['x1'], 'y1':result_sorted[1]['y1'], 'x2':result_sorted[1]['x2'], 'y2':result_sorted[1]['y2']}
	LP_dict['num3_locations'] = {'x1':result_sorted[2]['x1'], 'y1':result_sorted[2]['y1'], 'x2':result_sorted[2]['x2'], 'y2':result_sorted[2]['y2']}
	LP_dict['num4_locations'] = {'x1':result_sorted[3]['x1'], 'y1':result_sorted[3]['y1'], 'x2':result_sorted[3]['x2'], 'y2':result_sorted[3]['y2']}
	LP_dict['num5_locations'] = {'x1':result_sorted[4]['x1'], 'y1':result_sorted[4]['y1'], 'x2':result_sorted[4]['x2'], 'y2':result_sorted[4]['y2']}
	LP_dict['num6_locations'] = {'x1':result_sorted[5]['x1'], 'y1':result_sorted[5]['y1'], 'x2':result_sorted[5]['x2'], 'y2':result_sorted[5]['y2']}
	LP_dict['num7_locations'] = {'x1':result_sorted[6]['x1'], 'y1':result_sorted[6]['y1'], 'x2':result_sorted[6]['x2'], 'y2':result_sorted[6]['y2']}
	
	with open('C:/Users/Administrator/Desktop/AirSimData/annotations/%s.json' % image_name[:-4], 'w') as f:
		f.write(json.dumps(LP_dict, sort_keys=True, indent=4, separators=(',', ':'), cls=UserEncoder))
	

	