import cv2
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont


# DATAROOT = "d:/Documents/AirSim/final_dataset/scene1/noontime_foggy/"
# DATAROOT = "E:/dataset/final_dataset/front_noontime_lightrain/"
DATAROOT = r"D:\Documents\AirSim\BigNew\scene3\back_night_heavyrain/"
# DATAROOT = "C:/Users/Administrator/Desktop/AirSimData/"
DATASAVEROOT = "C:/Users/Administrator/Desktop/AirSimData/img/"
filename = "0"+str(np.random.randint(1,1500)).zfill(4)+".png"
# filename = "00674.png"
print(filename)

img = cv2.imread(DATAROOT + "raw/" + filename)
cv2.namedWindow("images")
# cv2.imshow("images", img)
# cv2.waitKey(0)
img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

char_table = {0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 10:"A", 11:"B",
 12:"C", 13:"D", 14:"E", 15:"F", 16:"G", 17:"H", 18:"J", 19:"K", 20:"L", 21:"M", 22:"N", 23:"P", 24:"Q",
  25:"R", 26:"S", 27:"T", 28:"U", 29:"V", 30:"W", 31:"X", 32:"Y", 33:"Z", 34:"皖", 35:"桂", 36:"贵", 37:"粤",
   38:"甘", 39:"京", 40:"冀", 41:"闽", 42:"渝", 43:"琼", 44:"吉", 45:"赣", 46:"豫", 47:"黑", 48:"湘", 49:"苏",
    50:"辽", 51:"蒙", 52:"宁", 53:"沪", 54:"浙", 55:"青", 56:"鄂", 57:"津", 58:"陕", 59:"新", 60:"鲁", 61:"云",
     62:"川", 63:"藏", 64:"晋", 66:"挂"}

license_plate_content = []



with open(DATAROOT+"annotations/"+filename[:-3] + "json", "r") as f:
	annotations = f.read()
	LP_dict = json.loads(annotations)
	
	font = ImageFont.truetype("C:/Users/Administrator/Desktop/songti.ttf", 40)
	fillColor = (255,255,255)
	position = (LP_dict['vertices_locations']['x2'], LP_dict['vertices_locations']['y2'])

	license_plate_content.append('license_plate_number:\n')
	for num in LP_dict['license_plate_number']:
		license_plate_content.append(char_table[LP_dict['license_plate_number'][num]])
	#license_plate_content.append('\nangle:\n')
	#license_plate_content.append(str(LP_dict['angle']))
	#license_plate_content.append('\ndistance:\n')
	#license_plate_content.append(str(LP_dict['distance']))
	
	print(license_plate_content)
	text = "".join(license_plate_content)
	if isinstance(text, str):
		text = text
		decoded = False
	else:
		text = text.decode(encoding)
		decoded = True
	draw = ImageDraw.Draw(img_PIL)
	draw.text(position, text, font=font, fill=fillColor)
	img_OpenCV = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)

	cv2.line(img_OpenCV, (LP_dict['vertices_locations']['x1'], LP_dict['vertices_locations']['y1']),
		(LP_dict['vertices_locations']['x2'], LP_dict['vertices_locations']['y2']), (0,0,255), 2)
	cv2.line(img_OpenCV, (LP_dict['vertices_locations']['x2'], LP_dict['vertices_locations']['y2']),
		(LP_dict['vertices_locations']['x3'], LP_dict['vertices_locations']['y3']), (0,0,255), 2)
	cv2.line(img_OpenCV, (LP_dict['vertices_locations']['x3'], LP_dict['vertices_locations']['y3']),
		(LP_dict['vertices_locations']['x4'], LP_dict['vertices_locations']['y4']), (0,0,255), 2)
	cv2.line(img_OpenCV, (LP_dict['vertices_locations']['x4'], LP_dict['vertices_locations']['y4']),
		(LP_dict['vertices_locations']['x1'], LP_dict['vertices_locations']['y1']), (0,0,255), 2)
	
	img_OpenCV = cv2.resize(img_OpenCV,(1280,720),interpolation=cv2.INTER_CUBIC)

	cv2.imshow("images", img_OpenCV)
	#cv2.imwrite(DATASAVEROOT + filename, img)
	cv2.waitKey(0)

cv2.destroyAllWindows()
