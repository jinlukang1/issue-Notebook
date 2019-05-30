import os,shutil
import json
import cv2
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont

roots = []
roots.append(r"G:\dataset\Car_dataset\noontime\sc1_foggy")
# roots.append(r"G:\reco_dataset\BigNewEnergy_reco_dataset\night\sc1_back_night_heavyrain")
filename = "0"+str(np.random.randint(1,200)).zfill(4)+".png"
# filename = "00555.png"

length = 1600.0
width = 500.0

def cal(vertices_locations, x, y):
    vertices_x1 = vertices_locations['x1']
    vertices_x2 = vertices_locations['x2']
    vertices_x3 = vertices_locations['x3']
    vertices_x4 = vertices_locations['x4']
    vertices_y1 = vertices_locations['y1']
    vertices_y2 = vertices_locations['y2']
    vertices_y3 = vertices_locations['y3']
    vertices_y4 = vertices_locations['y4']
    R1 = vertices_x1+(vertices_x2-vertices_x1)*x/length
    R2 = vertices_x4+(vertices_x3-vertices_x4)*x/length
    x_fin = R1*(width-y)/width+R2*y/width
    C1 = vertices_y1+(vertices_y4-vertices_y1)*y/width
    C2 = vertices_y2+(vertices_y3-vertices_y2)*y/width
    y_fin = C1*(length-x)/length+C2*x/length
    return int(round(x_fin)), int(round(y_fin))


def location_transfer(vertices_locations={}, num_location={}):#为了抹消之前四舍五入的影响简单做了些数值上的修改
    num_x1 = num_location['x1']-10
    num_y1 = num_location['y1']-10
    num_x2 = num_location['x2']+10
    num_y2 = num_location['y2']+10
    num_x1_fin, num_y1_fin = cal(vertices_locations, num_x1, num_y1)
    num_x3_fin, num_y3_fin = cal(vertices_locations, num_x2, num_y2)
    num_x2_fin, num_y2_fin = cal(vertices_locations, num_x2, num_y1)
    num_x4_fin, num_y4_fin = cal(vertices_locations, num_x1, num_y2)
    new_dict = {'x1':num_x1_fin-0, 'y1':num_y1_fin-0, 'x2':num_x2_fin+0, 'y2':num_y2_fin-0, 
    'x3':num_x3_fin+0, 'y3':num_y3_fin+0, 'x4':num_x4_fin-0, 'y4':num_y4_fin+0}
    return new_dict

def loaction_modify(num_location={}):
    num_x1 = num_location['x1']
    num_x2 = num_location['x2']
    num_x3 = num_location['x3']
    num_x4 = num_location['x4']
    num_y1 = num_location['y1']
    num_y2 = num_location['y2']
    num_y3 = num_location['y3']
    num_y4 = num_location['y4']
    angle = (y2-y1)/(x2-x1)

def draw_line(img_OpenCV, num_location):
    cv2.line(img_OpenCV, (num_location['x1'], num_location['y1']),
        (num_location['x2'], num_location['y2']), (0,0,255), 1)
    cv2.line(img_OpenCV, (num_location['x2'], num_location['y2']),
        (num_location['x3'], num_location['y3']), (0,0,255), 1)
    cv2.line(img_OpenCV, (num_location['x3'], num_location['y3']),
        (num_location['x4'], num_location['y4']), (0,0,255), 1)
    cv2.line(img_OpenCV, (num_location['x4'], num_location['y4']),
        (num_location['x1'], num_location['y1']), (0,0,255), 1)


def main():
    for root in roots:
        with open(root+"/new_annotations/"+filename[:-3] + "json", "r") as f:
            annotations = f.read()
            LP_dict = json.loads(annotations)
            img_OpenCV = cv2.imread(root+"/raw/"+filename)
            # for key in LP_dict.keys():
                # if 'num' in key and 'locations' in key:
                    # LP_dict[key] = location_transfer(LP_dict['vertices_locations'], LP_dict[key])
            for key in LP_dict.keys():
                if 'locations' in key:
                    draw_line(img_OpenCV, LP_dict[key])
            print(filename)
            # img_OpenCV = img_OpenCV[min(LP_dict['vertices_locations']['y1'], LP_dict['vertices_locations']['y2']):max(LP_dict['vertices_locations']['y3'], LP_dict['vertices_locations']['y4']),
            # min(LP_dict['vertices_locations']['x1'], LP_dict['vertices_locations']['x4']):max(LP_dict['vertices_locations']['x2'], LP_dict['vertices_locations']['x3'])]
            print(LP_dict['vertices_locations']['x1'], LP_dict['vertices_locations']['x3'], LP_dict['vertices_locations']['y1'], LP_dict['vertices_locations']['y3'])
            # img_OpenCV = img_OpenCV[LP_dict['vertices_locations']['y1']:LP_dict['vertices_locations']['y3'], LP_dict['vertices_locations']['x1']:LP_dict['vertices_locations']['x3']]
            # h, w, c = img_OpenCV.shape
            # img_OpenCV = cv2.resize(img_OpenCV,(5*w, 5*h),interpolation=cv2.INTER_CUBIC)

            cv2.imshow("images", img_OpenCV)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()