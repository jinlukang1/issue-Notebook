import numpy as np
import glob
import os
import PIL.Image
from collections import Counter
import cv2
from PIL import Image, ImageDraw, ImageFont


color_select = [[255,255,0], [176,224,230], [255,97,0], [0,255,0], 
                [128,42,42], [255,0,0], [160,32,240], [255,255,255]]
def get_pos(img_path):
    with PIL.Image.open(img_path).convert('RGB') as image_pil:
        image_np = np.asarray(image_pil).astype(np.uint8)
        # print(image_np.shape)
        for xi in range(image_np.shape[0]):
            for yi in range(image_np.shape[1]):
                if image_np[xi, yi, 2] == 0:
                    image_np[xi, yi, :] = 0
                elif image_np[xi, yi, 2] == 1:
                    image_np[xi, yi, :] = color_select[0]
                elif image_np[xi, yi, 2] == 2:
                    image_np[xi, yi, :] = color_select[1]
                elif image_np[xi, yi, 2] == 3:
                    image_np[xi, yi, :] = color_select[2]
                elif image_np[xi, yi, 2] == 4:
                    image_np[xi, yi, :] = color_select[3]
                elif image_np[xi, yi, 2] == 5:
                    image_np[xi, yi, :] = color_select[4]
                elif image_np[xi, yi, 2] == 6:
                    image_np[xi, yi, :] = color_select[5]
                elif image_np[xi, yi, 2] == 7:
                    image_np[xi, yi, :] = color_select[6]
                else:
                    image_np[xi, yi, :] = color_select[7]
        return image_np

def get_seg(img_path, char_path):
    result = []
    with PIL.Image.open(img_path) as image_pil:
        image_np = np.asarray(image_pil).astype(np.uint8)
    char_list = np.loadtxt(char_path, dtype=int)
    # print(char_list)
    white_img = np.zeros((50, 160, 3)).astype(np.uint8)
    # for idx, pixel in enumerate(char_list):
    #     print('index:{}, pixel:{}'.format(idx, pixel))
    for xi in range(image_np.shape[0]):
        for yi in range(image_np.shape[1]):
            # result.append(image_np[xi, yi])
            if image_np[xi, yi] == char_list[0]+1:
                white_img[xi, yi, :] = color_select[0]
            elif image_np[xi, yi] == char_list[1]+1:
                white_img[xi, yi, :] = color_select[1]
            elif image_np[xi, yi] == char_list[2]+1:
                white_img[xi, yi, :] = color_select[2]
            elif image_np[xi, yi] == char_list[3]+1:
                white_img[xi, yi, :] = color_select[3]
            elif image_np[xi, yi] == char_list[4]+1:
                white_img[xi, yi, :] = color_select[4]
            elif image_np[xi, yi] == char_list[5]+1:
                white_img[xi, yi, :] = color_select[5]
            elif image_np[xi, yi] == char_list[6]+1:
                white_img[xi, yi, :] = color_select[6]
            elif image_np[xi, yi] == 0:
                white_img[xi, yi, :] = [0, 0, 0]
            else:
                white_img[xi, yi, :] = color_select[7]
    # print(Counter(result).most_common())
    return white_img

def get_txt(char_path):
    char_table = {0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 10:"A", 11:"B",
    12:"C", 13:"D", 14:"E", 15:"F", 16:"G", 17:"H", 18:"J", 19:"K", 20:"L", 21:"M", 22:"N", 23:"P", 24:"Q",
    25:"R", 26:"S", 27:"T", 28:"U", 29:"V", 30:"W", 31:"X", 32:"Y", 33:"Z", 34:"皖", 35:"桂", 36:"贵", 37:"粤",
    38:"甘", 39:"京", 40:"冀", 41:"闽", 42:"渝", 43:"琼", 44:"吉", 45:"赣", 46:"豫", 47:"黑", 48:"湘", 49:"苏",
    50:"辽", 51:"蒙", 52:"宁", 53:"沪", 54:"浙", 55:"青", 56:"鄂", 57:"津", 58:"陕", 59:"新", 60:"鲁", 61:"云",
    62:"川", 63:"藏", 64:"晋"}
    font = ImageFont.truetype("C:/Users/Administrator/Desktop/songti.ttf", 30)
    fillColor = (255,0,0)
    position = (0, 0)

    content = []
    pred_list = np.loadtxt(char_path, dtype=int)
    for num in pred_list:
        content.append(char_table[num])
    text = "".join(content)
    if isinstance(text, str):
        text = text
        decoded = False
    else:
        text = text.decode(encoding)
        decoded = True

    white_img = np.zeros((50, 160, 3)).astype(np.uint8)
    img_PIL = Image.fromarray(white_img)
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, text, font=font, fill=fillColor)
    image_np = np.asarray(img_PIL)
    return image_np


if __name__ == '__main__':



    error_path = 'error/pos_else'
    img_list = glob.glob(os.path.join(error_path,'*im.png'))
    img_list.sort()
    for index, each_img in enumerate(img_list):
        gt_txt = each_img.replace('im.png', 'gt.txt')
        pred_txt = each_img.replace('im.png', 'pred.txt')

        gt_img = get_txt(gt_txt)
        pred_img = get_txt(pred_txt)


        with PIL.Image.open(each_img) as image_pil:
            ori_img = np.asarray(image_pil).astype(np.uint8)
        pos_img = get_pos(each_img.replace('im', 'pos'))
        seg_img = get_seg(each_img.replace('im', 'seg'), each_img.replace('im.png', 'gt.txt'))
        # print('ori_img:{}, pos_img:{}, seg_img:{}'.format(ori_img.shape, pos_img.shape, seg_img.shape))
        # white_img = np.zeros((100, 160, 3)).astype(np.uint8)
        print('prograss:{}/{}'.format(index, len(img_list)))
        vtitch = np.vstack((ori_img, pos_img, seg_img, gt_img, pred_img))


        show_img = vtitch
        cv2.namedWindow('image')
        show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
        cv2.imshow('image',show_img)
        cv2.imwrite(os.path.join('show_img/pos_else',str(index).zfill(5)+".png"), show_img)
        cv2.waitKey(1000)
    cv2.destroyAllWindows()