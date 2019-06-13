import numpy as np
import glob
import os
import PIL.Image
from collections import Counter


color_select = [[255,255,0], [176,224,230], [255,97,0], [0,255,0], 
                [128,42,42], [255,0,0], [160,32,240], [255,255,255]]
def show_pos(img_path):
    with PIL.Image.open(img_path).convert('RGB') as image_pil:
        image_np = np.asarray(image_pil).astype(np.uint8)
        # print(image_np.shape)
        for xi in range(image_np.shape[0]):
            for yi in range(image_np.shape[1]):
                if image_np[xi, yi, 2] == 255:
                    image_np[xi, yi, :] = color_select[7]
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
                # else:
                #     image_np[xi, yi, 2] *= 10
        image_pil = PIL.Image.fromarray(image_np)
        image_pil.show()

def show_seg(img_path, char_path):
    result = []
    with PIL.Image.open(img_path) as image_pil:
        image_np = np.asarray(image_pil).astype(np.uint8)
        print(image_np.shape)
    char_list = np.loadtxt(char_path, dtype=int)
    # print(char_list)
    white_img = np.zeros((50, 160, 3))
    for idx, pixel in enumerate(char_list):
        print('index:{}, pixel:{}'.format(idx, pixel))
        for xi in range(image_np.shape[0]):
            for yi in range(image_np.shape[1]):
                result.append(image_np[xi, yi])
                if image_np[xi, yi] == pixel+1:
                    white_img[xi, yi, :] = color_select[idx]
                elif image_np[xi, yi] == 0:
                    white_img[xi, yi, :] = [0, 0, 0]
    print(Counter(result).most_common())
    show_img = PIL.Image.fromarray(np.uint8(white_img))
    show_img.show()


if __name__ == '__main__':
    # show_pos('00265.png')
    seg_path = 'pos_data'
    img_list = glob.glob(os.path.join(seg_path,'*.png'))
    img_list.sort()
    for each_img in img_list:
        show_pos(each_img)#, each_img.replace('png', 'txt'))