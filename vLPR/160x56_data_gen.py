import numpy as np
import os
import glob
import cv2


if __name__ == '__main__':
    virtrual_data = np.load('./npy_data/train_without_night_im.npy')
    real_data_list = glob.glob('./real_image/*.png')
    real_numpy = np.zeros([len(real_data_list), 56, 160, 3], dtype=np.uint8)
    r_list = []
    v_list = []
    for i, each_path in enumerate(real_data_list):
        img = cv2.imread(each_path)
        img = cv2.resize(img,(160,56),interpolation=cv2.INTER_CUBIC)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # real_numpy[i, :, :, :] = img
        r_list.append(img)

    for j in range(virtrual_data.shape[0]):
        each_img = virtrual_data[j]
        each_img = cv2.resize(each_img,(160,56),interpolation=cv2.INTER_CUBIC)
        each_img = cv2.cvtColor(each_img, cv2.COLOR_BGR2RGB)
        v_list.append(each_img)
        if j == 5000-1:
            break
        # print(each_img.shape)
    # real_list = 
    # print(virtrual_data.shape)
    for index, each_np_pic in enumerate(r_list[:int(0.2*len(r_list))]):
        cv2.imwrite(os.path.join('./v2r/testB', str(index).zfill(5) + '.png'), each_np_pic)
        # break
    print('testB is done!')
    for index, each_np_pic in enumerate(r_list[int(0.2*len(r_list)):]):
        cv2.imwrite(os.path.join('./v2r/trainB', str(index).zfill(5) + '.png'), each_np_pic)
        # break
    print('trainB is done!')
    for index, each_np_pic in enumerate(v_list[:int(0.2*len(r_list))]):
        cv2.imwrite(os.path.join('./v2r/testA', str(index).zfill(5) + '.png'), each_np_pic)
        # break
    print('testA is done!')
    for index, each_np_pic in enumerate(v_list[int(0.2*len(r_list)):]):
        cv2.imwrite(os.path.join('./v2r/trainA', str(index).zfill(5) + '.png'), each_np_pic)
        # break
    print('trainA is done!')

