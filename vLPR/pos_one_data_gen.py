import numpy as np
import os, sys
import cv2
import tqdm


if __name__ == '__main__':
    np_char = np.load('npy_data/all/all_car_recorder_val_char.npy')
    np_pos = np.load('npy_data/all/all_car_recorder_val_pos.npy')
    np_img = np.load('npy_data/all/all_car_recorder_val_im.npy')
    print('img:{}, pos:{}, char:{}'.format(np_img.shape, np_pos.shape, np_char.shape))

    out_one_im = np.zeros([np_img.shape[0], 224, 224, 3], dtype=np.uint8)
    for i in tqdm.tqdm(range(np_img.shape[0])):
        pos_one_w, pos_one_h = np.where(np_pos[i, :, :] == 1)
        pos_one_img = np_img[i, np.min(pos_one_w):np.max(pos_one_w), np.min(pos_one_h):np.max(pos_one_h)]
        pos_one_img = cv2.resize(pos_one_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        # print(pos_one_img.shape)
        out_one_im[i, :, :, :] = pos_one_img
        cv2.namedWindow("Image")
        cv2.imshow("Image", pos_one_img)
        cv2.waitKey(1000)

    # np.save('npy_data/all/all_car_recorder_val_pos_one_img.npy', out_one_im)
    print(out_one_im.shape)


        # cv2.namedWindow("Image")
        # cv2.imshow("Image", pos_one_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()