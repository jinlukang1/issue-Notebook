import os
import numpy as np
import glob
import cv2

def wan_gen(spilt='val_without_night'):
    img_data = np.load(os.path.join('npy_data', spilt+'_im.npy'))
    pos_data = np.load(os.path.join('npy_data', spilt+'_pos.npy'))
    gt_data = np.load(os.path.join('npy_data', spilt+'_gt.npy'))
    char_data = np.load(os.path.join('npy_data', spilt+'_char.npy'))
    print(img_data.shape, pos_data.shape, gt_data.shape, char_data.shape)

    wan_img_data_list = []
    wan_pos_data_list = []
    wan_gt_data_list = []
    wan_char_data_list = []

    for index in range(img_data.shape[0]):
        if char_data[index][0][0] == 34:
            wan_img_data_list.append(img_data[index])
            wan_pos_data_list.append(pos_data[index])
            wan_gt_data_list.append(gt_data[index])
            wan_char_data_list.append(char_data[index])

    wan_img_data = np.asarray(wan_img_data_list)
    wan_pos_data = np.asarray(wan_pos_data_list)
    wan_gt_data = np.asarray(wan_gt_data_list)
    wan_char_data = np.asarray(wan_char_data_list)
    print(wan_img_data.shape, wan_pos_data.shape, wan_gt_data.shape, wan_char_data.shape)

    return wan_img_data, wan_pos_data, wan_gt_data, wan_char_data
    # cv2.namedWindow('image')
    # cv2.imshow('image',wan_img_data[0])
    # print(wan_char_data[0])
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()

if __name__ == '__main__':
    out_path = 'npy_data'
    data_type = 'angle0_without_night'

    val_img = np.load(os.path.join(out_path, 'val_angle0_without_night_im.npy'))
    val_gt = np.load(os.path.join(out_path, 'val_angle0_without_night_gt.npy'))
    val_pos = np.load(os.path.join(out_path, 'val_angle0_without_night_pos.npy'))
    val_char = np.load(os.path.join(out_path, 'val_angle0_without_night_char.npy'))

    train_img = np.load(os.path.join(out_path, 'train_angle0_without_night_im.npy'))
    train_gt = np.load(os.path.join(out_path, 'train_angle0_without_night_gt.npy'))
    train_pos = np.load(os.path.join(out_path, 'train_angle0_without_night_pos.npy'))
    train_char = np.load(os.path.join(out_path, 'train_angle0_without_night_char.npy'))

    test_img = np.load(os.path.join(out_path, 'test_angle0_without_night_im.npy'))
    test_gt = np.load(os.path.join(out_path, 'test_angle0_without_night_gt.npy'))
    test_pos = np.load(os.path.join(out_path, 'test_angle0_without_night_pos.npy'))
    test_char = np.load(os.path.join(out_path, 'test_angle0_without_night_char.npy'))

    print(val_img.shape, train_img.shape, test_img.shape)
    # val_img, val_pos, val_gt, val_char = wan_gen('val_without_night')
    # test_img, test_pos, test_gt, test_char = wan_gen('test_without_night')
    # train_img, train_pos, train_gt, train_char = wan_gen('train_without_night')

    wan_img_data = np.concatenate((train_img, val_img, test_img), axis=0)
    wan_pos_data = np.concatenate((train_pos, val_pos, test_pos), axis=0)
    wan_gt_data = np.concatenate((train_gt, val_gt, test_gt), axis=0)
    wan_char_data = np.concatenate((train_char, val_char, test_char), axis=0)

    print(wan_img_data.shape, wan_pos_data.shape, wan_gt_data.shape, wan_char_data.shape)

    np.save(os.path.join(out_path, data_type + '_im.npy'), wan_img_data)
    np.save(os.path.join(out_path, data_type + '_gt.npy'), wan_gt_data)
    np.save(os.path.join(out_path, data_type + '_pos.npy'), wan_pos_data)
    np.save(os.path.join(out_path, data_type + '_char.npy'), wan_char_data)

    # real_img_data = np.load(os.path.join('real', 'real_train_im.npy'))
    # real_char_data = np.load(os.path.join('real', 'real_train_char.npy'))
    # print(real_img_data.shape)





    # real_img_list = glob.glob('real_image/*.png')
    # print(len(real_img_list))
    # out_im = np.zeros([len(real_img_list), 50, 160, 3], dtype=np.uint8)

    # for i, name in enumerate(real_img_list):
    #     img = cv2.imread(name)
    #     img = cv2.resize(img,(160,50),interpolation=cv2.INTER_CUBIC)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     out_im[i, :, :, :] = img

    # print(out_im.shape)
    # np.save('real_image_train.npy', out_im)


    # cv2.namedWindow('image')
    # cv2.imshow('image',cv2.cvtColor(out_im[1000], cv2.COLOR_BGR2RGB))
    # # print(out_im[0])
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()
        # break

