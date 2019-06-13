import numpy as np
import glob
import cv2
import os
import json
import tqdm


# def real():
#     split_list = ['train_aug']
#     data_dir = r'C:\Users\Administrator\Desktop\vLPR experiment\Car_test_data'
#     for split in split_list:
#         im_path = os.path.join(data_dir, split, 'image')
#         gt_path = os.path.join(data_dir, split, 'label')
#         pos_mask_path = os.path.join(data_dir, split, 'pos_mask')
#         char_path = os.path.join(data_dir, split, 'char')
#         file_path = os.path.join(data_dir, 'list', split + '.txt')
#         out_path = os.path.join(data_dir, split, 'npy')

#         with open(file_path) as f:
#             names = f.readlines()

#         out_im = np.zeros([len(names), 50, 160, 3], dtype=np.uint8)
#         out_gt = np.zeros([len(names), 50, 160], dtype=np.uint8)
#         out_pos = np.zeros([len(names), 50, 160], dtype=np.uint8)
#         out_char = np.zeros([len(names), 1, 7], dtype=np.uint8)
#         for i, name in enumerate(names):
#             print(i)
#             name = name[:-1]
#             im = cv2.imread(os.path.join(im_path, name + '.png'))
#             im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#             out_im[i, :, :, :] = im
#             gt = cv2.imread(os.path.join(gt_path, name + '.png'), 0)
#             out_gt[i, :, :] = gt
#             pos_mask = cv2.imread(os.path.join(
#                 pos_mask_path, name + '.png'), 0)
#             out_pos[i, :, :] = pos_mask
#             char = np.loadtxt(os.path.join(char_path, name + '.txt'))
#             out_char[i, 0, :] = char

#         np.save(os.path.join(out_path, split + '_im.npy'), out_im)
#         np.save(os.path.join(out_path, split + '_gt.npy'), out_gt)
#         np.save(os.path.join(out_path, split + '_pos.npy'), out_pos)
#         np.save(os.path.join(out_path, split + '_char.npy'), out_char)

def npy_gen(data_type = None):
    out_path = 'npy_data'
    # data_type = 'test_val'
    with open("Imagelist.txt") as f:
        names = f.readlines()
        # print(names)

        out_im = np.zeros([len(names), 50, 160, 3], dtype=np.uint8)
        out_gt = np.zeros([len(names), 50, 160], dtype=np.uint8)
        out_pos = np.zeros([len(names), 50, 160], dtype=np.uint8)
        out_char = np.zeros([len(names), 1, 7], dtype=np.uint8)

        for i, name in tqdm.tqdm(enumerate(names)):
            name = name[:-1]

            img = cv2.imread(name)
            img = cv2.resize(img,(160,50),interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out_im[i, :, :, :] = img

            gt = cv2.imread(name.replace('raw', 'seg_anno'))
            for xi in range(gt.shape[0]):
                for yi in range(gt.shape[1]):
                    if gt[xi, yi, 2] == 255:
                        gt[xi, yi, :] = 0
                    else:
                        gt[xi, yi, :] += 1
            gt = cv2.resize(gt,(160,50), interpolation=cv2.INTER_NEAREST)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            out_gt[i, :, :] = gt[:, :, 2]

            pos_mask = cv2.imread(name.replace('raw', 'pos_anno'))
            for xi in range(pos_mask.shape[0]):
                for yi in range(pos_mask.shape[1]):
                    if pos_mask[xi, yi, 2] == 255:
                        pos_mask[xi, yi, :] = 0
            pos_mask = cv2.resize(pos_mask,(160,50), interpolation=cv2.INTER_NEAREST)
            pos_mask = cv2.cvtColor(pos_mask, cv2.COLOR_BGR2RGB)
            out_pos[i, :, :] = pos_mask[:, :, 2]

            with open(name.replace('raw', 'new_annotations').replace('png', 'json'), "r") as f:
                data_annotation = f.read()
                LP_dict = json.loads(data_annotation)
                for j, num in enumerate(LP_dict['license_plate_number'].keys()):
                    out_char[i, 0, j] = LP_dict['license_plate_number'][num]

        np.save(os.path.join(out_path, data_type + '_im.npy'), out_im)
        np.save(os.path.join(out_path, data_type + '_gt.npy'), out_gt)
        np.save(os.path.join(out_path, data_type + '_pos.npy'), out_pos)
        np.save(os.path.join(out_path, data_type + '_char.npy'), out_char)


if __name__ == "__main__":
    npy_gen()
