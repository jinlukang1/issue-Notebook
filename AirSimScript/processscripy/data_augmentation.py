# -*- coding:utf-8 -*-
import os, sys
import numpy as np
import random
from tqdm import tqdm
import cv2

#车牌数据的数据增强方式
#噪声
class random_noise(object):
    def __init__(self):
       self.noise = 1

    def __call__(self, im, gt, pos):
        im = im.copy()

        # im = self.gauss_noise(im)#有问题
        im = self.salt_and_pepper_noise(im)
        return im, gt, pos

    def gauss_noise(self, image, mean=0, var=0.001):
        '''
            添加高斯噪声
            mean : 均值
            var : 方差
        '''
        image = np.array(image / 255., dtype=float)
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out * 255)
        # cv.imshow("gasuss", out)
        return out

    def salt_and_pepper_noise(self, img, proportion=0.01):
        noise_img = img
        height, width = noise_img.shape[0], noise_img.shape[1]
        num = int(height * width * proportion)  # 多少个像素点添加椒盐噪声
        for i in range(num):
            w = random.randint(0, width - 1)
            h = random.randint(0, height - 1)
            if random.randint(0, 1) == 0:
                noise_img[h, w] = 0
            else:
                noise_img[h, w] = 255

        return noise_img


#对比度
class random_contrast(object):
    def __init__(self, threhold):
        self.threhold = threhold
    def __call__(self, np_img, gt, pos):
        contrast = self.threhold
        # print(contrast)
        mean = np.mean(np_img.copy())
        np_img = np_img - mean
        np_img = np.clip(np_img + mean*contrast, 0, 255)
        np_img = np_img.astype(np.uint8)
        return np_img, gt, pos


#高斯模糊
class gaussian_blurry(object):
    def __init__(self, ksize = (9,9), sigemaX=0, sigemaY=0):
        self.sigemaX = sigemaX
        self.sigemaY = sigemaY
        self.ksize = ksize
    def __call__(self, np_img, gt, pos):
        blur_img = np_img.copy()
        blur_img = cv2.GaussianBlur(blur_img, ksize = self.ksize, sigmaX = self.sigemaX, sigmaY = self.sigemaY)
        return blur_img, gt, pos

#运动模糊
class motion_blurry(object):
    def __init__(self, degree=12, angle=45):
        self.degree = degree
        self.angle = angle
    def __call__(self, np_img, gt, pos):
        M = cv2.getRotationMatrix2D((self.degree / 2, self.degree / 2), self.angle, 1)
        motion_blur_kernel = np.diag(np.ones(self.degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (self.degree, self.degree))
        motion_blur_kernel = motion_blur_kernel / self.degree
        blur_img = np_img.copy()
        blur_img = cv2.filter2D(blur_img, -1, motion_blur_kernel)
        cv2.normalize(blur_img, blur_img, 0, 255, cv2.NORM_MINMAX)
        blur_img = np.array(blur_img, dtype=np.uint8)

        return blur_img, gt, pos



#遮挡
class random_occlusion(object):
    def __init__(self, threhold):
        self.threhold = threhold
    def __call__(self, np_img, gt, pos):
        occ_img = np_img.copy()
        num = self.threhold
        height, width = occ_img.shape[0], occ_img.shape[1]
        for i in range(num):
            w = random.randint(0, width - 10)
            h = random.randint(0, height - 10)
            color = random.randint(0,255)
            for j in range(10):
                for k in range(10):
                    occ_img[h+k, w+j] = color
        return occ_img, gt, pos

#亮度
class random_bright(object):
    def __init__(self, threhold):
        self.threhold = threhold
    def __call__(self, np_img, gt, pos):
        bright = self.threhold
        np_img = np.clip(np_img.copy() * bright, 0, 255)
        np_img = np_img.astype(np.uint8)
        return np_img, gt, pos

class ori(object):
    def __init__(self, threhold=0):
        self.threhold = threhold
    def __call__(self, np_img, gt, pos):
        return np_img, gt, pos

#显示图片       
def show_img(np_img):
    cv2.namedWindow("Image")
    h, w, c = np_img.shape
    np_img = np_img[...,::-1]
    np_img = cv2.resize(np_img, (5*w, 5*h), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Image", np_img)
    cv2.waitKey (0)
    cv2.destroyAllWindows()

class random_crop(object):
    def __init__(self, threhold=8):
        self.threhold = threhold
    def __call__(self, np_img, gt, pos):
        crop = random.randint(3,self.threhold)#左闭右闭
        select = random.randint(1,4)
        h,w,c = np_img.shape
        if select == 1:
            np_img = np_img[:,crop:,:]
            gt = gt[:,crop:]
            pos = pos[:,crop:]
            np_img = cv2.resize(np_img, (w, h), interpolation=cv2.INTER_CUBIC)
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
            pos = cv2.resize(pos, (w, h), interpolation=cv2.INTER_NEAREST)
        elif select == 2:
            np_img = np_img[crop:,:,:]
            gt = gt[crop:,:]
            pos = pos[crop:,:]
            np_img = cv2.resize(np_img, (w, h), interpolation=cv2.INTER_CUBIC)
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
            pos = cv2.resize(pos, (w, h), interpolation=cv2.INTER_NEAREST)
        elif select == 3:
            np_img = np_img[:,:-crop,:]
            gt = gt[:,:-crop]
            pos = pos[:,:-crop]
            np_img = cv2.resize(np_img, (w, h), interpolation=cv2.INTER_CUBIC)
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
            pos = cv2.resize(pos, (w, h), interpolation=cv2.INTER_NEAREST)
        elif select == 4:
            np_img = np_img[:-crop,:,:]
            gt = gt[:-crop,:]
            pos = pos[:-crop,:]
            np_img = cv2.resize(np_img, (w, h), interpolation=cv2.INTER_CUBIC)
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
            pos = cv2.resize(pos, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            return np_img, gt, pos
        return np_img, gt, pos

class random_bg(object):
    def __init__(self, threhold=100):
        self.threhold = threhold
    def __call__(self, np_img):
        point_list = self.getPointList(np_img)
        for [W, H] in point_list:
            w_roi, h_roi, c = self.mask.shape
            roi = np_img[W:W+w_roi, H:H+h_roi]
            img_bg1 = cv2.bitwise_and(roi, roi, mask=self.mask[:, :, 0])
            img_fg2 = cv2.bitwise_and(self.crop_img, self.crop_img, mask=self.mask_inv[:, :, 0])
            dst = cv2.add(img_bg1, img_fg2)
            np_img[W:W+w_roi, H:H+h_roi] = dst

        return np_img
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.imshow('image', np_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def getPointList(self, np_img):
        w, h, c = np_img.shape
        w_num, h_num = int(w/100), int(h/100)
        point_list = []
        for i in range(w_num):
            for j in range(h_num):
                point_list.append([i*100, j*100])
        return point_list



    def getBG(self, path='./', x=0, y=0, w=0, h=0):
        bg_img = cv2.imread(path)
        crop_img = bg_img[y:y+h, x:x+w, :]
        self.crop_img = crop_img
        _, self.mask = cv2.threshold(self.crop_img, 127, 255, cv2.THRESH_BINARY)
        self.mask_inv = cv2.bitwise_not(self.mask)

if __name__ == '__main__':
    npy_data_img = r'C:\Users\Administrator\Desktop\vLPR experiment\npy_data\without_night/train_without_night_im.npy'
    npy_data_char = r'C:\Users\Administrator\Desktop\vLPR experiment\npy_data\without_night/train_without_night_char.npy'
    npy_data_pos = r'C:\Users\Administrator\Desktop\vLPR experiment\npy_data\without_night/train_without_night_pos.npy'
    npy_data_gt = r'C:\Users\Administrator\Desktop\vLPR experiment\npy_data\without_night/train_without_night_gt.npy'
    np_imgs = np.load(npy_data_img)
    np_gts = np.load(npy_data_gt)
    np_chars = np.load(npy_data_char)
    np_poses = np.load(npy_data_pos)
    
    test_index = 10

    np_test_img = np_imgs[test_index]
    np_test_gt = np_gts[test_index]
    np_test_pos = np_poses[test_index]

    

    # out_im = np.zeros([np_imgs.shape[0]*7, 50, 160, 3], dtype=np.uint8)
    # out_gt = np.zeros([np_imgs.shape[0]*7, 50, 160], dtype=np.uint8)
    # out_pos = np.zeros([np_imgs.shape[0]*7, 50, 160], dtype=np.uint8)
    # out_char = np.zeros([np_imgs.shape[0]*7, 1, 7], dtype=np.uint8)

    # for i in tqdm(range(np_imgs.shape[0])):
    #     # numpy.random.seed(5)
    #     each_np_img = np_imgs[i]
    #     each_np_gt = np_gts[i]
    #     each_np_char = np_chars[i]
    #     each_np_pos = np_poses[i]
    #     data_orim = ori()
    #     data_bright = random_bright(threhold=3*np.random.rand())
    #     data_contrast = random_contrast(threhold=2*np.random.rand())
    #     data_noise = random_noise()
    #     data_occlusion = random_occlusion(threhold=random.randint(1,9))
    #     random_ksize = (random.randint(2,18) // 4) * 2 + 1
    #     data_gaussblur = gaussian_blurry(ksize = (random_ksize,random_ksize), sigemaX=0, sigemaY=0)
    #     data_motionblur = motion_blurry(degree=random.randint(1,8), angle=random.randint(0,90))
    #     data_transforms = [data_orim, data_bright, data_contrast, data_noise, data_occlusion, data_gaussblur, data_motionblur]
    #     # print(each_np_img.shape)
    #     for index, each_transform in enumerate(data_transforms):
    #         trans_data, _ = each_transform(each_np_img, each_np_gt)
    #         # show_img(trans_data)
    #         out_im[i*7+index, :, :, :] = trans_data
    #         out_gt[i*7+index, :, :] = each_np_gt
    #         out_pos[i*7+index, :, :] = each_np_pos
    #         out_char[i*7+index] = each_np_char

    # print('im:{}, gt:{}, pos:{}, char:{}'.format(out_im.shape, out_gt.shape, out_pos.shape, out_char.shape))
    # out_path = 'npy_data_with_aug'
    # data_type = 'all_car_recorder_with_aug_val'
    # np.save(os.path.join(out_path, data_type + '_im.npy'), out_im)
    # np.save(os.path.join(out_path, data_type + '_gt.npy'), out_gt)
    # np.save(os.path.join(out_path, data_type + '_pos.npy'), out_pos)
    # np.save(os.path.join(out_path, data_type + '_char.npy'), out_char)


    # data_bright = random_bright(threhold=3*np.random.rand())
    # data_contrast = random_contrast(threhold=2*np.random.rand())
    # data_noise = random_noise()
    # data_occlusion = random_occlusion(threhold=random.randint(1,9))
    # random_ksize = (random.randint(2,18) // 4) * 2 + 1
    # data_gaussblur = gaussian_blurry(ksize = (random_ksize,random_ksize), sigemaX=0, sigemaY=0)
    # data_motionblur = motion_blurry(degree=random.randint(1,8), angle=random.randint(90,135))

    # bright_img, _ = data_bright(np_test_img, np_test_gt)
    # contrast_img, _ = data_contrast(np_test_img, np_test_gt)
    # noise_img, _ = data_noise(np_test_img, np_test_gt)
    # occl_img, _ = data_occlusion(np_test_img, np_test_gt)
    # Gblur_img, _ = data_gaussblur(np_test_img, np_test_gt)
    # Mblur_img, _ = data_motionblur(np_test_img, np_test_gt)

    data_crop = random_crop()
    croped_img , crop_gt, croped_pos = data_crop(np_test_img, np_test_gt, np_test_pos)

    crop_gt = crop_gt[:,:,np.newaxis]
    croped_pos = croped_pos[:,:,np.newaxis]
    gt = np.concatenate((crop_gt, crop_gt, crop_gt), axis=-1)
    pos = np.concatenate((croped_pos, croped_pos, croped_pos), axis=-1)
    show_im = np.vstack((gt, pos, croped_img))
    print(show_im.shape)
    show_img(show_im)