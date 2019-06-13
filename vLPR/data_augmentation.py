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

    def __call__(self, im, gt):
        im = im.copy()

        # im = self.gauss_noise(im)#有问题
        im = self.salt_and_pepper_noise(im)
        return im, gt

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
    def __call__(self, np_img, gt):
        contrast = self.threhold
        # print(contrast)
        mean = np.mean(np_img.copy())
        np_img = np_img - mean
        np_img = np.clip(np_img + mean*contrast, 0, 255)
        np_img = np_img.astype(np.uint8)
        return np_img, gt


#高斯模糊
class gaussian_blurry(object):
    def __init__(self, ksize = (9,9), sigemaX=0, sigemaY=0):
        self.sigemaX = sigemaX
        self.sigemaY = sigemaY
        self.ksize = ksize
    def __call__(self, np_img, gt):
        blur_img = np_img.copy()
        blur_img = cv2.GaussianBlur(blur_img, ksize = self.ksize, sigmaX = self.sigemaX, sigmaY = self.sigemaY)
        return blur_img, gt

#运动模糊
class motion_blurry(object):
    def __init__(self, degree=12, angle=45):
        self.degree = degree
        self.angle = angle
    def __call__(self, np_img, gt):
        M = cv2.getRotationMatrix2D((self.degree / 2, self.degree / 2), self.angle, 1)
        motion_blur_kernel = np.diag(np.ones(self.degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (self.degree, self.degree))
        motion_blur_kernel = motion_blur_kernel / self.degree
        blur_img = np_img.copy()
        blur_img = cv2.filter2D(blur_img, -1, motion_blur_kernel)
        cv2.normalize(blur_img, blur_img, 0, 255, cv2.NORM_MINMAX)
        blur_img = np.array(blur_img, dtype=np.uint8)

        return blur_img, gt



#遮挡
class random_occlusion(object):
    def __init__(self, threhold):
        self.threhold = threhold
    def __call__(self, np_img, gt):
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
        return occ_img, gt

#亮度
class random_bright(object):
    def __init__(self, threhold):
        self.threhold = threhold
    def __call__(self, np_img, gt):
        bright = self.threhold
        np_img = np.clip(np_img.copy() * bright, 0, 255)
        np_img = np_img.astype(np.uint8)
        return np_img, gt

class ori(object):
    def __init__(self, threhold=0):
        self.threhold = threhold
    def __call__(self, np_img, gt):
        return np_img, gt

#显示图片       
def show_img(np_img):
    cv2.namedWindow("Image")
    h, w, c = np_img.shape
    np_img = np_img[...,::-1]
    np_img = cv2.resize(np_img, (5*w, 5*h), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Image", np_img)
    cv2.waitKey (0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    npy_data_img = 'npy_data/all/all_car_recorder_val_im.npy'
    npy_data_char = 'npy_data/all/all_car_recorder_val_char.npy'
    npy_data_pos = 'npy_data/all/all_car_recorder_val_pos.npy'
    npy_data_gt = 'npy_data/all/all_car_recorder_val_gt.npy'
    np_imgs = np.load(npy_data_img)
    np_gts = np.load(npy_data_gt)
    np_chars = np.load(npy_data_char)
    np_poses = np.load(npy_data_pos)

    np_test_img = np_imgs[5000]
    np_test_gt = np_gts[10000]

    out_im = np.zeros([np_imgs.shape[0]*7, 50, 160, 3], dtype=np.uint8)
    out_gt = np.zeros([np_imgs.shape[0]*7, 50, 160], dtype=np.uint8)
    out_pos = np.zeros([np_imgs.shape[0]*7, 50, 160], dtype=np.uint8)
    out_char = np.zeros([np_imgs.shape[0]*7, 1, 7], dtype=np.uint8)

    for i in tqdm(range(np_imgs.shape[0])):
        # numpy.random.seed(5)
        each_np_img = np_imgs[i]
        each_np_gt = np_gts[i]
        each_np_char = np_chars[i]
        each_np_pos = np_poses[i]
        data_orim = ori()
        data_bright = random_bright(threhold=3*np.random.rand())
        data_contrast = random_contrast(threhold=2*np.random.rand())
        data_noise = random_noise()
        data_occlusion = random_occlusion(threhold=random.randint(1,9))
        random_ksize = (random.randint(2,18) // 4) * 2 + 1
        data_gaussblur = gaussian_blurry(ksize = (random_ksize,random_ksize), sigemaX=0, sigemaY=0)
        data_motionblur = motion_blurry(degree=random.randint(1,9), angle=random.randint(0,90))
        data_transforms = [data_orim, data_bright, data_contrast, data_noise, data_occlusion, data_gaussblur, data_motionblur]
        # print(each_np_img.shape)
        for index, each_transform in enumerate(data_transforms):
            trans_data, _ = each_transform(each_np_img, each_np_gt)
            # show_img(trans_data)
            out_im[i*7+index, :, :, :] = trans_data
            out_gt[i*7+index, :, :] = each_np_gt
            out_pos[i*7+index, :, :] = each_np_pos
            out_char[i*7+index] = each_np_char

    print('im:{}, gt:{}, pos:{}, char:{}'.format(out_im.shape, out_gt.shape, out_pos.shape, out_char.shape))
    out_path = 'npy_data_with_aug'
    data_type = 'all_car_recorder_with_aug_val'
    np.save(os.path.join(out_path, data_type + '_im.npy'), out_im)
    np.save(os.path.join(out_path, data_type + '_gt.npy'), out_gt)
    np.save(os.path.join(out_path, data_type + '_pos.npy'), out_pos)
    np.save(os.path.join(out_path, data_type + '_char.npy'), out_char)


    # data_bright = random_bright(threhold=3*np.random.rand())
    # data_contrast = random_contrast(threhold=2*np.random.rand())
    # data_noise = random_noise()
    # data_occlusion = random_occlusion(threhold=random.randint(1,9))
    # random_ksize = (random.randint(2,18) // 4) * 2 + 1
    # data_gaussblur = gaussian_blurry(ksize = (random_ksize,random_ksize), sigemaX=0, sigemaY=0)
    # data_motionblur = motion_blurry(degree=random.randint(1,9), angle=random.randint(0,90))

    # bright_img, _ = data_bright(np_test_img, np_test_gt)
    # contrast_img, _ = data_contrast(np_test_img, np_test_gt)
    # noise_img, _ = data_noise(np_test_img, np_test_gt)
    # occl_img, _ = data_occlusion(np_test_img, np_test_gt)
    # Gblur_img, _ = data_gaussblur(np_test_img, np_test_gt)
    # Mblur_img, _ = data_motionblur(np_test_img, np_test_gt)

    # show_img(bright_img)
