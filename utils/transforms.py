# coding: utf8
import numpy as np
import random
import numbers
import cv2 as cv
import os
import cv2
import random

class Random_gamma(object):
    '''
    随机gama变换
    '''
    def __init__(self, range=[0.7, 1.3]):
        self.range = range

    def __call__(self, imgs):
        randon_gamma = random.randrange(self.range[0]*1000, self.range[1]*1000)/1000
        t, h, w, c = imgs.shape
        res_imgs = []
        for i in range(t):
            res_imgs.append(self.adjust_gamma(imgs[i], gamma=randon_gamma))
        return np.asarray(res_imgs)

    def adjust_gamma(self, image, gamma):
        image = np.power(image, gamma)
        return image

    def __repr__(self):
        return self.__class__.__name__ + '(range={0})'.format(self.range)




class RCropResize(object):
    """ 把 64帧200*200输入随机crop和resize成224*224，Crops the given seq Images
    input images: (T x H x W x C)
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        margin: 先随机crop x*x大小的图片，x是180-200的随机数.
    """

    def __init__(self, size, margin):
        self.margin = margin
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped and resized image.
        """
        t, h, w, c = imgs.shape  # t=64, h=w=200, c=3
        th, tw = self.size  # th=tw
        # RandomCrop:
        m = random.randint(0, 20)
        l = h - m
        i = random.randint(0, m)
        j = random.randint(0, m)
        crop_imgs = imgs[:, i:i + l, j:j + l, :]
        # Resize:
        res_imgs = np.zeros((t, th, tw, c))
        for n in range(t):
            img = []
            img = cv.resize(crop_imgs[n, :, :, :], (th, tw))
            res_imgs[n, :, :, :] = img

        # Color Jitter
        randomintr = random.randint(-15, 15) / 255
        randomintg = random.randint(-15, 15) / 255
        randomintb = random.randint(-15, 15) / 255
        res_imgs[:, :, :, 0] += randomintr
        res_imgs[:, :, :, 1] += randomintg
        res_imgs[:, :, :, 2] += randomintb
        res_imgs[res_imgs < -1] = -1
        res_imgs[res_imgs > 1] = 1

        return res_imgs

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class CenterCrop(object):
    """Crops the given seq Images at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        randomintr = random.randint(-15, 15) / 255
        randomintg = random.randint(-15, 15) / 255
        randomintb = random.randint(-15, 15) / 255
        imgs[:, :, :, 0] += randomintr
        imgs[:, :, :, 1] += randomintg
        imgs[:, :, :, 2] += randomintb
        imgs[imgs < -1] = -1
        imgs[imgs > 1] = 1

        t, h, w, c = imgs.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))
        # print('o')

        return imgs[:, i:i + th, j:j + tw, :]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
