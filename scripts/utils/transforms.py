import math
import random
import numpy as np

from scipy import ndimage
from skimage.io import imread
from skimage import img_as_float
from skimage.transform import resize
from skimage.exposure import adjust_gamma
import skimage.transform as tf

def reflect_pad(img, pad_amt):
    #print(pad_amt + pad_amt)
    if pad_amt == 13:
        #choice = np.random.randint(0,1)
        pad_x = pad_amt + 1
        pad_y = pad_amt
    else:
        pad_x = pad_amt
        pad_y = pad_amt

    img = np.pad(img, ((pad_y,pad_x),(pad_y,pad_x),(0,0)),
                 mode='reflect')
    return img

def random_crop(imgs):
    img_h, img_w, _ = imgs[0].shape

    crop_amt = np.random.uniform(0.8, 1.0)
    new_w = int(img_w * crop_amt)
    new_h = int(img_h * crop_amt)
    new_x = np.random.randint(0, int(img_w - new_w))
    new_y = np.random.randint(0, int(img_h - new_h))

    imgs = [x[new_y:new_y+new_h,new_x:new_x+new_w, :] for x in imgs]
    # scale the images back to their original size
    #imgs = [resize(x, (101,101), preserve_range=True, mode='reflect') for x in imgs]
    return imgs 

def shear(img, shear_amt):
    shear = tf.AffineTransform(shear=shear_amt)
    img = tf.warp(img, inverse_map=shear, mode='reflect')
    return img

def rotate(img, rot_amt):
    rot = tf.AffineTransform(rotation=rot_amt)
    img = tf.warp(img, inverse_map=rot, mode='reflect')
    return img

def mt_noise(img):
    # brightness
    bright_scale = np.random.uniform(0.95, 1.05)
    img = img * bright_scale

    # gamma adjustment
    gamma_ = 1. - np.random.uniform(-0.5, 0.5)
    img = adjust_gamma(img, gamma=gamma_, gain=1)

    img += np.random.normal(scale=0.005, size=img.shape)

    img = np.clip(img, 0., 1.)

    return img

def zoom(img, zoom_amt, imsize):
    h,w,_ = img.shape
        
    # Bounding box of the zoomed-in region within the input array
    zh = int(np.round(h / zoom_factor))
    zw = int(np.round(w / zoom_factor))
    top = (h - zh) // 2
    left = (w - zw) // 2

    out = img[top:top+zh, left:left+zw]

    return out 

def augment_img(imgs, imsize, mt=False):
    '''randomly horizontal flip'''
    #random.seed()
    img_h, img_w, _ = imgs[0].shape

    # remove small masks
    if imgs[1].sum() < 150:
        imgs[1] = np.zeros((imgs[1].shape))
    
    if random.random() > 0.5:
        # flip lr
        imgs = [np.fliplr(x) for x in imgs]    

    if random.random() > 0.5:
        # flip ud
        imgs = [np.flipud(x) for x in imgs]

    #if np.random.rand() > 0.5:
        # random 90 degree rotation
    #    num_rot = np.random.randint(0,3)
    #    imgs = [np.rot90(x, k=num_rot) for x in imgs]
    
    if random.random() > 0.5:
        # random shear
        shear_amt = random.uniform(-0.1, 0.1)
        imgs = [shear(x, shear_amt) for x in imgs]
   
    if random.random() > 0.5:
        # random rotation
        rot_amt = random.uniform(-0.17, 0.17)
        imgs = [rotate(x, rot_amt) for x in imgs]

    if random.random() > 0.5:
        # random crop
        imgs = random_crop(imgs)

    #if np.random.rand() > 0.5:
        # random zooms
    #    np.random.un(1., 1.20)

    if not mt:
        if random.random() > 0.5:
            # brightness
            bright_scale = random.uniform(0.92, 1.08)
            imgs[0] = imgs[0] * bright_scale

        if random.random() > 0.5:
            # gamma adjustment
            gamma_ = 1. - random.uniform(-0.5, 0.5)
            imgs[0] = adjust_gamma(imgs[0], gamma=gamma_, gain=1)

            #imgs[0] += np.random.normal(scale=0.005, size=imgs[0].shape)

    #print(imgs[0].shape)
    # reflect pad the images to 128x128
    if imgs[0].shape != (101,101,3):
        imgs = [resize(x, (101,101), preserve_range=True, mode='reflect') for x in imgs]
    imgs = [reflect_pad(x, int((imsize-x.shape[0]) / 2)) for x in imgs]
    #else:
    #    imgs = [resize(x, (imsize,imsize), preserve_range=True, mode='reflect') for x in imgs]

    imgs[0] = np.clip(imgs[0], 0., 1.)

    #print(imgs[0].shape)
    #print()
    #print()

    # transpose for pytorch 
    imgs = [x.transpose((2,0,1)) for x in imgs]
    return imgs
