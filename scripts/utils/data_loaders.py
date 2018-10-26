import os
import cv2
import sys
import glob
import math
import time
import random
import pickle

import numpy as np

from scipy import ndimage
from skimage.io import imread
from skimage import img_as_float
from skimage.transform import resize
import skimage.filters as fltrs

from sklearn.model_selection import KFold

import torch
import torchvision as vsn
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from utils.transforms import augment_img, reflect_pad, mt_noise

#cv2.setNumThreads(0)

def add_depth_channels(image_tensor):
    _, h, w = image_tensor.size()
    for row, const in enumerate(np.linspace(0, 1, h)):
        image_tensor[1, row, :] = const
    image_tensor[2] = image_tensor[0] * image_tensor[1]
    return image_tensor

class MaskDataset(data.Dataset):
    '''Generic dataloader for a pascal VOC format folder'''
    def __init__(self, imsize=128, img_ids=None, img_paths=None, num_folds=None,
                 mask_paths=None, valid=False, small_msk_ids=None):
        self.valid = valid
        self.imsize = imsize
        self.img_ids = img_ids
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.num_folds = num_folds
        self.small_msk_ids = small_msk_ids
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):
        random.seed()

        # super sample small masks
        #if random.random() > 0.25 or self.valid:
        img = img_as_float(imread(self.img_paths + self.img_ids[index]))[:,:,:3]
        #img[:,:,2] = 1. - fltrs.laplace(img[:,:,0])
        msk = imread(self.mask_paths + self.img_ids[index]).astype(np.bool)
        msk = np.expand_dims(msk, axis=-1)
        #else:
        #    small_idx = random.randint(0, len(self.small_msk_ids))
        #    img = img_as_float(imread('../data/train/small_masks/images/' + self.small_msk_ids[small_idx]))[:,:,:3]
        #    msk = imread('../data/train/small_masks/masks/' + self.small_msk_ids[small_idx]).astype(np.bool)
        #    msk = np.expand_dims(msk, axis=-1) 
        
        if not self.valid:
            img_np, msk_np  = augment_img([img, msk], imsize=self.imsize)
        else:
            #img_np = resize(np.asarray(img), (self.imsize, self.imsize), 
            #            preserve_range=True, mode='reflect')
            #msk_np = resize(msk, (self.imsize, self.imsize),
            #                preserve_range=True, mode='reflect')
            img_np = reflect_pad(img, int((self.imsize - img.shape[0]) / 2))
            msk_np = reflect_pad(msk, int((self.imsize - msk.shape[0]) / 2)) 
            img_np = img_np.transpose((2,0,1)).astype(np.float32)
            msk_np = msk_np.transpose((2,0,1)).astype(np.float32)
    

        #print(img_np.shape, msk_np.shape)

        # get image ready for torch
        img_tch = self.normalize(torch.from_numpy(img_np.astype(np.float32)))

        msk_tch = torch.from_numpy(msk_np.astype(np.float32))

        msk_half_np = np.expand_dims(resize(msk_np[0].astype(np.float32), (64,64), preserve_range=True), axis=-1).transpose(2,0,1)
        msk_half_tch = torch.from_numpy(msk_half_np.astype(np.float32))
        msk_qrtr_np = np.expand_dims(resize(msk_np[0].astype(np.float32), (32,32), preserve_range=True), axis=-1).transpose(2,0,1)
        msk_qrtr_tch = torch.from_numpy(msk_qrtr_np.astype(np.float32))
        msk_eigt_np = np.expand_dims(resize(msk_np[0].astype(np.float32), (16,16), preserve_range=True), axis=-1).transpose(2,0,1)
        msk_eigt_tch = torch.from_numpy(msk_eigt_np.astype(np.float32))
        msk_sixteen_np = np.expand_dims(resize(msk_np[0].astype(np.float32), (8,8), preserve_range=True), axis=-1).transpose(2,0,1)
        msk_sixteen_tch = torch.from_numpy(msk_sixteen_np.astype(np.float32))

        #img_tch = add_depth_channels(img_tch)

        out_dict = {'img': img_tch,
                    'msk': [msk_tch, msk_half_tch, msk_qrtr_tch, msk_eigt_tch, msk_sixteen_tch],
                    #'msk_dwnsample': [msk_half_tch, msk_qrtr_tch, msk_eigt_tch, msk_sixteen_tch],
                    #'msk_qrtr': msk_qrtr_tch, 
                    #'msk_eigt': msk_eigt_tch,
                    'has_msk': msk_tch.sum() > 0,
                    'id': self.img_ids[index].replace('.png', '')}

        return out_dict

    def __len__(self):
        if self.valid:
            return int(len(self.img_ids) * (1. / self.num_folds))
        else:
            return int(len(self.img_ids) * ((self.num_folds - 1.) / self.num_folds))


class MaskDataset_MT(data.Dataset):
    '''Generic dataloader for a pascal VOC format folder'''
    def __init__(self, imsize=128, labeled_ids=None, labeled_img_paths=None, 
                 unlabeled_index=0, unlabeled_ids=None, unlabeled_img_paths=None, 
                 unlabeled_ratio=0.5, mask_paths=None):
        self.imsize = imsize
        self.labeled_ids = labeled_ids
        self.labeled_img_paths = labeled_img_paths
        self.unlabeled_idx = unlabeled_index
        self.unlabeled_ids= unlabeled_ids
        self.unlabeled_ratio = unlabeled_ratio
        self.unlabeled_img_paths = unlabeled_img_paths
        self.mask_paths = mask_paths
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):
        random.seed()

        # choose whether to pull labeled or unlabaled
        labeled = 1 if random.random() > self.unlabeled_ratio else 0 # print(imgs.size())
        if labeled == 1:
            img = img_as_float(imread(self.labeled_img_paths + 
                                      self.labeled_ids[index]))[:,:,:3]
            msk = imread(self.mask_paths + self.labeled_ids[index]).astype(np.bool)
            msk = np.expand_dims(msk, axis=-1)
        else:
            img = img_as_float(imread(self.unlabeled_img_paths + 
                                      self.unlabeled_ids[self.unlabeled_idx]))[:,:,:3]
            msk = np.ones((101, 101, 1)) * -1.
            self.unlabeled_idx += 1
            # if we go through all the labeled images, shuffle them and start counter over
            if self.unlabeled_idx >= len(self.unlabeled_ids):
                self.unlabeled_ids = shuffle(self.unlabeled_ids)
                self.unlabeled_idx = 0

        # the geometric augmentions have to be the same
        img, msk = augment_img([img, msk], imsize=self.imsize, mt=True)
        # brightness, gamma, and gaussian noise can be different
        img_a = mt_noise(img)
        img_b = mt_noise(img)

        msk_tch = torch.from_numpy(msk.astype(np.float32))
        msk_half_tch = torch.from_numpy(resize(msk.astype(np.float32), (64,64), preserve_range=True))
        msk_qrtr_tch = torch.from_numpy(resize(msk.astype(np.float32), (32,32), preserve_range=True))

        out_dict = {'img_a': self.normalize(torch.from_numpy(img_a.astype(np.float32))),
                    'img_b': self.normalize(torch.from_numpy(img_b.astype(np.float32))),
                    'msk': msk_tch,
                    'msk_half': msk_half_tch,
                    'msk_qrtr': msk_qrtr_tch,
                    'has_msk': msk_tch.sum() > 0,
                    'is_labeled': torch.tensor(labeled).long()}

        return out_dict

    def __len__(self):
        return int(len(self.labeled_ids) * 0.8)

class MaskTestDataset(data.Dataset):
    '''Dataset for loading the test Images'''
    def __init__(self, imsize=128, img_ids=None, img_paths=None):
        self.imsize = imsize
        self.img_ids = img_ids
        self.img_paths = img_paths
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):

        img = img_as_float(imread('../data/test/images/' + self.img_ids[index]))[:,:,:3]
        # scale up image to 202 or keep at 101, reflect pad to get network sizes
        if self.imsize == 256:
            img = resize(img, (202, 202), preserve_range=True, mode='reflect')
            img = reflect_pad(img, 27)
        else:
            img = reflect_pad(img, 13)
        img_lr = np.fliplr(img)
        
        #print(img.shape, img_lr.shape)

        img = img.transpose((2,0,1)).astype(np.float32)
        img_lr = img_lr.transpose((2,0,1)).astype(np.float32)

        img_tch = self.normalize(torch.from_numpy(img))
        img_lr_tch = self.normalize(torch.from_numpy(img_lr))
       
        #img_tch = add_depth_channels(img_tch)
        #img_lr_tch = add_depth_channels(img_lr_tch)

        out_dict = {'img': img_tch,
                    'img_lr': img_lr_tch,
                    'id': self.img_ids[index].replace('.png', ''),
                    'blank': torch.tensor(os.stat('../data/test/images/' + self.img_ids[index]).st_size != 107)}

        return out_dict

    def __len__(self):
        return len(self.img_ids)


def get_data_loaders(imsize=128, batch_size=16, num_folds=5, fold=0):
    '''sets up the torch data loaders for training'''
    img_ids = [os.path.basename(x) for x in glob.glob('../data/train/images/*.png')]
    kf = KFold(n_splits=num_folds, shuffle=True)

    img_idx = list(range(len(img_ids)))
    splits = list(kf.split(img_idx))
    train_idx, valid_idx = splits[fold]

    small_msk_ids = list(set([os.path.basename(x) for x in glob.glob('../data/train/small_masks/images/*.png')]) & 
                         set([img_ids[idx] for idx in train_idx]))

    print('Supersampling {} small masks'.format(len(small_msk_ids)))

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)  

    # set up the datasets
    train_dataset = MaskDataset(imsize=imsize, img_ids=img_ids, num_folds=num_folds,
                                  img_paths='../data/train/images/',
                                  mask_paths='../data/train/masks/',
                                  small_msk_ids=small_msk_ids)
    valid_dataset = MaskDataset(imsize=imsize, img_ids=img_ids, num_folds=num_folds,
                                  img_paths='../data/train/images/',
                                  mask_paths='../data/train/masks/',
                                  valid=True)

    # set up the data loaders
    train_loader = data.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       sampler=train_sampler,
                                       num_workers=4,
                                       pin_memory=True,
                                       drop_last=True)

    valid_loader = data.DataLoader(valid_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       sampler=valid_sampler,
                                       num_workers=4,
                                       pin_memory=True)

    return train_loader, valid_loader

def get_data_mt_loaders(imsize=128, batch_size=16, num_folds=5, fold=0, unlabeled_ratio=0.5):
    '''sets up the torch data loaders for training'''
    img_ids = [os.path.basename(x) for x in glob.glob('../data/train/images/*.png')]
    kf = KFold(n_splits=num_folds, shuffle=True)

    img_idx = list(range(len(img_ids)))
    splits = list(kf.split(img_idx))
    train_idx, valid_idx = splits[fold]
    
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)  

    unlabeled_ids = [os.path.basename(x) for x in glob.glob('../data/test/images/*.png')]
    
    # set up the datasets
    train_dataset = MaskDataset_MT(imsize=imsize, labeled_ids=img_ids,
                                   unlabeled_ids=unlabeled_ids,
                                   unlabeled_ratio=unlabeled_ratio,
                                   labeled_img_paths='../data/train/images/',
                                   unlabeled_img_paths='../data/test/images/',
                                   mask_paths='../data/train/masks/')

    valid_dataset = MaskDataset(imsize=imsize, img_ids=img_ids,
                                  img_paths='../data/train/images/',
                                  mask_paths='../data/train/masks/',
                                  valid=True)

    # set up the data loaders
    train_loader = data.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       sampler=train_sampler,
                                       num_workers=4,
                                       pin_memory=True,
                                       drop_last=True)

    valid_loader = data.DataLoader(valid_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       sampler=valid_sampler,
                                       num_workers=4,
                                       pin_memory=True)

    return train_loader, valid_loader


def get_test_loader(imsize=128, batch_size=16):
    '''sets up the torch data loaders for training'''
    img_ids = [os.path.basename(x) for x in glob.glob('../data/test/images/*.png')]
    print('Found {} test images'.format(len(img_ids)))

    # set up the datasets
    test_dataset = MaskTestDataset(imsize=imsize, img_ids=img_ids,
                                   img_paths='../data/test/images/')

    # set up the data loaders
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  pin_memory=True,
                                  drop_last=False)

    return test_loader

if __name__ == '__main__':
    train_loader, valid_loader = get_data_loaders(imsize=128, batch_size=32)

    for i, data in enumerate(train_loader):
        if i == 1:
            break
        img = data['img']
        msk = data['msk']

        img_grid = vsn.utils.make_grid(img, normalize=True)
        msk_grid = vsn.utils.make_grid(msk)

        vsn.utils.save_image(img_grid, '../imgs/train_imgs.png')
        vsn.utils.save_image(msk_grid, '../imgs/train_msks.png')
