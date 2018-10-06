import os
import sys
import glob
import time
import math
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision as vsn

from skimage.transform import resize

from models.nets import ResUNet
from utils.data_loaders import get_test_loader
from utils.data_vis import plot_from_torch

from utils.evaluations import DiceLoss, calc_metric

parser = argparse.ArgumentParser(description='Make Preds')
parser.add_argument('--imsize', default=128, type=int,
                    help='imsize to use for training')
parser.add_argument('--batch_size', default=128, type=int, 
                    help='size of batches')
parser.add_argument('--gpu', default=1, type=int, 
                    help='which gpu to run')
parser.add_argument('--weight_folder', default='resunet.pth', type=str,
                    help='which folder the weight files are in')
parser.add_argument('--debug', action='store_true', 
                    help='whether to display debug info')
parser.add_argument('--flip_tta', action='store_true', 
                    help='whether to horizontal flip TTA')
parser.add_argument('--use_bool', action='store_true',
                    help='whether to use empty predictions')
args = parser.parse_args()


weights = glob.glob(args.weight_folder + '*.pth')
print('Found {} models'.format(len(weights)))

OUT_FILE = '../subm/averaged_{}_resunet_models.csv'.format(len(weights))

# get the loaders
test_loader = get_test_loader(imsize=args.imsize, batch_size=args.batch_size)

net = ResUNet(use_bool=args.use_bool)
if args.gpu == 99:
     net = nn.DataParallel(net, device_ids=[0,1]).cuda()
else:
     torch.cuda.set_device(args.gpu)
     cudnn.benchmark = True
     net.cuda()

def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted 
    (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b

    print(run_lengths)
    return run_lengths

def make_preds():
    # keep list of prediction per batch
    test_ids = []
    averaged_mask_preds = []
    averaged_bool_preds = []
    all_blanks = []
    for w_idx, wf in enumerate(weights):
        print('Doing model {} of {}'.format(w_idx+1, len(weights)))
        net.load_state_dict(torch.load(wf, 
                                       map_location=lambda storage, 
                                       loc: storage))
        net.eval() 
        for batch_idx, data in enumerate(test_loader): 
            test_imgs = data['img'].cuda(async=True)
            batch_ids = data['id']
            blanks = data['blank']
            # get predictions
            preds = net(test_imgs)
            if args.use_bool:
                bool_preds = preds[1].sigmoid()
                preds = preds[0].sigmoid()
            else:
                preds = preds.sigmoid()
               
            if args.flip_tta:
                test_imgs_lr = data['img_lr'].cuda(async=True)
                preds_lr_ = net(test_imgs_lr)
                if args.use_bool:
                    bool_lr = preds_lr_[1].sigmoid()
                    preds_lr_ = preds_lr_[0].sigmoid()
                    bool_preds = (bool_lr + bool_preds) / 2.
                else:
                    preds_lr_ = preds_lr_.sigmoid()
                
                preds_lr = np.zeros((preds_lr_.size())).astype(np.float32)
                preds_lr = np.copy(preds_lr_.data.cpu().numpy()[:,:,:,::-1])
                
                preds = (preds + torch.from_numpy(preds_lr).cuda()) / 2.

            if w_idx == 0:
                #print(preds.size())
                test_ids.extend(batch_ids)
                all_blanks.extend(blanks.data.cpu())
                averaged_mask_preds.append(preds.data.view(-1, args.imsize, args.imsize).cpu())
                if args.use_bool:
                    #print(bool_preds)
                    averaged_bool_preds.append(bool_preds.data.cpu())
            else:
                averaged_mask_preds[batch_idx] += preds.data.view(-1, args.imsize, args.imsize).cpu()
                averaged_bool_preds[batch_idx] += bool_preds.data.cpu()
    
    print('Predicted mask sizes')
    for i in range(len(averaged_mask_preds)):
        print(averaged_mask_preds[i].size())

    avg_preds = torch.cat(averaged_mask_preds, dim=0)
    #print(avg_preds)
    avg_preds /= float(len(weights))
    #print(avg_preds)
    print(avg_preds.size())
    
    # set masks to 0 with low probability of having mask 
    if args.use_bool:
        bool_preds = torch.cat(averaged_bool_preds, dim=0)
        #print(bool_preds)
        bool_preds /= float(len(weights))
        bool_preds = bool_preds > 0.5
        print('number of non-empty predicted masks', bool_preds.sum())
        #print(bool_preds.size())
        avg_preds *= bool_preds.view(bool_preds.size(0),1,1).expand_as(avg_preds).float()
    
    blanks = torch.Tensor(all_blanks).view(-1,1,1)
    print('blanks', blanks.size())
    avg_preds *= blanks.expand_as(avg_preds).float()

    #if args.debug:
        #img_grid = vsn.utils.make_grid(test_imgs, normalize=True)
    #    msk_grid = vsn.utils.make_grid(avg_preds)
    #    vsn.utils.save_image(msk_grid, '../imgs/test_pred.png')
        #vsn.utils.save_image(img_grid, '../imgs/test_imgs.png')

    pred_np = avg_preds.cpu().numpy()

    print(pred_np.shape)
    # keep track of losses
    rles = []
    ids = []
    for j in range(pred_np.shape[0]):
        if args.imsize == 256:
            predicted_mask = resize(pred_np[j][27:229, 27:229], (101,101),
                                    preserve_range=True)
        else:
            predicted_mask = pred_np[j][13:114, 13:114]
            predicted_mask = np.where(predicted_mask > 0.4, 1, 0)
            rles.append(RLenc(predicted_mask.astype(np.int32)))
            ids.append(test_ids[j])
            
    print(len(ids), len(rles))
    #print({'id':ids[:4], 'rle_mask':rles[:4]})

    subm = pd.DataFrame.from_dict({'id':ids, 'rle_mask':rles}, orient='index').T
    subm.to_csv(OUT_FILE, index=False)

    subm.index.names = ['id']
    subm.columns = ['id', 'rle_mask']
    print(subm.head())
    print(subm.tail())

if __name__ == '__main__':
    with torch.no_grad():
        make_preds()

