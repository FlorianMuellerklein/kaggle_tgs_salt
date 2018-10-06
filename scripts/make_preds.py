import os
import sys
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
parser.add_argument('--fold_num', default=0, type=int, 
                    help='which fold to make predictions for')
parser.add_argument('--weight_file', default='resunet.pth', type=str,
                    help='which weight file to make predictions for')
#parser.add_argument('--model_name', default='resunet', type=str,
 #                   help='name of model for saving/loading weights')
#parser.add_argument('--exp_name', default='tgs_slt', type=str,
#                    help='name of experiment for saving files')
parser.add_argument('--debug', action='store_true', 
                    help='whether to display debug info')
parser.add_argument('--flip_tta', action='store_true', 
                    help='whether to horizontal flip TTA')
#parser.add_argument('--use_mt', action='store_true',
#                    help='whether to use mean teacher model')
#parser.add_argument('--use_swa', action='store_true',
#                    help='whether to use mean teacher model')
parser.add_argument('--use_bool', action='store_true',
                    help='whether to use empty predictions')
args = parser.parse_args()

# set model filenames
#model_params = [args.model_name, args.exp_name, args.fold_num]
#if args.use_mt:
#    MODEL_CKPT = '../model_weights/best_meanteacher_{}_{}_fold-{}.pth'.format(*model_params)
#elif args.use_swa:
#    MODEL_CKPT = '../model_weights/swa_{}_{}_fold-{}.pth'.format(*model_params)
#else:
#    MODEL_CKPT = '../model_weights/best_{}_{}_fold-{}.pth'.format(*model_params)

MODEL_CKPT = args.weight_file
OUT_FILE = '../subm/' + os.path.basename(MODEL_CKPT.replace('pth', 'csv'))
print('Saving to {}'.format(OUT_FILE))

# get the loaders
test_loader = get_test_loader(imsize=args.imsize, batch_size=args.batch_size)

net = ResUNet(use_bool=True)
if args.gpu == 99:
     net = nn.DataParallel(net, device_ids=[0,1]).cuda()
else:
     torch.cuda.set_device(args.gpu)
     cudnn.benchmark = True
     net.cuda()

net.load_state_dict(torch.load(MODEL_CKPT, 
                                   map_location=lambda storage, 
                                   loc: storage))

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
    net.eval() 
    # keep track of losses
    rles = []
    ids = []
    # no gradients during validation
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i % 20 == 0:
                print('Done {} batches out of {}'.format(i, len(test_loader.dataset) // args.batch_size))
            
            test_imgs = data['img'].cuda(async=True)
            test_ids = data['id']
            blanks = data['blank']
            # get predictions
            preds, chck_preds = net(test_imgs)
            preds = preds.sigmoid()
            chck_preds = chck_preds.sigmoid() > 0.5

            if args.flip_tta:
                test_imgs_lr = data['img_lr'].cuda(async=True)
                preds_lr, check_lr = net(test_imgs_lr)
                preds_lr_ = preds_lr.sigmoid()
                check_lr = check_lr.sigmoid() > 0.5

                chck_preds = (check_lr + chck_preds) / 2.
                preds_lr = np.zeros((preds_lr_.size())).astype(np.float32)
                preds_lr = np.copy(preds_lr_.data.cpu().numpy()[:,:,:,::-1])
                
                preds = (preds + torch.from_numpy(preds_lr).cuda()) / 2. 

            # set masks to 0 with low probability of having mask 
            if args.use_bool:
                chck_preds = chck_preds > 0.5
                preds *= chck_preds.view(chck_preds.size(0),1,1,1).expand_as(preds).float()
            preds *= blanks.view(blanks.size(0),1,1,1).expand_as(preds).float().cuda()

            if args.debug and i == 0:
                img_grid = vsn.utils.make_grid(test_imgs, normalize=True)
                msk_grid = vsn.utils.make_grid(preds)

                if args.flip_tta:
                    img_lr_grid = vsn.utils.make_grid(test_imgs_lr, normalize=True)
                    vsn.utils.save_image(img_lr_grid, '../imgs/test_imgs_lr.png')

                vsn.utils.save_image(img_grid, '../imgs/test_imgs.png')
                vsn.utils.save_image(msk_grid, '../imgs/test_pred.png') 

            pred_np = preds.squeeze().data.cpu().numpy()

            #print(pred_np.shape)

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
    #if args.use_mt:
    #    subm.to_csv('../subm/{}_{}_mt_fold-{}.csv'.format(args.model_name, args.exp_name, args.fold_num), index=False)
    #elif args.use_swa:
    #    subm.to_csv('../subm/{}_{}_swa_fold-{}.csv'.format(args.model_name, args.exp_name, args.fold_num), index=False)
    #else:  
    #    subm.to_csv('../subm/{}_{}_best_fold-{}.csv'.format(args.model_name, args.exp_name, args.fold_num), index=False)
    subm.to_csv(OUT_FILE, index=False)

    subm.index.names = ['id']
    subm.columns = ['id', 'rle_mask']
    print(subm.head())

if __name__ == '__main__':
    make_preds()

