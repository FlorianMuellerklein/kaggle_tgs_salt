import math
import glob

import numpy as np
import pandas as pd

from skimage.io import imread

#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(101, 101)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    if str(mask_rle) != str(np.nan):
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths

        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

submissions = glob.glob('../subm/best_resunet_tgs_slt_cycle*.csv')
print('Found {} submissions to merge'.format(len(submissions)))

folds = []
for subm in submissions:
    #print('reading {}'.format(subm))
    folds.append(pd.read_csv(subm))
#for i in range(5):
#    folds.append(pd.read_csv('../subm/resunet_tgs_slt_fold-{}.csv'.format(i)))
print(folds[0].head())


def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

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


ids = []
rles = []

for i in range(len(folds[0])):
    if i % 100 == 0:
        print('Done {} of {}'.format(i, len(folds[0])))
    mask = np.zeros((101, 101))
    for j in range(len(folds)):
        mask += rle_decode(folds[j].iloc[i]['rle_mask'])
        # zero out small masks
        #if np.count_nonzero(mask) < 15:
        #    mask = np.zeros((mask.shape))
    # majority vote for ensembled mask
    mask = np.where(mask >= np.round(len(folds) * 0.5), 1, 0)
    ids.append(folds[0].iloc[i]['id'])
    rles.append(RLenc(mask.astype(np.int32)))
    
ensmb_preds = pd.DataFrame({'id': ids, 'rle_mask': rles})
ensmb_preds.head()

ensmb_preds.to_csv('../subm/ensemble_swa_best_blended-{}.csv'.format(len(folds)), index=False)

