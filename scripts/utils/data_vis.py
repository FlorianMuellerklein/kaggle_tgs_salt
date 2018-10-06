import math
import torch
import pickle
import numpy as np

from skimage.transform import resize

import matplotlib.pyplot as plt
import matplotlib.patches as patches

#from utils.helpers import convert_xyxy, decode_boxes

def plot_img_annots(img, coords=None, mask=None, scale=None, num_anchors=1):
    '''Takes np img, and list of bounding boxes'''
    #strides = [128, 64, 32, 16, 8]
    strides = [8, 16, 32, 64, 128]
    img_h, img_w, _ = img.shape

    fig, ax = plt.subplots(nrows=len(coords), ncols=2,figsize=(18,18))
        
    for i in range(len(coords)):
        # add grid lines to image
        #img[:,::strides[i],:] = [0,0,0]
        #img[::strides[i],:,:] = [0,0,0]

        boxes = coords[i]
        #boxes = decode_boxes(boxes, 600, num_anchors)

        #fig, ax = plt.subplots(len(mask)
        ax[i,0].imshow( (img - np.amin(img)) / np.amax(img) )
        
        if boxes is not None:
            for j in range(len(boxes)):
                box = boxes[j]
                if int(mask[i][j]) > 0:
                    edgecolor = 'r'
                    linewidth = 3
                    linestyle = '-'
                    ax[i,0].text(box[0]*img_w, box[1]*img_h, mask[i][j], fontsize=16, color='w', weight='bold',
                               bbox={'facecolor':'black', 'pad':2})
                #else:
                #    edgecolor = 'y'
                #    linewidth = 2
                #    linestyle = ':'
                    rect = patches.Rectangle((box[0] * img_w, box[1] * img_h),
                                         (box[2] - box[0]) * img_w,
                                         (box[3] - box[1]) * img_h,
                                         edgecolor=edgecolor, facecolor='none',
                                         linewidth=linewidth, linestyle=linestyle)
                    ax[i,0].add_patch(rect)
                
        if mask is not None:
            sz_h = math.ceil(img_h / strides[i])
            sz_w = math.ceil(img_w / strides[i])
            mask[i][mask[i] == -1] = 0
            mask[i] = np.sum(mask[i].reshape(sz_h, sz_w, num_anchors), axis=2)
            mask[i][mask[i] > 0] = 1
            ax[i,1].imshow(mask[i].astype(np.float32), cmap='Reds', interpolation='none')
        #ax[i].tight_layout()
    #plt.savefig('../../imgs/ground_truth_box.png')
    plt.tight_layout()
    plt.show()

def plot_from_torch(img_tch, targets, num_scales, imsize, num_anchors):
    #strides = [128, 64, 32, 16, 8]
    strides = [8, 16, 32, 64, 128]
    # get numpy img
    img_np = img_tch.numpy().transpose(1,2,0)

    targets[:,1:] = decode_boxes(targets[:,1:], 600, num_scales)
    
    # get numpy grid
    if targets is not None:
        grids = []
        coords = []
        starting_idx = 0
        for i in range(num_scales):
            sz = math.ceil(imsize / strides[i])
            mask = targets[starting_idx:(sz*sz*num_anchors)+starting_idx, 0].data.numpy()
            box = targets[starting_idx:(sz*sz*num_anchors)+starting_idx, 1:].data.numpy()
            starting_idx += sz*sz*num_anchors
            grids.append(mask)
            coords.append(box)
    else:
        grids = None
        coords = None
    
    print(len(grids), len(coords))
    # get numpy coords
    #if targets is not None:
    #    coords_np = targets[:,1:].cpu().numpy()
    #else:
    #    coords_np = None
     
    plot_img_annots(img_np, coords=coords, mask=grids, num_anchors=num_anchors)


def draw_boxes(img, boxes, threshold, filename):
    # get img dimensions
    img_h, img_w, _ = img.shape
    
    # display the image
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    # go through each prediction
    for i in range(boxes.shape[0]):
        # check if confidence is above threshold
        if boxes[i][0] >= threshold:
            
            box = boxes[i][2:]
            # draw box
            rect = patches.Rectangle((box[0] * img_w, box[1] * img_h),
                                     (box[2] - box[0]) * img_w,
                                     (box[3] - box[1]) * img_h,
                                     edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(box[0]*img_w, box[1]*img_h, class_dict[boxes[i][1]], fontsize=16, color='w', weight='bold',
                        bbox={'facecolor':'black', 'pad':2})
            
    #plt.savefig("../imgs/{}.png".format(filename), dpi=400)
    plt.show()
