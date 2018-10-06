import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, gamma= 0.3, smooth=0, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        return torch.log((2 * torch.sum(input * target) + self.smooth) / (
                torch.sum(input) + torch.sum(target) + self.smooth + self.eps))

class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, eps=1e-8):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        #print(input.size(), target.size())
        assert input.size() == target.size()

        #input = input.transpose(1,2).transpose(2,3).contiguous().view(-1,1)
        #target = target.transpose(1,2).transpose(2,3).contiguous().view(-1,1)
        
        p  = input.sigmoid()
        p = p.clamp(min=self.eps, max=1.-self.eps)

        pt = p * target + (1.-p) * (1-target)
        # from Heng, don't apply focal weight to predictions with prob < 0.1
        #pt[pt < 0.1] = 0.

        w = (1.-pt).pow(self.gamma)

        loss = F.binary_cross_entropy_with_logits(input, target, w)

        return loss

class ConsistencyLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(ConsistencyLoss, self).__init__()
        self.eps = eps
        self.focal_loss = FocalLoss2d()

    def forward(self, input, target):
        assert input.size() == target.size()
        input_sigmoid = input.sigmoid()
        target_sigmoid = target.sigmoid()
        return F.binary_cross_entropy(input_sigmoid, target_sigmoid)

def calc_iou(actual,pred):
  intersection = np.count_nonzero(actual*pred)
  union = np.count_nonzero(actual+pred)
  iou_result = intersection/union if union!=0 else 0.
  return iou_result

def calc_ious(actuals,preds):
  ious_ = np.array([calc_iou(a,p) for a,p in zip(actuals,preds)])
  return ious_

def calc_precisions(thresholds,ious):
  thresholds = np.reshape(thresholds,(1,-1))
  ious = np.reshape(ious,(-1,1))
  ps = ious>thresholds
  mps = ps.mean(axis=1)
  return mps
  
def indiv_scores(masks,preds):
  ious = calc_ious(masks,preds)
  thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  
  precisions = calc_precisions(thresholds,ious)
  
  ###### Adjust score for empty masks
  emptyMasks = np.count_nonzero(masks.reshape((len(masks),-1)),axis=1)==0
  emptyPreds = np.count_nonzero(preds.reshape((len(preds),-1)),axis=1)==0
  adjust = (emptyMasks==emptyPreds).astype(np.float)
  precisions[emptyMasks] = adjust[emptyMasks]
  ###################
  return precisions

def calc_metric(masks, preds):
    preds = np.where(preds > 0.5, 1, 0)
    return np.mean(indiv_scores(masks,preds))

def get_iou_vector(A, B):
    batch_size = A.shape[0]
    B = np.where(B > 0.5, 1, 0)
    metric = []
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
            metric.append(1)
            continue

        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = np.sum(intersection > 0) / np.sum(union > 0)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)
