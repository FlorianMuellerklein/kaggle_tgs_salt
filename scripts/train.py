import sys
import time
import math
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision as vsn

from models.nets import ResUNet
from utils.data_loaders import get_data_loaders
from utils.data_vis import plot_from_torch

from utils.evaluations import FocalLoss2d, DiceLoss, get_iou_vector
import utils.lovasz_losses as L
from utils.helpers import bn_update 

parser = argparse.ArgumentParser(description='TGS Salt')
parser.add_argument('--imsize', default=128, type=int,
                    help='imsize to use for training')
parser.add_argument('--batch_size', default=32, type=int, 
                    help='size of batches')
parser.add_argument('--num_folds', default=5, type=int, 
                    help='number of cross val folds')
parser.add_argument('--epochs', default=1000, type=int, 
                    help='number of epochs')
parser.add_argument('--lr_max', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_min', default=0.0003, type=float,
                    help='min lr for cosine annealing')
parser.add_argument('--lr_rampup', default=1, type=int,
                    help='how long to ramp up the unsupervised weight')
parser.add_argument('--lr_rampdown', default=65, type=int,
                    help='how long to ramp up the unsupervised weight')
parser.add_argument('--num_cycles', default=5, type=int, 
                    help='number of learning rate cycles')
parser.add_argument('--l2', default=1e-4, type=float,
                    help='l2 regularization for model')
parser.add_argument('--lambda_bool', default=0.5, type=float,
                    help='lambda value for coordinate loss')
parser.add_argument('--es_patience', default=50, type=int, 
                    help='early stopping patience')
parser.add_argument('--lr_patience', default=20, type=int, 
                    help='early stopping patience')
parser.add_argument('--gpu', default=1, type=int, 
                    help='which gpu to run')
parser.add_argument('--model_name', default='resunet', type=str,
                    help='name of model for saving/loading weights')
parser.add_argument('--exp_name', default='tgs_slt', type=str,
                    help='name of experiment for saving files')
parser.add_argument('--debug', action='store_true', 
                    help='whether to display debug info')
parser.add_argument('--cos_anneal', action='store_true', 
                    help='whether to use cosine annealing for learning rate')
parser.add_argument('--freeze_bn', action='store_true', 
                    help='freeze batch norm during finetuning')
parser.add_argument('--use_lovasz', action='store_true', 
                    help='whether to use focal loss during finetuning')
parser.add_argument('--swa', action='store_true', 
                    help='whether to use stochastic weight averaging')
args = parser.parse_args()

if args.gpu == 99:
    device = torch.device("cuda:0")
    #torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

focal_loss = FocalLoss2d()
bce = nn.BCEWithLogitsLoss()

# training function
def train(net, optimizer, train_loader, freeze_bn=False, use_lovasz=False, swa=False):
    '''
    uses the data loader to grab a batch of images
    pushes images through network and gathers predictions
    updates network weights by evaluating the loss functions
    '''
    # set network to train mode
    #if args.gpu == 99:
    net.train()
    #else:
    #    net.train(mode=True, freeze_bn=args.freeze_bn)
    # keep track of our loss
    iter_loss = 0.

    # loop over the images for the desired amount
    for i, data in enumerate(train_loader):
        imgs = data['img'].to(device)
        msks = data['msk'].to(device)
        msk_bool = data['has_msk'].float().to(device)

        if args.debug and i == 0:
            img_grid = vsn.utils.make_grid(imgs, normalize=True)
            msk_grid = vsn.utils.make_grid(msks)

            vsn.utils.save_image(img_grid, '../imgs/train_imgs.png')
            vsn.utils.save_image(msk_grid, '../imgs/train_msks.png')
             
        # get predictions
        msk_preds, bool_preds = net(imgs)
        msk_blend = msk_preds * bool_preds.view(imgs.size(0), 1, 1, 1)
    
        # calculate loss
        if use_lovasz:
            loss = L.lovasz_hinge(msk_preds, msks)
        else:
            loss = focal_loss(msk_preds, msks)
            loss += L.lovasz_hinge(msk_preds, msks)
        loss += args.lambda_bool * bce(bool_preds, msk_bool.view(-1,1))
        loss += args.lambda_bool * focal_loss(msk_blend, msks)
        
        # zero gradients from previous run
        optimizer.zero_grad()
        #calculate gradients
        loss.backward()
        # update weights
        optimizer.step()

        # get training stats
        iter_loss += loss.item()
        # make a cool terminal output
        sys.stdout.write('\r')
        sys.stdout.write('B: {:>3}/{:<3} | {:.4}'.format(i+1, 
                                            len(train_loader),
                                            loss.item()))

    epoch_loss = iter_loss / (len(train_loader.dataset) / args.batch_size)
    print('\n' + 'Avg Train Loss: {:.4}'.format(epoch_loss))

    return epoch_loss

# validation function
def valid(net, optimizer, valid_loader, use_lovasz=False, save_imgs=False, fold_num=0):
    net.eval() 
    # keep track of losses
    val_ious = []
    val_iter_loss = 0.
    # no gradients during validation
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            valid_imgs = data['img'].to(device)
            valid_msks = data['msk'].to(device)
            valid_msk_bool = data['has_msk'].float().to(device)
            # get predictions
            msk_vpreds, bool_vpreds = net(valid_imgs)
            msk_blend_vpreds = msk_vpreds * bool_vpreds.view(valid_imgs.size(0), 1, 1, 1)

            if save_imgs:
                img_grid = vsn.utils.make_grid(valid_imgs, normalize=True)
                msk_grid = vsn.utils.make_grid(msk_vpreds)

                vsn.utils.save_image(img_grid, '../imgs/valid_imgs_fold-{}.png'.format(fold_num))
                vsn.utils.save_image(msk_grid, '../imgs/valid_msks_fold-{}.png'.format(fold_num))
        
            # calculate loss
            if use_lovasz:
                vloss = L.lovasz_hinge(msk_vpreds, valid_msks)
                #vloss += focal_loss(msk_vpreds, valid_msks)
                #vloss -= dice(msk_vpreds.sigmoid(), valid_msks)
            else:
                vloss = focal_loss(msk_vpreds, valid_msks)
                vloss += L.lovasz_hinge(msk_vpreds, valid_msks)
                #vloss -= dice(msk_vpreds.sigmoid(), valid_msks)
            
            vloss += args.lambda_bool * bce(bool_vpreds, valid_msk_bool.view(-1,1))
            vloss += args.lambda_bool * focal_loss(msk_blend_vpreds, valid_msks)

            #vloss += args.lambda_dice * dice(msk_vpreds.sigmoid(), valid_msks)
            # get validation stats
            val_iter_loss += vloss.item()
            
            val_ious.append(get_iou_vector(valid_msks.cpu().numpy()[:,:,13:114, 13:114], 
                                           msk_vpreds.sigmoid().cpu().numpy()[:,:,13:114, 13:114]))
            
    epoch_vloss = val_iter_loss / (len(valid_loader.dataset) / args.batch_size)
    print('Avg Eval Loss: {:.4}, Avg IOU: {:.4}'.format(epoch_vloss, np.mean(val_ious)))
    return epoch_vloss, np.mean(val_ious)

def train_network(net, fold=0, model_ckpt=None):
    # train the network, allow for keyboard interrupt
    try:
        # define optimizer
        optimizer = optim.SGD(net.parameters(), lr=args.lr_max, momentum=0.9, weight_decay=args.l2)
        # get the loaders
        train_loader, valid_loader = get_data_loaders(imsize=args.imsize,
                                                      batch_size=args.batch_size,
                                                      num_folds=args.num_folds,
                                                      fold=fold)

        # training flags
        swa = False
        use_lovasz = False
        freeze_bn = False
        save_imgs = False
        train_losses = []
        valid_losses = []
        valid_ious = []

        valid_patience = 0
        best_val_metric = 1000.0
        best_val_iou = 0.0
        cycle = 0
        swa_n = 0
        t_ = 0

        print('Training ...')
        for e in range(args.epochs):
            print('\n' + 'Epoch {}/{}'.format(e, args.epochs))

            start = time.time()
            
            # LR warm-up            
            #if e < args.lr_rampup:
            #    lr = args.lr_max * (min(e, args.lr_rampup) / args.lr_rampup)
            
            # if we get to the end of lr period, save swa weights
            if t_ >= args.lr_rampdown:
                # if we are using swa save off the current weights before updating
                if args.swa:
                    torch.save(net.state_dict(), '../swa/cycle_{}.pth'.format(cycle))
                    #swa_n += 1
                # reset the counter
                t_ = 0
                cycle += 1
                torch.save(net.state_dict(), 
                           '../model_weights/{}_{}_cycle-{}_fold-{}.pth'.format(args.model_name,
                                                                                     args.exp_name,
                                                                                     cycle,
                                                                                     fold))
                save_imgs = True
            else:
                save_imgs = False
           
            for params in optimizer.param_groups:
                #print('t_', t_)
                if args.cos_anneal and e > args.lr_rampup:
                    params['lr'] = (args.lr_min + 0.5 * (args.lr_max - args.lr_min) *
                                   (1 + np.cos(np.pi * t_ / args.lr_rampdown)))
                elif e < args.lr_rampup:
                    params['lr'] = args.lr_max * (min(t_+1, args.lr_rampup) / args.lr_rampup)

                print('Learning rate set to {:.4}'.format(optimizer.param_groups[0]['lr']))

            t_l = train(net, optimizer, train_loader, freeze_bn, use_lovasz)
            v_l, viou = valid(net, optimizer, valid_loader, use_lovasz, save_imgs, fold)

            #if swa:
               
            # save the model on best validation loss
            #if not args.cos_anneal:
            if viou > best_val_iou:
                net.eval()
                torch.save(net.state_dict(), model_ckpt)
                best_val_metric = v_l
                best_val_iou = viou
                valid_patience = 0
            else:
                valid_patience += 1

            # if the model stops improving by a certain num epoch, stop
            if cycle >= args.num_cycles:
                break

            # if the model doesn't improve for n epochs, reduce learning rate
            if cycle  >= 1: 
                if args.use_lovasz:
                    print('switching to lovasz')
                    use_lovasz = True
                   
                #dice_weight += 0.5
                if not args.cos_anneal:
                    print('Reducing learning rate by {}'.format(args.lr_scale))
                    for params in optimizer.param_groups:
                        params['lr'] *= args.lr_scale
            
            train_losses.append(t_l)
            valid_losses.append(v_l)
            valid_ious.append(viou)
 
            #if e in LR_SCHED:
            #    print('Reducing learning rate by {}'.format(args.lr_scale))
            #    for params in optimizer.param_groups:
            #        params['lr'] *= args.lr_scale

            t_ += 1
            print('Time: {}'.format(time.time()-start))

    except KeyboardInterrupt:
        pass

    if args.swa:
        for i in range(cycle):
            if i == 0:
                net.load_state_dict(torch.load('../swa/cycle_{}.pth'.format(i),
		                                 map_location=lambda storage, loc: storage))
            else:
                alpha = 1. / (i + 1.)
                prev = ResUNet()
                prev.load_state_dict(torch.load('../swa/cycle_{}.pth'.format(i),
		                                 map_location=lambda storage, loc: storage))
	        # average weights
                for param_c, param_p in zip(net.parameters(), prev.parameters()):
                    param_c.data *= (1.0 - alpha)
                    param_c.data += param_p.data.to(device) * alpha

        bn_update(train_loader, net, args.gpu)

    net.eval()
    torch.save(net.state_dict(), '../model_weights/swa_{}_{}_fold-{}.pth'.format(args.model_name, 
                                                                                 args.exp_name, fold))

    import pandas as pd

    out_dict = {'train_losses': train_losses,
                'valid_losses': valid_losses,
                'valid_ious': valid_ious}

    out_log = pd.DataFrame(out_dict)
    out_log.to_csv('../logs/resunet_fold-{}.csv'.format(fold), index=False)

    return best_val_iou

def train_folds():
    best_ious = []
    for fold in range(args.num_folds):

        #if fold > 0:
        #    break

        # set model filenames
        model_params = [args.model_name, args.exp_name, fold]
        MODEL_CKPT = '../model_weights/best_{}_{}_fold-{}.pth'.format(*model_params)

        net = ResUNet(use_bool=True)
        
        if args.gpu == 99:
            net = nn.parallel.DataParallel(net)
        net.to(device)

        print(net.parameters)
        print('Starting fold {} ...'.format(fold))
        best_ious.append(train_network(net, fold, model_ckpt=MODEL_CKPT))

    print('Average IOU:', np.mean(best_ious))

if __name__ == '__main__':
    train_folds()
