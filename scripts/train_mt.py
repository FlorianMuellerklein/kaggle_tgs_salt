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

#from apex.fp16_utils import FP16_Optimizer 

from models.nets import ResUNet
from utils.data_loaders import get_data_mt_loaders
from utils.data_vis import plot_from_torch

from utils.evaluations import FocalLoss2d, DiceLoss, get_iou_vector, ConsistencyLoss
import utils.lovasz_losses as L

parser = argparse.ArgumentParser(description='TGS Salt')
parser.add_argument('--imsize', default=128, type=int,
                    help='imsize to use for training')
parser.add_argument('--batch_size', default=32, type=int, 
                    help='size of batches')
parser.add_argument('--num_folds', default=5, type=int, 
                    help='number of cross val folds')
parser.add_argument('--epochs', default=1000, type=int, 
                    help='number of epochs')
#parser.add_argument('--lr', default=0.01, type=float,
#                    help='learning rate')
parser.add_argument('--lr_max', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_min', default=0.0003, type=float,
                    help='min lr for cosine annealing')
parser.add_argument('--num_cycles', default=7, type=int,
                    help='how many cyclic lr cycles')
parser.add_argument('--unlab_ratio', default=0.25, type=float,
                    help='ratio of unlabeled data in batch')
parser.add_argument('--wt_max', default=1, type=float,
                    help='max weight on consistency loss')
parser.add_argument('--rampup', default=10, type=int,
                    help='how long to ramp up the unsupervised weight')
parser.add_argument('--lr_rampup', default=1, type=int,
                    help='how long to ramp up the unsupervised weight')
parser.add_argument('--lr_rampdown', default=50, type=int,
                    help='how long to ramp up the unsupervised weight')
parser.add_argument('--lr_scale', default=0.1, type=float,
                    help='how much to reduce lr on plateau')
parser.add_argument('--l2', default=1e-5, type=float,
                    help='l2 regularization for model')
parser.add_argument('--lambda_dice', default=1.0, type=float,
                    help='lambda value for coordinate loss')
parser.add_argument('--es_patience', default=30, type=int, 
                    help='early stopping patience')
parser.add_argument('--lr_patience', default=20, type=int, 
                    help='early stopping patience')
parser.add_argument('--gpu', default=1, type=int, 
                    help='which gpu for training')
parser.add_argument('--model_name', default='resunet_mt', type=str,
                    help='name of model for saving/loading weights')
parser.add_argument('--exp_name', default='tgs_slt', type=str,
                    help='name of experiment for saving files')
parser.add_argument('--debug', action='store_true', 
                    help='whether to display debug info')
parser.add_argument('--load_best', action='store_true', 
                    help='load the previous best net to continue training')
parser.add_argument('--freeze_bn', action='store_true', 
                    help='freeze batch norm during finetuning')
parser.add_argument('--use_lovasz', action='store_true', 
                    help='whether to use focal loss during finetuning')
parser.add_argument('--cos_anneal', action='store_true', 
                    help='whether to use cosine annealing for learning rate')
args = parser.parse_args()

# define the loss functions
focal_loss = FocalLoss2d()
bce = nn.BCEWithLogitsLoss()
cl = ConsistencyLoss()
dice = DiceLoss()

# set up the dual gpu
#device = torch.device('cuda:1' if args.gpu else 'cpu')
#device_b = torch.device('cuda:0' if args.gpu else 'cpu')

def update_teacher(teacher, student, alpha=0.99):
    for param_t, param_s in zip(teacher.parameters(), student.parameters()):
        param_t.data *= alpha
        param_t.data += param_s.data * (1. - alpha)

# training function
def train(student, teacher, optimizer, train_loader, 
          w_t=0., e=0, freeze_bn=False, use_lovasz=False):
    '''
    uses the data loader to grab a batch of images
    pushes images through network and gathers predictions
    updates network weights by evaluating the loss functions
    '''
    # set network to train mode
    student.train(True, freeze_bn)
    teacher.train(True, freeze_bn)
    # keep track of our loss
    iter_loss = 0.
    iter_closs = 0.

    # loop over the images for the desired amount
    for i, data in enumerate(train_loader):
        imgs_a = data['img_a'].cuda()
        imgs_b = data['img_b'].cuda()
        msks = data['msk'].cuda()
        labeled_bool = data['is_labeled'].cuda()
        has_msk = data['has_msk'].float().cuda()

        mask = labeled_bool == 1

        if args.debug and i == 0:
            #print('{} labeled, {} total'.format(len(imgs_a[mask]), len(imgs_a)))
            
            img_a_grid = vsn.utils.make_grid(imgs_a, normalize=True)
            img_b_grid = vsn.utils.make_grid(imgs_b, normalize=True)
            msk_grid = vsn.utils.make_grid(msks)

            vsn.utils.save_image(img_a_grid, '../imgs/train_mt_imgs_a.png')
            vsn.utils.save_image(img_b_grid, '../imgs/train_mt_imgs_b.png')
            vsn.utils.save_image(msk_grid, '../imgs/train_msks.png')
            
        # zero gradients from previous run
        optimizer.zero_grad()

        # get predictions
        preds_a, bool_a = student(imgs_a)
        # calculate bce loss 
        if use_lovasz:
            loss_s = L.lovasz_hinge(preds_a[mask], msks[mask])
        else:
            loss_s = focal_loss(preds_a[mask], msks[mask])
            loss_s += L.lovasz_hinge(preds_a[mask], msks[mask])

        loss_s += bce(bool_a[mask], has_msk[mask].view(-1,1))

        # get the teacher predictions
        with torch.no_grad():
            preds_b, bool_b = teacher(imgs_b)
            
        loss_c = cl(preds_a, preds_b)
        loss_c += cl(bool_a, bool_b)
        loss = loss_s + w_t * loss_c

        #calculate gradients
        loss.backward()
        # update weights
        optimizer.step()

        # get training stats
        iter_loss += loss_s.item()
        iter_closs += loss_c.item()

        # update the teacher weights
        grad_step = e * (len(train_loader.dataset) / args.batch_size) + (i+1)
        alpha = min(1. - 1. / (grad_step + 1), 0.99)
        update_teacher(teacher, student, alpha=alpha)
        # make a cool terminal output
        sys.stdout.write('\r')
        sys.stdout.write('B: {:>3}/{:<3} | loss: {:.4} | step: {} alpha: {:.4}'.format(i+1, 
                                            len(train_loader),
                                            loss.item(),
                                            int(grad_step),
                                            alpha))

    epoch_loss = iter_loss / (len(train_loader.dataset) / args.batch_size)
    epoch_closs = iter_closs  / (len(train_loader.dataset) / args.batch_size)
    print('\n' + 'Avg Train Loss: {:.4}, Avg Consist. Loss: {:.4}'.format(epoch_loss,
                                                                          epoch_closs))

    return epoch_loss

# validation function
def valid(net, optimizer, valid_loader, mtype='student', use_lovasz=False):
    net.eval() 
    # keep track of losses
    val_ious = []
    val_iter_loss = 0.
    # no gradients during validation
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            valid_imgs = data['img'].cuda()
            valid_msks = data['msk'].cuda()
            valid_msk_bool = data['has_msk'].float().cuda()
            # get predictions
            msk_vpreds, bool_v = net(valid_imgs)
            # calculate loss
            if use_lovasz:
                vloss = L.lovasz_hinge(msk_vpreds, valid_msks)
            else:
                vloss = focal_loss(msk_vpreds, valid_msks)
                vloss += L.lovasz_hinge(msk_vpreds, valid_msks)
            vloss += bce(bool_v, valid_msk_bool.view(-1,1))
            # get validation stats
            val_iter_loss += vloss.item()
            
            val_ious.append(get_iou_vector(valid_msks.cpu().numpy()[:,:,13:114, 13:114], 
                                           msk_vpreds.sigmoid().cpu().numpy()[:,:,13:114, 13:114]))
            
    epoch_vloss = val_iter_loss / (len(valid_loader.dataset) / args.batch_size)
    print('{} Avg Eval Loss: {:.4}, Avg IOU: {:.4}'.format(mtype, epoch_vloss, np.mean(val_ious)))
    return epoch_vloss, np.mean(val_ious)

def train_network(student, teacher, fold=0, model_ckpt=None):
    # train the network, allow for keyboard interrupt
    try:
        # define optimizer
        optimizer = optim.SGD(student.parameters(), lr=args.lr_max, momentum=0.9)
        # get the loaders
        train_loader, valid_loader = get_data_mt_loaders(imsize=args.imsize,
                                                             batch_size=args.batch_size,
                                                             num_folds=args.num_folds,
                                                             fold=fold,
                                                             unlabeled_ratio=args.unlab_ratio)


        # start training with BN on
        use_wt = True
        use_lovasz = False
        freeze_bn = False
        train_losses = []
        valid_losses_s = []
        valid_ious_s = []
        valid_losses_t = []
        valid_ious_t = []

        valid_patience = 0
        best_val_metric = 1000.0
        best_val_iou = 0.0
        cycle = 0
        t_ = 0
        w_t = 0.
        mt_counter = 0

        print('Training ...')
        for e in range(args.epochs):
            print('\n' + 'Epoch {}/{}'.format(e, args.epochs))
            # LR warm-up            
            #if e < args.lr_rampup:
            #    lr = args.lr * (min(t_+1, args.lr_rampup) / args.lr_rampup)
            
            # if we get to the end of lr period, save swa weights
            if t_ >= args.lr_rampdown:
                # reset the counter
                t_ = 0
                cycle += 1
                save_imgs = True
                torch.save(net.state_dict(), 
                           '../model_weights/{}_{}_cycle-{}_fold-{}.pth'.format(args.model_name,
                                                                                args.exp_name,
                                                                                cycle,
                                                                                fold))

            for params in optimizer.param_groups:
                if args.cos_anneal:
                    params['lr'] = (args.lr_min + 0.5 * (args.lr_max - args.lr_min) *
                                   (1 + np.cos(np.pi * t_ / args.lr_rampdown)))

                #elif e < args.lr_rampup:
                #    params['lr'] = args.lr * (min(t_+1, args.lr_rampup) / args.lr_rampup)

                print('Learning rate set to {:.4}'.format(optimizer.param_groups[0]['lr']))

                start = time.time()

            t_l = train(student, teacher, optimizer, train_loader, w_t, e, freeze_bn, use_lovasz)
            v_l_s, viou_s = valid(student, optimizer, valid_loader, 'student', use_lovasz)
            v_l_t, viou_t = valid(teacher, optimizer, valid_loader, 'teacher', use_lovasz)

            # save the model on best validation loss
            if viou_t > best_val_iou:
                teacher.eval()
                torch.save(teacher.state_dict(), model_ckpt)
                best_val_metric = v_l_t
                best_val_iou = viou_t
                valid_patience = 0
            # only start using the patience values when we get past the rampup period
            else:
                valid_patience += 1

            # if the model stops improving by a certain num epoch, stop
            if cycle == args.num_cycles:
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
 

            # if the model doesn't improve for n epochs, reduce learning rate
            if valid_patience == args.lr_patience:
                if args.freeze_bn:
                    freeze_bn = True
                    #batch_size = batch_size // 2 if batch_size // 2 >= 16 else 16 
                if args.use_lovasz:
                    use_lovasz = True
                #use_wt = True
                #w_t = args.wt_max

                #print('Reducing learning rate by {}'.format(args.lr_scale))
                #for params in optimizer.param_groups:
                #    params['lr'] *= args.lr_scale

            # record losses
            train_losses.append(t_l)
            valid_losses_s.append(v_l_s)
            valid_ious_s.append(viou_s)
            valid_losses_t.append(v_l_t)
            valid_ious_t.append(viou_t)
            # update w_t
            w_t = args.wt_max * math.exp(-5 * (1. - (min(t_, args.rampup) / args.rampup))**2)
            t_ += 1
            print('Setting w_t to {:.4}'.format(w_t))
            print('Time: {}'.format(time.time()-start))

    except KeyboardInterrupt:
        pass

    import pandas as pd

    out_dict = {'train_losses': train_losses,
                'valid_losses_s': valid_losses_s,
                'valid_ious_s': valid_ious_s,
                'valid_losses_t': valid_losses_t,
                'valid_ious_t': valid_ious_t}

    out_log = pd.DataFrame(out_dict)
    out_log.to_csv('../logs/mt_fold-{}.csv'.format(fold), index=False)

    return best_val_iou

def train_folds():
    best_ious = []
    for fold in range(args.num_folds):

        #if fold > 0:
        #    break

        # set model filenames
        model_params = [args.model_name, args.exp_name, fold]
        MODEL_CKPT = '../model_weights/best_mt_{}_{}_fold-{}.pth'.format(*model_params)

        if args.load_best:
            net.load_state_dict(torch.load(MODEL_CKPT, 
                                           map_location=lambda storage, 
                                           loc: storage))

        student = ResUNet(use_bool=True)
        teacher = ResUNet(use_bool=True)
        for param in teacher.parameters():
            param.detach_()
            
        if args.gpu == 99:
            student = nn.DataParallel(student, device_ids=[0,1]).cuda()
            teacher = nn.DataParallel(teacher, device_ids=[0,1]).cuda()
        else:
            torch.cuda.set_device(args.gpu)
            cudnn.benchmark = True
            student.cuda()
            teacher.cuda()

        print('Starting fold {} ...'.format(fold))
        best_ious.append(train_network(student, teacher, fold, model_ckpt=MODEL_CKPT))

    print('Average IOU:', np.mean(best_ious))

if __name__ == '__main__':
    train_folds()
