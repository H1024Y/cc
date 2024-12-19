# author: 
# data: 
import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from lib.EDNet_v2 import EDNet, EDNet2
from utils.data_val import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, get_coef,cal_ual
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from torch import optim
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()

def train(train_loader, model, optimizer, epoch, save_path, writer):
    """
    train function
    """
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, edges, fixs) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda(device=device_ids[0])
            gts = gts.cuda(device=device_ids[0])
            edges = edges.cuda(device=device_ids[0])
            fixs = fixs.cuda(device=device_ids[0])
            preds = model(images)

            ual_coef = get_coef(iter_percentage=i/total_step, method='cos')
            ual_loss = cal_ual(seg_logits=preds[1], seg_gts=gts)
            ual_loss *= ual_coef

            loss_init = structure_loss(preds[0], gts)*0.125 + structure_loss(preds[3], gts)*0.25 + \
                        structure_loss(preds[2], gts)*0.5
            
            loss_final = structure_loss(preds[1], gts)

            loss = loss_init+loss_final+ual_loss

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            #在最开始、每20次迭代、迭代完时，记录一次
            if i % 50 == 0 or i == total_step or i == 1:
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                        format(epoch, opt.epoch, i, total_step, loss.data))
                # TensorboardX-Loss
                writer.add_scalars('Loss_Statistics',
                                   { 'Loss_total': loss.data},
                                   global_step=step)
                # TensorboardX-Training Data
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('GT', grid_image, step)
                grid_image = make_grid(edges[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Edge', grid_image, step)
                # grid_image = make_grid(fixs[0].clone().cpu().data, 1, normalize=True)
                # writer.add_image('Fix', grid_image, step)

                # TensorboardX-Outputs
                res = preds[0][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_cam', torch.tensor(res), step, dataformats='HW')

                res = preds[1][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_final', torch.tensor(res), step, dataformats='HW')

                res = preds[3][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_down', torch.tensor(res), step, dataformats='HW')

                res = preds[2][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_up', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step #求average loss
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch % 80 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer):
    """
    validation function
    """
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        mae_sum_edge = 0
        for i in range(test_loader.size):
            image, gt,  name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            # edge = np.asarray(edge, np.float32)
            gt /= (gt.max() + 1e-8)
            # edge /= (edge.max() + 1e-8)
            image = image.cuda(device=device_ids[0])

            result = model(image)

            res = F.upsample(result[1], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])


        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
            best_epoch = 1
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        logging.info(
            '[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))
        # if best_mae < 0.03293:
        #     torch.save(model.state_dict(), save_path + 'Net_epoch_mae_0.03293_{}.pth'.format(epoch))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=800, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=32, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=576, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=80, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--train_root', type=str, default='/data0/hcm/dataset/COD/TrainDataset/',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='/data0/hcm/dataset/COD/TestDataset/COD10K/',
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str, 
                        default='./snapshot/EDNet_V2_576/',
                        help='the path to save model and log')
    opt = parser.parse_args()

    setup_seed(981023)

    # set the device for training
    # if opt.gpu_id == '0':
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #     print('USE GPU 0')
    # elif opt.gpu_id == '1':
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    #     print('USE GPU 1')
    # cudnn.benchmark = True

    # # build the model
    # model = Network(channels=32).cuda()

    if opt.gpu_id == '0':
        tmp = "2,3"
        os.environ["CUDA_VISIBLE_DEVICES"] = tmp
        print('USE GPU '+tmp)
    elif opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
        print('USE GPU 1,2,3')
    cudnn.benchmark = True

    # build the model
    device_ids = [0,1]
    model = torch.nn.DataParallel(EDNet(channels=64), device_ids=device_ids)
    model = model.cuda(device=device_ids[0])

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Imgs/',
                              gt_root=opt.train_root + 'GT/',
                              edge_root=opt.train_root + 'Edge/',
                              fix_root=opt.train_root + 'Fix/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=8)
    val_loader = test_dataset(image_root=opt.val_root + 'Imgs/',
                              gt_root=opt.val_root + 'GT/',
                              testsize=opt.trainsize)
    total_step = len(train_loader)


    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))

    step = 0
    writer = SummaryWriter(save_path + 'summary')
    best_mae = 1
    best_epoch = 0

    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=50, eta_min=1e-6)
    print("Start train...")
    for epoch in range(1, opt.epoch):
        # cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        # writer.add_scalar('learning_rate', cur_lr, global_step=epoch)

        cosine_schedule.step()
        writer.add_scalar('learning_rate', cosine_schedule.get_lr()[0], global_step=epoch)
        logging.info('>>> current lr: {}'.format(cosine_schedule.get_lr()[0]))
        
        train(train_loader, model, optimizer, epoch, save_path, writer)
        val(val_loader, model, epoch, save_path, writer)

