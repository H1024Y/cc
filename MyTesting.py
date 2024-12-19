import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
# from scipy import misc
import cv2
from lib.EDNet_v2 import EDNet
from utils.data_val import test_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--pth_path', type=str, default=r'snapshot/EDNet_V2_4/Net_epoch_best.pth')
opt = parser.parse_args()

for _data_name in ['CAMO', 'CHAMELEON','COD10K','NC4K']:
# for _data_name in ['CHAMELEON']:
    data_path = r'/data0/hcm/dataset/COD/TestDataset/{}/'.format(_data_name)
    save_path = './res/{}/{}/'.format(opt.pth_path.split('/')[-2], _data_name)
    model = EDNet(channels=64)  # can be different under diverse backbone
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(opt.pth_path, map_location=torch.device('cpu')).items()})
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    # edge_root = '{}/Edge/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name, _ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        result = model(image)
        print('> {} - {}'.format(_data_name, name))

        res = F.upsample(result[1], size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path + name, res * 255)
