from lib.Modules_v2 import FPN, MFFL, MCEDM, MFFL2
from lib.ops.modules import MSDeformAttn
import timm
import torch.nn as nn
import torch.nn.functional as F
import torch
from timm.models.layers import DropPath, trunc_normal_
import math


class EDNet(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=32):
        super(EDNet, self).__init__()
        # ---- ResNet Backbone ----
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        self.fpn = FPN(256, channels)
        self.mffl = MFFL(channels)
        self.mceam = MCEDM(channels)

        self.mffl.apply(self._init_weights)
        self.mceam.apply(self._init_weights)
        self.fpn.apply(self._init_weights)
        self.apply(self._init_deform_weights)

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        image = x
        # Feature Extraction
        en_feats = self.shared_encoder(x)
        x0, x1, x2, x3, x4 = en_feats
        p1, p2, p3, f1, f2, f3, f4 = self.fpn(x1, x2, x3, x4)

        prior_cam = self.mffl(x2, x3, x4)  # bs, 1, 12, 12

        pred_cam = F.interpolate(prior_cam, size=image.size()[2:], mode='bilinear', align_corners=False)
        out1, out2, out3 = self.mceam([p1, p2, p3], prior_cam, image)
        return pred_cam, out1, out2, out3


class EDNet2(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=32):
        super(EDNet2, self).__init__()
        # ---- ResNet Backbone ----
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        self.fpn = FPN(256, channels)
        self.mffl = MFFL2(channels)
        self.mceam = MCEDM(channels)

        self.mffl.apply(self._init_weights)
        self.mceam.apply(self._init_weights)
        self.fpn.apply(self._init_weights)
        self.apply(self._init_deform_weights)

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        image = x
        # Feature Extraction
        en_feats = self.shared_encoder(x)
        x0, x1, x2, x3, x4 = en_feats
        p1, p2, p3, f1, f2, f3, f4 = self.fpn(x1, x2, x3, x4)

        prior_cam = self.mffl(f2, f3, f4)  # bs, 1, 12, 12

        pred_cam = F.interpolate(prior_cam, size=image.size()[2:], mode='bilinear', align_corners=False)
        out1, out2, out3 = self.mceam([p1, p2, p3], prior_cam, image)
        return pred_cam, out1, out2, out3

if __name__ == '__main__':
    image = torch.rand(2, 3, 384, 384).cuda()
    model = EDNet(64).cuda()
    pred_cam, out1, out2, out3 = model(image)
    print(pred_cam.shape)
    print(out1.shape)
    print(out2.shape)
    print(out3.shape)
