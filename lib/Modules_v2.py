# -*- coding: utf-8 -*-
import math

import torch.nn as nn

affine_par = True
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch.autograd import Variable
import numpy as np
from functools import partial
from lib.ops.modules import MSDeformAttn
from timm.models.layers import DropPath
from torch.nn.init import normal_
import torch.utils.checkpoint as cp

"""
ASPP
"""


class PAM(nn.Module):
    def __init__(self, in_channels):
        super(PAM, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x

        return out


class GPM(nn.Module):
    def __init__(self, dilation_series=[6, 12, 18], padding_series=[6, 12, 18], depth=128):
        # def __init__(self, dilation_series=[2, 5, 7], padding_series=[2, 5, 7], depth=128):
        super(GPM, self).__init__()
        self.branch_main = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BasicConv2d(2048, depth, kernel_size=1, stride=1)
        )
        self.branch0 = BasicConv2d(2048, depth, kernel_size=1, stride=1)
        self.branch1 = BasicConv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[0],
                                   dilation=dilation_series[0])
        self.branch2 = BasicConv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[1],
                                   dilation=dilation_series[1])
        self.branch3 = BasicConv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[2],
                                   dilation=dilation_series[2])
        self.head = nn.Sequential(
            BasicConv2d(depth * 5, 256, kernel_size=3, padding=1),
            PAM(256)
        )
        self.out = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64, affine=affine_par),
            nn.PReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 1, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # out = self.conv2d_list[0](x)
        # mulBranches = [conv2d_l(x) for conv2d_l in self.conv2d_list]
        size = x.shape[2:]
        branch_main = self.branch_main(x)
        branch_main = F.interpolate(branch_main, size=size, mode='bilinear', align_corners=True)
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        out = torch.cat([branch_main, branch0, branch1, branch2, branch3], 1)
        out = self.head(out)
        out = self.out(out)
        return out


"""
ASPP
in_channel是输入特征通道维度
depth是输出特征通道维度
"""


class ASPP(nn.Module):
    def __init__(self, in_channel=2048, depth=128):
        super(ASPP, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]
        # mean.shape = torch.Size([8, 3, 1, 1])
        image_features = self.mean(x)
        # conv.shape = torch.Size([8, 3, 1, 1])
        image_features = self.conv(image_features)
        # interpolate.shape = torch.Size([8, 3, 32, 32])
        image_features = F.interpolate(image_features, size=size, mode='bilinear', align_corners=False)

        # block1.shape = torch.Size([8, 3, 32, 32])
        atrous_block1 = self.atrous_block1(x)

        # block6.shape = torch.Size([8, 3, 32, 32])
        atrous_block6 = self.atrous_block6(x)

        # block12.shape = torch.Size([8, 3, 32, 32])
        atrous_block12 = self.atrous_block12(x)

        # block18.shape = torch.Size([8, 3, 32, 32])
        atrous_block18 = self.atrous_block18(x)

        # torch.cat.shape = torch.Size([8, 15, 32, 32])
        # conv_1x1.shape = torch.Size([8, 3, 32, 32])
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))

        return net


"""
Laplacian Pyramid
"""


def get_kernel_gussian(kernel_size, Sigma=1, in_channels=128):
    kernel_weights = cv2.getGaussianKernel(ksize=kernel_size, sigma=Sigma)
    kernel_weights = kernel_weights * kernel_weights.T
    kernel_weights = np.expand_dims(kernel_weights, axis=0)
    kernel_weights = np.repeat(kernel_weights, in_channels, axis=0)
    kernel_weights = np.expand_dims(kernel_weights, axis=1)
    return kernel_weights


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class LP(nn.Module):
    def __init__(self, in_channel, depth):
        super(LP, self).__init__()
        self.in_channel = in_channel
        self.ca = CALayer(in_channel)
        self.out = nn.Sequential(
            BasicConv2d(in_channel*5, depth, kernel_size=1),
        )

    def forward(self, x):
        _, c, h, w = x.size()
        # ## parameters
        kernet_shapes = [3, 5, 7, 9]
        k_value = np.power(2, 1 / 3)
        sigma = 1.6

        ## Kernel weights for Laplacian pyramid
        Sigma1_kernel = get_kernel_gussian(kernel_size=kernet_shapes[0], Sigma=sigma * np.power(k_value, 1), in_channels=self.in_channel).astype(np.float32)
        Sigma2_kernel = get_kernel_gussian(kernel_size=kernet_shapes[1], Sigma=sigma * np.power(k_value, 2), in_channels=self.in_channel).astype(np.float32)
        Sigma3_kernel = get_kernel_gussian(kernel_size=kernet_shapes[2], Sigma=sigma * np.power(k_value, 3), in_channels=self.in_channel).astype(np.float32)
        Sigma4_kernel = get_kernel_gussian(kernel_size=kernet_shapes[3], Sigma=sigma * np.power(k_value, 4), in_channels=self.in_channel).astype(np.float32)

        g0 = x
        g1 = F.conv2d(x, torch.tensor(Sigma1_kernel).to(x.device), None, stride=1, padding=1, groups=c)
        g2 = F.conv2d(x, torch.tensor(Sigma2_kernel).to(x.device), None, stride=1, padding=2, groups=c)
        g3 = F.conv2d(x, torch.tensor(Sigma3_kernel).to(x.device), None, stride=1, padding=3, groups=c)
        g4 = F.conv2d(x, torch.tensor(Sigma4_kernel).to(x.device), None, stride=1, padding=4, groups=c)

        ## Laplacian Pyramid
        L0 = g0
        L1 = g0 - g1
        L2 = g1 - g2
        L3 = g2 - g3
        L4 = g3 - g4

        m0 = self.ca(L0)
        m1 = self.ca(L1)
        m2 = self.ca(L2)
        m3 = self.ca(L3)
        m4 = self.ca(L4)
        m = torch.cat([m0, m1, m2, m3, m4], dim=1)
        out = self.out(m)
        return out


"""
Muti-scale Frequency Feature Localization
"""


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs_only_one(x, h, w):
    # bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8),
                                             (h // 16, w // 16),
                                             (h // 32, w // 32)], device=x.device)
    deform_inputs = [reference_points, spatial_shapes, level_start_index]

    return deform_inputs


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x1 = x[:, 0:H*W, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x2 = x[:, H*W:H*W+H*W//4, :].transpose(1, 2).view(B, C, H//2, W//2).contiguous()
        x3 = x[:, H*W+H*W//4:, :].transpose(1, 2).view(B, C, H//4, W //4).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MFFL(nn.Module):
    def __init__(self, in_channel, depth=64, num_heads=4, n_points=4, n_levels=3, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False, drop=0., drop_path=0., cffn_ratio=0.25):
        super(MFFL, self).__init__()
        self.depth = depth
        self.lp2 = LP(in_channel*8, depth)
        self.lp3 = LP(in_channel*16, depth)
        self.lp4 = LP(in_channel*32, depth)

        self.level_embed = nn.Parameter(torch.zeros(3, depth))

        self.with_cp = with_cp
        self.query_norm = norm_layer(depth)
        self.feat_norm = norm_layer(depth)
        self.attn = MSDeformAttn(d_model=depth, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(init_values * torch.ones(depth), requires_grad=True)

        self.ffn = ConvFFN(in_features=depth, hidden_features=int(depth * cffn_ratio), drop=drop)
        self.ffn_norm = norm_layer(depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.conv_1x1_output = nn.Conv2d(depth * 3, 1, 1, 1)
        normal_(self.level_embed)

    def forward(self, f2, f3, f4):
        B, C, H, W = f2.shape
        f2 = self.lp2(f2)
        f3 = self.lp3(f3)
        f4 = self.lp4(f4)

        f2 = f2.view(B, self.depth, -1).permute(0, 2, 1)
        f3 = f3.view(B, self.depth, -1).permute(0, 2, 1)
        f4 = f4.view(B, self.depth, -1).permute(0, 2, 1)

        f2, f3, f4 = self._add_level_embed(f2, f3, f4)

        f = torch.cat([f2, f3, f4], dim=1)

        deform_inputs = deform_inputs_only_one(f2, H * 8, W * 8)

        def _inner_forward(feat, h, w):
            B, N, C = feat.shape
            c1 = self.attn(self.query_norm(feat), deform_inputs[0],
                           self.feat_norm(feat), deform_inputs[1],
                           deform_inputs[2], None)

            c1 = c1 + self.drop_path(self.ffn(self.ffn_norm(c1), h, w))

            c_select1, c_select2, c_select3 = c1[:, :h * w, :], c1[:, h * w:h * w + h * w // 4, :], c1[:, h * w + h * w // 4:, :]
            c_select1 = c_select1.permute(0, 2, 1).reshape(B, C, h, w)
            c_select2 = F.interpolate(c_select2.permute(0, 2, 1).reshape(B, C, h // 2, w // 2), scale_factor=2, mode='bilinear', align_corners=False)
            c_select3 = F.interpolate(c_select3.permute(0, 2, 1).reshape(B, C, h // 4, w // 4), scale_factor=4, mode='bilinear', align_corners=False)
            c_select = torch.cat([c_select1, c_select2, c_select3],dim=1)
            location = self.conv_1x1_output(c_select)
            return location

        if self.with_cp and f.requires_grad:
            out = cp.checkpoint(_inner_forward, f, H, W)
        else:
            out = _inner_forward(f, H, W)

        return out

    def _add_level_embed(self, f2, f3, f4):
        f2 = f2 + self.level_embed[0]
        f3 = f3 + self.level_embed[1]
        f4 = f4 + self.level_embed[2]
        return f2, f3, f4


class MFFL2(nn.Module):
    def __init__(self, in_channel, depth=64):
        super(MFFL2, self).__init__()
        self.depth = depth
        self.lp2 = LP(in_channel, depth)
        self.lp3 = LP(in_channel, depth)
        self.lp4 = LP(in_channel, depth)

        self.atrous_block1 = BasicConv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = BasicConv2d(in_channel, depth, 3, 1, padding=3, dilation=3)
        self.atrous_block12 = BasicConv2d(in_channel, depth, 3, 1, padding=6, dilation=6)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(depth * 3, 3)

        self.conv_1x1_output = nn.Conv2d(depth * 3, 1, 1, 1)

    def forward(self, f2, f3, f4):
        B, C, H, W = f2.shape
        f2 = self.atrous_block1(self.lp2(f2))
        f3 = self.atrous_block6(self.lp3(f3))
        f4 = self.atrous_block6(self.lp4(f4))
        f2_num = self.avg_pool(f2).view(B, -1)
        f3_num = self.avg_pool(f3).view(B, -1)
        f4_num = self.avg_pool(f4).view(B, -1)
        weight = self.fc(torch.cat([f2_num, f3_num, f4_num], dim=1))
        weight = torch.chunk(torch.softmax(weight, dim=1), chunks=3, dim=1)
        f2 = f2 * weight[0].unsqueeze(-1).unsqueeze(-1)
        f3 = F.interpolate(f3, size=(H, W), mode='bilinear', align_corners=False) * weight[1].unsqueeze(-1).unsqueeze(-1)
        f4 = F.interpolate(f4, size=(H, W), mode='bilinear', align_corners=False) * weight[2].unsqueeze(-1).unsqueeze(-1)
        f = torch.cat([f2, f3, f4], dim=1)
        out = self.conv_1x1_output(f)
        return out

"""
FPN
"""


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x


class ETM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ETM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = BasicConv2d(in_channels, out_channels, 1)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channels, out_channels, 3, padding=1)
        self.conv_res = BasicConv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()

        # Lateral layers
        self.T1 = ETM(in_channels, out_channels)
        self.T2 = ETM(in_channels * 2, out_channels)
        self.T3 = ETM(in_channels * 4, out_channels)
        self.T4 = ETM(in_channels * 8, out_channels)

        # Smooth layers
        self.smooth1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, f1, f2, f3, f4):
        f1 = self.T1(f1)
        f2 = self.T2(f2)
        f3 = self.T3(f3)
        f4 = self.T4(f4)

        p3 = self._interpolate_add(f4, f3)
        p2 = self._interpolate_add(p3, f2)
        p1 = self._interpolate_add(p2, f1)

        p3 = self.smooth1(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth3(p1)

        return p1, p2, p3, f1, f2, f3, f4

    def _interpolate_add(self, x, y):
        '''interpolate and add two feature maps.

        Args:
          x: (Variable) top feature map to be interpolated.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the interpolated feature map
        with `F.interpolate(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        interpolated feature map size: [N,_,16,16]

        So we choose bilinear interpolate which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y


"""
边缘扩散模块
"""

class OGSA(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(OGSA, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, prior_cam):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature
                attention: B * N * N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B*N*C/8
        proj_query += torch.from_numpy(self.get_2d_sincos_pos_embed(C // 8, width)).float().unsqueeze(0).to(x.device)
        proj_key = (self.key_conv(x).mul(prior_cam.expand(-1, x.size()[1] // 8, -1, -1))).view(m_batchsize, -1, width * height)  # B*C/8*N
        proj_key += torch.from_numpy(self.get_2d_sincos_pos_embed(C // 8, width)).float().transpose(0, 1).contiguous().unsqueeze(0).to(x.device)
        energy = torch.bmm(proj_query, proj_key)  # batch的matmul B*N*N
        attention = self.softmax(energy / (C // 8) ** 0.5)  # B * (N) * (N)
        proj_value = (self.value_conv(x).mul(prior_cam.expand(-1, x.size()[1], -1, -1))).view(m_batchsize, -1, width * height)  # B * C * N
        proj_value += torch.from_numpy(self.get_2d_sincos_pos_embed(C, width)).float().transpose(0, 1).contiguous().unsqueeze(0).to(x.device)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B*C*N
        out = out.view(m_batchsize, C, width, height)  # B*C*H*W

        out = self.gamma * out + x
        return out

    def get_2d_sincos_pos_embed(self, embed_dim, grid_size, cls_token=False, extra_tokens=0):
        """
        grid_size: int of the grid height and width
        return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token and extra_tokens > 0:
            pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
        return pos_embed

    def get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

        emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.
        omega = 1. / 10000 ** omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    def positionalencoding1d(self, length, d_model):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe


"""
mask guided attention
"""


class MGA(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(MGA, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, prior_cam):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature
                attention: B * N * N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        query = self.query_conv(x).view(m_batchsize, C, -1)  # B*C*HW

        mask = torch.sigmoid(prior_cam)
        mask = torch.cat([mask, 1 - mask], dim=1)
        key = self.key_conv(mask).view(m_batchsize, 2, -1).permute(0, 2, 1)  # B*HW*2

        energy = torch.bmm(query, key)  # batch的matmul B*C*2
        attention = self.softmax(energy / (width * height) ** 0.5)
        value = self.value_conv(mask).view(m_batchsize, 2, -1)  # B * 2 * HW

        out = torch.bmm(attention, value)  # B*2*HW
        out = out.view(m_batchsize, C, width, height)  # B*C*H*W

        out = self.gamma * out + x
        return out

    def get_2d_sincos_pos_embed(self, embed_dim, grid_size, cls_token=False, extra_tokens=0):
        """
        grid_size: int of the grid height and width
        return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token and extra_tokens > 0:
            pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
        return pos_embed

    def get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

        emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.
        omega = 1. / 10000 ** omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    def positionalencoding1d(self, length, d_model):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe


"""
    guided filter
"""


# BoxFilter==MeanFilter
class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def diff_x(self, input, r):
        assert input.dim() == 4

        left = input[:, :, r:2 * r + 1]
        middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
        right = input[:, :, -1:] - input[:, :, -2 * r - 1:    -r - 1]

        output = torch.cat([left, middle, right], dim=2)

        return output

    def diff_y(self, input, r):
        assert input.dim() == 4

        left = input[:, :, :, r:2 * r + 1]
        middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
        right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1:    -r - 1]

        output = torch.cat([left, middle, right], dim=3)

        return output

    def forward(self, x):
        assert x.dim() == 4
        return self.diff_y(self.diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)

    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = self.boxfilter(Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))

        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N
        # cov_xy
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        return mean_A * x + mean_b

'''
fusion
'''
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


"""
Edge Diffusion module
"""


class EDM(nn.Module):
    def __init__(self, in_channels, r):
        super(EDM, self).__init__()
        self.mga = MGA(in_channels)
        self.GF = GuidedFilter(r=r, eps=1e-2)
        self.out = nn.Sequential(
            BasicConv2d(in_channels * 2, in_channels, 3, padding=1),
            BasicConv2d(in_channels, in_channels, 3, padding=1),
            nn.Conv2d(in_channels, 1, 1))

    def forward(self, x, prior_cam):
        _, _, H, W = x.size()
        prior_cam_up = F.interpolate(prior_cam, size=(H, W), mode='bilinear', align_corners=False)
        prior_attn_out = self.mga(x, prior_cam_up)
        edge_diffsion = self.GF(x, prior_cam_up.expand(-1, x.size()[1], -1, -1))
        out = self.out(torch.cat([prior_attn_out, edge_diffsion], dim=1)) + prior_cam_up
        return out

class EDM2(nn.Module):
    def __init__(self, in_channels, r):
        super(EDM2, self).__init__()
        self.mga = MGA(in_channels)
        self.GF = GuidedFilter(r=r, eps=1e-2)
        self.out = nn.Sequential(
            BasicConv2d(in_channels * 2, in_channels, 3, padding=1),
            BasicConv2d(in_channels, in_channels, 3, padding=1),
            nn.Conv2d(in_channels, 1, 1))
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x, prior_cam):
        _, _, H, W = x.size()
        prior_cam_up = F.interpolate(prior_cam, size=(H, W), mode='bilinear', align_corners=False)
        prior_attn_out = self.mga(x, prior_cam_up)
        edge_diffsion = self.GF(x, prior_cam_up.expand(-1, x.size()[1], -1, -1))
        f1 = edge_diffsion + prior_attn_out * self.ca(prior_attn_out)
        f2 = prior_attn_out - f1 * self.sa(f1)
        out = self.out(torch.cat([f1, f2], dim=1)) + prior_cam_up
        return out

"""
Muti scale Edge Diffusion module
"""


class MCEDM(nn.Module):
    def __init__(self, in_channels):
        super(MCEDM, self).__init__()
        self.edm2 = EDM2(in_channels, r=4)
        self.edm3 = EDM2(in_channels, r=8)
        self.edm4 = EDM2(in_channels, r=16)

    def forward(self, x, prior_cam, pic):
        p1, p2, p3 = x
        prior3 = self.edm2(p3, prior_cam)
        out3 = F.interpolate(prior3, size=pic.size()[2:], mode='bilinear', align_corners=False)

        prior2 = self.edm3(p2, prior3)
        out2 = F.interpolate(prior2, size=pic.size()[2:], mode='bilinear', align_corners=False)

        prior1 = self.edm4(p1, prior2)
        out1 = F.interpolate(prior1, size=pic.size()[2:], mode='bilinear', align_corners=False)

        return out1, out2, out3


if __name__ == "__main__":
    feature = torch.rand(2, 2048, 12, 12).cuda()
    model = MFFL().cuda()
    pred_cam = model(feature, "cuda")
    print(pred_cam.shape)
