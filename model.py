# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict

from config import config


BN_MOMENTUM = 0.1
logging.basicConfig(filename=config.LOG_DIR, filemode='w', level=logging.INFO, format='%(asctime)s => %(message)s')
# logging.basicConfig(level=logging.INFO, format='%(name)s :: %(asctime)s => %(message)s')
# logging.basicConfig(level=logging.INFO, format='%(asctime)s => %(message)s')


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck_CAFFE(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_CAFFE, self).__init__()
        # add stride to conv1x1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseFix(nn.Module):
    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        super(PoseFix, self).__init__()
        self.conv1 = nn.Conv2d(3+cfg.MODEL.NUM_JOINTS, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            # layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            
            for layer in layers:
                if isinstance(layer, nn.Conv2d):
                    layer.weight.data.normal_(mean=0, std=0.01)
            
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x, hm):
        x = torch.cat([x, hm], axis=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x

    def init_weights(self, cfg, pretrained='', download=True):
        # if os.path.isfile(pretrained) or download:
        #     # pretrained_state_dict = torch.load(pretrained)
        #     logging.info('=> loading pretrained model {}'.format(pretrained))
        #     # self.load_state_dict(pretrained_state_dict, strict=False)
        #     try:
        #         checkpoint = torch.load(pretrained)
        #     except FileNotFoundError as e:
        #         logging.info('=> Pretrained weight does not exsist.')
        #         logging.info('=> Downlaod from torch.hub')
                
        #         model_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
        #         checkpoint = torch.hub.load_state_dict_from_url(model_url, progress=True)
                
        #     if isinstance(checkpoint, OrderedDict):
        #         state_dict = checkpoint
        #     elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        #         state_dict_old = checkpoint['state_dict']
        #         state_dict = OrderedDict()
        #         # delete 'module.' because it is saved from DataParallel module
        #         for key in state_dict_old.keys():
        #             if key.startswith('module.'):
        #                 # state_dict[key[7:]] = state_dict[key]
        #                 # state_dict.pop(key)
        #                 state_dict[key[7:]] = state_dict_old[key]
        #             else:
        #                 state_dict[key] = state_dict_old[key]
        #     else:
        #         raise RuntimeError(
        #             'No state_dict found in checkpoint file {}'.format(pretrained))
        #     self.load_state_dict(state_dict, strict=False)
        #     logging.info("=> Backborn model weight's loaded.")
        # else:
        #     logging.info('=> download ResNet pretrained model')
        #     logging.info("=> from pytorch model zoo")
            
        #     return ValueError
        #     # model_dir_dict = {
        #     #     50:  'resnet50-19c8e357.pth',
        #     #     101: 'resnet101-5d3b4d8f.pth',
        #     #     152: 'resnet152-b121ed2d.pth',
        #     # }
            
        #     # checkpoint = torch.hub.load_state_dict_from_url(f'https://s3.amazonaws.com/pytorch/models/{model_dir_dict[int(cfg.MODEL.EXTRA.NUM_LAYERS)]}')
        #     # if isinstance(checkpoint, OrderedDict):
        #     #     state_dict = checkpoint
        #     # elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        #     #     state_dict_old = checkpoint['state_dict']
        #     #     state_dict = OrderedDict()
        #     #     # delete 'module.' because it is saved from DataParallel module
        #     #     for key in state_dict_old.keys():
        #     #         if key.startswith('module.'):
        #     #             # state_dict[key[7:]] = state_dict[key]
        #     #             # state_dict.pop(key)
        #     #             state_dict[key[7:]] = state_dict_old[key]
        #     #         else:
        #     #             state_dict[key] = state_dict_old[key]
        #     # else:
        #     #     raise RuntimeError(
        #     #         'No state_dict found in checkpoint file {}'.format(pretrained))
        #     # self.load_state_dict(state_dict, strict=False)
        #     # logging.info('=> pretrained model loaded')
        

        logging.info('Init deconv weights from normal distribution')
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                logging.info('Init {}.weight as normal(0, 0.01)'.format(name))
                logging.info('Init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.01)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                logging.info('Init {}.weight as 1'.format(name))
                logging.info('Init {}.bias as 0'.format(name))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        logging.info('Init final conv weights from normal distribution')
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                logging.info('Init {}.weight as normal(0, 0.001)'.format(name))
                logging.info('Init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        logging.info('')
        logging.info('')
        logging.info('')


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    style = cfg.MODEL.STYLE

    block_class, layers = resnet_spec[num_layers]

    if style == 'caffe':
        block_class = Bottleneck_CAFFE

    model = PoseFix(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg, cfg.MODEL.PRETRAINED)

    logging.info(f"Model's Builded. (Num layers: {num_layers})")
    return model


def get_model_name(cfg):
    name = cfg.MODEL.NAME
    full_name = cfg.MODEL.NAME
    extra = cfg.MODEL.EXTRA
    if name in ['pose_resnet']:
        name = '{model}_{num_layers}'.format(
            model=name,
            num_layers=extra.NUM_LAYERS)
        deconv_suffix = ''.join(
            'd{}'.format(num_filters)
            for num_filters in extra.NUM_DECONV_FILTERS)
        full_name = '{height}x{width}_{name}_{deconv_suffix}'.format(
            height=cfg.MODEL.IMAGE_SIZE[1],
            width=cfg.MODEL.IMAGE_SIZE[0],
            name=name,
            deconv_suffix=deconv_suffix)
    else:
        raise ValueError('Unkown model: {}'.format(cfg.MODEL))

    return name, full_name


if __name__ == "__main__":
    from config import config
    from torchsummaryX import summary
    
    pose_net = get_pose_net(config, is_train=True)
    pose_net.eval()
    
    inps = torch.zeros(32, 3, 384, 288)
    hms  = torch.zeros(32, 17, 384, 288)
    # outs = pose_net(inps)
    # print("Output's shape:", outs.shape)
    
    summary(pose_net, inps, hms)
    
    