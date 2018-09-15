import torch
from torch import nn
import torch
import math
import torch.nn.functional as F
from collections import OrderedDict


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

"""
class MSDNet(nn.Module):
    def __init__(self):
        super(MSDNet, self).__init__()

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0)

    def forward(self, x):


        return x

"""
class FirstScaleLayer(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24),
                 num_init_features=64, bn_size=4, drop_rate=0,num_classes=1000, **kwargs):

        super(FirstScaleLayer, self).__init__()

        # First convolution
        self.conv0 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=2, padding=1, bias=False))
        ]))
        self.scales = nn.ModuleList()


        num_features = num_init_features
        loop = 0
        for i, num_layers in enumerate(block_config):
            print('block: {} have {} layers with num_input_features: {} output_features: {}'.
                  format(i, num_layers, num_features, growth_rate * (2**(i+1))))
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.scales.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            loop += 1
            print('trans num_features: {}'.format(num_features))
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.scales.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
                loop+=1

        # Final batch norm
        #self.scales.add_module('norm5', nn.BatchNorm2d(num_features))

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.classes = num_classes
        self.adaptivepool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax()
        self.relulast = nn.ReLU(inplace=True)
        self.classifier1 = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.conv0(x)
        print(self.scales)
        for i, layer in enumerate(self.scales):
            print('{}: {}'.format(i, layer))
            x = layer(x)
            if i == 0:
                x_dense1 = x
            elif i == 2:
                x_dense2 = x
            elif i == 3:
                x_dense3 = x
        """
        x_dense1 = self.scales[1](x_conv0)
        x_trans1 = self.scales[1](x_dense1)
        x_dense2 = self.scales[2](x_trans1)
        x_trans2 = self.scales[3](x_dense2)
        x_dense3 = self.scales[4](x_trans2)
"""
        out = F.relu(x, inplace=True)
        out = self.adaptivepool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier1(out)
        return out


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            print('{}th layer at Denseblock has {} input_features'.format(i, num_input_features))
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

        self.add_module('norm2', nn.BatchNorm2d(num_output_features))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(num_output_features, num_output_features,
                                          kernel_size=3, stride=2, padding=3, bias=False))


def msdn18(num_class, drop_rate, pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model =FirstScaleLayer(num_init_features=32, growth_rate=32, block_config=(4, 8, 6), num_classes=num_class, drop_rate=drop_rate,
                     **kwargs)
    return model