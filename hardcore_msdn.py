import torch
from torch import nn
import torch
import math
import torch.nn.functional as F
from collections import OrderedDict
from config import Config


class MSDNet(nn.Module):
    def __init__(self, depth, num_init_features, growth_rate, num_classes,
                 drop_rate, block_config=(4, 8, 6), **kwargs):
        super(MSDNet, self).__init__()
        self.net = nn.ModuleList()
        self.first_layer = FirstScaleLayer(num_init_features=32, growth_rate=32,
                                           block_config=(4, 8, 6), num_classes=num_classes,
                                           drop_rate=drop_rate, **kwargs)
        self.classifier = _Classifier(32, 1024, num_classes)
        self.net.add_module('Scale_Layer', self.first_layer)
        self.depth_module = nn.ModuleList()
        self.scaledown = nn.ModuleList()
        self.scale_size = block_config.size()
        num_features = num_init_features

        self.depth = depth
        self.depth1_scale1_para = _ResidualLayer(32, 32)
        self.depth2_scale1_para = _ResidualLayer(64, 64)

        self.depth1_scale1_bottle = _BottleNeck(160, 80)
        self.depth1_scale2_bottle = _BottleNeck(336, 168)
        self.depth1_scale3_bottle = _BottleNeck(336, 168)

        self.depth2_scale1_cat = _TwoResidualLayer(80, 128)
        self.depth1_scale2_cat = _TwoResidualLayer(168, 336)


    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_layer(x)
        x_dense1, x_dense2, x_dense3 = x[0], x[1], x[2]

        print('x_dense1: {}, x_dense2: {}, x_dense3: {}'.format(x_dense1.shape, x_dense2.shape, x_dense3.shape))
        x_depth1_scale1_bottle = self.depth1_scale1_bottle(x_dense1)
        x_depth1_scale2_bottle = self.depth1_scale2_bottle(x_dense2)
        x_depth1_scale3_bottle = self.depth1_scale3_bottle(x_dense3)
        print('Bottle: x_depth1_scale1 {}, x_depth1_scale2 {}, x_depth1_scale3 {}'.format(x_depth1_scale1_bottle.shape,
                                                                                          x_depth1_scale2_bottle.shape,
                                                                                          x_depth1_scale3_bottle.shape))



        x_depth1_scale1_parallel = self.depth1_scale1_para(x_dense1)
        x_depth2_scale1_parallel = self.depth2_scale1_para(x_dense1)
        print('x_depth1_scale1_paralle: {} and x_depth2_scale1_parallel: {}'.format(x_depth1_scale1_parallel, x_depth2_scale1_parallel))



        x_depth1_scale2_cat = self.depth1_scale1_cat([x_depth1_scale1_bottle, x_dense2])
        print('x_depth1_scale2_cat: {}'.format(x_depth1_scale2_cat.shape))



        print('x_depth1_scale1 bottle: {}, x_depth1_scale1')
        #x_dense2_scale1
        x_depth2_scale1 = self.depth2_scale1(x_dense2)

        x_depth1_scale2 = self.depth1_scale2
        #x_depth1_scale3

        #x_depth2_scale3 =
        out = self.classifier(x)#(x_depth2_scale3)

        return out


class _Classifier(nn.Module):
    def __init__(self, input_feature, linear_features, classes):
        super(_Classifier, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('norm', nn.BatchNorm2d(input_feature))
        self.feature.add_module('relu', nn.ReLU(inplace=True))
        self.feature.add_module('avgppol', nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(linear_features, classes)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


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
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, **kwargs):

        super(FirstScaleLayer, self).__init__()

        # First convolution
        self.conv0 = nn.Sequential(OrderedDict([
            ('scale1_conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=2, padding=1, bias=False))
        ]))
        self.scales = nn.ModuleList()

        num_features = num_init_features
        loop = 0
        for i, num_layers in enumerate(block_config):
            # print('block: {} have {} layers with num_input_features: {} output_features: {}'.
            # format(i, num_layers, num_features, growth_rate * (2**(i+1))))
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.scales.add_module('scale1_denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            loop += 1
            # print('trans num_features: {}'.format(num_features))
            if i != len(block_config) - 1:
                bottle = _BottleNeck(num_features, num_features // 2)
                trans = _Transition(num_features // 2)
                self.scales.add_module('scale1_bottleneck%d' % (i + 1), bottle)
                self.scales.add_module('scale1_transition%d' % (i + 1), trans)
                num_features = num_features // 2
                loop += 1

        # Final batch norm
        # self.scales.add_module('norm5', nn.BatchNorm2d(num_features))

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

    def forward(self, x):
        output = []
        x = self.conv0(x)
        # print(self.scales)
        for i, layer in enumerate(self.scales):
            # print('{}: {}'.format(i, layer))
            x = layer(x)
            if i % 3 == 0:
                output.append(x)
        return output


class _ResidualLayer(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_ResidualLayer, self).__init__()
        self.conv = nn.Sequential(
            ('norm1', nn.BatchNorm2d(num_input_features)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(num_input_features, num_output_features,
                                kernel_size=3, stride=1, padding=1, bias=False))
        )

    def forward(self, x):
        x_conv = self.conv(x)
        if Config.debug:
            print('shape for residual layer: {} and {}'.format(x.shape, x_conv.shape))
        x = torch.cat([x, x_conv], dim=1)
        return x


class _TwoResidualLayer(nn.Module):
    def __init__(self, up_features, cur_features):
        super(_TwoResidualLayer, self).__init__()
        self.sample_down = nn.Sequential(
            ('norm1', nn.BatchNorm2d(up_features)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(up_features, up_features, kernel_size=3, stride=2, padding=2, bias=False))
        )
        self.current_conv = nn.Sequential(
            ('norm1', nn.BatchNorm2d(cur_features)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(cur_features, cur_features, kernel_size=3, stride=1, padding=1, bias=False))
        )

    def forward(self, *input):
        up_feature, current_feature = input[0], input[1]
        up_feature = self.sample_down(up_feature)
        current_feature_conv = self.current_conv(current_feature)
        if Config.debug:
            print('shape for two residual layer: {} and {}'.format(up_feature.shape, current_feature_conv.shape))
        x_cat = torch.cat([up_feature, current_feature_conv], dim=1)
        x_cat = torch.cat([x_cat, current_feature])
        return x_cat


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
            # print('{}th layer at Denseblock has {} input_features'.format(i, num_input_features))
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _BottleNeck(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_BottleNeck, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, num_output_features,
                                           kernel_size=1, stride=1, bias=False))


class _Transition(nn.Sequential):
    def __init__(self, num_input_features):
        super(_Transition, self).__init__()
        self.add_module('norm2', nn.BatchNorm2d(num_input_features))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(num_input_features,num_input_features,
                                           kernel_size=3, stride=2, padding=3, bias=False))


def msdn18(num_class, drop_rate, pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FirstScaleLayer(num_init_features=32, growth_rate=32, block_config=(4, 8, 6), num_classes=num_class,
                            drop_rate=drop_rate,
                            **kwargs)
    return model