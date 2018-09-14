import torch
from torch import nn
import torch
import math


class MSDN(nn.Module):
    def __init__(self, block, layers, num_classes=1000, fc=1, ss=False, **kwargs):
        self.inplanes = 64
        super(MSDN, self).__init__()
        self.ss = ss
        self.fc_num = fc
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 64, layers[0])
        self.layer2 = self._make_layer(block, 64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 192, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, 768, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.layer2avgpool = nn.AdaptiveAvgPool2d(1)
        self.layer3avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1280 * block.expansion, 1280 * block.expansion)
        self.bnlast = nn.BatchNorm2d(1280 * block.expansion)
        self.relulast = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1280 * block.expansion, num_classes)
        self.bnl = nn.BatchNorm2d(1280 * block.expansion)
        self.dropout = nn.Dropout2d(0.5)
        self.layer2fc = nn.Linear(128 * block.expansion, num_classes)
        self.layer3fc = nn.Linear(256 * block.expansion, num_classes)

        self.downsamplelayer1 = DenseLayer(64, 64, stride=2)
        self.downsamplelayer2 = DenseLayer(192, 256, stride=2)
        self.downsamplelayer3 = DenseLayer(512, 512, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.softmax = nn.Softmax(num_classes)
        self.sigmoid = nn.Sigmoid()
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1:# or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x_layer1down = self.downsamplelayer1(x)
        #print('x_layer1down.shape: {}, x.shape: {}'.format(x_layer1down.shape, x.shape))
        #print(self.layer2(x).shape)
        x_layer2 = torch.cat((x_layer1down, self.layer2(x)), dim=1)
        #print('x_layer2.shape: ', x_layer2.shape)
        x_layer2down = self.downsamplelayer2(x_layer2)
        #print('x_layer2down.shape: ',x_layer2down.shape)

        #print(self.layer3(x_layer2).shape)
        x_layer3 = torch.cat((x_layer2down, self.layer3(x_layer2)), dim=1)
        #print('x_layer3.shape: ', x_layer3.shape)
        x_layer3down = self.downsamplelayer3(x_layer3)
        #print('x_layer3down.shape: ', x_layer3down.shape)
        #print(self.layer4(x_layer3).shape)
        x_layer4 = torch.cat((x_layer3down, self.layer4(x_layer3)), dim=1)
        #print('x_layer4.shape: ', x_layer4.shape)

        x_avg = self.avgpool(x_layer4)
        #x_view = x_avg.view(x_avg.size(0), -1)
        x = self.fc1(x_avg)
        print(x.shape)
        #if self.fc_num == 2:
        x = self.bnlast(x)
        x = self.relulast(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bnl(x)

        x = self.softmax(x)
        if self.ss == True:
            x_layer2_ss_pool = self.layer2avgpool(x_layer2)
            x_layer2_ss = x_layer2_ss_pool.view(x_layer2_ss_pool.size(0), -1)
            x_layer2_ss = self.layer2fc(x_layer2_ss)

            x_layer3_ss = self.layer3avgpool(x_layer3)
            x_layer3_ss = x_layer3_ss.view(x_layer3_ss.size(0), -1)
            x_layer3_ss = self.layer3fc(x_layer3_ss)
            return x_layer2_ss, x_layer3_ss, x
        else:
            return x




def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


class DenseLayer(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(DenseLayer, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


def msdn18(num_class,ss, dense_type='sum', **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if dense_type == 'sum':
        model = MSDN(BasicBlock, [2, 2, 2, 2], num_classes=num_class, ss=ss, **kwargs)
    return model


def msdn34(num_class,ss, dense_type='sum', **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if dense_type == 'sum':
        model = MSDN(BasicBlock, [3, 4, 6, 3], num_classes=num_class, ss=ss, **kwargs)
    return model