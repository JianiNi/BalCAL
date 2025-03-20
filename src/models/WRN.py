# Code from https://github.com/deepmind/deepmind-research/blob/master/adversarial_robustness/pytorch/model_zoo.py
# (Gowal et al 2020)

from typing import Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
SVHN_MEAN = (0.5, 0.5, 0.5)
SVHN_STD = (0.5, 0.5, 0.5)

_ACTIVATION = {
    'relu': nn.ReLU,
    'swish': nn.SiLU,
}

    
class _Block(nn.Module):
    """
    WideResNet Block.
    Arguments:
        in_planes (int): number of input planes.
        out_planes (int): number of output filters.
        stride (int): stride of convolution.
        activation_fn (nn.Module): activation function.
    """
    def __init__(self, in_planes, out_planes, stride, activation_fn=nn.ReLU):
        super().__init__()
        self.batchnorm_0 = nn.BatchNorm2d(in_planes, momentum=0.01)
        self.relu_0 = activation_fn(inplace=True)
        # We manually pad to obtain the same effect as `SAME` (necessary when `stride` is different than 1).
        self.conv_0 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                padding=0, bias=False)
        self.batchnorm_1 = nn.BatchNorm2d(out_planes, momentum=0.01)
        self.relu_1 = activation_fn(inplace=True)
        self.conv_1 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.has_shortcut = in_planes != out_planes
        if self.has_shortcut:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, 
                                      stride=stride, padding=0, bias=False)
        else:
            self.shortcut = None
        self._stride = stride

    def forward(self, x):
        if self.has_shortcut:
            x = self.relu_0(self.batchnorm_0(x))
        else:
            out = self.relu_0(self.batchnorm_0(x))
        v = x if self.has_shortcut else out
        if self._stride == 1:
            v = F.pad(v, (1, 1, 1, 1))
        elif self._stride == 2:
            v = F.pad(v, (0, 1, 0, 1))
        else:
            raise ValueError('Unsupported `stride`.')
        out = self.conv_0(v)
        out = self.relu_1(self.batchnorm_1(out))
        out = self.conv_1(out)
        out = torch.add(self.shortcut(x) if self.has_shortcut else x, out)
        return out


class _BlockGroup(nn.Module):
    """
    WideResNet block group.
    Arguments:
        in_planes (int): number of input planes.
        out_planes (int): number of output filters.
        stride (int): stride of convolution.
        activation_fn (nn.Module): activation function.
    """
    def __init__(self, num_blocks, in_planes, out_planes, stride, activation_fn=nn.ReLU):
        super().__init__()
        block = []
        for i in range(num_blocks):
            block.append(
                _Block(i == 0 and in_planes or out_planes, 
                       out_planes,
                       i == 0 and stride or 1,
                       activation_fn=activation_fn)
            )
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class WideResNet(nn.Module):
    """
    WideResNet model
    Arguments:
        num_classes (int): number of output classes.
        depth (int): number of layers.
        width (int): width factor.
        activation_fn (nn.Module): activation function.
        mean (tuple): mean of dataset.
        std (tuple): standard deviation of dataset.
        padding (int): padding.
        num_input_channels (int): number of channels in the input.
    """
    def __init__(self,
                 num_classes: int = 10,
                 depth: int = 28,
                 width: int = 10,
                 activation_fn: nn.Module = nn.ReLU,
                 mean: Union[Tuple[float, ...], float] = CIFAR10_MEAN,
                 std: Union[Tuple[float, ...], float] = CIFAR10_STD,
                 padding: int = 0,
                 num_input_channels: int = 3):
        super().__init__()
        self.padding = padding
        num_channels = [16, 16 * width, 32 * width, 64 * width]
        assert (depth - 4) % 6 == 0
        num_blocks = (depth - 4) // 6
        self.num_input_channels = num_input_channels
        self.return_z = False
        self.init_conv = nn.Conv2d(num_input_channels, num_channels[0],
                                   kernel_size=3, stride=1, padding=1, bias=False)
        self.layer = nn.Sequential(
            _BlockGroup(num_blocks, num_channels[0], num_channels[1], 1,
                        activation_fn=activation_fn),
            _BlockGroup(num_blocks, num_channels[1], num_channels[2], 2,
                        activation_fn=activation_fn),
            _BlockGroup(num_blocks, num_channels[2], num_channels[3], 2,
                        activation_fn=activation_fn))
        self.batchnorm = nn.BatchNorm2d(num_channels[3], momentum=0.01)
        self.relu = activation_fn(inplace=True)
        self.logits = nn.Linear(num_channels[3], num_classes)
        self.num_channels = num_channels[3]
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def forward(self, x):
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        if not hasattr(self, 'num_input_channels'):
            self.num_input_channels=3
        if not hasattr(self, 'return_z'):
            self.return_z = False
        out = self.init_conv(x)
        out = self.layer(out)
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 8 if self.num_input_channels == 3 else 7)
        out = out.view(-1, self.num_channels)
        if self.return_z:
            return self.logits(out), out
        else:
            return self.logits(out)


class Normalization(nn.Module):
    """
    Standardizes the input data.
    Arguments:
        mean (list): mean.
        std (float): standard deviation.
        device (str or torch.device): device to be used.
    Returns:
        (input - mean) / std
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        num_channels = len(mean)
        self.mean = torch.FloatTensor(mean).view(1, num_channels, 1, 1)
        self.sigma = torch.FloatTensor(std).view(1, num_channels, 1, 1)
        self.mean_cuda, self.sigma_cuda = None, None

    def forward(self, x):
        if x.is_cuda:
            if self.mean_cuda is None:
                self.mean_cuda = self.mean.cuda()
                self.sigma_cuda = self.sigma.cuda()
            out = (x - self.mean_cuda) / self.sigma_cuda
        else:
            out = (x - self.mean) / self.sigma
        return out
    
class WideResNet_BalCAL(nn.Module):
    """
    WideResNet model
    Arguments:
        num_classes (int): number of output classes.
        depth (int): number of layers.
        width (int): width factor.
        activation_fn (nn.Module): activation function.
        mean (tuple): mean of dataset.
        std (tuple): standard deviation of dataset.
        padding (int): padding.
        num_input_channels (int): number of channels in the input.
    """
    def __init__(self,
                 num_classes: int = 10,
                 depth: int = 28,
                 width: int = 10,
                 lamd: float=0.8, 
                 activation_fn: nn.Module = nn.ReLU,
                 mean: Union[Tuple[float, ...], float] = CIFAR10_MEAN,
                 std: Union[Tuple[float, ...], float] = CIFAR10_STD,
                 padding: int = 0,
                 num_input_channels: int = 3):
        super().__init__()
        self.padding = padding
        num_channels = [16, 16 * width, 32 * width, 64 * width]
        assert (depth - 4) % 6 == 0
        num_blocks = (depth - 4) // 6
        self.num_input_channels = num_input_channels
        self.return_z = False
        self.init_conv = nn.Conv2d(num_input_channels, num_channels[0],
                                   kernel_size=3, stride=1, padding=1, bias=False)
        self.layer = nn.Sequential(
            _BlockGroup(num_blocks, num_channels[0], num_channels[1], 1,
                        activation_fn=activation_fn),
            _BlockGroup(num_blocks, num_channels[1], num_channels[2], 2,
                        activation_fn=activation_fn),
            _BlockGroup(num_blocks, num_channels[2], num_channels[3], 2,
                        activation_fn=activation_fn))
        self.batchnorm = nn.BatchNorm2d(num_channels[3], momentum=0.01)
        self.relu = activation_fn(inplace=True)
        self.num_channels = num_channels[3]
        
        P = self.generate_random_orthogonal_matrix(num_channels[3], num_classes)     
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = torch.sqrt(torch.tensor(num_classes / (num_classes-1))) * torch.matmul(P, I-((1/num_classes) * one))
        self.ori_M = M.cuda()

        self.logits = nn.Linear(num_channels[3],  num_classes)
        
        self.lamd = nn.Parameter(torch.tensor(lamd))

        self.adaptor = Adaptor(num_channels[3], num_channels[2], activation_fn=activation_fn)
        self.batchnorm_etf = nn.BatchNorm2d(num_channels[3], momentum=0.01)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        a = torch.rand(size=(feat_in, num_classes))
        P, _ = torch.linalg.qr(a)
        # P = torch.tensor(P).float()
        # assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-07), torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        return P
    
    def shared_parameters(self):
        return chain(
            self.init_conv.parameters(),
            self.layer.parameters(),
            self.batchnorm.parameters()
        )
    
    def linear_parameters(self):
        return self.logits.parameters()
    
    def ETF_parameters(self):
        return chain(
            self.BN_H.parameters(),
            self.adaptor.parameters(),
            self.batchnorm_etf.parameters()
        )

    def forward(self, x, return_all = False):
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        if not hasattr(self, 'num_input_channels'):
            self.num_input_channels=3
        if not hasattr(self, 'return_z'):
            self.return_z = False
        out = self.init_conv(x)
        out = self.layer(out)

        
        out_ETF = self.adaptor(out)
        out_ETF = self.relu(self.batchnorm_etf(out_ETF))
        out_ETF = F.avg_pool2d(out_ETF, 8 if self.num_input_channels == 3 else 7)
        out_ETF = out_ETF.view(-1, self.num_channels)
           
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 8 if self.num_input_channels == 3 else 7)
        out = out.view(-1, self.num_channels)
        y_linear = self.logits(out)

        
        out_norm = torch.norm(out_ETF, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feat_ETF = torch.div(out_ETF, out_norm)

        y_ETF = torch.matmul(feat_ETF, self.ori_M)
        y = self.lamd * y_linear + (1-self.lamd)* y_ETF
        if return_all:
            return y_linear, feat_ETF, y
        else:
            return y
        
        
class Adaptor(nn.Module):
    def __init__(self, in_planes, bottleneck_planes, activation_fn=nn.ReLU):
        super().__init__()
        self.batchnorm_0 = nn.BatchNorm2d(in_planes, momentum=0.01)
        self.relu_0 = activation_fn(inplace=True)
        self.conv_0 = nn.Conv2d(in_planes, bottleneck_planes, kernel_size=3, stride=1,
                                padding=0, bias=False)
        self.batchnorm_1 = nn.BatchNorm2d(bottleneck_planes, momentum=0.01)
        self.relu_1 = activation_fn(inplace=True)
        self.conv_1 = nn.Conv2d(bottleneck_planes, in_planes, kernel_size=3, stride=1,
                                padding=1, bias=False)
        self._stride = 1

    def forward(self, x):
        out = self.relu_0(self.batchnorm_0(x))
        if self._stride == 1:
            out = F.pad(out, (1, 1, 1, 1))
        elif self._stride == 2:
            out = F.pad(out, (0, 1, 0, 1))
        else:
            raise ValueError('Unsupported `stride`.')
        out = self.conv_0(out)
        out = self.relu_1(self.batchnorm_1(out))
        out = self.conv_1(out)
        return x + out


