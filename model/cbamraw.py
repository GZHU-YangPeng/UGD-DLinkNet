# cbamraw.py - 原版 CBAM 接口对齐改进版

import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class channel_attention(nn.Module):
    def __init__(self, in_channel, ratio=16, pool_types=['avg', 'max']):
        super(channel_attention, self).__init__()
        self.pool_types = pool_types
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(in_channel, in_channel // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // ratio, in_channel)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        channel_att_sum = None
        b, c, h, w = x.size()

        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (h, w))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (h, w))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (h, w))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = self.logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum += channel_att_raw

        scale = self.sigmoid(channel_att_sum).view(b, c, 1, 1).expand_as(x)
        return x * scale

    @staticmethod
    def logsumexp_2d(tensor):
        tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
        s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
        outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
        return outputs


class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = self.sigmoid(x_out)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat([
            torch.max(x, dim=1, keepdim=True)[0],
            torch.mean(x, dim=1, keepdim=True)
        ], dim=1)


class cbam(nn.Module):
    def __init__(self, in_channel, ratio=16, kernel_size=7, pool_types=['avg', 'max'], no_spatial=False):
        super(cbam, self).__init__()
        # 通道注意力模块
        self.channel_attention = channel_attention(in_channel=in_channel,
                                                    ratio=ratio,
                                                    pool_types=pool_types)
        # 是否跳过空间注意力
        self.no_spatial = no_spatial
        if not no_spatial:
            self.spatial_attention = spatial_attention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        if not self.no_spatial:
            x = self.spatial_attention(x)
        return x
    
    
    
    # ---------------------------------------------------- #
# （2）自注意力机制
class self_attention(nn.Module):
    # 初始化，卷积核大小为7*7
    def __init__(self, in_channal = 512, kernel_size=1):
        # 继承父类初始化方法
        super(self_attention, self).__init__()

        # # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        # padding = kernel_size // 2
        # 7*7卷积融合通道信息 [b,in_c,h,w]==>[b,1,h,w]
        self.conv_k = nn.Conv2d(in_channels=in_channal, out_channels=1, kernel_size=kernel_size,
                                 bias=True)
        self.conv_q = nn.Conv2d(in_channels=in_channal, out_channels=1, kernel_size=kernel_size,
                                 bias=True)
        self.conv_v = nn.Conv2d(in_channels=in_channal, out_channels=1, kernel_size=kernel_size,
                                 bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.conv = nn.Conv2d(in_channels=1, out_channels=in_channal, kernel_size=kernel_size,
                                 bias=True)
        # self.conv = nn.Sequential(
        #                 nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
        #                       padding=padding, bias=False),
        #                 nn.BatchNorm2d(1),
        #                 nn.Sigmoid(),
        #                   )
        # sigmoid函数
        # self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        k = self.conv_k(inputs)
        q = self.conv_q(inputs)
        v = self.conv_v(inputs)
        k_q = self.softmax(k * q)
        outputs =self.conv(k_q * v)

        return outputs