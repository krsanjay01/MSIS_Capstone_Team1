import torch
import torch.nn as nn
import torch.nn.functional as F


# SelfAttention Module
class SelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8, dropout=0.0):
        super(SelfAttention, self).__init__()
        self.in_dim = in_dim
        self.heads = heads

        self.query = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.key = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.value = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.scale = (in_dim // heads) ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm([in_dim, 1, 1])

    def forward(self, x):
        b, c, h, w = x.size()

        q = self.query(x).view(b, self.heads, c // self.heads, h * w)
        k = self.key(x).view(b, self.heads, c // self.heads, h * w)
        v = self.value(x).view(b, self.heads, c // self.heads, h * w)

        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v.transpose(-2, -1))
        out = out.view(b, c, h, w)
        out = self.proj(out)
        out = self.norm(out + x)

        return out


# Conv_Layer remains unchanged
class Conv_Layer(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride,
                 padding=0, dilation=1, bias=True, activ=None, norm=None,
                 pool=None):
        super(Conv_Layer, self).__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv', nn.Conv2d(in_c, out_c, kernel_size=kernel,
                                               stride=stride, dilation=dilation, padding=padding, bias=bias))

        if activ == 'leak':
            activ = nn.LeakyReLU(inplace=True)
        elif activ == 'relu':
            activ = nn.ReLU(inplace=True)
        elif activ == 'pleak':
            activ = nn.PReLU()
        elif activ == 'gelu':
            activ = nn.GELU()
        elif activ == 'selu':
            activ = nn.SELU()
        elif activ == 'sigmoid':
            activ = nn.Sigmoid()
        elif activ == 'softmax':
            activ = nn.Softmax(dim=1)
        elif activ == 'tanh':
            activ = nn.Tanh()
        else:
            activ = None

        if norm == 'bn':
            norm = nn.BatchNorm2d(out_c)
        elif norm == 'ln':
            norm = nn.LayerNorm([out_c, 1, 1])
        else:
            norm = None

        if pool == 'max':
            pool = nn.MaxPool2d(2, 2)
        elif pool == 'avg':
            pool = nn.AvgPool2d(2, 2)
        else:
            pool = None

        if norm is not None:
            self.conv.add_module('norm', norm)

        if activ is not None:
            self.conv.add_module('activ', activ)

        if pool is not None:
            self.conv.add_module('pool', pool)

    def forward(self, x):
        x = self.conv(x)
        return x


# DeConv_Layer remains unchanged
class DeConv_Layer(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride,
                 padding=0, activ=None, norm=None,
                 pool=None, bias=True):
        super(DeConv_Layer, self).__init__()
        self.deconv = nn.Sequential()
        self.deconv.add_module('deconv', nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel,
                                                            stride=stride, padding=padding, bias=bias))

        if activ == 'leak':
            activ = nn.LeakyReLU(inplace=True)
        elif activ == 'relu':
            activ = nn.ReLU(inplace=True)
        elif activ == 'pleak':
            activ = nn.PReLU()
        elif activ == 'gelu':
            activ = nn.GELU()
        elif activ == 'selu':
            activ = nn.SELU()
        elif activ == 'sigmoid':
            activ = nn.Sigmoid()
        elif activ == 'softmax':
            activ = nn.Softmax(dim=1)
        else:
            activ = None

        if norm == 'bn':
            norm = nn.BatchNorm2d(out_c)
        elif norm == 'ln':
            norm = nn.LayerNorm([out_c, 1, 1])
        else:
            norm = None

        if pool == 'max':
            pool = nn.MaxPool2d(2, 2)
        elif pool == 'avg':
            pool = nn.AvgPool2d(2, 2)
        else:
            pool = None

        if norm is not None:
            self.deconv.add_module('norm', norm)

        if activ is not None:
            self.deconv.add_module('activ', activ)

        if pool is not None:
            self.deconv.add_module('pool', pool)

    def forward(self, x):
        x = self.deconv(x)
        return x


# Updated Conv_Block with optional Self-Attention
class Conv_Block(nn.Module):
    def __init__(self, in_c, out_c, activ=None, pool=None, norm='bn', use_transformer=False, embed_size=256, num_heads=8, num_layers=6):
        super(Conv_Block, self).__init__()
        self.c1 = Conv_Layer(in_c, out_c, 3, 1, activ=activ, norm=norm, padding=1)
        self.c2 = Conv_Layer(out_c, out_c, 3, 1, activ=activ, norm=norm, padding=1)

        # Transformer Block (optional)
        self.use_transformer = use_transformer
        if use_transformer:
            self.transformer = SelfAttention(embed_size=embed_size, num_heads=num_heads, num_layers=num_layers)

        if pool == 'up_stride':
            self.pool = DeConv_Layer(out_c, out_c, 2, 2, norm=norm)
        elif pool == 'up_bilinear':
            self.pool = nn.Upsample(scale_factor=2, mode=pool[3:], align_corners=True)
        elif pool == 'up_nearest':
            self.pool = nn.Upsample(scale_factor=2, mode=pool[3:], align_corners=True)
        elif pool == 'down_max':
            self.pool = nn.MaxPool2d(2, 2)
        elif pool == 'down_stride':
            self.c2 = Conv_Layer(out_c, out_c, 3, 2, activ=activ, norm=norm, padding=1)
            self.pool = None
        else:
            self.pool = None

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)

        if self.use_transformer:
            x = self.transformer(x)

        if self.pool:
            return x, self.pool(x)
        else:
            return x, None  # Always return a tuple



# Utility function remains unchanged
def concat_curr(prev, curr):
    diffY = prev.size()[2] - curr.size()[2]
    diffX = prev.size()[3] - curr.size()[3]

    curr = F.pad(curr, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
    x = torch.cat([prev, curr], dim=1)
    return x
