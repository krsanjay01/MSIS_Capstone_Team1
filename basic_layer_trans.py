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
        self.norm = nn.BatchNorm2d(in_dim)  # Use BatchNorm2d instead of LayerNorm

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



# Conv_Layer remains unchanged except for padding placement
class Conv_Layer(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, padding=0,
                 dilation=1, bias=True, activ=None, norm=None):
        super(Conv_Layer, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel,
                              stride=stride, dilation=dilation, padding=padding, bias=bias)

        layers = [self.conv]

        if norm == 'bn':
            layers.append(nn.BatchNorm2d(out_c))
        elif norm == 'ln':
            layers.append(nn.LayerNorm([out_c, 1, 1]))

        if activ == 'leak':
            layers.append(nn.LeakyReLU(inplace=True))
        elif activ == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activ == 'pleak':
            layers.append(nn.PReLU())
        elif activ == 'gelu':
            layers.append(nn.GELU())
        elif activ == 'selu':
            layers.append(nn.SELU())
        elif activ == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activ == 'softmax':
            layers.append(nn.Softmax(dim=1))
        elif activ == 'tanh':
            layers.append(nn.Tanh())

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


# DeConv_Layer remains unchanged except for padding placement
class DeConv_Layer(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, padding=0,
                 activ=None, norm=None, bias=True):
        super(DeConv_Layer, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel,
                                         stride=stride, padding=padding, bias=bias)

        layers = [self.deconv]

        if norm == 'bn':
            layers.append(nn.BatchNorm2d(out_c))
        elif norm == 'ln':
            layers.append(nn.LayerNorm([out_c, 1, 1]))

        if activ == 'leak':
            layers.append(nn.LeakyReLU(inplace=True))
        elif activ == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activ == 'pleak':
            layers.append(nn.PReLU())
        elif activ == 'gelu':
            layers.append(nn.GELU())
        elif activ == 'selu':
            layers.append(nn.SELU())
        elif activ == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activ == 'softmax':
            layers.append(nn.Softmax(dim=1))
        elif activ == 'tanh':
            layers.append(nn.Tanh())

        self.deconv = nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv(x)


# Updated Conv_Block with optional Self-Attention
class Conv_Block(nn.Module):
    def __init__(self, in_c, out_c, activ=None, pool=None, norm='bn', use_transformer=False, embed_size=256, num_heads=8):
        super(Conv_Block, self).__init__()
        self.c1 = Conv_Layer(in_c, out_c, 3, 1, activ=activ, norm=norm, padding=1)
        self.c2 = Conv_Layer(out_c, out_c, 3, 1, activ=activ, norm=norm, padding=1)

        # Transformer Block (optional) xyz
        self.use_transformer = use_transformer
        if use_transformer:
            self.transformer = SelfAttention(in_dim=out_c, heads=num_heads, dropout=0.1)

        if pool == 'up_stride':
            self.pool = DeConv_Layer(out_c, out_c, 2, 2, norm=norm)
        elif pool == 'up_bilinear':
            self.pool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif pool == 'up_nearest':
            self.pool = nn.Upsample(scale_factor=2, mode='nearest')
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
            return x, x  # Ensure two outputs are always returned


# Utility function remains unchanged
def concat_curr(prev, curr):
    diffY = prev.size()[2] - curr.size()[2]
    diffX = prev.size()[3] - curr.size()[3]

    curr = F.pad(curr, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
    return torch.cat([prev, curr], dim=1)
