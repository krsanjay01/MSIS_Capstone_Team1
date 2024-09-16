import torch
import torch.nn as nn
import torch.nn.functional as F

# Conv_Layer class with device parameter
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

        # Initialize normalization
        self.norm_layer = None
        if norm == 'bn':
            self.norm_layer = nn.BatchNorm2d(out_c)

        if pool == 'max':
            pool = nn.MaxPool2d(2, 2)
        elif pool == 'avg':
            pool = nn.AvgPool2d(2, 2)

        # Add layers to the sequential module
        if self.norm_layer is not None:
            self.conv.add_module('norm', self.norm_layer)

        if not pool is None:
            self.conv.add_module('pool', pool)

        if not activ is None:
            self.conv.add_module('activ', activ)

    def forward(self, x):
        x = x.to(self.device)  # Ensure x is on the correct device
        x = self.conv(x)

        return x


# DeConv_Layer with device parameter
class DeConv_Layer(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, padding=0,
                 activ=None, norm=None, bias=True):
        super(DeConv_Layer, self).__init__()
        '''self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel,
                                         stride=stride, padding=padding, bias=bias).to(self.device)'''

        self.deconv = nn.Sequential()
        self.deconv.add_module('deconv', nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel,
                                                            stride=stride, padding=padding, bias=bias))
        layers = [self.deconv]

        if norm == 'bn':
            layers.append(nn.BatchNorm2d(out_c).to(self.device))
        elif norm == 'ln':
            layers.append(nn.LayerNorm([out_c, 1, 1]).to(self.device))

        if activ == 'leak':
            layers.append(nn.LeakyReLU(inplace=True).to(self.device))
        elif activ == 'relu':
            layers.append(nn.ReLU(inplace=True).to(self.device))
        elif activ == 'pleak':
            layers.append(nn.PReLU().to(self.device))
        elif activ == 'gelu':
            layers.append(nn.GELU().to(self.device))
        elif activ == 'selu':
            layers.append(nn.SELU().to(self.device))
        elif activ == 'sigmoid':
            layers.append(nn.Sigmoid().to(self.device))
        elif activ == 'softmax':
            layers.append(nn.Softmax(dim=1).to(self.device))
        elif activ == 'tanh':
            layers.append(nn.Tanh().to(self.device))

        self.deconv = nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(self.device)
        return self.deconv(x)


# Conv_Block with device parameter (no transformer option)
class Conv_Block(nn.Module):
    def __init__(self, in_c, out_c, activ=None, pool=None, norm='bn', device='cpu'):
        super(Conv_Block, self).__init__()
        self.device = device

        # Add conditional logic to skip batch norm for 1x1 spatial dimensions
        self.c1 = Conv_Layer(in_c, out_c, 3, 1, activ=activ, norm=norm if norm != 'bn' or out_c > 1 else None,
                             padding=1)
        self.c2 = Conv_Layer(out_c, out_c, 3, 1, activ=activ, norm=norm if norm != 'bn' or out_c > 1 else None,
                             padding=1)

        if pool == 'up_stride':
            self.pool = DeConv_Layer(out_c, out_c, 2, 2, norm=norm, device=self.device)
        elif pool == 'up_bilinear':
            self.pool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True).to(self.device)
        elif pool == 'up_nearest':
            self.pool = nn.Upsample(scale_factor=2, mode='nearest').to(self.device)
        elif pool == 'down_max':
            self.pool = nn.MaxPool2d(2, 2).to(self.device)
        elif pool == 'down_stride':
            self.c2 = Conv_Layer(out_c, out_c, 3, 2, activ=activ, norm=norm, padding=1)
            self.pool = None
        else:
            self.pool = None

    def forward(self, x):
        # Apply conv layers normally but conditionally apply batch norm if spatial dims > 1x1
        x = x.to(self.device)  # Ensure x is on the correct device
        x = self.c1(x)
        if x.size(2) > 1 and x.size(3) > 1:
            x = self.c2(x)  # Apply second conv and norm only if spatial dims > 1x1

        if self.pool:
            return x, self.pool(x)
        else:
            return 0, x


# Utility function remains unchanged
def concat_curr(prev, curr):
    diffY = prev.size()[2] - curr.size()[2]
    diffX = prev.size()[3] - curr.size()[3]

    curr = F.pad(curr, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
    return torch.cat([prev, curr], dim=1)