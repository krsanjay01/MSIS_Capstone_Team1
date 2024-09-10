from basic_layers import *
from torch import nn
from trans_unet.vit_seg_modeling import Transformer  # Importing transformer and encoder from vit_seg_modeling.py


class UnetWithTransformer(nn.Module):
    def __init__(self, device, config=None, inp_ch=1, out_ch=1, arch=16, depth=3, activ='leak', concat=None, vis=False):
        super(UnetWithTransformer, self).__init__()

        self.activ = activ
        self.device = device
        self.out_ch = out_ch
        self.inp_ch = inp_ch
        self.depth = depth
        self.arch = arch
        self.concat = None
        self.vis = vis

        self.arch_n = []
        self.dec = []  # Decoder layers
        self.layers = []
        self.skip = []

        # Use the Transformer as encoder
        self.encoder = Transformer(config, img_size=224, vis=vis)

        # Throttling layer: Converts transformer output to match decoder input dimensions
        self.throttling_layer = nn.Sequential(
            nn.Conv2d(config.hidden_size, arch, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # Initialize the decoder and final layer as in the original model
        self.check_concat(concat)
        self.prep_arch_list()
        self.organize_arch()
        self.prep_params()

    def check_concat(self, con):
        if con is None:
            self.concat = [1] * self.depth
        elif len(con) > self.depth:
            self.concat = con[:self.depth]
            self.concat = 2 * con
            self.concat[self.concat == 0] = 1
        elif len(con) < self.depth:
            self.concat = con + [0] * (self.depth - len(con))
            self.concat = 2 * con
            self.concat[self.concat == 0] = 1
        else:
            self.concat = 2 * con
            self.concat[self.concat == 0] = 1

    def prep_arch_list(self):
        for dl in range(0, self.depth + 1):
            self.arch_n.append((2 ** (dl - 1)) * self.arch)

        self.arch_n[0] = self.inp_ch

    def organize_arch(self):
        self.layers = [Conv_Block(self.arch_n[-1], self.arch_n[-1], activ=self.activ, pool='up_stride')]

        for idx in range(len(self.arch_n) - 2):
            self.dec.append(
                Conv_Block(self.concat[- (idx + 1)] * self.arch_n[- (idx + 1)], self.arch_n[- (idx + 2)],
                           activ=self.activ, pool='up_stride'))
        self.dec.append(Conv_Block(self.concat[0] * self.arch, self.arch, activ=self.activ))
        self.layers.append(Conv_Layer(self.arch, self.out_ch, 1, 1, norm=None, activ='tanh'))

    def prep_params(self):
        for blk_idx in range(len(self.dec)):
            self.add_module(f'dec_{blk_idx + 1}', self.dec[blk_idx])

        self.add_module(f'mid', self.layers[0])
        self.add_module(f'final', self.layers[1])

    def forward(self, img):
        # Transformer encoder step
        x, attn_weights, features = self.encoder(img)

        # Throttling layer
        x = self.throttling_layer(x)

        h_skip = []
        for conv in self.enc:
            hs, x = conv(x)
            h_skip.append(hs)

        _, x = self.mid(x)

        for l_idx in range(len(self.dec)):
            if self.concat[-(l_idx + 1)] == 2:
                _, x = self.dec[l_idx](concat_curr(h_skip[-(l_idx + 1)], x))
            else:
                _, x = self.dec[l_idx](x)

        x = self.final(x)
        return x


# ----------------Test---------------------------

if __name__ == '__main__':
    import torch

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    x = torch.randn(1, 1, 128, 128).to(device)

    net = UnetWithTransformer(device).to(device)
    y = net(x)
    print(y.shape)
