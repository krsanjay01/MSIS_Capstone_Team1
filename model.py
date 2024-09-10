from basic_layers import *
from torch import nn
from trans_unet.vit_seg_modeling import Transformer
from trans_unet.vit_seg_configs import get_b16_config


class Unet(nn.Module):
    def __init__(self, device, config=None, inp_ch=1, out_ch=1,
                 arch=16, depth=3, activ='leak', concat=None):
        super(Unet, self).__init__()

        # Load the configuration, if none is provided, we will default to ViT-B_16
        if config is None:
            config = get_b16_config()

        self.activ = activ
        self.device = device
        self.out_ch = out_ch
        self.inp_ch = inp_ch
        self.depth = depth
        self.arch = arch
        self.concat = None

        self.arch_n = []
        self.enc = []
        self.dec = []
        self.layers = []
        self.skip = []

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
        for idx in range(len(self.arch_n) - 1):
            self.enc.append(
                Conv_Block(self.arch_n[idx], self.arch_n[idx + 1], activ=self.activ, pool='down_max'))

            # Debug print to inspect self.arch_n[-1]
            print(f"self.arch_n: {self.arch_n}")
            print(f"self.arch_n[-1]: {self.arch_n[-1]}")

            # Assuming the Conv_Block outputs feature maps of shape (B, C, H, W)
            # Modify this part to correctly handle the shape of last_feature_map
            if isinstance(self.arch_n[-1], int):  # If it’s an integer, we are missing dimensions.
                feature_dim = self.arch_n[-1]
                seq_length = self.arch_n[-2] ** 2  # Assuming H == W, and we use the second last element for dimensions.
            else:
                # Assuming it's a shape tuple/list (C, H, W) or similar
                last_feature_map = self.arch_n[-1]
                seq_length = last_feature_map[-1] * last_feature_map[-2]  # Assuming last_feature_map is (C, H, W)
                feature_dim = last_feature_map[1]  # Assuming last shape is (C, H, W) and we need C

            # More debug prints to verify the calculated dimensions
            print(f"seq_length: {seq_length}, feature_dim: {feature_dim}")

        # Add the transformer component here
        self.enc.append(lambda x: x.flatten(2).transpose(1, 2))  # Flatten and transpose to (B, N, D)
        self.enc.append(Transformer(d_model=feature_dim, num_patches=seq_length))  # Or relevant params
        self.enc.append(
            lambda x: x.transpose(1, 2).view(-1, feature_dim, last_feature_map[-2],
                                             last_feature_map[-1]))  # Restore shape

        self.layers = [Conv_Block(self.arch_n[-1], self.arch_n[-1], activ=self.activ, pool='up_stride')]

        for idx in range(len(self.arch_n) - 2):
            self.dec.append(
                Conv_Block(self.concat[- (idx + 1)] * self.arch_n[- (idx + 1)], self.arch_n[- (idx + 2)],
                           activ=self.activ, pool='up_stride'))
        self.dec.append(Conv_Block(self.concat[0] * self.arch, self.arch, activ=self.activ))
        self.layers.append(Conv_Layer(self.arch, self.out_ch, 1, 1, norm=None, activ='tanh'))

    def prep_params(self):
        for blk_idx in range(len(self.enc)):
            self.add_module(f'enc_{blk_idx + 1}', self.enc[blk_idx])

        self.add_module(f'mid', self.layers[0])

        for blk_idx in range(len(self.dec)):
            self.add_module(f'dec_{blk_idx + 1}', self.dec[blk_idx])

        self.add_module(f'final', self.layers[1])

    """def forward(self, img):
        h = img
        h_skip = []

        for conv in self.enc:
            hs, h = conv(h)
            h_skip.append(hs)

        _, h = self.mid(h)

        for l_idx in range(len(self.dec)):
            if self.concat[-(l_idx + 1)] == 2:
                _, h = self.dec[l_idx](concat_curr(h_skip[-(l_idx + 1)], h))
            else:
                _, h = self.dec[l_idx](h)

        h = self.final(h)

        return h"""

    def forward(self, img):
        h = img
        h_skip = []

        for conv in self.enc:
            hs, h = conv(h)
            h_skip.append(hs)

        _, h = self.mid(h)

        for l_idx in range(len(self.dec)):
            if self.concat[-(l_idx + 1)] == 2:
                _, h = self.dec[l_idx](concat_curr(h_skip[-(l_idx + 1)], h))
            else:
                _, h = self.dec[l_idx](h)

        h = self.final(h)

        return h


# ----------------Test---------------------------

if __name__ == '__main__':
    import torch

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    x = torch.randn(1, 1, 128, 128).to(device)

    net = Conv_Layer(1,1,4,2, 2).to(device)
    y = net(x)
