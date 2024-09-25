from basic_layers import *
from torch import nn
import math


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

    def forward(self, x):
        # Assuming input shape is (batch_size, channels, height, width)
        b, c, h, w = x.size()
        x = x.view(b, c, h * w).permute(2, 0, 1)  # Reshape to (sequence_length, batch_size, embedding_dim)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = x.permute(1, 2, 0).view(b, c, h, w)  # Reshape back to (batch_size, channels, height, width)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class UnetWithTransformer(nn.Module):
    def __init__(self, device, inp_ch=1, out_ch=1,
                 arch=16, depth=3, activ='leak', concat=None,
                 trans_nhead=4, trans_dim=64, trans_layers=4):
        super(UnetWithTransformer, self).__init__()

        self.activ = activ
        self.device = device
        self.out_ch = out_ch
        self.inp_ch = inp_ch
        self.depth = depth
        self.arch = arch
        self.concat = None

        self.trans_nhead = trans_nhead  # Number of attention heads
        self.trans_dim = trans_dim  # Dimension of transformer input
        self.trans_layers = trans_layers  # Number of transformer layers

        self.arch_n = []
        self.enc = []
        self.dec = []
        self.layers = []
        self.skip = []

        self.check_concat(concat)
        self.prep_arch_list()
        self.organize_arch()
        self.prep_params()

        # Transformer block
        self.transformer = TransformerBlock(d_model=self.arch_n[-1], nhead=self.trans_nhead, num_layers=self.trans_layers)

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

    def forward(self, img):
        h = img
        h_skip = []

        for conv in self.enc:
            hs, h = conv(h)
            h_skip.append(hs)

        _, h = self.mid(h)

        # Apply the transformer block here
        h = self.transformer(h)

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

    net = UnetWithTransformer(device, inp_ch=1, out_ch=1, arch=16, depth=3, activ='leak').to(device)
    y = net(x)
    print(y.shape)
