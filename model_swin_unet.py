from swin_layer import *
from torch import nn


class SwinTransformer(nn.Module):
    def __init__(self, device, config=None, inp_ch=1, out_ch=1, arch=16, activ='leak', concat=None, vis=False, img_size=224, patch_size=4, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., num_classes=1000):
        super(SwinTransformer, self).__init__()

        self.activ = activ
        self.device = device
        self.out_ch = out_ch
        self.inp_ch = inp_ch
        self.arch = arch
        self.concat = None
        self.vis = vis

        self.arch_n = []
        self.dec = []  # Decoder layers
        self.layers = []
        self.skip = []

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_chans=inp_ch, embed_dim=embed_dim)

        # Build stages with Swin Transformer Blocks
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList([
                SwinTransformerBlock(dim=embed_dim, input_resolution=(img_size // (2 ** i_layer), img_size // (2 ** i_layer)), 
                                     num_heads=num_heads[i_layer], window_size=window_size, shift_size=(i_layer % 2) * window_size // 2)
                for _ in range(depths[i_layer])
            ])
            self.layers.append(layer)
            embed_dim *= 2  # Double the dimensions at each stage

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

   

    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            for block in layer:
                x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x


# ----------------Test---------------------------

if __name__ == '__main__':
    import torch

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    x = torch.randn(1, 1, 128, 128).to(device)

    net = SwinTransformer(device).to(device)
    y = net(x)
    print(y.shape)
