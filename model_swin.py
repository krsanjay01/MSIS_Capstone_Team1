from swin_unet.vision_transformer import SwinUnet as ViT_seg
from torch import nn
from swin_unet.config import get_config


class SwinUnet(nn.Module):
    def __init__(self, device, args):
        super(SwinUnet, self).__init__()

        config = get_config(args)

        self.net = ViT_seg(config, device, img_size=args.img_size, num_classes=args.num_classes).to(device)
        self.net .load_from(config)

    def forward(self, x):
        self.net.forward(x)


# ----------------Test---------------------------

if __name__ == '__main__':
    import torch

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    x = torch.randn(1, 1, 128, 128).to(device)

    net = SwinUnet(device).to(device)
    y = net(x)
    print(y.shape)
