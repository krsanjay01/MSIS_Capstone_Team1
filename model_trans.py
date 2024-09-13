from basic_layers import *
from torch import nn
import numpy as np
import torch
from trans_unet.vit_seg_modeling import Transformer  # Importing transformer and encoder from vit_seg_modeling.py
from trans_unet.vit_seg_configs import get_b16_config


class UnetWithTransformer(nn.Module):
    def __init__(self, device, inp_ch=1, out_ch=1, arch=16, depth=3, activ='leak', concat=None, train=True, vis=False, config=None):
        super(UnetWithTransformer, self).__init__()

        self.activ = activ
        self.device = device
        self.out_ch = out_ch
        self.inp_ch = inp_ch
        self.depth = depth
        self.arch = arch
        self.concat = None
        self.vis = vis

        # If config is None, use a default configuration (like ViT-B_16)
        if config is None:
            config = get_b16_config()  # Initialize default ViT-B_16 config



        self.arch_n = []
        self.dec = []  # Decoder layers
        self.layers = []
        self.skip = []

        # Use the Transformer as encoder
        transformer = Transformer(config, img_size=224, vis=vis)
        self.transformer = transformer
        self.encoder = transformer
        # Freeze the first N layers of the transformer encoder
        self.freeze_transformer_layers(num_layers_to_freeze=2)

        # Throttling layer: Converts transformer output to match decoder input dimensions
        self.throttling_layer = nn.Sequential(
            nn.Conv2d(config.hidden_size, arch, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # Load pre-trained weights if a path is provided
        if config.pretrained_path and train:
            self.load_pretrained_weights(config.pretrained_path)

        # Initialize the decoder and final layer as in the original model
        self.check_concat(concat)
        self.prep_arch_list()
        self.organize_arch(config)
        self.prep_params()

        # Add a Conv2d layer to reduce the output channels from 32 to 3
        self.channel_projection = nn.Conv2d(32, 3, kernel_size=1)

    def freeze_transformer_layers(self, num_layers_to_freeze):
        """
               Freezes the first num_layers_to_freeze layers of the transformer encoder.
               """
        """
                Freezes the first num_layers_to_freeze layers of the transformer encoder.
                """
        # Access the encoder's layers (self.transformer.encoder.layer is a ModuleList)
        for i, block in enumerate(self.transformer.encoder.layer):
            if i < num_layers_to_freeze:
                for param in block.parameters():
                    param.requires_grad = False

    def load_pretrained_weights(self, pretrained_path):
        """
        Load the pre-trained ViT weights from the .npz file.
        """
        # Load the weights from the .npz file
        pretrained_weights = np.load(pretrained_path)

        # Loop over all layers and assign weights from the .npz file
        for name, param in self.encoder.named_parameters():
            layer_name = name.replace('.', '/')
            if layer_name in pretrained_weights:
                print(f"Loading weight for layer: {layer_name}")
                param.data = torch.from_numpy(pretrained_weights[layer_name])

        print("Pre-trained weights loaded successfully.")

    def check_concat(self, con):
        # Ensure con is not None and is a list
        if con is None:
            con = [1] * (self.depth + 1)  # Initialize with 1s if None
        elif isinstance(con, np.ndarray):  # If con is a NumPy array, convert it to a list
            con = con.tolist()

        # If the length of con is less than depth + 1, pad with 1s
        if len(con) < (self.depth + 1):
            con = con + [1] * ((self.depth + 1) - len(con))

        # If the length of con is greater than depth + 1, truncate to match the expected size
        elif len(con) > (self.depth + 1):
            con = con[:self.depth + 1]

        # Multiply non-zero values by 2 and replace any 0s with 1
        self.concat = [2 * c if c != 0 else 1 for c in con]

        # Debugging print statements
        print(f"Final concat after check_concat: {self.concat}")
        print(f"Length of concat: {len(self.concat)}")

    def prep_arch_list(self):
        self.arch_n = []
        for dl in range(0, self.depth + 2):  # Ensuring depth + 2 elements
            self.arch_n.append((2 ** (dl - 1)) * self.arch)

        self.arch_n[0] = self.inp_ch  # Initialize the first element to the input channels


    def organize_arch(self, config):
        self.layers = [Conv_Block(self.arch_n[-1], self.arch_n[-1], activ=self.activ, pool='up_stride')]

        # Use depth + 1 to ensure safe access in arch_n and concat
        for idx in range(len(self.arch_n) - 2):
            self.dec.append(
                Conv_Block(self.concat[-(idx + 1)] * self.arch_n[-(idx + 1)], self.arch_n[-(idx + 2)],
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

        # Reshape transformer output
        batch_size, num_patches, hidden_size = x.shape
        height = width = int(num_patches ** 0.5)  # Assuming a square grid of patches
        x = x.permute(0, 2, 1).contiguous().view(batch_size, hidden_size, height, width)

        # Apply throttling layer
        x = self.throttling_layer(x)

        # Project the output channels from 32 to 3 to match residuals
        x = self.channel_projection(x)

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
