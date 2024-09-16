from basic_layer_trans import *
from torch import nn
import numpy as np
import torch
from trans_unet.vit_seg_modeling import Transformer  # Importing transformer and encoder from vit_seg_modeling.py
from trans_unet.vit_seg_configs import get_r50_b16_config
from trans_unet.vit_seg_configs import get_b16_config


class UnetWithTransformer(nn.Module):
    def __init__(self, device, inp_ch=1, out_ch=1, arch=16, depth=3, activ='leak', concat=None, train=True, vis=False,
                 config=None):
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
            config.img_size = 224  # Set image size to 200x200

        self.arch_n = []
        self.enc = []
        self.dec = []  # Decoder layers
        self.layers = []
        self.skip = []

        # Use the Transformer as encoder
        transformer = Transformer(config, img_size=224, vis=vis)
        self.transformer = transformer.to(self.device)  # Move to device

        self.encoder = transformer.to(self.device)  # Ensure the encoder is on the correct device
        # Freeze the first N layers of the transformer encoder
        self.freeze_transformer_layers(num_layers_to_freeze=2)

        # Layer to adjust the number of channels before feeding into transformer
        self.channel_adjustment = nn.Conv2d(512, 1024, kernel_size=1, stride=1).to(self.device)  # 512 -> 1024 channels

        # Throttling layer: Converts transformer output to match decoder input dimensions
        self.throttling_layer = nn.Sequential(
            nn.Conv2d(config.hidden_size, 32, kernel_size=1).to(self.device),  # Move to device
            nn.ReLU(inplace=True).to(self.device)  # Move to device
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
        self.channel_projection = nn.Conv2d(32, 3, kernel_size=1).to(self.device)  # Move to device

    def freeze_transformer_layers(self, num_layers_to_freeze):
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

    def prep_arch_list(self):
        self.arch_n = []
        for dl in range(0, self.depth + 2):  # Ensuring depth + 2 elements
            self.arch_n.append((2 ** (dl - 1)) * self.arch)

        self.arch_n[0] = self.inp_ch  # Initialize the first element to the input channels

    def organize_arch(self, config):

        for idx in range(len(self.arch_n) - 1):
            self.enc.append(
                Conv_Block(self.arch_n[idx], self.arch_n[idx + 1], activ=self.activ, pool='down_max'))

        # Add a layer to reduce channels from 512 (output of encoder) to 512 (expected by transformer)
        self.reduce_channels = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1).to(self.device)

        self.layers = [Conv_Block(self.arch_n[-1], self.arch_n[-1], activ=self.activ, pool='up_stride')]

        # Use depth + 1 to ensure safe access in arch_n and concat
        for idx in range(len(self.arch_n) - 2):
            self.dec.append(
                Conv_Block(self.concat[-(idx + 1)] * self.arch_n[-(idx + 1)], self.arch_n[-(idx + 2)],
                           activ=self.activ, pool='up_stride'))

        # Define a convolution layer to reduce skip tensor channels before concatenation
        self.conv_skip = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=1)  # Example channels

        self.dec.append(Conv_Block(self.concat[0] * self.arch, self.arch, activ=self.activ))

        self.layers.append(Conv_Layer(self.arch, self.out_ch, 1, 1, norm=None, activ='tanh'))

    def prep_params(self):
        for blk_idx in range(len(self.dec)):
            self.add_module(f'dec_{blk_idx + 1}', self.dec[blk_idx])

        self.add_module(f'mid', self.layers[0])
        self.add_module(f'final', self.layers[1])

    def forward(self, img):
        # Ensure img is on the correct device and dtype
        img = img.to(self.device, dtype=torch.float32)
        h_skip = []

        # Move encoder Conv_Block layers to MPS device
        for idx, enc_layer in enumerate(self.enc):
            enc_layer = enc_layer.to(self.device)  # Move layer to MPS
            _, img = enc_layer(img)  # We only want to pass `img`, the second element
            h_skip.append(img)  # Save the output of each encoder layer for skip connections

        # **Apply the Conv2d layer to reduce the number of channels**
        img = self.reduce_channels(img)  # Reduce channels from 512 to 1024
        print(f"Shape after reduce_channels: {img.shape}")  # Add print statement

        # **Ensure the spatial resolution is aligned with the Vision Transformer input**
        img_size = 224  # Adjust this size according to your transformer patch size
        img = F.interpolate(img, size=(img_size, img_size), mode='bilinear', align_corners=True)

        print(f"Shape before Transformer input: {img.shape}")  # Add print statement

        # Transformer encoder step
        x, attn_weights, features = self.encoder(img.to(self.device))

        # Reshape transformer output
        batch_size, num_patches, hidden_size = x.shape
        height = width = int(num_patches ** 0.5)
        x = x.permute(0, 2, 1).contiguous().view(batch_size, hidden_size, height, width).to(self.device)

        # Apply throttling layer to convert transformer output to fit UNet decoder input
        self.throttling_layer = self.throttling_layer.to(self.device)  # Ensure throttling layer is on MPS
        x = self.throttling_layer(x)

        # Decoder loop with concatenation and skip connections
        for l_idx in range(len(self.dec)):
            if self.concat[-(l_idx + 1)] == 2:
                # Concatenate skip connection from encoder
                skip_tensor = h_skip[-(l_idx + 1)]

                # Interpolate skip_tensor if dimensions don't match
                if skip_tensor.size(2) != x.size(2) or skip_tensor.size(3) != x.size(3):
                    skip_tensor = F.interpolate(skip_tensor, size=(x.size(2), x.size(3)), mode='bilinear',
                                                align_corners=True)

                # Concatenate along the channel dimension
                x = torch.cat([x, skip_tensor], dim=1)

                # Adjust the number of channels to match the expected input size of the next decoder layer
                if x.size(1) != self.dec[l_idx].c1.conv[0].in_channels:  # Check if the channels mismatch
                    channel_adjustment_layer = nn.Conv2d(in_channels=x.size(1),
                                                         out_channels=self.dec[l_idx].c1.conv[0].in_channels,
                                                         kernel_size=1).to(self.device)
                    x = channel_adjustment_layer(x)

            # Apply decoder layer
            _, x = self.dec[l_idx](x)

        # Final projection to output channels
        h = self.final(x)

        return h


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
