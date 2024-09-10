from torch import nn
import torch.nn.functional as F
from einops import rearrange
import torch


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

        if shift_size > 0:
            self.attn_mask = self.create_mask(input_resolution, window_size, shift_size)
        else:
            self.attn_mask = None

    def create_mask(self, input_resolution, window_size, shift_size):
        # Create a mask for shifted windows
        H, W = input_resolution
        img_mask = torch.zeros((1, H, W, 1))

        h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = self.window_partition(img_mask, window_size)
        mask_windows = mask_windows.view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift if shift_size > 0
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = self.window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Window-based multi-head self-attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # Reverse windows back to original size
        shifted_x = self.window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)
        x = shortcut + x

        # Feed-forward network
        x = x + self.mlp(self.norm2(x))
        return x
    
    # Partitioning an image into windows
    def window_partition(x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = rearrange(x, 'b h1 w1 h2 w2 c -> (b h1 w1) (h2 w2) c')
        return windows

    # Reverse the window partitioning to reconstruct the image
    def window_reverse(windows, window_size, H, W):
        B = int(windows.shape[0] // (H * W // window_size // window_size))
        x = rearrange(windows, '(b h1 w1) (h2 w2) c -> b (h1 h2) (w1 w2) c', 
                    b=B, h1=H//window_size, w1=W//window_size, h2=window_size, w2=window_size)
        return x





class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        _, _, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image dimensions must be divisible by patch size"
        x = self.proj(x).flatten(2).transpose(1, 2)  # B x num_patches x embed_dim
        return x
    

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5  # Scale factor for attention
        
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        
        # Relative position encoding
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Add relative position bias
        attn = attn + self.relative_position_bias_table
        
        if mask is not None:
            attn = attn + mask
        
        attn = self.softmax(attn)
        attn = attn @ v
        
        attn = attn.transpose(1, 2).reshape(B_, N, C)
        return self.proj(attn)


# -------- Functions ----------------------------------------------------------

def concat_curr(prev, curr):
    diffY = prev.size()[2] - curr.size()[2]
    diffX = prev.size()[3] - curr.size()[3]

    curr = F.pad(curr, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

    x = torch.cat([prev, curr], dim=1)
    return x
