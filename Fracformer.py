import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.fft import rfft, irfft
from torch.nn import Sequential, LayerNorm, Linear, ReLU, Dropout
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import SPSD
from SPSD import PoolType, getPool1d, getPool2d


class EpsilonReLU(nn.Module):
    """ReLU with epsilon added to avoid numerical issues with zero-sized tensors."""
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return F.relu(x) + self.epsilon


class SignalEmbedSoft(nn.Module):
    """abstract signal to Patch Embedding
    """
    def __init__(self, sig_len = 750, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        num_patches = sig_len // patch_size
        self.sig_len = sig_len
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, L = x.shape
        assert L == self.sig_len, \
            f"Input signal length ({L}) doesn't match model ({self.sig_len})."
        x = self.proj(x).transpose(1, 2)
        return x


class ImgEmbed(nn.Module):
    """abstract image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, layerNum = 1):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj2 = nn.Sequential(*[
            nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size) for _ in range(layerNum - 1)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        x = self.proj1(x)
        x = self.proj2(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, x.size(1))
        # print(x.shape)
        return x


class ImgEmbed4SPHS(nn.Module):
    """abstract image to Patch Embedding
    """
    def __init__(self, img_height=512, img_weight=512, patch_size=16, in_chans=1, embed_dim=256, layerNum = 1):
        super().__init__()
        self.H = img_height // patch_size
        self.W = img_weight // patch_size
        self.num_patches = self.H * self.W
        self.patch_size = patch_size

        self.proj1 = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj2 = nn.Sequential(*[
            nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size) for _ in range(layerNum - 1)
        ])

    def forward(self, x):
        x = self.proj1(x)
        x = self.proj2(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalSPSDFilter(nn.Module):
    def __init__(self, dim, len, minAlpha = 0, maxAlpha = 3, offset = 3, poolType = PoolType.MAX):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(len, dim, dtype=torch.float32) * 0.02)
        self.len = len
        self.spsd = SPSD.SPSD1D(minAlpha = minAlpha, maxAlpha = maxAlpha, N = len, offset = offset, poolType = poolType)

    def forward(self, x):
        device = x.device
        x, maxHolder = self.spsd(x.transpose(1, 2))
        x = x.transpose(1, 2).to(device)
        x = x * self.weight.to(device)
        return x, maxHolder.to(device)


class GlobalSPSDFilter2D(nn.Module):
    def __init__(self, dim, len, minAlpha = 0, maxAlpha = 3, offset = 3, poolType = PoolType.MAX):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(len, dim, dtype=torch.float32) * 0.02)
        self.len = len
        self.spsd = SPSD.SPSD2D(minAlpha = minAlpha, maxAlpha = maxAlpha, N = len, offset = offset, poolType = poolType)

    def forward(self, x):
        # x : B * H * W * C
        device = x.device
        x, maxHolder = self.spsd(x.permute(0, 3, 1, 2))  # x : B * C * N
        x = x.transpose(1, 2).to(device)
        x = x * self.weight.to(device)
        return x, maxHolder.to(device)


class Block(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, len = 64, minAlpha = 0, maxAlpha = 3, 
                 offset = 3, poolType = PoolType.MAX):
        super().__init__()
        self.epsilonrelu = EpsilonReLU()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalSPSDFilter(dim, len, minAlpha, maxAlpha, offset, poolType)
        self.drop_path = nn.Dropout(0.1)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)

        self.shortcut = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        x0, maxHolder = self.filter(self.epsilonrelu(self.norm1(x)))
        x0 = self.drop_path(self.mlp(self.norm2(x0)))
        return x + x0


class ResBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, len = 64, minAlpha = 0, maxAlpha = 3, 
                    offset = 3, poolType = PoolType.MAX):
        super().__init__()
        self.epsilonrelu = EpsilonReLU()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.filter = GlobalSPSDFilter(dim, len, minAlpha, maxAlpha, offset, poolType)
        self.drop_path = nn.Dropout(0.1)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.alphaEmbed = nn.Sequential(
            nn.Linear(1, len),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.norm3 = norm_layer(dim)

        self.shortcut = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        x0, maxHolder = self.filter(self.epsilonrelu(self.norm1(x)))
        x0 = x0 + self.alphaEmbed(maxHolder).transpose(1, 2)
        x0 = self.drop_path(self.mlp(self.norm2(x0)))
        return x + x0


class ResBlock2D(nn.Module):
    
        def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, 
                    norm_layer=nn.LayerNorm, len = 64, minAlpha = 0, maxAlpha = 3, 
                        offset = 3, poolType = PoolType.MAX):
            super().__init__()
            self.epsilonrelu = EpsilonReLU()
            self.norm1 = norm_layer(dim)
            self.norm2 = norm_layer(dim)
            self.norm3 = norm_layer(dim)
            self.filter = GlobalSPSDFilter2D(dim, len, minAlpha, maxAlpha, offset, poolType)
            self.drop_path = nn.Dropout(0.1)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.alphaEmbed = nn.Sequential(
                nn.Linear(1, len),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.norm3 = norm_layer(dim)
    
            self.shortcut = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(dim)
            )
    
        def forward(self, x):
            x0, maxHolder = self.filter(self.epsilonrelu(self.norm1(x)))
            x0 = x0 + self.alphaEmbed(maxHolder).transpose(1, 2)
            x0 = self.drop_path(self.mlp(self.norm2(x0)))
            x = x.view(x.size(0), -1, x.size(-1))
            return x + x0

class SIFTBlock(nn.Module):

    def __init__(self, dim, H, W, N1 = 5, minAlpha = 0, maxAlpha = 9, N2 = 9, offset = 3, poolType = PoolType.MAX):
        super().__init__()
        self.dim = dim
        self.N1  = N1
        self.minAlpha = minAlpha
        self.maxAlpha = maxAlpha
        self.N2  = N2
        self.offset = offset
        self.poolType = poolType
        self.softplus = nn.Softplus()
        self.sphs = SPSD.SIFT(N1, minAlpha, maxAlpha, N2, offset, poolType)
        self.filter = torch.nn.Parameter(torch.randn(dim, H, W, N2, dtype=torch.float32))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        assert len(x.shape) == 4, "Only accept tensor input with dimension 4."
        x = self.softplus(x)
        x = self.sphs(x)
        self.filter = self.filter.to(x.device)
        torch.nn.functional.softmax(self.filter, dim=-1)
        x = torch.mul(x, self.filter)
        x = torch.sum(x, dim=-1)
        return x


class Fracformer(nn.Module):
    def __init__(self, img_height=512, img_weight=512, patch_size=16, in_chans=1, embed_dim=256, 
                 N1=5, minAlpha=0, maxAlpha=9, N2=9, offset=3, poolType=PoolType.MAX, 
                 mixer_depth=2, mixer_head=8, mlp_ratio=4.0, num_classes=4):
        super().__init__()
        self.img_height = img_height
        self.img_weight = img_weight
        self.patch_size = patch_size
        self.H = img_height // patch_size
        self.W = img_weight // patch_size
        self.num_patches = self.H * self.W
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.mixer_depth = mixer_depth
        self.mixer_head = mixer_head

        self.embed = ImgEmbed4SPHS(img_height, img_weight, patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.H, self.W))
        self.sphs_block = SIFTBlock(embed_dim, self.H, self.W, N1, minAlpha, maxAlpha, N2, offset, poolType)
        self.mixer_blocks = TransformerEncoder(
            TransformerEncoderLayer(
                d_model = embed_dim,
                nhead = mixer_head,
                dim_feedforward = int(embed_dim * mlp_ratio),
                dropout = 0.1,
                activation = 'gelu',
                batch_first = True,
            ),
            num_layers=mixer_depth,
            norm=torch.nn.LayerNorm(embed_dim)
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = Linear(embed_dim, num_classes)
        self.pool = nn.MaxPool1d(kernel_size=self.num_patches)

    def forward(self, x):
        # pre-processing
        x = self.embed(x)
        x = x + self.pos_embed

        # SPHS block
        x = self.sphs_block(x)

        # Middle blocks
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1)
        x = torch.nn.functional.max_pool1d(x, 4)
        x = x.permute(0, 2, 1)

        # Mixer blocks
        x = self.mixer_blocks(x)

        # post-processing
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.avg_pool1d(x, x.shape[-1])
        x = x.view(x.size(0), -1)
        x = self.head(x)
    
        return x
    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Fracformer()
    x = torch.randn(2, 1, 512, 512)
    y = net(x)
    print(y.shape)

