import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

class TimeEmbedding(nn.Module):
    def __init__(self, time_steps, in_channels, dim):
        assert in_channels % 2 == 0
        super().__init__()
        emb = torch.arange(0, in_channels, step=2) / in_channels * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(time_steps).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [time_steps, in_channels // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [time_steps, in_channels // 2, 2]
        emb = emb.view(time_steps, in_channels)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(in_channels, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, t):
        return self.timembedding(t)

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """添加了fuse功能的标准卷积模块
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, bn=True, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d),dilation=d, groups=g, bias=False)
        self.bn = nn.GroupNorm(min(32, c2), c2) if bn is True else nn.Identity()
        self.act = nn.SiLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    
    def forward(self, x, temb=None):
        return self.act(self.bn(self.conv(x)))

class TransConv(nn.Module):
    """添加了fuse功能的标准卷积模块
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, bn=True, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.ConvTranspose2d(c1, c2, k, s, autopad(k, p, d), dilation=d, groups=g, bias=False)
        self.bn = nn.GroupNorm(min(32, c2), c2) if bn is True else nn.Identity()
        self.act = nn.SiLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    
    def forward(self, x, temb=None):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x, temb=None):
        return self.act(self.conv(x))

class ShortCut(nn.Module):
    """添加了fuse功能的标准卷积模块
    """
    def __init__(self, c1, c2):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, 1)
    
    def forward(self, x):
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, c1, c2, tdim, ratio=2.0, shortcut=True, act=nn.SiLU(inplace=True), attn=False):
        super().__init__()
        c_ = int(c2 * ratio)
        self.shortcut_k = shortcut
        if shortcut:
            self.shortcut = ShortCut(c1, c2)
        self.conv1 = Conv(c1, c_, 3, 1, act=False)
        self.conv2 = Conv(c_, c2, 3, 1, act=act)
        self.temb_proj = nn.Sequential(
            nn.Linear(tdim, c2),
            act,
            nn.Linear(c2, c_),
        )
        self.attn = AttnBlock(c2) if attn else nn.Identity()

    def forward(self, x, temb):
        if self.shortcut_k:
            shortcut = self.shortcut(x)
        x = self.conv1(x)
        x += self.temb_proj(temb)[:, :, None, None]
        x = self.conv2(x)
        
        if self.shortcut_k:
            x = x + shortcut
            return self.attn(x)
        else:
            x = self.attn(x)
            return x

class GhostResBlock(nn.Module):
    def __init__(self, c1, c2, tdim, ratio=0.5, shortcut=True, act=nn.SiLU(inplace=True)):
        super().__init__()
        c_ = int(c2 * ratio)
        self.shortcut = shortcut if c1 == c2 else False
        self.conv1 = Conv(c1, c_, 3, 1, act=act)
        self.conv2 = Conv(c_, c_, 3, 1, g=c_, act=act)
        self.temb_proj = nn.Sequential(
            act,
            nn.Linear(tdim, c_),
        )

    def forward(self, x, temb):
        if self.shortcut:
            shortcut = x
        x1 = self.conv1(x)
        x1 += self.temb_proj(temb)[:, :, None, None]
        x2 = self.conv2(x1)
        x = torch.cat([x1, x2], dim=1)
        if self.shortcut:
            return x + shortcut
        else:
            return x
        
class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_qkv = nn.Conv2d(in_ch, in_ch*3, 3, stride=1, padding=1)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_qkv, self.proj]:    
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x, temb=None):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        qkv = self.proj_qkv(x)
        q, k, v = qkv.chunk(3, 1)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x, temb=None):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)
    

class UNet(nn.Module):
    __act__ = nn.SiLU(inplace=False)
    __Block__ = ResBlock
    def __init__(self,
                 time_steps,
                 depth=4,
                 in_channels=3,
                 out_channels=3,
                 dims=[64, 128, 256, 512, 1024],
                 num_blocks=[2,2,3,2,3],
                 attn=[False, False, True, False],
                  ):
        super(UNet, self).__init__()
        tdim = dims[0]*4
        self.time_embedding = TimeEmbedding(time_steps, dims[0], dims[0]*4)
        self.stem = Conv(in_channels, dims[0], 3, 1, act=self.__act__)

        # 下行过程
        self.downs = nn.ModuleList()
        for i in range(depth):
            for _ in range(num_blocks[i]):
                self.downs.append(self.__Block__(dims[i], dims[i], tdim, act=self.__act__, attn=attn[i]))
            self.downs.append(Conv(dims[i], dims[i+1], 3, 2, act=self.__act__))

        # 中间过程
        self.mid_block = nn.ModuleList()
        for _ in range(num_blocks[-1]):
            # self.mid_block.append(self.__Block__(dims[-1], dims[-1], tdim, act=self.__act__))
            self.mid_block.append(AttnBlock(dims[-1]))
            
        # 上行过程
        self.ups = nn.ModuleList()
        for i in reversed(range(depth)):
            self.ups.append(TransConv(dims[i+1], dims[i], 4, 2, p=1, act=self.__act__))
            for _ in range(num_blocks[i]):
                self.ups.append(self.__Block__(dims[i]*2, dims[i], tdim, act=self.__act__, attn=attn[i]))

        # self.proj = TransConv(dims[0], out_channels, 4, 2, p=1, act=self.__act__)
        self.proj = nn.Sequential(nn.GroupNorm(32, dims[0]),
                                nn.Conv2d(dims[0], out_channels, 3, 1, 1))
        self.initialize()
            
    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, t):
        # time step embedding
        temb = self.time_embedding(t)
        # unet downsampleing
        skips = []
        x = self.stem(x, temb)
        for block in self.downs:
            x = block(x, temb)
            if isinstance(block, self.__Block__):
                skips.append(x)

        # unet middle
        for block in self.mid_block:
            x = block(x, temb)

        # unet upsampleing
        for block in self.ups:
            if isinstance(block, self.__Block__):
                x = torch.cat([skips.pop()* 2**(-0.5), x], dim=1)
            x = block(x, temb)
        x = self.proj(x)
        return x

class GhostUNet(UNet):
    __act__ = nn.SiLU(inplace=False)
    __Block__ = GhostResBlock

if __name__ == "__main__":
    batch_size = 8
    time_steps = 100
    x = torch.randn(batch_size, 3, 256, 256)
    t = torch.randint(time_steps, (batch_size, ))
    model = UNet(time_steps)
    # model = TransConv(3, 8, 4, 2, p=1)
    y = model(x, t)
    print(model, y.shape)