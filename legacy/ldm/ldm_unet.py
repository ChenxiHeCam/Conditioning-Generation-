"""
3D U-Net denoiser for the Stage 2 latent diffusion model.

Inputs (forward):
  - z_noisy:  (B, 8, 32, 32, 32)   noisy normalized latent
  - ic_delta: (B, 1, 64, 64, 64)   initial density patch
  - ic_vbv:   (B, 1, 64, 64, 64)   initial velocity patch
  - params:   (B, 4)               astrophysical parameters
  - redshift: (B,)                 scalar redshift
  - sigma:    (B,)                 EDM noise level

Output:
  - x_hat: (B, 8, 32, 32, 32)      predicted denoised latent (in v-pred sense)

Architecture (per design review):
  - Latent normalization is done OUTSIDE; this module sees already-normalized z.
  - IC stem: a small learned conv stem (2 -> 16 -> 32 channels, stride-2 once
    to take 64^3 -> 32^3) instead of trilinear downsample, so small-scale IC
    modes survive.
  - Spatial backbone: 32 -> 16 -> 8, channels 64 -> 128 -> 256.
  - Self-attention: windowed (window=8) at 16^3, global at 8^3.
  - Conditioning vector: concat(4 params, redshift, sigma) -> MLP(256) -> 256d
    embedding -> FiLM (scale + shift) injected at each ResBlock after GroupNorm.
  - GroupNorm groups: 8 at 64ch, 32 at 128/256ch (per the reviewer's note: GN(32)
    on 64 channels = 2 ch/group, too noisy).
  - CFG dropout: handled outside by zeroing the embedding and/or IC channels.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def sinusoidal_embedding(x, dim=128, max_period=10000.0):
    """Fourier feature embedding for a scalar (or 1-D tensor)."""
    if x.ndim == 0:
        x = x.unsqueeze(0)
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(half, device=x.device, dtype=torch.float32) / half)
    args = x[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ConditioningEmbedding(nn.Module):
    """Combine (params, redshift, sigma) -> single 256-d vector."""
    def __init__(self, n_params=4, emb_dim=128, out_dim=256):
        super().__init__()
        self.n_params = n_params
        # Embed params as a JOINT MLP (per agent review: don't sum)
        self.param_mlp = nn.Sequential(
            nn.Linear(n_params, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        # Final MLP combines param-embed + sinusoidal(z) + sinusoidal(sigma)
        self.combine = nn.Sequential(
            nn.Linear(emb_dim * 3, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.out_dim = out_dim
        self.emb_dim = emb_dim

    def forward(self, params, redshift, sigma):
        p = self.param_mlp(params)
        z = sinusoidal_embedding(redshift, dim=self.emb_dim)
        s = sinusoidal_embedding(torch.log(sigma.clamp_min(1e-6)) * 0.25,
                                 dim=self.emb_dim)
        return self.combine(torch.cat([p, z, s], dim=-1))


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def gn_groups(ch):
    """Group count: 8 for <=64 ch, 32 for larger; clamp to a divisor of ch."""
    target = 8 if ch <= 64 else 32
    # ch may not be divisible by target (e.g. skip-concat 48+96=144); fall back
    for g in (target, 16, 8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class FiLM(nn.Module):
    """Per-channel scale+shift from a 256-d conditioning vector."""
    def __init__(self, cond_dim, ch):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * ch)
        # zero-init so initial behaviour is identity
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        # x: (B, C, D, H, W); cond: (B, cond_dim)
        sc_sh = self.proj(cond)
        scale, shift = sc_sh.chunk(2, dim=-1)
        # (B, C, 1, 1, 1)
        scale = scale[:, :, None, None, None]
        shift = shift[:, :, None, None, None]
        return x * (1 + scale) + shift


class ResBlock3D(nn.Module):
    """ResBlock with FiLM conditioning."""
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(gn_groups(in_ch), in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.film1 = FiLM(cond_dim, out_ch)
        self.norm2 = nn.GroupNorm(gn_groups(out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.film2 = FiLM(cond_dim, out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.film1(h, cond)
        h = self.conv2(F.silu(self.norm2(h)))
        h = self.film2(h, cond)
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class GlobalAttention3D(nn.Module):
    """Standard multi-head self-attention over all spatial tokens.
    Use only at the lowest resolution (8^3 = 512 tokens)."""
    def __init__(self, ch, heads=4):
        super().__init__()
        assert ch % heads == 0
        self.heads = heads
        self.norm = nn.GroupNorm(gn_groups(ch), ch)
        self.qkv = nn.Conv3d(ch, 3 * ch, 1)
        self.proj = nn.Conv3d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, D, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        # Reshape to (B, heads, head_dim, N) where N = D*H*W
        def split(t):
            return t.reshape(B, self.heads, C // self.heads, D * H * W)
        q, k, v = split(q), split(k), split(v)
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) / math.sqrt(C // self.heads)
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(B, C, D, H, W)
        return x + self.proj(out)


class WindowedAttention3D(nn.Module):
    """Non-overlapping 3D windowed self-attention.
    At 16^3 spatial with window=8: 8 windows, 512 tokens per window."""
    def __init__(self, ch, heads=4, window=8):
        super().__init__()
        assert ch % heads == 0
        self.heads = heads
        self.window = window
        self.norm = nn.GroupNorm(gn_groups(ch), ch)
        self.qkv = nn.Conv3d(ch, 3 * ch, 1)
        self.proj = nn.Conv3d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, D, H, W = x.shape
        W_ = self.window
        assert D % W_ == 0 and H % W_ == 0 and W % W_ == 0, \
            f"spatial dims {(D,H,W)} must be divisible by window {W_}"
        h = self.norm(x)
        qkv = self.qkv(h)
        # (B, 3C, D, H, W) -> windows of (D/W_, H/W_, W/W_) of size W_^3
        # Reshape to (B*nW, 3C, W_, W_, W_) where nW = (D*H*W)/W_^3
        nD, nH, nW = D // W_, H // W_, W // W_
        qkv = qkv.reshape(B, 3 * C, nD, W_, nH, W_, nW, W_)
        qkv = qkv.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        # now (B, nD, nH, nW, 3C, W_, W_, W_)
        qkv = qkv.reshape(B * nD * nH * nW, 3 * C, W_, W_, W_)
        q, k, v = qkv.chunk(3, dim=1)
        def split(t):
            return t.reshape(-1, self.heads, C // self.heads, W_ * W_ * W_)
        q, k, v = split(q), split(k), split(v)
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) / math.sqrt(C // self.heads)
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        # back to spatial windows
        out = out.reshape(B * nD * nH * nW, C, W_, W_, W_)
        out = out.reshape(B, nD, nH, nW, C, W_, W_, W_)
        out = out.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        out = out.reshape(B, C, D, H, W)
        return x + self.proj(out)


# ---------------------------------------------------------------------------
# IC stem (learned downsampler 64^3 -> 32^3)
# ---------------------------------------------------------------------------

class ICStem(nn.Module):
    """1 stride-2 conv to take (B, 2, 64, 64, 64) -> (B, 32, 32, 32, 32)."""
    def __init__(self, in_ch=2, mid_ch=16, out_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(mid_ch, out_ch, 4, stride=2, padding=1),
            nn.SiLU(),
        )
        self.out_ch = out_ch

    def forward(self, ic_delta, ic_vbv):
        x = torch.cat([ic_delta, ic_vbv], dim=1)
        return self.net(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class Downsample3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv3d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x):
        return self.op(x)


class Upsample3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.ConvTranspose3d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x):
        return self.op(x)


class LDMUNet3D(nn.Module):
    def __init__(self,
                 latent_ch=8,
                 ic_stem_out=32,
                 base_ch=48,
                 ch_mults=(1, 2, 4),
                 n_res_per_level=2,
                 cond_dim=256,
                 attn_levels=(1, 2),    # windowed at level 1 (16^3), global at level 2 (8^3)
                 window=8,
                 heads=4,
                 n_params=4):
        super().__init__()
        self.latent_ch = latent_ch
        self.cond = ConditioningEmbedding(n_params=n_params, out_dim=cond_dim)
        self.ic_stem = ICStem(in_ch=2, mid_ch=16, out_ch=ic_stem_out)

        # Input projection
        in_total = latent_ch + ic_stem_out
        self.proj_in = nn.Conv3d(in_total, base_ch * ch_mults[0], 3, padding=1)

        # Encoder
        self.downs = nn.ModuleList()
        chs = [base_ch * m for m in ch_mults]
        self.chs = chs

        # store skip-channel counts so the decoder can use them
        skip_chs = []
        cur_ch = chs[0]
        for level, ch in enumerate(chs):
            blocks = nn.ModuleList()
            for _ in range(n_res_per_level):
                blocks.append(ResBlock3D(cur_ch, ch, cond_dim))
                cur_ch = ch
            attn = None
            if level in attn_levels:
                # level 1 -> windowed, level 2 (bottleneck) -> global
                # level numbering from 0 at top of encoder
                if level == len(chs) - 1:
                    attn = GlobalAttention3D(cur_ch, heads=heads)
                else:
                    attn = WindowedAttention3D(cur_ch, heads=heads, window=window)
            is_last = (level == len(chs) - 1)
            self.downs.append(nn.ModuleList([
                blocks,
                attn if attn is not None else nn.Identity(),
                Downsample3D(cur_ch) if not is_last else nn.Identity(),
            ]))
            skip_chs.append(cur_ch)

        # Bottleneck (extra blocks at lowest level)
        self.mid_block1 = ResBlock3D(cur_ch, cur_ch, cond_dim)
        self.mid_attn   = GlobalAttention3D(cur_ch, heads=heads)
        self.mid_block2 = ResBlock3D(cur_ch, cur_ch, cond_dim)

        # Decoder
        self.ups = nn.ModuleList()
        for level in reversed(range(len(chs))):
            ch = chs[level]
            # is_bottom_iter: first iteration (level == len(chs)-1, deepest level)
            is_bottom_iter = (level == len(chs) - 1)
            # Upsample takes us FROM previous (deeper) cur_ch to this level
            up = Upsample3D(cur_ch) if not is_bottom_iter else nn.Identity()
            blocks = nn.ModuleList()
            for r in range(n_res_per_level):
                in_c = cur_ch + (skip_chs[level] if r == 0 else 0)
                blocks.append(ResBlock3D(in_c, ch, cond_dim))
                cur_ch = ch
            attn = None
            if level in attn_levels:
                if level == len(chs) - 1:
                    attn = GlobalAttention3D(cur_ch, heads=heads)
                else:
                    attn = WindowedAttention3D(cur_ch, heads=heads, window=window)
            self.ups.append(nn.ModuleList([
                blocks,
                attn if attn is not None else nn.Identity(),
                up,
            ]))

        # Output head
        self.norm_out = nn.GroupNorm(gn_groups(cur_ch), cur_ch)
        self.conv_out = nn.Conv3d(cur_ch, latent_ch, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, z_noisy, ic_delta, ic_vbv, params, redshift, sigma):
        """
        z_noisy   : (B, 8, 32, 32, 32)
        ic_delta  : (B, 1, 64, 64, 64)
        ic_vbv    : (B, 1, 64, 64, 64)
        params    : (B, 4)
        redshift  : (B,)
        sigma     : (B,)
        returns   : (B, 8, 32, 32, 32)
        """
        cond = self.cond(params, redshift, sigma)
        ic_feat = self.ic_stem(ic_delta, ic_vbv)   # (B, 32, 32, 32, 32)
        x = torch.cat([z_noisy, ic_feat], dim=1)
        x = self.proj_in(x)

        skips = []
        for (blocks, attn, down) in self.downs:
            for b in blocks:
                x = b(x, cond)
            x = attn(x)
            skips.append(x)
            x = down(x)

        x = self.mid_block1(x, cond)
        x = self.mid_attn(x)
        x = self.mid_block2(x, cond)

        for (blocks, attn, up) in self.ups:
            x = up(x)
            skip = skips.pop()
            for i, b in enumerate(blocks):
                if i == 0:
                    x = torch.cat([x, skip], dim=1)
                x = b(x, cond)
            x = attn(x)

        x = F.silu(self.norm_out(x))
        return self.conv_out(x)


if __name__ == '__main__':
    # Smoke test
    model = LDMUNet3D()
    n = sum(p.numel() for p in model.parameters())
    print(f'LDMUNet3D params: {n/1e6:.2f} M')

    B = 2
    z = torch.randn(B, 8, 32, 32, 32)
    ic_d = torch.randn(B, 1, 64, 64, 64)
    ic_v = torch.randn(B, 1, 64, 64, 64)
    params = torch.randn(B, 4)
    redshift = torch.tensor([10.0, 8.0])
    sigma = torch.tensor([0.5, 2.0])
    out = model(z, ic_d, ic_v, params, redshift, sigma)
    print('output shape:', out.shape)
    # Check that initial output is near zero (zero-init final conv)
    print('initial output abs mean:', out.abs().mean().item())
