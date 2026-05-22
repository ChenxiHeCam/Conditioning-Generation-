"""
3D Variational Autoencoder for 21cm brightness temperature fields.
Compresses (1, 64, 64, 64) patches to (latent_ch, 16, 16, 16) latents.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_norm(ch):
    return nn.GroupNorm(min(8, ch), ch)


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            make_norm(in_ch), nn.SiLU(),
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            make_norm(out_ch), nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class Encoder3D(nn.Module):
    """64^3 → 16^3 with 2 downsample stages."""
    def __init__(self, in_ch=1, latent_ch=4, base_ch=64, ch_mults=(1, 2)):
        super().__init__()
        self.init_conv = nn.Conv3d(in_ch, base_ch, 3, padding=1)

        self.stages = nn.ModuleList()
        ch = base_ch
        for mult in ch_mults:
            out = base_ch * mult
            self.stages.append(nn.Sequential(
                ResBlock3D(ch, out),
                ResBlock3D(out, out),
                nn.Conv3d(out, out, 4, stride=2, padding=1),   # 2× down
            ))
            ch = out

        self.mid = nn.Sequential(ResBlock3D(ch, ch), ResBlock3D(ch, ch))
        self.to_latent = nn.Conv3d(ch, latent_ch * 2, 1)   # outputs mean & logvar

    def forward(self, x):
        h = self.init_conv(x)
        for stage in self.stages:
            h = stage(h)
        h = self.mid(h)
        mean, logvar = self.to_latent(h).chunk(2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        return mean, logvar


class Decoder3D(nn.Module):
    """16^3 → 64^3 mirror of Encoder."""
    def __init__(self, out_ch=1, latent_ch=4, base_ch=64, ch_mults=(1, 2)):
        super().__init__()
        # Start at deepest channel count
        ch = base_ch * ch_mults[-1]
        self.init_conv = nn.Conv3d(latent_ch, ch, 3, padding=1)
        self.mid = nn.Sequential(ResBlock3D(ch, ch), ResBlock3D(ch, ch))

        self.stages = nn.ModuleList()
        for mult in reversed(ch_mults):
            out = base_ch * mult
            self.stages.append(nn.Sequential(
                nn.ConvTranspose3d(ch, out, 4, stride=2, padding=1),   # 2× up
                ResBlock3D(out, out),
                ResBlock3D(out, out),
            ))
            ch = out

        self.out = nn.Sequential(
            make_norm(ch), nn.SiLU(),
            nn.Conv3d(ch, out_ch, 3, padding=1),
        )

    def forward(self, z):
        h = self.mid(self.init_conv(z))
        for stage in self.stages:
            h = stage(h)
        return self.out(h)


class VAE3D(nn.Module):
    def __init__(self, in_ch=1, latent_ch=4, base_ch=64, ch_mults=(1, 2)):
        super().__init__()
        self.encoder = Encoder3D(in_ch, latent_ch, base_ch, ch_mults)
        self.decoder = Decoder3D(in_ch, latent_ch, base_ch, ch_mults)

    # ------------------------------------------------------------------
    def encode(self, x):
        mean, logvar = self.encoder(x)
        z = mean + torch.randn_like(mean) * (0.5 * logvar).exp()
        return z, mean, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, mean, logvar

    # ------------------------------------------------------------------
    @staticmethod
    def kl_loss(mean, logvar):
        return -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

    @staticmethod
    def recon_loss(recon, target):
        return F.mse_loss(recon, target)

    def loss(self, x, kl_weight=1e-4):
        recon, mean, logvar = self(x)
        l_recon = self.recon_loss(recon, x)
        l_kl    = self.kl_loss(mean, logvar)
        return l_recon + kl_weight * l_kl, l_recon, l_kl
