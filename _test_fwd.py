import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")
sys.path.insert(0, "/root/21cm_gen")
import torch

dev = "cuda"
print("torch", torch.__version__, "device:", torch.cuda.get_device_name(0))

print("\n--- pixel forward ---", flush=True)
from ldm_unet import LDMUNet3D
m = LDMUNet3D(latent_ch=1, ic_stem_downsamples=0, base_ch=32, ch_mults=(1, 2, 4, 4),
              attn_levels=(2, 3)).to(dev).eval()
print(f"  params {sum(p.numel() for p in m.parameters())/1e6:.1f}M", flush=True)
B = 4
z    = torch.randn(B, 1, 64, 64, 64, device=dev)
ic_d = torch.randn(B, 1, 64, 64, 64, device=dev)
ic_v = torch.randn(B, 1, 64, 64, 64, device=dev)
par  = torch.randn(B, 4, device=dev)
zr   = torch.full((B,), 10.0, device=dev)
sigma = torch.full((B,), 1.0, device=dev)
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out = m(z, ic_d, ic_v, par, zr, sigma)
torch.cuda.synchronize()
print(f"  fwd OK in {time.time()-t0:.1f}s, shape={out.shape}", flush=True)

print("\n--- vqgan forward ---", flush=True)
from models.vqgan3d import VQGAN3D
g = VQGAN3D(in_ch=1, embed_dim=8, n_embed=1024, base_ch=128).to(dev).eval()
print(f"  params {sum(p.numel() for p in g.parameters())/1e6:.1f}M", flush=True)
x = torch.randn(4, 1, 64, 64, 64, device=dev)
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    recon, commit, idx = g(x)
torch.cuda.synchronize()
print(f"  fwd OK in {time.time()-t0:.1f}s, shape={recon.shape}", flush=True)

print("\n--- stylegan forward ---", flush=True)
from models.stylegan3d import StyleGenerator3D
G = StyleGenerator3D(base_ch=32).to(dev).eval()
print(f"  params {sum(p.numel() for p in G.parameters())/1e6:.1f}M", flush=True)
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out = G(ic_d, ic_v, par, zr)
torch.cuda.synchronize()
print(f"  fwd OK in {time.time()-t0:.1f}s, shape={out.shape}", flush=True)
print("\nALL MODELS FORWARD OK")
