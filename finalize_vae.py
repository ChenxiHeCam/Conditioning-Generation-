
import os, sys, argparse, torch, numpy as np
sys.path.insert(0,"/root/21cm_gen")
from dataset import T21Dataset
from models.vae import VAE3D
from torch.utils.data import DataLoader, ConcatDataset

ap=argparse.ArgumentParser()
ap.add_argument("--ckpt_in", required=True)
ap.add_argument("--ckpt_out", required=True)
ap.add_argument("--latent_ch", type=int, required=True)
ap.add_argument("--base_ch", type=int, default=128)
ap.add_argument("--ch_mults", nargs="+", type=int, default=[1,2])
ap.add_argument("--latent_spatial", type=int, default=16)
args=ap.parse_args()
dev="cuda"

ck=torch.load(args.ckpt_in, map_location=dev)
vae=VAE3D(in_ch=1, latent_ch=args.latent_ch, base_ch=args.base_ch, ch_mults=tuple(args.ch_mults)).to(dev).eval()
vae.load_state_dict(ck["model"])

# accumulate per-channel mean/std on train set
sets=[]
for root in ["/root/autodl-tmp/ASR21cm/varying_IC","/root/autodl-tmp/ASR21cm/varying_astro"]:
    try:
        sets.append(T21Dataset(root,64,redshifts=[8,9,10,11,12],split="train",
                               load_ic=False,max_per_z=50,primary_z=10,holdout_frac=0.25))
    except Exception as e: print("skip",root,e)
ds=ConcatDataset(sets)
ld=DataLoader(ds,batch_size=8,shuffle=False,num_workers=0)
C=args.latent_ch
sm=torch.zeros(C,device=dev); sm2=torch.zeros(C,device=dev); n=0
with torch.no_grad():
    for i,b in enumerate(ld):
        x=b["patch"].to(dev)
        mu,_=vae.encoder(x)
        f=mu.transpose(0,1).reshape(C,-1)
        sm+=f.sum(1); sm2+=(f**2).sum(1); n+=f.shape[1]
        if (i+1)%80==0: print(f"  batch {i+1}/{len(ld)}")
mean=(sm/n).cpu(); var=(sm2/n - (sm/n)**2).clamp_min(0).cpu(); std=var.sqrt()
print(f"per-ch mean range [{mean.min():.3f},{mean.max():.3f}], std range [{std.min():.3f},{std.max():.3f}]")

out=dict(ck)
out["latent_mean"]=mean; out["latent_std"]=std
out["model_config"]=dict(in_ch=1,latent_ch=args.latent_ch,base_ch=args.base_ch,
                         ch_mults=list(args.ch_mults),downsamples=len(args.ch_mults),
                         latent_shape=[args.latent_ch,args.latent_spatial,args.latent_spatial,args.latent_spatial])
torch.save(out, args.ckpt_out)
print("saved:", args.ckpt_out)
