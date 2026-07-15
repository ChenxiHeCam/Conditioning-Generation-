# Architecture flow chart of the v5+v4logk pipeline (matplotlib, no GPU needed).
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.size': 10})

fig, ax = plt.subplots(figsize=(13.5, 7.2))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 7.2); ax.axis('off')

C = {'input': '#dbeafe', 'ldm': '#fde68a', 'vae': '#bbf7d0',
     'res': '#fbcfe8', 'out': '#e5e7eb', 'note': '#ffffff'}


def box(x, y, w, h, text, color, fs=10, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
                                fc=color, ec='k', lw=lw))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs)


def arrow(x1, y1, x2, y2, text='', curve=0.0, fs=9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 connectionstyle=f'arc3,rad={curve}',
                 arrowstyle='-|>', mutation_scale=16, lw=1.5, color='k'))
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.18, text, ha='center', fontsize=fs,
                style='italic')


# ---- inputs (left column) ----
box(0.3, 5.6, 2.5, 0.9, 'IC density $\\delta(\\vec x)$\n$64^3$ patch', C['input'])
box(0.3, 4.4, 2.5, 0.9, 'IC velocity $v_{bv}(\\vec x)$\n$64^3$ patch', C['input'])
box(0.3, 3.2, 2.5, 0.9, 'astro params $\\theta\\in\\mathbb{R}^4$\n+ redshift $z$', C['input'])
box(0.3, 1.4, 2.5, 0.9, 'Gaussian noise\n$z_T\\sim\\mathcal{N}(0,\\sigma_{max}^2)$\n$(8,16^3)$', C['input'], fs=9)

# ---- LDM ----
box(3.8, 2.6, 3.0, 2.6,
    'LDM denoiser U-Net\n(EDM, 5.7M params)\n\n'
    'base 48, mults (1,2)\nwindowed attn @ $8^3$\nglobal attn @ $4^3$\n'
    'FiLM($\\theta$, $z$, $\\log\\sigma$)\n30 Heun steps', C['ldm'])
box(3.8, 5.5, 3.0, 0.8, 'IC stem: 2$\\times$ stride-2 conv\n$64^3\\to16^3$ features', C['ldm'], fs=9)

arrow(2.8, 6.05, 4.0, 6.3, '')
arrow(2.8, 4.85, 3.9, 5.6, '')
arrow(5.3, 5.5, 5.3, 5.25, 'concat')
arrow(2.8, 3.65, 3.8, 3.65, 'FiLM')
arrow(2.8, 1.85, 4.4, 2.6, '')

# ---- VAE decoder ----
box(7.6, 3.3, 2.4, 1.5,
    'VAE decoder\n(42.6M, frozen)\n$(8,16^3)\\to 64^3$\n8$\\times$ compression', C['vae'])
arrow(6.8, 3.9, 7.6, 4.0, '$\\hat z\\,\\sigma_{lat}+\\mu_{lat}$')

# ---- residual ----
box(10.7, 3.3, 2.5, 1.5,
    'Cond. residual U-Net\n(6.1M)  $\\hat r(\\vec x)$\nFiLM($\\theta$, $z$)\n'
    'zero-init output', C['res'])
arrow(10.0, 4.05, 10.7, 4.05, '$x_{low}$')
# IC skip to residual
arrow(1.55, 5.6, 11.9, 4.8, '', curve=-0.25)
ax.text(7.2, 6.6, 'IC fields + params skip directly to the residual stage',
        fontsize=8.5, style='italic')

# ---- output ----
box(10.7, 1.0, 2.5, 1.3,
    'final patch\n$x_{low}+\\hat r$  ($64^3$, mK)', C['out'])
arrow(11.95, 3.3, 11.95, 2.3, '+')

box(7.0, 0.6, 2.9, 1.2,
    'Hann-blended tiling\n343 patches, 50% overlap\n$\\to 256^3$ cube (~30 s)', C['out'], fs=9)
arrow(10.7, 1.4, 9.9, 1.3, '')

ax.text(0.3, 0.25,
        'Training: VAE (149 ep) $\\to$ LDM on frozen latents (250 ep) $\\to$ residual stage-1 VAE pairs (80 ep) '
        '$\\to$ stage-2 60/40 VAE/LDM-sampled pairs (30 ep, log-$k$ ratio PS loss)',
        fontsize=9)

ax.set_title('21cm latent diffusion emulator — inference pipeline (v5 + v4logk)',
             fontsize=13)
for ext in ('png', 'pdf'):
    fig.savefig(f'D:/Astro/repo/results_local/flowchart_pipeline.{ext}',
                dpi=250, bbox_inches='tight')
print('saved flowchart_pipeline.png/pdf')
