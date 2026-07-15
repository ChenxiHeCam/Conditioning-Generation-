# Compression-ablation 4-panel figure from eval_models_pub.py metrics json.
#
#   python plot_compression_ablation.py \
#     --metrics /root/autodl-tmp/model_compare_figs/model_metrics.json \
#     --mapping "1:pixel" "4:4x latent" "8:8x latent" "16:16x latent" \
#     --out_dir /root/autodl-tmp/model_compare_figs
import os, argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.25,
                     'savefig.dpi': 250, 'lines.linewidth': 2})

p = argparse.ArgumentParser()
p.add_argument('--metrics', required=True)
p.add_argument('--mapping', nargs='+', required=True, help='compression:label pairs')
p.add_argument('--out_dir', required=True)
args = p.parse_args()

summ = json.load(open(args.metrics))['summary']
comp, rel, ps_err, bi_err, secs = [], [], [], [], []
for spec in args.mapping:
    c, label = spec.split(':', 1)
    m = summ[label]
    comp.append(float(c))
    rel.append(m['rel_mse'])
    b = m['ps_bands']
    ps_err.append(100 * np.mean([abs(b['large'] - 1), abs(b['mid'] - 1),
                                 abs(b['small'] - 1)]))
    bi_err.append(100 * abs(m['bispec_eq_mean_ratio'] - 1))
    secs.append(m['sec_per_cube'])

fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
panels = [(rel, 'median relative MSE', 'log'),
          (ps_err, r'PS band error  $|\Delta^2_{\rm ratio}-1|$  [%]', 'linear'),
          (bi_err, r'equilateral bispectrum error  [%]', 'linear'),
          (secs, r'inference per $256^3$ cube  [s]', 'log')]
for ax, (y, ylab, yscale) in zip(axes, panels):
    ax.plot(comp, y, 'o-', color='tab:blue')
    ax.set_xscale('log', base=2)
    ax.set_yscale(yscale)
    ax.set_xticks(comp)
    ax.set_xticklabels([f'{int(c)}×' for c in comp])
    ax.set_xlabel('VAE compression factor')
    ax.set_ylabel(ylab)
fig.suptitle('Compression ablation — quality and speed vs latent compression', y=1.04)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(args.out_dir, f'M7_compression_ablation.{ext}'),
                bbox_inches='tight')
print('saved M7_compression_ablation')
