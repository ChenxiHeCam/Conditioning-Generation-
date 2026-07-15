"""Histogram of relative MSE and bias across test set, per compression chain.

bias = std_true / std_gen - 1

Reads .npz files from infer_baselines.py outputs:
  /path/to/eval_results/ff_<tag>_test_z*/000_T21_*.npz

Usage:
  python plot_bias_hist.py --eval_root /path/to/eval_results --out plots/
"""
import os, argparse, glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def collect_chain(eval_root, tag):
    """Read all per-cube npz files for one chain (across z's)."""
    pattern = os.path.join(eval_root, f'ff_{tag}_test_z*', '*.npz')
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    rel_mse, bias, z_list = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        t = d['true_norm']; g = d['gen_norm']
        rel = float(((t - g)**2).sum() / max((t**2).sum(), 1e-30))
        st = float(t.std()); sg = float(g.std()) + 1e-30
        rel_mse.append(rel); bias.append(st / sg - 1.0)
        # extract z from folder name ff_<tag>_test_z<N>
        z_list.append(int(os.path.basename(os.path.dirname(f)).split('_z')[-1].split('_')[0]))
    return dict(tag=tag,
                rel_mse=np.array(rel_mse),
                bias=np.array(bias),
                z=np.array(z_list),
                files=files)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eval_root', default='/home/ch2067/rds/hpc-work/21cm_gen/eval_results')
    p.add_argument('--out', default='plots/')
    p.add_argument('--tags', nargs='+', default=['2x', '4x', '8x', '16x'])
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    chains = []
    for tag in args.tags:
        d = collect_chain(args.eval_root, tag)
        if d is None:
            print(f"WARN: no data for {tag}, skipping")
            continue
        chains.append(d)
        print(f"{tag}: {len(d['rel_mse'])} cubes  median rel_MSE={np.median(d['rel_mse']):.3f}  bias median={np.median(d['bias']):+.3f}")

    if not chains: return

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    # ---------- Figure 1: rel_MSE histogram ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    edges = np.linspace(0, max(0.05, np.percentile(np.concatenate([c['rel_mse'] for c in chains]), 95)), 40)
    for i, c in enumerate(chains):
        ax.hist(c['rel_mse'], bins=edges, density=True, alpha=0.5,
                color=colors[i], label=f"{c['tag']} (med={np.median(c['rel_mse']):.3f})")
    ax.set_xlabel('relative MSE'); ax.set_ylabel('density')
    ax.set_title(f'rel_MSE distribution across test set (n={len(chains[0]["rel_mse"])} cubes/chain)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'rel_mse_hist.png'), dpi=150)
    plt.savefig(os.path.join(args.out, 'rel_mse_hist.pdf'))
    plt.close()
    print(f"saved {args.out}/rel_mse_hist.png")

    # ---------- Figure 2: bias histogram ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    all_bias = np.concatenate([c['bias'] for c in chains])
    edges = np.linspace(np.percentile(all_bias, 2), np.percentile(all_bias, 98), 40)
    for i, c in enumerate(chains):
        ax.hist(c['bias'], bins=edges, density=True, alpha=0.5,
                color=colors[i], label=f"{c['tag']} (med={np.median(c['bias']):+.3f})")
    ax.axvline(0, color='k', ls=':', alpha=0.5, label='unbiased')
    ax.set_xlabel(r'bias = $\sigma_{\rm true}/\sigma_{\rm gen} - 1$'); ax.set_ylabel('density')
    ax.set_title('Variance bias across test set')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'bias_hist.png'), dpi=150)
    plt.savefig(os.path.join(args.out, 'bias_hist.pdf'))
    plt.close()
    print(f"saved {args.out}/bias_hist.png")

    # ---------- Per-z breakdown JSON ----------
    summary = {}
    for c in chains:
        per_z = {}
        for z in sorted(set(c['z'].tolist())):
            mask = c['z'] == z
            per_z[int(z)] = dict(
                n=int(mask.sum()),
                rel_mse_median=float(np.median(c['rel_mse'][mask])),
                rel_mse_mean=float(np.mean(c['rel_mse'][mask])),
                bias_median=float(np.median(c['bias'][mask])),
                bias_mean=float(np.mean(c['bias'][mask])),
            )
        summary[c['tag']] = dict(per_z=per_z, overall=dict(
            rel_mse_median=float(np.median(c['rel_mse'])),
            bias_median=float(np.median(c['bias'])),
        ))
    with open(os.path.join(args.out, 'rel_mse_bias_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"saved {args.out}/rel_mse_bias_summary.json")


if __name__ == '__main__':
    main()
