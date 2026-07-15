"""
Quick loss curve viewer. Run on login node (no GPU needed).
    python plot_loss.py --log_dir checkpoints/flow
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(log_path, out_path):
    data = np.loadtxt(log_path, delimiter=',', skiprows=1)
    if data.ndim == 1:
        data = data[None]
    epochs, tr, vl = data[:, 0], data[:, 1], data[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, yscale in zip(axes, ['linear', 'log']):
        ax.plot(epochs, tr, label='Train')
        ax.plot(epochs, vl, label='Val', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_yscale(yscale)
        ax.set_title(f'Loss ({yscale} scale)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Print current stats
    best_val_ep = int(epochs[np.argmin(vl)])
    print(f"  Latest epoch : {int(epochs[-1])}")
    print(f"  Latest train : {tr[-1]:.6f}")
    print(f"  Latest val   : {vl[-1]:.6f}")
    print(f"  Best val     : {vl.min():.6f}  (epoch {best_val_ep})")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"  Saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log_dir', default='/home/ch2067/rds/hpc-work/21cm_gen/checkpoints/flow')
    args = p.parse_args()

    for name in ['log.csv']:
        path = os.path.join(args.log_dir, name)
        if os.path.exists(path):
            out = path.replace('.csv', '_plot.png')
            print(f"\n{path}")
            plot(path, out)
        else:
            print(f"Not found: {path}")


if __name__ == '__main__':
    main()
