"""Convert all .mat T21+IC cubes to .npy in parallel.

Run on CSD3 via sbatch icelake CPU partition.
Output: same dirname but files end in .npy alongside .mat.
"""
import os, sys, glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import scipy.io
import numpy as np

DATA = '/home/ch2067/rds/hpc-work/ASR21cm/datasets'

PATTERNS = [
    f'{DATA}/varying_IC/T21_cubes/*.mat',
    f'{DATA}/varying_IC/IC_cubes/*.mat',
    f'{DATA}/varying_astro/T21_cubes/*.mat',
    f'{DATA}/varying_astro/IC_cubes/*.mat',
]

def convert(mat_path):
    npy = mat_path[:-4] + '.npy'
    if os.path.exists(npy):
        return 'skip', mat_path
    try:
        d = scipy.io.loadmat(mat_path)
        # find the main array (skip __ keys)
        keys = [k for k in d.keys() if not k.startswith('__')]
        if not keys:
            return 'no-array', mat_path
        # take the biggest numeric array
        arr = None
        for k in keys:
            v = d[k]
            if isinstance(v, np.ndarray) and v.dtype.kind in 'fi':
                if arr is None or v.size > arr.size:
                    arr = v
        if arr is None:
            return 'no-numeric', mat_path
        np.save(npy, arr.astype(np.float32))
        return 'ok', mat_path
    except Exception as e:
        return f'err:{type(e).__name__}:{str(e)[:60]}', mat_path

def main():
    files = []
    for p in PATTERNS:
        files.extend(glob.glob(p))
    print(f'Total .mat files: {len(files)}', flush=True)
    if not files:
        return
    n_ok = n_skip = n_err = 0
    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))
    print(f'Using {n_workers} workers', flush=True)
    t0 = __import__('time').time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for i, (status, path) in enumerate(ex.map(convert, files, chunksize=4)):
            if status == 'ok':       n_ok += 1
            elif status == 'skip':   n_skip += 1
            else:
                n_err += 1
                if n_err <= 10:
                    print(f'ERR: {status} -- {path}', flush=True)
            if (i + 1) % 200 == 0:
                dt = __import__('time').time() - t0
                rate = (i + 1) / dt
                eta = (len(files) - i - 1) / rate
                print(f'  {i+1}/{len(files)} ok={n_ok} skip={n_skip} err={n_err} '
                      f'rate={rate:.1f}/s eta={eta/60:.1f}min', flush=True)
    print(f'DONE ok={n_ok} skip={n_skip} err={n_err}')

if __name__ == '__main__':
    main()
