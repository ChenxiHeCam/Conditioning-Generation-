"""
Dataset for 21cm brightness temperature fields.

Supports two data formats:
  varying_astro  — fixed IC, varying astrophysical parameters
                   T21:  T21_cube_z10__diffusion_NNNN.mat
                   IC:   delta1000.mat / vbv1000.mat  (shared)
                   params: parameters__diffusion_NNNN.params

  varying_IC     — varying ICs, fixed astrophysical parameters
                   T21:  T21_cube_z{Z}__Npix256_IC{N}.mat
                   IC:   delta_Npix256_IC{N}.mat / vbv_Npix256_IC{N}.mat
                   params: fixed (from matlab script defaults)

Format is auto-detected from the T21 filenames.
Datasets can be combined via torch.utils.data.ConcatDataset.
"""
import os
import re
import torch
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset

try:
    import scipy.io as sio
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    import h5py
    H5PY_OK = True
except ImportError:
    H5PY_OK = False


# -------------------------------------------------------------------------
# Parameter spec
# -------------------------------------------------------------------------

PARAM_KEYS = ['MyStar_II', 'MyVc', 'MyFX', 'DelayParam', 'redshift']
PARAM_DIM  = len(PARAM_KEYS)   # 5

# Parameters that span multiple orders of magnitude — apply log10 before normalising
LOG_PARAMS = {'MyStar_II', 'MyFX'}

# Fixed astrophysical parameters used in varying_IC simulations
# (from run_diffusion_sim_IC.m: fstarII=0.05, circ_V=4.2, f_X=1.0, delay=0.75)
VARYING_IC_FIXED_PARAMS = {
    'MyStar_II':  0.05,
    'MyVc':       4.2,
    'MyFX':       1.0,
    'DelayParam': 0.75,
}


# -------------------------------------------------------------------------
# I/O helpers
# -------------------------------------------------------------------------

def load_mat(path):
    """Load a field cube, preferring pre-converted .npy for speed."""
    npy = path.replace('.mat', '.npy')
    if os.path.exists(npy):
        return np.load(npy)
    # Fall back to parsing the original .mat
    if SCIPY_OK:
        try:
            data = sio.loadmat(path)
            for key in data:
                if key.startswith('__'):
                    continue
                arr = np.array(data[key], dtype=np.float32)
                arr = arr.squeeze()
                if arr.ndim == 3:
                    return arr
        except Exception:
            pass
    if H5PY_OK:
        try:
            with h5py.File(path, 'r') as f:
                for key in f:
                    arr = np.array(f[key], dtype=np.float32)
                    arr = arr.squeeze()
                    if arr.ndim == 3:
                        return arr
        except OSError:
            pass
    raise IOError(f"Cannot load {path}")


def parse_params(path):
    """Parse a .params file into a float dict."""
    params = {}
    with open(path, 'r') as f:
        for line in f:
            if '=' in line:
                k, v = line.split('=', 1)
                try:
                    params[k.strip()] = float(v.strip())
                except ValueError:
                    pass
    return params


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------

class T21Dataset(Dataset):
    def __init__(self, data_root, patch_size=64, redshifts=None,
                 split='train', val_frac=0.1, test_frac=0.1, seed=42,
                 npix=256, ic_cache_size=60, patches_per_cube=1,
                 load_ic=True):
        """
        data_root:       root directory containing T21_cubes/, IC_cubes/, parameters/
        patch_size:      spatial size of cubic patches to extract
        redshifts:       list of int redshifts to include (None = all)
        split:           'train', 'val', or 'test'
        val_frac:        fraction held out per redshift for validation
        test_frac:       fraction held out per redshift for test (stratified by redshift)
        seed:            random seed for split
        npix:            resolution to use for varying_IC files (256 or 512)
        ic_cache_size:   max number of IC fields to keep in RAM simultaneously
        patches_per_cube: number of random patches drawn from each file per epoch
        load_ic:         if False, skip loading IC fields (zeros returned); use for VAE training
        """
        self.data_root       = data_root
        self.patch_size      = patch_size
        self.patches_per_cube = patches_per_cube
        self.load_ic         = load_ic
        self.npix            = npix
        self.t21_dir      = os.path.join(data_root, 'T21_cubes')
        self.param_dir    = os.path.join(data_root, 'parameters')
        self.ic_dir       = os.path.join(data_root, 'IC_cubes')
        self._ic_cache    = OrderedDict()
        self._ic_cache_max = ic_cache_size

        # Collect and filter T21 files
        all_files = sorted(f for f in os.listdir(self.t21_dir)
                           if f.endswith('.mat') and not f.startswith('.') and 'files_to' not in f)
        if redshifts:
            all_files = [f for f in all_files if self._redshift_of(f) in redshifts]

        # Auto-detect dataset format
        self.fmt = self._detect_format(all_files)
        print(f"Dataset format: {self.fmt}  ({len(all_files)} files)")

        # For varying_IC keep only the requested Npix resolution
        if self.fmt == 'varying_IC':
            all_files = [f for f in all_files if f'Npix{npix}' in f]
            print(f"  After Npix{npix} filter: {len(all_files)} files")

        # Stratified 3-way split by redshift so every z is represented in val/test.
        # Within each redshift group: test_frac → test, val_frac → val, rest → train.
        from collections import defaultdict
        groups = defaultdict(list)
        for f in all_files:
            groups[self._redshift_of(f)].append(f)

        rng = np.random.default_rng(seed)
        train_files, val_files, test_files = [], [], []
        for z in sorted(groups):
            g = [groups[z][i] for i in rng.permutation(len(groups[z]))]
            n_test = max(1, int(len(g) * test_frac))
            n_val  = max(1, int(len(g) * val_frac))
            test_files.extend(g[:n_test])
            val_files.extend(g[n_test:n_test + n_val])
            train_files.extend(g[n_test + n_val:])

        self.files = {'train': train_files, 'val': val_files, 'test': test_files}[split]
        print(f"  split={split}: {len(self.files)} files  "
              f"(train={len(train_files)} val={len(val_files)} test={len(test_files)})")

        # Pre-build IC paths for varying_astro shared IC
        if self.fmt == 'varying_astro':
            self._init_shared_ic()

        # Load params and compute normalisation stats
        self._load_all_params()
        self._compute_norm_stats()

    # ------------------------------------------------------------------
    # Format detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_format(files):
        for f in files:
            if re.search(r'_IC\d+\.mat$', f):
                return 'varying_IC'
            if 'diffusion_' in f:
                return 'varying_astro'
        return 'varying_astro'

    @staticmethod
    def _redshift_of(fname):
        m = re.search(r'_z(\d+)_', fname) or re.search(r'_z(\d+)__', fname)
        return int(m.group(1)) if m else None

    # ------------------------------------------------------------------
    # Shared IC (varying_astro)
    # ------------------------------------------------------------------

    def _init_shared_ic(self):
        """Load single shared IC for varying_astro (delta + vbv)."""
        self._shared_delta = None
        self._shared_vbv   = None
        if not os.path.isdir(self.ic_dir):
            return
        files = os.listdir(self.ic_dir)
        d_files = [f for f in files if 'delta' in f.lower() and f.endswith('.mat')]
        v_files = [f for f in files if 'vbv'   in f.lower() and f.endswith('.mat')]
        if d_files:
            arr = load_mat(os.path.join(self.ic_dir, d_files[0]))
            self._shared_delta = (arr - arr.mean()) / (arr.std() + 1e-8)
            print(f"  Shared IC delta: {self._shared_delta.shape}")
        if v_files:
            arr = load_mat(os.path.join(self.ic_dir, v_files[0]))
            self._shared_vbv = (arr - arr.mean()) / (arr.std() + 1e-8)
            print(f"  Shared IC vbv:   {self._shared_vbv.shape}")

    # ------------------------------------------------------------------
    # IC loading for varying_IC (lazy with LRU cache)
    # ------------------------------------------------------------------

    def _ic_paths_for(self, fname):
        """Return (delta_path, vbv_path) for a given T21 filename."""
        if self.fmt == 'varying_astro':
            d = (os.path.join(self.ic_dir,
                 [f for f in os.listdir(self.ic_dir) if 'delta' in f.lower()][0])
                 if self._shared_delta is not None and os.path.isdir(self.ic_dir) else None)
            return None, None   # handled separately via _shared_delta / _shared_vbv

        # varying_IC: extract IC number from filename
        # e.g. T21_cube_z10__Npix256_IC42.mat → IC42
        m = re.search(r'_IC(\d+)\.mat$', fname)
        if not m:
            return None, None
        ic_id = m.group(1)
        d = os.path.join(self.ic_dir, f'delta_Npix{self.npix}_IC{ic_id}.mat')
        v = os.path.join(self.ic_dir, f'vbv_Npix{self.npix}_IC{ic_id}.mat')
        return (d if os.path.exists(d) else None,
                v if os.path.exists(v) else None)

    def _load_ic_cached(self, path):
        """Load and normalise an IC field, with LRU cache."""
        if path is None:
            return None
        if path not in self._ic_cache:
            if len(self._ic_cache) >= self._ic_cache_max:
                self._ic_cache.popitem(last=False)   # evict oldest
            arr = load_mat(path)
            self._ic_cache[path] = (arr - arr.mean()) / (arr.std() + 1e-8)
        else:
            # Move to end (most recently used)
            self._ic_cache.move_to_end(path)
        return self._ic_cache[path]

    # ------------------------------------------------------------------
    # Parameter loading
    # ------------------------------------------------------------------

    def _load_all_params(self):
        self.raw_params = {}
        for fname in self.files:
            if self.fmt == 'varying_astro':
                m = re.search(r'diffusion_(\d+)', fname)
                z = re.search(r'_z(\d+)_', fname) or re.search(r'_z(\d+)__', fname)
                if not m:
                    continue
                pfile = os.path.join(self.param_dir,
                                     f"parameters__diffusion_{m.group(1)}.params")
                if os.path.exists(pfile):
                    p = parse_params(pfile)
                    p['redshift'] = float(z.group(1)) if z else 10.0
                    for k in LOG_PARAMS:
                        if k in p and p[k] > 0:
                            p[k] = np.log10(p[k])
                    self.raw_params[fname] = p

            else:  # varying_IC
                z = re.search(r'_z(\d+)_', fname) or re.search(r'_z(\d+)__', fname)
                p = dict(VARYING_IC_FIXED_PARAMS)
                p['redshift'] = float(z.group(1)) if z else 10.0
                for k in LOG_PARAMS:
                    if k in p and p[k] > 0:
                        p[k] = np.log10(p[k])
                self.raw_params[fname] = p

    def _compute_norm_stats(self):
        # Parameter normalisation
        vals = {k: [] for k in PARAM_KEYS}
        for p in self.raw_params.values():
            for k in PARAM_KEYS:
                if k in p:
                    vals[k].append(p[k])
        self.p_mean = {k: float(np.mean(v)) if v else 0.0 for k, v in vals.items()}
        self.p_std  = {k: float(np.std(v))  + 1e-8  if v else 1.0 for k, v in vals.items()}

        # T21 normalisation from a small sample
        sample = self.files[:min(50, len(self.files))]
        vals_t21 = []
        for f in sample:
            try:
                cube = load_mat(os.path.join(self.t21_dir, f))
                vals_t21.append(cube.ravel()[::200])
            except Exception:
                pass
        if vals_t21:
            all_v = np.concatenate(vals_t21)
            self.t21_mean = float(all_v.mean())
            self.t21_std  = float(all_v.std()) + 1e-8
        else:
            self.t21_mean, self.t21_std = 0.0, 1.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def param_vector(self, fname):
        p = self.raw_params.get(fname, {})
        return torch.tensor(
            [(p.get(k, self.p_mean[k]) - self.p_mean[k]) / self.p_std[k]
             for k in PARAM_KEYS],
            dtype=torch.float32)

    def __len__(self):
        return len(self.files) * self.patches_per_cube

    def __getitem__(self, idx):
        file_idx = idx // self.patches_per_cube
        fname = self.files[file_idx]
        try:
            cube = load_mat(os.path.join(self.t21_dir, fname))
        except IOError:
            # Corrupt / unreadable file — return a neighbour sample instead
            return self.__getitem__((idx + 1) % len(self))
        cube  = (cube - self.t21_mean) / self.t21_std

        # Random cubic patch
        N, ps = cube.shape[0], self.patch_size
        if N >= ps:
            i = np.random.randint(0, N - ps + 1)
            j = np.random.randint(0, N - ps + 1)
            k = np.random.randint(0, N - ps + 1)
            patch = cube[i:i+ps, j:j+ps, k:k+ps]
        else:
            pad   = [(0, max(0, ps - s)) for s in cube.shape]
            patch = np.pad(cube, pad)[:ps, :ps, :ps]
            i = j = k = 0

        patch = torch.from_numpy(patch).unsqueeze(0)   # (1, ps, ps, ps)

        # IC patches at the SAME spatial location
        ic_delta = torch.zeros(1, ps, ps, ps)
        ic_vbv   = torch.zeros(1, ps, ps, ps)

        if not self.load_ic:
            pass  # return zeros — used by VAE training which doesn't need IC

        elif self.fmt == 'varying_astro':
            # Shared IC fields preloaded in _init_shared_ic
            for field, target in [(self._shared_delta, 'delta'),
                                   (self._shared_vbv,   'vbv')]:
                if field is None:
                    continue
                N_ic = field.shape[0]
                if N_ic >= ps:
                    arr = field[i:i+ps, j:j+ps, k:k+ps].copy()
                else:
                    arr = field[:ps, :ps, :ps].copy()
                t = torch.from_numpy(arr).unsqueeze(0)
                if target == 'delta':
                    ic_delta = t
                else:
                    ic_vbv = t

        else:  # varying_IC — load per-sim IC
            d_path, v_path = self._ic_paths_for(fname)
            for path, target in [(d_path, 'delta'), (v_path, 'vbv')]:
                field = self._load_ic_cached(path)
                if field is None:
                    continue
                N_ic = field.shape[0]
                if N_ic >= ps:
                    arr = field[i:i+ps, j:j+ps, k:k+ps].copy()
                else:
                    arr = field[:ps, :ps, :ps].copy()
                t = torch.from_numpy(arr).unsqueeze(0)
                if target == 'delta':
                    ic_delta = t
                else:
                    ic_vbv = t

        return {
            'patch':    patch,
            'ic_delta': ic_delta,
            'ic_vbv':   ic_vbv,
            'params':   self.param_vector(fname),
        }
