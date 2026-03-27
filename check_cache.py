"""Diagnostic: check behavioral feature quality in the v5 cache."""
import json, glob, numpy as np, os

cache_files = sorted(glob.glob('checkpoints/feature_cache_v5/*.npz'))
print(f'Found {len(cache_files)} cached videos')

zero_std_count = 0
no_behavioral = 0
ok_count = 0

for cf in cache_files:
    d = np.load(cf, allow_pickle=True)
    keys = list(d.keys())
    if 'behavioral' not in keys:
        no_behavioral += 1
        continue
    b = d['behavioral']
    std_per_dim = b.std(axis=0)
    zero_dims = (std_per_dim < 0.01).sum()
    if zero_dims > 4:
        zero_std_count += 1
    else:
        ok_count += 1
    if ok_count <= 5 or zero_dims > 4:
        name = os.path.basename(cf)
        print(f'  {name}: shape={b.shape}  zero_dims={zero_dims}/9  std={std_per_dim.round(3).tolist()}')

print()
print(f'Summary: ok={ok_count}  poor_quality={zero_std_count}  no_behavioral={no_behavioral}')
