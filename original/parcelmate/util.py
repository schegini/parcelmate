import sys
import os
import h5py

def stderr(s):
    sys.stderr.write(s)
    sys.stderr.flush()


def save_h5_data(
        data,
        path,
        verbose=True,
        indent=0
):
    dirpath = os.path.dirname(path)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    if verbose:
        stderr('%sSaving to %s\n' % (' ' * indent, path))
    with h5py.File(path, 'w') as f:
        for key in data:
            f.create_dataset(key, data=data[key])


def load_h5_data(path, verbose=True, indent=0):
    if verbose:
        stderr('%sLoading from %s\n' % (' ' * indent, path))
    out = {}
    with h5py.File(path, 'r') as f:
        for key in f.keys():
            out[key] = f[key][:]

    return out