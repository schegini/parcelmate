import os
import sys
import copy
import itertools
import subprocess
import argparse

import yaml

from parcelmate.cfg import get_cfg
from parcelmate.bin.make_jobs import write_job


def _stderr(s):
    sys.stderr.write(s)
    sys.stderr.flush()


def _set_nested(d, dotted_key, value):
    """Set a value in a nested dict using a dotted key, e.g. `parcellation.n_networks`."""
    keys = dotted_key.split('.')
    node = d
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def _tag(combo):
    """Build a filesystem-safe tag from a dict of {dotted_key: value}."""
    parts = []
    for key, value in combo.items():
        short = key.split('.')[-1]
        v = str(value).replace('/', '-').replace(' ', '')
        parts.append('%s-%s' % (short, v))
    return '__'.join(parts)


def expand_grid(grid):
    """Cartesian product of a {dotted_key: [values]} grid -> list of {dotted_key: value} dicts."""
    keys = list(grid.keys())
    value_lists = [grid[k] if isinstance(grid[k], (list, tuple)) else [grid[k]] for k in keys]
    return [dict(zip(keys, values)) for values in itertools.product(*value_lists)]


def build_sweep(spec, outdir):
    """Generate one config YAML per grid point. Returns the manifest (list of run dicts)."""
    base = get_cfg(spec['base_config'])
    name = spec.get('name', 'sweep')

    base_output_dir = base.get('output_dir', 'results')
    output_root = spec.get('output_root', os.path.join(os.path.dirname(base_output_dir.rstrip('/')), name))

    config_dir = os.path.join(outdir, 'configs')
    os.makedirs(config_dir, exist_ok=True)

    manifest = []
    for combo in expand_grid(spec['grid']):
        tag = _tag(combo)
        cfg = copy.deepcopy(base)
        for key, value in combo.items():
            _set_nested(cfg, key, value)
        cfg['output_dir'] = os.path.join(output_root, tag)

        config_path = os.path.join(config_dir, '%s__%s.yaml' % (name, tag))
        with open(config_path, 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        manifest.append(dict(
            tag=tag,
            params=combo,
            config_path=os.path.abspath(config_path),
            output_dir=cfg['output_dir'],
        ))
        _stderr('Generated %s -> %s\n' % (tag, cfg['output_dir']))

    manifest_path = os.path.join(outdir, 'manifest.yaml')
    with open(manifest_path, 'w') as f:
        yaml.safe_dump(dict(name=name, output_root=output_root, runs=manifest), f, sort_keys=False)
    _stderr('Wrote manifest: %s (%d runs)\n' % (manifest_path, len(manifest)))

    return manifest


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        'Generate a config sweep from a base config + grid spec, then generate and (optionally) submit SLURM jobs.'
    )
    argparser.add_argument('spec_path', help='Path to a sweep spec YAML (base_config, name, grid, [output_root]).')
    argparser.add_argument('-o', '--outdir', default=None,
                           help='Directory for generated configs, .pbs scripts, and manifest. Default: sweep_<name>.')
    argparser.add_argument('--no-submit', dest='submit', action='store_false',
                           help='Generate everything but do not sbatch. Default is to submit.')
    # SLURM passthrough (mirrors make_jobs.py)
    argparser.add_argument('-t', '--time', type=int, default=24)
    argparser.add_argument('-n', '--n_cores', type=int, default=1)
    argparser.add_argument('-m', '--memory', type=int, default=8)
    argparser.add_argument('-g', '--use_gpu', action='store_true')
    argparser.add_argument('-a', '--slurm_account', default=None)
    argparser.add_argument('-P', '--slurm_partition', default=None)
    argparser.add_argument('-e', '--exclude', nargs='+')
    argparser.add_argument('-C', '--constraint', default=None)
    args = argparser.parse_args()

    spec = get_cfg(args.spec_path)
    name = spec.get('name', 'sweep')
    outdir = args.outdir or ('sweep_%s' % name)
    os.makedirs(outdir, exist_ok=True)

    manifest = build_sweep(spec, outdir)

    job_dir = os.path.join(outdir, 'jobs')
    pbs_paths = []
    for run in manifest:
        pbs_paths.append(write_job(
            run['config_path'],
            outdir=job_dir,
            time=args.time,
            n_cores=args.n_cores,
            memory=args.memory,
            use_gpu=args.use_gpu,
            slurm_account=args.slurm_account,
            slurm_partition=args.slurm_partition,
            slurm_constraint=args.constraint,
            exclude=args.exclude,
        ))
    _stderr('Generated %d job scripts in %s\n' % (len(pbs_paths), job_dir))

    if not args.submit:
        _stderr('Skipping submission (--no-submit). Submit with: for f in %s/*.pbs; do sbatch "$f"; done\n' % job_dir)
        sys.exit(0)

    if subprocess.call(['bash', '-c', 'command -v sbatch >/dev/null 2>&1']) != 0:
        _stderr('WARNING: sbatch not found on PATH; jobs were generated in %s but not submitted.\n' % job_dir)
        sys.exit(0)

    for pbs in pbs_paths:
        result = subprocess.run(['sbatch', pbs], capture_output=True, text=True)
        if result.returncode == 0:
            _stderr('Submitted %s: %s' % (os.path.basename(pbs), result.stdout))
        else:
            _stderr('FAILED to submit %s: %s\n' % (os.path.basename(pbs), result.stderr))
