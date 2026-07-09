import os
import argparse

base = """#!/bin/bash
#
#SBATCH --job-name=%s
#SBATCH --output="%s-%%N-%%j.out"
#SBATCH --time=%d:00:00
#SBATCH --mem=%dG
#SBATCH --ntasks=%d
"""


DEFAULT_SCRATCH_DIR = '/nlp/scr/schegini'


def write_job(
        path,
        outdir='./',
        time=24,
        n_cores=1,
        memory=8,
        use_gpu=False,
        slurm_account=None,
        slurm_partition=None,
        slurm_constraint=None,
        exclude=None,
        scratch_dir=DEFAULT_SCRATCH_DIR,
):
    """Generate a single SLURM batch script (`.pbs`) that runs the pipeline for one config.

    Returns the path to the generated `.pbs` file.
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if isinstance(exclude, (list, tuple)):
        exclude = ','.join(exclude)

    job_name = os.path.splitext(os.path.basename(path))[0]
    filename = os.path.join(outdir, job_name + '.pbs')
    with open(filename, 'w') as f:
        f.write(base % (job_name, job_name, time, memory, n_cores))
        if use_gpu:
            f.write('#SBATCH --gres=gpu:1\n')
        if slurm_account:
            f.write('#SBATCH --account=%s\n' % slurm_account)
        if slurm_partition:
            f.write('#SBATCH --partition=%s\n' % slurm_partition)
        if slurm_constraint:
            f.write('#SBATCH --constraint=%s\n' % slurm_constraint)
        if exclude:
            f.write('#SBATCH --exclude=%s\n' % exclude)
        f.write('\n\nset -e\n\n')
        if scratch_dir:
            # Redirect the HuggingFace cache, uv cache, and project venv onto
            # group scratch so the job never writes multi-GB artifacts into the
            # 20GB /sailhome home quota, regardless of the compute node's shell
            # config. See CLUSTER.md ("disk quota").
            f.write('# Keep caches + venv off the /sailhome home quota (see CLUSTER.md).\n')
            f.write('export HF_HOME=%s/.cache/huggingface\n' % scratch_dir)
            f.write('export UV_CACHE_DIR=%s/.cache/uv\n' % scratch_dir)
            f.write('export UV_PROJECT_ENVIRONMENT=%s/parcelmate/.venv\n' % scratch_dir)
            f.write('mkdir -p %s/.cache\n\n' % scratch_dir)
        # Ensure uv is available (installs to ~/.local/bin if missing)
        f.write('if ! command -v uv &> /dev/null; then\n')
        f.write('    if [ -x "$HOME/.local/bin/uv" ]; then\n')
        f.write('        export PATH="$HOME/.local/bin:$PATH"\n')
        f.write('    else\n')
        f.write('        curl -LsSf https://astral.sh/uv/install.sh | sh\n')
        f.write('        export PATH="$HOME/.local/bin:$PATH"\n')
        f.write('    fi\n')
        f.write('fi\n\n')
        # Sync venv and run
        f.write('uv sync\n')
        f.write('uv run python -m parcelmate.bin.main %s\n' % path)

    return filename


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Generate SLURM batch jobs to run parcellations specified in one or more config (YAML) files.
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) to config file(s).')
    argparser.add_argument('-t', '--time', type=int, default=24, help='Maximum number of hours to train models')
    argparser.add_argument('-n', '--n_cores', type=int, default=1, help='Number of cores to request')
    argparser.add_argument('-m', '--memory', type=int, default=8, help='Number of GB of memory to request')
    argparser.add_argument('-g', '--use_gpu', action='store_true', help='Whether to request a GPU node')
    argparser.add_argument('-a', '--slurm_account', default=None, help='Value for SLURM --account setting, if applicable')
    argparser.add_argument('-P', '--slurm_partition', default=None, help='Value for SLURM --partition setting, if applicable')
    argparser.add_argument('-e', '--exclude', nargs='+', help='Nodes to exclude')
    argparser.add_argument('-C', '--constraint', default=None, help='Value for SLURM --constraint setting (e.g. GPU_CC>=7.5, GPU_TYP:A100)')
    argparser.add_argument('-o', '--outdir', default='./', help='Directory in which to place generated batch scripts')
    argparser.add_argument('--scratch', default=DEFAULT_SCRATCH_DIR,
                           help='Group scratch base for HF/uv caches + venv, kept off the home quota. '
                                'Pass "" to disable the redirect. Default: %s' % DEFAULT_SCRATCH_DIR)
    args = argparser.parse_args()

    for path in args.paths:
        write_job(
            path,
            outdir=args.outdir,
            time=args.time,
            n_cores=args.n_cores,
            memory=args.memory,
            use_gpu=args.use_gpu,
            slurm_account=args.slurm_account,
            slurm_partition=args.slurm_partition,
            slurm_constraint=args.constraint,
            exclude=args.exclude,
            scratch_dir=args.scratch,
        )
