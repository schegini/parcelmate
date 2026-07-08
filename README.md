# Parcelmate

## How to Run

There are two main approaches to running `parcelmate`: running locally/interactively as a Python module, or generating SLURM jobs for cluster execution.

### 1. Running Locally or Interactively

You can execute the main script as a Python module. It takes an optional YAML configuration file and lets you specify which steps of the pipeline to run.

```bash
python -m parcelmate.bin.main path/to/your_config.yaml
```

**Key Arguments:**
- `config_path`: (Optional) The path to your YAML config file.
- `-s`, `--steps`: A space-delimited list of specific steps you want to run (e.g., `connectivity`, `parcellation`, `subnetwork_extraction`, `plot_connectivity`, etc.). By default, it runs `all`.
- `-O`, `--overwrite`: Add this flag if you want to force recomputing outputs even if they already exist.

### Subnetwork knockout controls (Project 3)

The `subnetwork_knockout` step knocks out each extracted subnetwork **individually**
(plus, optionally, the union of all of them) and compares each against
size-matched **random baselines** drawn from the un-knocked-out complement, so
selectivity can be tested — a subnetwork matters only if knocking it out hurts
more than removing the same number of random neurons. Each condition is evaluated
with both connectivity and next-token LM loss/perplexity; results are collected
into `<output_dir>/knockout/loss_summary.csv`.

Configure it with a `knockout` block (see `parcelmate/configs/knockout.yaml` for a
small offline example):

```yaml
knockout:
  knockout_mode: mean   # 'mean' clamps knocked-out units to their cross-domain
                        # mean activation ("mean-out"); 'zero' clamps to zero
  n_baseline: 3         # size-matched random controls per condition
  baseline_seed: 0
  networks: null        # null = every subnetwork; or a list of indices, e.g. [0, 2]
  include_union: true   # also knock out the union of all subnetworks
  include_healthy: true # run a no-perturbation reference for matched loss
  eval_loss: true
  loss_n_tokens: null   # cap tokens used for the loss pass (null = all drawn)
```

Run just this step (it consumes the outputs of `subnetwork_extraction`):

```bash
python -m parcelmate.bin.main your_config.yaml -s subnetwork_knockout
```

`subnetwork_knockout` also writes per-domain comparison plots
(`<output_dir>/plots/knockout/knockout_<domain>.png`). Regenerate just the plots
from an existing summary with `-s plot_knockout`.

To sweep knockout settings (e.g. mean-out vs zero-out) across SLURM jobs and
collect the results into one dashboard, see the
[Sweeping the subnetwork knockout controls](CLUSTER.md#sweeping-the-subnetwork-knockout-controls-project-3)
section — the knockout runs automatically in any sweep whose base config has a
`knockout:` block (`parcelmate/configs/sweep_knockout.yaml` is a ready example).

### 2. Generating SLURM Jobs (For the Stanford SC Cluster)

To run on the Stanford SC Cluster, there is a dedicated script to help you generate SLURM batch job files (`.pbs`) from your configuration files. 

```bash
python -m parcelmate.bin.make_jobs path/to/your_config.yaml
```

**Useful SLURM Arguments:**
- `-t <hours>`: Max time for the job (default is 24).
- `-m <GB>`: Memory requested in GB (default is 8).
- `-n <cores>`: Number of cores to request (default is 1).
- `-g`, `--use_gpu`: Flag to request a GPU node.
- `-a <account>`, `--slurm_account <account>`: Define your SLURM `--account` parameter.
- `-P <partition>`, `--slurm_partition <partition>`: Define your SLURM `--partition` parameter.
- `-e <nodes>`, `--exclude <nodes>`: Nodes to exclude.
- `-o <dir>`, `--outdir <dir>`: Directory to save the generated `.pbs` scripts (defaults to `./`).

Once `make_jobs.py` generates the `.pbs` file, you can submit it to the cluster scheduler using `sbatch` (for more cluster details, see [CLUSTER.md](CLUSTER.md)):

```bash
sbatch path/to/generated_job.pbs
```
