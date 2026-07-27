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

The networks come **only** from the parcellation of the healthy model.
`knockout_mode` and `knockout_thresh` both accept a list, and every combination
runs against that *same* parcellation: the mode decides what knocked-out units
are clamped to, the threshold decides how much of a network counts as "in it",
and both are applied after the networks have been found. So mean-out vs zero-out
and 0.5 vs 0.9 are all like-for-like comparisons within one run.

Conditions are named `network<i>_thresh<t>_<mode>` (e.g. `network0_thresh0.9_mean`),
and `loss_summary.csv` carries `network`, `thresh` and `mode` columns so every
combination lines up per network — the knockout plot draws modes side by side and
thresholds one above the other.

Configure it with a `knockout` block (see `parcelmate/configs/knockout.yaml` for a
small offline example):

```yaml
knockout:
  knockout_mode: [mean, zero]  # run both against ONE parcellation and compare;
                               # 'mean' clamps knocked-out units to their
                               # cross-domain mean activation ("mean-out"),
                               # 'zero' clamps to zero. A single string
                               # (e.g. `mean`) runs just that one mode.
  knockout_thresh: [0.5, 0.9]  # membership cutoff for "in the network"; a list
                               # runs each against the SAME parcellation. A bare
                               # number (e.g. 0.5) runs just that one. Higher =
                               # sparser selection; a network with no units left
                               # is skipped at that threshold.
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

`subnetwork_knockout` also writes per-network comparison plots
(`<output_dir>/plots/knockout/knockout_<network>_<metric>.png`, e.g.
`knockout_network0_loss.png`), each showing the five conditions — healthy,
mean-out network, mean-out random, zero-out network, zero-out random — faceted by
domain, with one row per threshold. Regenerate just the plots from an existing
summary with `-s plot_knockout`.

Mode and threshold are both handled *within* a single run (above), not by the
sweep — a separate job per value would re-parcellate and knock out different
neurons, so the thing you meant to vary would be confounded with a fresh
stochastic parcellation. To
sweep settings that each warrant their own parcellation (e.g. `n_networks`)
across SLURM jobs and collect the results into one dashboard, see the
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
