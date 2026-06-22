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
