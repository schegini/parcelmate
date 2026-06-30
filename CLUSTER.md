# Stanford SC Cluster Instructions

## Getting Started

### Login via SSH
- The cluster primary login node is `sc.stanford.edu`.
- SSH directly if you are already connected on the Stanford Network.
- If logging in from outside the Stanford Network, you will need to use **Stanford VPN - Full Tunnel** or hop through another node that has a public network interface (eg. `scdt.stanford.edu`).

> **Important!**
> Do NOT run resource intensive processes on `sc` headnode (NO vscode, ipython, tensorboard...etc), they will be killed automatically.

### WebGUI via Open OnDemand
Open OnDemand is an open-sourced HPC web-portal.
- Access the SC cluster Open OnDemand portal at **https://sc.stanford.edu** and login with your `CSID`.

### Group Affiliation (SLURM Account)
- The SC Cluster is a "condominium-type" cluster. Each user is linked with one or more `account` according to their association.
- When submitting your job, define the `--account` parameter in `srun` or `sbatch` script in order to use the respective compute resources defined in each SLURM `partition`.

### Home Directory and Storage
- Home directory is on `/sailhome/$CSID` with a 20GB quota. This space is meant as a landing space.
- Do not store research dataset in this space; use central storage provided by your group.

> **Backup your data**
> Your home-directory is snapshotted daily. Most other group storage servers are NOT backed-up. You are responsible to make sure important and difficult-to-reproduce data are backed-up.

### Data-transfer (SCDT)
- Designated host for data-transfer: `scdt.stanford.edu`.
- Keep any parallelism to the minimal (check `top`).
- For downloads from cloud providers, the use of `rclone` is suggested.
- We also allow vscode, ipython, tensorboard to be ran on SCDT to an extent, please note these processes will be wiped out every 24 hours.

## Job Submission

### SLURM - Overview
The SC cluster uses SLURM for job scheduling.

### Interactive Jobs (shell access)
Interactive jobs are for real-time interaction (prototyping, testing, debugging).
To request an interactive session, SSH to `sc.stanford.edu` and use the `srun` command:

```bash
srun --account=your_group_account --partition=my_partition --pty bash
```

To request an interactive session with a GPU:
```bash
srun --account=your_group_account --partition=my_partition --nodelist=node1 --gres=gpu:1 --pty bash
```

### Batch Jobs
For submitting batch jobs (real-time interaction not required):
```bash
sbatch my_script.sh
```
*(You can reference a sample submit script at: `/sailhome/software/sample-batch.sh`)*

### GPU
Users can request a specific type of GPU or specify a vRAM/arch constraint:
```bash
# Request 1 H100 GPU from any nodes in mypartition
srun --account=your_group_account --partition=mypartition --gres=gpu:h100:1 --pty bash

# Request 1 GPU with 80G vRAM from any nodes in mypartition
srun --account=your_group_account --partition=mypartition --gres=gpu:1 --constraint=80G --pty bash
```

## Running Parcelmate Jobs

The full workflow to run a config on the cluster:

```bash
# 1. Connect to the LOGIN node. Only `sc` has the SLURM client (sbatch/srun/squeue).
#    `scdt` is data-transfer only and will report "Command 'sbatch' not found".
ssh <CSID>@sc.stanford.edu
cd ~/parcelmate

# 2. Generate the batch script. The -a/-P/-g flags are baked into the .pbs as
#    #SBATCH --account / --partition / --gres directives.
python3 -m parcelmate.bin.make_jobs parcelmate/configs/cory-shain.yaml -g -a nlp -P sphinx

# 3. Submit the script with sbatch.
sbatch cory-shain.pbs

# 4. Monitor and read output.
squeue -u <CSID>            # or: pestat -u <CSID> -G
tail -f cory-shain-*.out    # job stdout/stderr lands in ~/parcelmate
```

> **Submit `.pbs` files with `sbatch`, not `srun`.**
> - A `.pbs` file is a batch script. `sbatch script.pbs` parses its `#SBATCH`
>   directives and queues the job. `srun script.pbs` instead tries to *execute the
>   file as a command* and ignores the directives entirely.
> - Because the account/partition/GPU are already in the script, you do **not**
>   pass `--account`/`--partition`/`--gres` to `sbatch`.
> - Do not submit from inside an interactive `srun --pty bash` session — that shell
>   already holds the allocation's resources, so the nested job hangs with
>   "Requested nodes are busy".

### Running interactively with `srun`

Use `srun` for real-time work — prototyping, debugging, or watching output live —
rather than for submitting the `.pbs` script. With `srun` you run the **actual
command** (the python module), not the `.pbs` file.

**Option A — interactive shell, then run by hand:**

```bash
# from the sc login node; this drops you onto a compute node with the GPU held by THIS shell
srun --account=nlp --partition=sphinx --gres=gpu:1 --pty bash

cd ~/parcelmate
uv sync
uv run python -m parcelmate.bin.main parcelmate/configs/cory-shain.yaml
```

Once you have the shell, run the python command directly — do **not** `srun`
again inside it (that nests an allocation and hangs with "Requested nodes are busy").

**Option B — run a command in one shot (no interactive shell):**

```bash
srun --account=nlp --partition=sphinx --gres=gpu:1 \
  bash -c 'cd ~/parcelmate && uv sync && uv run python -m parcelmate.bin.main parcelmate/configs/cory-shain.yaml'
```

This blocks your terminal and streams output live until the job finishes.

**`sbatch` vs `srun` at a glance:**

| | `sbatch cory-shain.pbs` | `srun ... --pty bash` |
|---|---|---|
| Runs | the `.pbs` script unattended | a command you type, interactively |
| Survives logout | yes | no (dies when shell closes) |
| Best for | real/long runs | debugging, quick tests, live output |
| What you pass | the script file | the python command directly |

For real runs, prefer `sbatch`. Reach for `srun` only when you want to sit on a
node and iterate.

### Cleaning up outputs

Job logs (`*.out`) and the pipeline `output_dir` (e.g. `test/` or `results/`,
set in the config) accumulate in `~/parcelmate`. To clear them without touching
code:

```bash
cd ~/parcelmate
rm -f cory-shain*.out cory-shain*.pbs   # SLURM logs + generated scripts
rm -rf test                             # pipeline output_dir (check your config first)
```

## Usage Policy

> **Important**
> The cluster is a shared resource.
> - Running job or CPU intensive process on `sc.stanford.edu` (headnode) and `scdt.stanford.edu` (data-transfer) themselves is strictly prohibited.
> - No direct access (eg. SSH) to compute node unless otherwise arranged (you may SSH to the compute node where you have an active running job).

## Useful CLI/Tools

### SLURM CLI

`pestat` is a tool for a quick/overall view of the cluster:
- **Status of each node on the cluster** (with GPU usage): `pestat -G`
- **Status of each node within a partition**: `pestat -p mypartition -G`
- **Status of a specific node**: `pestat -n mynode -G`
- **List nodes that have a job owned by a specific user**: `pestat -u myuser -G`

Standard Slurm commands:
- **View all jobs queued in a specific partition**: `squeue -p mypartition`
- **View detailed information of a specific job**: `scontrol show job jobid`
- **Cancel a job**: `scancel "jobid"`

### SC-Specific Tools

- **`showaccount`**: Show user cluster, user, and account affiliation.
- **`showjob <jobid>`**: Detailed output for a specific job, including state, time limits, required nodes, etc.
- **`showalloc <partition>`**: View memory, CPU, and GPU allocations for nodes in a partition.
- **`sgpu -g <partition>`**: Show comprehensive GPU status including total GPUs, current GPU utilization/memory usage, and usage broken down by user.
