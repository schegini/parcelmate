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
