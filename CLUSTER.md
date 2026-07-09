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

- Access the SC cluster Open OnDemand portal at **[https://sc.stanford.edu](https://sc.stanford.edu)** and login with your `CSID`.



### Group Affiliation (SLURM Account)

- The SC Cluster is a "condominium-type" cluster. Each user is linked with one or more `account` according to their association.
- When submitting your job, define the `--account` parameter in `srun` or `sbatch` script in order to use the respective compute resources defined in each SLURM `partition`.



### Home Directory and Storage

- Home directory is on `/sailhome/$CSID` with a 20GB quota. This space is meant as a landing space.
- Do not store research dataset in this space; use central storage provided by your group.

> **Backup your data**
> Your home-directory is snapshotted daily. Most other group storage servers are NOT backed-up. You are responsible to make sure important and difficult-to-reproduce data are backed-up.

> **Disk quota**
> The 20GB home quota is easy to blow with the HuggingFace cache, the uv cache,
> and the torch venv. `make_jobs.py` / `sweep.py` bake these redirects into every
> generated `.pbs` so jobs never write them into home (each job gets its own venv,
> keyed on the job name, so concurrent sweep jobs never race on `uv sync`):
> ```bash
> export HF_HOME=/nlp/scr/<CSID>/.cache/huggingface
> export UV_CACHE_DIR=/nlp/scr/<CSID>/.cache/uv
> export UV_PROJECT_ENVIRONMENT=/nlp/scr/<CSID>/parcelmate/venvs/<job_name>
> ```
> Override the base with `--scratch /nlp/scr/<CSID>` (default is `schegini`'s), or
> `--scratch ""` to disable. For interactive sessions, add the same three lines to
> your `~/.bashrc`. Also keep every config's `output_dir` / `output_root` under
> `/nlp/scr/...`, never a relative path (which lands in home).



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

*(You can reference a sample submit script at:* `/sailhome/software/sample-batch.sh`*)*

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

> **Submit** `.pbs` **files with** `sbatch`**, not** `srun`**.**
>
> - A `.pbs` file is a batch script. `sbatch script.pbs` parses its `#SBATCH`
> directives and queues the job. `srun script.pbs` instead tries to *execute the
> file as a command* and ignores the directives entirely.
> - Because the account/partition/GPU are already in the script, you do **not**
> pass `--account`/`--partition`/`--gres` to `sbatch`.
> - Do not submit from inside an interactive `srun --pty bash` session — that shell
> already holds the allocation's resources, so the nested job hangs with
> "Requested nodes are busy".



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

`sbatch` **vs** `srun` **at a glance:**


|                 | `sbatch cory-shain.pbs`      | `srun ... --pty bash`               |
| --------------- | ---------------------------- | ----------------------------------- |
| Runs            | the `.pbs` script unattended | a command you type, interactively   |
| Survives logout | yes                          | no (dies when shell closes)         |
| Best for        | real/long runs               | debugging, quick tests, live output |
| What you pass   | the script file              | the python command directly         |


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

## Running Sweeps

To try several hyperparameter values at once (rather than editing one config by
hand and resubmitting), use the sweep harness. It generates one config + one
`.pbs` per grid point, submits them all, and collects each run's plots into a
single dashboard for side-by-side review.

Everything below runs on the `sc` login node, the same as `make_jobs` — it is
pure file I/O plus `sbatch` calls (no compute), so it is headnode-safe. The
actual pipeline work happens on compute nodes via SLURM.

### 1. Write a sweep spec

A sweep spec is a small YAML file (see
`parcelmate/configs/sweep_nnetworks.yaml`) with a base config and a grid:

```yaml
base_config: parcelmate/configs/cory-shain.yaml
name: nnetworks
output_root: /nlp/scr/schegini/parcelmate/sweeps/nnetworks   # per-run outputs go here
grid:
  parcellation.n_networks: [25, 50, 75, 100]
```

- Dotted keys index into the base config (`parcellation.n_networks` sets
`parcellation: {n_networks: ...}`). Any config key works.
- Each key maps to a list; the sweep runs the **cartesian product** of all keys,
so adding a second key multiplies the number of runs.
- Each run gets its own `output_dir` under `output_root`, tagged by its params
(e.g. `.../nnetworks/n_networks-50`), so runs never clobber each other.



### 2. Generate + submit the sweep

```bash
ssh <CSID>@sc.stanford.edu
cd ~/parcelmate
git pull    # ensure sweep.py / collect.py are present

# Generates configs + .pbs AND sbatches them. -a/-P/-g are baked into every
# .pbs exactly as with make_jobs.
python3 -m parcelmate.bin.sweep parcelmate/configs/sweep_nnetworks.yaml -g -a nlp -P sphinx
```

This creates a `sweep_<name>/` directory holding `configs/`, `jobs/`, and a
`manifest.yaml` that records every run's params, config path, and output_dir.

- Add `--no-submit` to generate everything **without** submitting (inspect the
configs/`.pbs` first, then `for f in sweep_<name>/jobs/*.pbs; do sbatch "$f"; done`).
- SLURM flags (`-t`, `-m`, `-n`, `-g`, `-a`, `-P`, `-e`, `-C`) match `make_jobs`
and apply to every generated job.

Monitor as usual:

```bash
squeue -u <CSID>            # one job per grid point
```



### 3. Collect outputs for review

Once the jobs finish (or to check partial progress), build the dashboard:

```bash
python3 -m parcelmate.bin.collect sweep_nnetworks/manifest.yaml
# -> sweep_nnetworks/dashboard.html  (plots laid out side by side)
# -> sweep_nnetworks/index.md        (same, renders in VSCode/GitHub)
```

Runs that have not produced plots yet show "no outputs yet", so it is safe to
re-run `collect` at any time.

The `sc` headnode is headless, so you cannot open `dashboard.html` there
directly. Options:

- **VSCode Remote-SSH** into `sc` and open `sweep_nnetworks/index.md` in the
markdown preview — images render in place.
- **Open OnDemand** ([https://sc.stanford.edu](https://sc.stanford.edu)) to browse the file via the portal.
- **Pull it to your local machine** via the data-transfer host and open locally.
The dashboard links plots by path relative to the sweep dir, and the plots
themselves live under `output_root`, so copy both for a self-contained view:
  ```bash
  rsync -av <CSID>@scdt.stanford.edu:~/parcelmate/sweep_nnetworks/ ./sweep_nnetworks/
  rsync -av <CSID>@scdt.stanford.edu:/nlp/scr/schegini/parcelmate/sweeps/nnetworks/ \
    ./nlp/scr/schegini/parcelmate/sweeps/nnetworks/
  open sweep_nnetworks/dashboard.html
  ```



### Sweeping the subnetwork knockout controls (Project 3)

The knockout controls are part of the default pipeline: `main.py` with the
default `all` steps runs `subnetwork_knockout` at the end, and every generated
`.pbs` calls `main.py <config>` with no `-s` flag. So **any sweep whose base
config includes a `knockout:` block runs the knockout automatically** — no code
or job-template changes needed. A ready-made base config and spec ship in the
repo:

- `parcelmate/configs/knockout_cluster.yaml` — base config (gpt2, a few domains,
  a `knockout:` block; see the README for every knob).
- `parcelmate/configs/sweep_knockout.yaml` — spec that sweeps
  `knockout.knockout_mode: [mean, zero]` (mean-out vs zero-out).

Full workflow after logging in:

```bash
ssh <CSID>@sc.stanford.edu
cd ~/parcelmate
git pull

# Optional: inspect what will run first.
python3 -m parcelmate.bin.sweep parcelmate/configs/sweep_knockout.yaml --no-submit
less sweep_knockout/configs/*.yaml

# Generate configs + .pbs and submit (GPU node; use your account/partition).
python3 -m parcelmate.bin.sweep parcelmate/configs/sweep_knockout.yaml -g -a nlp -P sphinx -t 24 -m 16

squeue -u <CSID>            # one job per grid point (mean, zero)
```

Each run resumes from any connectivity/parcellation outputs already on disk
(`overwrite` defaults off), so re-submitting only recomputes what is missing.

**Watch the cost.** The knockout step re-runs connectivity + loss once per
condition: one per network in `knockout.networks`, plus `1 + n_baseline` per
network, plus the union — each over every domain in `connectivity.domains`.
Keep `networks` short (e.g. `[0, 1, 2]`) and `n_baseline` small (e.g. 3) while
iterating; cap the loss pass with `knockout.loss_n_tokens`. If a knockout
selects more than half of all units, its random baseline is skipped (no equal
sized disjoint set exists) and the job logs a warning — raise
`knockout.knockout_thresh` toward 1.0 for a sparser, baseline-able selection.

Collect results the same way as any sweep:

```bash
python3 -m parcelmate.bin.collect sweep_knockout/manifest.yaml
```

The dashboard shows the per-domain knockout-loss bar charts
(`plots/knockout/knockout_<domain>.png`: healthy vs subnetwork-knockout vs
baseline). The exact numbers live in each run's
`<output_dir>/knockout/loss_summary.csv` — pull them alongside the plots:

```bash
rsync -av <CSID>@scdt.stanford.edu:/nlp/scr/schegini/parcelmate/sweeps/knockout/ \
  ./nlp/scr/schegini/parcelmate/sweeps/knockout/
```



## Usage Policy

> **Important**
> The cluster is a shared resource.
>
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

- `showaccount`: Show user cluster, user, and account affiliation.
- `showjob <jobid>`: Detailed output for a specific job, including state, time limits, required nodes, etc.
- `showalloc <partition>`: View memory, CPU, and GPU allocations for nodes in a partition.
- `sgpu -g <partition>`: Show comprehensive GPU status including total GPUs, current GPU utilization/memory usage, and usage broken down by user.

