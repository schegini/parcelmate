# The original parcelmate, as a side-by-side baseline

`README.md` in this directory is upstream's own, left untouched. This file is the
fork's note about what this directory is and how to use it.

## What this is

A copy of **`coryshain/parcelmate` at commit `39bfe13`** — the point this fork
diverged (`git merge-base HEAD upstream/main`). Upstream has not moved since, so
this is also its current `main`.

It exists to answer one question: **are the fork's results a property of the
method, or of the fork's changes to the code?**

It is deliberately *not* a copy of the fork. It carries the original's pipeline
plus three specific additions (below) and nothing else — no size-matched random
baselines, no LM-loss evaluation, no threshold sweep.

## Compatibility patches

Three changes were needed to make 2025-era code run correctly against today's
libraries. Each is marked in-place with a `BASELINE PATCH` comment.

**Patch 1 — `model.py`, HuggingFace dataset loading.**
`dataset='wikitext'` → `'Salesforce/wikitext'`, `'bookcorpus'` →
`'bookcorpus/bookcorpus'`, and `_data_kwargs['trust_remote_code'] = True` removed.
Bare ids no longer resolve and `trust_remote_code` was removed in `datasets>=3.0`,
so **without this the original cannot load wikitext at all**.

`tldr17` needed more: `webis/tldr-17` is a script-only repo and `datasets>=4`
removed script support outright, so it dies with `RuntimeError: Dataset scripts
are no longer supported, but found tldr-17.py` (this killed the first cluster
run, job 16402125). It now loads HF's auto-converted parquet branch through the
packaged `parquet` builder — the same text by another route. `get_dataset` picks
the `content` key for it, and that selection code is byte-identical to the fork's,
so both codebases read the same field of the same corpus.

All of this matches the fork's own fixes (`aca114a` and its tldr17 handling).

**Patch 2 — `data.py`, CPU fallback in `correlate` (1 line).**
`use_gpu = use_gpu and torch.cuda.is_available()`. The original unconditionally
calls `torch.cuda.get_device_properties(0)`, which crashes without CUDA. A no-op
on a GPU node.

**Patch 3 — `model.py`, `PerturbedLayer.forward` (1 line). This one changes
results.** Was `self.layer.forward(*args, **kwargs)`, now `self.layer(...)`.
Calling `.forward()` directly bypasses PyTorch's forward hooks, and
**transformers v5 collects `output_hidden_states` via those hooks**. So every
wrapped block silently dropped its own hidden state:

| lesioned layers | hidden states returned (of 13) |
|---|---|
| embedding only | 13 |
| one block | 12 |
| all 12 blocks | **1** |

A knockout spanning the whole model therefore produced connectivity over **768
units instead of 9984** — the embedding layer alone — while the healthy run used
all 9984. Measured on transformers 5.12.1, which is what `uv.lock` pins.

The lesion itself always applied correctly; only the *recording* of activations
was lost. So loss-based results are unaffected, but any connectivity or stability
computed **after** a knockout was measuring a fraction of the model.

## The three additions

Marked in-place with `ADDED (n)` comments.

**(1) Mean vectors for mean-out.** `compute_mean_activations` collects each
neuron's mean activation and caches it to `mean_activations.h5`. Zero-out sets a
lesioned neuron to 0, which is off-distribution — no neuron rests at 0, so part of
the damage is the shock of an impossible value rather than the loss of the
network. Mean-out clamps to a mean instead, so the neuron goes uninformative while
staying in range. Both the cross-domain aggregate (token-weighted, the default
clamp) and each per-corpus mean vector are stored, so a network can also be
clamped to one corpus's mean.

**(2) No union.** The original OR-ed every column of the shared parcellation
together and lesioned the union of all subnetworks in one pass, so nothing could
be attributed to any single network. `select_knockout` now requires a
`network_ix` and **refuses** a multi-network selection; `run_knockout` loops over
networks, writing `knockout/network<i>_<mode>/`. Both modes run against the same
parcellation and the same per-network unit selection, so mean-out and zero-out
lesion identical units and are directly comparable.

**(3) Distilled stability.** `summarize_knockout_stability` reduces each
condition's stability matrix to one number — the median of its strict lower
triangle — and writes `knockout/stability_summary.csv` with `condition, network,
mode, scope, median_stability, n_pairs`. Lower triangle because the matrix is
symmetric with a trivial unit diagonal; median because one degenerate pair
shouldn't drag the summary the way a mean would. Scopes are `betweendomain` and
`withindomain_<domain>`.

Two things this had to get right:

- **Lesioned units are excluded explicitly**, via a per-condition
  `knockout_selection.h5`, not by filtering NaN. A lesioned unit is constant so
  its connectivity *should* be undefined, but `correlate` mean-centres in
  float32: clamping to a large non-zero mean leaves rounding residue, the norm
  isn't exactly 0, and the unit yields **noise correlations instead of NaN**. In a
  test run only 196 of 736 lesioned units went NaN under mean-out versus 612 of
  736 under zero-out. Filtering NaN alone would drop different unit counts per
  mode and bias the mean-vs-zero comparison. Both modes now drop the identical
  set.
- `plot_stability` is **untouched**, still zero-filling NaN as upstream wrote it.
  It and this summary can therefore disagree on knocked-out conditions; the
  summary is the one to trust after a lesion.

`n_pairs` is worth watching: with `n_samples: 2` the within-domain median is taken
over a single sample pair, so it is that pair, not a median. Raise
`connectivity.n_samples` if you want the within-domain numbers to mean much.

**(4) Cross-domain mean-out (GOAL.md project 2).** `knockout.mean_domain` clamps
a subnetwork to **one corpus's** mean and then evaluates the lesioned model on
**the corpora it was not clamped to**. Neither upstream nor the fork does this:
upstream has no mean-out at all, and the fork clamps to a corpus but still
evaluates on every corpus including that one.

Holding the clamp corpus out is the point. With it left in, the frozen units sit
at exactly the mean of the text being read — the least informative case. The
question worth asking is what happens when the frozen network is *wrong* for the
input.

```yaml
knockout:
  mean_domain: each        # one condition per corpus, each holding its own out
  # mean_domain: wikitext  # just that one
  # mean_domain: null      # cross-domain aggregate, evaluated on everything
```

Conditions are named `network<i>_mean-<corpus>` and the summary carries a
`clamp_domain` column, so mean-out-clamped-to-X can be read against zero-out on
the same network. Zero-out has no clamp corpus, so it runs once over every
domain and stays the reference.

Two consequences to plan for. Conditions multiply by the number of corpora —
`networks x (1 zero + n_corpora mean)` — so trim `networks` or `domains` before
running. And each mean-out condition sees one fewer corpus, so its
`betweendomain` scope has `n-1` domain averages; with only two corpora
configured, that scope disappears entirely (it needs at least two).

`configs/project2.yaml` is a ready config: match_iter2's settings plus
`mean_domain: each`.

**(5) Healthy reference.** A lone stability number says nothing — there is no way
to tell a damaged model from one that was never very stable to begin with. The
summary now carries a `mode='healthy'` row per lesioned network.

It costs **no model passes**. The run root's `connectivity/` already *is* the
unperturbed model, written by the pipeline's `connectivity` step; it had simply
never been distilled. The reference is computed by re-reading those files.

The reference is per network, over exactly the units that network's lesion
removed. Comparing an all-units healthy against a lesion missing 12% of the model
would fold "these are different neurons" into the gap you are trying to read.

Rebuild the summary for a run that has already finished — no GPU, no re-running:

```bash
python -m parcelmate.bin.main configs/match_iter2.yaml -s summarize_stability
```

It loads every connectivity matrix (~10 GB for a 4-domain gpt2 run), so give it a
CPU node rather than the login node.

## Verifying what changed

```bash
# from the repo root
diff <(git show upstream/main:parcelmate/model.py) original/parcelmate/model.py
diff <(git show upstream/main:parcelmate/plot.py)  original/parcelmate/plot.py
diff <(git show upstream/main:parcelmate/data.py)  original/parcelmate/data.py

# the fork vs this baseline
diff -ru original/parcelmate parcelmate
```

Scaffolding that does not exist upstream and so cannot change its behaviour:
`pyproject.toml` + `uv.lock` (copied from the fork so both sides resolve
**identical** dependency versions — otherwise you would be comparing library
versions as much as code), `configs/`, `jobs/`, and this file.

## Running it

The one rule: **run from inside this directory.** `python -m parcelmate.bin.main`
resolves the package from the current working directory first, so running from the
repo root would silently execute the *fork's* code. The cluster job asserts this;
by hand, check with:

```bash
cd original
python -c "import parcelmate, os; print(os.path.abspath(parcelmate.__file__))"
```

`configs/match_iter2.yaml` copies every connectivity and parcellation setting from
the fork's `parcelmate/configs/knockout_iter2.yaml`, so the code is the only
variable:

```bash
ssh <CSID>@sc.stanford.edu
cd ~/parcelmate/original
sbatch jobs/match_iter2.pbs        # submit from THIS dir; the job cds to $SLURM_SUBMIT_DIR
squeue -u <CSID>
```

Outputs land in `/nlp/scr/schegini/parcelmate/original/match_iter2`, a separate
tree from every fork run — both pipelines resume from whatever h5s they find on
disk, so a shared `output_dir` would silently blend the two codebases' outputs.

**Cost.** The knockout re-runs connectivity once per condition, and conditions
multiply as `networks x modes`. With `networks: null` (every shared subnetwork)
and both modes that can be a lot of passes; set `networks: [0, 1, 2]` while
iterating.

## What to compare afterwards

Both runs share the same output layout, so compare the same files across the two
`output_dir`s:

- `connectivity/connectivity_<domain>_avg.h5` — the most direct test, and
  deterministic given the same data, so large differences mean a real code
  divergence rather than sampling noise.
- `parcellation/` sample scores — stochastic, so compare distributions, not
  per-index identity.
- `knockout/stability_summary.csv` — the distilled scalar, per network and mode.
