# The original parcelmate, as a side-by-side baseline

`README.md` in this directory is upstream's own, left untouched. This file is the
fork's note about what this directory is and how to use it.

## What this is

A verbatim copy of **`coryshain/parcelmate` at commit `39bfe13`** — the point this
fork diverged (`git merge-base HEAD upstream/main`). Upstream has not moved since,
so this is also its current `main`.

It exists to answer one question: **are the fork's results a property of the
method, or of the fork's changes to the code?**

## What was changed, and why

Two patches, both required to make 2025-era code run against today's libraries.
Each is marked in-place with a `BASELINE PATCH` comment.

**Patch 1 — `parcelmate/model.py`, HuggingFace dataset ids (3 lines).**
`dataset='wikitext'` → `'Salesforce/wikitext'`, `'bookcorpus'` →
`'bookcorpus/bookcorpus'`, and `_data_kwargs['trust_remote_code'] = True` removed.
Bare dataset ids no longer resolve and `trust_remote_code` was removed in
`datasets>=3.0`, so **without this the original cannot load wikitext at all**.
This is the same fix the fork made in `aca114a`.

**Patch 2 — `parcelmate/data.py`, CPU fallback in `correlate` (1 line).**
`use_gpu = use_gpu and torch.cuda.is_available()`. The original unconditionally
calls `torch.cuda.get_device_properties(0)`, which crashes on a machine without
CUDA. A no-op on a GPU node — it only unblocks running the smoke config locally.

Neither patch touches the connectivity, parcellation, or subnetwork maths. To
confirm that for yourself:

```bash
# from the repo root -- shows ONLY the two patches above
diff <(git show upstream/main:parcelmate/model.py) original/parcelmate/model.py
diff <(git show upstream/main:parcelmate/data.py)  original/parcelmate/data.py
```

Everything else here is scaffolding that does not exist upstream and so cannot
change its behaviour: `pyproject.toml` + `uv.lock` (copied from the fork, so both
sides resolve **identical** dependency versions — otherwise you would be comparing
library versions as much as code), `configs/`, `jobs/`, and this file.

## Comparing the two codebases

```bash
# Side-by-side of the whole package. The fork is ~1400 lines ahead, mostly
# knockout machinery in model.py/plot.py that has no upstream counterpart.
diff -ru original/parcelmate parcelmate

# Just the shared pipeline, ignoring the fork's additions:
diff -u original/parcelmate/data.py parcelmate/data.py
diff -u original/parcelmate/util.py parcelmate/util.py
git diff upstream/main HEAD --stat -- parcelmate/
```

Known fork-side changes to the **shared** code path (as opposed to net-new
knockout code), i.e. the candidates for any results difference:

| File | Change | Numerically relevant? |
|---|---|---|
| `data.py` | CPU fallback in `correlate` | No — same maths, different device |
| `util.py` | `f[key][:]` → `f[key][()]` in `load_h5_data` | No — read-side only, handles scalar datasets |
| `constants.py` | raw-string regex, extra name constants | No — silences a `SyntaxWarning` |
| `model.py` | dataset ids / `trust_remote_code` | No — same corpora, current ids |
| `model.py`, `plot.py` | knockout + stability additions | Net-new; not on the shared path |

That table is the prediction: **the shared pipeline should produce equivalent
results, and the run below is what tests it.** Parcellation is stochastic, so
"equivalent" means the stability/score distributions line up, not that network
indices match.

## Running it

The one rule: **run from inside this directory.** `python -m parcelmate.bin.main`
resolves the package from the current working directory first, so running from the
repo root would silently execute the *fork's* code. The cluster job asserts this;
when running by hand, check it:

```bash
cd original
python -c "import parcelmate, os; print(os.path.abspath(parcelmate.__file__))"
```

**Local smoke test** (CPU, ~1 min, numbers meaningless — just proves it runs).
Verified passing on 2026-07-30: connectivity for both domains → parcellation →
`subnetwork/parcellation_shared_avg.h5`.

```bash
cd original
../.venv/bin/python -m parcelmate.bin.main configs/smoke.yaml \
    -s connectivity parcellation subnetwork_extraction
```

Reusing the fork's `.venv` is just the fast path — it is built from the same
`pyproject.toml`/`uv.lock` copied here. For a genuinely isolated environment use
`uv run python -m ...` instead, which syncs a separate venv from this directory's
own copies.

Both domains in `configs/smoke.yaml` are synthetic and generated locally, so the
smoke test needs no dataset downloads (only the gpt2 weights, likely already
cached). Note it uses **two** domains deliberately: `subnetwork_extraction`
indexes `domains[0]`/`domains[1]`, so any single-domain config dies with a
`KeyError`. That is true of the fork too — the fork's own `bare.yaml` would hit
it — so it is shared behaviour, not a difference between the codebases.

**Matched cluster run** — `configs/match_iter2.yaml` copies every setting from the
fork's `parcelmate/configs/knockout_iter2.yaml`, so the code is the only variable:

```bash
ssh <CSID>@sc.stanford.edu
cd ~/parcelmate/original
sbatch jobs/match_iter2.pbs        # submit from THIS dir; the job cds to $SLURM_SUBMIT_DIR
squeue -u <CSID>
```

Outputs land in `/nlp/scr/schegini/parcelmate/original/match_iter2`, a separate
tree from every fork run — both pipelines resume from whatever h5s they find on
disk, so a shared `output_dir` would silently blend the two codebases' outputs.

The job runs `connectivity → parcellation → subnetwork_extraction` plus plots. It
does **not** run upstream's `subnetwork_knockout`: that is upstream's own earlier
knockout implementation, not the fork's selectivity analysis, so it is not a
like-for-like comparison and it is not free.

## What to compare afterwards

The two runs share the same output layout, so compare the same files across the
two `output_dir`s:

- `connectivity/connectivity_<domain>_avg.h5` — do the connectivity matrices agree?
  This is the most direct test; it is deterministic given the same data, so large
  differences here mean a real code divergence rather than sampling noise.
- `parcellation/` sample scores and the stability plots — parcellation is
  stochastic, so compare distributions, not per-index identity.
- `plots/` — the heatmaps are the fastest eyeball check.

If the connectivity matrices agree and the stability distributions overlap, the
fork's results are the method's, not the fork's edits.
