# Project 3 — Subnetwork Knockout Selectivity: Run Log & Findings

**Goal (GOAL.md #3):** For a given number of neurons in a subnetwork, run a
baseline that knocks out the *same number* of neurons drawn randomly from the
remaining (un-knocked-out) units. Is the damage *selective* to the subnetwork,
or does knocking out any equal-sized random set hurt just as much? Also compare
**zero-out** vs **mean-out** (clamp knocked-out units to their cross-domain mean
activation) as the perturbation.

**Selectivity criterion:** a subnetwork is "special" for a domain iff its
knockout raises LM loss on that domain *more than* the size-matched random
baseline (baseline = same #units, disjoint, from the complement).

**Pipeline per run:** `connectivity → parcellation → subnetwork_extraction →
subnetwork_knockout`. Knockout re-runs connectivity + LM loss once per
condition: healthy, each `network<i>`, `network<i>_baseline<s>`, `union`,
`union_baseline<s>`, over every domain.

**Infra notes:**
- Runs on Stanford SC cluster (`sc` login node, `nlp` account, `sphinx`
  partition, 1 GPU). Outputs land in `/nlp/scr/schegini/parcelmate/...` and are
  rsynced into `parcelmate/c_outputs/<iter>/`.
- Quota fix (HEAD `0b003db`): HF/uv caches + per-job venv redirected to scratch.
  Earlier (Jul 8) knockout sweep failed with `Errno 122 Disk quota exceeded`
  because the HF cache hit the 20GB home quota — fixed, not a code bug in the
  knockout logic itself.

**Tests:** `test/test_knockout.py` (12 unit tests, stdlib `unittest`, no
GPU/model needed) guards the Project-3 invariants — inclusive threshold,
single-network vs union selection, and crucially that the random baseline is
**size-matched** and **disjoint** from the knockout. Run with
`python3 -m unittest discover -s test -p "test_*.py" -v`.

---

## Iteration index

| Iter | Purpose | Model | domains | n_networks | networks | n_baseline | tokens | status |
|------|---------|-------|---------|------------|----------|-----------|--------|--------|
| 0 (smoke) | validate pipeline + sync path | gpt2 | wikitext,tldr17 | 10 | [0,1] | 1 | 5k | done (outputs complete; cosmetic FAILED) |
| 1 | full Project-3 mean vs zero | gpt2 | wikitext,tldr17,codeparrot,random | 50 | [0,1,2] | 3 | 50k | done (COMPLETED, clean selectivity) |
| 2 | threshold × mode sweep | gpt2 | (4 domains) | 50 | [0-4] | 3 | 50k | done — **thresh 0.5 wins; 0.9 kills signal** |
| 3 | model-scale | gpt2-medium, gpt2-large | (4 domains) | 50 | [0,1,2] | 3 | 50k | OOM at `-m 48`; re-running at `-m 384` (resumes from cached connectivity) |

---

## Iteration 0 — smoke test

**Rationale:** tiny/cheap end-to-end validation that (a) jobs run to completion
under the quota fix, (b) mean-out and zero-out both produce a `loss_summary.csv`
+ knockout plots, and (c) the rsync-into-`c_outputs` path works. Not for
scientific conclusions (5k tokens, 10 networks — very noisy).

**Outputs:** `c_outputs/iter0_smoke/` (loss summaries, knockout plots, dashboard,
logs). Full intermediate h5s left on scratch at
`/nlp/scr/schegini/parcelmate/sweeps/knockout_smoke/`.

**Result — pipeline validated.** Both mean & zero runs completed connectivity
→ parcellation → subnetwork_extraction → knockout for all conditions and wrote
`loss_summary.csv` + all plots.

**Selectivity signal is real (mean-out):** `network0` knockout raises LM loss
above *both* healthy and its size-matched random baseline —
- wikitext: healthy 4.33 → **network0 KO 9.66** vs network0_baseline 8.34
- tldr17:   healthy 3.92 → **network0 KO 8.42** vs network0_baseline 7.56

i.e. removing *this specific* subnetwork hurts more than removing an equal number
of random units. `network1` had ~no effect (it is a near-empty/unimportant
network at n_networks=10). This is exactly the Project-3 control working.

### Issues found → fixes for next iterations
1. **Cosmetic job failure.** Jobs exit non-zero via a `PyGILState_Release ...
   Fatal Python error` at interpreter *shutdown* — a lingering HF-datasets
   streaming thread (tldr17/Reddit is streamed) tearing down during finalize.
   Happens *after* all outputs are written, so it is harmless. **→ Detect run
   success by presence of `loss_summary.csv` + plots, NOT by SLURM exit code.**
2. **Networks too big → baselines skipped.** At n_networks=10, single networks
   exceed half the units (`union` = 6193/9984), so `_run_baselines` skips them
   ("knocked out exceeds complement") — we lose the control exactly where the
   effect is largest. **→ Iteration 1 uses n_networks=50** so individual
   networks are ~200 units and baseline-able (union will still skip, expected).
3. **Cross-mode parcellation mismatch (confound).** mean and zero are *separate*
   pipeline runs with independently sampled parcellations, so `network1(mean)` ≠
   `network1(zero)` — per-index mean-vs-zero comparison is invalid. The
   *within-run* selectivity test (KO vs its own baseline) is still valid.
   **→ Treat mean & zero as independent selectivity tests for now; a shared-
   parcellation mode is a candidate code change if we want a matched comparison.**

---

## Iteration 1 — full Project-3 run (gpt2, 50 networks, 3 baselines, 50k tokens)

**Both jobs COMPLETED cleanly** (no shutdown crash this time — datasets were
already cached, so no lingering streaming thread). Outputs in `c_outputs/iter1/`
(loss summaries, knockout bar-plots per domain, `collect` dashboard). Full h5s on
scratch at `/nlp/scr/schegini/parcelmate/sweeps/knockout/`.

**Headline: the selectivity control passes decisively.** For nearly every
subnetwork×domain, the real knockout raises LM loss above the *maximum* of its 3
size-matched random baselines (`selective? = YES` = KO beats even the worst
baseline draw). Baselines sit essentially at the healthy loss — removing an equal
number of *random* units barely hurts, while removing *the subnetwork* hurts a
lot. That is exactly the Project-3 claim: the damage is localized, not a generic
capacity effect.

Selectivity = knockout loss − mean(baseline loss), gpt2. (network indices differ
between the mean and zero runs — independent parcellations; see issue 3.)

**mean-out**
| domain | net | healthy | KO | base_mean | KO−base | selective |
|---|---|---|---|---|---|---|
| codeparrot | network2 | 2.16 | **3.77** | 2.10 | +1.66 | YES |
| codeparrot | network1 | 2.16 | 2.46 | 2.41 | +0.04 | ~ (not code-selective) |
| wikitext | network1 | 3.90 | **5.60** | 3.90 | +1.71 | YES |
| wikitext | network2 | 3.90 | **5.01** | 3.93 | +1.08 | YES |
| tldr17 | network1 | 3.68 | **5.17** | 3.73 | +1.44 | YES |
| tldr17 | network2 | 3.68 | **4.58** | 3.73 | +0.85 | YES |
| union | (all) | — | 9–13 | 5.6–7.6 | +3.7–5.6 | YES (huge) |

**zero-out** (network0/network2): same story — network0 & network2 both
selective across wikitext/tldr17/codeparrot (e.g. wikitext network2 KO 6.15 vs
base 3.95; tldr17 network2 KO 5.70 vs 3.76). So the selectivity is **robust to
the perturbation mode** (mean-out vs zero-out).

**Notes / caveats**
- **`random` domain is a weak control**, as intended: healthy loss is already
  ~11.8 (model can't predict random tokens), so KO/baseline differences are large
  but noisy and not linguistically meaningful. Good sanity check, not a result.
- **network1 is language- but not code-selective** (codeparrot KO≈baseline) —
  first hint that different subnetworks carry different domain functions
  (relevant to Project 2). network2 hurts *all* domains incl. code.

---

## Iteration 2 — knockout threshold × perturbation mode (gpt2, networks [0–4])

All 4 jobs COMPLETED. Outputs in `c_outputs/iter2/`. **Key result: the knockout
threshold matters a lot, and higher is NOT better.**

| mode | thresh | networks with units | selectivity |
|---|---|---|---|
| mean | 0.5 | network0, network1, network4 | strong (KO−base up to +2.0, mostly YES) |
| mean | **0.9** | **only network1** | **gone** (KO ≈ healthy ≈ baseline) |
| zero | 0.5 | network1, network2, network3 | strong (network1/3 YES +1.2–2.5) |
| zero | **0.9** | **only network1** | **gone** |

**Interpretation.** At `thresh=0.9` almost no unit has >0.9 single-network
membership (parcellation is soft), so selections are tiny/empty and knocking them
out does nothing. The function lives in the **broad 0.5–0.9 membership mass** of a
subnetwork, not a small high-confidence core. **→ Keep `knockout_thresh=0.5`.**
GOAL.md's "raise threshold toward 1.0" only helps keep the *union* selection
under half the units (baseline-able); for *individual* networks 0.5 is already
baseline-able and 0.9 over-prunes.

**Secondary observations (thresh 0.5).**
- Some individual networks show baseline > knockout on a domain (e.g. mean
  codeparrot network0: KO 4.01 < base 4.79; zero network2 across domains) — noisy
  single draws where a random baseline happens to hit a critical unit. The
  *language* domains (wikitext/tldr17) are consistently clean; more baselines
  (n_baseline↑) would tighten the null. **→ candidate: n_baseline 3→5.**
- Confirms iter1: network1 is language-selective but weak on code.
- **`union` knockouts are catastrophic** but also have elevated baselines (they
  remove a large fraction of units); the single-network results are the clean
  ones. Union is a whole-model sanity check, not the selectivity headline.
- **Empty networks**: at 50 networks some indices have 0 units above threshold
  and are skipped (network0 empty in the mean run, network1 empty in the zero
  run). Iteration 2's higher threshold + population of networks [0–4] probes this.
