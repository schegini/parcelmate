import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from parcelmate.constants import *
from parcelmate.util import stderr, load_h5_data


def plot_connectivity(
        output_dir='results',
        verbose=True,
        indent=0
):
    connectivity_dir = os.path.join(output_dir, CONNECTIVITY_NAME)
    plot_dir = os.path.join(output_dir, PLOT_DIR, CONNECTIVITY_NAME)

    if verbose:
        stderr('Plotting connectivity\n')
    indent += 2

    for path in os.listdir(connectivity_dir):
        match = INPUT_NAME_RE.match(path)
        if match and match.group(1) == CONNECTIVITY_NAME:
            domain = match.group(2)
        else:
            continue
        key = match.group(3)
        if key != 'avg':
            continue
        filepath = os.path.join(connectivity_dir, path)
        data = load_h5_data(filepath, verbose=verbose, indent=indent)
        R = np.nan_to_num(data['connectivity'])
        coordinates = data['coordinates'][:, 0]  # 0th dimension is layer
        layers = np.unique(coordinates)
        for layer in layers:
            sel = coordinates == layer
            _R = R[sel, :][:, sel]

            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            filepath = os.path.join(plot_dir, '%s_%s_L%s.png' % (CONNECTIVITY_NAME, domain, layer))
            ax = sns.clustermap(
                _R,
                cmap='coolwarm',
                vmin=-1,
                vmax=1,
                xticklabels=False,
                yticklabels=False,
                annot=False,
            )
            ax.ax_row_dendrogram.set_visible(False)
            ax.ax_col_dendrogram.set_visible(False)
            fig = ax._figure
            fig.savefig(filepath, dpi=150)
            plt.close('all')


def plot_parcellation(
        output_dir='results',
        verbose=True,
        indent=0
):
    connectivity_dir = os.path.join(output_dir, CONNECTIVITY_NAME)
    subnetwork_dir = os.path.join(output_dir, SUBNETWORK_NAME)
    plot_dir = os.path.join(output_dir, PLOT_DIR, PARCELLATION_NAME)

    if verbose:
        stderr('Plotting parcellations\n')
    indent += 2

    parents = (subnetwork_dir, connectivity_dir)
    for parent in parents:
        paths = os.listdir(parent)
        for path in paths:
            match = INPUT_NAME_RE.match(path)
            if match and match.group(1) in (CONNECTIVITY_NAME, PARCELLATION_NAME):
                domain = match.group(2)
            else:
                continue
            key = match.group(3)
            if key != 'avg':
                continue
            filepath = os.path.join(parent, path)
            data = load_h5_data(filepath, verbose=verbose, indent=indent)
            if 'parcellation' not in data:
                continue
            parcellation = data['parcellation']
            coordinates = data['coordinates'][:, 0]  # 0th dimension is layer
            counts_by_layer = {x: y for x, y in zip(*np.unique(coordinates, return_counts=True))}
            layers = sorted(list(counts_by_layer.keys()))
            n_layers = len(layers)
            n_units = max(*counts_by_layer.values())
            n_networks = parcellation.shape[-1]
            for i in range(n_networks):
                out = np.zeros((n_units, n_layers))
                for j, layer in enumerate(layers):
                    sel = coordinates == layer
                    out[:, j] = parcellation[sel, i]

                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)

                filepath = os.path.join(plot_dir, '%s_%s_network%d.png' % (PARCELLATION_NAME, domain, i + 1))
                ax = sns.heatmap(
                    pd.DataFrame(out, index=range(n_units), columns=layers),
                    cmap='Blues',
                    vmin=0,
                    vmax=1,
                    xticklabels=True,
                    yticklabels=False,
                    annot=False,
                )
                fig = ax.get_figure()
                fig.savefig(filepath, dpi=150)
                plt.close('all')


def plot_stability(
        output_dir='results',
        verbose=True,
        indent=0
):
    connectivity_dir = os.path.join(output_dir, CONNECTIVITY_NAME)
    plot_dir = os.path.join(output_dir, PLOT_DIR, STABILITY_NAME)

    if verbose:
        stderr('%sPlotting stability\n' % (' ' * indent))
    indent += 2

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    samples_by_domain = {}
    averages_by_domain = {}
    coordinates = None
    for path in os.listdir(connectivity_dir):
        match = INPUT_NAME_RE.match(path)
        if match and match.group(1) == CONNECTIVITY_NAME:
            domain = match.group(2)
        else:
            continue
        key = match.group(3)
        if key == 'avg':
            R_by_domain = averages_by_domain
        else:
            key = int(key[len(SAMPLE_NAME):])
            R_by_domain = samples_by_domain
        if not domain in R_by_domain:
            R_by_domain[domain] = {}
        filepath = os.path.join(connectivity_dir, path)
        data = load_h5_data(filepath, verbose=verbose, indent=indent)
        R = np.nan_to_num(data['connectivity'])
        R = np.abs(R)
        R_by_domain[domain][key] = R
        if coordinates is None:
            coordinates = data['coordinates'][:, 0]  # 0th dimension is layer

    layers = np.unique(coordinates)

    for domain in samples_by_domain:
        n = len(samples_by_domain[domain])
        R = np.zeros((n, n))
        R_by_layer = {layer: np.zeros((n, n)) for layer in layers}
        labels = sorted(list(samples_by_domain[domain].keys()))
        for i, key1 in enumerate(labels):
            if key1 == 'avg':
                continue
            R1 = samples_by_domain[domain][key1]
            ix = np.tril_indices(R1.shape[0], k=-1)
            for j, key2 in enumerate(labels):
                if key2 == 'avg':
                    continue
                R2 = samples_by_domain[domain][key2]
                R[i, j] = np.corrcoef(R1[ix], R2[ix])[0, 1]
                for layer in layers:
                    sel = coordinates == layer
                    _R1 = R1[sel, :][:, sel]
                    _R2 = R2[sel, :][:, sel]
                    _ix = np.tril_indices(_R1.shape[0], k=-1)
                    R_by_layer[layer][i, j] = np.corrcoef(_R1[_ix], _R2[_ix])[0, 1]

        filepath = os.path.join(plot_dir, 'withindomain_%s.png' % domain)
        ax = sns.heatmap(
            pd.DataFrame(R, index=labels, columns=labels),
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            xticklabels=True,
            yticklabels=True,
            annot=True
        )
        fig = ax.get_figure()
        fig.savefig(filepath, dpi=150)
        plt.close('all')

        for layer in R_by_layer:
            filepath = os.path.join(plot_dir, 'withindomain_%s_L%s.png' % (domain, layer))
            ax = sns.heatmap(
                pd.DataFrame(R_by_layer[layer], index=labels, columns=labels),
                cmap='coolwarm',
                vmin=-1,
                vmax=1,
                xticklabels=True,
                yticklabels=True,
                annot=True
            )
            fig = ax.get_figure()
            fig.savefig(filepath, dpi=150)
            plt.close('all')

    labels = sorted(list(averages_by_domain.keys()))
    n = len(labels)
    R = np.zeros((n, n))
    R_by_layer = {layer: np.zeros((n, n)) for layer in layers}
    for i, domain1 in enumerate(labels):
        R1 = averages_by_domain[domain1]['avg']
        ix = np.tril_indices(R1.shape[0], k=-1)
        for j, domain2 in enumerate(labels):
            R2 = averages_by_domain[domain2]['avg']
            R[i, j] = np.corrcoef(R1[ix], R2[ix])[0, 1]
            for layer in layers:
                sel = coordinates == layer
                _R1 = R1[sel, :][:, sel]
                _R2 = R2[sel, :][:, sel]
                _ix = np.tril_indices(_R1.shape[0], k=-1)
                R_by_layer[layer][i, j] = np.corrcoef(_R1[_ix], _R2[_ix])[0, 1]

    filepath = os.path.join(plot_dir, 'betweendomain.png')
    ax = sns.heatmap(
        pd.DataFrame(R, index=labels, columns=labels),
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        xticklabels=True,
        yticklabels=True,
        annot=True
    )
    ax.tick_params(axis='x', rotation=45)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close('all')

    for layer in R_by_layer:
        filepath = os.path.join(plot_dir, 'betweendomain_L%s.png' % layer)
        ax = sns.heatmap(
            pd.DataFrame(R_by_layer[layer], index=labels, columns=labels),
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            xticklabels=True,
            yticklabels=True,
            annot=True
        )
        ax.tick_params(axis='x', rotation=45)
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close('all')


# ---------------------------------------------------------------------------
# ADDED (3): distilled stability.
#
# `plot_stability` above is left exactly as upstream wrote it, heatmaps and all.
# What follows is additive: it reduces each stability matrix to ONE number so
# that "how much did this lesion destabilize the model?" can be read off a table
# and compared across conditions, instead of eyeballed across heatmaps.
#
# The number is the median of the strict lower triangle of the stability matrix.
# Lower triangle because the matrix is symmetric with a trivial unit diagonal, so
# the strict lower triangle holds each distinct sample pair exactly once; median
# because a single degenerate pair should not drag the summary the way a mean
# would.
#
# One deliberate difference from `plot_stability`: it calls `np.nan_to_num` on
# the connectivity, which turns a knocked-out neuron's undefined correlations
# into zeros and folds them into the stability estimate as if they were real
# measurements. A lesioned neuron is constant, so its connectivity is genuinely
# undefined -- here those pairs are DROPPED instead. Zero-filling would make
# stability look like it fell simply because more cells read zero. The heatmaps
# and this summary can therefore disagree on knocked-out conditions; the summary
# is the one to trust after a lesion.
# ---------------------------------------------------------------------------
def _masked_corr(a, b):
    """Pearson correlation of two vectors, dropping positions that are NaN in
    either. Returns NaN if fewer than two finite pairs survive, or if either side
    is constant (correlation undefined)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 2:
        return np.nan
    va, vb = a[m], b[m]
    if np.std(va) == 0 or np.std(vb) == 0:
        return np.nan
    return float(np.corrcoef(va, vb)[0, 1])


def load_knockout_selection(condition_dir):
    """The boolean unit mask a knockout condition lesioned, or None if the
    condition did not record one (e.g. the healthy run)."""
    path = os.path.join(condition_dir, '%s_selection%s' % (KNOCKOUT_NAME, EXTENSION))
    if not os.path.exists(path):
        return None
    return np.asarray(load_h5_data(path, verbose=False)['selection']).astype(bool)


def compute_stability_matrices(output_dir='results', drop=None, domains=None,
                               verbose=False, indent=0):
    """Stability matrices for one run/condition directory, NaNs preserved.

    Within-domain stability correlates each pair of connectivity *samples* of a
    domain; between-domain stability correlates each pair of domain *averages*.

    ``drop`` is an optional boolean mask over units to exclude before
    correlating. When None, a ``knockout_selection.h5`` in ``output_dir`` is used
    if present. Passing it explicitly is what lets the HEALTHY run be measured
    over exactly the units some lesion removed, so the two are comparable.

    ``domains`` optionally restricts which corpora are read. Needed because a
    cross-domain mean-out condition holds its clamp corpus out, so its
    ``betweendomain`` figure covers fewer domain pairs than an unrestricted
    reference -- comparing the two would mostly measure which corpus was
    dropped. Passing the same subset makes them like-for-like.

    Returns ``{'within': {domain: (labels, R)}, 'between': (labels, R) | None}``
    or None if there is no connectivity to read.
    """
    connectivity_dir = os.path.join(output_dir, CONNECTIVITY_NAME)
    if not os.path.isdir(connectivity_dir):
        return None

    # Units this condition lesioned, if `run_knockout` recorded them. They are
    # dropped outright rather than left to NaN-filtering, which catches only the
    # ones whose connectivity came out exactly undefined -- see run_knockout.
    if drop is None:
        drop = load_knockout_selection(output_dir)
    else:
        drop = np.asarray(drop).astype(bool)

    samples_by_domain = {}
    averages_by_domain = {}
    for path in os.listdir(connectivity_dir):
        match = INPUT_NAME_RE.match(path)
        if match and match.group(1) == CONNECTIVITY_NAME:
            domain = match.group(2)
        else:
            continue
        if domains is not None and domain not in domains:
            continue
        key = match.group(3)
        if key == 'avg':
            R_by_domain = averages_by_domain
        else:
            key = int(key[len(SAMPLE_NAME):])
            R_by_domain = samples_by_domain
        if domain not in R_by_domain:
            R_by_domain[domain] = {}
        data = load_h5_data(os.path.join(connectivity_dir, path), verbose=verbose, indent=indent)
        # NaNs deliberately kept here (see module note above).
        R = np.abs(np.asarray(data['connectivity'], dtype=float))
        if drop is not None and drop.shape[0] == R.shape[0]:
            keep = ~drop
            R = R[keep, :][:, keep]
        R_by_domain[domain][key] = R

    def _pairwise(mats_by_label):
        labels = sorted(mats_by_label.keys())
        n = len(labels)
        R = np.full((n, n), np.nan)
        for i, li in enumerate(labels):
            Ri = mats_by_label[li]
            ix = np.tril_indices(Ri.shape[0], k=-1)
            for j, lj in enumerate(labels):
                R[i, j] = _masked_corr(Ri[ix], mats_by_label[lj][ix])
        return labels, R

    within = {}
    for domain, mats in samples_by_domain.items():
        if len(mats) >= 2:
            within[domain] = _pairwise(mats)

    between = None
    if len(averages_by_domain) >= 2:
        between = _pairwise({d: averages_by_domain[d]['avg'] for d in averages_by_domain})

    return dict(within=within, between=between)


def distill_stability(mat):
    """Reduce a stability matrix to a single scalar: the median of its strict
    lower triangle, ignoring NaN cells. Returns ``(median, n_pairs)``, with
    median NaN when no finite pair survives."""
    mat = np.asarray(mat, dtype=float)
    vals = mat[np.tril_indices(mat.shape[0], k=-1)]
    finite = vals[np.isfinite(vals)]
    if not finite.size:
        return np.nan, 0
    return float(np.median(finite)), int(finite.size)


def _stability_rows(stability, condition, network, mode, clamp_domain):
    """One row per scope for an already-computed stability result."""
    scopes = {}
    if stability['between'] is not None:
        scopes['betweendomain'] = stability['between'][1]
    for domain, (_labels, R) in stability['within'].items():
        scopes['withindomain_%s' % domain] = R

    rows = []
    for scope, mat in scopes.items():
        median, n_pairs = distill_stability(mat)
        rows.append(dict(
            condition=condition,
            network=network,
            mode=mode,
            clamp_domain=clamp_domain,
            # For a cross-domain mean-out the clamp corpus IS the corpus held
            # out of the evaluation, so the two coincide. `holdout` is the column
            # to group on when comparing against matched references.
            holdout=clamp_domain,
            scope=scope,
            median_stability=median,
            n_pairs=n_pairs,
        ))
    return rows


def summarize_knockout_stability(knockout_root, healthy_dir=None, verbose=True, indent=0):
    """Distil every knockout condition under ``knockout_root`` into one row per
    (condition, scope) and write ``stability_summary.csv``.

    Scopes are ``betweendomain`` and ``withindomain_<domain>``. Condition names
    are ``network<i>_<mode>``, so the mode and network columns let mean-out and
    zero-out be compared network by network.

    ADDED (5): when ``healthy_dir`` points at the run root (whose
    ``connectivity/`` is the UNPERTURBED model), one ``mode='healthy'`` row is
    emitted per lesioned network, computed over exactly the units that network's
    lesion removed. Without a reference, a lone stability number says nothing --
    there is no way to tell a damaged model from one that was never very stable.
    Matching the unit set per network matters because otherwise part of any
    healthy-vs-lesion gap is just "these are different neurons"; at 1232 units
    that is 12% of the model.

    Returns the DataFrame, or None if there was nothing to summarize."""
    if not os.path.isdir(knockout_root):
        return None

    rows = []
    masks_by_network = {}
    holdout_sets = {}
    for condition in sorted(os.listdir(knockout_root)):
        condition_dir = os.path.join(knockout_root, condition)
        if not os.path.isdir(condition_dir):
            continue
        stability = compute_stability_matrices(condition_dir, verbose=False, indent=indent)
        if stability is None:
            continue

        network, _, mode = condition.rpartition('_')
        # `mean-<corpus>` marks a clamp to one corpus's mean, with that corpus
        # held out of the evaluation (see run_knockout).
        clamp_domain = ''
        if mode.startswith('mean-'):
            mode, clamp_domain = 'mean', mode[len('mean-'):]
        if mode not in ('zero', 'mean'):  # not a name this module wrote
            network, mode, clamp_domain = condition, '', ''

        rows.extend(_stability_rows(stability, condition, network, mode, clamp_domain))

        # Every mode of a given network lesions the same units, so one healthy
        # reference per network covers them all.
        mask = load_knockout_selection(condition_dir)
        if mask is not None and network not in masks_by_network:
            masks_by_network[network] = mask
        # A cross-domain mean-out condition saw only the corpora it was NOT
        # clamped to; record that subset so matched references can be built.
        if clamp_domain:
            evaluated = sorted(stability['within'])
            if evaluated:
                holdout_sets.setdefault(network, {})[clamp_domain] = evaluated

    # ADDED (5): the unperturbed reference, one row per lesioned network, over
    # that network's surviving units. No model runs needed -- the run root's
    # connectivity/ IS the healthy model, already written by the pipeline.
    if healthy_dir and os.path.isdir(os.path.join(healthy_dir, CONNECTIVITY_NAME)):
        for network, mask in sorted(masks_by_network.items()):
            if verbose:
                stderr('%sHealthy reference for %s (%d units held out)\n' % (
                    ' ' * indent, network, int(mask.sum())))
            stability = compute_stability_matrices(
                healthy_dir, drop=mask, verbose=False, indent=indent
            )
            if stability is None:
                continue
            rows.extend(_stability_rows(
                stability, 'healthy_vs_%s' % network, network, 'healthy', ''
            ))

            # Matched-subset references. A mean-out condition that held out
            # corpus X has a `betweendomain` figure over the remaining corpora
            # only; an unrestricted reference covers more pairs, so the two
            # differ mostly by which corpus was dropped. Emit healthy AND
            # zero-out restricted to the same subset, tagged with `holdout`, so
            # every mean-out row has something legitimate to be read against.
            # Only `betweendomain` needs this -- within-domain figures do not
            # depend on which other corpora were present.
            for holdout, evaluated in sorted(holdout_sets.get(network, {}).items()):
                for ref_dir, ref_label, ref_mode in (
                        (healthy_dir, 'healthy_vs_%s' % network, 'healthy'),
                        (os.path.join(knockout_root, '%s_zero' % network),
                         '%s_zero' % network, 'zero'),
                ):
                    if not os.path.isdir(os.path.join(ref_dir, CONNECTIVITY_NAME)):
                        continue
                    ref = compute_stability_matrices(
                        ref_dir, drop=mask, domains=evaluated,
                        verbose=False, indent=indent
                    )
                    if ref is None or ref['between'] is None:
                        continue
                    median, n_pairs = distill_stability(ref['between'][1])
                    rows.append(dict(
                        condition='%s_ex-%s' % (ref_label, holdout),
                        network=network,
                        mode=ref_mode,
                        clamp_domain='',
                        holdout=holdout,
                        scope='betweendomain',
                        median_stability=median,
                        n_pairs=n_pairs,
                    ))

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values(['scope', 'condition']).reset_index(drop=True)
    out_path = os.path.join(knockout_root, '%s_summary.csv' % STABILITY_NAME)
    df.to_csv(out_path, index=False)
    if verbose:
        stderr('%sWrote stability summary to %s\n' % (' ' * indent, out_path))
    return df
