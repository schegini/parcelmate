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


def compute_stability_matrices(output_dir='results', verbose=False, indent=0):
    """Stability matrices for one run/condition directory, NaNs preserved.

    Within-domain stability correlates each pair of connectivity *samples* of a
    domain; between-domain stability correlates each pair of domain *averages*.

    Returns ``{'within': {domain: (labels, R)}, 'between': (labels, R) | None}``
    or None if there is no connectivity to read.
    """
    connectivity_dir = os.path.join(output_dir, CONNECTIVITY_NAME)
    if not os.path.isdir(connectivity_dir):
        return None

    # Units this condition lesioned, if `run_knockout` recorded them. They are
    # dropped outright rather than left to NaN-filtering, which catches only the
    # ones whose connectivity came out exactly undefined -- see run_knockout.
    drop = None
    selection_path = os.path.join(output_dir, '%s_selection%s' % (KNOCKOUT_NAME, EXTENSION))
    if os.path.exists(selection_path):
        drop = np.asarray(load_h5_data(selection_path, verbose=False)['selection']).astype(bool)

    samples_by_domain = {}
    averages_by_domain = {}
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


def summarize_knockout_stability(knockout_root, verbose=True, indent=0):
    """Distil every knockout condition under ``knockout_root`` into one row per
    (condition, scope) and write ``stability_summary.csv``.

    Scopes are ``betweendomain`` and ``withindomain_<domain>``. Condition names
    are ``network<i>_<mode>``, so the mode and network columns let mean-out and
    zero-out be compared network by network. Returns the DataFrame, or None if
    there was nothing to summarize."""
    if not os.path.isdir(knockout_root):
        return None

    rows = []
    for condition in sorted(os.listdir(knockout_root)):
        condition_dir = os.path.join(knockout_root, condition)
        if not os.path.isdir(condition_dir):
            continue
        stability = compute_stability_matrices(condition_dir, verbose=False, indent=indent)
        if stability is None:
            continue

        network, _, mode = condition.rpartition('_')
        if mode not in ('zero', 'mean'):  # not a name this module wrote
            network, mode = condition, ''

        scopes = {}
        if stability['between'] is not None:
            scopes['betweendomain'] = stability['between'][1]
        for domain, (_labels, R) in stability['within'].items():
            scopes['withindomain_%s' % domain] = R

        for scope, mat in scopes.items():
            median, n_pairs = distill_stability(mat)
            rows.append(dict(
                condition=condition,
                network=network,
                mode=mode,
                scope=scope,
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
