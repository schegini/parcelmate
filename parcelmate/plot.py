import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from parcelmate.constants import *
from parcelmate.util import stderr, load_h5_data


# ---------------------------------------------------------------------------
# Shared knockout-condition vocabulary
#
# Every per-network knockout figure (loss, perplexity, distilled stability) is
# built from the SAME five conditions in the SAME order and colours, so the bars
# line up across plots and the network-vs-random comparison is read at a glance:
#
#   healthy | mean-out network | mean-out random | zero-out network | zero-out random
#
# Hue encodes the knockout mode (blue = healthy reference, red = mean-out,
# orange = zero-out); within a mode the size-matched random baseline is the
# lighter, hatched bar so "real subnetwork vs its control" reads as a matched
# pair. Colours are drawn from the validated categorical palette (blue/red/orange
# = slots 1/8/6); the baselines are tints of their mode plus a hatch as a second,
# non-colour cue.
# ---------------------------------------------------------------------------
CONDITION_ORDER = ['healthy', 'mean_net', 'mean_rand', 'zero_net', 'zero_rand']
CONDITION_STYLE = {
    'healthy':   dict(label='healthy',         color='#2a78d6', hatch=None),
    'mean_net':  dict(label='mean-out network', color='#e34948', hatch=None),
    'mean_rand': dict(label='mean-out random',  color='#f1a6a5', hatch='///'),
    'zero_net':  dict(label='zero-out network', color='#eb6834', hatch=None),
    'zero_rand': dict(label='zero-out random',  color='#f6b89f', hatch='///'),
}
# Which (kind, mode) each condition slot corresponds to in the summaries.
_CONDITION_SELECTOR = {
    'mean_net':  ('knockout', 'mean'),
    'mean_rand': ('baseline', 'mean'),
    'zero_net':  ('knockout', 'zero'),
    'zero_rand': ('baseline', 'zero'),
}


def _masked_corr(a, b):
    """Pearson correlation of two vectors after dropping any position that is NaN
    in either vector. Knocked-out neurons have undefined (NaN) connectivity, so
    dropping NaNs here automatically removes them from the stability computation
    (as opposed to zero-filling, which would fold them in as spurious zeros).
    Returns NaN when fewer than two finite pairs survive or either side is
    constant."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 2:
        return np.nan
    va, vb = a[m], b[m]
    if np.std(va) == 0 or np.std(vb) == 0:
        return np.nan
    return float(np.corrcoef(va, vb)[0, 1])


def _tri_values(mat, k=-1):
    """Strict lower-triangle entries of a square matrix as a flat vector."""
    mat = np.asarray(mat, dtype=float)
    ix = np.tril_indices(mat.shape[0], k=k)
    return mat[ix]


def _parse_condition_name(name):
    """Classify a knockout condition directory / summary label into
    ``(kind, mode, network)``.

    ``kind``    -> 'healthy' | 'knockout' | 'baseline'
    ``mode``    -> 'mean' | 'zero' | '' (legacy single-mode runs)
    ``network`` -> e.g. 'network0' (or 'healthy'/'union').

    Mirrors the parsing in ``summarize_knockout_loss`` so the stability summary
    lines up with the loss summary condition-for-condition."""
    if name == HEALTHY_NAME:
        return 'healthy', '', HEALTHY_NAME
    if BASELINE_NAME in name:
        kind = 'baseline'
    else:
        kind = 'knockout'
    mode = ''
    for _m in ('mean', 'zero'):
        if ('_%s' % _m) in name or name == _m:
            mode = _m
            break
    if mode:
        network = name.split('_%s' % mode)[0]
    else:
        # Legacy single-mode name; strip a trailing _baselineN if present.
        network = name.split('_%s' % BASELINE_NAME)[0]
    return kind, mode, network


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


# ---------------------------------------------------------------------------
# Stability computation (shared by the per-condition heatmaps and the distilled
# per-network summary).
# ---------------------------------------------------------------------------
def compute_stability(
        output_dir='results',
        verbose=True,
        indent=0
):
    """Compute stability (reproducibility) matrices from the connectivity of one
    run/condition directory.

    Within-domain stability correlates each pair of connectivity *samples* of a
    domain; between-domain stability correlates each pair of domain *averages*.
    Correlations are taken over the strict lower triangle of the (absolute-value)
    connectivity matrices, dropping any neuron pair that is NaN in either operand
    -- so knocked-out neurons (whose connectivity is undefined) are excluded
    rather than zero-filled.

    Returns a dict::

        {
          'layers': [...],
          'within': {domain: {'labels': [...], 'R': arr, 'by_layer': {layer: arr}}},
          'between': {'labels': [...], 'R': arr, 'by_layer': {layer: arr}} | None,
        }
    """
    connectivity_dir = os.path.join(output_dir, CONNECTIVITY_NAME)
    if not os.path.isdir(connectivity_dir):
        return None

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
        if domain not in R_by_domain:
            R_by_domain[domain] = {}
        filepath = os.path.join(connectivity_dir, path)
        data = load_h5_data(filepath, verbose=verbose, indent=indent)
        # Keep NaNs: knocked-out neurons are undefined and must be dropped (not
        # zero-filled) when correlating.
        R = np.abs(np.asarray(data['connectivity'], dtype=float))
        R_by_domain[domain][key] = R
        if coordinates is None:
            coordinates = data['coordinates'][:, 0]  # 0th dimension is layer

    if coordinates is None:
        return None
    layers = np.unique(coordinates)

    def _pairwise(mats_by_label):
        """Correlation matrix over a dict {label: connectivity}; also per layer."""
        labels = sorted(mats_by_label.keys())
        n = len(labels)
        R = np.full((n, n), np.nan)
        R_by_layer = {layer: np.full((n, n), np.nan) for layer in layers}
        for i, li in enumerate(labels):
            Ri = mats_by_label[li]
            ix = np.tril_indices(Ri.shape[0], k=-1)
            for j, lj in enumerate(labels):
                Rj = mats_by_label[lj]
                R[i, j] = _masked_corr(Ri[ix], Rj[ix])
                for layer in layers:
                    sel = coordinates == layer
                    _Ri = Ri[sel, :][:, sel]
                    _Rj = Rj[sel, :][:, sel]
                    _ix = np.tril_indices(_Ri.shape[0], k=-1)
                    R_by_layer[layer][i, j] = _masked_corr(_Ri[_ix], _Rj[_ix])
        return labels, R, R_by_layer

    within = {}
    for domain, mats in samples_by_domain.items():
        labels, R, R_by_layer = _pairwise(mats)
        within[domain] = dict(labels=labels, R=R, by_layer=R_by_layer)

    between = None
    if len(averages_by_domain) >= 2:
        mats = {domain: averages_by_domain[domain]['avg'] for domain in averages_by_domain}
        labels, R, R_by_layer = _pairwise(mats)
        between = dict(labels=labels, R=R, by_layer=R_by_layer)

    return dict(layers=layers, within=within, between=between)


def _save_stability_heatmap(mat, labels, filepath, rotate_x=False):
    ax = sns.heatmap(
        pd.DataFrame(mat, index=labels, columns=labels),
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        xticklabels=True,
        yticklabels=True,
        annot=True,
    )
    if rotate_x:
        ax.tick_params(axis='x', rotation=45)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close('all')


def plot_stability(
        output_dir='results',
        verbose=True,
        indent=0
):
    plot_dir = os.path.join(output_dir, PLOT_DIR, STABILITY_NAME)

    if verbose:
        stderr('%sPlotting stability\n' % (' ' * indent))
    indent += 2

    stability = compute_stability(output_dir, verbose=verbose, indent=indent)
    if stability is None:
        return

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    for domain, block in stability['within'].items():
        labels = block['labels']
        if len(labels) < 2:
            continue
        _save_stability_heatmap(
            block['R'], labels,
            os.path.join(plot_dir, 'withindomain_%s.png' % domain))
        for layer, mat in block['by_layer'].items():
            _save_stability_heatmap(
                mat, labels,
                os.path.join(plot_dir, 'withindomain_%s_L%s.png' % (domain, layer)))

    between = stability['between']
    if between is not None:
        labels = between['labels']
        _save_stability_heatmap(
            between['R'], labels,
            os.path.join(plot_dir, 'betweendomain.png'), rotate_x=True)
        for layer, mat in between['by_layer'].items():
            _save_stability_heatmap(
                mat, labels,
                os.path.join(plot_dir, 'betweendomain_L%s.png' % layer), rotate_x=True)


def summarize_knockout_stability(knockout_root, verbose=True, indent=0):
    """Distil each knockout condition's stability heatmaps into a single number
    per scope: the median of the strict lower triangle of the stability matrix
    (NaN cells dropped). Scopes are ``betweendomain`` and ``withindomain_<domain>``.

    Writes ``stability_summary.csv`` alongside the loss summary and returns the
    DataFrame (or None if nothing to summarize)."""
    if not os.path.isdir(knockout_root):
        return None

    rows = []
    for name in sorted(os.listdir(knockout_root)):
        condition_dir = os.path.join(knockout_root, name)
        if not os.path.isdir(condition_dir):
            continue
        if not os.path.isdir(os.path.join(condition_dir, CONNECTIVITY_NAME)):
            continue
        stability = compute_stability(condition_dir, verbose=False, indent=indent)
        if stability is None:
            continue
        kind, mode, network = _parse_condition_name(name)

        scopes = {}
        if stability['between'] is not None:
            scopes['betweendomain'] = stability['between']['R']
        for domain, block in stability['within'].items():
            scopes['withindomain_%s' % domain] = block['R']

        for scope, mat in scopes.items():
            vals = _tri_values(mat)
            finite = vals[np.isfinite(vals)]
            rows.append(dict(
                condition=name,
                kind=kind,
                mode=mode,
                network=network,
                scope=scope,
                median_stability=float(np.median(finite)) if finite.size else np.nan,
                n_pairs=int(finite.size),
            ))

    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values(['scope', 'condition']).reset_index(drop=True)
    out_path = os.path.join(knockout_root, '%s_summary.csv' % STABILITY_NAME)
    df.to_csv(out_path, index=False)
    if verbose:
        stderr('%sWrote stability summary to %s\n' % (' ' * indent, out_path))
    return df


# ---------------------------------------------------------------------------
# Per-network knockout bar plots (loss / perplexity / distilled stability).
# ---------------------------------------------------------------------------
def _condition_value(sub, condition):
    """Aggregate the value(s) for one condition slot in a per-(network, facet)
    slice of a summary DataFrame. Returns ``(mean, lo_err, hi_err)`` where the
    error arms give the min/max range across baseline draws (0 for single-value
    conditions), or None when the slot has no data."""
    if condition == 'healthy':
        rows = sub[sub['kind'] == 'healthy']
    else:
        kind, mode = _CONDITION_SELECTOR[condition]
        rows = sub[(sub['kind'] == kind) & (sub['mode'] == mode)]
    if rows.empty:
        return None
    vals = rows['value'].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    m = float(vals.mean())
    return m, m - float(vals.min()), float(vals.max()) - m


def _draw_condition_bars(ax, sub, conditions):
    """Draw the (up to five) condition bars on ``ax`` from a per-facet slice.
    ``conditions`` is the ordered list of condition keys to attempt. Returns the
    subset actually drawn (those with data)."""
    drawn = []
    for i, cond in enumerate(conditions):
        agg = _condition_value(sub, cond)
        if agg is None:
            continue
        m, lo, hi = agg
        style = CONDITION_STYLE[cond]
        ax.bar([i], [m], width=0.8, color=style['color'], hatch=style['hatch'],
               edgecolor='white', linewidth=0.6, zorder=2)
        if lo > 0 or hi > 0:
            ax.errorbar([i], [m], yerr=[[lo], [hi]], fmt='none',
                        ecolor='#333333', elinewidth=1, capsize=3, zorder=3)
        drawn.append(cond)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([''] * len(conditions))
    ax.tick_params(axis='x', length=0)
    ax.grid(axis='y', color='#e5e5e5', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    return drawn


def _condition_legend(fig, conditions):
    handles = [mpatches.Patch(facecolor=CONDITION_STYLE[c]['color'],
                              hatch=CONDITION_STYLE[c]['hatch'],
                              edgecolor='white', label=CONDITION_STYLE[c]['label'])
               for c in conditions]
    fig.legend(handles=handles, loc='lower center', ncol=len(conditions),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0))


def _available_conditions(df):
    """Ordered condition slots present in a summary DataFrame (healthy always
    kept as the reference; mode-specific slots kept only if that mode ran)."""
    modes = set(df['mode'].dropna().astype(str))
    conditions = ['healthy']
    for cond in CONDITION_ORDER[1:]:
        _, mode = _CONDITION_SELECTOR[cond]
        if mode in modes:
            conditions.append(cond)
    return conditions


def _prep_knockout_df(df):
    """Normalize a loss summary for plotting: backfill legacy mode/network
    columns and drop the union condition (no longer analyzed)."""
    df = df.copy()
    if 'mode' not in df.columns:
        df['mode'] = ''
    if 'network' not in df.columns:
        df['network'] = df['condition']
    df['mode'] = df['mode'].fillna('').astype(str)
    df['network'] = df['network'].astype(str)
    df = df[df['network'] != 'union']
    return df


def _network_sort_key(network):
    """Sort 'network0','network1',... numerically; other labels lexically after."""
    if network.startswith('network'):
        try:
            return (0, int(network[len('network'):]))
        except ValueError:
            pass
    return (1, network)


def plot_knockout_loss(
        output_dir='results',
        verbose=True,
        indent=0
):
    """Per-network bar plots of LM loss and perplexity across the five knockout
    conditions (healthy / mean-out network / mean-out random / zero-out network /
    zero-out random). One figure per network per metric, faceted by domain so
    each domain keeps its own y-scale (perplexities differ by orders of magnitude
    across corpora)."""
    knockout_root = os.path.join(output_dir, KNOCKOUT_NAME)
    summary_path = os.path.join(knockout_root, '%s_summary.csv' % LOSS_NAME)
    plot_dir = os.path.join(output_dir, PLOT_DIR, KNOCKOUT_NAME)

    if not os.path.exists(summary_path):
        if verbose:
            stderr('%sNo knockout loss summary at %s; skipping plot\n' % (' ' * indent, summary_path))
        return

    if verbose:
        stderr('%sPlotting knockout loss/perplexity by network\n' % (' ' * indent))

    df = pd.read_csv(summary_path)
    if df.empty:
        return
    df = _prep_knockout_df(df)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    conditions = _available_conditions(df)
    networks = sorted({n for n in df['network'] if n.startswith('network')},
                      key=_network_sort_key)
    domains = sorted(df['domain'].unique())

    metrics = [('loss', 'loss', 'LM loss (cross-entropy)', False),
               ('perplexity', 'perplexity', 'perplexity', True)]

    for network in networks:
        net_df = df[(df['network'] == network) | (df['kind'] == 'healthy')]
        for col, tag, ylabel, log_y in metrics:
            m_df = net_df.rename(columns={col: 'value'})
            fig, axes = plt.subplots(
                1, len(domains),
                figsize=(max(3.0 * len(domains), 4.5), 4.2),
                squeeze=False)
            axes = axes[0]
            for ax, domain in zip(axes, domains):
                sub = m_df[m_df['domain'] == domain]
                _draw_condition_bars(ax, sub, conditions)
                if log_y:
                    ax.set_yscale('log')
                ax.set_title(domain, fontsize=10)
            axes[0].set_ylabel(ylabel)
            _condition_legend(fig, conditions)
            fig.suptitle('Knockout %s: %s' % (ylabel, network), fontsize=11)
            fig.tight_layout(rect=(0, 0.06, 1, 0.96))
            fig.savefig(os.path.join(plot_dir, 'knockout_%s_%s.png' % (network, tag)), dpi=150)
            plt.close('all')


def plot_knockout_stability(
        output_dir='results',
        verbose=True,
        indent=0
):
    """Per-network bar plots of distilled stability (median lower-triangle
    correlation) across the five knockout conditions. One figure per network,
    faceted by scope (between-domain and each within-domain), mirroring the
    loss/perplexity plots."""
    knockout_root = os.path.join(output_dir, KNOCKOUT_NAME)
    summary_path = os.path.join(knockout_root, '%s_summary.csv' % STABILITY_NAME)
    plot_dir = os.path.join(output_dir, PLOT_DIR, '%s_%s' % (STABILITY_NAME, KNOCKOUT_NAME))

    if not os.path.exists(summary_path):
        if verbose:
            stderr('%sNo knockout stability summary at %s; skipping plot\n' % (' ' * indent, summary_path))
        return

    if verbose:
        stderr('%sPlotting knockout stability by network\n' % (' ' * indent))

    df = pd.read_csv(summary_path)
    if df.empty:
        return
    df = _prep_knockout_df(df).rename(columns={'median_stability': 'value'})
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    conditions = _available_conditions(df)
    networks = sorted({n for n in df['network'] if n.startswith('network')},
                      key=_network_sort_key)
    # Between-domain first, then within-domain scopes alphabetically.
    scopes = sorted(df['scope'].unique(),
                    key=lambda s: (0, s) if s == 'betweendomain' else (1, s))

    for network in networks:
        net_df = df[(df['network'] == network) | (df['kind'] == 'healthy')]
        fig, axes = plt.subplots(
            1, len(scopes),
            figsize=(max(3.0 * len(scopes), 4.5), 4.2),
            squeeze=False)
        axes = axes[0]
        for ax, scope in zip(axes, scopes):
            sub = net_df[net_df['scope'] == scope]
            _draw_condition_bars(ax, sub, conditions)
            ax.axhline(0, color='#999999', linewidth=0.8, zorder=1)
            ax.set_title(scope.replace('withindomain_', 'within: '), fontsize=9)
        axes[0].set_ylabel('median stability (lower-triangle corr.)')
        _condition_legend(fig, conditions)
        fig.suptitle('Knockout stability: %s' % network, fontsize=11)
        fig.tight_layout(rect=(0, 0.06, 1, 0.96))
        fig.savefig(os.path.join(plot_dir, 'stability_%s.png' % network), dpi=150)
        plt.close('all')
