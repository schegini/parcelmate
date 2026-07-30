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
