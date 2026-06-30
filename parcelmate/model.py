import math
import os
import copy
import numpy as np
from scipy import optimize
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import MiniBatchKMeans
import torch
from transformers import AutoModel, AutoTokenizer

from parcelmate.constants import *
from parcelmate.data import *
from parcelmate.util import *
from parcelmate.plot import *


class PerturbedModel(torch.nn.Module):
    def __init__(self, model, perturbation_coordinates, perturbation_values=None, *args, **kwargs):
        super(PerturbedModel, self).__init__(*args, **kwargs)
        self.model = model
        self.perturbation_coordinates = perturbation_coordinates
        if perturbation_values is None:  # Default to zero (knockout)
            perturbation_values = np.zeros(len(self.perturbation_coordinates))
        assert len(perturbation_values) == len(perturbation_coordinates), \
            'perturbation_values must match perturbation_coordinates'
        self.perturbation_values = perturbation_values

        layers_attr = 'h'
        layer_indices = np.unique(perturbation_coordinates[:, 0])  # 0th dimension is layer
        layers = getattr(self.model, layers_attr)
        perturbation_coordinate_tensors = {}
        perturbation_value_tensors = {}
        for l_ix in layer_indices:
            if l_ix == 0:
                key = 'embedding'
            else:
                key = l_ix - 1 # Shifted down bc of embedding layer
            sel = perturbation_coordinates[:, 0] == l_ix  # 0th dimension is layer
            layer_coordinates = perturbation_coordinates[sel][:, 1]  # 1st dimension is hidden unit
            layer_coordinates = torch.nn.Parameter(
                torch.as_tensor(
                    layer_coordinates
                ),
                requires_grad=False
            )
            perturbation_coordinate_tensors[key] = layer_coordinates
            layer_values = perturbation_values[sel]
            layer_values = torch.nn.Parameter(
                torch.as_tensor(
                    layer_values,
                    dtype=model.dtype
                ),
                requires_grad=False
            )
            perturbation_value_tensors[key] = layer_values

        self.perturbation_coordinate_tensors = perturbation_coordinate_tensors
        self.perturbation_value_tensors = perturbation_value_tensors

        for l_ix in layer_indices:
            if l_ix == 0:
                _l_ix = 'embedding'
                source_layer = self.model.drop
            else:
                _l_ix = l_ix - 1  # Shifted down bc of embedding layer
                source_layer = layers[_l_ix]
            layer = PerturbedLayer(
                source_layer,
                perturbation_coordinates=self.perturbation_coordinate_tensors[_l_ix],
                perturbation_values=self.perturbation_value_tensors[_l_ix]
            )
            if _l_ix == 'embedding':
                self.model.drop = layer
            else:
                layers[_l_ix] = layer

    def forward(self, *args, **kwargs):
        out = self.model.forward(*args, **kwargs)
        return out


class PerturbedLayer(torch.nn.Module):
    def __init__(self, layer, perturbation_coordinates=None, perturbation_values=None, *args, **kwargs):
        super(PerturbedLayer, self).__init__(*args, **kwargs)
        self.layer = layer
        self.perturbation_coordinates = perturbation_coordinates
        self.perturbation_values = perturbation_values

    def forward(self, *args, **kwargs):
        out = self.layer.forward(*args, **kwargs)
        if self.perturbation_coordinates is not None:
            if isinstance(out, torch.Tensor):
                out[..., self.perturbation_coordinates] = self.perturbation_values
            else:
                out0 = out[0]
                out0[..., self.perturbation_coordinates] = self.perturbation_values
                out = (out0,) + out[1:]

        return out


def get_model_and_tokenizer(model_name, knockout_probs=None, knockout_thresh=0.5, coordinates=None):
    model = AutoModel.from_pretrained(model_name)
    if knockout_probs is not None:
        assert coordinates is not None, 'coordinates must be provided if knockout_probs is not None'
        sel = None
        for ix in range(knockout_probs.shape[1]):
            inv_mask_ = knockout_probs[:, ix] >= knockout_thresh
            if sel is None:
                sel = inv_mask_
            else:
                sel |= inv_mask_
        perturbation_coordinates = coordinates[sel]
        perturbation_values = None
        model = PerturbedModel(
            model,
            perturbation_coordinates=perturbation_coordinates,
            perturbation_values=perturbation_values
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def get_timecourses(
        model,
        input_ids,
        attention_mask,
        batch_size=8,
        highpass=None,
        lowpass=None,
        step=0.2,
        timecourse_pca_components=None,
        timecourse_ica_components=None,
        verbose=True,
        indent=0,
        **kwargs
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if verbose:
        stderr('%sGetting timecourses\n' % (' ' * indent))
    timecourses = None
    coordinates = None
    t = 0
    T = int(attention_mask.detach().numpy().sum())
    B = int(math.ceil(input_ids.size(0) / batch_size))
    indent += 2
    for i in range(0, input_ids.size(0), batch_size):
        if verbose:
            stderr('\r%sBatch %d/%d' % (' ' * indent, i // batch_size + 1, B))
        _input_ids = input_ids[i:i + batch_size].to(device)
        _attention_mask = attention_mask[i:i + batch_size].to(device)
        states = model(
            input_ids=_input_ids,
            attention_mask=_attention_mask,
            output_hidden_states=True,
            **kwargs
        ).hidden_states
        mask = _attention_mask.detach().cpu().numpy().astype(bool)
        _t = int(mask.sum())
        if timecourses is None:
            out_shape = (sum(x.shape[-1] for x in states), T)
            timecourses = np.zeros(out_shape, dtype=np.float32)
        if coordinates is None:
            coordinates = np.zeros((sum(x.shape[-1] for x in states), 2), dtype=np.int32)
        h = 0
        for s, state in enumerate(states):
            _h = state.size(-1)
            timecourses[h:h + _h, t:t + _t] = bandpass(
                state.detach().cpu().numpy()[mask].T,
                step=step,
                lower=highpass,
                upper=lowpass
            )
            coordinates[h:h + _h, 0] = s
            coordinates[h:h + _h, 1] = np.arange(_h)
            h += _h
        t += _t
    if verbose:
        stderr('\n')

    model.to('cpu')
    torch.cuda.empty_cache()

    if timecourse_pca_components:
        t = timecourses.shape[-1]
        n_components = min(timecourse_pca_components, t)
        if verbose:
            stderr('%sPCA transforming (n components = %s)' % (' ' * indent, n_components))
        t1 = time.time()
        n_components = min(n_components, t)
        m = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components, svd_solver='auto', whiten=True))
        ])
        timecourses = m.fit_transform(timecourses)
        stderr(' (%0.2fs)\n' % (time.time() - t1))
    if timecourse_ica_components:
        t = timecourses.shape[-1]
        n_components = min(timecourse_ica_components, t)
        n_components = min(n_components, t)
        if verbose:
            stderr('%sICA transforming (n components = %s)' % (' ' * indent, n_components))
        t1 = time.time()
        m = Pipeline([
            ('scaler', StandardScaler()),
            ('ica', FastICA(n_components=n_components, whiten='unit-variance'))
        ])
        timecourses = m.fit_transform(timecourses)
        stderr(' (%0.2fs)\n' % (time.time() - t1))

    return dict(
        timecourses=timecourses,  # <n_neurons, n_tokens/n_components>
        coordinates=coordinates  # <n_neurons>
    )


def get_connectivity(timecourses, n_components=None):
    X = timecourses
    if n_components:
        m = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components))
        ])
        X = m.fit_transform(X)
    R = correlate(X, rowvar=True)

    return R


def sample_parcellations(
        connectivity,
        n_networks=50,
        n_samples=100,
        binarize_connectivity=True,
        connectivity_pca_components=None,
        connectivity_ica_components=None,
        clustering_kwargs=None,
        verbose=True,
        indent=0
):
    if verbose:
        stderr('%sSampling (n_networks=%d)\n' % (' ' * indent, n_networks))
    indent += 2

    if clustering_kwargs is None:
        clustering_kwargs = {}
    X = connectivity
    if binarize_connectivity:
        X = (X > np.quantile(X, 0.9, axis=1)).astype(int)
    if connectivity_pca_components:
        n_components = connectivity_pca_components
        if n_components == 'auto':
            n_components = n_networks - 1
        if verbose:
            stderr('%sPCA transforming (n components = %s)' % (' ' * indent, n_components))
        t1 = time.time()
        n_components = min(n_components, X.shape[-1])
        m = PCA(n_components=n_components, svd_solver='auto', whiten=True)
        X = m.fit_transform(X)
        stderr(' (%0.2fs)\n' % (time.time() - t1))
    if connectivity_ica_components:
        n_components = connectivity_ica_components
        if n_components == 'auto':
            n_components = n_networks - 1
        n_components = min(n_components, X.shape[-1])
        if verbose:
            stderr('%sICA transforming (n components = %s)' % (' ' * indent, n_components))
        t1 = time.time()
        m = FastICA(n_components=n_components, whiten='unit-variance')
        X = m.fit_transform(X)
        stderr(' (%0.2fs)\n' % (time.time() - t1))

    if verbose:
        stderr('%sDrawing samples\n' % (' ' * indent))
    indent += 2
    n_units = X.shape[0]
    samples = np.zeros((n_samples, n_units))
    scores = np.zeros(n_samples)
    for i in range(n_samples):
        if verbose and n_samples > 1:
            stderr('\r%sSample %d/%d' % (' ' * indent, i + 1, n_samples))
        m = MiniBatchKMeans(n_clusters=n_networks, **clustering_kwargs)
        _sample = m.fit_predict(X)
        _score = m.inertia_
        samples[i, :] = _sample
        scores[i] = _score

    if n_samples > 1:
        stderr('\n')

    return dict(
        samples=samples,  # <n_samples, n_units>
        scores=scores  # <n_samples>
    )


def _align_samples(
        samples,
        w=None,
        n_alignments=None,
        shuffle=False,
        greedy=True,
        verbose=True,
        indent=0
):
    if w is None:
        _w = 1
    else:
        _w = w[0]
    n_samples = samples.shape[0]
    n_units = samples.shape[1]
    n_networks = samples.max() + 1
    reference = (samples[0][None, ...] == np.arange(n_networks)[..., None]).astype(float)
    parcellation = None
    C = 0

    # Align subsequent samples
    if shuffle:
        s_ix = np.random.permutation(n_samples)
        samples = samples[s_ix]
    n = n_alignments
    if n is None:
        n = n_samples
    i = 0
    for i_cum in range(n):
        if verbose:
            stderr('\r%sAlignment %d/%d' % (' ' * indent, i_cum + 1, n))

        if w is not None:
            _w = w[i]
        else:
            _w = 1
        if _w == 0:
            continue

        if len(samples.shape) == 2:
            s = (samples[i][None, ...] == np.arange(n_networks)[..., None])
        else:
            s = samples[i].T
        s = s.astype(float)
        _reference = standardize_array(reference)
        _s = standardize_array(s)
        scores = np.dot(
            _reference,
            _s.T,
        ) / n_units

        _, ix_r = optimize.linear_sum_assignment(scores, maximize=True)
        s = s[ix_r]
        if parcellation is None:
            parcellation = s * _w
        else:
            parcellation = parcellation + s * _w
        if greedy:
            reference = parcellation
        C += _w
        i += 1
        if i >= n_samples:
            i = 0
            if shuffle:
                s_ix = np.random.permutation(n_samples)
                samples = samples[s_ix]

    if verbose and n > 0:
        stderr('\n')

    parcellation = parcellation / C

    return parcellation


def align_samples(
        samples,
        scores,
        n_alignments=None,
        weight_samples=False,
        verbose=True,
        indent=0
):
    if verbose:
        stderr('%sAligning samples\n' % (' ' * indent))
    indent += 1

    s_ix = np.argsort(scores)
    samples = samples[s_ix]
    scores = scores[s_ix]
    if weight_samples:
        w = 1 - scores  # Flip to upweight lower inertia
    else:
        w = None

    parcellation = _align_samples(
        samples,
        w=w,
        n_alignments=n_alignments,
        shuffle=False,
        greedy=True,
        verbose=verbose,
        indent=indent + 2
    ).T

    indent -= 1

    return parcellation


def run_connectivity(
        model_name='gpt2',
        output_dir=OUTPUT_DIR,
        n_samples=N_SAMPLES,
        domains=('wikitext', 'bookcorpus', 'agnews', 'tldr17', 'codeparrot', 'random', 'whitespace'),
        seq_len=1024,
        n_tokens=None,
        split='train',
        take=100000,
        wrap=True,
        shuffle=True,
        batch_size=8,
        highpass=None,
        lowpass=None,
        step=0.2,
        timecourse_pca_components=None,
        timecourse_ica_components=None,
        eps=1e-3,
        data_kwargs=None,
        model_kwargs=None,
        knockout_filepath=None,
        knockout_thresh=0.5,
        overwrite=False,
        verbose=True,
        indent=0
):
    if data_kwargs is None:
        data_kwargs = {}
    if model_kwargs is None:
        model_kwargs = {}
    if n_tokens is None:
        n_tokens = (N_TOKENS // (seq_len * batch_size)) * seq_len * batch_size

    connectivity_dir = os.path.join(output_dir, CONNECTIVITY_NAME)

    knockout_probs = knockout_coordinates = None
    if knockout_filepath is not None:
        data = load_h5_data(knockout_filepath, verbose=verbose, indent=indent)
        assert 'parcellation' in data, 'If provided, knockout_filepath must contain the field "parcellation"'
        knockout_probs = data['parcellation']
        knockout_coordinates = data['coordinates']

    model, tokenizer = get_model_and_tokenizer(
        model_name,
        knockout_probs=knockout_probs,
        coordinates=knockout_coordinates,
        knockout_thresh=knockout_thresh
    )

    if isinstance(domains, str):
        domains = (domains,)

    for domain in domains:
        if verbose:
            stderr('%sRunning connectivity for %s\n' % (' ' * indent, domain))
        indent += 2
        _data_kwargs = copy.deepcopy(data_kwargs)
        if domain == 'wikitext':
            _data_kwargs.update(dict(
                dataset='Salesforce/wikitext',
                tokenizer= tokenizer,
                name='wikitext-103-raw-v1',
            ))
        elif domain == 'bookcorpus':
            _data_kwargs.update(dict(
                dataset='Yuti/bookcorpus'
            ))
        elif domain == 'agnews':
            _data_kwargs.update(dict(
                dataset='fancyzhx/ag_news'
            ))
        elif domain == 'codeparrot':
            _data_kwargs.update(dict(
                dataset='codeparrot/codeparrot-clean'
            ))
        elif domain == 'tldr17':
            _data_kwargs.update(dict(
                dataset='webis/tldr-17'
            ))
        elif domain == 'random':
            _data_kwargs.update(dict(
                dataset='random'
            ))
        elif domain == 'whitespace':
            _data_kwargs.update(dict(
                dataset='whitespace'
            ))
        else:
            raise ValueError('Unrecognized input data name: %s' % domain)

        _data_kwargs['tokenizer'] = tokenizer

        input_ids, attention_mask = get_dataset(
            n_tokens=n_tokens * n_samples,
            split=split,
            take=take,
            seq_len=seq_len,
            wrap=wrap,
            shuffle=shuffle,
            verbose=verbose,
            indent=indent,
            **_data_kwargs
        )

        if not os.path.exists(connectivity_dir):
            os.makedirs(connectivity_dir)

        if verbose:
            stderr('%sQuerying model\n' % (' ' * indent))
        n = int(np.ceil(len(input_ids) / n_samples))
        connectivity = []
        coordinates = None
        indent += 2
        new = False
        for i in range(0, len(input_ids), n):
            t0 = time.time()
            filepath = os.path.join(
                connectivity_dir,
                '%s_%s_%s%d%s' % (
                    CONNECTIVITY_NAME,
                    domain,
                    SAMPLE_NAME,
                    i // n + 1,
                    EXTENSION
                )
            )
            if verbose:
                stderr('%sSample %d/%d\n' % (' ' * indent, i // n + 1, n_samples))
            if os.path.exists(filepath) and not overwrite:
                out = load_h5_data(filepath, verbose=False)
            else:
                out = {}
            indent += 2
            if 'connectivity' not in out or 'coordinates' not in out:
                _input_ids = input_ids[i:i+n]
                _attention_mask = attention_mask[i:i+n]
                out = get_timecourses(
                    model,
                    _input_ids,
                    _attention_mask,
                    batch_size=batch_size,
                    highpass=highpass,
                    lowpass=lowpass,
                    step=step,
                    timecourse_pca_components=timecourse_pca_components,
                    timecourse_ica_components=timecourse_ica_components,
                    verbose=verbose,
                    indent=indent,
                    **model_kwargs
                )
                timecourses = out['timecourses']
                coordinates = out['coordinates']
                _connectivity = get_connectivity(timecourses)
                save = True
                new = True
            else:
                _connectivity = out['connectivity']
                coordinates = out['coordinates']
                save = False
            connectivity.append(_connectivity)
            if n_samples > 1 and save:
                save_h5_data(
                    dict(
                        connectivity=_connectivity,
                        coordinates=coordinates
                    ),
                    filepath,
                    verbose=verbose,
                    indent=indent
                )
            if verbose:
                stderr('%sElapsed time: %.2f s\n' % (' ' * indent, time.time() - t0))
            indent -= 2
        indent -= 2
        if n_samples > 1:
            connectivity = fisher_average(*connectivity, eps=eps)
        else:
            connectivity = connectivity[0]
        filepath = os.path.join(
            connectivity_dir,
            '%s_%s_avg%s' % (
                CONNECTIVITY_NAME,
                domain,
                EXTENSION
            ),
        )
        save = True
        if os.path.exists(filepath) and not overwrite:
            out = load_h5_data(filepath, verbose=False)
            if 'connectivity' in out and 'coordinates' in out and not new:
                save = False
        if save:
            save_h5_data(
                dict(
                    connectivity=connectivity,
                    coordinates=coordinates
                ),
                filepath,
                verbose=verbose,
                indent=indent
            )
        indent -= 2


def run_parcellation(
        output_dir=OUTPUT_DIR,
        n_networks=50,
        n_samples=100,
        binarize_connectivity=True,
        connectivity_pca_components=200,
        connectivity_ica_components=None,
        clustering_kwargs=None,
        n_alignments=None,
        weight_samples=False,
        overwrite=False,
        verbose=True,
        indent=0
):
    connectivity_dir = os.path.join(output_dir, CONNECTIVITY_NAME)

    for path in os.listdir(connectivity_dir):
        t0 = time.time()
        match = INPUT_NAME_RE.match(path)
        if not match:
            continue
        inpath = os.path.join(connectivity_dir, path)
        data = load_h5_data(inpath, verbose=verbose, indent=indent)

        if overwrite or not 'parcellation' in data:
            R = np.nan_to_num(data['connectivity'])
            R = np.abs(R)

            sample = sample_parcellations(
                R,
                n_networks=n_networks,
                n_samples=n_samples,
                binarize_connectivity=binarize_connectivity,
                connectivity_pca_components=connectivity_pca_components,
                connectivity_ica_components=connectivity_ica_components,
                clustering_kwargs=clustering_kwargs,
                verbose=verbose,
                indent=indent + 2
            )
            parcellation = align_samples(
                sample['samples'],
                sample['scores'],
                n_alignments=n_alignments,
                weight_samples=weight_samples,
                verbose=verbose,
                indent=indent + 2
            )
            data['parcellation'] = parcellation

            save_h5_data(
                data,
                inpath,
                verbose=verbose,
                indent=indent + 2
            )

            if verbose:
                stderr('%sElapsed time: %.2f s\n' % (' ' * (indent + 2), time.time() - t0))


def run_subnetwork_extraction(
        output_dir=OUTPUT_DIR,
        verbose=True,
        indent=0
):
    connectivity_dir = os.path.join(output_dir, CONNECTIVITY_NAME)
    subnetwork_dir = os.path.join(output_dir, SUBNETWORK_NAME)

    if verbose:
        stderr('Extracting subnetworks\n')
    indent += 2

    parcellations = {}
    coordinates = None
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
        if 'parcellation' not in data:
            continue
        if coordinates is None:
            coordinates = data['coordinates']
        parcellations[domain] = data['parcellation']

    shared_subnetworks = {}
    domains = sorted(list(parcellations.keys()))
    n_domains = len(domains)
    for d1 in range(len(domains)):
        domain1 = domains[d1]
        for d2 in range(d1 + 1, len(domains)):
            domain2 = domains[d2]
            parcellation1 = parcellations[domain1].T  # <n_networks, n_units>
            parcellation2 = parcellations[domain2].T  # <n_networks, n_units>
            n_networks = parcellation1.shape[0]
            n_units = parcellation1.shape[1]

            _parcellation1 = standardize_array(parcellation1)
            _parcellation2 = standardize_array(parcellation2)
            scores = np.dot(
                _parcellation1,
                _parcellation2.T,
            ) / n_units
            alignment1 = np.argmax(scores, axis=1)
            alignment2 = np.argmax(scores, axis=0)
            matches = np.arange(n_networks) == alignment2[alignment1]
            ix1 = np.arange(n_networks)[matches]
            ix2 = alignment1[matches]
            if domain1 not in shared_subnetworks:
                shared_subnetworks[domain1] = {}
            if domain2 not in shared_subnetworks:
                shared_subnetworks[domain2] = {}
            shared_subnetworks[domain1][domain2] = {int(x):int(y) for x, y in zip(ix1, ix2)}
            shared_subnetworks[domain2][domain1] = {int(y):int(x) for x, y in zip(ix1, ix2)}

    networks = []
    for start in shared_subnetworks[domains[0]][domains[1]]:
        d_ix = 0
        n_ix = start
        network = []
        while d_ix < len(domains):
            domain = domains[d_ix]
            network.append(parcellations[domain][..., n_ix])
            if d_ix < n_domains - 1 and n_ix in shared_subnetworks[domain][domains[d_ix + 1]]:
                n_ix = shared_subnetworks[domain][domains[d_ix + 1]][n_ix]
                d_ix += 1
            else:
                break

        if len(network) == len(domains):
            network = np.stack(network, axis=0).mean(axis=0)
            networks.append(network)

    networks = np.stack(networks, axis=1)

    if not os.path.exists(subnetwork_dir):
        os.makedirs(subnetwork_dir)

    save_h5_data(
        dict(
            parcellation=networks,
            coordinates=coordinates
        ),
        os.path.join(
            subnetwork_dir,
            '%s_%s_%s%s' % (
                PARCELLATION_NAME,
                'shared',
                'avg',
                EXTENSION
            )
        ),
        verbose=verbose,
        indent=indent
    )


def run_knockout(
        output_dir=os.path.join(OUTPUT_DIR, KNOCKOUT_NAME),
        model_name='gpt2',
        connectivity_kwargs=None,
        steps=('plot_stability',),
        verbose=True,
        indent=0
):
    if connectivity_kwargs is None:
        connectivity_kwargs = {}
    subnetwork_dir = os.path.join(output_dir, SUBNETWORK_NAME)
    knockout_dir = os.path.join(output_dir, 'knockout')

    if verbose:
        stderr('Running knockout\n')
    indent += 2

    if not os.path.exists(knockout_dir):
        os.makedirs(knockout_dir)

    for path in os.listdir(subnetwork_dir):
        match = INPUT_NAME_RE.match(path)
        if not match:
            continue
        knockout_filepath = os.path.join(subnetwork_dir, path)
        data = load_h5_data(knockout_filepath, verbose=False)
        if 'parcellation' not in data:
            continue

        run_connectivity(
            model_name=model_name,
            output_dir=knockout_dir,
            knockout_filepath=knockout_filepath,
            knockout_thresh=0.5,
            verbose=verbose,
            indent=indent,
            **connectivity_kwargs
        )

        for step in steps:
            if step == 'plot_stability':
                plot_stability(
                    output_dir=knockout_dir,
                    verbose=verbose,
                    indent=indent
                )
            else:
                raise ValueError('Unrecognized step: %s' % step)
