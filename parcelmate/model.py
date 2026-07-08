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
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from parcelmate.constants import *
from parcelmate.data import *
from parcelmate.util import *
from parcelmate.plot import *


def get_transformer_body(model):
    """Return the transformer stack that exposes ``.h`` (blocks) and ``.drop``
    (embedding dropout). ``AutoModel`` (e.g. GPT2Model) is itself the body,
    while ``AutoModelForCausalLM`` (e.g. GPT2LMHeadModel) nests it under
    ``.transformer``."""
    if hasattr(model, 'transformer'):
        return model.transformer
    return model


class PerturbedModel(torch.nn.Module):
    def __init__(self, model, perturbation_coordinates, perturbation_values=None, *args, **kwargs):
        super(PerturbedModel, self).__init__(*args, **kwargs)
        self.model = model
        body = get_transformer_body(self.model)
        self.perturbation_coordinates = perturbation_coordinates
        if perturbation_values is None:  # Default to zero (knockout)
            perturbation_values = np.zeros(len(self.perturbation_coordinates))
        assert len(perturbation_values) == len(perturbation_coordinates), \
            'perturbation_values must match perturbation_coordinates'
        self.perturbation_values = perturbation_values

        layers_attr = 'h'
        layer_indices = np.unique(perturbation_coordinates[:, 0])  # 0th dimension is layer
        layers = getattr(body, layers_attr)
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
                source_layer = body.drop
            else:
                _l_ix = l_ix - 1  # Shifted down bc of embedding layer
                source_layer = layers[_l_ix]
            layer = PerturbedLayer(
                source_layer,
                perturbation_coordinates=self.perturbation_coordinate_tensors[_l_ix],
                perturbation_values=self.perturbation_value_tensors[_l_ix]
            )
            if _l_ix == 'embedding':
                body.drop = layer
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


def select_knockout(coordinates, knockout_probs, knockout_thresh=0.5, network_ix=None):
    """Build a boolean selection mask (and the matching coordinates) for a
    knockout from soft network-membership probabilities.

    ``knockout_probs`` is <n_units> (single network) or <n_units, n_networks>.
    When ``network_ix`` is None and multiple networks are present, the selection
    is the union of every network at/above ``knockout_thresh`` (the original
    behavior); otherwise a single network column is used.
    """
    probs = np.asarray(knockout_probs)
    if probs.ndim == 1:
        sel = probs >= knockout_thresh
    elif network_ix is None:
        sel = (probs >= knockout_thresh).any(axis=1)
    else:
        sel = probs[:, network_ix] >= knockout_thresh
    return coordinates[sel], sel


def sample_baseline_selection(sel, seed=0):
    """Given a knockout selection mask ``sel`` over all units, draw an equally
    sized set of units uniformly at random from the *complement* (units that
    were NOT knocked out), globally across all layers."""
    sel = np.asarray(sel).astype(bool)
    n = int(sel.sum())
    complement_ix = np.where(~sel)[0]
    assert n <= len(complement_ix), \
        'Cannot draw %d baseline units from a complement of size %d' % (n, len(complement_ix))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(complement_ix, size=n, replace=False)
    baseline_sel = np.zeros_like(sel)
    baseline_sel[chosen] = True
    return baseline_sel


def get_model_and_tokenizer(
        model_name,
        for_causal_lm=False,
        perturbation_coordinates=None,
        perturbation_values=None,
):
    if for_causal_lm:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
    if perturbation_coordinates is not None and len(perturbation_coordinates) > 0:
        model = PerturbedModel(
            model,
            perturbation_coordinates=perturbation_coordinates,
            perturbation_values=perturbation_values,
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


def get_lm_loss(
        model,
        input_ids,
        attention_mask,
        batch_size=8,
        verbose=True,
        indent=0,
        **kwargs
):
    """Compute mean next-token cross-entropy (and perplexity) over the given
    inputs. Padding positions are excluded via the attention mask, and the loss
    is token-weighted (not batch-averaged) so it is comparable across draws."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    if verbose:
        stderr('%sComputing LM loss\n' % (' ' * indent))
    total_loss = 0.0
    total_tokens = 0
    B = int(math.ceil(input_ids.size(0) / batch_size))
    indent += 2
    with torch.no_grad():
        for i in range(0, input_ids.size(0), batch_size):
            if verbose:
                stderr('\r%sBatch %d/%d' % (' ' * indent, i // batch_size + 1, B))
            _input_ids = input_ids[i:i + batch_size].to(device)
            _attention_mask = attention_mask[i:i + batch_size].to(device)
            logits = model(
                input_ids=_input_ids,
                attention_mask=_attention_mask,
                **kwargs
            ).logits
            # Shift so token t predicts token t+1.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = _input_ids[..., 1:].contiguous()
            shift_mask = _attention_mask[..., 1:].contiguous().reshape(-1)
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction='none'
            )
            loss = loss * shift_mask
            total_loss += float(loss.sum().detach().cpu())
            total_tokens += int(shift_mask.sum().detach().cpu())
    if verbose:
        stderr('\n')

    model.to('cpu')
    torch.cuda.empty_cache()

    mean_loss = total_loss / max(total_tokens, 1)
    perplexity = float(np.exp(mean_loss))

    return dict(
        loss=mean_loss,
        perplexity=perplexity,
        n_tokens=total_tokens
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


def resolve_domain_data_kwargs(domain, tokenizer, data_kwargs=None):
    """Map a domain name to the ``get_dataset`` kwargs that load it."""
    _data_kwargs = copy.deepcopy(data_kwargs) if data_kwargs else {}
    if domain == 'wikitext':
        _data_kwargs.update(dict(
            dataset='Salesforce/wikitext',
            name='wikitext-103-raw-v1',
        ))
    elif domain == 'bookcorpus':
        # Load the parquet files directly via the packaged 'parquet' builder so
        # datasets never attempts dataset-script resolution (the repo's legacy
        # bookcorpus.py script is rejected by datasets >= 3.x).
        _data_kwargs.update(dict(
            dataset='parquet',
            data_files='hf://datasets/Yuti/bookcorpus/data/train-*.parquet',
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
        # webis/tldr-17 is script-only; load HF's auto-converted parquet branch
        # directly via the packaged 'parquet' builder to avoid script resolution.
        _data_kwargs.update(dict(
            dataset='parquet',
            data_files='hf://datasets/webis/tldr-17@refs%2Fconvert%2Fparquet/default/partial-train/*.parquet'
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

    return _data_kwargs


def compute_mean_activations(
        model_name='gpt2',
        output_dir=OUTPUT_DIR,
        domains=('wikitext', 'bookcorpus', 'agnews', 'tldr17', 'codeparrot', 'random', 'whitespace'),
        seq_len=1024,
        n_tokens=None,
        split='train',
        take=100000,
        wrap=True,
        shuffle=True,
        batch_size=8,
        data_kwargs=None,
        model_kwargs=None,
        overwrite=False,
        verbose=True,
        indent=0
):
    """Compute each neuron's mean activation over a text sample spanning ALL
    domains (a single cross-domain scalar per unit). Used as the "mean-out"
    clamp value: knocking a neuron out sets it to this global mean rather than
    to zero. Cached to ``mean_activations.h5`` under ``output_dir``.

    Returns ``(mean_activations, coordinates)`` aligned row-for-row with the
    coordinates produced by ``get_timecourses``.
    """
    if model_kwargs is None:
        model_kwargs = {}
    if n_tokens is None:
        n_tokens = (N_TOKENS // (seq_len * batch_size)) * seq_len * batch_size

    filepath = os.path.join(output_dir, '%s%s' % (MEAN_ACTIVATION_NAME, EXTENSION))
    if os.path.exists(filepath) and not overwrite:
        data = load_h5_data(filepath, verbose=verbose, indent=indent)
        if 'mean_activations' in data and 'coordinates' in data:
            return data['mean_activations'], data['coordinates']

    if isinstance(domains, str):
        domains = (domains,)

    if verbose:
        stderr('%sComputing cross-domain mean activations\n' % (' ' * indent))
    indent += 2

    model, tokenizer = get_model_and_tokenizer(model_name)

    running_sum = None
    running_count = 0
    coordinates = None
    for domain in domains:
        if verbose:
            stderr('%sDomain %s\n' % (' ' * indent, domain))
        _data_kwargs = resolve_domain_data_kwargs(domain, tokenizer, data_kwargs)
        input_ids, attention_mask = get_dataset(
            n_tokens=n_tokens,
            split=split,
            take=take,
            seq_len=seq_len,
            wrap=wrap,
            shuffle=shuffle,
            verbose=verbose,
            indent=indent + 2,
            **_data_kwargs
        )
        # Raw activations only: no bandpass / PCA / ICA transforms.
        out = get_timecourses(
            model,
            input_ids,
            attention_mask,
            batch_size=batch_size,
            highpass=None,
            lowpass=None,
            verbose=verbose,
            indent=indent + 2,
            **model_kwargs
        )
        timecourses = out['timecourses']  # <n_units, n_tokens>
        coordinates = out['coordinates']
        # Weight by token count so domains contribute proportionally.
        if running_sum is None:
            running_sum = timecourses.sum(axis=1).astype(np.float64)
        else:
            running_sum += timecourses.sum(axis=1)
        running_count += timecourses.shape[1]

    mean_activations = (running_sum / max(running_count, 1)).astype(np.float32)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_h5_data(
        dict(
            mean_activations=mean_activations,
            coordinates=coordinates
        ),
        filepath,
        verbose=verbose,
        indent=indent
    )

    return mean_activations, coordinates


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
        perturbation_coordinates=None,
        perturbation_values=None,
        eval_loss=False,
        loss_n_tokens=None,
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

    # A single model (loaded with an LM head when loss is requested) is reused
    # for both the connectivity and loss passes so the perturbation is identical.
    model, tokenizer = get_model_and_tokenizer(
        model_name,
        for_causal_lm=eval_loss,
        perturbation_coordinates=perturbation_coordinates,
        perturbation_values=perturbation_values,
    )

    if isinstance(domains, str):
        domains = (domains,)

    losses = {}
    for domain in domains:
        if verbose:
            stderr('%sRunning connectivity for %s\n' % (' ' * indent, domain))
        indent += 2
        _data_kwargs = resolve_domain_data_kwargs(domain, tokenizer, data_kwargs)

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

        if eval_loss:
            _input_ids = input_ids
            _attention_mask = attention_mask
            if loss_n_tokens is not None:
                n_seq = int(np.ceil(loss_n_tokens / seq_len))
                _input_ids = _input_ids[:n_seq]
                _attention_mask = _attention_mask[:n_seq]
            loss_out = get_lm_loss(
                model,
                _input_ids,
                _attention_mask,
                batch_size=batch_size,
                verbose=verbose,
                indent=indent,
            )
            losses[domain] = loss_out
            if verbose:
                stderr('%sLoss=%.4f  Perplexity=%.2f  (%s)\n' % (
                    ' ' * indent, loss_out['loss'], loss_out['perplexity'], domain))
        indent -= 2

    if eval_loss and losses:
        loss_data = {}
        for domain, loss_out in losses.items():
            loss_data['%s_loss' % domain] = np.float32(loss_out['loss'])
            loss_data['%s_perplexity' % domain] = np.float32(loss_out['perplexity'])
            loss_data['%s_n_tokens' % domain] = np.int64(loss_out['n_tokens'])
        save_h5_data(
            loss_data,
            os.path.join(output_dir, '%s%s' % (LOSS_NAME, EXTENSION)),
            verbose=verbose,
            indent=indent
        )


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


def _run_knockout_condition(
        name,
        knockout_root,
        model_name,
        coordinates,
        sel,
        mean_activations=None,
        connectivity_kwargs=None,
        eval_loss=True,
        loss_n_tokens=None,
        steps=(),
        overwrite=False,
        verbose=True,
        indent=0
):
    """Run one knockout condition (a single selection ``sel`` over all units)
    into its own subdirectory ``knockout_root/name``. When ``mean_activations``
    is provided the knocked-out units are clamped to their cross-domain mean
    ("mean-out"); otherwise they are zeroed."""
    condition_dir = os.path.join(knockout_root, name)
    if sel is None:
        perturbation_coordinates = None
        perturbation_values = None
        n_ko = 0
    else:
        sel = np.asarray(sel).astype(bool)
        perturbation_coordinates = coordinates[sel]
        n_ko = int(sel.sum())
        if mean_activations is not None:
            perturbation_values = np.asarray(mean_activations)[sel]
        else:
            perturbation_values = None
    if verbose:
        stderr('%sCondition %s (%d units knocked out)\n' % (' ' * indent, name, n_ko))

    run_connectivity(
        model_name=model_name,
        output_dir=condition_dir,
        perturbation_coordinates=perturbation_coordinates,
        perturbation_values=perturbation_values,
        eval_loss=eval_loss,
        loss_n_tokens=loss_n_tokens,
        overwrite=overwrite,
        verbose=verbose,
        indent=indent + 2,
        **(connectivity_kwargs or {})
    )

    for step in steps:
        if step == 'plot_stability':
            plot_stability(
                output_dir=condition_dir,
                verbose=verbose,
                indent=indent + 2
            )
        else:
            raise ValueError('Unrecognized step: %s' % step)


def _run_baselines(
        prefix,
        knockout_root,
        model_name,
        coordinates,
        sel,
        mean_activations=None,
        connectivity_kwargs=None,
        n_baseline=3,
        baseline_seed=0,
        eval_loss=True,
        loss_n_tokens=None,
        steps=(),
        overwrite=False,
        verbose=True,
        indent=0
):
    """Run ``n_baseline`` size-matched random controls for the knockout ``sel``.
    Skips (with a warning) when the knockout is larger than its complement, in
    which case a disjoint equal-sized random draw cannot exist."""
    sel = np.asarray(sel).astype(bool)
    n_ko = int(sel.sum())
    n_complement = int((~sel).sum())
    if n_ko > n_complement:
        if verbose:
            stderr('%sSkipping baselines for %s: %d units knocked out exceeds '
                   'complement of %d\n' % (' ' * indent, prefix, n_ko, n_complement))
        return
    for s in range(n_baseline):
        baseline_sel = sample_baseline_selection(sel, seed=baseline_seed + s)
        _run_knockout_condition(
            '%s_%s%d' % (prefix, BASELINE_NAME, s),
            knockout_root, model_name, coordinates, sel=baseline_sel,
            mean_activations=mean_activations, connectivity_kwargs=connectivity_kwargs,
            eval_loss=eval_loss, loss_n_tokens=loss_n_tokens, steps=steps,
            overwrite=overwrite, verbose=verbose, indent=indent
        )


def run_knockout(
        output_dir=OUTPUT_DIR,
        model_name='gpt2',
        connectivity_kwargs=None,
        knockout_thresh=0.5,
        knockout_mode='mean',
        n_baseline=3,
        baseline_seed=0,
        networks=None,
        include_union=True,
        include_healthy=True,
        eval_loss=True,
        loss_n_tokens=None,
        steps=('plot_stability',),
        overwrite=False,
        verbose=True,
        indent=0
):
    """Knock out each extracted subnetwork individually (plus, optionally, the
    union of all of them), each against ``n_baseline`` size-matched controls
    drawn uniformly at random from the un-knocked-out complement. Every
    condition is evaluated with both connectivity and LM loss so selectivity can
    be tested: a subnetwork is "special" only if knocking it out hurts more than
    knocking out the same number of random neurons.
    """
    connectivity_kwargs = dict(connectivity_kwargs or {})
    # These are supplied explicitly below; drop from the passthrough to avoid
    # duplicate-keyword errors.
    if 'model_name' in connectivity_kwargs:
        model_name = connectivity_kwargs.pop('model_name')
    connectivity_kwargs.pop('overwrite', None)

    assert knockout_mode in ('mean', 'zero'), 'knockout_mode must be "mean" or "zero"'

    subnetwork_dir = os.path.join(output_dir, SUBNETWORK_NAME)
    knockout_root = os.path.join(output_dir, KNOCKOUT_NAME)

    if verbose:
        stderr('Running knockout (mode=%s)\n' % knockout_mode)
    indent += 2

    if not os.path.exists(knockout_root):
        os.makedirs(knockout_root)

    # Cross-domain mean activations (computed once, cached) for "mean-out".
    mean_activations = None
    if knockout_mode == 'mean':
        mean_allowed = {'domains', 'seq_len', 'n_tokens', 'split', 'take',
                        'wrap', 'shuffle', 'batch_size', 'data_kwargs', 'model_kwargs'}
        mean_kwargs = {k: v for k, v in connectivity_kwargs.items() if k in mean_allowed}
        mean_activations, mean_coordinates = compute_mean_activations(
            model_name=model_name,
            output_dir=knockout_root,
            overwrite=overwrite,
            verbose=verbose,
            indent=indent,
            **mean_kwargs
        )

    for path in os.listdir(subnetwork_dir):
        match = INPUT_NAME_RE.match(path)
        if not match:
            continue
        subnetwork_filepath = os.path.join(subnetwork_dir, path)
        data = load_h5_data(subnetwork_filepath, verbose=False)
        if 'parcellation' not in data:
            continue
        probs = np.asarray(data['parcellation'])  # <n_units, n_networks>
        coordinates = np.asarray(data['coordinates'])
        if probs.ndim == 1:
            probs = probs[:, None]
        n_networks = probs.shape[1]

        if mean_activations is not None:
            assert len(mean_activations) == coordinates.shape[0], \
                'mean_activations (%d) and subnetwork coordinates (%d) are misaligned' % (
                    len(mean_activations), coordinates.shape[0])

        # Healthy reference (no perturbation) for matched loss/connectivity.
        if include_healthy:
            _run_knockout_condition(
                HEALTHY_NAME, knockout_root, model_name, coordinates, sel=None,
                mean_activations=None, connectivity_kwargs=connectivity_kwargs,
                eval_loss=eval_loss, loss_n_tokens=loss_n_tokens, steps=steps,
                overwrite=overwrite, verbose=verbose, indent=indent
            )

        net_indices = list(range(n_networks)) if networks is None else list(networks)
        for network_ix in net_indices:
            _, sel = select_knockout(coordinates, probs, knockout_thresh, network_ix=network_ix)
            if int(sel.sum()) == 0:
                if verbose:
                    stderr('%sNetwork %d has no units above threshold; skipping\n' % (
                        ' ' * indent, network_ix))
                continue
            # Real subnetwork knockout.
            _run_knockout_condition(
                'network%d' % network_ix, knockout_root, model_name, coordinates, sel=sel,
                mean_activations=mean_activations, connectivity_kwargs=connectivity_kwargs,
                eval_loss=eval_loss, loss_n_tokens=loss_n_tokens, steps=steps,
                overwrite=overwrite, verbose=verbose, indent=indent
            )
            # Size-matched random baselines from the complement.
            _run_baselines(
                'network%d' % network_ix, knockout_root, model_name, coordinates, sel,
                mean_activations=mean_activations, connectivity_kwargs=connectivity_kwargs,
                n_baseline=n_baseline, baseline_seed=baseline_seed,
                eval_loss=eval_loss, loss_n_tokens=loss_n_tokens, steps=steps,
                overwrite=overwrite, verbose=verbose, indent=indent
            )

        # Union of all subnetworks (original behavior) + its baselines.
        if include_union and n_networks > 1:
            _, sel = select_knockout(coordinates, probs, knockout_thresh, network_ix=None)
            if int(sel.sum()) > 0:
                _run_knockout_condition(
                    'union', knockout_root, model_name, coordinates, sel=sel,
                    mean_activations=mean_activations, connectivity_kwargs=connectivity_kwargs,
                    eval_loss=eval_loss, loss_n_tokens=loss_n_tokens, steps=steps,
                    overwrite=overwrite, verbose=verbose, indent=indent
                )
                _run_baselines(
                    'union', knockout_root, model_name, coordinates, sel,
                    mean_activations=mean_activations, connectivity_kwargs=connectivity_kwargs,
                    n_baseline=n_baseline, baseline_seed=baseline_seed,
                    eval_loss=eval_loss, loss_n_tokens=loss_n_tokens, steps=steps,
                    overwrite=overwrite, verbose=verbose, indent=indent
                )

    if eval_loss:
        summarize_knockout_loss(knockout_root, verbose=verbose, indent=indent)
        plot_knockout_loss(output_dir=output_dir, verbose=verbose, indent=indent)


def summarize_knockout_loss(knockout_root, verbose=True, indent=0):
    """Collect per-condition ``loss.h5`` files under ``knockout_root`` into a
    tidy CSV comparing healthy vs subnetwork-knockout vs baseline-knockout."""
    import pandas as pd

    rows = []
    for name in sorted(os.listdir(knockout_root)):
        condition_dir = os.path.join(knockout_root, name)
        loss_path = os.path.join(condition_dir, '%s%s' % (LOSS_NAME, EXTENSION))
        if not os.path.isdir(condition_dir) or not os.path.exists(loss_path):
            continue
        data = load_h5_data(loss_path, verbose=False)
        domains = sorted({k.rsplit('_', 1)[0] for k in data
                          if k.endswith('_loss')})
        if name == HEALTHY_NAME:
            kind = 'healthy'
        elif BASELINE_NAME in name:
            kind = 'baseline'
        else:
            kind = 'knockout'
        for domain in domains:
            rows.append(dict(
                condition=name,
                kind=kind,
                domain=domain,
                loss=float(data['%s_loss' % domain]),
                perplexity=float(data['%s_perplexity' % domain]),
                n_tokens=int(data['%s_n_tokens' % domain]),
            ))

    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values(['domain', 'condition']).reset_index(drop=True)
    out_path = os.path.join(knockout_root, '%s_summary.csv' % LOSS_NAME)
    df.to_csv(out_path, index=False)
    if verbose:
        stderr('%sWrote loss summary to %s\n' % (' ' * indent, out_path))

    return df
