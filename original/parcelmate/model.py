import math
import os
import copy
import numpy as np
import pandas as pd  # ADDED (6): loss summary
from scipy import optimize
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import MiniBatchKMeans
import torch
import torch.nn.functional as F  # ADDED (6): LM loss
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from parcelmate.constants import *
from parcelmate.data import *
from parcelmate.util import *
from parcelmate.plot import *


def get_transformer_body(model):
    """ADDED (6): return the transformer stack exposing ``.h`` (blocks) and
    ``.drop`` (embedding dropout). ``AutoModel`` (GPT2Model) is itself the body,
    while ``AutoModelForCausalLM`` (GPT2LMHeadModel) nests it under
    ``.transformer``. Without this, lesioning a causal-LM model raises
    AttributeError, so LM loss could not be measured under a knockout at all."""
    if hasattr(model, 'transformer'):
        return model.transformer
    return model


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

        # ADDED (6): was `self.model`; the body indirection lets the same lesion
        # apply to AutoModel and AutoModelForCausalLM alike.
        body = get_transformer_body(self.model)
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
        # BASELINE PATCH 3: was `self.layer.forward(*args, **kwargs)`. Calling
        # .forward() directly bypasses PyTorch's forward hooks, and transformers
        # v5 collects `output_hidden_states` VIA those hooks. So every wrapped
        # block silently dropped its hidden state: lesioning all 12 GPT-2 blocks
        # returned 1 hidden state instead of 13, and connectivity was computed
        # over 768 units instead of 9984. The lesion itself always applied
        # correctly -- only the recording of activations was lost.
        out = self.layer(*args, **kwargs)
        if self.perturbation_coordinates is not None:
            if isinstance(out, torch.Tensor):
                out[..., self.perturbation_coordinates] = self.perturbation_values
            else:
                out0 = out[0]
                out0[..., self.perturbation_coordinates] = self.perturbation_values
                out = (out0,) + out[1:]

        return out


def get_model_and_tokenizer(
        model_name,
        knockout_probs=None,
        knockout_thresh=0.5,
        coordinates=None,
        network_ix=None,      # ADDED (2): which single network to lesion
        knockout_values=None, # ADDED (1): mean-out clamp values, <n_units>
        for_causal_lm=False,  # ADDED (6): load the LM head, for loss
        knockout_selection=None  # ADDED (7): explicit mask, overrides network_ix
):
    if for_causal_lm:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
    if knockout_probs is not None:
        assert coordinates is not None, 'coordinates must be provided if knockout_probs is not None'
        if knockout_selection is not None:
            # ADDED (7): a random size-matched baseline is an arbitrary set of
            # units, not one the parcellation can name, so it comes in as an
            # explicit mask rather than a network index.
            sel = np.asarray(knockout_selection).astype(bool)
            perturbation_coordinates = coordinates[sel]
        else:
            perturbation_coordinates, sel = select_knockout(
                coordinates, knockout_probs, knockout_thresh=knockout_thresh, network_ix=network_ix
            )
        # None -> PerturbedModel defaults to zeros, i.e. zero-out. Otherwise take
        # the clamp value of each selected unit from the full-length mean vector,
        # which is aligned row-for-row with `coordinates`.
        perturbation_values = None
        if knockout_values is not None:
            knockout_values = np.asarray(knockout_values)
            assert knockout_values.shape[0] == sel.shape[0], \
                'knockout_values must be aligned with coordinates (%d vs %d)' % (
                    knockout_values.shape[0], sel.shape[0])
            perturbation_values = knockout_values[sel]
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


def get_lm_loss(
        model,
        input_ids,
        attention_mask,
        batch_size=8,
        verbose=True,
        indent=0,
        **kwargs
):
    """ADDED (6): mean next-token cross-entropy (and perplexity) over the given
    inputs.

    Stability measures whether the model is self-consistent; it cannot tell a
    healthy model from one that is consistently broken. Loss measures whether
    the model got *worse*, which is the question a knockout is actually asking.

    Padding is excluded via the attention mask, and the loss is token-weighted
    rather than batch-averaged so it is comparable across draws of different
    length."""
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

    return dict(
        loss=mean_loss,
        perplexity=float(np.exp(mean_loss)),
        n_tokens=total_tokens
    )


def run_condition_loss(
        model_name='gpt2',
        domains=(),
        knockout_filepath=None,
        knockout_thresh=0.5,
        knockout_network_ix=None,
        knockout_values=None,
        knockout_selection=None,  # ADDED (7)
        seq_len=1024,
        n_tokens=None,
        split='train',
        take=100000,
        wrap=True,
        shuffle=True,
        batch_size=8,
        data_kwargs=None,
        model_kwargs=None,
        verbose=True,
        indent=0,
        **_ignored
):
    """ADDED (6): next-token loss for ONE knockout condition, on each domain it
    is evaluated over. Returns ``{domain: {loss, perplexity, n_tokens}}``.

    The lesion is applied exactly as in `run_connectivity` -- same selection,
    same clamp values -- so the loss and the connectivity describe the same
    perturbed model.
    """
    if model_kwargs is None:
        model_kwargs = {}
    if n_tokens is None:
        n_tokens = (N_TOKENS // (seq_len * batch_size)) * seq_len * batch_size
    if isinstance(domains, str):
        domains = (domains,)

    knockout_probs = knockout_coordinates = None
    if knockout_filepath is not None:
        data = load_h5_data(knockout_filepath, verbose=False)
        knockout_probs = data['parcellation']
        knockout_coordinates = data['coordinates']

    model, tokenizer = get_model_and_tokenizer(
        model_name,
        knockout_probs=knockout_probs,
        coordinates=knockout_coordinates,
        knockout_thresh=knockout_thresh,
        network_ix=knockout_network_ix,
        knockout_values=knockout_values,
        knockout_selection=knockout_selection,
        for_causal_lm=True
    )

    out = {}
    for domain in domains:
        _data_kwargs = resolve_domain_data_kwargs(domain, tokenizer, data_kwargs)
        input_ids, attention_mask = get_dataset(
            n_tokens=n_tokens,
            split=split,
            take=take,
            seq_len=seq_len,
            wrap=wrap,
            shuffle=shuffle,
            verbose=False,
            indent=indent,
            **_data_kwargs
        )
        out[domain] = get_lm_loss(
            model,
            input_ids,
            attention_mask,
            batch_size=batch_size,
            verbose=verbose,
            indent=indent,
            **model_kwargs
        )
        if verbose:
            stderr('%s%s: loss %.4f (ppl %.1f)\n' % (
                ' ' * indent, domain, out[domain]['loss'], out[domain]['perplexity']))

    return out


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
        seed=None,  # ADDED (8): reproducible parcellation, see below
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
        # ADDED (8): a DISTINCT seed per sample. Membership is the fraction of
        # samples that agree, so the samples must stay independent -- putting a
        # fixed `random_state` in clustering_kwargs would make all n_samples
        # clusterings identical and collapse every membership value to 0 or 1,
        # silently destroying the soft parcellation. Deriving per-sample seeds
        # keeps the samples different while making the whole run reproducible,
        # which is what lets one parcellation be replicated against another.
        _clustering_kwargs = dict(clustering_kwargs)
        if seed is not None:
            _clustering_kwargs.setdefault('random_state', int(seed) * 100003 + i)
        m = MiniBatchKMeans(n_clusters=n_networks, **_clustering_kwargs)
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
    """Map a domain name onto the `get_dataset` kwargs that load it.

    ADDED: lifted verbatim out of `run_connectivity` (no behavioural change) so
    that mean-activation collection can draw from the same corpora by the same
    rules. Single source of truth for what a domain name means.
    """
    if data_kwargs is None:
        data_kwargs = {}
    _data_kwargs = copy.deepcopy(data_kwargs)
    if domain == 'wikitext':
        _data_kwargs.update(dict(
            dataset='Salesforce/wikitext',  # BASELINE PATCH 1 (was 'wikitext')
            name='wikitext-103-raw-v1',
        ))
    elif domain == 'bookcorpus':
        _data_kwargs.update(dict(
            dataset='bookcorpus/bookcorpus'  # BASELINE PATCH 1 (was 'bookcorpus')
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
        # BASELINE PATCH 1 (was dataset='webis/tldr-17'): that repo is
        # script-only, and datasets>=4 removed script support outright
        # ("Dataset scripts are no longer supported, but found tldr-17.py"), so
        # the original cannot load this corpus at all. Load HF's auto-converted
        # parquet branch through the packaged 'parquet' builder instead, which
        # is the same text by another route. Matches the fork's own fix.
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
    # BASELINE PATCH 1: `trust_remote_code` was removed in datasets>=3.0 and
    # now raises. Original line: _data_kwargs['trust_remote_code'] = True
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
    """ADDED (1): collect each neuron's mean activation, to be used as the
    "mean-out" clamp value.

    Zero-out sets a knocked-out neuron to 0, which is off-distribution: no
    neuron's resting state is 0, so part of the damage is the shock of an
    impossible value rather than the loss of the network. Mean-out clamps to a
    mean activation instead, so the neuron goes uninformative while staying in
    its normal range.

    Returns ``(mean_activations, domain_means, coordinates)``:
    ``mean_activations`` is the token-weighted mean over ALL domains (<n_units>),
    the default clamp; ``domain_means`` maps each domain to its own mean vector
    (<n_units>), so a network can instead be clamped to one corpus's mean; and
    ``coordinates`` is aligned row-for-row with both, exactly as
    `get_timecourses` returns them.

    Cached to ``mean_activations.h5`` under ``output_dir`` -- this is a full
    forward pass over every domain, so it is computed once per run and reused by
    every mean-out condition.
    """
    if model_kwargs is None:
        model_kwargs = {}
    if n_tokens is None:
        n_tokens = (N_TOKENS // (seq_len * batch_size)) * seq_len * batch_size

    if isinstance(domains, str):
        domains = (domains,)

    filepath = os.path.join(output_dir, '%s%s' % (MEAN_ACTIVATION_NAME, EXTENSION))
    if os.path.exists(filepath) and not overwrite:
        data = load_h5_data(filepath, verbose=verbose, indent=indent)
        cached = unpack_mean_activations(data)
        # Only reuse a cache that covers every domain asked for.
        if cached is not None and all(d in cached[1] for d in domains):
            return cached

    if verbose:
        stderr('%sComputing mean activations for mean-out\n' % (' ' * indent))
    indent += 2

    model, tokenizer = get_model_and_tokenizer(model_name)

    domain_means = {}
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
        # Raw activations: no bandpass, no PCA/ICA. The clamp value has to be on
        # the same scale as what the layer actually emits at inference.
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
        domain_sum = timecourses.sum(axis=1).astype(np.float64)
        domain_count = timecourses.shape[1]
        domain_means[domain] = (domain_sum / max(domain_count, 1)).astype(np.float32)
        # Token-weighted, so the aggregate mixes domains in proportion to how
        # much text each contributed rather than treating a tiny domain as equal.
        if running_sum is None:
            running_sum = domain_sum.copy()
        else:
            running_sum += domain_sum
        running_count += domain_count

    mean_activations = (running_sum / max(running_count, 1)).astype(np.float32)

    # `domains` is stored alongside so the stacked per-corpus means can be
    # unpacked back into a dict; h5 has no dict type.
    domain_names = list(domain_means)
    save_h5_data(
        dict(
            mean_activations=mean_activations,
            domain_means=np.stack([domain_means[d] for d in domain_names], axis=0),
            domains=np.array([d.encode('utf-8') for d in domain_names]),
            coordinates=coordinates
        ),
        filepath,
        verbose=verbose,
        indent=indent
    )

    return mean_activations, domain_means, coordinates


def unpack_mean_activations(data):
    """ADDED (1): rebuild ``(mean_activations, domain_means, coordinates)`` from a
    loaded ``mean_activations.h5``, or None if the payload is incomplete."""
    required = ('mean_activations', 'domain_means', 'domains', 'coordinates')
    if not all(k in data for k in required):
        return None
    names = [d.decode('utf-8') if isinstance(d, bytes) else str(d)
             for d in data['domains']]
    domain_means = {name: np.asarray(row)
                    for name, row in zip(names, np.asarray(data['domain_means']))}
    return data['mean_activations'], domain_means, data['coordinates']


def sample_baseline_selection(sel, seed=0):
    """ADDED (7): given a knockout mask, draw an equally sized set of units
    uniformly at random from the *complement* — units the real lesion did NOT
    touch — globally across layers.

    This is the control the whole per-network claim rests on. Freezing 1358
    neurons is expected to hurt; the question is whether freezing *these* 1358
    hurts more than freezing any 1358. Without it, every result is equally
    consistent with a generic capacity effect.
    """
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


def _mode_label(mode, mean_domain=None):
    """ADDED (4): condition-name token for a knockout mode. A corpus-specific
    mean-out clamp is tagged with the corpus it was clamped to (e.g.
    ``mean-wikitext``), so runs against different clamp corpora land in distinct
    directories. The ``mean``/``zero`` prefix stays first so the mode is still
    parseable from the name."""
    if mode == 'mean' and mean_domain is not None:
        return 'mean-%s' % mean_domain
    return mode


def select_knockout(coordinates, knockout_probs, knockout_thresh=0.5, network_ix=None):
    """ADDED (2): build the boolean unit mask for knocking out ONE network.

    `knockout_probs` is <n_units, n_networks> soft membership. The original code
    OR-ed every column together and lesioned the union of all shared subnetworks
    in a single pass, which cannot attribute damage to any particular network --
    so `network_ix` is required here whenever more than one network is present.

    Returns ``(perturbation_coordinates, sel)``.
    """
    probs = np.asarray(knockout_probs)
    if probs.ndim == 1:
        sel = probs >= knockout_thresh
    else:
        assert network_ix is not None or probs.shape[1] == 1, \
            'network_ix is required: refusing to knock out the union of %d networks ' \
            'at once (damage could not be attributed to any one of them)' % probs.shape[1]
        sel = probs[:, network_ix or 0] >= knockout_thresh
    return coordinates[sel], sel


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
        knockout_network_ix=None,  # ADDED (2): lesion one network, not the union
        knockout_values=None,      # ADDED (1): mean-out clamp values, <n_units>
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
        knockout_thresh=knockout_thresh,
        network_ix=knockout_network_ix,
        knockout_values=knockout_values
    )

    if isinstance(domains, str):
        domains = (domains,)

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
        indent -= 2


def run_parcellation(
        output_dir=OUTPUT_DIR,
        n_networks=50,
        n_samples=100,
        binarize_connectivity=True,
        connectivity_pca_components=200,
        connectivity_ica_components=None,
        clustering_kwargs=None,
        seed=None,  # ADDED (8)
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
                seed=seed,
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
        knockout_mode=('zero', 'mean'),  # ADDED (1): 'mean' = mean-out
        knockout_thresh=0.5,
        networks=None,                   # ADDED (2): which networks; None = each
        mean_domain=None,                # ADDED (4): clamp corpus, hold it out
        eval_loss=True,                  # ADDED (6): next-token loss per condition
        n_baseline=2,                    # ADDED (7): size-matched random controls
        baseline_seed=0,
        steps=('plot_stability',),
        overwrite=False,
        verbose=True,
        indent=0
):
    """Knock out each shared subnetwork INDIVIDUALLY, under each perturbation mode.

    Changed from the original in two ways:

    (2) The original lesioned the *union* of every shared subnetwork in one pass
        (`get_model_and_tokenizer` OR-ed all columns of the parcellation), so a
        change in connectivity could not be attributed to any one network. Each
        network is now knocked out on its own, into its own output directory.

    (1) `knockout_mode` selects what the lesioned units are clamped to: 'zero'
        (the original behaviour) or 'mean' (mean-out, using the mean activations
        collected by `compute_mean_activations`). Both modes are driven from the
        SAME parcellation and the same per-network selection, so mean-out and
        zero-out lesion identical units and are directly comparable.

    Conditions are written to ``<output_dir>/knockout/network<i>_<mode>/``.
    """
    if connectivity_kwargs is None:
        connectivity_kwargs = {}
    # `model_name` is also a legitimate connectivity key; take it from there when
    # present, otherwise Python raises on the duplicate keyword below.
    connectivity_kwargs = dict(connectivity_kwargs)
    model_name = connectivity_kwargs.pop('model_name', model_name)
    if isinstance(knockout_mode, str):
        knockout_mode = (knockout_mode,)
    for mode in knockout_mode:
        assert mode in ('zero', 'mean'), 'Unrecognized knockout_mode: %s' % mode

    # ADDED (4): the corpora available to evaluate on, and which of them (if any)
    # supply the mean-out clamp. `mean_domain` accepts None (clamp to the
    # cross-domain aggregate and evaluate on everything, the prior behaviour), a
    # corpus name, 'each' (one condition per corpus), or an explicit list.
    all_domains = connectivity_kwargs.get(
        'domains', ('wikitext', 'bookcorpus', 'agnews', 'tldr17', 'codeparrot', 'random', 'whitespace')
    )
    if isinstance(all_domains, str):
        all_domains = (all_domains,)
    all_domains = list(all_domains)

    if mean_domain is None:
        clamp_domains = [None]
    elif mean_domain == 'each':
        clamp_domains = list(all_domains)
    elif isinstance(mean_domain, str):
        clamp_domains = [mean_domain]
    else:
        clamp_domains = list(mean_domain)
    for d in clamp_domains:
        if d is None:
            continue
        assert d in all_domains, \
            'mean_domain %r is not among the evaluated domains: %s' % (d, ', '.join(all_domains))
        assert len(all_domains) > 1, \
            'Holding out %r leaves no domains to evaluate on' % d

    mode_specs = []
    for mode in knockout_mode:
        if mode == 'mean':
            mode_specs.extend((mode, d) for d in clamp_domains)
        else:
            mode_specs.append((mode, None))

    subnetwork_dir = os.path.join(output_dir, SUBNETWORK_NAME)
    knockout_dir = os.path.join(output_dir, KNOCKOUT_NAME)

    if verbose:
        stderr('Running knockout\n')
    indent += 2

    if not os.path.exists(knockout_dir):
        os.makedirs(knockout_dir)

    loss_rows = []  # ADDED (6)

    # Mean-out clamp values. Computed once (a full pass over every domain) and
    # shared by every mean-out condition; skipped entirely if only zeroing out.
    mean_activations = None
    domain_means = {}
    if 'mean' in knockout_mode:
        mean_activations, domain_means, mean_coordinates = compute_mean_activations(
            model_name=model_name,
            output_dir=output_dir,
            overwrite=overwrite,
            verbose=verbose,
            indent=indent,
            **{k: v for k, v in connectivity_kwargs.items()
               if k in ('domains', 'seq_len', 'n_tokens', 'split', 'take', 'wrap',
                        'shuffle', 'batch_size', 'data_kwargs', 'model_kwargs')}
        )

    for path in sorted(os.listdir(subnetwork_dir)):
        match = INPUT_NAME_RE.match(path)
        if not match:
            continue
        knockout_filepath = os.path.join(subnetwork_dir, path)
        data = load_h5_data(knockout_filepath, verbose=False)
        if 'parcellation' not in data:
            continue

        parcellation = np.asarray(data['parcellation'])
        if parcellation.ndim == 1:
            parcellation = parcellation[:, None]
        n_networks = parcellation.shape[1]
        network_ixs = range(n_networks) if networks is None else networks

        if mean_activations is not None:
            # The clamp vector is indexed by the same unit ordering as the
            # parcellation; if that ever drifts, mean-out would clamp the wrong
            # neurons to plausible-looking values and fail silently.
            assert np.array_equal(np.asarray(mean_coordinates), np.asarray(data['coordinates'])), \
                'Mean-activation coordinates do not match the parcellation coordinates'

        for network_ix in network_ixs:
            assert 0 <= network_ix < n_networks, \
                'Network %d out of range (%d networks found)' % (network_ix, n_networks)
            n_units = int((parcellation[:, network_ix] >= knockout_thresh).sum())
            if not n_units:
                if verbose:
                    stderr('%sSkipping network%d: no units at/above thresh %s\n' % (
                        ' ' * indent, network_ix, knockout_thresh))
                continue

            for mode, clamp_domain in mode_specs:
                condition = 'network%d_%s' % (network_ix, _mode_label(mode, clamp_domain))
                condition_dir = os.path.join(knockout_dir, condition)

                # ADDED (4): what the units are clamped to, and what the lesioned
                # model is then fed.
                _connectivity_kwargs = dict(connectivity_kwargs)
                if mode == 'mean' and clamp_domain is not None:
                    knockout_values = domain_means[clamp_domain]
                    _connectivity_kwargs['domains'] = [
                        d for d in all_domains if d != clamp_domain
                    ]
                else:
                    knockout_values = mean_activations if mode == 'mean' else None

                if verbose:
                    if clamp_domain is None:
                        stderr('%sCondition %s (%d units)\n' % (
                            ' ' * indent, condition, n_units))
                    else:
                        stderr('%sCondition %s (%d units) -- clamped to %s, evaluated on %s\n' % (
                            ' ' * indent, condition, n_units, clamp_domain,
                            ', '.join(_connectivity_kwargs['domains'])))

                run_connectivity(
                    model_name=model_name,
                    output_dir=condition_dir,
                    knockout_filepath=knockout_filepath,
                    knockout_thresh=knockout_thresh,
                    knockout_network_ix=network_ix,
                    knockout_values=knockout_values,
                    overwrite=overwrite,
                    verbose=verbose,
                    indent=indent + 2,
                    **_connectivity_kwargs
                )

                # ADDED (6): next-token loss for this condition, on the same
                # domains the connectivity was computed over.
                if eval_loss:
                    losses = run_condition_loss(
                        model_name=model_name,
                        knockout_filepath=knockout_filepath,
                        knockout_thresh=knockout_thresh,
                        knockout_network_ix=network_ix,
                        knockout_values=knockout_values,
                        verbose=verbose,
                        indent=indent + 2,
                        **_connectivity_kwargs
                    )
                    for domain, res in losses.items():
                        loss_rows.append(dict(
                            condition=condition,
                            kind='knockout',
                            network='network%d' % network_ix,
                            mode=mode,
                            clamp_domain=clamp_domain or '',
                            domain=domain,
                            **res
                        ))

                    # ADDED (7): size-matched random controls for THIS condition.
                    # Same number of units, drawn from the complement so they are
                    # disjoint from the real lesion, and clamped to the very same
                    # values. The only thing that differs is *which* units --
                    # which is exactly the comparison the per-network claim needs.
                    #
                    # Loss only, deliberately: connectivity for baselines would
                    # roughly triple the run, and the corrected stability analysis
                    # showed no usable signal, so loss is the operative metric.
                    _, real_sel = select_knockout(
                        data['coordinates'], parcellation,
                        knockout_thresh=knockout_thresh, network_ix=network_ix
                    )
                    if n_baseline and int(real_sel.sum()) > len(real_sel) - int(real_sel.sum()):
                        if verbose:
                            stderr('%sSkipping baselines for %s: %d units leaves too '
                                   'small a complement for a disjoint draw\n' % (
                                       ' ' * indent, condition, int(real_sel.sum())))
                    else:
                        for b in range(n_baseline):
                            base_sel = sample_baseline_selection(real_sel, seed=baseline_seed + b)
                            if verbose:
                                stderr('%s  baseline %d/%d (%d random units)\n' % (
                                    ' ' * indent, b + 1, n_baseline, int(base_sel.sum())))
                            b_losses = run_condition_loss(
                                model_name=model_name,
                                knockout_filepath=knockout_filepath,
                                knockout_selection=base_sel,
                                knockout_values=knockout_values,
                                verbose=False,
                                indent=indent + 4,
                                **_connectivity_kwargs
                            )
                            for domain, res in b_losses.items():
                                loss_rows.append(dict(
                                    condition='%s_%s%d' % (condition, BASELINE_NAME, b),
                                    kind='baseline',
                                    network='network%d' % network_ix,
                                    mode=mode,
                                    clamp_domain=clamp_domain or '',
                                    domain=domain,
                                    **res
                                ))

                # ADDED (3): record exactly which units were lesioned, so the
                # stability summary can exclude them explicitly.
                _, sel = select_knockout(
                    data['coordinates'], parcellation,
                    knockout_thresh=knockout_thresh, network_ix=network_ix
                )
                save_h5_data(
                    dict(selection=sel.astype(np.uint8)),
                    os.path.join(condition_dir, '%s_selection%s' % (KNOCKOUT_NAME, EXTENSION)),
                    verbose=False
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

    # ADDED (6): the unperturbed loss. Unlike the stability reference this DOES
    # need model passes -- loss cannot be recovered from stored connectivity --
    # but it is one pass per domain, and without it every knockout loss is a
    # number with nothing to be worse than.
    if eval_loss:
        if verbose:
            stderr('%sHealthy (unperturbed) loss reference\n' % (' ' * indent))
        healthy_losses = run_condition_loss(
            model_name=model_name,
            knockout_filepath=None,
            verbose=verbose,
            indent=indent + 2,
            **connectivity_kwargs
        )
        for domain, res in healthy_losses.items():
            loss_rows.append(dict(
                condition=HEALTHY_NAME, kind='healthy', network='', mode='healthy',
                clamp_domain='', domain=domain, **res
            ))

        if loss_rows:
            df = pd.DataFrame(loss_rows)
            # Every knockout row gains `delta_loss`: how much worse than healthy
            # on the SAME domain. That difference, not the raw loss, is the
            # quantity to compare across conditions.
            healthy_by_domain = df[df['mode'] == 'healthy'].set_index('domain')['loss']
            df['delta_loss'] = df.apply(
                lambda r: r['loss'] - healthy_by_domain.get(r['domain'], np.nan), axis=1
            )
            df = df.sort_values(['domain', 'network', 'mode', 'clamp_domain'])
            loss_path = os.path.join(knockout_dir, '%s_summary.csv' % LOSS_NAME)
            df.to_csv(loss_path, index=False)
            if verbose:
                stderr('%sWrote loss summary to %s\n' % (' ' * indent, loss_path))

    # ADDED (3): distil every condition's stability matrix to one number.
    # ADDED (5): `healthy_dir` is the run root, whose connectivity/ is the
    # unperturbed model -- so the reference costs no extra model passes.
    summarize_knockout_stability(
        knockout_dir, healthy_dir=output_dir, verbose=verbose, indent=indent
    )
