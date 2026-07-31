import time
import numpy as np
from scipy import signal
import torch
import datasets

from parcelmate.util import stderr


# ADDED (9): tokenized-corpus cache, keyed on everything that determines the
# returned tensors. `get_dataset` streams and tokenizes up to `take` documents
# on EVERY call, and the knockout pipeline calls it once per domain per
# condition -- including once per random baseline. Those tensors are identical
# whenever the arguments are, so the repeated work is pure waste, and it is what
# made a large `n_baseline` unaffordable. See `seed` below for why the tensors
# are now identical in the first place.
_DATASET_CACHE = {}
_DATASET_CACHE_MAX = 32


def clear_dataset_cache():
    """ADDED (9): drop the tokenized-corpus cache (tests, or to reclaim memory)."""
    _DATASET_CACHE.clear()


def _dataset_cache_key(dataset, tokenizer, n_tokens, seq_len, split, take, wrap,
                       shuffle, seed, kwargs):
    """ADDED (9): identity of a tokenized draw. Anything that changes the token
    tensors must appear here, or a stale draw would be silently reused."""
    tokenizer_id = (
        getattr(tokenizer, 'name_or_path', None),
        getattr(tokenizer, 'vocab_size', None),
        len(tokenizer) if hasattr(tokenizer, '__len__') else None,
    )
    return (
        repr(dataset), tokenizer_id, n_tokens, seq_len, split, take, wrap,
        shuffle, seed, repr(sorted(kwargs.items())),
    )


class BaselineDataset:
    def __init__(self, dataset_type, seq_len, tokenizer=None, seed=None):
        assert dataset_type in ('whitespace', 'random'), 'Unknown dataset type: %s' % dataset_type
        self.dataset_type = dataset_type
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        # ADDED (9): the 'random' corpus drew tokens from the unseeded global
        # numpy RNG, so it was a different corpus on every call -- the same
        # unpaired-comparison problem as the shuffle below, in its purest form.
        self.rng = np.random.default_rng(seed)
        if self.dataset_type == 'random':
            assert self.tokenizer is not None, 'Tokenizer must be provided for random dataset type'
            tokens = self.tokenizer.get_vocab()
            special_tokens = set(self.tokenizer.all_special_tokens)
            tokens = [tokens[x] for x in tokens if not x in special_tokens]
            self.tokens = tokens
        else:
            self.tokens = {}

    def __iter__(self):
        return self

    def __next__(self):
        if self.dataset_type == 'whitespace':
            text = ' ' * self.seq_len
        elif self.dataset_type == 'random':
            toks = self.rng.choice(self.tokens, size=self.seq_len, replace=True)
            text = self.tokenizer.decode(toks)
        else:
            raise ValueError('Unknown dataset type: %s' % self.dataset_type)
        return text

    def take(self, n):
        out = []
        print()
        for i in range(n):
            print('\r%d' % i)
            out.append(next(self))
        print()
        out = [next(self) for _ in range(n)]
        return out


def get_dataset(
        dataset,
        tokenizer,
        n_tokens,
        seq_len,
        split='train',
        take=100000,
        wrap=True,
        shuffle=True,
        seed=0,      # ADDED (9): reproducible corpus draw; see below
        cache=True,  # ADDED (9)
        verbose=True,
        indent=0,
        **kwargs
):
    """ADDED (9) -- `seed` and `cache`.

    `seed` fixes *which* documents are drawn. This used to be
    `np.random.shuffle(dataset)` on the unseeded global numpy RNG, and nothing in
    the package seeded it, so every call returned a DIFFERENT sample of the
    corpus. The knockout pipeline calls this once per domain per condition, so a
    knockout and its size-matched random baselines were each scored on different
    text: corpus-sampling variance sat *inside* the very comparison the
    selectivity claim rests on, indistinguishable from lesion effect. It also
    inflated the apparent baseline noise floor, which is the yardstick the
    clamp-corpus effect was measured against.

    With a fixed seed every condition sees byte-identical tokens, so
    knockout-vs-baseline becomes a *paired* contrast and can be tested per
    sequence rather than across the handful of baseline draws.

    Pass `seed=None` to restore the old nondeterministic behaviour; it exists
    only so the pre-fix sampling can be reproduced deliberately.

    `cache` memoizes the tokenized tensors. Streaming and tokenizing `take`
    documents dominates the cost of a loss-only baseline, so without this a large
    `n_baseline` is unaffordable. Callers get a copy and may mutate freely.
    """
    if verbose:
        stderr('%sGetting input data\n' % (' ' * indent))
    assert seq_len > 0, 'seq_len must be positive'

    cache_key = None
    if cache:
        cache_key = _dataset_cache_key(
            dataset, tokenizer, n_tokens, seq_len, split, take, wrap, shuffle,
            seed, kwargs
        )
        cached = _DATASET_CACHE.get(cache_key)
        if cached is not None:
            if verbose:
                stderr('%s  (reusing cached tokenization)\n' % (' ' * indent))
            return cached[0].clone(), cached[1].clone()

    if dataset in ('whitespace', 'random'):
        dataset = BaselineDataset(dataset, seq_len, tokenizer=tokenizer, seed=seed)
    else:
        dataset = datasets.load_dataset(dataset, split=split, streaming=True, **kwargs)
        if take:
            dataset = dataset.take(take)
        key = None
        _dataset = []
        for instance in dataset:
            if key is None:
                if 'text' in instance:
                    key = 'text'
                elif 'content' in instance:
                    key = 'content'
                else:
                    raise ValueError('no known content key found in dataset')
            _dataset.append(instance[key])
        dataset = _dataset
        if shuffle and take:
            if seed is None:  # ADDED (9): legacy, nondeterministic
                np.random.shuffle(dataset)
            else:
                order = np.random.default_rng(seed).permutation(len(dataset))
                dataset = [dataset[i] for i in order]
        assert split, 'split must be specified when loading a HuggingFace dataset'

    _n_tokens = 0
    input_ids = None
    attention_mask = None
    for instance in dataset:
        toks = tokenizer(instance)
        ids, mask = toks['input_ids'], toks['attention_mask']
        if _n_tokens + len(ids) > n_tokens:
            ids = ids[:n_tokens - _n_tokens]
            mask = mask[:n_tokens - _n_tokens]
        __n_tokens = len(ids)
        if wrap:
            if input_ids is None:
                input_ids = [[]]
                attention_mask = [[]]
            elif len(input_ids[-1]) == seq_len:  # Start a new batch item
                input_ids.append([])
                attention_mask.append([])
            n = seq_len - len(input_ids[-1])
            assert n > 0, 'non-positive n found when wrapping input data. len(input_ids[-1]) = %d' % len(input_ids[-1])
            input_ids[-1].extend(ids[:n])
            attention_mask[-1].extend(mask[:n])
            ids = ids[n:]
            mask = mask[n:]
            while ids:  # Wrap
                n = min(len(ids), seq_len)
                input_ids.append(ids[:n])
                attention_mask.append(mask[:n])
                ids = ids[n:]
                mask = mask[n:]
        else:
            if input_ids is None:
                input_ids = []
                attention_mask = []
            input_ids.append(ids)
            attention_mask.append(mask)
        _n_tokens += __n_tokens
        if _n_tokens >= n_tokens:
            break

    assert n_tokens == _n_tokens, ('%d tokens requested but the dataset only contains %d tokens.'
                                   ' Consider increasing the value of `take`.'
                                   % (n_tokens, _n_tokens))

    input_ids = torch.as_tensor(pad(input_ids))
    attention_mask = torch.as_tensor(pad(attention_mask))

    if cache:  # ADDED (9)
        if len(_DATASET_CACHE) >= _DATASET_CACHE_MAX:
            _DATASET_CACHE.pop(next(iter(_DATASET_CACHE)))
        _DATASET_CACHE[cache_key] = (input_ids, attention_mask)
        return input_ids.clone(), attention_mask.clone()

    return input_ids, attention_mask


def pad(arr, max_len=None, pad_value=0, right=True):
    if max_len is None:
        max_len = max(len(x) for x in arr)
    if right:
        out = [x + [pad_value] * (max_len - len(x)) for x in arr]
    else:
        out = [[pad_value] * (max_len - len(x)) + x for x in arr]
    return out


def standardize_array(arr, axis=-1):
    out = (arr - arr.mean(axis=axis, keepdims=True)) / arr.std(axis=axis, keepdims=True)
    out = np.where(np.isfinite(out), out, np.zeros_like(out))

    return out


def minmax_normalize_array(arr, axis=None):
    out = arr - arr.min(axis=axis, keepdims=True)
    out = out / out.max(axis=axis, keepdims=True)
    out = np.where(np.isfinite(out), out, np.zeros_like(out))

    return out


def get_bandpass_filter(step=None, lower=None, upper=None, order=5):
    assert lower is not None or upper is not None, 'At least one of the lower (hi-pass) or upper (lo-pass) ' + \
                                                   'parameters must be provided.'
    assert step is not None, 'step must be provided.'
    fs = 1 / step
    Wn = []
    btype = None
    if lower is not None:
        Wn.append(lower)
        btype = 'highpass'
    if upper is not None:
        Wn.append(upper)
        if btype is None:
            btype = 'lowpass'
        else:
            btype = 'bandpass'
    if len(Wn) == 1:
        Wn = Wn[0]

    return signal.butter(order, Wn, fs=fs, btype=btype)


def bandpass(arr, step=None, lower=None, upper=None, order=5, axis=-1):
    if (lower is None and upper is None) or step is None:
        return arr
    b, a = get_bandpass_filter(step=step, lower=lower, upper=upper, order=order)
    out = signal.lfilter(b, a, arr, axis=axis)

    return out


def correlate(X, rowvar=True, use_gpu=True):
    if rowvar:
        X = X.T
    t = X.shape[0]
    X -= X.mean(axis=0, keepdims=True)
    X /= np.linalg.norm(X, axis=0, keepdims=True)

    # BASELINE PATCH 2: fall back to the CPU path off-GPU; the original
    # unconditionally calls torch.cuda.get_device_properties(0) below and
    # crashes on a machine without CUDA. No-op on a GPU node.
    use_gpu = use_gpu and torch.cuda.is_available()
    if use_gpu:
        X_ = torch.as_tensor(X)
        n_bytes = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        n_bytes *= 0.9  # Shrink allocation to avoid edge cases
        assert n_bytes > 0, 'No memory available on GPU'
        assert n_bytes / 8 > t, 'Not enough GPU memory to compute correlation matrix'
        k = int(n_bytes / (t * 8))
        k = min(k, X.shape[1])
        device = torch.device('cuda:0')
        R = np.zeros((X.shape[1], X.shape[1]), dtype=X.dtype)
        X1 = torch.zeros([t, k], device=device)
        X2 = torch.zeros([t, k], device=device)
        for i in range(0, X.shape[1], k):
            for j in range(0, X.shape[1], k):
                ni = min(k, X.shape[1] - i)
                nj = min(k, X.shape[1] - j)
                _X1 = X1[:, :ni]
                _X2 = X2[:, :nj]
                _X1[:,:] = X_[:, i:i + k]
                _X2[:,:] = X_[:, j:j + k]
                R[i:i + k, j:j + k] = (_X1.T @ _X2).detach().cpu().numpy()
        del X1, X2
        torch.cuda.empty_cache()
    else:
        R = X.T @ X

    return R


def fisher(arr, eps=1e-3):
    return np.arctanh(np.multiply(arr, 1 - eps, out=arr), out=arr)


def fisher_average(*arrs, eps=1e-3):
    out = None
    for arr in arrs:
        if out is None:
            out = fisher(arr, eps=eps)
        else:
            out += fisher(arr, eps=eps)
    out /= len(arrs)
    out = np.tanh(out, out=out)

    return out

