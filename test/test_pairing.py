"""Unit tests for the paired-evaluation fix.

These guard the change that makes a knockout and its size-matched random
baselines comparable *on the same text*:

    python3 -m unittest discover -s test -v
    # or, if pytest is available:
    pytest test/test_pairing.py -v

The scientific claim they protect: `get_dataset` used to shuffle the corpus with
the unseeded global numpy RNG, so every call returned a different sample. Since
the knockout pipeline calls it once per condition, a knockout and its baselines
were each scored on different text, and corpus-sampling variance sat inside the
selectivity comparison as if it were lesion effect. If the seeding silently
breaks, that confound comes back and nothing downstream will notice.

The second half covers the per-sequence losses that seeding makes meaningful --
without identical tokens across conditions, a per-sequence contrast is not
paired and the vectors are not comparable element-wise.

No GPU, no model download, and no network access: the corpus paths are exercised
through the built-in `random`/`whitespace` generators and a stubbed
`datasets.load_dataset`, and the loss path through a fake model.
"""
import unittest

import numpy as np
import torch

from parcelmate import data as pm_data
from parcelmate.data import get_dataset, clear_dataset_cache, BaselineDataset
from parcelmate.model import get_lm_loss


class FakeTokenizer:
    """Minimal stand-in for an HF tokenizer: whitespace tokens hashed into a
    small vocabulary. Counts calls so the cache can be shown to prevent work."""

    name_or_path = 'fake-tokenizer'
    vocab_size = 64

    def __init__(self):
        self.n_calls = 0
        self.all_special_tokens = ['<pad>']

    def __len__(self):
        return self.vocab_size

    def get_vocab(self):
        vocab = {'w%d' % i: i for i in range(self.vocab_size)}
        vocab['<pad>'] = 0
        return vocab

    def decode(self, toks):
        return ' '.join('w%d' % int(t) for t in toks)

    def __call__(self, text):
        self.n_calls += 1
        ids = [(hash(w) % self.vocab_size) for w in text.split()] or [1]
        return {'input_ids': ids, 'attention_mask': [1] * len(ids)}


class FakeCausalLM(torch.nn.Module):
    """Returns fixed, position-dependent logits so per-sequence losses differ
    across sequences but are exactly reproducible."""

    def __init__(self, vocab_size=64):
        super().__init__()
        self.vocab_size = vocab_size
        # A parameter so `.to(device)` / `.dtype` behave like a real model.
        self.scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        b, t = input_ids.shape
        pos = torch.arange(t, dtype=torch.float32).reshape(1, t, 1)
        vocab = torch.arange(self.vocab_size, dtype=torch.float32).reshape(1, 1, -1)
        logits = torch.sin(pos + vocab).expand(b, t, self.vocab_size).contiguous()
        return type('Out', (), {'logits': logits})()


def _stub_hf_dataset(monkey_target, docs):
    """Replace `datasets.load_dataset` with one returning `docs` in fixed order,
    so any variation in the output comes from the shuffle, not the source."""
    class _Streamed:
        def __init__(self, rows):
            self.rows = rows

        def take(self, n):
            return _Streamed(self.rows[:n])

        def __iter__(self):
            return iter(self.rows)

    def _load_dataset(*args, **kwargs):
        return _Streamed([{'text': d} for d in docs])

    monkey_target.datasets.load_dataset = _load_dataset


class TestDatasetSeeding(unittest.TestCase):
    """The corpus draw must be a function of the seed and nothing else."""

    def setUp(self):
        clear_dataset_cache()
        self.tok = FakeTokenizer()
        self._real_load = pm_data.datasets.load_dataset
        # 200 distinct documents; a shuffle that changes will change which of
        # them land in the first `n_tokens` tokens.
        docs = ['w%d w%d w%d w%d' % (i, i + 1, i + 2, i + 3) for i in range(200)]
        _stub_hf_dataset(pm_data, docs)

    def tearDown(self):
        pm_data.datasets.load_dataset = self._real_load
        clear_dataset_cache()

    def _draw(self, seed, cache=False):
        return get_dataset(
            dataset='stub', tokenizer=self.tok, n_tokens=64, seq_len=16,
            take=200, shuffle=True, seed=seed, cache=cache, verbose=False
        )

    def test_same_seed_gives_identical_tokens(self):
        # THE central invariant: this is what makes knockout vs baseline paired.
        a_ids, a_mask = self._draw(0)
        b_ids, b_mask = self._draw(0)
        self.assertTrue(torch.equal(a_ids, b_ids))
        self.assertTrue(torch.equal(a_mask, b_mask))

    def test_different_seed_gives_different_tokens(self):
        # Guards against the seed being ignored entirely (which would also make
        # the test above pass).
        a_ids, _ = self._draw(0)
        b_ids, _ = self._draw(12345)
        self.assertFalse(torch.equal(a_ids, b_ids))

    def test_seed_none_restores_legacy_nondeterminism(self):
        # Kept reachable only so the pre-fix sampling can be reproduced.
        np.random.seed(1)
        a_ids, _ = self._draw(None)
        np.random.seed(2)
        b_ids, _ = self._draw(None)
        self.assertFalse(torch.equal(a_ids, b_ids))

    def test_default_seed_is_deterministic(self):
        # A config that never mentions a seed must still get paired evaluation;
        # the bug was precisely that the default was nondeterministic.
        a_ids, _ = get_dataset(
            dataset='stub', tokenizer=self.tok, n_tokens=64, seq_len=16,
            take=200, shuffle=True, cache=False, verbose=False
        )
        b_ids, _ = get_dataset(
            dataset='stub', tokenizer=self.tok, n_tokens=64, seq_len=16,
            take=200, shuffle=True, cache=False, verbose=False
        )
        self.assertTrue(torch.equal(a_ids, b_ids))


class TestRandomCorpusSeeding(unittest.TestCase):
    """The synthetic 'random' corpus drew from the global RNG too."""

    def setUp(self):
        clear_dataset_cache()
        self.tok = FakeTokenizer()

    def tearDown(self):
        clear_dataset_cache()

    def test_random_corpus_is_reproducible(self):
        a = BaselineDataset('random', 8, tokenizer=self.tok, seed=3)
        b = BaselineDataset('random', 8, tokenizer=self.tok, seed=3)
        self.assertEqual([next(a) for _ in range(4)], [next(b) for _ in range(4)])

    def test_random_corpus_varies_with_seed(self):
        a = BaselineDataset('random', 8, tokenizer=self.tok, seed=3)
        b = BaselineDataset('random', 8, tokenizer=self.tok, seed=4)
        self.assertNotEqual([next(a) for _ in range(4)], [next(b) for _ in range(4)])

    def test_random_domain_through_get_dataset_is_reproducible(self):
        kw = dict(dataset='random', tokenizer=self.tok, n_tokens=64, seq_len=16,
                  cache=False, verbose=False)
        a_ids, _ = get_dataset(seed=7, **kw)
        b_ids, _ = get_dataset(seed=7, **kw)
        self.assertTrue(torch.equal(a_ids, b_ids))


class TestDatasetCache(unittest.TestCase):
    """The cache is what makes a large n_baseline affordable."""

    def setUp(self):
        clear_dataset_cache()
        self.tok = FakeTokenizer()

    def tearDown(self):
        clear_dataset_cache()

    def _draw(self, **over):
        kw = dict(dataset='whitespace', tokenizer=self.tok, n_tokens=64,
                  seq_len=16, cache=True, verbose=False)
        kw.update(over)
        return get_dataset(**kw)

    def test_second_call_does_no_tokenization(self):
        self._draw()
        after_first = self.tok.n_calls
        self.assertGreater(after_first, 0)
        self._draw()
        self.assertEqual(self.tok.n_calls, after_first)

    def test_cached_result_equals_uncached(self):
        cached_ids, cached_mask = self._draw()
        clear_dataset_cache()
        fresh_ids, fresh_mask = self._draw(cache=False)
        self.assertTrue(torch.equal(cached_ids, fresh_ids))
        self.assertTrue(torch.equal(cached_mask, fresh_mask))

    def test_caller_gets_a_copy(self):
        # Every condition mutates nothing today, but a cache that hands out its
        # own tensors would silently corrupt every later condition if one ever
        # did -- and that corruption would look like a lesion effect.
        a_ids, _ = self._draw()
        a_ids[0, 0] = 999
        b_ids, _ = self._draw()
        self.assertNotEqual(int(b_ids[0, 0]), 999)

    def test_seed_is_part_of_the_cache_key(self):
        a_ids, _ = self._draw(dataset='random', seed=1)
        b_ids, _ = self._draw(dataset='random', seed=2)
        self.assertFalse(torch.equal(a_ids, b_ids))

    def test_n_tokens_is_part_of_the_cache_key(self):
        a_ids, _ = self._draw(n_tokens=64)
        b_ids, _ = self._draw(n_tokens=32)
        self.assertNotEqual(a_ids.numel(), b_ids.numel())


class TestPerSequenceLoss(unittest.TestCase):
    """Per-sequence losses must decompose the scalar loss exactly, or a paired
    test over sequences would not be measuring the reported effect."""

    def setUp(self):
        self.model = FakeCausalLM()
        torch.manual_seed(0)
        self.input_ids = torch.randint(0, 64, (7, 16))
        self.attention_mask = torch.ones_like(self.input_ids)

    def _loss(self, **over):
        kw = dict(model=self.model, input_ids=self.input_ids,
                  attention_mask=self.attention_mask, batch_size=3, verbose=False)
        kw.update(over)
        return get_lm_loss(**kw)

    def test_one_entry_per_sequence(self):
        out = self._loss()
        self.assertEqual(out['n_seqs'], self.input_ids.size(0))
        self.assertEqual(out['seq_loss'].shape, (self.input_ids.size(0),))
        self.assertEqual(out['seq_tokens'].shape, (self.input_ids.size(0),))

    def test_token_weighted_mean_recovers_scalar_loss(self):
        # The scalar loss is token-weighted, so the per-sequence vector must be
        # recombined the same way. An unweighted mean would silently disagree
        # whenever sequences differ in length.
        out = self._loss()
        recombined = (
            np.nansum(out['seq_loss'] * out['seq_tokens']) / out['seq_tokens'].sum()
        )
        self.assertAlmostEqual(recombined, out['loss'], places=6)

    def test_token_counts_sum_to_total(self):
        out = self._loss()
        self.assertEqual(int(out['seq_tokens'].sum()), out['n_tokens'])

    def test_batching_does_not_change_results(self):
        # Sequences are split across batches; a reshape error would show up as
        # per-sequence values that depend on batch_size.
        a = self._loss(batch_size=3)
        b = self._loss(batch_size=7)
        np.testing.assert_allclose(a['seq_loss'], b['seq_loss'], rtol=1e-6)

    def test_padding_is_excluded_and_empty_sequences_are_nan(self):
        mask = torch.ones_like(self.input_ids)
        mask[2, 4:] = 0   # partially padded
        mask[5, :] = 0    # fully padded -> no predicted tokens at all
        out = self._loss(attention_mask=mask)
        # A fully padded sequence must not read as a perfectly predicted one.
        self.assertTrue(np.isnan(out['seq_loss'][5]))
        self.assertEqual(int(out['seq_tokens'][5]), 0)
        self.assertEqual(int(out['seq_tokens'][2]), 3)  # 4 kept tokens, shifted
        self.assertFalse(np.isnan(out['seq_loss'][2]))

    def test_sequences_are_aligned_across_conditions(self):
        # The property the paired test depends on: same tokens in, same
        # per-sequence vector out, so element i means the same sequence in a
        # knockout run and in its baseline run.
        a = self._loss()
        b = self._loss()
        np.testing.assert_allclose(a['seq_loss'], b['seq_loss'], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
