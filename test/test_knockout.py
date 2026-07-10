"""Unit tests for the Project-3 subnetwork-knockout core logic.

These cover the *pure* pieces of the knockout pipeline — selection, size-matched
baseline sampling, and loss-summary aggregation — none of which need a GPU or a
model download, so they run fast anywhere:

    python3 -m unittest discover -s test -v
    # or, if pytest is available:
    pytest test/test_knockout.py -v

The scientific claim these guard: a subnetwork knockout must be compared against
a random baseline of *the same size*, drawn *disjointly* from the un-knocked-out
complement. If either the size-matching or the disjointness silently breaks, the
selectivity result in GOAL.md #3 becomes meaningless, so those are asserted
explicitly here.
"""
import os
import tempfile
import unittest

import numpy as np

from parcelmate.model import (
    select_knockout,
    sample_baseline_selection,
    summarize_knockout_loss,
)
from parcelmate.util import save_h5_data
from parcelmate.constants import LOSS_NAME, EXTENSION, HEALTHY_NAME, BASELINE_NAME


def _coords(n_units):
    """Dummy (layer, unit) coordinates, one row per unit."""
    return np.stack([np.zeros(n_units, dtype=int), np.arange(n_units)], axis=1)


class TestSelectKnockout(unittest.TestCase):
    def test_1d_probs_threshold_is_inclusive(self):
        # probs exactly at the threshold must be selected (>=, not >).
        probs = np.array([0.2, 0.5, 0.8, 0.49])
        coords = _coords(4)
        sel_coords, sel = select_knockout(coords, probs, knockout_thresh=0.5)
        np.testing.assert_array_equal(sel, [False, True, True, False])
        # returned coordinates match the mask
        np.testing.assert_array_equal(sel_coords, coords[sel])

    def test_2d_single_network_uses_only_that_column(self):
        # Unit 0 is high in net0 only; unit 2 high in net1 only.
        probs = np.array([[0.9, 0.1],
                          [0.1, 0.1],
                          [0.2, 0.95]])
        coords = _coords(3)
        _, sel0 = select_knockout(coords, probs, knockout_thresh=0.5, network_ix=0)
        _, sel1 = select_knockout(coords, probs, knockout_thresh=0.5, network_ix=1)
        np.testing.assert_array_equal(sel0, [True, False, False])
        np.testing.assert_array_equal(sel1, [False, False, True])

    def test_2d_union_when_network_ix_none(self):
        # network_ix=None -> union of every network at/above threshold.
        probs = np.array([[0.9, 0.1],
                          [0.1, 0.1],
                          [0.2, 0.95]])
        coords = _coords(3)
        _, sel = select_knockout(coords, probs, knockout_thresh=0.5, network_ix=None)
        np.testing.assert_array_equal(sel, [True, False, True])

    def test_higher_threshold_selects_fewer_units(self):
        # Monotonicity: raising the threshold can only shrink the selection.
        rng = np.random.default_rng(0)
        probs = rng.random((200, 5))
        coords = _coords(200)
        _, lo = select_knockout(coords, probs, knockout_thresh=0.5, network_ix=2)
        _, hi = select_knockout(coords, probs, knockout_thresh=0.9, network_ix=2)
        self.assertLessEqual(int(hi.sum()), int(lo.sum()))
        # every high-threshold unit is also selected at the lower threshold
        self.assertTrue(np.all(lo[hi]))


class TestBaselineSelection(unittest.TestCase):
    def test_baseline_is_size_matched_and_disjoint(self):
        sel = np.array([True, True, False, False, False, False])
        baseline = sample_baseline_selection(sel, seed=0)
        # same number of knocked-out units ...
        self.assertEqual(int(baseline.sum()), int(sel.sum()))
        # ... and drawn entirely from the complement (no overlap).
        self.assertFalse(np.any(baseline & sel))
        self.assertEqual(baseline.dtype, np.bool_)
        self.assertEqual(baseline.shape, sel.shape)

    def test_deterministic_given_seed(self):
        sel = np.zeros(50, dtype=bool)
        sel[:10] = True
        a = sample_baseline_selection(sel, seed=7)
        b = sample_baseline_selection(sel, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_generally_differ(self):
        sel = np.zeros(100, dtype=bool)
        sel[:20] = True
        a = sample_baseline_selection(sel, seed=0)
        b = sample_baseline_selection(sel, seed=1)
        self.assertFalse(np.array_equal(a, b))

    def test_full_complement_is_deterministic(self):
        # If the knockout takes exactly half, the baseline must be the entire
        # complement (only one possible disjoint size-matched draw).
        sel = np.array([True, True, False, False])
        baseline = sample_baseline_selection(sel, seed=123)
        np.testing.assert_array_equal(baseline, [False, False, True, True])

    def test_raises_when_knockout_exceeds_complement(self):
        # 3 knocked out, only 1 free unit -> cannot draw a disjoint match.
        sel = np.array([True, True, True, False])
        with self.assertRaises(AssertionError):
            sample_baseline_selection(sel, seed=0)


class TestSummarizeKnockoutLoss(unittest.TestCase):
    def _write_condition(self, root, name, losses):
        """losses: {domain: (loss, perplexity, n_tokens)}."""
        data = {}
        for domain, (loss, ppl, ntok) in losses.items():
            data['%s_loss' % domain] = float(loss)
            data['%s_perplexity' % domain] = float(ppl)
            data['%s_n_tokens' % domain] = int(ntok)
        save_h5_data(data, os.path.join(root, name, '%s%s' % (LOSS_NAME, EXTENSION)),
                     verbose=False)

    def test_kind_classification_and_values(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_condition(root, HEALTHY_NAME, {'wikitext': (4.0, 54.6, 5000)})
            self._write_condition(root, 'network0', {'wikitext': (9.0, 8103.0, 5000)})
            self._write_condition(root, 'network0_%s0' % BASELINE_NAME,
                                  {'wikitext': (8.0, 2981.0, 5000)})
            df = summarize_knockout_loss(root, verbose=False)

            kinds = dict(zip(df['condition'], df['kind']))
            self.assertEqual(kinds[HEALTHY_NAME], 'healthy')
            self.assertEqual(kinds['network0'], 'knockout')
            self.assertEqual(kinds['network0_%s0' % BASELINE_NAME], 'baseline')

            # values round-trip through the h5 correctly
            row = df[df['condition'] == 'network0'].iloc[0]
            self.assertAlmostEqual(row['loss'], 9.0, places=5)
            self.assertEqual(int(row['n_tokens']), 5000)

            # CSV written next to the conditions
            self.assertTrue(os.path.exists(os.path.join(root, '%s_summary.csv' % LOSS_NAME)))

    def test_multiple_domains_one_row_each(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_condition(root, 'network1',
                                  {'wikitext': (5.0, 148.0, 5000),
                                   'tldr17': (6.0, 403.0, 5000)})
            df = summarize_knockout_loss(root, verbose=False)
            self.assertEqual(set(df['domain']), {'wikitext', 'tldr17'})
            self.assertEqual(len(df), 2)

    def test_empty_root_returns_none(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(summarize_knockout_loss(root, verbose=False))


if __name__ == '__main__':
    unittest.main()
