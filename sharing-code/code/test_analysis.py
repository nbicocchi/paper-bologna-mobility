"""Unit tests and checks against the original, unmodified result exports."""
import unittest
import numpy as np
import pandas as pd
from analyse import load_sites, aggregate, reconcile, paired_contrast
from common import OUT, FEATURES


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sites = load_sites()
        cls.summary = aggregate(cls.sites)

    def test_source_reconciliation(self):
        self.assertEqual(len(reconcile(self.summary)), 108)

    def test_unique_and_complete(self):
        keys = ['mode', 'year', 'site', 'family', 'feature']
        self.assertFalse(self.sites.duplicated(keys).any())
        self.assertEqual(len(self.summary), 36)
        for _, group in self.sites.groupby(['mode', 'year', 'site']):
            self.assertEqual(len(group), 9)
            self.assertEqual(group.observed.nunique(), 1)
            self.assertEqual(group.n.nunique(), 1)

    def test_zero_devices_not_lost(self):
        f = self.sites[(self.sites['mode'] == 'car') & (self.sites.year == 2024)
                       & (self.sites.family == 'Boosting') & (self.sites.feature == 'Context')]
        zero = f[f.observed == 0]
        self.assertEqual(len(zero), 8)
        self.assertTrue(zero.wape.isna().all())
        self.assertGreater(zero.ae.sum(), 0)
        self.assertTrue(np.allclose(zero.ae, zero.signed))

    def test_zero_contrast(self):
        f = self.sites[(self.sites['mode'] == 'bike') & (self.sites.year == 2023)
                       & (self.sites.family == 'Poisson')]
        result = paired_contrast(f, 'Context', 'Context', np.random.default_rng(1))
        for key in ['delta_pp', 'lo_pp', 'hi_pp']:
            self.assertEqual(result[key], 0)

    def test_increment_telescopes(self):
        c = pd.read_csv(OUT / 'contrasts.csv')
        for _, group in c.groupby(['mode', 'year', 'family']):
            increments = group[group.kind == 'increment'].delta_pp.sum()
            full = group[(group.kind == 'vs_context') & (group.feature == 'Geometry')].delta_pp.iloc[0]
            self.assertAlmostEqual(increments, full, places=10)
        # Identical car boosting predictions must also have identical intervals.
        for year in [2023, 2024]:
            f = c[(c['mode'] == 'car') & (c.year == year) & (c.family == 'Boosting')
                  & (c.kind == 'vs_context')]
            self.assertTrue(np.allclose(f.lo_pp, f.lo_pp.iloc[0]))
            self.assertTrue(np.allclose(f.hi_pp, f.hi_pp.iloc[0]))

    def test_persistence_partitions(self):
        p = pd.read_csv(OUT / 'persistence.csv')
        counts = p[['improve_both', 'improve_2023_only', 'improve_2024_only', 'improve_neither']].sum(axis=1)
        self.assertTrue((counts == p.common_positive_sites).all())

    def test_absolute_errors_bound_bias(self):
        self.assertTrue((self.sites.ae + 1e-6 >= abs(self.sites.signed)).all())

    def test_fixture_pooled_contrast(self):
        # Ratio of sums, not average percentage: the quiet site must not dominate.
        frame = pd.DataFrame([
            dict(site='a', unit='a', feature='Context', observed=100, ae=20),
            dict(site='b', unit='b', feature='Context', observed=1, ae=5),
            dict(site='a', unit='a', feature='Citywide', observed=100, ae=10),
            dict(site='b', unit='b', feature='Citywide', observed=1, ae=6)])
        r = paired_contrast(frame, 'Citywide', 'Context', np.random.default_rng(1))
        self.assertAlmostEqual(r['delta_pp'], -900 / 101)


if __name__ == '__main__':
    unittest.main()
