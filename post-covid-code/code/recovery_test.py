"""Definition, audit, panel and numerical consistency checks (no network access)."""
import unittest
import re
import numpy as np
import pandas as pd
from recovery_prepare import ROOT, CACHE, OUT, YEARS, HOURS, clock_dates
from recovery_analyse import cells, panel, standardized, spatial_ci


class Definitions(unittest.TestCase):
    def test_manuscript_is_self_contained(self):
        text_root = ROOT.parent / 'paper-car-sharing-text'
        text = (text_root / 'main.tex').read_text()
        historical = r'original strict|previous (?:version|draft|panel)|earlier (?:version|draft)|rather than 22|22 (?:detectors|devices)'
        self.assertIsNone(re.search(historical, text, flags=re.IGNORECASE))
        self.assertNotIn('Original strict panel', (text_root / 'tables/recovery-sensitivity.tex').read_text())

    def test_clock_dates(self):
        self.assertEqual(clock_dates(2020), ['2020-03-29', '2020-10-25'])
        self.assertEqual(clock_dates(2025), ['2025-03-30', '2025-10-26'])

    def test_block_bootstrap_constant_ratio(self):
        idx = pd.Index([1, 2, 3])
        lo, hi = spatial_ci(pd.Series([8., 16., 24.], index=idx),
                            pd.Series([10., 20., 30.], index=idx),
                            pd.Series(['a', 'a', 'b'], index=idx))
        np.testing.assert_allclose([lo, hi], [80, 80])

    def test_coverage_boundaries_and_zero_denominator(self):
        index = pd.MultiIndex.from_product([[1, 2, 3, 4, 5], YEARS, range(1, 13), [0, 1]],
                                           names=['site', 'year', 'month', 'weekend'])
        c = pd.DataFrame({'total': 100., 'days': np.where(index.get_level_values('weekend') == 0, 4, 2)}, index=index)
        c.loc[(3, 2020, 1, 0), 'days'] = 3
        c.loc[(4, 2019, slice(None), slice(None)), 'total'] = 0
        c.loc[(5, 2025, slice(None), slice(None)), 'total'] = 0
        rows = []
        for site in range(1, 6):
            for year in YEARS:
                n = 179 if (site, year) == (2, 2020) else 180
                rows.extend([{'site': site, 'year': year, 'unstable': False}] * n)
        self.assertEqual(set(panel(pd.DataFrame(rows), c, 180, 4, 2)), {1, 5})
        c.loc[(1, 2020, 1, 1), 'days'] = 1
        self.assertEqual(set(panel(pd.DataFrame(rows), c, 180, 4, 2)), {5})
        self.assertEqual(set(panel(pd.DataFrame(rows), c)), {1, 2, 3, 5})
        c.loc[(1, 2020, 1, 1), 'days'] = 0
        c.loc[(3, 2020, 1, 0), 'days'] = 1
        self.assertEqual(set(panel(pd.DataFrame(rows), c)), {2, 5})

    def test_calendar_weights(self):
        index = pd.MultiIndex.from_product([[1], YEARS, range(1, 13), [0, 1]],
                                           names=['site', 'year', 'month', 'weekend'])
        values = np.where(index.get_level_values('weekend') == 0, 14., 7.)
        c = pd.DataFrame({name: values / 24 for name in HOURS}, index=index)
        c['total'] = values
        c['days'] = 100  # Observation counts must not become aggregation weights.
        s = standardized(c, [1])
        np.testing.assert_allclose(s.total, 12.)
        np.testing.assert_allclose(s[HOURS].sum(axis=1), s.total)


class Artifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = pd.read_parquet(CACHE / 'daily.parquet')

    def test_audit_reconciles(self):
        a = pd.read_csv(OUT / 'audit.csv')
        excluded = ['inactive_or_not_loop', 'invalid_identity_position_date', 'invalid_counts',
                    'conflicting_alias_rows', 'identical_alias_rows', 'clock_change_days', 'incomplete_days']
        np.testing.assert_array_equal(a.raw_rows - a[excluded].sum(axis=1), a.complete_days)
        self.assertEqual(len(self.d), a.complete_days.sum())

    def test_daily_integrity(self):
        d = self.d
        self.assertFalse(d.duplicated(['site', 'date']).any())
        self.assertTrue(np.isfinite(d[HOURS]).all().all())
        self.assertTrue(d[HOURS].ge(0).all().all())
        np.testing.assert_allclose(d[HOURS].sum(axis=1), d.total)
        for year in YEARS:
            self.assertFalse(d.loc[d.year.eq(year), 'date'].dt.strftime('%Y-%m-%d').isin(clock_dates(year)).any())

    def test_balanced_panel(self):
        c = cells(self.d)
        ids = panel(self.d, c)
        saved = pd.read_csv(OUT / 'panel-sites.csv').site
        self.assertEqual(set(ids), set(saved))
        s = standardized(c, ids)
        self.assertEqual(len(s), len(ids)*7*12)
        np.testing.assert_allclose(s[HOURS].sum(axis=1), s.total)
        self.assertFalse(self.d.loc[self.d.site.isin(ids), 'unstable'].any())

    def test_indices_and_profiles(self):
        a = pd.read_csv(OUT / 'annual.csv')
        self.assertAlmostEqual(a.loc[a.year.eq(2019), 'index'].iloc[0], 100)
        self.assertTrue((a.lower <= a['index'] + 1e-10).all())
        self.assertTrue((a.upper >= a['index'] - 1e-10).all())
        h = pd.read_csv(OUT / 'hourly.csv')
        np.testing.assert_allclose(h.groupby(['year', 'type']).share.sum(), 100)
        m = pd.read_csv(OUT / 'monthly.csv')
        self.assertEqual(len(m), 84)
        np.testing.assert_allclose(m.loc[m.year.eq(2019), 'index'], 100)
        expected = pd.read_csv(OUT / 'standardized-site-year.csv').pivot(index='site', columns='year', values='total')
        np.testing.assert_allclose(a['index'], 100*expected.sum()/expected[2019].sum())

    def test_network_membership(self):
        network = pd.read_csv(OUT / 'network-membership.csv')
        selected = pd.read_csv(OUT / 'panel-sites.csv')
        self.assertFalse(network.site.duplicated().any())
        self.assertEqual(set(network.loc[network.accepted_panel, 'site']), set(selected.site))
        self.assertGreater(len(network), len(selected))


if __name__ == '__main__':
    unittest.main(verbosity=2)
