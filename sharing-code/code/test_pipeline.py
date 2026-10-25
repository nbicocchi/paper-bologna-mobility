"""Leakage, split integrity, matched targets, raw-cohort and numerical checks."""
import json
import unittest
import numpy as np
import pandas as pd
from common import CACHE, OUT, ROOT, FEATURES, metrics, period, calendar, xy
from features import event_counts
from predict import feature_sets, history_tables, add_history, HISTORY
from analyse import aggregate, paired_contrast
from describe import map_periods, spatial_grid, rhythms


class ArithmeticTests(unittest.TestCase):
    def test_calendar_and_dst(self):
        for mode in ['bike', 'car']:
            c = calendar(mode)
            self.assertFalse(c.duplicated(['date', 'regime']).any())
            self.assertEqual(int(c.exposure.sum()), 731 * 24)
        for date in ['2023-03-26', '2023-10-29']:
            t = pd.Series(pd.to_datetime([date + ' 05:00', date + ' 06:00']).tz_localize('Europe/Rome'))
            d, r = period(t, 'bike')
            self.assertNotEqual(d.iloc[0], d.iloc[1])
            self.assertEqual(r.tolist(), ['night', 'day'])

    def test_no_future_counts_in_predictors(self):
        for mode in ['bike', 'car']:
            for scenario in ['new', 'known']:
                specs = feature_sets(mode, scenario)
                self.assertEqual(list(specs), FEATURES)
                for columns in specs.values():
                    self.assertFalse(set(columns) & {'count', 'rate', 'year', 'prediction'})
                    self.assertEqual(set(columns) & set(HISTORY), set(HISTORY) if scenario == 'known' else set())
                for a, b in zip(FEATURES[:-1], FEATURES[1:]):
                    self.assertTrue(set(specs[a]).issubset(specs[b]))

    def test_historical_profiles_ignore_test_labels(self):
        train = pd.DataFrame(dict(site=['a', 'a'], direction=[0, 0], regime=['day', 'day'],
            dow=[0, 0], year=[2023, 2023], count=[20., 60.], exposure=[2., 3.]))
        tables = history_tables(train)
        test = train.iloc[:1].assign(year=2024, count=100000.)
        result = add_history(test, tables)
        changed = add_history(test.assign(count=0.), tables)
        self.assertTrue(np.allclose(result[HISTORY], changed[HISTORY]))
        self.assertAlmostEqual(np.exp(result.log_site_rate.iloc[0]), 16.)
        with self.assertRaises(ValueError):
            history_tables(test)
        unseen_weekday = add_history(test.assign(dow=6), tables)
        self.assertTrue(unseen_weekday.history_weekday_fallback.all())
        self.assertAlmostEqual(np.exp(unseen_weekday.log_weekday_rate.iloc[0]), 16.)
        unseen_regime = add_history(test.assign(regime='night', dow=6), tables)
        self.assertTrue(unseen_regime.history_regime_fallback.all())
        self.assertAlmostEqual(np.exp(unseen_regime.log_site_rate.iloc[0]), 16.)
        with self.assertRaises(ValueError):
            add_history(test.assign(site='unknown'), tables)

    def test_event_timestamp_position_and_outside_study(self):
        # Departure at first period, return at next period, third return outside study.
        departures = event_counts(np.array([0, 0, 0]), np.ones(3, bool), 2)
        returns = event_counts(np.array([1, 1, -1]), np.ones(3, bool), 2)
        np.testing.assert_array_equal(departures, [3, 0])
        np.testing.assert_array_equal(returns, [0, 2])

    def test_ratio_of_sums_and_zero_totals(self):
        frame = pd.DataFrame([
            dict(site='a', unit='a', feature='Context', observed=100, ae=20, n=2),
            dict(site='b', unit='b', feature='Context', observed=1, ae=5, n=2),
            dict(site='a', unit='a', feature='Citywide', observed=100, ae=10, n=2),
            dict(site='b', unit='b', feature='Citywide', observed=1, ae=6, n=2)])
        r = paired_contrast(frame, 'Citywide', 'Context', np.random.default_rng(1))
        self.assertAlmostEqual(r['delta_pp'], -900 / 101)
        zero = metrics([0, 0], [1, 2])
        self.assertTrue(np.isnan(zero['wape']))
        self.assertEqual(zero['ae'], 3)
        self.assertEqual(zero['signed'], 3)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sites = pd.read_csv(OUT / 'sites.csv')
        cls.summary = pd.read_csv(OUT / 'summary.csv')

    def test_standalone_source_code(self):
        for p in (ROOT / 'code').glob('*.py'):
            self.assertNotIn('bike' + '-sharing-code/', p.read_text())
            self.assertNotIn('car' + '-sharing-code/', p.read_text())

    def test_predictions_match_between_scenarios(self):
        keys = ['site', 'date', 'regime', 'direction', 'family', 'feature']
        for mode in ['bike', 'car']:
            p = pd.read_parquet(OUT / f'{mode}-predictions.parquet')
            self.assertFalse(p.duplicated(['scenario'] + keys).any())
            expected = pd.read_parquet(CACHE / f'{mode}-model-input.parquet')
            y = p.loc[p.year.eq(2024)]
            a = y.loc[y.scenario.eq('new')].set_index(keys).sort_index()
            b = y.loc[y.scenario.eq('known')].set_index(keys).sort_index()
            pd.testing.assert_index_equal(a.index, b.index)
            np.testing.assert_array_equal(a[['count', 'exposure']], b[['count', 'exposure']])
            for scenario, year in [('new', 2023), ('new', 2024), ('known', 2024)]:
                q = p.loc[p.scenario.eq(scenario) & p.year.eq(year)]
                n = len(expected.loc[expected.year.eq(year)])
                self.assertTrue(q.groupby(['family', 'feature']).size().eq(n).all())
                self.assertEqual(q.groupby(['family', 'feature']).ngroups, 9)
            for record in json.loads((OUT / f'{mode}-prediction-design.json').read_text())['splits']:
                self.assertEqual(record['train_years'], [2023])
                self.assertLessEqual(record['train_date_max'], '2023-12-31')
                if record['scenario'] == 'new':
                    self.assertTrue(set(record['train_sites']).isdisjoint(record['test_sites']))
                    if mode == 'car':
                        self.assertGreater(record['minimum_distance_km'], .25)
                else:
                    self.assertTrue(set(record['test_sites']).issubset(record['train_sites']))
                    self.assertGreaterEqual(record['test_date_min'], '2024-01-01')

    def test_metrics_reconciled_and_all_configurations(self):
        self.assertEqual(len(self.summary), 54)
        actual = aggregate(self.sites).sort_values(['mode', 'scenario', 'year', 'family', 'feature']).reset_index(drop=True)
        np.testing.assert_allclose(actual.pooled_wape, self.summary.pooled_wape, rtol=1e-10)
        self.assertFalse(self.sites.duplicated(['mode', 'scenario', 'year', 'family', 'feature', 'site']).any())
        self.assertTrue((self.sites.ae + 1e-6 >= self.sites.signed.abs()).all())
        for _, group in self.sites.groupby(['mode', 'scenario', 'year', 'site']):
            self.assertEqual(len(group), 9)
            self.assertEqual(group.observed.nunique(), 1)
            self.assertEqual(group.n.nunique(), 1)
        zero = self.sites.loc[self.sites.observed.eq(0)]
        self.assertTrue(zero.wape.isna().all())
        np.testing.assert_allclose(zero.ae, zero.signed)

    def test_paired_contrasts_and_persistence(self):
        c = pd.read_csv(OUT / 'contrasts.csv')
        self.assertEqual(len(c), 72)
        for _, g in c.groupby(['mode', 'scenario', 'year', 'family']):
            self.assertAlmostEqual(g.loc[g.kind.eq('increment'), 'delta_pp'].sum(),
                g.loc[g.kind.eq('vs_context') & g.feature.eq('Geometry'), 'delta_pp'].iloc[0], places=9)
        p = pd.read_csv(OUT / 'persistence.csv')
        self.assertTrue(p[['improve_both', 'improve_2023_only', 'improve_2024_only', 'improve_neither']].sum(axis=1).eq(p.common_positive_sites).all())

    def test_privacy_and_cohorts(self):
        for mode in ['bike', 'car']:
            t = pd.read_parquet(CACHE / f'{mode}-trips.parquet')
            self.assertFalse(set(t.columns) & {'Codice utente', 'Targa', 'BikeID'})
            self.assertTrue(np.isfinite(t.seconds).all())
            self.assertTrue(t.seconds.ge(30).all())
            if mode == 'bike':
                self.assertTrue(t.displacement.ge(.1).all())
            audit = pd.read_csv(OUT / f'{mode}-rental-audit.csv')
            self.assertEqual(len(t), audit.retained.sum())
            self.assertEqual(set(t.start.dt.year), {2023, 2024})

    def test_training_only_motor_network_and_grouping(self):
        sites = pd.read_csv(OUT / 'car-sites.csv')
        counts = pd.read_parquet(CACHE / 'car-counts.parquet')
        coverage = counts.loc[counts.date.str.startswith('2023')].groupby('site').date.nunique()
        self.assertEqual(set(sites.site), set(coverage.index))
        self.assertTrue(coverage.ge(180).all())
        self.assertTrue(sites.groupby('cluster').block.nunique().eq(1).all())
        self.assertTrue(sites.groupby('cluster').fold.nunique().eq(1).all())
        centres = sites.groupby('cluster')[['x', 'y']].mean()
        gx, gy = np.floor(centres.x).astype(int), np.floor(centres.y).astype(int)
        expected_blocks = gx.astype(str) + ':' + gy.astype(str)
        pd.testing.assert_series_equal(expected_blocks, sites.groupby('cluster').block.first(), check_names=False)
        np.testing.assert_array_equal((gx + 2 * gy) % 5, sites.groupby('cluster').fold.first())

    def test_matched_map_exposure_and_regional_partition(self):
        g = pd.read_csv(OUT / 'spatial-grid.csv')
        self.assertFalse(g.duplicated(['mode', 'window', 'gx', 'gy']).any())
        self.assertTrue(g.normalised_balance.between(-1, 1).all())
        self.assertTrue(g.groupby('window').exposure.nunique().eq(1).all())
        regions = pd.read_csv(OUT / 'rhythm-regions.csv')
        for mode in ['bike', 'car']:
            a = g.loc[g['mode'].eq(mode) & g.in_viewport].groupby('window')[['arrivals', 'departures']].sum()
            b = regions.loc[regions['mode'].eq(mode)].groupby('window')[['arrivals', 'departures']].sum()
            np.testing.assert_allclose(a, b)


if __name__ == '__main__':
    unittest.main()
