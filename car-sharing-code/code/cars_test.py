"""Deterministic boundary, leakage, reconciliation and artifact-integrity tests."""
import json
import unittest
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from cars_common import ROOT, OUT, CACHE, TEXT, DST_DATES, HOLIDAYS, period, chord_distance, metrics, sha
from cars_predict import UniformScale


class Definitions(unittest.TestCase):
    def test_calendar_periods(self):
        t = pd.Series(pd.to_datetime(['2023-01-01 00:00', '2023-01-01 05:59', '2023-01-01 06:00',
                                     '2023-01-01 22:59', '2023-01-01 23:00']))
        self.assertEqual(list(period(t)), ['night', 'night', 'day', 'day', 'night'])
        self.assertIn('2023-10-04', HOLIDAYS)
        self.assertIn('2024-04-01', HOLIDAYS)

    def test_chords(self):
        ox, oy = np.array([0., 0., 1.]), np.zeros(3)
        dx, dy = np.array([2., 0., 2.]), np.zeros(3)
        u, delta = chord_distance(ox, oy, dx, dy, 1., .25)
        self.assertAlmostEqual(u[0], .5)
        self.assertAlmostEqual(delta[0], .25)
        self.assertEqual(u[1], 0.)  # Degenerate round trip is not a crossing.
        self.assertEqual(u[2], 0.)  # Endpoint projection is not an interior chord.
        self.assertTrue(np.isfinite(delta).all())

    def test_scaling_and_metrics(self):
        model = UniformScale().fit(np.array([[1.], [2.]]), np.array([3., 6.]), np.array([7., 17.]))
        np.testing.assert_allclose(model.predict([[2.]]), [6.])
        self.assertEqual(metrics([10, 20], [5, 25])['wape'], 1 / 3)
        self.assertEqual(metrics([10, 20], [5, 25])['bias'], 0.)
        self.assertTrue(np.isnan(metrics([0], [1])['wape']))


class Artifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sites = pd.read_csv(OUT / 'sites.csv').set_index('site')
        cls.pred = pd.read_parquet(OUT / 'predictions.parquet')
        cls.counts = pd.read_parquet(CACHE / 'counts.parquet')

    def test_input_hashes(self):
        manifest = json.loads((OUT / 'input-manifest.json').read_text())
        for path, checksum in manifest['inputs'].items():
            self.assertEqual(sha(ROOT / path), checksum, path)

    def test_rental_audit_and_privacy(self):
        audit = pd.read_csv(OUT / 'rental-audit.csv')
        removed = ['exact_duplicates', 'invalid_start_year', 'outside_endpoint_box', 'invalid_or_short_duration']
        np.testing.assert_array_equal(audit.initial - audit[removed].sum(axis=1), audit.retained)
        d = pd.read_parquet(CACHE / 'trips.parquet')
        self.assertEqual(len(d), audit.retained.sum())
        self.assertTrue(d.seconds.ge(30).all())
        self.assertGreater(d.displacement.lt(.1).sum(), 0)
        for name in ('Targa', 'Codice utente', 'user_id', 'plate'):
            self.assertNotIn(name, d.columns)

    def test_counter_audit(self):
        audit = pd.read_csv(OUT / 'counter-audit.csv')
        removed = ['not_active_loop', 'invalid_key_coordinates_date', 'invalid_hour_values',
                   'conflicting_station_day', 'duplicate_station_day_aliases', 'clock_change_dates']
        np.testing.assert_array_equal(audit.initial - audit[removed].sum(axis=1), audit.unique_station_days)
        np.testing.assert_array_equal(2 * audit.unique_station_days - audit.day_incomplete_quality - audit.night_incomplete_quality,
                                      audit.valid_periods)
        self.assertFalse(self.counts.duplicated(['site', 'date', 'regime']).any())
        self.assertFalse(self.counts.date.isin(DST_DATES).any())
        self.assertTrue(self.counts['count'].ge(0).all())
        np.testing.assert_array_equal(self.counts.exposure, self.counts.regime.map({'day': 17, 'night': 7}))

    def test_features(self):
        for filename in ('features.parquet', 'features-short.parquet'):
            d = pd.read_parquet(CACHE / filename)
            self.assertFalse(d.duplicated(['site', 'date', 'regime']).any())
            for c in ('global_trips', 'starts_500m', 'ends_500m', 'chord_100', 'chord_250'):
                self.assertTrue(d[c].ge(0).all())
            self.assertTrue(d.chord_100.le(d.chord_250).all())
            self.assertTrue(d.chord_250.le(d.global_trips).all())
            self.assertTrue(d.groupby(['date', 'regime']).global_trips.nunique().eq(1).all())

    def test_spatial_exclusion(self):
        design = json.loads((OUT / 'prediction-design.json').read_text())
        tests = []
        for fold in design['folds']:
            train = self.sites.loc[fold['train_sites']]
            test = self.sites.loc[fold['test_sites']]
            self.assertFalse(set(train.index) & set(test.index))
            self.assertFalse(set(train.cluster) & set(test.cluster))
            self.assertFalse(set(train.block) & set(test.block))
            distance = cKDTree(test[['x', 'y']]).query(train[['x', 'y']])[0]
            self.assertTrue((distance > .25).all())
            tests.extend(test.index)
        self.assertEqual(sorted(tests), sorted(self.sites.index))

    def test_matched_predictions(self):
        d = self.pred
        self.assertTrue(np.isfinite(d.prediction).all())
        self.assertTrue(d.prediction.ge(0).all())
        self.assertFalse(d.duplicated(['model', 'site', 'date', 'regime']).any())
        self.assertTrue(d.groupby(['site', 'date', 'regime']).model.nunique().eq(11).all())
        summary = pd.read_csv(OUT / 'prediction-summary.csv')
        for r in summary.itertuples():
            z = d.loc[d.year.eq(r.year) & d.model.eq(r.model)]
            self.assertAlmostEqual(metrics(z['count'], z.prediction)['wape'], r.pooled_wape)
            self.assertEqual(len(z), len(self.counts.loc[self.counts.date.str.startswith(str(r.year))]))

    def test_tables_and_figures(self):
        for name in ('monthly', 'hourly', 'distributions', 'activity', 'balances', 'folds', 'prediction', 'observed-predicted'):
            self.assertGreater((TEXT / f'figures/cars-{name}.pdf').stat().st_size, 1000)
        for path in (TEXT / 'tables').glob('cars-*.tex'):
            self.assertNotIn('nan', path.read_text().lower(), path.name)

    def test_generated_provenance(self):
        provenance = json.loads((OUT / 'reproducibility.json').read_text())
        for name, checksum in provenance['code_sha256'].items():
            self.assertEqual(sha(ROOT / 'code' / name), checksum)
        for name, checksum in provenance['generated_tables'].items():
            self.assertEqual(sha(TEXT / 'tables' / name), checksum)


if __name__ == '__main__':
    unittest.main(verbosity=2)
