"""Check pooled weekday/weekend distributions without reopening trip records."""
import json
import unittest
import numpy as np
import pandas as pd
from common import INPUT
from distributions import aggregate_measurements


class DistributionTests(unittest.TestCase):
    def test_monthly_counts_reconcile_with_annual_cohorts(self):
        monthly = pd.read_csv(INPUT / 'distribution-monthly.csv')
        self.assertFalse(monthly.duplicated(['mode', 'month']).any())
        self.assertTrue((monthly['count'] > 0).all())
        monthly['year'] = pd.to_datetime(monthly.month).dt.year.astype('int64')
        actual = monthly.groupby(['mode', 'year'])['count'].sum()
        cohorts = pd.DataFrame(json.loads((INPUT / 'distribution-design.json').read_text())['cohorts'])
        expected = cohorts.groupby(['mode', 'year']).retained.sum()
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_pooling_cohorts_and_daytypes(self):
        design = json.loads((INPUT / 'distribution-design.json').read_text())
        cohorts = pd.DataFrame(design['cohorts'])
        summary = pd.read_csv(INPUT / 'distribution-summary.csv')
        self.assertEqual(set(summary.daytype), {'Weekday', 'Weekend'})
        for mode in ['bike', 'car']:
            n = summary[(summary['mode'] == mode) & (summary.metric == 'displacement_km')].n.sum()
            self.assertEqual(n, cohorts[cohorts['mode'] == mode].retained.sum())
        expected_bike = sum(json.loads((INPUT/f'bike-audit-{year}.json').read_text())['valid_duration_retained']
                            for year in [2023, 2024])
        actual = summary[(summary['mode'] == 'bike') & (summary.metric == 'duration_min')].n.sum()
        self.assertEqual(actual, expected_bike)
        hourly = pd.read_csv(INPUT / 'distribution-hourly.csv')
        for (mode, daytype), frame in hourly.groupby(['mode', 'daytype']):
            self.assertEqual(set(frame.hour), set(range(24)))
            self.assertAlmostEqual(frame.share.sum(), 1)
            expected = summary[(summary['mode'] == mode) & (summary.daytype == daytype)
                               & (summary.metric == 'displacement_km')].n.iloc[0]
            self.assertEqual(frame['count'].sum(), expected)

    def test_histogram_mass_and_monotonicity(self):
        histogram = pd.read_csv(INPUT / 'distribution-histograms.csv')
        for _, frame in histogram.groupby(['mode', 'daytype', 'metric']):
            self.assertEqual(frame.n.nunique(), 1)
            self.assertEqual(frame['count'].sum(), frame.n.iloc[0])
            self.assertEqual(frame.cumulative_count.iloc[-1], frame.n.iloc[0])
            self.assertTrue((np.diff(frame.cumulative_count) >= 0).all())

    def test_fixture_quantiles_and_missing(self):
        hist, summary = aggregate_measurements({('bike', 'Weekday', 'duration_min'):
                                               [np.array([1., 2., np.nan]), np.array([3., 4.])]})
        self.assertEqual(summary.n.iloc[0], 4)
        self.assertEqual(summary['median'].iloc[0], 2.5)
        self.assertEqual(hist['count'].sum(), 4)


if __name__ == '__main__':
    unittest.main()
