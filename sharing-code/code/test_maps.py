"""Map arithmetic and cartographic provenance, without raw rental coordinates."""
import unittest
import numpy as np
import pandas as pd
from common import INPUT
from maps import basemap, load_grid, project, rhythm_summaries, MINIMUM


class MapTests(unittest.TestCase):
    def test_sensor_map_matches_experimental_networks(self):
        car = pd.read_csv(INPUT / 'car-counter-locations.csv')
        scores = pd.read_csv(INPUT / 'car-sites.csv')
        self.assertEqual(len(car), 752)
        self.assertEqual(set(car.site), set(scores.query('year == 2023').site))
        groups = car.groupby('cluster')
        self.assertEqual(groups.ngroups, 246)
        self.assertEqual(car.block.nunique(), 63)
        self.assertTrue(groups.block.nunique().eq(1).all())
        self.assertTrue(groups.fold.nunique().eq(1).all())
        centres = groups[['x', 'y']].mean()
        expected_blocks = (np.floor(centres.x).astype(int).astype(str) + ':'
                           + np.floor(centres.y).astype(int).astype(str))
        self.assertTrue(expected_blocks.eq(groups.block.first()).all())
        bike = pd.read_csv(INPUT / 'bike-counter-locations.csv')
        selected = set(pd.read_csv(INPUT / 'bike-sites.csv').site)
        self.assertEqual(selected, {'Ercolani', 'Parri', 'Sabotino', 'San Donato'})
        self.assertTrue(selected.issubset(set(bike.site)))
        _, extent = basemap()
        for frame in [car, bike[bike.site.isin(selected)]]:
            self.assertTrue(np.isfinite(frame[['x', 'y']]).all().all())
            x, y = project(frame.x, frame.y)
            self.assertTrue(((x >= extent[0]) & (x <= extent[1])).all())
            self.assertTrue(((y >= extent[2]) & (y <= extent[3])).all())

    def test_original_values_reconciled(self):
        for mode in MINIMUM:
            original = pd.read_csv(INPUT / f'{mode}-map-grid.csv')
            grid = load_grid(mode)
            original_rate = original.events_per_hour if mode == 'bike' else original.activity
            original_balance = original.ratio if mode == 'bike' else original.balance
            self.assertTrue(np.allclose(grid.events_per_hour, original_rate))
            self.assertTrue(np.allclose(grid.normalised_balance, original_balance))

    def test_balance_and_thresholds(self):
        for mode in MINIMUM:
            grid = load_grid(mode)
            self.assertTrue(grid.normalised_balance.between(-1, 1).all())
            self.assertTrue((grid.net == grid.arrivals-grid.departures).all())
            self.assertTrue((grid.displayed == (grid.events >= MINIMUM[mode])).all())
            self.assertEqual(set(grid.window), {0, 1, 2, 3})
            self.assertTrue((grid.groupby('window').exposure.nunique() == 1).all())

    def test_cached_basemap_hashes_and_extent(self):
        image, extent = basemap()
        self.assertEqual(image.shape, (768, 768, 3))
        x, y = project(np.array([-7, 8]), np.array([-5, 6]))
        self.assertLessEqual(extent[0], x.min())
        self.assertGreaterEqual(extent[1], x.max())
        self.assertLessEqual(extent[2], y.min())
        self.assertGreaterEqual(extent[3], y.max())

    def test_rhythm_region_partition_and_reversals(self):
        regions, reversals, concordance = rhythm_summaries()
        for mode in MINIMUM:
            grid = load_grid(mode)
            expected = grid[grid.in_viewport].groupby('window')[['arrivals', 'departures']].sum()
            actual = regions[regions['mode'] == mode].groupby('window')[['arrivals', 'departures']].sum()
            self.assertTrue(np.allclose(actual, expected))
        self.assertTrue((reversals.reversing_cells <= reversals.common_displayed_cells).all())
        self.assertTrue(reversals.reversal_fraction.between(0, 1).all())
        self.assertTrue(concordance.activity_rank_correlation.between(-1, 1).all())
        self.assertTrue(concordance.balance_correlation.between(-1, 1).all())


if __name__ == '__main__':
    unittest.main()
