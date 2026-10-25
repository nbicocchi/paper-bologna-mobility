"""Cartographic projection and cached-source provenance, without network requests."""
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from common import OUT, REPO, LAT0, LON0, sha
from basemap import project
from describe import draw_counter_network, FOLD_COLOURS


class BasemapTests(unittest.TestCase):
    def test_counter_network_displays_stored_folds(self):
        networks = {
            'bike': pd.DataFrame({'site': ['Ercolani'], 'x': [0.], 'y': [0.]}),
            'car': pd.DataFrame({'x': [.1, .2, 1.1, 2.1, 3.1, 4.1], 'y': [.1] * 6,
                                 'block': ['0:0', '0:0', '1:0', '2:0', '3:0', '4:0'],
                                 'fold': [0, 0, 1, 2, 3, 4]}),
        }
        original = networks['car'].copy(deep=True)
        with patch('describe.backdrop'), patch('describe.save') as save:
            draw_counter_network(networks, None, None, (-1, 6, -1, 2))
        fig, name = save.call_args.args
        try:
            self.assertEqual(name, 'counter-network')
            ax = fig.axes[1]
            self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()],
                             [f'Fold {i}' for i in range(1, 6)])
            self.assertEqual(len(ax.patches), 5)
            for cell, colour in zip(ax.patches, FOLD_COLOURS):
                np.testing.assert_allclose(cell.get_edgecolor(), to_rgba(colour))
            np.testing.assert_allclose(ax.collections[0].get_facecolors(),
                                      [to_rgba(FOLD_COLOURS[f], .9) for f in original.fold])
            pd.testing.assert_frame_equal(networks['car'], original)
        finally:
            plt.close(fig)

    def test_projection_matches_geographical_reference(self):
        x, y = project(0, 0)
        self.assertAlmostEqual(np.rad2deg(x / 6378137.), LON0)
        self.assertAlmostEqual(np.rad2deg(2 * np.arctan(np.exp(y / 6378137.)) - np.pi / 2), LAT0)
        mx, my = project(np.array([-7, 8]), np.array([-5, 6]))
        self.assertGreater(mx[1], mx[0])
        self.assertGreater(my[1], my[0])

    def test_cached_tiles_and_attribution(self):
        source = json.loads((OUT / 'basemap-sources.json').read_text())
        self.assertIn('OpenStreetMap', source['attribution'])
        self.assertTrue(source['tiles'])
        for tile in source['tiles']:
            path = REPO / tile['file']
            self.assertEqual(sha(path), tile['sha256'])
            self.assertEqual(plt.imread(path).shape[:2], (256, 256))
            self.assertTrue(tile['url'].startswith('https://tile.openstreetmap.org/'))


if __name__ == '__main__':
    unittest.main()
