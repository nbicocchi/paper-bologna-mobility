"""Read-only source registry and canonical names; no upstream imports."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
OUT = ROOT / 'results'
INPUT = ROOT / 'inputs'
TEXT = REPO / 'sharing-text'
SEED = 20261025
REPLICATES = 2000
FEATURES = ['Context', 'Citywide', 'Endpoints', 'Geometry']
MAPPING = {
    'bike': {
        'calendar_weather': ('Boosting', 'Context'),
        'global_sharing': ('Boosting', 'Citywide'),
        'local_endpoints': ('Boosting', 'Endpoints'),
        'chord_proxy': ('Boosting', 'Geometry'),
        'poisson_base': ('Poisson', 'Context'),
        'poisson_global': ('Poisson', 'Citywide'),
        'poisson_local': ('Poisson', 'Endpoints'),
        'poisson_sharing': ('Poisson', 'Geometry'),
        'uniform_chord': ('Scaling', 'Geometry'),
    },
    'car': {
        **{f'{prefix}_{suffix}': (family, feature)
           for prefix, family in [('boosting', 'Boosting'), ('poisson', 'Poisson')]
           for suffix, feature in [('base', 'Context'), ('global', 'Citywide'),
                                   ('local', 'Endpoints'), ('geometry', 'Geometry')]},
        'scaling': ('Scaling', 'Geometry'),
    },
}
SOURCES = {
    'bike-counter-locations.csv': 'bike-sharing-code/results/audit-counters.csv',
    'car-counter-locations.csv': 'car-sharing-code/results/cars/sites.csv',
    'bike-summary.csv': 'bike-sharing-code/results/prediction-summary.csv',
    'bike-sites.csv': 'bike-sharing-code/results/prediction-site-diagnostics.csv',
    'bike-audit-2023.json': 'bike-sharing-code/results/audit-sharing-2023.json',
    'bike-audit-2024.json': 'bike-sharing-code/results/audit-sharing-2024.json',
    'bike-design.json': 'bike-sharing-code/results/prediction-design.json',
    'bike-selection.json': 'bike-sharing-code/results/counter-selection.json',
    'bike-coverage.csv': 'bike-sharing-code/results/site-coverage.csv',
    'car-summary.csv': 'car-sharing-code/results/cars/prediction-summary.csv',
    'car-sites.csv': 'car-sharing-code/results/cars/site-scores.csv',
    'car-audit.csv': 'car-sharing-code/results/cars/rental-audit.csv',
    'car-design.json': 'car-sharing-code/results/cars/prediction-design.json',
    'car-counter-design.json': 'car-sharing-code/results/cars/counter-design.json',
    'car-reliability.json': 'car-sharing-code/results/cars/counter-reliability.json',
    'bike-map-grid.csv': 'bike-sharing-code/results/sharing-map-grid.csv',
    'bike-map-design.json': 'bike-sharing-code/results/sharing-map-design.json',
    'car-map-grid.csv': 'car-sharing-code/results/cars/sharing-map-grid.csv',
    'basemap.json': 'bike-sharing-code/results/city-basemap.json',
    **{f'osm-12-{x}-{y}.png':
       f'bike-sharing-code/code/dati_esterni/basemap/osm-12-{x}-{y}.png'
       for x in range(2176, 2179) for y in range(1480, 1483)},
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def year_from_split(split):
    return {'spatial_2023': 2023, 'spatiotemporal_2024': 2024}[split]
