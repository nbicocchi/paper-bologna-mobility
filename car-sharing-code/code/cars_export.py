"""Export computed car-study tables and macros; no hand-entered empirical values."""
import importlib.metadata
import platform
import numpy as np
import pandas as pd
from cars_common import ROOT, OUT, TEXT, dump, sha

TABLES = TEXT / 'tables'


def table(name, columns, rows, alignment):
    text = '\\begin{tabular}{' + alignment + '}\n\\toprule\n'
    text += ' & '.join(columns) + ' \\\\\n\\midrule\n'
    text += ''.join(' & '.join(map(str, row)) + ' \\\\\n' for row in rows)
    text += '\\bottomrule\n\\end{tabular}\n'
    (TABLES / f'cars-{name}.tex').write_text(text)


def main():
    audit = pd.read_csv(OUT / 'rental-audit.csv').set_index('year')
    rows = []
    labels = [('initial', 'Initial records'), ('exact_duplicates', 'Exact duplicates'),
              ('invalid_start_year', 'Invalid departure/year'), ('outside_endpoint_box', 'Endpoints outside study box'),
              ('invalid_or_short_duration', 'Invalid duration or below 30 s'), ('retained', 'Retained rentals')]
    for key, label in labels:
        rows.append([label] + [f'{int(audit.loc[y, key]):,}' for y in (2023, 2024)])
    table('audit', ['', '2023', '2024'], rows, 'lrr')
    counter = pd.read_csv(OUT / 'counter-audit.csv').set_index('year')
    labels = [('initial', 'Initial device-day rows'), ('not_active_loop', 'Not active loop type'),
              ('invalid_key_coordinates_date', 'Invalid ID/position/date'), ('invalid_hour_values', 'Invalid hourly counts'),
              ('conflicting_station_day', 'Conflicting device-day rows'),
              ('duplicate_station_day_aliases', 'Repeated device-day aliases'),
              ('clock_change_dates', 'Clock-change device-days'), ('unique_station_days', 'Candidate device-days'),
              ('day_incomplete_quality', 'Incomplete day periods'), ('night_incomplete_quality', 'Incomplete night periods'),
              ('valid_periods', 'Complete periods before site eligibility')]
    rows = [[label] + [f'{int(counter.loc[y, key]):,}' for y in (2023, 2024)] for key, label in labels]
    table('counter-audit', ['', '2023', '2024'], rows, 'lrr')
    summary = pd.read_csv(OUT / 'trip-summary.csv')
    rows = [[int(r.year), r.regime.title(), f'{r.n:,}', f'{r.median_displacement_km:.2f}',
             f'{r.median_duration_min:.1f}', f'{r.near_origin_pct:.1f}', f'{r.over_24h_pct:.1f}'] for r in summary.itertuples()]
    table('trips', ['Year', 'Period', '$n$', 'km$^a$', 'min$^b$', 'Near$^c$', 'Long$^d$'], rows, 'rlrrrrr')
    od = pd.read_csv(OUT / 'od-comparison.csv')
    rows = []
    for size, g in od.groupby('cell_km'):
        g = g.set_index('year')
        rows.append([f'{size:g}', f'{g.loc[2023, "js_distance"]:.3f}', f'{g.loc[2024, "js_distance"]:.3f}'])
    table('od', ['Cell (km)', '2023', '2024'], rows, 'rrr')
    results = pd.read_csv(OUT / 'prediction-summary.csv').set_index(['year', 'model'])
    for year in (2023, 2024):
        rows = []
        for family, label in [('boosting', 'Boosting'), ('poisson', 'Poisson')]:
            for features, feature_label in [('base', 'No sharing'), ('global', 'Citywide sharing'),
                                             ('local', 'Local endpoints'), ('geometry', 'OD geometry')]:
                r = results.loc[(year, f'{family}_{features}')]
                rows.append([label, feature_label, f'{r.macro_wape * 100:,.1f}',
                             f'{r.pooled_wape * 100:.1f}', f'{r.cluster_macro_wape * 100:.1f}'])
        r = results.loc[(year, 'scaling')]
        rows.append(['Scaling', '250 m chord', f'{r.macro_wape * 100:,.1f}',
                     f'{r.pooled_wape * 100:.1f}', f'{r.cluster_macro_wape * 100:.1f}'])
        table(f'prediction-{year}', ['Model', 'Features', 'Macro', 'Pooled', 'Cluster'], rows, 'llrrr')
    rows = []
    for year in (2023, 2024):
        for family in ('boosting', 'poisson'):
            for feature in ('base', 'geometry'):
                r = results.loc[(year, f'{family}_{feature}')]
                rows.append([year, family.title(), 'No sharing' if feature == 'base' else 'Geometry',
                             f'{r.median_wape * 100:.1f}', f'{int(r.sites_positive_total)}/{int(r.sites)}'])
    table('robustness', ['Year', 'Model', 'Inputs', 'Median', '$S_+/S$'], rows, 'rllrr')
    bootstrap = pd.read_csv(OUT / 'paired-bootstrap.csv')
    rows = []
    for r in bootstrap.itertuples():
        if r.model == 'scaling' or r.model.endswith('_short'): continue
        family, feature = r.model.split('_', 1)
        rows.append([int(r.year), family.title(), {'global': 'Citywide', 'local': 'Endpoints', 'geometry': 'Geometry'}[feature],
                     f'{r.delta_macro_wape * 100:+.1f}',
                     f'[{r.lo * 100:+.1f}, {r.hi * 100:+.1f}]'])
    table('contrasts', ['Year', 'Model', 'Added input', '$\\Delta$', '95\\% interval'], rows, 'rllrr')
    rows = []
    for year in (2023, 2024):
        for family in ('boosting', 'poisson'):
            r = results.loc[(year, f'{family}_geometry_short')]
            rows.append([year, family.title(), f'{r.macro_wape * 100:.1f}', f'{r.pooled_wape * 100:.1f}'])
    table('sensitivity', ['Year', 'Model', 'Macro', 'Pooled'], rows, 'rlrr')
    sites = pd.read_csv(OUT / 'sites.csv')
    coverage = pd.read_csv(OUT / 'counter-coverage.csv')
    rows = []
    for year in (2023, 2024):
        z = coverage.loc[coverage.year.eq(year) & coverage.site.isin(sites.site)]
        rows.append([year, z.site.nunique(), f'{z.periods.sum():,}',
                     f'{z.dates.median():.0f}', f'{z.dates.min():.0f}--{z.dates.max():.0f}'])
    table('coverage', ['Year', 'Devices', 'Periods', 'Median dates', 'Range'], rows, 'rrrrr')
    values = {'RawCars': f'{audit.initial.sum():,}', 'RetainedCars': f'{audit.retained.sum():,}',
              'CarDevices': len(sites), 'CarClusters': sites.cluster.nunique(), 'CarBlocks': sites.block.nunique()}
    for year, word in [(2023, 'Spatial'), (2024, 'Forward')]:
        for family, fword in [('boosting', 'Boost'), ('poisson', 'Poisson')]:
            for feature, vword in [('base', 'Base'), ('global', 'Global'), ('local', 'Local'), ('geometry', 'Geometry')]:
                r = results.loc[(year, f'{family}_{feature}')]
                for measure, mword in [('macro_wape', 'Macro'), ('pooled_wape', 'Pooled')]:
                    values[word + fword + vword + mword] = f'{r[measure] * 100:.1f}'
    (TABLES / 'cars-numbers.tex').write_text('% Generated by cars_export.py; do not edit.\n' +
        ''.join('\\newcommand{\\' + key + '}{' + str(value) + '}\n' for key, value in values.items()))
    packages = ['numpy', 'pandas', 'scipy', 'matplotlib', 'scikit-learn', 'pyarrow', 'requests']
    dump('reproducibility.json', {'python': platform.python_version(), 'platform': platform.platform(),
        'packages': {p: importlib.metadata.version(p) for p in packages},
        'code_sha256': {p.name: sha(p) for p in sorted((ROOT / 'code').glob('*.py'))},
        'generated_tables': {p.name: sha(p) for p in sorted(TABLES.glob('cars-*.tex'))}})


if __name__ == '__main__':
    main()
