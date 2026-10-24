"""Audit supplied rental/count snapshots; join official measurement completeness."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from cars_common import ROOT, OUT, CACHE, DST_DATES, xy, period, dump, sha, VERSION


def rentals():
    frames, audits, sensitivities = [], [], []
    for year in (2023, 2024):
        d = pd.read_csv(ROOT / f'dataset/car-sharing-{year}.csv', sep=';', low_memory=False)
        audit = {'year': year, 'initial': len(d)}
        audit['exact_duplicates'] = int(d.duplicated().sum())
        d = d.drop_duplicates().copy()
        # Identifiers participate only in exact duplicate detection, then are discarded.
        d = d.drop(columns=['Codice utente', 'Targa'])
        d['start'] = pd.to_datetime(d['Inizio noleggio'], format='mixed', dayfirst=True, errors='coerce')
        d['end'] = pd.to_datetime(d['Fine noleggio'], format='mixed', dayfirst=True, errors='coerce')
        for new, old in [('olat', 'Latitudine Start'), ('olon', 'Longitudine Start'),
                         ('dlat', 'Latitudine Stop'), ('dlon', 'Longitudine Stop')]:
            d[new] = pd.to_numeric(d[old].astype(str).str.replace(',', '.'), errors='coerce')
        d['seconds'] = (d.end - d.start).dt.total_seconds()
        d['ox'], d['oy'] = xy(d.olat, d.olon)
        d['dx'], d['dy'] = xy(d.dlat, d.dlon)
        d['displacement'] = np.hypot(d.dx - d.ox, d.dy - d.oy)
        d['recorded_km'] = (pd.to_numeric(d['Distanza percorsa (Km)'].astype(str).str.replace(',', '.'),
                                        errors='coerce') if 'Distanza percorsa (Km)' in d else np.nan)

        def reject(name, valid):
            nonlocal d
            audit[name] = int((~valid).sum())
            d = d.loc[valid].copy()

        reject('invalid_start_year', d.start.notna() & d.start.dt.year.eq(year))
        inside = d.olat.between(44.40, 44.60) & d.dlat.between(44.40, 44.60)
        inside &= d.olon.between(11.20, 11.50) & d.dlon.between(11.20, 11.50)
        reject('outside_endpoint_box', inside)
        reject('invalid_or_short_duration', np.isfinite(d.seconds) & d.seconds.ge(30))
        # A return near the origin is a meaningful car rental, not necessarily an error.
        audit['retained'] = len(d)
        audit['near_origin_100m'] = int(d.displacement.lt(.1).sum())
        audit['over_24h'] = int(d.seconds.gt(86400).sum())
        audit['end_outside_study'] = int((~d.end.dt.year.isin([2023, 2024])).sum())
        for hours in (2, 6, 24):
            z = d.loc[d.seconds.le(hours * 3600)]
            sensitivities.append({'year': year, 'maximum_rental_hours': hours, 'retained': len(z),
                                  'night_fraction': float((period(z.start) == 'night').mean()),
                                  'median_displacement_km': float(z.displacement.median())})
        d['date'] = d.start.dt.strftime('%Y-%m-%d')
        d['regime'] = period(d.start)
        d['year'] = year
        columns = ['start', 'end', 'seconds', 'ox', 'oy', 'dx', 'dy', 'displacement',
                   'recorded_km', 'date', 'regime', 'year']
        frames.append(d[columns])
        audits.append(audit)
    trips = pd.concat(frames, ignore_index=True)
    trips.to_parquet(CACHE / 'trips.parquet', index=False)
    pd.DataFrame(audits).to_csv(OUT / 'rental-audit.csv', index=False)
    pd.DataFrame(sensitivities).to_csv(OUT / 'rental-sensitivity.csv', index=False)


def accuracy(year):
    source = CACHE / f'source/accuratezza-spire-anno-{year}.csv'
    a = pd.read_csv(source, sep=';', low_memory=False)
    # The Opendatasoft CSV export uses human-readable labels.
    a = a.rename(columns={a.columns[0]: 'date', a.columns[1]: 'loop'})
    hours = list(a.columns[2:26])
    values = a[hours].apply(lambda v: pd.to_numeric(v.astype(str).str.replace('%', '').str.replace(',', '.'), errors='coerce'))
    values.columns = [f'a{i}' for i in range(24)]
    a = pd.concat([a[['date', 'loop']], values], axis=1)
    a['date'] = a.date.str[:10]
    # Conflicting quality rows cannot be used to assert complete measurement.
    conflict = a.groupby(['date', 'loop'])[values.columns].transform('nunique').gt(1).any(axis=1)
    return a.loc[~conflict].drop_duplicates(['date', 'loop'])


def counts():
    frames, audits = [], []
    for year in (2023, 2024):
        d = pd.read_csv(ROOT / f'dataset/rilevazione-autoveicoli-tramite-spire-anno-{year}.csv',
                        sep=';', low_memory=False)
        hours = list(d.columns[2:26])
        d = d.rename(columns={'data': 'date', 'codice spira': 'loop',
                              'ID_univoco_stazione_spira': 'site', 'latitudine': 'lat',
                              'longitudine': 'lon', 'Nome via': 'street'})
        audit = {'year': year, 'initial': len(d)}

        def reject(name, valid):
            nonlocal d
            audit[name] = int((~valid).sum())
            d = d.loc[valid].copy()

        reject('not_active_loop', d.tipologia.str.lower().eq('spira') & d.stato.eq('A'))
        reject('invalid_key_coordinates_date', d.site.notna() & d.lat.between(44.4, 44.6)
               & d.lon.between(11.2, 11.5) & pd.to_datetime(d.date, errors='coerce').dt.year.eq(year))
        d['site'] = d.site.astype(int)
        values = d[hours].apply(pd.to_numeric, errors='coerce')
        d[hours] = values
        reject('invalid_hour_values', np.isfinite(values).all(axis=1) & values.ge(0).all(axis=1))
        conflict = d.groupby(['site', 'date'])[hours].transform('nunique').gt(1).any(axis=1)
        reject('conflicting_station_day', ~conflict)
        a = accuracy(year)
        d = d.merge(a, on=['date', 'loop'], how='left', validate='many_to_one')
        audit['no_quality_match_rows'] = int(d.a0.isna().sum())
        # Duplicated aliases have identical counts; retain the most complete quality record.
        acols = [f'a{i}' for i in range(24)]
        d['quality_hours'] = d[acols].eq(100).sum(axis=1)
        d = d.sort_values(['site', 'date', 'quality_hours', 'loop'], ascending=[True, True, False, True])
        audit['duplicate_station_day_aliases'] = int(d.duplicated(['site', 'date']).sum())
        d = d.drop_duplicates(['site', 'date'])
        reject('clock_change_dates', ~d.date.isin(DST_DATES))
        audit['unique_station_days'] = len(d)
        for regime, indices in [('day', list(range(6, 23))), ('night', list(range(6)) + [23])]:
            valid = d[[f'a{i}' for i in indices]].eq(100).all(axis=1)
            audit[f'{regime}_incomplete_quality'] = int((~valid).sum())
            z = d.loc[valid].copy()
            z['count'] = z[[hours[i] for i in indices]].sum(axis=1)
            z['regime'] = regime
            z['exposure'] = len(indices)
            frames.append(z[['site', 'date', 'regime', 'exposure', 'count', 'lat', 'lon', 'street']])
        audit['valid_periods'] = sum(len(x) for x in frames[-2:])
        audits.append(audit)
    d = pd.concat(frames, ignore_index=True)
    # Identity is defined in the municipal documentation; changing aliases are not locations.
    meta = d.groupby('site').agg(lat=('lat', 'median'), lon=('lon', 'median'), street=('street', 'first'))
    meta['x'], meta['y'] = xy(meta.lat, meta.lon)
    z = d.join(meta[['x', 'y']], on='site')
    rowx, rowy = xy(z.lat, z.lon)
    shifted = pd.Series(np.hypot(rowx - z.x, rowy - z.y) > .1, index=z.index)
    unstable = set(z.loc[shifted, 'site'])
    d = d.loc[~d.site.isin(unstable)].copy()
    coverage = d.assign(year=d.date.str[:4].astype(int)).groupby(['site', 'year']).agg(
        dates=('date', 'nunique'), periods=('date', 'size'))
    coverage.to_csv(OUT / 'counter-coverage.csv')
    eligible = coverage.xs(2023, level='year').query('dates >= 180').index
    meta = meta.loc[meta.index.isin(eligible) & ~meta.index.isin(unstable)].copy()
    coordinates = meta[['x', 'y']].to_numpy()
    pairs = cKDTree(coordinates).query_pairs(.1, output_type='ndarray')
    if len(pairs):
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(meta), len(meta)))
        _, labels = connected_components(graph, directed=False)
    else:
        labels = np.arange(len(meta))
    meta['cluster'] = labels
    centers = meta.groupby('cluster')[['x', 'y']].mean()
    # Fixed kilometre grid, checkerboard-style allocation, independent of traffic values.
    centers['gx'] = np.floor(centers.x).astype(int)
    centers['gy'] = np.floor(centers.y).astype(int)
    centers['block'] = centers.gx.astype(str) + ':' + centers.gy.astype(str)
    centers['fold'] = (centers.gx + 2 * centers.gy) % 5
    meta = meta.join(centers[['block', 'fold']], on='cluster')
    meta['radius'] = np.hypot(meta.x, meta.y)
    meta.to_csv(OUT / 'sites.csv')
    d = d.loc[d.site.isin(meta.index)].drop(columns=['lat', 'lon', 'street'])
    d.to_parquet(CACHE / 'counts.parquet', index=False)
    pd.DataFrame(audits).to_csv(OUT / 'counter-audit.csv', index=False)
    dump('counter-design.json', {'unstable_sites_over_100m': sorted(unstable),
         'eligibility': 'At least 180 distinct valid dates in 2023; no 2024 performance selection',
         'sites': len(meta), 'clusters': int(meta.cluster.nunique()), 'blocks': int(meta.block.nunique()),
         'periods': len(d), 'quality': 'Every source hour must have official accuracy exactly 100 percent',
         'time': 'Calendar-date day 06--23 and night 00--06 plus 23--24; DST change dates excluded'})


def main():
    rentals()
    counts()
    inputs = [ROOT / f'dataset/car-sharing-{y}.csv' for y in (2023, 2024)]
    inputs += [ROOT / f'dataset/rilevazione-autoveicoli-tramite-spire-anno-{y}.csv' for y in (2023, 2024)]
    inputs += [ROOT / 'dataset/meteo-bologna-ilmeteo-2019-2025.csv']
    inputs += sorted((CACHE / 'source').glob('*'))
    dump('input-manifest.json', {'version': VERSION,
        'inputs': {str(p.relative_to(ROOT)): sha(p) for p in inputs}})


if __name__ == '__main__':
    main()
