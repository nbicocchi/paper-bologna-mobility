"""Rebuild de-identified cohorts and counts directly from dataset/, never other papers."""
import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from common import (DATA, CACHE, OUT, EXCLUDED_BIKE, DST_DATES, directories,
                    xy, period, calendar, sha, dump)


def rentals(mode):
    frames, audits, coverage = [], [], []
    for year in [2023, 2024]:
        path = DATA / f'{mode}-sharing-{year}.csv'
        raw = pd.read_csv(path, sep=';', low_memory=False)
        audit = dict(mode=mode, year=year, raw=len(raw), exact_duplicates=int(raw.duplicated().sum()))
        raw = raw.drop_duplicates()
        # IDs are used for exact deduplication only, and never retained in intermediates.
        t = pd.DataFrame(index=raw.index)
        for new, old in [('start', 'Inizio noleggio'), ('end', 'Fine noleggio')]:
            t[new] = pd.to_datetime(raw[old], format='mixed', dayfirst=True, errors='coerce')
        for new, old in [('olat', 'Latitudine Start'), ('olon', 'Longitudine Start'),
                         ('dlat', 'Latitudine Stop'), ('dlon', 'Longitudine Stop')]:
            t[new] = pd.to_numeric(raw[old].astype(str).str.replace(',', '.'), errors='coerce')
        t['seconds'] = (t.end - t.start).dt.total_seconds()
        t['ox'], t['oy'] = xy(t.olat, t.olon)
        t['dx'], t['dy'] = xy(t.dlat, t.dlon)
        t['displacement'] = np.hypot(t.dx - t.ox, t.dy - t.oy)
        valid = (t.start.notna() & t.start.dt.year.eq(year) & t.olat.between(44.4, 44.6)
                 & t.dlat.between(44.4, 44.6) & t.olon.between(11.2, 11.5) & t.dlon.between(11.2, 11.5))
        duration = np.isfinite(t.seconds) & t.seconds.ge(30)
        if mode == 'bike':
            t['type'] = raw['Tipo bici'].str.lower().map({'e-bike': 'electric', 'ebike': 'electric',
                'bike': 'mechanical', 'muscolare': 'mechanical', 'muscolari': 'mechanical', 'normal bike': 'mechanical'})
            valid &= t.type.notna() & t.displacement.ge(.1)
            date, regime = period(t.start, mode)
            c = pd.DataFrame({'date': date, 'regime': regime, 'candidates': valid.astype(int),
                              'retained': (valid & duration).astype(int)}).groupby(['date', 'regime']).sum()
            coverage.append(c)
        audit['endpoint_candidates'] = int(valid.sum())
        audit['invalid_duration_among_candidates'] = int((valid & ~duration).sum())
        t = t.loc[valid & duration].drop(columns=['olat', 'olon', 'dlat', 'dlon']).copy()
        t['date'], t['regime'] = period(t.start, mode)
        t['duration_min'] = t.seconds / 60
        if mode == 'bike':
            t['duration_min'] = t.duration_min.where(t.duration_min.between(.5, 180)
                                                    & (t.displacement / t.duration_min * 60 <= 45))
        audit['retained'] = len(t)
        audit['sha256'] = sha(path)
        frames.append(t)
        audits.append(audit)
        print(mode, year, 'retained rentals', len(t), flush=True)
    pd.concat(frames, ignore_index=True).to_parquet(CACHE / f'{mode}-trips.parquet', index=False)
    pd.DataFrame(audits).to_csv(OUT / f'{mode}-rental-audit.csv', index=False)
    if coverage:
        c = pd.concat(coverage).groupby(['date', 'regime']).sum().reset_index()
        c['usable'] = ~((c.candidates >= 100) & (c.retained < .01 * c.candidates))
        c.to_csv(OUT / 'bike-sharing-coverage.csv', index=False)


def bike_counts():
    d = pd.read_csv(DATA / 'colonnine-conta-bici.csv', sep=';', encoding='utf-8-sig')
    d['time'] = pd.to_datetime(d['Data'], utc=True)
    d = d.loc[d.time.dt.tz_convert('Europe/Rome').dt.year.isin([2023, 2024])].copy()
    d = d.rename(columns={'Dispositivo conta-bici': 'device', 'Direzione centro': 'inward',
                          'Direzione periferia': 'outward', 'Totale': 'total'})
    d['site'] = d.device.str.replace(r'_(I|II)$', '', regex=True)
    excluded = d.site.isin(EXCLUDED_BIKE)
    audit = dict(input_rows=len(d), excluded_sites=sorted(EXCLUDED_BIKE), excluded_rows=int(excluded.sum()))
    d = d.loc[~excluded].drop_duplicates().copy()
    conflict = d.duplicated(['device', 'time'], keep=False)
    audit['conflicting_rows'] = int(conflict.sum())
    d = d.loc[~conflict].sort_values(['device', 'time']).copy()
    loc = d['Geo Point'].str.split(',', expand=True).astype(float)
    d['x'], d['y'] = xy(loc[0], loc[1])
    zero = d[['total', 'inward', 'outward']].eq(0).all(axis=1)
    breaks = d.device.ne(d.device.shift()) | ~zero | ~zero.shift(fill_value=False) | d.time.diff().ne(pd.Timedelta(hours=1))
    suspect = zero & zero.groupby(breaks.cumsum()).transform('sum').ge(48)
    d['valid'] = (~suspect & d[['inward', 'outward', 'total']].notna().all(axis=1)
                  & d[['inward', 'outward', 'total']].ge(0).all(axis=1) & d.total.eq(d.inward + d.outward))
    audit['suspect_zero_hours'] = int(suspect.sum())
    training_rows = d.loc[d.time.dt.tz_convert('Europe/Rome').dt.year.eq(2023)]
    # Freeze the expected instrument inventory and coordinates before the test year.
    known_devices = set(training_rows.device)
    devices = training_rows.groupby('site').device.nunique()
    d = d.loc[d.device.isin(known_devices)].copy()
    h = d.groupby(['site', 'time']).agg(n=('device', 'nunique'), valid=('valid', 'all'),
                                       inward=('inward', 'sum'), outward=('outward', 'sum')).reset_index()
    h['valid'] &= h.n.eq(h.site.map(devices))
    h['date'], h['regime'] = period(h.time.dt.tz_convert('Europe/Rome'), 'bike')
    a = h.loc[h.valid].groupby(['site', 'date', 'regime']).agg(hours=('time', 'size'),
            inward=('inward', 'sum'), outward=('outward', 'sum')).reset_index()
    a = a.merge(calendar('bike'), on=['date', 'regime'], validate='many_to_one')
    a = a.loc[a.hours.eq(a.exposure) & a.date.between('2023-01-01', '2024-12-30')].copy()
    coverage = a.assign(year=a.date.str[:4].astype(int)).groupby(['site', 'year']).date.nunique().rename('dates')
    coverage.to_csv(OUT / 'bike-counter-coverage.csv')
    eligible = coverage.xs(2023, level='year').loc[lambda s: s >= 60].index
    sites = training_rows.groupby('site')[['x', 'y']].mean().loc[eligible].copy()
    sites['radius'] = np.hypot(sites.x, sites.y)
    sites['block'] = sites.index
    sites['fold'] = np.arange(len(sites))
    sites.to_csv(OUT / 'bike-sites.csv')
    a.loc[a.site.isin(eligible)].to_parquet(CACHE / 'bike-counts.parquet', index=False)
    audit['eligible_sites'] = eligible.tolist()
    dump(OUT / 'bike-counter-audit.json', audit)


def official_accuracy(year):
    name = f'accuratezza-spire-anno-{year}'
    path = CACHE / 'sources' / f'{name}.csv'
    url = f'https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/{name}/exports/csv'
    if not path.exists():
        print('Downloading official recording completeness:', year, flush=True)
        response = requests.get(url, timeout=600)
        response.raise_for_status()
        temporary = path.with_suffix('.part')
        temporary.write_bytes(response.content)
        temporary.replace(path)
    a = pd.read_csv(path, sep=';', low_memory=False)
    a = a.rename(columns={a.columns[0]: 'date', a.columns[1]: 'loop'})
    values = a[list(a.columns[2:26])].apply(lambda s:
        pd.to_numeric(s.astype(str).str.replace('%', '').str.replace(',', '.'), errors='coerce'))
    values.columns = [f'a{i}' for i in range(24)]
    a = pd.concat([a[['date', 'loop']], values], axis=1)
    a['date'] = a.date.str[:10]
    conflict = a.groupby(['date', 'loop'])[values.columns].transform('nunique').gt(1).any(axis=1)
    dump(CACHE / 'sources' / f'{name}.json', dict(url=url, sha256=sha(path)))
    return a.loc[~conflict].drop_duplicates(['date', 'loop'])


def car_counts():
    frames, audits = [], []
    for year in [2023, 2024]:
        d = pd.read_csv(DATA / f'rilevazione-autoveicoli-tramite-spire-anno-{year}.csv', sep=';', low_memory=False)
        hours = list(d.columns[2:26])
        d = d.rename(columns={'data': 'date', 'codice spira': 'loop', 'ID_univoco_stazione_spira': 'site',
                              'latitudine': 'lat', 'longitudine': 'lon', 'Nome via': 'street'})
        audit = dict(year=year, raw=len(d))
        valid = (d.tipologia.str.lower().eq('spira') & d.stato.eq('A') & d.site.notna()
                 & d.lat.between(44.4, 44.6) & d.lon.between(11.2, 11.5)
                 & pd.to_datetime(d.date, errors='coerce').dt.year.eq(year))
        d = d.loc[valid].copy()
        d['site'] = d.site.astype(int)
        d[hours] = d[hours].apply(pd.to_numeric, errors='coerce')
        d = d.loc[np.isfinite(d[hours]).all(axis=1) & d[hours].ge(0).all(axis=1)].copy()
        conflict = d.groupby(['site', 'date'])[hours].transform('nunique').gt(1).any(axis=1)
        audit['conflicting_rows'] = int(conflict.sum())
        d = d.loc[~conflict].merge(official_accuracy(year), on=['date', 'loop'], how='left', validate='many_to_one')
        d['quality_hours'] = d[[f'a{i}' for i in range(24)]].eq(100).sum(axis=1)
        d = d.sort_values(['site', 'date', 'quality_hours', 'loop'], ascending=[True, True, False, True])
        d = d.drop_duplicates(['site', 'date'])
        d = d.loc[~d.date.isin(DST_DATES)]
        for regime, indices in [('day', list(range(6, 23))), ('night', list(range(6)) + [23])]:
            valid = d[[f'a{i}' for i in indices]].eq(100).all(axis=1)
            z = d.loc[valid].copy()
            z['count'] = z[[hours[i] for i in indices]].sum(axis=1)
            z['regime'], z['exposure'] = regime, len(indices)
            frames.append(z[['site', 'date', 'regime', 'exposure', 'count', 'lat', 'lon', 'street']])
        audit['complete_periods'] = sum(len(f) for f in frames[-2:])
        audits.append(audit)
    d = pd.concat(frames, ignore_index=True)
    # Locations and eligibility use 2023 only; 2024 cannot select the training cohort.
    train = d.loc[d.date.str.startswith('2023')]
    meta = train.groupby('site').agg(lat=('lat', 'median'), lon=('lon', 'median'), street=('street', 'first'))
    meta['x'], meta['y'] = xy(meta.lat, meta.lon)
    rowx, rowy = xy(train.lat, train.lon)
    shifted = np.hypot(rowx - train.site.map(meta.x), rowy - train.site.map(meta.y)) > .1
    unstable = set(train.loc[shifted, 'site'])
    coverage = train.groupby('site').date.nunique()
    meta = meta.loc[coverage.loc[coverage >= 180].index.difference(list(unstable))].copy()
    pairs = cKDTree(meta[['x', 'y']]).query_pairs(.1, output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(meta), len(meta)))
    _, labels = connected_components(graph, directed=False)
    meta['cluster'] = labels
    centres = meta.groupby('cluster')[['x', 'y']].mean()
    gx, gy = np.floor(centres.x).astype(int), np.floor(centres.y).astype(int)
    centres['block'] = gx.astype(str) + ':' + gy.astype(str)
    centres['fold'] = (gx + 2 * gy) % 5
    meta = meta.join(centres[['block', 'fold']], on='cluster')
    meta['radius'] = np.hypot(meta.x, meta.y)
    meta.to_csv(OUT / 'car-sites.csv')
    d = d.loc[d.site.isin(meta.index)].copy()
    rowx, rowy = xy(d.lat, d.lon)
    moved = np.hypot(rowx - d.site.map(meta.x), rowy - d.site.map(meta.y)) > .1
    d = d.loc[~moved].drop(columns=['lat', 'lon', 'street'])
    d.to_parquet(CACHE / 'car-counts.parquet', index=False)
    pd.DataFrame(audits).to_csv(OUT / 'car-counter-audit.csv', index=False)
    dump(OUT / 'car-counter-design.json', dict(sites=len(meta), clusters=int(meta.cluster.nunique()),
        blocks=int(meta.block.nunique()), unstable_2023=sorted(unstable), moved_periods_excluded=int(moved.sum()),
        eligibility='2023 only: >=180 usable dates and coordinates stable within 100 m'))


def main():
    directories()
    for mode in ['bike', 'car']:
        rentals(mode)
    bike_counts()
    car_counts()


if __name__ == '__main__':
    main()
