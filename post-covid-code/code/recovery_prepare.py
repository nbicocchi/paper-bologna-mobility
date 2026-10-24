"""Audit a longitudinal detector panel. Missing recording is never zero traffic."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'code/dati_esterni/recovery'
OUT = ROOT / 'results/recovery'
YEARS = list(range(2019, 2026))
HOURS = [f'h{h:02}' for h in range(24)]


def clock_dates(year):
    dates = pd.date_range(f'{year}-01-01', f'{year}-12-31')
    return dates[(dates.dayofweek == 6) & dates.month.isin([3, 10])].to_series().groupby(lambda d: d.month).max().dt.strftime('%Y-%m-%d').tolist()


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    audits, frames, positions = [], [], []
    for year in YEARS:
        print('Preparing', year, flush=True)
        d = pd.read_csv(ROOT / f'dataset/rilevazione-autoveicoli-tramite-spire-anno-{year}.csv', sep=';', low_memory=False)
        d = d.rename(columns=dict(zip(d.columns[2:26], HOURS)) | {
            'data': 'date', 'codice spira': 'loop', 'ID_univoco_stazione_spira': 'site',
            'latitudine': 'lat', 'longitudine': 'lon', 'Nome via': 'street'})
        audit = {'year': year, 'raw_rows': len(d), 'first_date': d.date.min(), 'last_date': d.date.max()}

        def reject(name, valid):
            nonlocal d
            audit[name] = int((~valid).sum())
            d = d.loc[valid].copy()

        reject('inactive_or_not_loop', d.tipologia.str.lower().eq('spira') & d.stato.eq('A'))
        reject('invalid_identity_position_date', d.site.notna() & d.lat.between(44.4, 44.6)
               & d.lon.between(11.2, 11.5) & pd.to_datetime(d.date, errors='coerce').dt.year.eq(year))
        d['site'] = d.site.astype(int)
        d[HOURS] = d[HOURS].apply(pd.to_numeric, errors='coerce')
        reject('invalid_counts', np.isfinite(d[HOURS]).all(axis=1) & d[HOURS].ge(0).all(axis=1))
        conflicts = d.groupby(['site', 'date'])[HOURS].transform('nunique').gt(1).any(axis=1)
        reject('conflicting_alias_rows', ~conflicts)
        a = pd.read_csv(CACHE / f'source/accuratezza-spire-anno-{year}.csv', sep=';', low_memory=False)
        acols = [f'a{i:02}' for i in range(24)]
        a = a.rename(columns=dict(zip(a.columns[:26], ['date', 'loop'] + acols)))
        a['date'] = a.date.str[:10]
        a[acols] = a[acols].apply(lambda s: pd.to_numeric(s.astype(str).str.replace('%', '').str.replace(',', '.'), errors='coerce'))
        bad = a.groupby(['date', 'loop'])[acols].transform('nunique').gt(1).any(axis=1)
        audit['conflicting_quality_rows'] = int(bad.sum())
        a = a.loc[~bad, ['date', 'loop'] + acols].drop_duplicates(['date', 'loop'])
        d = d.merge(a, on=['date', 'loop'], how='left', validate='many_to_one')
        audit['unmatched_quality_rows'] = int(d[acols].isna().all(axis=1).sum())
        d['complete_hours'] = d[acols].eq(100).sum(axis=1)
        d = d.sort_values(['site', 'date', 'complete_hours', 'loop'], ascending=[True, True, False, True])
        audit['identical_alias_rows'] = int(d.duplicated(['site', 'date']).sum())
        d = d.drop_duplicates(['site', 'date'])
        reject('clock_change_days', ~d.date.isin(clock_dates(year)))
        positions.append(d[['site', 'lat', 'lon', 'street']].drop_duplicates())
        audit['candidate_days'] = len(d)
        reject('incomplete_days', d.complete_hours.eq(24))
        d['total'] = d[HOURS].sum(axis=1)
        audit['complete_days'] = len(d)
        audit['zero_complete_days'] = int(d.total.eq(0).sum())
        audit['sites_complete'] = int(d.site.nunique())
        d['year'] = year
        frames.append(d[['site', 'date', 'year', 'total'] + HOURS])
        audits.append(audit)
    p = pd.concat(positions).drop_duplicates()
    meta = p.groupby('site').agg(lat=('lat', 'median'), lon=('lon', 'median'), street=('street', 'first'))
    p = p.join(meta[['lat', 'lon']], on='site', rsuffix='_median')
    distance = 6371 * np.hypot(np.deg2rad(p.lat - p.lat_median), np.deg2rad(p.lon - p.lon_median) * np.cos(np.deg2rad(44.4949)))
    unstable = sorted(p.loc[distance.gt(.1), 'site'].unique().tolist())
    meta['x'] = 6371 * np.deg2rad(meta.lon - 11.3426) * np.cos(np.deg2rad(44.4949))
    meta['y'] = 6371 * np.deg2rad(meta.lat - 44.4949)
    meta['block'] = np.floor(meta.x).astype(int).astype(str) + ':' + np.floor(meta.y).astype(int).astype(str)
    meta['radius_km'] = np.hypot(meta.x, meta.y)
    d = pd.concat(frames, ignore_index=True)
    d['date'] = pd.to_datetime(d.date)
    d['month'] = d.date.dt.month
    d['dow'] = d.date.dt.dayofweek
    d['weekend'] = d.dow.ge(5).astype(int)
    # Gregorian Easter algorithm is supplied by scipy; national and local public holidays.
    from dateutil.easter import easter
    holidays = set()
    for year in YEARS:
        holidays.update(pd.to_datetime([f'{year}-{s}' for s in ['01-01', '01-06', '04-25', '05-01', '06-02', '08-15', '10-04', '11-01', '12-08', '12-25', '12-26']]))
        holidays.update([pd.Timestamp(easter(year)), pd.Timestamp(easter(year)) + pd.Timedelta(days=1)])
    d['holiday'] = d.date.isin(holidays)
    d['unstable'] = d.site.isin(unstable)
    d.to_parquet(CACHE / 'daily.parquet', index=False)
    meta.to_csv(OUT / 'sites.csv')
    pd.DataFrame(audits).to_csv(OUT / 'audit.csv', index=False)
    coverage = d.groupby(['site', 'year']).agg(days=('date', 'size'), zero_days=('total', lambda s: s.eq(0).sum()), total=('total', 'sum'))
    coverage.to_csv(OUT / 'coverage.csv')
    (OUT / 'design.json').write_text(json.dumps({'years': YEARS, 'unstable_sites': unstable,
        'position_tolerance_m': 100, 'quality': 'All 24 hours exactly 100%; no upscaling',
        'primary_minimum_days_per_year': 120, 'minimum_month_type_days': {'weekday': 2, 'weekend': 1},
        'holidays': 'Italian national holidays, Easter and Monday, Bologna patronal day excluded',
        'standardization': 'Equal months; weekdays 5/7, weekends 2/7; fixed all-year panel',
        'bootstrap': '2000 resamples of fixed 1 km spatial blocks; seed 20261025'}, indent=2))


if __name__ == '__main__':
    main()
