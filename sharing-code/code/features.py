"""Contemporaneous endpoint/chord predictors, rebuilt from the local raw-data cohorts."""
import numpy as np
import pandas as pd
from common import DATA, CACHE, OUT, HOLIDAYS, calendar, period, chord_distance

SHARING = ['global_trips', 'starts_500m', 'ends_500m', 'chord_100', 'chord_250']


def event_counts(positions, selected, size):
    selected = np.asarray(selected) & (positions >= 0)
    return np.bincount(positions[selected], minlength=size)


def build(mode):
    t = pd.read_parquet(CACHE / f'{mode}-trips.parquet')
    sites = pd.read_csv(OUT / f'{mode}-sites.csv')
    periods = calendar(mode)
    index = pd.MultiIndex.from_frame(periods[['date', 'regime']])
    pos = index.get_indexer(pd.MultiIndex.from_frame(t[['date', 'regime']]))
    adate, aregime = period(t.end, mode)
    arrival_pos = index.get_indexer(pd.MultiIndex.from_arrays([adate, aregime]))
    size = len(index)
    global_counts = event_counts(pos, np.ones(len(t), dtype=bool), size)
    electric = event_counts(pos, t.type.eq('electric'), size) if mode == 'bike' else None
    ox, oy, dx, dy = [t[c].to_numpy() for c in ['ox', 'oy', 'dx', 'dy']]
    frames = []
    for s in sites.itertuples():
        z = periods.copy()
        z['site'] = s.site
        z['global_trips'] = global_counts
        if electric is not None:
            z['electric_share'] = np.divide(electric, global_counts, out=np.zeros(size), where=global_counts > 0)
        z['starts_500m'] = event_counts(pos, np.hypot(ox - s.x, oy - s.y) <= .5, size)
        # Returns must be assigned to their own timestamp, not their rental's departure.
        z['ends_500m'] = event_counts(arrival_pos, np.hypot(dx - s.x, dy - s.y) <= .5, size)
        u, delta = chord_distance(ox, oy, dx, dy, s.x, s.y)
        inward = (dx - ox) * (-s.x) + (dy - oy) * (-s.y) > 0
        for radius in [100, 250]:
            cross = (u > 0) & (u < 1) & (delta <= radius / 1000)
            if mode == 'bike':
                for direction, selected in [('inward', inward), ('outward', ~inward)]:
                    z[f'chord_{radius}_{direction}'] = event_counts(pos, cross & selected, size)
            else:
                z[f'chord_{radius}'] = event_counts(pos, cross, size)
        frames.append(z.drop(columns='exposure'))
    features = pd.concat(frames, ignore_index=True)
    d = pd.read_parquet(CACHE / f'{mode}-counts.parquet')
    d = d.merge(features, on=['site', 'date', 'regime'], validate='one_to_one')
    d = d.merge(sites.drop(columns=['street', 'lat', 'lon'], errors='ignore'), on='site', validate='many_to_one')
    weather = pd.read_csv(DATA / 'meteo-bologna-ilmeteo-2019-2025.csv', sep=';')
    weather = weather[['date', 'temperature_mean_c', 'precipitation_mm']].drop_duplicates('date')
    d = d.merge(weather, on='date', how='left', validate='many_to_one')
    dates = pd.to_datetime(d.date)
    d['year'], d['dow'] = dates.dt.year, dates.dt.weekday
    d['sin_year'] = np.sin(2 * np.pi * dates.dt.dayofyear / 365.25)
    d['cos_year'] = np.cos(2 * np.pi * dates.dt.dayofyear / 365.25)
    d['night'] = d.regime.eq('night').astype(int)
    d['holiday'] = d.date.isin(HOLIDAYS).astype(int)
    if mode == 'bike':
        coverage = pd.read_csv(OUT / 'bike-sharing-coverage.csv')
        d = d.merge(coverage[['date', 'regime', 'usable']], on=['date', 'regime'], validate='many_to_one')
        d = d.loc[d.usable].drop(columns='usable')
        directions = []
        for direction in ['inward', 'outward']:
            z = d.copy()
            z['direction'] = int(direction == 'outward')
            z['count'] = z[direction]
            for radius in [100, 250]:
                z[f'chord_{radius}'] = z[f'chord_{radius}_{direction}']
            directions.append(z)
        d = pd.concat(directions, ignore_index=True)
        d = d.drop(columns=['inward', 'outward'] + [f'chord_{r}_{v}' for r in [100, 250] for v in ['inward', 'outward']])
    else:
        d['direction'] = 0
    for column in SHARING:
        d[column] /= d.exposure
    d['rate'] = d['count'] / d.exposure
    d.to_parquet(CACHE / f'{mode}-model-input.parquet', index=False)
    print(mode, 'model rows', len(d), 'sites', d.site.nunique(), flush=True)


if __name__ == '__main__':
    for mode in ['bike', 'car']:
        build(mode)
