"""Car-rental descriptions, endpoint maps and contemporaneous predictor features."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cars_common import OUT, CACHE, TEXT, DST_DATES, SEED, period, chord_distance, dump

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, name):
    fig.savefig(TEXT / f'figures/cars-{name}.pdf', bbox_inches='tight')
    fig.savefig(TEXT / f'figures/cars-{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def descriptions(trips):
    rows = []
    for (year, regime), g in trips.groupby(['year', 'regime']):
        distance = g.recorded_km.where(g.recorded_km.between(0, 1000))
        rows.append({'year': year, 'regime': regime, 'n': len(g),
                     'median_displacement_km': g.displacement.median(),
                     'median_duration_min': g.seconds.median() / 60,
                     'p95_duration_min': g.seconds.quantile(.95) / 60,
                     'near_origin_pct': 100 * g.displacement.lt(.1).mean(),
                     'over_24h_pct': 100 * g.seconds.gt(86400).mean(),
                     'recorded_distance_n': distance.notna().sum(),
                     'recorded_distance_median': distance.median() if distance.notna().any() else None})
    pd.DataFrame(rows).to_csv(OUT / 'trip-summary.csv', index=False)
    monthly = trips.groupby([trips.start.dt.to_period('M'), 'year']).size().rename('n').reset_index()
    monthly.to_csv(OUT / 'monthly.csv', index=False)
    fig, ax = plt.subplots(figsize=(7.1, 2.3))
    for year, g in monthly.groupby('year'):
        ax.plot(g.start.dt.month, g.n / 1000, marker='o', label=str(year))
    ax.set(xlabel='Departure month', ylabel='Retained rentals (thousands)', xticks=range(1, 13))
    ax.legend(frameon=False)
    fig.tight_layout(); save(fig, 'monthly')
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5), sharey=True)
    hourly_rows = []
    for ax, weekend in zip(axes, (False, True)):
        for year in (2023, 2024):
            g = trips.loc[trips.year.eq(year) & trips.start.dt.weekday.ge(5).eq(weekend)]
            counts = g.start.dt.hour.value_counts().reindex(range(24), fill_value=0).sort_index()
            ax.plot(counts.index, counts / counts.sum() * 100, label=str(year))
            hourly_rows.extend({'year': year, 'weekend': weekend, 'hour': h, 'n': int(n)} for h, n in counts.items())
        ax.axvspan(-.5, 5.5, color='grey', alpha=.12)
        ax.axvspan(22.5, 23.5, color='grey', alpha=.12)
        ax.set(title='Weekend' if weekend else 'Weekday', xlabel='Departure hour', xlim=(-.5, 23.5))
    axes[0].set_ylabel('Within-group share (%)'); axes[1].legend(frameon=False)
    fig.tight_layout(); save(fig, 'hourly')
    pd.DataFrame(hourly_rows).to_csv(OUT / 'hourly.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5))
    for year, g in trips.groupby('year'):
        for ax, values in zip(axes, (g.displacement, g.seconds / 60)):
            v = np.sort(values.to_numpy())
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), label=str(year))
    axes[0].set(xlabel='Endpoint displacement (km)', ylabel='Cumulative fraction', xlim=(0, 20))
    axes[1].set(xlabel='Recorded rental duration (min)', xscale='log', xlim=(.5, 10000))
    axes[1].legend(frameon=False); fig.tight_layout(); save(fig, 'distributions')
    rows = []
    rng = np.random.default_rng(SEED)
    for size in (.5, 1., 2.):
        z = trips.copy()
        z['od'] = list(zip(np.floor(z.ox / size).astype(int), np.floor(z.oy / size).astype(int),
                           np.floor(z.dx / size).astype(int), np.floor(z.dy / size).astype(int)))
        for year, g in z.groupby('year'):
            counts = g.groupby(['od', 'regime']).size().unstack(fill_value=0)
            distance = jensenshannon(counts.day, counts.night, base=2)
            m = min(20000, int(counts.sum().min()))
            draws = []
            null_draws = []
            pooled = counts.sum(axis=1) / counts.to_numpy().sum()
            for _ in range(30):
                samples = [rng.multinomial(m, counts[c] / counts[c].sum()) for c in ('day', 'night')]
                draws.append(jensenshannon(*samples, base=2))
                null_draws.append(jensenshannon(rng.multinomial(m, pooled), rng.multinomial(m, pooled), base=2))
            rows.append({'year': year, 'cell_km': size, 'js_distance': distance,
                         'equal_sample_n': m, 'equal_sample_js_mean': np.mean(draws),
                         'equal_sample_js_sd': np.std(draws), 'pooled_null_js_mean': np.mean(null_draws),
                         'pooled_null_js_sd': np.std(null_draws)})
    pd.DataFrame(rows).to_csv(OUT / 'od-comparison.csv', index=False)
    dump('sharing-coverage.json', [{'year': int(year), 'dates_with_retained_departures': g.date.nunique(),
        'minimum_daily_departures': int(g.groupby('date').size().min()),
        'start_min': str(g.start.min()), 'start_max': str(g.start.max()),
        'end_after_2024': int(g.end.ge('2025-01-01').sum())} for year, g in trips.groupby('year')])


def endpoint_maps(trips, sites):
    dates = pd.date_range('2023-01-01', '2024-12-31').strftime('%Y-%m-%d')
    dates = dates[~dates.isin(DST_DATES)]
    rows = []
    for event, xs, ys, timecol in [('departures', 'ox', 'oy', 'start'), ('arrivals', 'dx', 'dy', 'end')]:
        t = trips[timecol]
        keep = t.dt.strftime('%Y-%m-%d').isin(dates)
        z = trips.loc[keep]
        h = t.loc[keep].dt.hour
        window = np.select([h.between(6, 9), h.between(10, 14), h.between(15, 19)], [0, 1, 2], default=3)
        grid = pd.DataFrame({'gx': np.floor(z[xs] / .5).astype(int),
                             'gy': np.floor(z[ys] / .5).astype(int), 'window': window})
        rows.append(grid.groupby(['gx', 'gy', 'window']).size().rename(event))
    grid = pd.concat(rows, axis=1).fillna(0).reset_index()
    grid['events'] = grid.arrivals + grid.departures
    grid['exposure'] = np.array([4, 5, 5, 10])[grid.window] * len(dates)
    grid['activity'] = grid.events / grid.exposure
    grid['balance'] = (grid.arrivals - grid.departures) / grid.events
    grid['displayed'] = grid.events.ge(50)
    grid.to_csv(OUT / 'sharing-map-grid.csv', index=False)
    titles = ['06–10', '10–15', '15–20', '20–06 (calendar-date bins)']
    shown = grid.loc[grid.displayed]
    for name, col, cmap, vmin, vmax in [('activity', 'activity', 'viridis', 0, shown.activity.max()),
                                      ('balances', 'balance', 'RdBu_r', -1, 1)]:
        fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.3), sharex=True, sharey=True)
        for i, ax in enumerate(axes.flat):
            z = shown.loc[shown.window.eq(i)]
            points = ax.scatter((z.gx + .5) * .5, (z.gy + .5) * .5, c=z[col], marker='s',
                                s=16, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
            ax.scatter(sites.x, sites.y, marker='.', s=1, color='black', alpha=.2)
            ax.set(title=titles[i], aspect='equal', xlim=(-12, 13), ylim=(-11, 12))
        for ax in axes[1]: ax.set_xlabel('East of reference point (km)')
        for ax in axes[:, 0]: ax.set_ylabel('North of reference point (km)')
        fig.colorbar(points, ax=axes, shrink=.8, label='Endpoint events per hour' if col == 'activity' else '(Arrivals − departures) / events')
        save(fig, name)


def build_features(trips, sites, short=False):
    if short:
        trips = trips.loc[trips.seconds.le(6 * 3600)].copy()
    dates = pd.date_range('2023-01-01', '2024-12-31').strftime('%Y-%m-%d')
    index = pd.MultiIndex.from_product([dates, ['day', 'night']], names=['date', 'regime'])
    key = pd.MultiIndex.from_frame(trips[['date', 'regime']])
    pos = index.get_indexer(key)
    arrivals = pd.DataFrame({'date': trips.end.dt.strftime('%Y-%m-%d'), 'regime': period(trips.end)})
    arrival_pos = index.get_indexer(pd.MultiIndex.from_frame(arrivals))
    total = np.bincount(pos, minlength=len(index))
    ox, oy, dx, dy = [trips[c].to_numpy() for c in ('ox', 'oy', 'dx', 'dy')]
    frames = []
    for site, s in sites.iterrows():
        u, delta = chord_distance(ox, oy, dx, dy, s.x, s.y)
        z = index.to_frame(index=False)
        z['site'] = site
        z['global_trips'] = total
        starts = np.hypot(ox - s.x, oy - s.y) <= .5
        ends = (np.hypot(dx - s.x, dy - s.y) <= .5) & (arrival_pos >= 0)
        z['starts_500m'] = np.bincount(pos[starts], minlength=len(index))
        z['ends_500m'] = np.bincount(arrival_pos[ends], minlength=len(index))
        for radius in (100, 250):
            mask = (u > 0) & (u < 1) & (delta <= radius / 1000)
            z[f'chord_{radius}'] = np.bincount(pos[mask], minlength=len(index))
        frames.append(z)
    features = pd.concat(frames, ignore_index=True)
    features.to_parquet(CACHE / ('features-short.parquet' if short else 'features.parquet'), index=False)


def main():
    trips = pd.read_parquet(CACHE / 'trips.parquet')
    sites = pd.read_csv(OUT / 'sites.csv', index_col='site')
    descriptions(trips)
    endpoint_maps(trips, sites)
    build_features(trips, sites)
    build_features(trips, sites, short=True)


if __name__ == '__main__':
    main()
