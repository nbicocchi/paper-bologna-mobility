"""Fresh trip distributions and matched-date urban-rhythm maps from the raw-data cohorts."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from common import CACHE, OUT, TEXT, period
from basemap import load as load_basemap, backdrop, project, ATTRIBUTION

WINDOWS = ['06–10', '10–15', '15–20', '20–06']
LIMITS = (-7, 8, -5, 6)
MINIMUM = {'bike': 200, 'car': 50}
FOLD_COLOURS = ['#0072b2', '#e69f00', '#009e73', '#cc79a7', '#d55e00']


def save(fig, name):
    fig.savefig(TEXT / 'figures' / f'{name}.pdf', bbox_inches='tight',
                metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def window(time):
    return np.select([time.dt.hour.between(6, 9), time.dt.hour.between(10, 14),
                      time.dt.hour.between(15, 19)], [0, 1, 2], default=3)


def map_periods():
    """Same actual hours and continuous service-date windows for both services."""
    from common import DST_DATES
    time = pd.Series(pd.date_range('2023-01-01 06:00', '2025-01-01 06:00',
                     tz='Europe/Rome', freq='h', inclusive='left')).dt.tz_localize(None)
    date, regime = period(time, 'bike')
    h = pd.DataFrame({'date': date, 'regime': regime, 'window': window(time),
                      'valid': time.dt.year.isin([2023, 2024]) & ~time.dt.strftime('%Y-%m-%d').isin(DST_DATES)})
    coverage = pd.read_csv(OUT / 'bike-sharing-coverage.csv')
    h = h.merge(coverage[['date', 'regime', 'usable']], on=['date', 'regime'], how='left', validate='many_to_one')
    h['valid'] &= h.usable.fillna(False).astype(bool)
    periods = h.groupby(['date', 'window']).agg(exposure=('valid', 'size'), valid=('valid', 'all')).reset_index()
    return periods.loc[periods.valid].drop(columns='valid')


def spatial_grid(t, periods, mode):
    counts = []
    for prefix, timestamp, event in [('o', 'start', 'departures'), ('d', 'end', 'arrivals')]:
        date, _ = period(t[timestamp], 'bike')
        z = pd.DataFrame({'date': date, 'window': window(t[timestamp]),
                          'gx': np.floor(t[prefix + 'x'] / .5).astype(int),
                          'gy': np.floor(t[prefix + 'y'] / .5).astype(int)})
        z = z.merge(periods[['date', 'window']], on=['date', 'window'], validate='many_to_one')
        counts.append(z.groupby(['window', 'gx', 'gy']).size().rename(event))
    grid = pd.concat(counts, axis=1).fillna(0).reset_index()
    grid['events'] = grid.arrivals + grid.departures
    grid['exposure'] = grid.window.map(periods.groupby('window').exposure.sum())
    grid['events_per_hour'] = grid.events / grid.exposure
    grid['normalised_balance'] = (grid.arrivals - grid.departures) / grid.events
    grid['displayed'] = grid.events >= MINIMUM[mode]
    grid['in_viewport'] = grid.gx.between(-14, 15) & grid.gy.between(-10, 11)
    grid['mode'] = mode
    return grid


def rhythms(grids):
    regions, reversals, concordance = [], [], []
    for mode, grid in grids.items():
        z = grid.loc[grid.in_viewport].copy()
        x, y = (z.gx + .5) * .5, (z.gy + .5) * .5
        z['region'] = np.select([np.hypot(x, y) < 1.5, y >= 1.5], ['Centre', 'Northern band'], default='Remaining area')
        r = z.groupby(['region', 'window'])[['arrivals', 'departures']].sum().reset_index()
        r['normalised_balance'] = (r.arrivals - r.departures) / (r.arrivals + r.departures)
        r['mode'] = mode
        regions.append(r)
        common = z.loc[z.displayed].pivot(index=['gx', 'gy'], columns='window', values='normalised_balance')[[0, 2]].dropna()
        reversals.append(dict(mode=mode, common_displayed_cells=len(common), reversing_cells=int((common[0] * common[2] < 0).sum()),
                              reversal_fraction=float((common[0] * common[2] < 0).mean())))
    for w in range(4):
        parts = [g.loc[g.in_viewport & g.displayed & g.window.eq(w)].set_index(['gx', 'gy']).normalised_balance.rename(m)
                 for m, g in grids.items()]
        paired = pd.concat(parts, axis=1, join='inner')
        concordance.append(dict(window=w, common_displayed_cells=len(paired), balance_correlation=paired.bike.corr(paired.car)))
    return pd.concat(regions), pd.DataFrame(reversals), pd.DataFrame(concordance)


def draw_maps(grids):
    networks = {mode: pd.read_csv(OUT / f'{mode}-sites.csv') for mode in ['bike', 'car']}
    all_sites = pd.concat(networks.values(), ignore_index=True)
    car = networks['car']
    blocks = np.array([list(map(int, block.split(':'))) for block in car.block.unique()])
    network_bounds = (min(all_sites.x.min(), blocks[:, 0].min()) - .6,
                      max(all_sites.x.max(), blocks[:, 0].max() + 1) + .6,
                      min(all_sites.y.min(), blocks[:, 1].min()) - .6,
                      max(all_sites.y.max(), blocks[:, 1].max() + 1) + .6)
    bounds = (min(LIMITS[0], network_bounds[0]), max(LIMITS[1], network_bounds[1]),
              min(LIMITS[2], network_bounds[2]), max(LIMITS[3], network_bounds[3]))
    context, extent = load_basemap(bounds)
    index = pd.MultiIndex.from_product([range(-10, 12), range(-14, 16)], names=['gy', 'gx'])
    x, y = np.arange(-7, 8.5, .5), np.arange(-5, 6.5, .5)
    mx, my = project(*np.meshgrid(x, y))
    for name, measure in [('sharing-activity', 'events_per_hour'), ('sharing-balance', 'normalised_balance')]:
        fig, axes = plt.subplots(2, 4, figsize=(13, 6.3), layout='constrained', sharex=True, sharey=True)
        for row, (mode, grid) in enumerate(grids.items()):
            maximum = grid.loc[grid.displayed & grid.in_viewport, 'events_per_hour'].max()
            norm = Normalize(0, maximum) if measure == 'events_per_hour' else TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
            for w, ax in enumerate(axes[row]):
                z = grid.loc[grid.window.eq(w)].set_index(['gy', 'gx']).reindex(index)
                values = z[measure].to_numpy().reshape(22, 30)
                values = np.ma.masked_where(~z.displayed.fillna(False).astype(bool).to_numpy().reshape(22, 30), values)
                backdrop(ax, context, extent, LIMITS)
                mesh = ax.pcolormesh(mx, my, values, cmap='YlOrRd' if measure == 'events_per_hour' else 'RdBu_r',
                                     norm=norm, rasterized=True, alpha=.8, zorder=1)
                ax.plot(*project(0, 0), '+', color='black', markersize=5, zorder=3)
                ax.set_title(f'({chr(97 + row * 4 + w)}) {WINDOWS[w]}')
                if row == 1:
                    ax.set_xlabel('East of centre (km)')
                if w == 0:
                    ax.set_ylabel(f'{"Bike" if mode == "bike" else "Car"}-sharing\nNorth of centre (km)')
            fig.colorbar(mesh, ax=axes[row].tolist(), shrink=.8,
                         label='Endpoint events / hour' if measure == 'events_per_hour' else '(Arrivals − departures) / events')
        fig.supxlabel(ATTRIBUTION + '\n500 m cells · matched hours · cross: city reference point · uncoloured: suppressed, not zero', fontsize=8)
        save(fig, name)
    draw_counter_network(networks, context, extent, network_bounds)


def draw_counter_network(networks, context, extent, bounds):
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch, Polygon

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True, layout='constrained')
    for mode, ax in zip(['bike', 'car'], axes):
        sites = networks[mode]
        backdrop(ax, context, extent, bounds)
        sx, sy = project(sites.x, sites.y)
        colours = sites.fold.map(dict(enumerate(FOLD_COLOURS))) if mode == 'car' else '#0072b2'
        ax.scatter(sx, sy, s=40 if mode == 'bike' else 10, alpha=.9, color=colours,
                    edgecolors='white', linewidths=.4, zorder=3)
        if mode == 'car':
            for block, members in sites.groupby('block'):
                if members.fold.nunique() != 1:
                    raise ValueError(f'Spatial block {block} belongs to multiple folds')
                colour = FOLD_COLOURS[int(members.fold.iloc[0])]
                gx, gy = map(int, block.split(':'))
                bx, by = project(np.array([gx, gx+1, gx+1, gx]), np.array([gy, gy, gy+1, gy+1]))
                ax.add_patch(Polygon(np.column_stack([bx, by]), lw=.8,
                                     facecolor=to_rgba(colour, .18), edgecolor=colour, zorder=2))
            handles = [Patch(facecolor=to_rgba(colour, .18), edgecolor=colour, label=f'Fold {fold + 1}')
                       for fold, colour in enumerate(FOLD_COLOURS)]
            ax.legend(handles=handles, title='Spatial validation', loc='lower left',
                      fontsize=7, title_fontsize=8, framealpha=.95)
        else:
            for s in sites.itertuples():
                ax.annotate(s.site.replace('_', ' '), project(s.x, s.y), xytext=(4, 4), textcoords='offset points', fontsize=8,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1), zorder=4)
        ax.set(title=f'Bicycle counters: {len(sites)} locations' if mode == 'bike' else f'Motor-vehicle counters: {len(sites)} devices',
               xlabel='East of centre (km)', ylabel='North of centre (km)', aspect='equal')
    fig.supxlabel(ATTRIBUTION + '\nCartographic context only; not the historical routing network', fontsize=8)
    save(fig, 'counter-network')


def distributions(trips):
    monthly_rows, hourly_rows, summary_rows = [], [], []
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), layout='constrained')
    for row, (mode, t) in enumerate(trips.items()):
        monthly = t.groupby(t.start.dt.strftime('%Y-%m')).size()
        monthly_rows += [dict(mode=mode, month=m, count=int(n)) for m, n in monthly.items()]
        dates = pd.to_datetime(monthly.index)
        axes[row, 0].plot(dates, monthly / 1000, '-o', markersize=3)
        axes[row, 0].set(ylabel=f'{"Bike" if mode == "bike" else "Car"} rentals (thousands)')
        axes[row, 0].tick_params(axis='x', rotation=45, labelsize=7)
        for daytype, mask, color in [('Weekday', t.start.dt.weekday < 5, '#0072b2'), ('Weekend', t.start.dt.weekday >= 5, '#d55e00')]:
            z = t.loc[mask]
            counts = z.start.dt.hour.value_counts().reindex(range(24), fill_value=0).sort_index()
            hourly_rows += [dict(mode=mode, daytype=daytype, hour=h, count=int(n), share=n / len(z)) for h, n in counts.items()]
            axes[row, 1].plot(counts.index, counts / len(z) * 100, label=daytype, color=color)
            for column, metric in [(2, 'displacement'), (3, 'duration_min')]:
                values = np.sort(z[metric].dropna().to_numpy())
                selection = np.linspace(0, len(values) - 1, min(1200, len(values))).astype(int)
                q = np.quantile(values, [.25, .5, .75, .95, .99])
                summary_rows.append(dict(mode=mode, daytype=daytype, metric=metric, n=len(values),
                                         q25=q[0], median=q[1], q75=q[2], p95=q[3], p99=q[4]))
                axes[row, column].plot(values[selection], (selection + 1) / len(values), color=color,
                                      label=f'{daytype}: {q[1]:.2f}')
        axes[row, 1].set(xlabel='Departure hour', ylabel='Departures (%)', xlim=(0, 23))
        axes[row, 1].legend(fontsize=7)
        axes[row, 2].set(xlabel='Endpoint displacement (km)', ylabel='Cumulative fraction', xlim=(0, 20), ylim=(0, 1))
        axes[row, 3].set(xlabel='Rental duration (min)', xscale='log', ylabel='Cumulative fraction', ylim=(0, 1))
        for ax in axes[row, 2:]:
            ax.legend(fontsize=7)
        for column, ax in enumerate(axes[row]):
            ax.set_title(f'({chr(97 + 4 * row + column)})')
    save(fig, 'sharing-distributions')
    pd.DataFrame(monthly_rows).to_csv(OUT / 'distribution-monthly.csv', index=False)
    pd.DataFrame(hourly_rows).to_csv(OUT / 'distribution-hourly.csv', index=False)
    pd.DataFrame(summary_rows).to_csv(OUT / 'distribution-summary.csv', index=False)


def main():
    trips = {mode: pd.read_parquet(CACHE / f'{mode}-trips.parquet') for mode in ['bike', 'car']}
    distributions(trips)
    periods = map_periods()
    periods.to_csv(OUT / 'map-periods.csv', index=False)
    grids = {mode: spatial_grid(t, periods, mode) for mode, t in trips.items()}
    pd.concat(grids.values()).to_csv(OUT / 'spatial-grid.csv', index=False)
    for name, data in zip(['rhythm-regions', 'rhythm-reversals', 'rhythm-concordance'], rhythms(grids)):
        data.to_csv(OUT / f'{name}.csv', index=False)
    draw_maps(grids)


if __name__ == '__main__':
    main()
