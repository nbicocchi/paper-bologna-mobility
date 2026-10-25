"""Pool 2023--2024 scalar trip measurements; freeze only aggregate histograms."""
import json
import math
import numpy as np
import pandas as pd
from common import INPUT, OUT, REPO, TEXT, sha

TRIP_SOURCES = {
    'bike': ['bike-sharing-code/code/dati_esterni/trips-2023.parquet',
             'bike-sharing-code/code/dati_esterni/trips-2024.parquet'],
    'car': ['car-sharing-code/code/dati_esterni/cars/trips.parquet'],
}
SNAPSHOTS = ['distribution-histograms.csv', 'distribution-summary.csv', 'distribution-design.json',
             'distribution-hourly.csv', 'distribution-monthly.csv']


def aggregate_measurements(values):
    """Histogram counts and exact quantiles for one finite nonnegative group."""
    rows, summaries = [], []
    for (mode, daytype, metric), observations in values.items():
        data = np.concatenate(observations)
        data = data[np.isfinite(data)]
        if not len(data) or (data < 0).any():
            raise ValueError('Empty or negative distribution measurements')
        if metric == 'duration_min':
            upper = max(1., 10. ** math.ceil(math.log10(float(data.max()))))
            edges = np.geomspace(.5, upper, 401)
        else:
            # The analytical endpoint box is less than 40 km across.
            edges = np.linspace(0, 40, 801)
        counts, edges = np.histogram(data, bins=edges)
        if counts.sum() != len(data):
            raise ValueError('Histogram did not retain all valid measurements')
        cumulative = np.cumsum(counts)
        for left, right, count, total in zip(edges[:-1], edges[1:], counts, cumulative):
            rows.append(dict(mode=mode, daytype=daytype, metric=metric,
                             left=left, right=right, count=int(count),
                             cumulative_count=int(total), n=len(data)))
        q25, median, q75, p95, p99 = np.quantile(data, [.25, .5, .75, .95, .99])
        summaries.append(dict(mode=mode, daytype=daytype, metric=metric, n=len(data),
                              q25=q25, median=median, q75=q75, p95=p95, p99=p99))
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def freeze_distributions(frozen=False):
    """Read selected scalar columns only; never copy trip rows or endpoint positions."""
    if not frozen:
        values, source_records, cohorts, hourly, monthly = {}, [], [], [], []
        for mode, relatives in TRIP_SOURCES.items():
            for relative in relatives:
                path = REPO / relative
                before = sha(path)
                columns = ['start', 'duration', 'chord'] if mode == 'bike' else ['start', 'seconds', 'displacement']
                frame = pd.read_parquet(path, columns=columns)
                if frame.start.isna().any() or not frame.start.dt.year.isin([2023, 2024]).all():
                    raise ValueError('Unexpected source timestamp/year')
                # Harmonised calendar weekday of departure, not bike service date.
                daytype = np.where(frame.start.dt.weekday < 5, 'Weekday', 'Weekend')
                for month, count in frame.groupby(frame.start.dt.strftime('%Y-%m')).size().items():
                    monthly.append(dict(mode=mode, month=month, count=int(count)))
                for year in [2023, 2024]:
                    count = int(frame.start.dt.year.eq(year).sum())
                    if count:
                        cohorts.append(dict(mode=mode, year=year, retained=count))
                for label in ['Weekday', 'Weekend']:
                    subset = frame.loc[daytype == label]
                    for hour, count in enumerate(np.bincount(subset.start.dt.hour, minlength=24)):
                        hourly.append(dict(mode=mode, daytype=label, hour=hour, count=int(count)))
                    duration = subset.duration if mode == 'bike' else subset.seconds / 60
                    displacement = subset.chord if mode == 'bike' else subset.displacement
                    for metric, data in [('duration_min', duration), ('displacement_km', displacement)]:
                        values.setdefault((mode, label, metric), []).append(data.to_numpy())
                if sha(path) != before:
                    raise RuntimeError('Trip source changed during read')
                source_records.append(dict(source=relative, sha256=before,
                    snapshot=None, access='Read-only scalar projection; no row-level snapshot'))
        histograms, summary = aggregate_measurements(values)
        histograms.to_csv(INPUT / SNAPSHOTS[0], index=False)
        summary.to_csv(INPUT / SNAPSHOTS[1], index=False)
        hourly = pd.DataFrame(hourly).groupby(['mode', 'daytype', 'hour'], as_index=False)['count'].sum()
        hourly['share'] = hourly['count'] / hourly.groupby(['mode', 'daytype'])['count'].transform('sum')
        hourly.to_csv(INPUT / SNAPSHOTS[3], index=False)
        monthly = pd.DataFrame(monthly).groupby(['mode', 'month'], as_index=False)['count'].sum()
        monthly.to_csv(INPUT / SNAPSHOTS[4], index=False)
        design = dict(years=[2023, 2024], pooling='All retained trips pooled, not equal-year averaging',
                      daytype='Departure calendar date: Monday-Friday / Saturday-Sunday; holidays not reclassified',
                       hourly='Reported local departure hour; normalised within mode/daytype after pooling years',
                       monthly='Retained departures by local calendar month; absent months not imputed as zero',
                      distance='Straight-line endpoint displacement in km, not recorded route distance',
                      duration={'bike': 'Source valid-duration subset: 0.5-180 min and endpoint speed <=45 km/h',
                                'car': 'Retained rental duration, at least 0.5 min; no upper cap; may include stops'},
                      distance_cohort={'bike': 'Retained cohort, endpoint displacement >=100 m',
                                       'car': 'Retained cohort including near-origin returns'},
                      privacy='Only start, duration/seconds and chord/displacement read; no IDs or coordinates',
                      sources=source_records, cohorts=cohorts)
        (INPUT / SNAPSHOTS[2]).write_text(json.dumps(design, indent=2) + '\n')
    design = json.loads((INPUT / SNAPSHOTS[2]).read_text())
    records = list(design['sources'])
    for name in SNAPSHOTS:
        records.append(dict(source=str((INPUT / name).relative_to(REPO)), snapshot=name,
                            sha256=sha(INPUT / name)))
    return records


def draw_distributions():
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    histograms = pd.read_csv(INPUT / SNAPSHOTS[0])
    summary = pd.read_csv(INPUT / SNAPSHOTS[1])
    histograms.to_csv(OUT / SNAPSHOTS[0], index=False)
    summary.to_csv(OUT / SNAPSHOTS[1], index=False)
    hourly = pd.read_csv(INPUT / SNAPSHOTS[3])
    hourly.to_csv(OUT / SNAPSHOTS[3], index=False)
    fig, axes = plt.subplots(2, 4, figsize=(13.4, 7.4), layout='constrained')
    monthly = pd.read_csv(INPUT / SNAPSHOTS[4])
    monthly.to_csv(OUT / SNAPSHOTS[4], index=False)
    for row, (mode, title) in enumerate(zip(['bike', 'car'], ['Bike-sharing', 'Car-sharing'])):
        ax = axes[row, 0]
        series = monthly[monthly['mode'] == mode].set_index('month')['count']
        months = pd.date_range('2023-01-01', '2024-12-01', freq='MS')
        counts = series.reindex(months.strftime('%Y-%m')).to_numpy(dtype=float)
        ax.plot(months, counts/1000, '-o', color='#0072b2', lw=1.7, ms=3)
        ax.axvline(pd.Timestamp('2024-01-01'), color='gray', ls=':', lw=1)
        ax.set(title=f'({chr(97+row*4)}) {title} · monthly use',
               ylabel='Rentals per month (thousands)', xlabel='Departure month',
               ylim=(0, np.nanmax(counts)/1000*1.12), xlim=(months[0], months[-1]))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        ax.grid(alpha=.2)
        ax = axes[row, 1]
        for label, color, style in [('Weekday', '#0072b2', '-o'), ('Weekend', '#d55e00', '--s')]:
            h = hourly[(hourly['mode'] == mode) & (hourly.daytype == label)]
            ax.plot(h.hour, 100*h.share, style, color=color, lw=1.7, ms=3, label=label)
        ax.set(title=f'({chr(98+row*4)}) {title} · timing', xlabel='Local departure hour',
               ylabel='Departure share (%)', xlim=(0, 23), xticks=[0, 6, 12, 18, 23],
               ylim=(0, 110*hourly.share.max()))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    for column, (metric, xlabel) in enumerate([
            ('displacement_km', 'Endpoint displacement (km)'),
            ('duration_min', 'Rental duration (minutes)')], start=2):
        for row, (mode, title) in enumerate(zip(['bike', 'car'], ['Bike-sharing', 'Car-sharing'])):
            ax = axes[row, column]
            for label, color, style in [('Weekday', '#0072b2', '-'), ('Weekend', '#d55e00', '--')]:
                h = histograms[(histograms['mode'] == mode) & (histograms.daytype == label)
                               & (histograms.metric == metric)].sort_values('left')
                s = summary[(summary['mode'] == mode) & (summary.daytype == label)
                            & (summary.metric == metric)].iloc[0]
                # Binned empirical CDF; all counts retained, no tail trimming.
                x = np.r_[h.left.iloc[0], h.right.to_numpy()]
                y = np.r_[0., 100*h.cumulative_count.to_numpy()/h.n.iloc[0]]
                ax.step(x, y, where='post', color=color, ls=style, lw=1.8,
                        label=f'{label} (median {s["median"]:.1f})')
            measure_title = 'distance' if metric == 'displacement_km' else 'duration'
            ax.set_title(f'({chr(97+row*4+column)}) {title} · {measure_title}')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Cumulative share (%)')
            ax.set_ylim(0, 100)
            ax.grid(alpha=.2)
            if metric == 'duration_min':
                ax.set_xscale('log')
                ax.set_xlim(.5, histograms[(histograms['mode'] == mode) &
                                          (histograms.metric == metric)].right.max())
            else:
                # Zoom to the central distribution; retain and label the full tail in outputs.
                ax.set_xlim(0, 20)
            ax.legend(fontsize=8, loc='lower right')
    fig.supxlabel('Monthly retained rentals at left; other columns pooled over 2023–2024\n'
                  'Weekday: Monday–Friday · Weekend: Saturday–Sunday', fontsize=9)
    fig.savefig(TEXT / 'figures' / 'sharing-distributions.pdf', bbox_inches='tight',
                metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
