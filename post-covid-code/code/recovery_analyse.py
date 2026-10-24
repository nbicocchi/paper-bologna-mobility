"""Fixed-panel, calendar-standardised recovery indices and transparent sensitivities."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from recovery_prepare import ROOT, CACHE, OUT, YEARS, HOURS

TEXT = ROOT.parent / 'paper-car-sharing-text'
FIG = TEXT / 'figures'
TAB = TEXT / 'tables'
SEED = 20261025


def cells(d, positive_only=False):
    z = d.loc[~d.unstable & ~d.holiday].copy()
    if positive_only:
        z = z.loc[z.total.gt(0)]
    g = z.groupby(['site', 'year', 'month', 'weekend'])
    means = g[['total'] + HOURS].mean()
    means['days'] = g.size()
    return means


def panel(d, c, minimum=120, weekday_min=2, weekend_min=1):
    annual = d.loc[~d.unstable].groupby(['site', 'year']).size().unstack().reindex(columns=YEARS)
    available = annual.ge(minimum).all(axis=1)
    valid = c.days.ge(np.where(c.index.get_level_values('weekend') == 0, weekday_min, weekend_min))
    n = valid.groupby('site').sum()
    ids = n.index[(n == 7 * 12 * 2) & available.reindex(n.index, fill_value=False)]
    base = c.loc[c.index.get_level_values('year') == 2019].groupby('site').total.mean()
    return ids.intersection(base.index[base.gt(0)])


def standardized(c, ids):
    z = c.loc[c.index.get_level_values('site').isin(ids)].reset_index()
    weights = np.where(z.weekend.eq(0), 5 / 7, 2 / 7)
    z[['total'] + HOURS] = z[['total'] + HOURS].mul(weights, axis=0)
    return z.groupby(['site', 'year', 'month'])[['total'] + HOURS].sum()


def spatial_ci(values, base, blocks, repetitions=2000):
    """Ratio of sums, block resampling with all devices retained inside sampled blocks."""
    a = pd.DataFrame({'a': values, 'b': base, 'block': blocks}).groupby('block')[['a', 'b']].sum()
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(a), size=(repetitions, len(a)))
    ratios = 100 * a.a.to_numpy()[picks].sum(axis=1) / a.b.to_numpy()[picks].sum(axis=1)
    return np.percentile(ratios, [2.5, 97.5])


def table(name, headers, rows, alignment=None):
    alignment = alignment or ('l' + 'r' * (len(headers) - 1))
    lines = [r'\begin{tabular}{' + alignment + '}', r'\toprule', ' & '.join(headers) + r' \\', r'\midrule']
    lines += [' & '.join(map(str, row)) + r' \\' for row in rows]
    lines += [r'\bottomrule', r'\end{tabular}']
    (TAB / f'recovery-{name}.tex').write_text('\n'.join(lines) + '\n')


def savefig(name):
    plt.savefig(FIG / f'recovery-{name}.pdf', bbox_inches='tight')
    plt.savefig(FIG / f'recovery-{name}.png', dpi=180, bbox_inches='tight')
    plt.close()


def main():
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)
    d = pd.read_parquet(CACHE / 'daily.parquet')
    meta = pd.read_csv(OUT / 'sites.csv', index_col='site')
    c = cells(d)
    c.to_parquet(CACHE / 'cells.parquet')
    ids = panel(d, c)
    if len(ids) < 10:
        raise RuntimeError(f'Only {len(ids)} balanced sites: inspect coverage before changing design')
    meta = meta.loc[ids]
    meta.to_csv(OUT / 'panel-sites.csv')
    s = standardized(c, ids)
    s.to_csv(OUT / 'standardized-site-month.csv')
    annual = s.groupby(['site', 'year']).mean()
    annual.to_csv(OUT / 'standardized-site-year.csv')
    av = annual.total.unstack('year')
    site_ratios = av.div(av[2019], axis=0) * 100
    site_ratios.to_csv(OUT / 'site-indices.csv')
    results = []
    for year in YEARS:
        ratios = site_ratios[year]
        lo, hi = spatial_ci(av[year], av[2019], meta.block)
        results.append({'year': year, 'index': 100 * av[year].sum() / av[2019].sum(),
                        'lower': lo, 'upper': hi, 'median_site_index': ratios.median(),
                        'q25': ratios.quantile(.25), 'q75': ratios.quantile(.75),
                        'sites_at_95': ratios.ge(95).mean() * 100,
                        'sites_at_100': ratios.ge(100).mean() * 100})
    a = pd.DataFrame(results)
    a.to_csv(OUT / 'annual.csv', index=False)
    monthly = s.total.groupby(['year', 'month']).sum().unstack('year')
    monthly = monthly.div(monthly[2019], axis=0) * 100
    m = monthly.stack().rename('index').reset_index().sort_values(['year', 'month'])
    m['date'] = pd.to_datetime(dict(year=m.year, month=m.month, day=1))
    m.to_csv(OUT / 'monthly.csv', index=False)
    bytype = []
    hourly = []
    for weekend, label in [(0, 'Weekday'), (1, 'Weekend')]:
        z = c.loc[c.index.get_level_values('site').isin(ids) & (c.index.get_level_values('weekend') == weekend)]
        v = z.groupby(['site', 'year'])[['total'] + HOURS].mean()
        sums = v.groupby('year').sum()
        for year in YEARS:
            bytype.append({'year': year, 'type': label, 'index': 100 * sums.loc[year, 'total'] / sums.loc[2019, 'total']})
            for hour in range(24):
                hourly.append({'year': year, 'type': label, 'hour': hour,
                    'index': 100 * sums.loc[year, HOURS[hour]] / sums.loc[2019, HOURS[hour]],
                    'share': 100 * sums.loc[year, HOURS[hour]] / sums.loc[year, 'total'],
                    'mean_count': sums.loc[year, HOURS[hour]] / len(ids)})
    bt, h = pd.DataFrame(bytype), pd.DataFrame(hourly)
    bt.to_csv(OUT / 'day-type.csv', index=False)
    h.to_csv(OUT / 'hourly.csv', index=False)
    peaks = []
    for year in YEARS:
        z = h.loc[(h.year == year) & h.type.eq('Weekday')]
        baseline = h.loc[h.year.eq(2019) & h.type.eq('Weekday')]
        for label, hours in [('Morning 07--09', [7, 8]), ('Evening 17--19', [17, 18]),
                             ('Midday 10--16', list(range(10, 16))), ('Night 00--06', list(range(6)))]:
            v = z.loc[z.hour.isin(hours)]
            b = baseline.loc[baseline.hour.isin(hours)]
            peaks.append({'year': year, 'window': label, 'index': 100 * v.mean_count.sum() / b.mean_count.sum(),
                          'share': v.share.sum(), 'share_change_pp': v.share.sum() - b.share.sum()})
    peaks = pd.DataFrame(peaks)
    peaks.to_csv(OUT / 'windows.csv', index=False)
    shapes = []
    for year in YEARS:
        for label in ['Weekday', 'Weekend']:
            p = h.loc[h.year.eq(year) & h.type.eq(label)].sort_values('hour').share.to_numpy()
            q = h.loc[h.year.eq(2019) & h.type.eq(label)].sort_values('hour').share.to_numpy()
            shapes.append({'year': year, 'type': label, 'js_distance': jensenshannon(p, q, base=2),
                           'total_variation': np.abs(p-q).sum()/200})
    pd.DataFrame(shapes).to_csv(OUT / 'shape.csv', index=False)
    # Vary annual coverage while retaining the primary monthly requirements.
    sensitivities = []
    designs = [(f'Balanced {n} days', n, 2, 1) for n in [90, 120, 180, 240]]
    for label, threshold, weekday_min, weekend_min in designs:
        cohort = panel(d, c, threshold, weekday_min, weekend_min)
        if len(cohort):
            v = standardized(c, cohort).total.groupby(['site', 'year']).mean().unstack('year')
            for year in YEARS:
                sensitivities.append({'design': label, 'sites': len(cohort), 'year': year,
                                       'index': 100 * v[year].sum()/v[2019].sum()})
        else:
            for year in YEARS:
                sensitivities.append({'design': label, 'sites': 0, 'year': year, 'index': np.nan})
    cp = cells(d, positive_only=True)
    positive_ids = panel(d.loc[d.total.gt(0)], cp).intersection(ids)
    if len(positive_ids):
        v = standardized(cp, positive_ids).total.groupby(['site', 'year']).mean().unstack('year')
        for year in YEARS:
            sensitivities.append({'design': 'Positive-day diagnostic', 'sites': len(positive_ids), 'year': year,
                                   'index': 100*v[year].sum()/v[2019].sum()})
    # Pairwise 2019/year panels relax seven-year survivor selection but change composition.
    for year in YEARS[1:]:
        sub = d.loc[d.year.isin([2019, year]) & ~d.unstable]
        cov = sub.groupby(['site', 'year']).size().unstack().ge(120).all(axis=1)
        cc = c.loc[c.index.get_level_values('year').isin([2019, year])]
        good = cc.days.ge(np.where(cc.index.get_level_values('weekend') == 0, 2, 1)).groupby('site').sum()
        pair = good.index[(good == 48) & cov.reindex(good.index, fill_value=False)]
        v = standardized(cc, pair).total.groupby(['site', 'year']).mean().unstack('year')
        valid = v[2019].gt(0)
        v = v.loc[valid]
        sensitivities.append({'design': 'Pairwise balanced', 'sites': len(v), 'year': year,
                               'index': 100*v[year].sum()/v[2019].sum()})
    sens = pd.DataFrame(sensitivities)
    # Give each spatial block equal influence after pooling counts inside that block.
    grouped = annual.reset_index().merge(meta[['block']], on='site').groupby(['block', 'year']).total.sum().unstack('year')
    block_index = 100 * grouped.div(grouped[2019], axis=0).mean()
    sens = pd.concat([sens, pd.DataFrame({'design': 'Equal spatial blocks', 'sites': len(ids),
                                         'year': YEARS, 'index': block_index.reindex(YEARS).to_numpy()})], ignore_index=True)
    sens.to_csv(OUT / 'sensitivity.csv', index=False)
    # Deliberately naive comparison, retained only to diagnose compositional confounding.
    naive = d.groupby('year').agg(total=('total', 'sum'), observed_days=('total', 'size'), sites=('site', 'nunique'))
    naive['raw_total_index'] = 100*naive.total/naive.loc[2019, 'total']
    naive['per_available_day_index'] = 100*(naive.total/naive.observed_days)/(naive.loc[2019, 'total']/naive.loc[2019, 'observed_days'])
    naive.to_csv(OUT / 'naive-comparison.csv')
    # Descriptive linear slope, month effects; Newey-West covariance allows serial dependence.
    import statsmodels.api as sm
    slopes = []
    for start in [2022, 2023]:
        z = m.loc[m.year.ge(start)].copy()
        X = pd.get_dummies(z.month.astype(str), drop_first=True, dtype=float)
        X['time_years'] = np.arange(len(z))/12
        fit = sm.OLS(z['index'].to_numpy(), sm.add_constant(X).to_numpy()).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
        coef = len(fit.params)-1
        slopes.append({'start': start, 'end': 2025, 'slope_pp_per_year': fit.params[coef],
                       'lower': fit.conf_int()[coef, 0], 'upper': fit.conf_int()[coef, 1], 'months': len(z)})
    pd.DataFrame(slopes).to_csv(OUT / 'trends.csv', index=False)
    peak2025 = peaks.loc[peaks.year.eq(2025)].set_index('window')
    info = {'sites': len(ids), 'blocks': int(meta.block.nunique()), 'complete_panel_days': int(d.site.isin(ids).sum()),
            'nonholiday_panel_days': int((d.site.isin(ids) & ~d.holiday).sum()),
            'zero_panel_days': int((d.site.isin(ids) & d.total.eq(0)).sum()),
            'minimum_month': m.loc[m.year.ge(2020)].sort_values('index').iloc[0][['year', 'month', 'index']].to_dict(),
            'trends': slopes, '2025_windows': peak2025.to_dict('index')}
    crossing = {}
    post = m.loc[m.year.ge(2020)].reset_index(drop=True)
    for threshold in [90, 95, 100]:
        starts = [i for i in range(len(post)-2) if post.loc[i:i+2, 'index'].ge(threshold).all()
                  and post.loc[i, 'date'] >= pd.Timestamp('2020-05-01')]
        crossing[str(threshold)] = post.loc[starts[0], 'date'].strftime('%Y-%m') if starts else None
    info['first_three_consecutive_months_at_threshold'] = crossing
    (OUT / 'summary.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)
    export(d, meta, a, m, bt, h, peaks, sens, site_ratios, info)


def export(d, meta, a, m, bt, h, peaks, sens, ratios, info):
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    all_sites = pd.read_csv(OUT / 'sites.csv')
    excluded = all_sites.loc[~all_sites.site.isin(meta.index)]
    all_sites['accepted_panel'] = all_sites.site.isin(meta.index)
    all_sites.to_csv(OUT / 'network-membership.csv', index=False)
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.scatter(excluded.x, excluded.y, s=12, color='#aaaaaa', alpha=.6,
               label=f'Other source detectors (n={len(excluded)})', zorder=1)
    ax.scatter(meta.x, meta.y, s=33, color='#087e9a', edgecolors='white', linewidths=.4,
               label=f'Accepted seven-year panel (n={len(meta)})', zorder=2)
    ax.set_aspect('equal')
    ax.set_xlabel('East of reference point (km)')
    ax.set_ylabel('North of reference point (km)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=.15)
    fig.tight_layout()
    savefig('network')
    plt.figure(figsize=(10, 3.4))
    plt.plot(m.date, m['index'], color='#126b86', lw=1.6)
    plt.axhline(100, color='black', ls='--', lw=.8, label='Same calendar month in 2019')
    plt.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-06-01'), color='grey', alpha=.17)
    plt.ylabel('Calendar-standardised index (2019 = 100)')
    plt.legend(loc='lower right')
    plt.grid(axis='y', alpha=.2)
    savefig('monthly')
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    ax[0].errorbar(a.year, a['index'], yerr=[a['index']-a.lower, a.upper-a['index']], fmt='o-', capsize=3, label='Ratio of standardised sums')
    ax[0].plot(a.year, a.median_site_index, 's--', label='Median detector ratio')
    for label in ['Weekday', 'Weekend']:
        z = bt.loc[bt.type.eq(label)]
        ax[1].plot(z.year, z['index'], 'o-', label=label)
    for x in ax:
        x.axhline(100, color='black', ls=':', lw=.8)
        x.set_ylabel('Index (2019 = 100)')
        x.legend(fontsize=8)
        x.grid(axis='y', alpha=.2)
    fig.tight_layout()
    savefig('annual')
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    for year in [2019, 2020, 2022, 2025]:
        z = h.loc[h.year.eq(year) & h.type.eq('Weekday')]
        ax[0].plot(z.hour, z.mean_count, label=str(year))
        ax[1].plot(z.hour, z.share, label=str(year))
    ax[0].set_ylabel('Mean detections per detector/hour')
    ax[1].set_ylabel('Share of weekday detections (%)')
    for x in ax:
        x.set_xlabel('Local clock hour')
        x.set_xticks(range(0, 24, 3))
        x.legend()
    fig.tight_layout()
    savefig('hourly')
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    z = meta.join(ratios[2025].rename('index'))
    im = ax[0].scatter(z.x, z.y, c=z['index'], cmap='RdBu_r', vmin=50, vmax=150, s=15)
    all_sites = pd.read_csv(OUT / 'sites.csv')
    ax[0].scatter(all_sites.x, all_sites.y, s=3, color='grey', alpha=.3, zorder=0, label='Other source devices')
    ax[0].set_aspect('equal')
    ax[0].legend(fontsize=7, loc='upper left')
    ax[0].set_xlabel('East of reference (km)')
    ax[0].set_ylabel('North of reference (km)')
    fig.colorbar(im, ax=ax[0], label='2025 / 2019 index (colour clipped)')
    sorted_values = np.sort(ratios[2025])
    ax[1].plot(sorted_values, np.arange(1, len(z)+1)/len(z))
    ax[1].axvline(100, color='black', ls='--', lw=.8)
    ax[1].set_xlabel('2025 / 2019 detector index')
    ax[1].set_ylabel('Cumulative fraction of detectors')
    fig.tight_layout()
    savefig('heterogeneity')
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for label, g in sens.groupby('design'):
        ax.plot(g.year, g['index'], marker='o', label=label)
    ax.axhline(100, color='black', ls=':', lw=.8)
    ax.set_ylabel('Index (2019 = 100)')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    savefig('sensitivity')
    audit = pd.read_csv(OUT / 'audit.csv')
    table('audit', ['Year', 'Raw rows', 'Complete days', 'Devices', 'Zero days'],
          [[int(r.year), f'{int(r.raw_rows):,}', f'{int(r.complete_days):,}', int(r.sites_complete), f'{int(r.zero_complete_days):,}'] for r in audit.itertuples()])
    table('annual', ['Year', 'Index', r'95\% interval', 'Median', r'$\geq95$ (\%)'],
          [[int(r.year), f'{r.index:.1f}', f'[{r.lower:.1f}, {r.upper:.1f}]', f'{r.median_site_index:.1f}', f'{r.sites_at_95:.1f}'] for r in a.itertuples()])
    table('windows', ['Window', '2020', '2022', '2025', r'$\Delta$ share$^a$'],
          [[label, *[f'{g.loc[g.year.eq(y), "index"].iloc[0]:.1f}' for y in [2020, 2022, 2025]],
            f'{g.loc[g.year.eq(2025), "share_change_pp"].iloc[0]:+.2f}'] for label, g in peaks.groupby('window', sort=False)])
    table('sensitivity', ['Design', '$S_{2025}$', '2020', '2023', '2025'],
          [[label, int(g.loc[g.year.eq(2025), 'sites'].iloc[0]), *[f'{g.loc[g.year.eq(y), "index"].iloc[0]:.1f}' if np.isfinite(g.loc[g.year.eq(y), 'index'].iloc[0]) else '--' for y in [2020, 2023, 2025]]] for label, g in sens.groupby('design', sort=False)])
    macros = {'PanelSites': str(info['sites']), 'PanelBlocks': str(info['blocks']),
              'PanelDays': f"{info['complete_panel_days']:,}", 'ZeroPanelDays': f"{info['zero_panel_days']:,}"}
    for r in a.itertuples():
        # Roman suffixes are valid TeX command names; digits are not.
        suffix = ['Nineteen', 'Twenty', 'TwentyOne', 'TwentyTwo', 'TwentyThree', 'TwentyFour', 'TwentyFive'][int(r.year)-2019]
        macros['Index'+suffix] = f'{r.index:.1f}'
    (TAB / 'recovery-numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k, v in macros.items())+'\n')


if __name__ == '__main__':
    main()
